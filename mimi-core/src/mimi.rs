// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::nn::MaybeQuantizedVarBuilder;
use crate::streaming::{StreamMask, StreamTensor, StreamingModule};
use crate::{conv, quantization, seanet, transformer};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::quantized_var_builder::VarBuilder as QuantizedVarBuilder;
use std::collections::HashMap;
use std::time::Instant;

/// Accumulated per-component timing (in seconds).
#[derive(Debug, Clone, Default)]
pub struct StepTimings {
    pub seanet_encode: f64,
    pub encoder_transformer: f64,
    pub downsample: f64,
    pub quantizer_encode: f64,
    pub quantizer_decode: f64,
    pub upsample: f64,
    pub decoder_transformer: f64,
    pub seanet_decode: f64,
    pub steps: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    pub seanet: seanet::Config,
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/b7d2bd5a/.hydra/config.yaml
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet_cfg = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: candle_nn::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: 0,
            norm: conv::Norm::WeightNorm,
            pad_mode: conv::PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer_cfg = transformer::Config {
            d_model: seanet_cfg.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            conv_kernel_size: 5,
            use_conv_bias: true,
            use_conv_block: false,
            cross_attention: None,
            max_period: 10000,
            gating: None,
            norm: crate::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,

            dim_feedforward: 2048,
            kv_repeat: 1,
            conv_layout: true, // see builders.py
            max_seq_len: 8192, // the transformer works at 25hz so this is ~5 mins.
            shared_cross_attn: false,
        };
        Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mimi {
    encoder: seanet::SeaNetEncoder,
    decoder: seanet::SeaNetDecoder,
    encoder_transformer: transformer::Transformer,
    decoder_transformer: transformer::Transformer,
    downsample: conv::ConvDownsample1d,
    upsample: conv::ConvTrUpsample1d,
    quantizer: quantization::SplitResidualVectorQuantizer,
    config: Config,
    timings: StepTimings,
}

impl Mimi {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        Self::new_(None, cfg, vb)
    }

    pub fn batched(batch_size: usize, cfg: Config, vb: VarBuilder) -> Result<Self> {
        Self::new_(Some(batch_size), cfg, vb)
    }

    fn new_(batch_size: Option<usize>, cfg: Config, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.seanet.dimension;
        let encoder = seanet::SeaNetEncoder::new(&cfg.seanet, vb.pp("encoder"))?;
        let decoder = seanet::SeaNetDecoder::new(&cfg.seanet, vb.pp("decoder"))?;
        let encoder_transformer = transformer::Transformer::new(
            batch_size,
            dim,
            &cfg.transformer,
            vb.pp("encoder_transformer"),
        )?;
        let decoder_transformer = transformer::Transformer::new(
            batch_size,
            dim,
            &cfg.transformer,
            vb.pp("decoder_transformer"),
        )?;
        let quantizer = quantization::SplitResidualVectorQuantizer::new(
            /* dim */ cfg.quantizer_dim,
            /* input_dim */ Some(dim),
            /* output_dim */ Some(dim),
            /* n_q */ cfg.quantizer_n_q,
            /* bins */ cfg.quantizer_bins,
            vb.pp("quantizer"),
        )?;
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;

        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
        // `upsample` and `downsample` only apply if frame_rate is different from encoder_frame_rate.
        let downsample = conv::ConvDownsample1d::new(
            /* stride */ downsample_stride,
            /* dim */ dim,
            /* causal */ true,
            /* learnt */ true,
            vb.pp("downsample"),
        )?;
        let upsample = conv::ConvTrUpsample1d::new(
            /* stride */ downsample_stride,
            /* dim */ dim,
            /* causal */ true,
            /* learnt */ true,
            vb.pp("upsample"),
        )?;

        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            config: cfg,
            timings: StepTimings::default(),
        })
    }

    /// Build Mimi with quantized transformer weights and regular weights for everything else.
    fn new_quantized_(
        batch_size: Option<usize>,
        cfg: Config,
        vb: VarBuilder,                           // for SEANet, quantizer, conv
        transformer_vb: MaybeQuantizedVarBuilder,  // for transformers (quantized)
    ) -> Result<Self> {
        let dim = cfg.seanet.dimension;
        let encoder = seanet::SeaNetEncoder::new(&cfg.seanet, vb.pp("encoder"))?;
        let decoder = seanet::SeaNetDecoder::new(&cfg.seanet, vb.pp("decoder"))?;
        let encoder_transformer = transformer::Transformer::new_maybe_quantized(
            batch_size,
            dim,
            &cfg.transformer,
            transformer_vb.pp("encoder_transformer"),
        )?;
        let decoder_transformer = transformer::Transformer::new_maybe_quantized(
            batch_size,
            dim,
            &cfg.transformer,
            transformer_vb.pp("decoder_transformer"),
        )?;
        let quantizer = quantization::SplitResidualVectorQuantizer::new(
            cfg.quantizer_dim,
            Some(dim),
            Some(dim),
            cfg.quantizer_n_q,
            cfg.quantizer_bins,
            vb.pp("quantizer"),
        )?;
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
        let downsample = conv::ConvDownsample1d::new(
            downsample_stride, dim, true, true, vb.pp("downsample"),
        )?;
        let upsample = conv::ConvTrUpsample1d::new(
            downsample_stride, dim, true, true, vb.pp("upsample"),
        )?;
        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            quantizer,
            downsample,
            upsample,
            config: cfg,
            timings: StepTimings::default(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn encode_pre_quantize(&mut self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.encoder.forward(xs)?;
        self.encoder_transformer.reset_state();
        let xs = self.encoder_transformer.forward(&xs)?;
        let xs = &xs[0];
        xs.apply(&self.downsample)
    }

    pub fn encode(&mut self, xs: &Tensor) -> Result<Tensor> {
        let t0 = Instant::now();
        let xs = self.encoder.forward(xs)?;
        let t1 = Instant::now();
        self.encoder_transformer.reset_state();
        let xs = self.encoder_transformer.forward(&xs)?;
        let xs = &xs[0];
        let t2 = Instant::now();
        let xs = xs.apply(&self.downsample)?;
        let t3 = Instant::now();
        let codes = self.quantizer.encode(&xs)?;
        let t4 = Instant::now();
        self.timings.seanet_encode += (t1 - t0).as_secs_f64();
        self.timings.encoder_transformer += (t2 - t1).as_secs_f64();
        self.timings.downsample += (t3 - t2).as_secs_f64();
        self.timings.quantizer_encode += (t4 - t3).as_secs_f64();
        self.timings.steps += 1;
        Ok(codes)
    }

    pub fn encode_step(&mut self, xs: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        let t0 = Instant::now();
        let xs = self.encoder.step(xs, m)?;
        let t1 = Instant::now();
        let xs = self.encoder_transformer.step(&xs, m)?;
        let t2 = Instant::now();
        let xs = self.downsample.step(&xs, m)?;
        let t3 = Instant::now();

        self.timings.seanet_encode += (t1 - t0).as_secs_f64();
        self.timings.encoder_transformer += (t2 - t1).as_secs_f64();
        self.timings.downsample += (t3 - t2).as_secs_f64();

        match xs.as_option() {
            None => Ok(().into()),
            Some(xs) => {
                let tq = Instant::now();
                let codes = self.quantizer.encode(xs)?;
                self.timings.quantizer_encode += tq.elapsed().as_secs_f64();
                self.timings.steps += 1;
                Ok(codes.into())
            }
        }
    }

    pub fn decode(&mut self, codes: &Tensor) -> Result<Tensor> {
        let t0 = Instant::now();
        let emb = self.quantizer.decode(codes)?;
        let t1 = Instant::now();
        let emb = emb.apply(&self.upsample)?;
        let t2 = Instant::now();
        self.decoder_transformer.reset_state();
        let outs = self.decoder_transformer.forward(&emb)?;
        let out = &outs[0];
        let t3 = Instant::now();
        let result = self.decoder.forward(out)?;
        let t4 = Instant::now();
        self.timings.quantizer_decode += (t1 - t0).as_secs_f64();
        self.timings.upsample += (t2 - t1).as_secs_f64();
        self.timings.decoder_transformer += (t3 - t2).as_secs_f64();
        self.timings.seanet_decode += (t4 - t3).as_secs_f64();
        Ok(result)
    }

    pub fn decode_step(&mut self, codes: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        let t0 = Instant::now();
        let emb = match codes.as_option() {
            Some(codes) => StreamTensor::from_tensor(self.quantizer.decode(codes)?),
            None => StreamTensor::empty(),
        };
        let t1 = Instant::now();
        let emb = self.upsample.step(&emb, m)?;
        let t2 = Instant::now();
        let out = self.decoder_transformer.step(&emb, m)?;
        let t3 = Instant::now();
        let result = self.decoder.step(&out, m)?;
        let t4 = Instant::now();

        self.timings.quantizer_decode += (t1 - t0).as_secs_f64();
        self.timings.upsample += (t2 - t1).as_secs_f64();
        self.timings.decoder_transformer += (t3 - t2).as_secs_f64();
        self.timings.seanet_decode += (t4 - t3).as_secs_f64();

        Ok(result)
    }

    pub fn timings(&self) -> &StepTimings {
        &self.timings
    }

    pub fn reset_timings(&mut self) {
        self.timings = StepTimings::default();
    }

    pub fn reset_state(&mut self) {
        self.encoder.reset_state();
        self.encoder_transformer.reset_state();
        self.decoder.reset_state();
        self.decoder_transformer.reset_state();
        self.upsample.reset_state();
        self.downsample.reset_state();
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize, batch_size: usize) -> Result<()> {
        self.encoder_transformer.reset_batch_idx(batch_idx, batch_size)?;
        self.encoder_transformer.reset_batch_idx(batch_idx, batch_size)?;
        self.encoder.reset_batch_idx(batch_idx, batch_size)?;
        self.decoder.reset_batch_idx(batch_idx, batch_size)?;
        self.upsample.reset_batch_idx(batch_idx, batch_size)?;
        self.downsample.reset_batch_idx(batch_idx, batch_size)?;
        Ok(())
    }
}

pub fn load(model_file: &str, num_codebooks: Option<usize>, dev: &Device) -> Result<Mimi> {
    if model_file.ends_with(".gguf") {
        return load_gguf(model_file, num_codebooks, dev);
    }
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, dev)? };
    let cfg = Config::v0_1(num_codebooks);
    let mimi = Mimi::new(cfg, vb)?;
    Ok(mimi)
}

/// Load from a quantized GGUF file.
/// Transformer weights stay quantized (Q8_0); everything else is dequantized to f32.
pub fn load_gguf(model_file: &str, num_codebooks: Option<usize>, dev: &Device) -> Result<Mimi> {
    // 1. Load GGUF, dequantize all tensors → regular VarBuilder (for SEANet, quantizer, conv)
    let mut file = std::fs::File::open(model_file)?;
    let content = candle::quantized::gguf_file::Content::read(&mut file)?;
    let mut tensors = HashMap::new();
    for name in content.tensor_infos.keys() {
        let qt = content.tensor(&mut file, name, dev)?;
        tensors.insert(name.clone(), qt.dequantize(dev)?);
    }
    let regular_vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, dev);

    // 2. Load GGUF again as quantized VarBuilder (for transformers)
    let quantized_vb = QuantizedVarBuilder::from_gguf(model_file, dev)?;

    let cfg = Config::v0_1(num_codebooks);
    Mimi::new_quantized_(None, cfg, regular_vb, MaybeQuantizedVarBuilder::Quantized(quantized_vb))
}

pub fn load_b(
    batch_size: Option<usize>,
    model_file: &str,
    num_codebooks: Option<usize>,
    dev: &Device,
) -> Result<Mimi> {
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, dev)? };
    let cfg = Config::v0_1(num_codebooks);
    let mimi = Mimi::new_(batch_size, cfg, vb)?;
    Ok(mimi)
}
