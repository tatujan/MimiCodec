// Mimi Audio Codec - extracted from kyutai-labs/moshi
// Original code: Copyright (c) Kyutai, all rights reserved.
// Licensed under MIT/Apache-2.0

pub use candle;
pub use candle_nn;

pub mod conv;
pub mod kv_cache;
pub mod mimi;
pub mod nn;
pub mod quantization;
pub mod seanet;
pub mod streaming;
pub mod transformer;
pub mod wav;

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

pub use streaming::{StreamMask, StreamTensor, StreamingModule};
