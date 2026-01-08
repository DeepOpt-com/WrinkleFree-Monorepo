//! GGUF types and constants.
//!
//! Ported from llama.cpp's gguf-py/gguf/constants.py

use std::fmt;

/// GGUF file magic number ("GGUF" in little-endian)
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Current GGUF version
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Block size for K-quants
pub const QK_K: usize = 256;

/// Block size for native BitNet kernels (from sgl-kernel)
pub const QK_I2_S_NATIVE: usize = 128;

/// GGUF value types for metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::InvalidValueType(value)),
        }
    }
}

/// GGML quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    TL1 = 38,
    TL2 = 39,
}

impl TryFrom<u32> for GgmlQuantType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            31 => Ok(Self::Q4_0_4_4),
            32 => Ok(Self::Q4_0_4_8),
            33 => Ok(Self::Q4_0_8_8),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            38 => Ok(Self::TL1),
            39 => Ok(Self::TL2),
            _ => Err(GgufError::InvalidQuantType(value)),
        }
    }
}

impl GgmlQuantType {
    /// Returns (block_size, type_size_bytes) for this quantization type.
    pub fn block_info(&self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 2 + 16),
            Self::Q4_1 => (32, 2 + 2 + 16),
            Self::Q5_0 => (32, 2 + 4 + 16),
            Self::Q5_1 => (32, 2 + 2 + 4 + 16),
            Self::Q8_0 => (32, 2 + 32),
            Self::Q8_1 => (32, 4 + 4 + 32),
            Self::Q2_K => (QK_K, 2 + 2 + QK_K / 16 + QK_K / 4),
            Self::Q3_K => (QK_K, 2 + QK_K / 4 + QK_K / 8 + 12),
            Self::Q4_K => (QK_K, 2 + 2 + QK_K / 2 + 12),
            Self::Q5_K => (QK_K, 2 + 2 + QK_K / 2 + QK_K / 8 + 12),
            Self::Q6_K => (QK_K, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16),
            Self::Q8_K => (QK_K, 4 + QK_K + QK_K / 8),
            Self::IQ2_XXS => (QK_K, 2 + QK_K / 4),
            Self::IQ2_XS => (QK_K, 2 + QK_K / 4 + QK_K / 32),
            Self::IQ3_XXS => (QK_K, 2 + QK_K / 4 + QK_K / 8),
            Self::IQ1_S => (QK_K, 2 + QK_K / 8 + QK_K / 16),
            Self::IQ4_NL => (32, 2 + 16),
            Self::IQ3_S => (QK_K, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4),
            Self::IQ2_S => (QK_K, 2 + QK_K / 4 + QK_K / 16),  // 256 elements, 82 bytes
            Self::IQ4_XS => (QK_K, 2 + 2 + QK_K / 2 + QK_K / 64),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::IQ1_M => (QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32),
            Self::BF16 => (1, 2),
            Self::Q4_0_4_4 => (32, 2 + 16),
            Self::Q4_0_4_8 => (32, 2 + 16),
            Self::Q4_0_8_8 => (32, 2 + 16),
            // Ternary quantization types (optimized for BitNet)
            // TQ1_0: Packed ternary, 5 weights per 8 bits (base-3 encoding)
            Self::TQ1_0 => (256, 64 + 4 + 2),  // 256 elements, 70 bytes
            // TQ2_0: 2-bit ternary (00=-1, 01=0, 10=+1), 4 weights per byte
            Self::TQ2_0 => (256, 64 + 2),  // 256 elements, 66 bytes
            Self::TL1 => (256, 70),
            Self::TL2 => (256, 66),
        }
    }

    /// Returns the number of bytes needed to store `n_elements` values.
    pub fn bytes_for_elements(&self, n_elements: usize) -> usize {
        let (block_size, type_size) = self.block_info();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        n_blocks * type_size
    }

    /// Returns true if this is a ternary quantization type.
    pub fn is_ternary(&self) -> bool {
        matches!(self, Self::TQ1_0 | Self::TQ2_0 | Self::IQ2_S | Self::TL1 | Self::TL2)
    }
}

/// Metadata value (can be scalar, string, or array)
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    /// Try to get as u32
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint8(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to get as i32
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            Self::Int8(v) => Some(*v as i32),
            Self::Int16(v) => Some(*v as i32),
            Self::Uint32(v) if *v <= i32::MAX as u32 => Some(*v as i32),
            _ => None,
        }
    }

    /// Try to get as u64
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to get as f32
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as array
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

/// Tensor information from GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight")
    pub name: String,
    /// Tensor shape (e.g., [4096, 4096])
    pub shape: Vec<usize>,
    /// Number of elements
    pub n_elements: usize,
    /// Quantization type
    pub dtype: GgmlQuantType,
    /// Offset in file to tensor data
    pub data_offset: usize,
    /// Size in bytes
    pub n_bytes: usize,
}

/// Model configuration extracted from GGUF metadata
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Architecture name (e.g., "llama", "qwen2", "bitnet")
    pub architecture: String,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Hidden size / embedding dimension
    pub hidden_size: u32,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Intermediate (FFN) dimension
    pub intermediate_size: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta (frequency base)
    pub rope_theta: f32,
    /// BOS token ID
    pub bos_token_id: Option<u32>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
    /// Pad token ID
    pub pad_token_id: Option<u32>,
    /// LRC rank (if using Low-Rank Correction)
    pub lrc_rank: Option<u32>,
}

/// GGUF parsing errors
#[derive(Debug)]
pub enum GgufError {
    /// Invalid GGUF magic number
    InvalidMagic(u32),
    /// Unsupported GGUF version
    UnsupportedVersion(u32),
    /// Invalid value type
    InvalidValueType(u32),
    /// Invalid quantization type
    InvalidQuantType(u32),
    /// Missing required metadata key
    MissingKey(String),
    /// I/O error
    Io(std::io::Error),
    /// File is too small
    FileTooSmall,
    /// Invalid string encoding
    InvalidUtf8,
    /// Tensor not found
    TensorNotFound(String),
    /// Invalid tensor dimensions
    InvalidDimensions,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic(m) => write!(f, "Invalid GGUF magic: 0x{:08x} (expected 0x{:08x})", m, GGUF_MAGIC),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported GGUF version: {} (expected {})", v, GGUF_VERSION),
            Self::InvalidValueType(t) => write!(f, "Invalid GGUF value type: {}", t),
            Self::InvalidQuantType(t) => write!(f, "Invalid quantization type: {}", t),
            Self::MissingKey(k) => write!(f, "Missing required metadata key: {}", k),
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::FileTooSmall => write!(f, "File is too small to be a valid GGUF"),
            Self::InvalidUtf8 => write!(f, "Invalid UTF-8 string in GGUF"),
            Self::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            Self::InvalidDimensions => write!(f, "Invalid tensor dimensions"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
