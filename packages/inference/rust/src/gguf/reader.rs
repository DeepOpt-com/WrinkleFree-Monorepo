//! GGUF file reader with memory-mapped access.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use super::types::*;

/// Memory-mapped GGUF file reader.
pub struct GgufReader {
    /// Memory-mapped file data
    mmap: Mmap,
    /// Byte order: true = little-endian (normal), false = big-endian (swapped)
    little_endian: bool,
    /// GGUF version
    pub version: u32,
    /// Data alignment
    pub alignment: usize,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor information
    pub tensors: Vec<GgufTensorInfo>,
    /// Offset to start of tensor data
    pub data_offset: usize,
    /// Model configuration (parsed from metadata)
    pub config: ModelConfig,
}

impl GgufReader {
    /// Open a GGUF file and parse its header.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 24 {
            return Err(GgufError::FileTooSmall);
        }

        let mut reader = Self {
            mmap,
            little_endian: true,
            version: 0,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            data_offset: 0,
            config: ModelConfig::default(),
        };

        reader.parse_header()?;
        reader.extract_config();

        Ok(reader)
    }

    /// Parse the GGUF header and all metadata/tensor info.
    fn parse_header(&mut self) -> Result<(), GgufError> {
        let mut offset = 0;

        // Check magic (always little-endian)
        let magic = self.read_u32_at(offset, true);
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }
        offset += 4;

        // Check version and determine byte order
        let version_raw = self.read_u32_at(offset, true);
        if version_raw & 0xFFFF == 0 {
            // Byte order is swapped
            self.little_endian = false;
            self.version = version_raw.swap_bytes();
        } else {
            self.version = version_raw;
        }

        if self.version < 2 || self.version > GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(self.version));
        }
        offset += 4;

        // Read tensor count and kv count
        let tensor_count = self.read_u64(offset);
        offset += 8;
        let kv_count = self.read_u64(offset);
        offset += 8;

        // Parse metadata
        for _ in 0..kv_count {
            let (key, value, consumed) = self.parse_kv_pair(offset)?;
            self.metadata.insert(key, value);
            offset += consumed;
        }

        // Check for custom alignment
        if let Some(GgufValue::Uint32(align)) = self.metadata.get("general.alignment") {
            self.alignment = *align as usize;
        }

        // Parse tensor info
        for _ in 0..tensor_count {
            let (tensor_info, consumed) = self.parse_tensor_info(offset)?;
            self.tensors.push(tensor_info);
            offset += consumed;
        }

        // Align to data offset
        let padding = offset % self.alignment;
        if padding != 0 {
            offset += self.alignment - padding;
        }
        self.data_offset = offset;

        // Debug: print calculated data offset
        eprintln!("=== GGUF DATA OFFSET DEBUG ===");
        eprintln!("  Position after tensor info: {} (pre-alignment)", offset - (if padding != 0 { self.alignment - padding } else { 0 }));
        eprintln!("  Alignment: {}", self.alignment);
        eprintln!("  Data offset (aligned): {}", self.data_offset);

        // Convert relative tensor offsets to absolute offsets
        // GGUF stores offsets relative to tensor data start
        for tensor in &mut self.tensors {
            tensor.data_offset += self.data_offset;
        }

        // Debug: print first I2_S tensor offset
        for tensor in &self.tensors {
            if tensor.dtype == GgmlQuantType::I2_S {
                eprintln!("  First I2_S tensor '{}': relative_offset={}, absolute_offset={}",
                    tensor.name, tensor.data_offset - self.data_offset, tensor.data_offset);
                break;
            }
        }

        Ok(())
    }

    /// Parse a key-value pair from metadata.
    fn parse_kv_pair(&self, offset: usize) -> Result<(String, GgufValue, usize), GgufError> {
        let mut consumed = 0;

        // Read key (string)
        let key_len = self.read_u64(offset + consumed) as usize;
        consumed += 8;
        let key = self.read_string(offset + consumed, key_len)?;
        consumed += key_len;

        // Read value type
        let value_type = GgufValueType::try_from(self.read_u32(offset + consumed))?;
        consumed += 4;

        // Read value
        let (value, value_consumed) = self.parse_value(offset + consumed, value_type)?;
        consumed += value_consumed;

        Ok((key, value, consumed))
    }

    /// Parse a value of the given type.
    fn parse_value(&self, offset: usize, vtype: GgufValueType) -> Result<(GgufValue, usize), GgufError> {
        match vtype {
            GgufValueType::Uint8 => Ok((GgufValue::Uint8(self.mmap[offset]), 1)),
            GgufValueType::Int8 => Ok((GgufValue::Int8(self.mmap[offset] as i8), 1)),
            GgufValueType::Uint16 => Ok((GgufValue::Uint16(self.read_u16(offset)), 2)),
            GgufValueType::Int16 => Ok((GgufValue::Int16(self.read_u16(offset) as i16), 2)),
            GgufValueType::Uint32 => Ok((GgufValue::Uint32(self.read_u32(offset)), 4)),
            GgufValueType::Int32 => Ok((GgufValue::Int32(self.read_u32(offset) as i32), 4)),
            GgufValueType::Float32 => Ok((GgufValue::Float32(f32::from_le_bytes(
                self.mmap[offset..offset + 4].try_into().unwrap()
            )), 4)),
            GgufValueType::Bool => Ok((GgufValue::Bool(self.mmap[offset] != 0), 1)),
            GgufValueType::String => {
                let len = self.read_u64(offset) as usize;
                let s = self.read_string(offset + 8, len)?;
                Ok((GgufValue::String(s), 8 + len))
            }
            GgufValueType::Array => {
                let mut consumed = 0;
                let element_type = GgufValueType::try_from(self.read_u32(offset))?;
                consumed += 4;
                let array_len = self.read_u64(offset + consumed) as usize;
                consumed += 8;

                let mut array = Vec::with_capacity(array_len);
                for _ in 0..array_len {
                    let (value, value_consumed) = self.parse_value(offset + consumed, element_type)?;
                    array.push(value);
                    consumed += value_consumed;
                }
                Ok((GgufValue::Array(array), consumed))
            }
            GgufValueType::Uint64 => Ok((GgufValue::Uint64(self.read_u64(offset)), 8)),
            GgufValueType::Int64 => Ok((GgufValue::Int64(self.read_u64(offset) as i64), 8)),
            GgufValueType::Float64 => Ok((GgufValue::Float64(f64::from_le_bytes(
                self.mmap[offset..offset + 8].try_into().unwrap()
            )), 8)),
        }
    }

    /// Parse tensor info entry.
    fn parse_tensor_info(&self, offset: usize) -> Result<(GgufTensorInfo, usize), GgufError> {
        let mut consumed = 0;

        // Read name
        let name_len = self.read_u64(offset + consumed) as usize;
        consumed += 8;
        let name = self.read_string(offset + consumed, name_len)?;
        consumed += name_len;

        // Read n_dims
        let n_dims = self.read_u32(offset + consumed) as usize;
        consumed += 4;

        // Read dimensions
        let mut shape = Vec::with_capacity(n_dims);
        let mut n_elements = 1usize;
        for _ in 0..n_dims {
            let dim = self.read_u64(offset + consumed) as usize;
            shape.push(dim);
            n_elements = n_elements.saturating_mul(dim);
            consumed += 8;
        }

        // Read dtype
        let dtype = GgmlQuantType::try_from(self.read_u32(offset + consumed))?;
        consumed += 4;

        // Read data offset (relative to tensor data start)
        let relative_offset = self.read_u64(offset + consumed) as usize;
        consumed += 8;

        let n_bytes = dtype.bytes_for_elements(n_elements);

        Ok((
            GgufTensorInfo {
                name,
                shape,
                n_elements,
                dtype,
                data_offset: relative_offset, // Store relative offset temporarily
                n_bytes,
            },
            consumed,
        ))
    }

    /// Extract model configuration from metadata.
    fn extract_config(&mut self) {
        // Get architecture
        self.config.architecture = self.get_string("general.architecture")
            .unwrap_or_default();

        let arch = &self.config.architecture;

        // Helper to get arch-specific key
        let get_arch_u32 = |reader: &Self, key: &str| -> Option<u32> {
            let full_key = key.replace("{arch}", arch);
            reader.get_u32(&full_key)
        };

        let get_arch_f32 = |reader: &Self, key: &str| -> Option<f32> {
            let full_key = key.replace("{arch}", arch);
            reader.get_f32(&full_key)
        };

        // Extract config values
        self.config.vocab_size = get_arch_u32(self, "{arch}.vocab_size").unwrap_or(0);
        self.config.hidden_size = get_arch_u32(self, "{arch}.embedding_length").unwrap_or(0);
        self.config.num_layers = get_arch_u32(self, "{arch}.block_count").unwrap_or(0);
        self.config.num_heads = get_arch_u32(self, "{arch}.attention.head_count").unwrap_or(0);
        self.config.num_kv_heads = get_arch_u32(self, "{arch}.attention.head_count_kv")
            .unwrap_or(self.config.num_heads);
        self.config.intermediate_size = get_arch_u32(self, "{arch}.feed_forward_length").unwrap_or(0);
        self.config.max_seq_len = get_arch_u32(self, "{arch}.context_length").unwrap_or(2048);
        self.config.rms_norm_eps = get_arch_f32(self, "{arch}.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        self.config.rope_theta = get_arch_f32(self, "{arch}.rope.freq_base")
            .unwrap_or(10000.0);

        eprintln!("=== GGUF CONFIG DEBUG ===");
        eprintln!("  architecture: {}", arch);
        eprintln!("  rope_theta: {} (key: {}.rope.freq_base)", self.config.rope_theta, arch);
        eprintln!("  vocab_size: {}, hidden_size: {}, num_layers: {}",
            self.config.vocab_size, self.config.hidden_size, self.config.num_layers);
        eprintln!("  num_heads: {}, num_kv_heads: {}", self.config.num_heads, self.config.num_kv_heads);

        // Tokenizer config
        self.config.bos_token_id = self.get_u32("tokenizer.ggml.bos_token_id");
        self.config.eos_token_id = self.get_u32("tokenizer.ggml.eos_token_id");
        self.config.pad_token_id = self.get_u32("tokenizer.ggml.padding_token_id");

        // LRC rank
        let lrc_key = format!("{}.lrc.rank", arch);
        self.config.lrc_rank = self.get_u32(&lrc_key);
    }

    // ========================================================================
    // Low-level read helpers
    // ========================================================================

    fn read_u16(&self, offset: usize) -> u16 {
        let bytes: [u8; 2] = self.mmap[offset..offset + 2].try_into().unwrap();
        if self.little_endian {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        }
    }

    fn read_u32(&self, offset: usize) -> u32 {
        self.read_u32_at(offset, self.little_endian)
    }

    fn read_u32_at(&self, offset: usize, little_endian: bool) -> u32 {
        let bytes: [u8; 4] = self.mmap[offset..offset + 4].try_into().unwrap();
        if little_endian {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        }
    }

    fn read_u64(&self, offset: usize) -> u64 {
        let bytes: [u8; 8] = self.mmap[offset..offset + 8].try_into().unwrap();
        if self.little_endian {
            u64::from_le_bytes(bytes)
        } else {
            u64::from_be_bytes(bytes)
        }
    }

    fn read_string(&self, offset: usize, len: usize) -> Result<String, GgufError> {
        String::from_utf8(self.mmap[offset..offset + len].to_vec())
            .map_err(|_| GgufError::InvalidUtf8)
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Get an i32 metadata value.
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.metadata.get(key).and_then(|v| v.as_i32())
    }

    /// Get a f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Get a string metadata value.
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.metadata.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    /// Get the architecture name.
    pub fn architecture(&self) -> &str {
        &self.config.architecture
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get tensor data as a byte slice.
    pub fn tensor_data(&self, tensor: &GgufTensorInfo) -> &[u8] {
        &self.mmap[tensor.data_offset..tensor.data_offset + tensor.n_bytes]
    }

    /// Get tensor data by name.
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8], GgufError> {
        let tensor = self.find_tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        Ok(self.tensor_data(tensor))
    }

    /// Get raw mmap reference (for advanced use).
    pub fn raw_data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get tokenizer vocabulary from metadata.
    pub fn get_vocab(&self) -> Option<Vec<String>> {
        if let Some(GgufValue::Array(tokens)) = self.metadata.get("tokenizer.ggml.tokens") {
            let vocab: Vec<String> = tokens
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if !vocab.is_empty() {
                return Some(vocab);
            }
        }
        None
    }

    /// Get BPE merges from metadata (for BPE tokenizers).
    pub fn get_merges(&self) -> Option<Vec<String>> {
        if let Some(GgufValue::Array(merges)) = self.metadata.get("tokenizer.ggml.merges") {
            let merges: Vec<String> = merges
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if !merges.is_empty() {
                return Some(merges);
            }
        }
        None
    }

    /// Print summary of the GGUF file.
    pub fn print_summary(&self) {
        println!("=== GGUF Summary ===");
        println!("Version: {}", self.version);
        println!("Architecture: {}", self.config.architecture);
        println!("Vocab size: {}", self.config.vocab_size);
        println!("Hidden size: {}", self.config.hidden_size);
        println!("Num layers: {}", self.config.num_layers);
        println!("Num heads: {}", self.config.num_heads);
        println!("Num KV heads: {}", self.config.num_kv_heads);
        println!("Intermediate size: {}", self.config.intermediate_size);
        println!("Max seq len: {}", self.config.max_seq_len);
        if let Some(lrc_rank) = self.config.lrc_rank {
            println!("LRC rank: {}", lrc_rank);
        }
        println!("\nTensors: {}", self.tensors.len());

        // Show first few tensors
        for (i, t) in self.tensors.iter().take(10).enumerate() {
            println!("  [{:3}] {} {:?} {:?} ({} bytes)",
                i, t.name, t.shape, t.dtype, t.n_bytes);
        }
        if self.tensors.len() > 10 {
            println!("  ... and {} more", self.tensors.len() - 10);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_block_info() {
        // Test a few key quantization types
        let (bs, ts) = GgmlQuantType::F32.block_info();
        assert_eq!(bs, 1);
        assert_eq!(ts, 4);

        let (bs, ts) = GgmlQuantType::F16.block_info();
        assert_eq!(bs, 1);
        assert_eq!(ts, 2);

        let (bs, ts) = GgmlQuantType::TQ2_0.block_info();
        assert_eq!(bs, 256);
        assert_eq!(ts, 66);
    }

    #[test]
    fn test_bytes_for_elements() {
        // F32: 4 bytes per element
        assert_eq!(GgmlQuantType::F32.bytes_for_elements(100), 400);

        // TQ2_0: 256 elements per 66 bytes
        assert_eq!(GgmlQuantType::TQ2_0.bytes_for_elements(256), 66);
        assert_eq!(GgmlQuantType::TQ2_0.bytes_for_elements(512), 132);
    }
}
