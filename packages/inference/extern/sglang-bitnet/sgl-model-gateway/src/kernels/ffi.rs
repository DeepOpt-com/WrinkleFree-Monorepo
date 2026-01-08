//! FFI bindings to native BitNet kernels.
//!
//! These bindings link to sgl-kernel/csrc/bitnet/bitnet_gemv.cpp

use std::ffi::c_int;

/// CPU capability flags
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CPUCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_dotprod: bool,
}

impl CPUCapabilities {
    /// Detect CPU capabilities at runtime.
    pub fn detect() -> Self {
        unsafe { detect_cpu_capabilities() }
    }

    /// Get a human-readable description of capabilities.
    pub fn description(&self) -> String {
        let mut caps = Vec::new();
        if self.has_avx512 {
            caps.push("AVX-512");
        }
        if self.has_avx2 {
            caps.push("AVX2");
        }
        if self.has_neon {
            caps.push("NEON");
        }
        if self.has_dotprod {
            caps.push("DotProd");
        }
        if caps.is_empty() {
            "None (scalar fallback)".to_string()
        } else {
            caps.join(", ")
        }
    }
}

/// Tile configuration for cache optimization
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Output tile size (default 256)
    pub bm: c_int,
    /// Input tile size (default 32)
    pub bk: c_int,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self { bm: 256, bk: 32 }
    }
}

// Link to the native BitNet kernel library
#[link(name = "sgl_kernel_bitnet", kind = "static")]
extern "C" {
    /// Detect CPU capabilities.
    fn detect_cpu_capabilities() -> CPUCapabilities;

    /// BitNet GEMV: y = W * x
    ///
    /// # Arguments
    /// * `n` - Input dimension (must be multiple of 128)
    /// * `result` - Output scalar (dot product result)
    /// * `packed_weights` - Packed 2-bit ternary weights
    /// * `activations` - INT8 activations
    fn bitnet_vec_dot_i2_i8(
        n: c_int,
        result: *mut f32,
        packed_weights: *const u8,
        activations: *const i8,
    );

    /// BitNet GEMM: Y = W * X (batched)
    ///
    /// # Arguments
    /// * `m` - Number of output features
    /// * `n` - Batch size
    /// * `k` - Number of input features (must be multiple of 128)
    /// * `output` - Output matrix (M x N)
    /// * `packed_weights` - Packed weights (M x K/4)
    /// * `activations` - Input activations (K x N)
    /// * `scale` - Weight scale factor
    /// * `config` - Tile configuration for cache optimization
    fn bitnet_gemm_i2_i8(
        m: c_int,
        n: c_int,
        k: c_int,
        output: *mut f32,
        packed_weights: *const u8,
        activations: *const i8,
        scale: f32,
        config: *const TileConfig,
    );

    /// Quantize activations to INT8.
    ///
    /// # Arguments
    /// * `n` - Number of elements
    /// * `output` - INT8 output
    /// * `input` - FP32 input
    /// * `scale` - Output scale factor
    fn quantize_activations_i8(
        n: c_int,
        output: *mut i8,
        input: *const f32,
        scale: *mut f32,
    );

    /// Auto-tune tile sizes for the current CPU.
    fn auto_tune_tiles(m: c_int, k: c_int) -> TileConfig;
}

/// Safe wrapper for BitNet kernels.
pub struct BitNetKernel {
    pub capabilities: CPUCapabilities,
    pub tile_config: TileConfig,
}

impl BitNetKernel {
    /// Create a new kernel wrapper with auto-detected capabilities.
    pub fn new() -> Self {
        let capabilities = CPUCapabilities::detect();
        Self {
            capabilities,
            tile_config: TileConfig::default(),
        }
    }

    /// Create with auto-tuned tile configuration.
    pub fn with_auto_tune(typical_m: i32, typical_k: i32) -> Self {
        let capabilities = CPUCapabilities::detect();
        let tile_config = unsafe { auto_tune_tiles(typical_m, typical_k) };
        Self {
            capabilities,
            tile_config,
        }
    }

    /// Compute dot product: result = packed_weights · activations
    ///
    /// # Safety
    /// - `n` must be a multiple of 128
    /// - `packed_weights` must have length n/4 bytes
    /// - `activations` must have length n
    pub fn vec_dot(&self, packed_weights: &[u8], activations: &[i8]) -> f32 {
        let n = activations.len();
        debug_assert!(n % 128 == 0, "n must be multiple of 128");
        debug_assert!(packed_weights.len() >= n / 4, "weights too short");

        let mut result = 0.0f32;
        unsafe {
            bitnet_vec_dot_i2_i8(
                n as c_int,
                &mut result,
                packed_weights.as_ptr(),
                activations.as_ptr(),
            );
        }
        result
    }

    /// Compute matrix multiplication: output = packed_weights * activations
    ///
    /// # Arguments
    /// * `m` - Output features
    /// * `n` - Batch size
    /// * `k` - Input features (must be multiple of 128)
    /// * `packed_weights` - Packed weights [M, K/4]
    /// * `activations` - Input [K, N]
    /// * `scale` - Weight scale
    ///
    /// # Returns
    /// Output matrix [M, N]
    pub fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        packed_weights: &[u8],
        activations: &[i8],
        scale: f32,
    ) -> Vec<f32> {
        debug_assert!(k % 128 == 0, "k must be multiple of 128");
        debug_assert!(packed_weights.len() >= m * k / 4, "weights too short");
        debug_assert!(activations.len() >= k * n, "activations too short");

        let mut output = vec![0.0f32; m * n];
        unsafe {
            bitnet_gemm_i2_i8(
                m as c_int,
                n as c_int,
                k as c_int,
                output.as_mut_ptr(),
                packed_weights.as_ptr(),
                activations.as_ptr(),
                scale,
                &self.tile_config,
            );
        }
        output
    }

    /// Quantize FP32 activations to INT8.
    ///
    /// # Returns
    /// (quantized activations, scale factor)
    pub fn quantize_activations(&self, input: &[f32]) -> (Vec<i8>, f32) {
        let n = input.len();
        let mut output = vec![0i8; n];
        let mut scale = 0.0f32;
        unsafe {
            quantize_activations_i8(
                n as c_int,
                output.as_mut_ptr(),
                input.as_ptr(),
                &mut scale,
            );
        }
        (output, scale)
    }
}

impl Default for BitNetKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if native kernels are available.
pub fn is_kernel_available() -> bool {
    let caps = CPUCapabilities::detect();
    caps.has_avx2 || caps.has_avx512 || caps.has_neon
}
