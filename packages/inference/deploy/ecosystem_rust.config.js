// pm2 ecosystem config for Rust DLM server
// NO PYTHON - uses Fast-dLLM v2 block diffusion via Rust FFI
module.exports = {
  apps: [
    {
      name: 'dlm-server',
      script: '/opt/wrinklefree/packages/inference/deploy/start_dlm_server.sh',
      interpreter: '/bin/bash',
      cwd: '/opt/wrinklefree',
      env: {
        // Use all CPUs for SIMD parallelism
        OMP_NUM_THREADS: require('os').cpus().length,
        MKL_NUM_THREADS: require('os').cpus().length,
        // Library paths for llama.cpp
        LD_LIBRARY_PATH: '/opt/wrinklefree/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp/build/src:/opt/wrinklefree/packages/inference/extern/sglang-bitnet/3rdparty/llama.cpp/build/ggml/src',
      },
      // Restart policy
      max_restarts: 10,
      restart_delay: 1000,
      // Logging
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/bitnet-rust-error.log',
      out_file: '/var/log/bitnet-rust-out.log',
      merge_logs: true,
    },
    {
      name: 'streamlit',
      script: '/opt/wrinklefree/packages/inference/deploy/start_streamlit.sh',
      interpreter: '/bin/bash',
      cwd: '/opt/wrinklefree',
      env: {
        BITNET_API_URL: 'http://localhost:30000',
      },
      // Restart policy
      max_restarts: 10,
      restart_delay: 1000,
      // Logging
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/streamlit-error.log',
      out_file: '/var/log/streamlit-out.log',
      merge_logs: true,
    },
  ],
};
