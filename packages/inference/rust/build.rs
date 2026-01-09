use std::process::Command;

const DEFAULT_VERSION: &str = "0.0.0";
const DEFAULT_PROJECT_NAME: &str = "wf-inference";

/// Set a compile-time environment variable with the SGL_MODEL_GATEWAY_ prefix
macro_rules! set_env {
    ($name:expr, $value:expr) => {
        println!("cargo:rustc-env=SGL_MODEL_GATEWAY_{}={}", $name, $value);
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=Cargo.toml");

    // Build C++ inference library when native-inference feature is enabled
    #[cfg(feature = "native-inference")]
    build_cpp_inference_engine()?;

    // Set version info environment variables
    let version = read_cargo_version().unwrap_or_else(|_| DEFAULT_VERSION.to_string());
    let target = std::env::var("TARGET").unwrap_or_else(|_| get_rustc_host().unwrap_or_default());
    let profile = std::env::var("PROFILE").unwrap_or_default();

    set_env!("PROJECT_NAME", DEFAULT_PROJECT_NAME);
    set_env!("VERSION", version);
    set_env!(
        "BUILD_TIME",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );
    set_env!(
        "BUILD_MODE",
        if profile == "release" {
            "release"
        } else {
            "debug"
        }
    );
    set_env!("TARGET_TRIPLE", target);
    set_env!(
        "GIT_BRANCH",
        git_branch().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "GIT_COMMIT",
        git_commit().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "GIT_STATUS",
        git_status().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "RUSTC_VERSION",
        rustc_version().unwrap_or_else(|| "unknown".into())
    );
    set_env!(
        "CARGO_VERSION",
        cargo_version().unwrap_or_else(|| "unknown".into())
    );

    Ok(())
}

fn read_cargo_version() -> Result<String, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string("Cargo.toml")?;
    let toml: toml::Value = toml::from_str(&content)?;
    toml.get("package")
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| "Missing version in Cargo.toml".into())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn git_branch() -> Option<String> {
    run_cmd("git", &["rev-parse", "--abbrev-ref", "HEAD"])
}

fn git_commit() -> Option<String> {
    run_cmd("git", &["rev-parse", "--short", "HEAD"])
}

fn git_status() -> Option<String> {
    run_cmd("git", &["status", "--porcelain"])
        .map(|s| if s.is_empty() { "clean" } else { "dirty" }.into())
}

fn rustc_version() -> Option<String> {
    run_cmd("rustc", &["--version"])
}

fn cargo_version() -> Option<String> {
    run_cmd("cargo", &["--version"])
}

fn get_rustc_host() -> Option<String> {
    run_cmd("rustc", &["-vV"])?
        .lines()
        .find(|l| l.starts_with("host: "))
        .and_then(|l| l.strip_prefix("host: "))
        .map(|s| s.trim().to_string())
}

/// Build the C++ inference engine library (only for llama-inference feature)
///
/// NOTE: The native-inference feature now uses pure Rust BitNet kernels
/// and does NOT require C++ compilation. This function is only called
/// when llama-inference is enabled (legacy mode with llama.cpp).
#[cfg(feature = "native-inference")]
fn build_cpp_inference_engine() -> Result<(), Box<dyn std::error::Error>> {
    // NOTE: BitNet kernels are now pure Rust (src/kernels/bitnet/)
    // No C++ compilation needed for basic native-inference

    // Only build llama.cpp wrapper when llama-inference feature is enabled
    #[cfg(feature = "llama-inference")]
    {
        use std::path::PathBuf;

        println!("cargo:rerun-if-changed=../cpp/llama_engine.cpp");
        println!("cargo:rerun-if-changed=../cpp/kv_cache.cpp");
        println!("cargo:rerun-if-changed=../cpp/bitnet_batch.cpp");

        let cpp_dir = PathBuf::from("../cpp");
        let llama_dir = PathBuf::from("../extern/llama.cpp");
        let llama_include = llama_dir.join("include");
        let ggml_include = llama_dir.join("ggml/include");
        let llama_lib_dir = llama_dir.join("build/src");
        let ggml_lib_dir = llama_dir.join("build/ggml/src");

        let target = std::env::var("TARGET").unwrap_or_default();
        let use_native = std::env::var("NATIVE_SIMD").is_ok();
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

        // Check if llama.cpp is built
        let llama_lib_exists = llama_lib_dir.join("libllama.so").exists()
            || llama_lib_dir.join("libllama.dylib").exists();

        if llama_lib_exists {
            // Build the inference engine wrapper around llama.cpp
            let mut engine_build = cc::Build::new();
            engine_build
                .cpp(true)
                .opt_level(3)
                .flag("-std=c++17")
                .flag("-DNDEBUG")
                .include(&cpp_dir)
                .include(&llama_include)
                .include(&ggml_include)
                .file(cpp_dir.join("llama_engine.cpp"))
                .file(cpp_dir.join("kv_cache.cpp"))
                .file(cpp_dir.join("bitnet_batch.cpp"));

            if use_native {
                engine_build.flag("-march=native");
                engine_build.flag("-mtune=native");
                println!("cargo:warning=Building C++ with -march=native for maximum SIMD optimization");
            } else if target.contains("x86_64") {
                engine_build.flag("-mavx2");
                engine_build.flag("-mfma");
            } else if target.contains("aarch64") {
                engine_build.flag("-march=armv8-a+simd");
            }

            engine_build.compile("sgl_kernel_inference");
            println!("cargo:rustc-link-lib=static=sgl_kernel_inference");

            // Link llama.cpp libraries
            let llama_lib_abs = std::fs::canonicalize(&llama_lib_dir)?;
            let ggml_lib_abs = std::fs::canonicalize(&ggml_lib_dir)?;
            println!("cargo:rustc-link-search=native={}", llama_lib_abs.display());
            println!("cargo:rustc-link-search=native={}", ggml_lib_abs.display());
            println!("cargo:rustc-link-lib=dylib=llama");
            println!("cargo:rustc-link-lib=dylib=ggml");

            // Link C++ standard library
            if target_os == "macos" {
                println!("cargo:rustc-link-lib=c++");
            } else {
                println!("cargo:rustc-link-lib=stdc++");
                println!("cargo:rustc-link-lib=gomp");
            }
        } else {
            panic!("llama-inference feature enabled but llama.cpp not built. Build it first:\n\
                   cd extern/llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build");
        }
    }

    Ok(())
}
