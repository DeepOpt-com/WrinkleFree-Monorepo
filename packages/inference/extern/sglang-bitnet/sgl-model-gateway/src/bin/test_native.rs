//! Simple test binary for native BitNet inference.
//!
//! This tests the C++ llama.cpp wrapper without the full gateway routing.
//!
//! Usage:
//!   cargo run --release --features native-inference --bin test_native -- <model_path>

use std::env;

#[cfg(feature = "native-inference")]
use sgl_model_gateway::grpc_client::native_engine::NativeEngineClient;
#[cfg(feature = "native-inference")]
use sgl_model_gateway::protocols::sampling_params::SamplingParams;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("microsoft/bitnet-b1.58-2B-4T");

    println!("=== Native BitNet Inference Test ===");
    println!("Model: {}", model_path);
    println!();

    #[cfg(feature = "native-inference")]
    {
        match NativeEngineClient::new(model_path).await {
            Ok(client) => {
                println!("Engine created successfully!");

                // Get model info
                match client.get_model_info().await {
                    Ok(info) => {
                        println!("Model info:");
                        println!("  vocab_size:   {}", info.vocab_size);
                        println!("  hidden_size:  {}", info.hidden_size);
                        println!("  num_layers:   {}", info.num_layers);
                        println!("  max_seq_len:  {}", info.max_seq_len);
                    }
                    Err(e) => println!("Failed to get model info: {}", e),
                }

                println!();
                println!("Testing generation...");

                let prompt = "The capital of France is";
                // Use greedy sampling (temp=0) for fair comparison with BitNet.cpp
                let params = SamplingParams {
                    temperature: Some(0.0),  // Greedy
                    top_p: Some(1.0),
                    max_new_tokens: Some(32),
                    ..Default::default()
                };

                println!("Prompt: \"{}\"", prompt);

                let start = std::time::Instant::now();
                match client.generate(prompt, &params).await {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        println!("Generated: \"{}\"", result.text);
                        println!();
                        println!("Stats:");
                        println!("  prompt tokens:     {}", result.num_prompt_tokens);
                        println!("  generated tokens:  {}", result.num_generated_tokens);
                        println!("  total time:        {:.2}ms", elapsed.as_secs_f64() * 1000.0);
                        if result.num_generated_tokens > 0 {
                            let tok_s = result.num_generated_tokens as f64 / elapsed.as_secs_f64();
                            println!("  tokens/sec:        {:.1}", tok_s);
                        }
                    }
                    Err(e) => {
                        println!("Generation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Failed to create engine: {}", e);
            }
        }
    }

    #[cfg(not(feature = "native-inference"))]
    {
        let _ = model_path;
        println!("ERROR: Native inference not enabled.");
        println!("Rebuild with: cargo build --release --features native-inference");
    }
}
