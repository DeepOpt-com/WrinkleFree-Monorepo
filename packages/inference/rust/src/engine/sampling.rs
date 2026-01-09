//! Token sampling strategies for generation.

use rand::Rng;

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for softmax (0 = greedy, 1 = normal, >1 = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample a token from logits with repetition penalty.
pub fn sample_token_with_penalty(
    logits: &[f32],
    config: &SamplingConfig,
    rng: &mut impl Rng,
    past_tokens: &[i32],
) -> usize {
    let mut logits = logits.to_vec();

    // Apply repetition penalty FIRST (before temperature)
    if config.repetition_penalty != 1.0 && !past_tokens.is_empty() {
        apply_repetition_penalty(&mut logits, past_tokens, config.repetition_penalty);
    }

    // Apply temperature
    if config.temperature > 0.0 && config.temperature != 1.0 {
        let inv_temp = 1.0 / config.temperature;
        for l in &mut logits {
            *l *= inv_temp;
        }
    }

    // For temperature = 0, use greedy
    if config.temperature == 0.0 {
        return argmax(&logits);
    }

    // Apply top-k filtering
    if config.top_k > 0 && config.top_k < logits.len() {
        logits = top_k_filter(&logits, config.top_k);
    }

    // Apply top-p filtering
    if config.top_p < 1.0 {
        logits = top_p_filter(&logits, config.top_p);
    }

    // Convert to probabilities
    let probs = softmax(&logits);

    // Sample from distribution
    sample_from_probs(&probs, rng)
}

/// Sample a token from logits (without repetition penalty).
pub fn sample_token(
    logits: &[f32],
    config: &SamplingConfig,
    rng: &mut impl Rng,
) -> usize {
    sample_token_with_penalty(logits, config, rng, &[])
}

/// Greedy sampling (argmax).
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Top-k sampling: keep only top k logits, set rest to -inf.
pub fn top_k_sampling(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }

    // Find k-th largest value
    let mut sorted: Vec<f32> = logits.iter().cloned().collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];

    // Zero out values below threshold
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn top_k_filter(logits: &[f32], k: usize) -> Vec<f32> {
    let mut result = logits.to_vec();
    top_k_sampling(&mut result, k);
    result
}

/// Top-p (nucleus) sampling: keep smallest set of tokens with cumulative prob >= p.
pub fn top_p_sampling(logits: &mut [f32], p: f32) {
    // Sort indices by logit value (descending)
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&i, &j| logits[j].partial_cmp(&logits[i]).unwrap_or(std::cmp::Ordering::Equal));

    // Compute softmax for sorted logits
    let sorted_logits: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    let probs = softmax(&sorted_logits);

    // Find cutoff index
    let mut cumsum = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Mask out tokens beyond cutoff
    for &idx in indices.iter().skip(cutoff_idx) {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn top_p_filter(logits: &[f32], p: f32) -> Vec<f32> {
    let mut result = logits.to_vec();
    top_p_sampling(&mut result, p);
    result
}

/// Apply repetition penalty to tokens that appeared in context.
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[i32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &token_id in token_ids {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Softmax function.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    logits.iter().map(|l| (l - max).exp() / exp_sum).collect()
}

/// Sample from a probability distribution.
fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_argmax() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        assert_eq!(sample_token(&logits, &config, &mut rng), 1);
    }

    #[test]
    fn test_top_k() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        top_k_sampling(&mut logits, 2);

        // Only top 2 should remain
        assert!(logits[0] == f32::NEG_INFINITY);
        assert!(logits[1] == 5.0);
        assert!(logits[2] == 3.0);
        assert!(logits[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher logits = higher probs
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}
