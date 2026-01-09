//! Simple tokenizer using GGUF vocabulary.
//!
//! Provides basic BPE tokenization and decoding using the vocabulary
//! embedded in GGUF model files.

use std::collections::HashMap;

/// Simple BPE tokenizer.
pub struct Tokenizer {
    /// Token string to ID mapping
    token_to_id: HashMap<String, i32>,
    /// ID to token string mapping
    id_to_token: Vec<String>,
    /// Special token IDs
    pub bos_token_id: Option<i32>,
    pub eos_token_id: Option<i32>,
    pub pad_token_id: Option<i32>,
}

impl Tokenizer {
    /// Create tokenizer from vocabulary.
    pub fn new(vocab: Vec<String>) -> Self {
        let token_to_id: HashMap<String, i32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as i32))
            .collect();

        Self {
            token_to_id,
            id_to_token: vocab,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
        }
    }

    /// Set special token IDs.
    pub fn with_special_tokens(
        mut self,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
    ) -> Self {
        self.bos_token_id = bos.map(|v| v as i32);
        self.eos_token_id = eos.map(|v| v as i32);
        self.pad_token_id = pad.map(|v| v as i32);
        self
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encode text to token IDs using simple byte-fallback tokenization.
    ///
    /// This is a simplified encoder that:
    /// 1. Tries to match whole words/subwords from the vocab
    /// 2. Falls back to byte tokens for unknown characters
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut tokens = Vec::new();

        // Add BOS token if available
        if let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        // Simple greedy tokenization
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let mut found = false;

            // Try to match longest token first (up to 20 chars)
            for len in (1..=20.min(bytes.len() - i)).rev() {
                if let Ok(substr) = std::str::from_utf8(&bytes[i..i + len]) {
                    // Try exact match
                    if let Some(&id) = self.token_to_id.get(substr) {
                        tokens.push(id);
                        i += len;
                        found = true;
                        break;
                    }

                    // Try with leading space (common in BPE)
                    let with_space = format!("▁{}", substr);
                    if let Some(&id) = self.token_to_id.get(&with_space) {
                        tokens.push(id);
                        i += len;
                        found = true;
                        break;
                    }

                    // Try with Ġ prefix (GPT-style)
                    let with_g = format!("Ġ{}", substr);
                    if let Some(&id) = self.token_to_id.get(&with_g) {
                        tokens.push(id);
                        i += len;
                        found = true;
                        break;
                    }
                }
            }

            // Fallback: use byte token
            if !found {
                let byte = bytes[i];
                // Try byte token format: <0xXX>
                let byte_token = format!("<0x{:02X}>", byte);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else {
                    // Last resort: use the byte value directly if in vocab range
                    if (byte as usize) < self.id_to_token.len() {
                        tokens.push(byte as i32);
                    }
                }
                i += 1;
            }
        }

        tokens
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut result = String::new();

        for &id in ids {
            // Skip special tokens in output
            if Some(id) == self.bos_token_id
                || Some(id) == self.eos_token_id
                || Some(id) == self.pad_token_id
            {
                continue;
            }

            if id >= 0 && (id as usize) < self.id_to_token.len() {
                let token = &self.id_to_token[id as usize];

                // Handle special token formats
                if token.starts_with("<0x") && token.ends_with('>') {
                    // Byte token: <0xXX>
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        result.push(byte as char);
                    }
                } else if token.starts_with("▁") {
                    // SentencePiece style: ▁ = space
                    result.push(' ');
                    result.push_str(&token[3..]); // Skip the ▁ character (3 bytes in UTF-8)
                } else if token.starts_with("Ġ") {
                    // GPT style: Ġ = space
                    result.push(' ');
                    result.push_str(&token[2..]); // Skip the Ġ character (2 bytes in UTF-8)
                } else if token == "<|begin_of_text|>" || token == "<s>" {
                    // Skip BOS markers
                } else if token == "<|end_of_text|>" || token == "</s>" {
                    // Stop at EOS
                    break;
                } else if token.starts_with("<|") && token.ends_with("|>") {
                    // Skip special control tokens
                } else {
                    result.push_str(token);
                }
            }
        }

        // Clean up leading space if present
        if result.starts_with(' ') {
            result = result[1..].to_string();
        }

        result
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: i32) -> Option<&str> {
        if id >= 0 && (id as usize) < self.id_to_token.len() {
            Some(&self.id_to_token[id as usize])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let vocab = vec![
            "hello".to_string(),
            "world".to_string(),
            " ".to_string(),
            "▁hello".to_string(),
        ];
        let tokenizer = Tokenizer::new(vocab);

        assert_eq!(tokenizer.vocab_size(), 4);
        assert_eq!(tokenizer.decode_token(0), Some("hello"));
        assert_eq!(tokenizer.decode_token(1), Some("world"));
    }

    #[test]
    fn test_decode_sentencepiece() {
        let vocab = vec![
            "▁The".to_string(),
            "▁capital".to_string(),
            "▁of".to_string(),
            "▁France".to_string(),
            "▁is".to_string(),
            "▁Paris".to_string(),
        ];
        let tokenizer = Tokenizer::new(vocab);

        let text = tokenizer.decode(&[0, 1, 2, 3, 4, 5]);
        assert_eq!(text, "The capital of France is Paris");
    }
}
