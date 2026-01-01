# Models API

Model architecture classes for CheaperTraining.

## MobileLLMConfig

Configuration dataclass for MobileLLM architecture.

```python
from cheapertraining.models import MobileLLMConfig

config = MobileLLMConfig(
    vocab_size=128256,       # Vocabulary size
    num_layers=22,           # Number of transformer layers
    num_heads=24,            # Number of attention heads
    num_kv_heads=6,          # Number of KV heads (for GQA)
    embed_dim=1536,          # Embedding dimension
    hidden_dim=6144,         # FFN hidden dimension
    max_seq_len=32768,       # Maximum sequence length
    rope_base=500000,        # RoPE base frequency
    use_qk_norm=True,        # Apply QK normalization
    tie_word_embeddings=True # Share input/output embeddings
)
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | 128256 | Size of vocabulary |
| `num_layers` | int | - | Number of transformer layers |
| `num_heads` | int | - | Number of attention heads |
| `num_kv_heads` | int | - | Number of key-value heads (for GQA) |
| `embed_dim` | int | - | Embedding dimension |
| `hidden_dim` | int | - | FFN intermediate dimension |
| `max_seq_len` | int | 32768 | Maximum sequence length |
| `rope_base` | float | 500000 | RoPE base frequency |
| `use_qk_norm` | bool | True | Whether to apply QK normalization |
| `tie_word_embeddings` | bool | True | Share input/output embeddings |
| `dropout` | float | 0.0 | Dropout probability |
| `norm_eps` | float | 1e-5 | Layer norm epsilon |

## get_mobilellm_config

Factory function for predefined configurations.

```python
from cheapertraining.models import get_mobilellm_config

config = get_mobilellm_config("950m")  # Returns MobileLLMConfig for 950M model
```

### Available Configurations

| Name | Parameters | Layers | Heads | Embed Dim |
|------|------------|--------|-------|-----------|
| `"140m"` | ~140M | 15 | 9 | 576 |
| `"360m"` | ~360M | 15 | 16 | 1024 |
| `"950m"` | ~950M | 22 | 24 | 1536 |
| `"7b"` | ~7B | 32 | 32 | 4096 |
| `"70b"` | ~70B | 80 | 64 | 8192 |
| `"671b"` | ~671B | 61 | 128 | 7168 |

## MobileLLM

Main model class implementing the MobileLLM architecture.

```python
from cheapertraining.models import MobileLLM, get_mobilellm_config

config = get_mobilellm_config("950m")
model = MobileLLM(config)

# Forward pass
output = model(input_ids, labels=labels)
loss = output.loss
logits = output.logits
```

### Methods

#### `forward(input_ids, attention_mask=None, labels=None)`

Forward pass through the model.

**Parameters:**
- `input_ids` (Tensor): Input token IDs, shape `(batch, seq_len)`
- `attention_mask` (Tensor, optional): Attention mask
- `labels` (Tensor, optional): Labels for loss computation

**Returns:**
- `ModelOutput` with `loss`, `logits`, `hidden_states`

#### `generate(input_ids, max_new_tokens=100, temperature=1.0, top_p=0.9)`

Generate text autoregressively.

**Parameters:**
- `input_ids` (Tensor): Prompt token IDs
- `max_new_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `top_p` (float): Top-p (nucleus) sampling threshold

**Returns:**
- `Tensor`: Generated token IDs

#### `num_parameters()`

Returns total number of parameters.

```python
print(f"Parameters: {model.num_parameters():,}")  # Parameters: 950,000,000
```

#### `enable_gradient_checkpointing()`

Enable gradient checkpointing for memory efficiency.

```python
model.enable_gradient_checkpointing()
```

## RMSNorm

Root Mean Square Layer Normalization.

```python
from cheapertraining.models.attention import RMSNorm

norm = RMSNorm(dim=1536, eps=1e-5)
output = norm(hidden_states)
```

## MultiHeadAttention

Multi-head attention with Grouped Query Attention and optional QK-Norm.

```python
from cheapertraining.models.attention import MultiHeadAttention

attention = MultiHeadAttention(
    embed_dim=1536,
    num_heads=24,
    num_kv_heads=6,
    use_qk_norm=True,
    rope_base=500000,
    max_seq_len=32768,
)

output = attention(hidden_states, attention_mask=mask)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embed_dim` | int | Model embedding dimension |
| `num_heads` | int | Number of query heads |
| `num_kv_heads` | int | Number of key-value heads |
| `use_qk_norm` | bool | Apply normalization to Q, K |
| `rope_base` | float | RoPE base frequency |
| `max_seq_len` | int | Maximum sequence length |

## FeedForward

SwiGLU feed-forward network.

```python
from cheapertraining.models.attention import FeedForward

ffn = FeedForward(embed_dim=1536, hidden_dim=6144)
output = ffn(hidden_states)
```

## TransformerBlock

Single transformer layer with attention and FFN.

```python
from cheapertraining.models.transformer import TransformerBlock

block = TransformerBlock(config, layer_idx=0)
output = block(hidden_states, attention_mask=mask)
```

## TransformerDecoder

Stack of transformer blocks.

```python
from cheapertraining.models.transformer import TransformerDecoder

decoder = TransformerDecoder(config)
output = decoder(hidden_states, attention_mask=mask)
```
