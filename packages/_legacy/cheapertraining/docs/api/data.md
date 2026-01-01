# Data API

Data loading, tokenization, and processing utilities.

## TokenizerWrapper

Wrapper around HuggingFace tokenizers with additional utilities.

```python
from cheapertraining.data import TokenizerWrapper

tokenizer = TokenizerWrapper("meta-llama/Llama-3.2-1B")

# Basic tokenization
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Batch tokenization
batch = tokenizer.batch_encode(["Text 1", "Text 2"], max_length=2048)
```

### Constructor

```python
TokenizerWrapper(
    model_name_or_path: str,
    max_length: int = 2048,
    padding: bool = True,
    truncation: bool = True,
)
```

### Methods

#### `encode(text, add_special_tokens=True) -> List[int]`

Encode text to token IDs.

#### `decode(token_ids, skip_special_tokens=True) -> str`

Decode token IDs to text.

#### `batch_encode(texts, max_length=None, return_tensors="pt") -> dict`

Batch encode multiple texts.

**Returns:**
- `dict` with `input_ids`, `attention_mask`

#### `apply_chat_template(messages) -> str`

Apply chat template to conversation.

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]
formatted = tokenizer.apply_chat_template(messages)
```

#### `create_labels_with_prompt_mask(input_ids, prompt_length) -> Tensor`

Create labels with prompt tokens masked (-100).

```python
labels = tokenizer.create_labels_with_prompt_mask(
    input_ids,
    prompt_length=50,
)
# labels[:50] = -100, labels[50:] = input_ids[50:]
```

### Properties

- `vocab_size`: Size of vocabulary
- `pad_token_id`: Padding token ID
- `eos_token_id`: End-of-sequence token ID
- `bos_token_id`: Beginning-of-sequence token ID

## MixedDataset

Weighted mixture of multiple datasets.

```python
from cheapertraining.data import MixedDataset

mixture = [
    {
        "name": "fineweb",
        "weight": 0.6,
        "path": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
    },
    {
        "name": "code",
        "weight": 0.3,
        "path": "bigcode/starcoderdata",
        "split": "train",
    },
    {
        "name": "math",
        "weight": 0.1,
        "path": "open-web-math/open-web-math",
        "split": "train",
    },
]

dataset = MixedDataset(
    mixture=mixture,
    tokenizer=tokenizer,
    streaming=True,
)
```

### Constructor

```python
MixedDataset(
    mixture: List[dict],
    tokenizer: TokenizerWrapper,
    streaming: bool = True,
    seed: int = 42,
)
```

### Mixture Entry Format

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | str | Yes | Dataset identifier |
| `weight` | float | Yes | Sampling weight |
| `path` | str | Yes | HuggingFace dataset path |
| `subset` | str | No | Dataset subset name |
| `split` | str | No | Dataset split (default: "train") |

Weights are normalized to sum to 1.0.

### Methods

#### `__iter__()`

Iterate over mixed samples.

```python
for sample in dataset:
    text = sample["text"]
```

## PackedDataset

Pack variable-length documents into fixed-length sequences.

```python
from cheapertraining.data import PackedDataset

packed = PackedDataset(
    dataset=mixed_dataset,
    max_length=2048,
    eos_token_id=tokenizer.eos_token_id,
)
```

### Constructor

```python
PackedDataset(
    dataset: Iterable,
    max_length: int,
    eos_token_id: int,
    drop_last: bool = True,
)
```

### How Packing Works

Documents are concatenated with EOS separators until reaching `max_length`:

```
[Doc1 tokens] [EOS] [Doc2 tokens] [EOS] [Doc3...] -> [2048 tokens]
```

This minimizes padding waste and improves training efficiency.

### Methods

#### `__iter__()`

Iterate over packed sequences.

```python
for batch in packed:
    input_ids = batch["input_ids"]  # Shape: (max_length,)
```

## PretrainDataset

Dataset for pretraining with automatic packing.

```python
from cheapertraining.data.datasets import PretrainDataset

dataset = PretrainDataset(
    data_path="HuggingFaceFW/fineweb-edu",
    tokenizer=tokenizer,
    max_length=2048,
    streaming=True,
)
```

### Constructor

```python
PretrainDataset(
    data_path: str,
    tokenizer: TokenizerWrapper,
    max_length: int = 2048,
    subset: str = None,
    split: str = "train",
    streaming: bool = True,
    packing: bool = True,
)
```

## SFTDataset

Dataset for supervised fine-tuning with prompt masking.

```python
from cheapertraining.data.datasets import SFTDataset

dataset = SFTDataset(
    data_path="HuggingFaceTB/smoltalk",
    tokenizer=tokenizer,
    max_length=4096,
)
```

### Constructor

```python
SFTDataset(
    data_path: str,
    tokenizer: TokenizerWrapper,
    max_length: int = 4096,
    subset: str = None,
    split: str = "train",
    format: str = "messages",  # or "instruction_response"
)
```

### Supported Formats

**Messages format:**
```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

**Instruction-response format:**
```json
{
  "instruction": "What is 2+2?",
  "response": "2+2 equals 4."
}
```

### Output Format

```python
{
    "input_ids": [...],      # Full sequence
    "attention_mask": [...], # Attention mask
    "labels": [...],         # -100 for prompt, token_ids for completion
}
```

## DataLoader Utilities

### create_dataloader

Create a DataLoader with distributed sampling.

```python
from cheapertraining.data import create_dataloader

dataloader = create_dataloader(
    dataset=dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    distributed=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | Dataset | - | Dataset to load |
| `batch_size` | int | - | Batch size |
| `num_workers` | int | 4 | Data loading workers |
| `pin_memory` | bool | True | Pin memory for GPU |
| `distributed` | bool | False | Use DistributedSampler |
| `drop_last` | bool | True | Drop incomplete batches |
| `shuffle` | bool | True | Shuffle data |
