# Tiny GPT-2

A small GPT-2 inference implementation built with PyTorch.

The goal of this project was to understand GPT-2 by rebuilding its main inference path from scratch, loading the real pretrained weights, and checking that the results match Hugging Face's implementation.

The current version supports:

* the full GPT-2 forward pass
* pretrained weight loading
* numerical comparison against Hugging Face
* simple greedy text generation

## What is implemented

The model follows the GPT-2 small architecture:

```text
Token IDs
   ↓
Token + position embeddings
   ↓
12 Transformer blocks
   ↓
Final LayerNorm
   ↓
Language model head
   ↓
Vocabulary logits
```

Each Transformer block contains:

```text
Input
  ↓
LayerNorm
  ↓
Causal self-attention
  ↓
Residual connection
  ↓
LayerNorm
  ↓
MLP
  ↓
Residual connection
```

The project includes:

* token embeddings
* positional embeddings
* QKV projection
* multi-head attention
* causal masking
* feed-forward layers
* GELU activation
* residual connections
* 12 decoder blocks
* final LayerNorm
* tied embedding and output weights
* greedy text generation

## Why I built this

Using GPT-2 with Hugging Face is very simple:

```python
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2"
)
```

But most of the model is hidden behind the library.

I wanted to understand:

* how attention works in practice
* how tensor shapes change through the model
* how Q, K and V are created
* how causal masking works
* how pretrained weights are loaded
* how to test whether a custom implementation is actually correct

This project is also a baseline for future inference optimization experiments.

## Pretrained weights

The model architecture is custom, but the trained weights come from the official GPT-2 checkpoint.

Hugging Face uses `Conv1D` for some layers, while this project uses `nn.Linear`.

Because the weight layouts are different, some weights need to be transposed when copied.

For example:

```text
Hugging Face Conv1D: [in_features, out_features]
PyTorch Linear:      [out_features, in_features]
```

This is handled by the custom weight loader.

## Testing

The project includes tests for:

* QKV weights
* QKV outputs
* attention head shapes
* attention output
* MLP output
* LayerNorm
* complete decoder blocks
* all 12 Transformer layers
* final model logits
* causal behavior
* multiple batch sizes and sequence lengths

Run the tests with:

```bash
pytest -v
```

The full model output is compared directly against Hugging Face GPT-2.

## Text generation

The model also supports simple greedy generation.

At each step:

```text
current sequence
      ↓
model forward pass
      ↓
take logits from the last token
      ↓
choose the highest-scoring token
      ↓
append it to the sequence
      ↓
repeat
```

Run the demo with:

```bash
python scripts/demo_generate.py
```

## Project structure

```text
src/tiny_transformer/
├── block.py
├── config.py
├── activations.py
├── transfer_model_param.py
├── generation.py
└── embeddings_map.py

tests/
├── conftest.py
├── test_attention.py
├── test_block.py
├── test_mlp.py
├── test_model.py
└── ...

scripts/
└── demo_generate.py
```

## Current scope

This project focuses on understanding and reproducing GPT-2 inference.

It does not currently include:

* training
* KV cache
* optimized generation
* top-k or top-p sampling
* FlashAttention
* quantization
* distributed inference

These may be explored later.

## Versions

### v0.1-gpt2-baseline

Full GPT-2 forward pass with pretrained weight loading and parity tests against Hugging Face.

### v0.2-generation

Adds greedy autoregressive text generation.

## Next steps

The next phase of the project is focused on performance.

Planned work includes:

* benchmarking the current implementation
* profiling GPU execution
* testing `torch.compile`
* testing PyTorch SDPA
* adding KV caching
* studying memory usage
* experimenting with Triton kernels

The goal is to move from:

```text
understanding the model
        ↓
proving correctness
        ↓
measuring performance
        ↓
optimizing bottlenecks
```
