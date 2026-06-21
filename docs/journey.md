# Building the transformer block

The first step consists of building a transformer block using PyTorch and HuggingFace.

## Tokenization + Embedding

I manually wrote the function used to retreive the sequence embeddings by taking in input the sequence's token IDs.  

```python
embedding_matrix = next(model.named_parameters("(wte)"))[1]

for id_ in word_ids:
    token_id = id_.item()
    embedding = embedding_matrix[token_id, :]
    embedding_list.append(embedding)
```

However there's an already existing function which does that `torch.nn.Embedding`.

### Experiment: Reconstruct GPT-2 input embeddings manually

Hypothesis:
GPT-2 hidden_states[0] equals token embeddings plus positional embeddings.

Manual formula:
x = wte(input_ids) + wpe(position_ids)

Result:
manual_x shape = [1, 5, 768]
reference shape = [1, 5, 768]
allclose = passed !

Conclusion:
The embedding stage is now understood and verified.

### Mistakes - Tokenization + Embedding

- Initially I only returned the token embeddings forgetting to take into account positional embeddings.
- I squeezed the batch dimension because it was equal to 1. But in practice we should keep it.
- Inspecting objects instead of using the existing public API : `next(model.named_parameters("(wte)"))[1]`.
- Asserting types instead of using `isinstance()`, which is more flexible (e.g., when comparing torch.Tensor subclasses). 