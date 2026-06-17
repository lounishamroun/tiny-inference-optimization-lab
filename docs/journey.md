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

### Mistake - Positional Embeddings 

Initially I only returned the token embeddings forgetting to take into account positional embeddings.
