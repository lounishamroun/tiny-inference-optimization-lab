# Building the transformer block

`You can access the main checkpoints by looking for commits containing the keyword 'checkpoint'`


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

### Mistakes - Attention

- I used `torch.tril` to create the causal mask. However, this function produces `0` values in the masked positions, whereas they should be `-inf` before the softmax operation, because softmax may still treat `0` as a valid score.

- This wasn’t a mistake, but rather a design issue. I was using a loop to compute attention for each head instead of computing the dot product in a single batched operation.

```python
for i in range(n_heads):
        Q_tmp=Q[:,:,i,:]
        K_tmp=K[:,:,i,:]
        # all following ops were occuring the loop
        ...
```