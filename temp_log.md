Ok so let's read the doc of nn.Embedding:
- It's a lookup table which returns an embedding according to the provided index.

Does our model has such object? : Yes:

if we print(model) we retreive those parameters token + positional embeddings:

GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
...
)

We have to retreive the Embedding objects: returning => [B, T, D].

embedding_object=next(model.named_parameters("Embedding"))[0] => This returns the Embedding.wte.weight type, not Embedding object.

```python
weights=next(model.named_parameters("wte"))[1]
embedding = nn.Embedding.from_pretrained(weights)
input = torch.tensor([0])
print(embedding(input))
```

So I managed to access the Embedding object.
