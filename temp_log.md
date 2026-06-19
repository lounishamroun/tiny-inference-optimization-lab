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

Here's my second attempt with fixed batch size:

    word_ids=word_ids
    embedding_list=[]
    for id_ in word_ids: 
        token_id=torch.tensor(id_.item()) #type int
        token_embedding_weights=next(model.named_parameters("wte"))[1]
        token_embedding_obj=nn.Embedding.from_pretrained(token_embedding_weights)
        positional_weights=next(model.named_parameters("wpe"))[1]
        positional_embedding_obj=nn.Embedding.from_pretrained(positional_weights)   
        final_embedding=token_embedding_obj(token_id)+positional_embedding_obj(token_id)
        embedding_list.append(final_embedding)
    
    embedding_tensor=torch.from_numpy(np.array(embedding_list)).unsqueeze(0) #=> torch.Size([1, 5, 768])
    
    return embedding_tensor #return tensor containing the embeddings


token_ids=tokenize_text(INPUT_TEXT=INPUT_TEXT)
embedding_for_seq=ids_to_embeddings(token_ids)
print(embedding_for_seq.shape)