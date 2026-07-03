
Ok so I added an assert, even though I'm not sure of its usefulness since the reshape will raise an error
if the dimension doesn't match:

```python
def multi_head_proj(embedding_projection,n_heads=12,head_dim=64):
    B,T,d_model=embedding_projection.shape[0],embedding_projection.shape[1],embedding_projection.shape[2]
    assert multi_head_proj.shape[-2]*multi_head_proj.shape[-1]==d_model,f"Can't reshape model dimension, model dimension = {d_model} | n_head x head_dim = {multi_head_proj.shape[-2]*multi_head_proj.shape[-1]} => n_head x head_dim must be equal to d_model"
    multi_head_proj=torch.reshape(embedding_projection,(B,T,n_heads,head_dim))
    return multi_head_proj
```
