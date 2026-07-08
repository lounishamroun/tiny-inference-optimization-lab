Ok so the next step seem to be the MLP which will increase dimension and squish the dimension back.

So I created a class "mlp", which will take the residual as an argument and merge it somehow with the embeddings which hadn't projection/attention applied to. 

So here in the main I will create a residual variable: 
```python
embeddings=ids_to_gpt2_input_embeddings(token_ids=token_ids,model=global_model)
reisudal=embeddings
```

Ok so in the paper we have the following sentence :
```text
We employ a residual connection [11] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself
```

So I think we should get back and change our  head_wise_attention_compute(qkv_proj) function in order to output LayerNorm(residual + original function output), oh no wait let's create a function which merges residuals and output and does layer norm so that we handle it in the main to better seperate concerns, having something like that :

residual=embeddings
...... (other lines of code)
qkv_attention=head_wise_attention_compute(qkv_proj)
LayerNormConcat(qkv_attention,residual)