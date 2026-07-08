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
LayerNormConcat(qkv_attention,residual).

So if I remember correctly, there's batch norm and layer norm.

Batch norm prevents distribution shift across batches while layer norm is just normalizing values across a certain dimension.

So here I guess we would normalize accross model parameters to have somehow similar feature distribution.

So I read that while batch normalization is effective it's tied to batch size since the mean and std are sampled from the current batch.

In batch norm : We basically compute stats for each neurons of a particular layer based on aggregated batches stats.
In layer norm: We compute stats across all neurons of a particular layer (instead of per neuron like in batch norm).

So I guess they say "layer norm" doesn't impose constraint since even there's a batch of 1 , you can still computes stats on all hidden states as opposed to batch norm where you would be limited (having to compute stats with only one neuron exemple).


Here's my first attempt at layernorm function:

def LayerNormConcat(x,residual_x,d_model):
    concat=x+residual_x
    layer_norm=nn.LayerNorm(normalized_shape=d_model,device=x.device)
    concat_norm=layer_norm(concat)
    return concat_norm

Now let's get back to the MLP.

In the paper they say this :

`two linear transformations with a ReLU activation in between`

This means that we need an expension layer + ReLu activation + The opposite operation
While the linear transformations are the same across different positions, they use different parameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
df f = 2048

Let's take 3072 for our case (idk if it's good just saw it on a website).

