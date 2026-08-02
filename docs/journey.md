# Building the transformer block

`You can access the main checkpoints by looking for commits containing the keyword 'checkpoint'`

To have an overview of how a transformer works, I recommend this website : https://poloclub.github.io/transformer-explainer/

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

The goal here is to compute the dot product reduces/sums over Dh=64 to produce one similarity score for each query-token/key-token pair.

Then spread that information by multiplying by the `V` matrix.

- I used `torch.tril` to create the causal mask. However, this function produces `0` values in the masked positions, whereas they should be `-inf` before the softmax operation, because softmax may still treat `0` as a valid score.

- This wasn’t a mistake, but rather a design issue. I was using a loop to compute attention for each head instead of computing the dot product in a single batched operation.

```python
for i in range(n_heads):
        Q_tmp=Q[:,:,i,:]
        K_tmp=K[:,:,i,:]
        # all following ops were occuring inside the loop
        ...
```

- In general, I should remember to use assertions only during development and prioritize tests when benchmarking, since certain assertions can consume GPU memory.

- One of my mistake was also to use condition on null values to create the causal mask instead of relying on position (indexing).

```python


```


## Transformer block

### Mixing info from previous tokens


```python
attention_matrix = softmax_scores @ mh_V
```

### Random Param Init

Initially, we randomly initialized the parameters (weights). However, we should use the pretrained weights from the GPT-2 architecture; otherwise, our results will be meaningless.

```python
self.Qw = nn.Linear(...)
self.Kw = nn.Linear(...)
self.Vw = nn.Linear(...)
self.final_projection = nn.Linear(...)

self.up_proj = nn.Linear(...)
self.down_proj = nn.Linear(...)
```

Hence, the next predicted word makes no sense.

```text
My favourite italian food is called `abouts`
```

Which is obviously wrong.

We need to use original GPT 2 parameters:

```python
for name, param in model.named_parameters():
    print(f'Name: {name} of shape :  {model.get_parameter(name).shape}')
```
```text
Name: transformer.wte.weight of shape :  torch.Size([50257, 768])
Name: transformer.wpe.weight of shape :  torch.Size([1024, 768])
Name: transformer.h.0.ln_1.weight of shape :  torch.Size([768])
Name: transformer.h.0.ln_1.bias of shape :  torch.Size([768])
```

...

## To put in the article:

```python
source_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
source_model.eval() #to ensure deterministic results
```
We shouldn't forget to set the source model in evaluation mode, to ensure deterministic result, some mechanisms such as dropout could interefer with the output.