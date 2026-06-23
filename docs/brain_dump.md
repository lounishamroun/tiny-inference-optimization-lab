## Q,K,V

So we should put a linear layer to our embeddings (weights and bias for each embedding row)
[B,T,d_model] => [..T,d_model] put weights on it. 

We gonna ignore the batch size at the moment

In entry we have embeddings => [T,d_model],

So each Q, K, V matrices will have the following shape:

[T,d_model] while having (d_model*T)+T parameters so one parameter per embedding dimension + bias for each token










