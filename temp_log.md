So we have in input of the projection function: [B,T,d_model]

Where =>  Embeddings [B,T,d_model] => (IN) 'projection function' (OUT) => [B,T,d_model] , [B,T,d_model] , [B,T,d_model] '3 different matrices
having the same shape as the input but with a linear layer '

So each matrices will have one parameter per row and per d_model, 
so the number of parameters should be the following for each matrices: (B * d_model * T) + T "the bias" 


We should to a sanity check in order to check both the shape and the number of parameters and maybe also compare with the OG model.

They seem to also be the presence of a residual connections (1 skipping projection and the other skipping MLPs)