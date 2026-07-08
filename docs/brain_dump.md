
Constructive feedback and critic on ingineer intuition, technical choices etc...

So I started by adding the dropout and outputting the residual:

class QKVProjection(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model=d_model
        self.dropout=nn.Dropout(p=0.1)
        self.Qw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Kw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Vw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        
    def forward(self,x:torch.tensor):
        x=self.dropout(x)
        residual=x
        Q=self.Qw(x)
        K=self.Kw(x)
        V=self.Vw(x)
        
        return [Q,K,V],residual

I did the same for the final mlp layer by integrating layer norm, like you said, I forgot there was a learnable parameter in this function and I also integrated dropout:

class FeedForward(nn.Module):
    def __init__(self,residual,d_model,d_expansion):
        super().__init__()
        self.residual=residual
        self.dropout=nn.Dropout(p=0.1)
        self.augmented=nn.Linear(in_features=d_model,out_features=d_expansion)
        self.activation=nn.ReLU()
        self.reduced=nn.Linear(in_features=d_expansion,out_features=d_model)
        self.layer_norm=nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self,x):
        mlp_input=x
        layer_norm=self.layer_norm.to(x.device)
        
        """ MLP """
        x=self.dropout(x)
        x=x+self.residual #residual concat
        residual=x
        x=self.augmented(x)
        x=self.activation(x)
        x=self.reduced(x)
        x=layer_norm(x)
        
        
        """ Assertions """
        assert x.shape == mlp_input.shape #checking invariance.
        
        """ Output 
        MLP forward shape => [batch_size,seq_length,d_model]
        new residual (unaffected by linear layer) shape => [batch_size,seq_length,d_model] 
        """
        return x,residual

However I read your recommendation on managing this the residual and layer norm in another decoder block.

So I guess I should do the same for dropout so that we seperate training data with linear ops.

I will start by merging

def tokenize_text(INPUT_TEXT) and def ids_to_gpt2_input_embeddings(token_ids,model)

So that I directly have a text to embedding function which output will be directly fed to the transformer.

I think its even better to create a dedicated EmbeddingMap class in a seperate module