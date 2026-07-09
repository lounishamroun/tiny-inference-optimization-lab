""" Glossary 
Decoder (GPT STYLE)
    Shapes:
        n_layers = 4 : Number of transformer blocks.
        d_model = 768 : Correponds to the length of our embeddings.
        n_heads = 12
        head_dim = 64
        batch_size = 1
        seq_length = 5 : Number of tokens extracted from the sequence 
        context_length = 1024
        vocab_size = 50257
    Euristics:
        dropout probability = 0.1
"""

from . import data_loader,embeddings_map
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings

_N_HEADS=12
_HEAD_DIM=64
_D_EXPANSION=3072

""" 
Returns an array of [Q,K,V] matrices each of shape [d_model,d_model] with randomly initialized parameters.
"""
class QKVProjection(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model=d_model
        self.Qw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Kw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Vw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        
    def forward(self,embeddings:torch.tensor):
        Q=self.Qw(embeddings)
        K=self.Kw(embeddings)
        V=self.Vw(embeddings)
        
        return Q,K,V
    

""" 
Takes one embedding projection [Q,K,V] to turn it into a multi-head tensor
input: [Q,K,V] projection of shape: [d_model,d_model]
output:[Q,K,V] projection of shape : ([B,T,(h),head_dim]) (h) being the number of heads
"""
class MultiHead():
    def __init__(self,n_heads,head_dim):
        self.n_heads=n_heads
        self.head_dim=head_dim
        
    def multi_head_proj(self,Q,K,V):
        qkv_array=[Q,K,V]
        proj_reshape=[]
        
        """ Assertions """
        B,T,d_model=Q.shape[0],Q.shape[-2],Q.shape[-1]
        assert self.n_heads*self.head_dim==d_model,f"Can't reshape model dimension, model dimension = {n_heads*head_dim} => n_head x head_dim must be equal to d_model"

        """ Reshaping """
        for proj in qkv_array:
                multi_head_projection=torch.reshape(proj,(B,T,self.n_heads,self.head_dim))
                proj_reshape.append(multi_head_projection) 
                
        """ Assertions """
        assert len(proj_reshape)==3, f"Tuple must contain 3 tensors not {len(proj_reshape)}"
        assert proj_reshape[0].shape==proj_reshape[1].shape==proj_reshape[2].shape
        mh_Q,mh_K,mh_V=proj_reshape
        return mh_Q,mh_K,mh_V
    """ 
    Input : Q,K,V heads of shape => [B, T, (h), d_head]
    Output : Merged heads of shape => [B, T, d_model]
    """
    def head_wise_attention_compute(self,mh_Q,mh_K,mh_V):
        batch_size = mh_Q.shape[0]
        seq_length = mh_Q.shape[1]
        n_heads = mh_Q.shape[-2]
        head_dim = mh_Q.shape[-1]
        d_model = n_heads * head_dim

        m = nn.Softmax(dim=-1)

        # Q, K, V initially: [B, T, H, Dh]
        # Move to attention-friendly layout.
        mh_Q = torch.movedim(mh_Q, (1, 2), (2, 1))  # [B, H, T, Dh]
        mh_K = torch.movedim(mh_K, (1, 2), (2, 1))  # [B, H, T, Dh]
        mh_V = torch.movedim(mh_V, (1, 2), (2, 1))  # [B, H, T, Dh]

        assert mh_Q.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])
        assert mh_K.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])
        assert mh_V.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])

        # Attention scores: [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = mh_Q @ mh_K.transpose(-2, -1)

        assert scores.shape == torch.Size([batch_size, n_heads, seq_length, seq_length])

        scaled_scores = scores / math.sqrt(head_dim)

        # Causal mask: True where key position j is in the future of query position i.
        # Shape: [T, T], broadcastable to [B, H, T, T]
        mask = torch.ones(
            (seq_length, seq_length),
            device=scaled_scores.device,
            dtype=torch.bool,
        )
        mask = torch.triu(mask, diagonal=1)

        # Replace future-token logits with -inf.
        masked_scores = scaled_scores.masked_fill(mask, float("-inf"))

        # Softmax over key-token dimension.
        softmax_scores = m(masked_scores)

        assert softmax_scores.shape == torch.Size([batch_size, n_heads, seq_length, seq_length])

        # Check each attention row sums to 1.
        row_sums = softmax_scores.sum(dim=-1)
        ones = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-6), (
            f"Attention rows do not sum to 1. "
            f"max diff = {(row_sums - ones).abs().max().item()}"
        )

        # Check future positions have zero probability after softmax.
        future_weights = softmax_scores.masked_select(mask)
        assert torch.allclose(
            future_weights,
            torch.zeros_like(future_weights),
            atol=1e-6,
        ), f"Future tokens are receiving attention. max={future_weights.max().item()}"

        # Attention output: [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        attention_matrix = softmax_scores @ mh_V

        assert attention_matrix.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])

        # Merge heads:
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        attention_matrix = torch.movedim(attention_matrix, (1, 2), (2, 1))
        attention_matrix = attention_matrix.reshape(batch_size, seq_length, d_model)

        assert attention_matrix.shape == torch.Size([batch_size, seq_length, d_model])

        # Sanity check for future optimization.
        if not attention_matrix.is_contiguous():
            warnings.warn("attention_matrix is not contiguous", UserWarning)

        return attention_matrix
    

"""
Input : Merged heads of shape => [B, T, d_model]
"""
class FeedForward(nn.Module):
    def __init__(self,d_model,d_expansion):
        super().__init__()
        self.augmented=nn.Linear(in_features=d_model,out_features=d_expansion)
        self.activation=nn.GELU()
        self.reduced=nn.Linear(in_features=d_expansion,out_features=d_model)
        self.layer_norm=nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self,x):
        mlp_input=x
        layer_norm=self.layer_norm.to(x.device)
        
        """ MLP """
        x=self.dropout(x)
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
        return x

""" 
INPUT : Embeddings
Output : Embedding matrix of shape => [batch_size,seq_length,d_model] 

"""
class TinyDecoderBlock(nn.Module):
    pass
    """
    def __init__(self,text):
        super().__init__()
        self.input_text=text
    """    

if __name__=="__main__":
    
    if torch.cuda.is_available():
        DEVICE="cuda"
    else:
        DEVICE="cpu"

    """ 
    Retreiving embeddings for an input text sequence
    Output shape => [B,T,d_model] 
    """

    INPUT_TEXT = data_loader.return_text("data/text.txt")
    embeddings=embeddings_map.TokenToEmbedding(INPUT_TEXT,device=DEVICE).map_embeddings()
    
    """ Perform Q,K,V projection and output each Q,K,V matrices """
    d_model=embeddings.shape[-1]
    proj_obj=QKVProjection(d_model).to(DEVICE)
    Q,K,V=proj_obj(embeddings=embeddings) 
    
    """ Turns each Q,K,V matrices into a multi-head paradigm"""
    multi_head=MultiHead(n_heads=_N_HEADS,head_dim=_HEAD_DIM)
    mh_Q,mh_K,mh_V=multi_head.multi_head_proj(Q,K,V)
    attention_matrix=multi_head.head_wise_attention_compute(mh_Q,mh_K,mh_V)
    print(attention_matrix.shape)
    

    

    
    
    


    



    
