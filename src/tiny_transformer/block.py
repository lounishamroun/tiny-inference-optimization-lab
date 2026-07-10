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


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
        
_N_HEADS=12
_HEAD_DIM=64
_D_EXPANSION=3072

""" 
Returns an array of [Q,K,V] matrices each of shape [d_model,d_model] with randomly initialized parameters.
"""
class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,head_dim):
        super().__init__()
        self.n_heads=n_heads
        self.head_dim=head_dim
        self.d_model=d_model
        self.Qw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Kw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Vw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.final_projection=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        
        
    def forward(self,embeddings:torch.tensor):
        
        batch_size=embeddings.shape[0]
        seq_length=embeddings.shape[1]
        
        """ I. Q, K, V Projection """
        
        Q=self.Qw(embeddings)
        K=self.Kw(embeddings)
        V=self.Vw(embeddings)
        
        
        proj_reshape=[]
        for proj in [Q,K,V]:   
            B,T,d_model=embeddings.shape
            assert self.n_heads*self.head_dim==d_model,f"Can't reshape model dimension, model dimension = {self.n_heads*self.head_dim} => n_head x head_dim must be equal to d_model"
            """ Multi-Head reshape """
            multi_head_projection=torch.reshape(proj,(B,T,self.n_heads,self.head_dim))
            proj_reshape.append(multi_head_projection) 

        assert len(proj_reshape)==3, f"Tuple must contain 3 tensors not {len(proj_reshape)}"
        assert proj_reshape[0].shape==proj_reshape[1].shape==proj_reshape[2].shape
       
       
        mh_Q,mh_K,mh_V=proj_reshape #Per proj multi-heads tensors
        
        
        """ II. Attention Compute """
        
        m = nn.Softmax(dim=-1)

        # Q, K, V initially: [B, T, H, Dh]
        # Move to attention-friendly layout.
        mh_Q = torch.movedim(mh_Q, (1, 2), (2, 1))  # [B, H, T, Dh]
        mh_K = torch.movedim(mh_K, (1, 2), (2, 1))  # [B, H, T, Dh]
        mh_V = torch.movedim(mh_V, (1, 2), (2, 1))  # [B, H, T, Dh]

        assert mh_Q.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])
        assert mh_K.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])
        assert mh_V.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])

        # Attention scores: [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = mh_Q @ mh_K.transpose(-2, -1)

 
        assert scores.shape == torch.Size([batch_size, self.n_heads, seq_length, seq_length])

        scaled_scores = scores / math.sqrt(self.head_dim)

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

        assert softmax_scores.shape == torch.Size([batch_size, self.n_heads, seq_length, seq_length])

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

        assert attention_matrix.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])

        # Merge heads:
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        attention_matrix = torch.movedim(attention_matrix, (1, 2), (2, 1))
        attention_matrix = attention_matrix.reshape(batch_size, seq_length, self.d_model)
        attention_matrix=self.final_projection(attention_matrix)

        assert attention_matrix.shape == torch.Size([batch_size, seq_length, self.d_model])

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
    
    def forward(self,x):
        mlp_input=x
        x=self.augmented(x)
        x=self.activation(x)
        x=self.reduced(x)
        
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

    def __init__(self,d_expansion,d_model,n_heads,head_dim):
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.head_dim=head_dim
        self.d_expansion=d_expansion
        self.layer_norm_1=nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2=nn.LayerNorm(normalized_shape=self.d_expansion)
        self.attention=CausalSelfAttention(d_model=self.d_model,n_heads=self.n_heads,head_dim=self.head_dim)
        self.mlp=FeedForward(d_model=self.d_model,d_expansion=self.d_expansion)        
        
    def forward(self,embeddings):
        """Computing Attention | Contract : [B,T,d_model] => Instance => [B,T,d_model] """
        pre_attention_residual=embeddings
        embeddings=self.layer_norm_1(embeddings)
        attention=self.attention(embeddings=embeddings)
        pre_mlp_residual=attention+pre_attention_residual
        attention=self.layer_norm_2(pre_mlp_residual)
        post_mlp=self.mlp(attention)
        output=pre_mlp_residual+post_mlp
        return output
        

if __name__=="__main__":

    """ 
    Retreiving embeddings for an input text sequence
    Output shape => [B,T,d_model] 
    """
    INPUT_TEXT = data_loader.return_text("data/text.txt")
    embeddings=embeddings_map.TokenToEmbedding(INPUT_TEXT,device=DEVICE).map_embeddings()
    d_model=embeddings.shape[-1]
    """ 
    Transformer block
    Output shape => [B,T,d_model] 
    """
    block=TinyDecoderBlock(d_expansion=_D_EXPANSION,
                     d_model=d_model,
                     n_heads=_N_HEADS,
                     head_dim=_HEAD_DIM).to(DEVICE)
    
    block_output=block(embeddings)

 
    
    


    



    
