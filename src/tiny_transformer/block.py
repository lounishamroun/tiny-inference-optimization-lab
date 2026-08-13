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

from . import data_loader,embeddings_map,get_model_param,config
from .get_model_param import gpt2_parameter_load_helper
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
        

#TO DO : causal mask test

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads,head_dim,qkv_proj_wgt,qkv_proj_bias,qkv_final_proj_wgt,qkv_final_proj_bias):
        super().__init__()
        self.n_heads=n_heads
        self.head_dim=head_dim
        self.d_model=d_model
    
        """Injecting GPT-2 parameters into the QKV projection layer"""
        self.qkv_proj=nn.Linear(in_features=self.d_model,out_features=3*self.d_model)

        with torch.no_grad():
            self.qkv_proj.weight.copy_(qkv_proj_wgt.T) #keep the same identity
            self.qkv_proj.bias.copy_(qkv_proj_bias) 
        
        """Final QKV projection"""
        self.final_projection=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        with torch.no_grad():
            self.final_projection.weight.copy_(qkv_final_proj_wgt.T)
            self.final_projection.bias.copy_(qkv_final_proj_bias)
        
        
    def _qkv_projection_helper(self,embeddings,batch_size,seq_length):
        """ 
        Returns Q,K,V matrices
        [
            Q shaped [B, T, H, Dh],
            K shaped [B, T, H, Dh],
            V shaped [B, T, H, Dh],
        ]
        """
            
        """ Unified projection """
        qkv=self.qkv_proj(embeddings)  #[batch_size,seq_length,d_model] @ [d_model,3*d_model] => Each token has it's QKV weighted matrices.
        qkv=torch.reshape(qkv, (batch_size,seq_length,3,self.d_model))
        #e.g: torch.Size([1, 7, 2304]) for a sequence containing 7 tokens.
        """ Q,K,V projections """
        Q,K,V=qkv[:,:,0,:],qkv[:,:,1,:],qkv[:,:,2,:]
    
        
        proj_reshape=[]
        for proj in [Q,K,V]:  #for each projection
            """ Assertions """
            assert self.n_heads*self.head_dim==self.d_model,f"Can't reshape model dimension, model dimension = {self.n_heads*self.head_dim} => n_head x head_dim must be equal to d_model"
            
            """ Multi-Head reshape """
            multi_head_projection=torch.reshape(proj,(batch_size,seq_length,self.n_heads,self.head_dim))
            proj_reshape.append(multi_head_projection) 

        assert len(proj_reshape)==3, f"Tuple must contain 3 tensors not {len(proj_reshape)}"
        assert proj_reshape[0].shape==proj_reshape[1].shape==proj_reshape[2].shape
       
       
        return proj_reshape #Per proj multi-heads  
    
    
    def _causal_attention_helper(self,multi_head_proj,batch_size,seq_length):
        m = nn.Softmax(dim=-1)
        mh_Q,mh_K,mh_V = multi_head_proj

        # Q, K, V initially: [B, T, H, Dh]
        # Move to attention-friendly layout.
        mh_Q = torch.movedim(mh_Q, (1, 2), (2, 1))  # [B, H, T, Dh]
        mh_K = torch.movedim(mh_K, (1, 2), (2, 1))  # [B, H, T, Dh]
        mh_V = torch.movedim(mh_V, (1, 2), (2, 1))  # [B, H, T, Dh]


        """ Multi-Head Assertions """
        assert mh_Q.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])
        assert mh_K.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])
        assert mh_V.shape == torch.Size([batch_size, self.n_heads, seq_length, self.head_dim])

        """ Attention Score """
        # Attention scores: [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = mh_Q @ mh_K.transpose(-2, -1)

        """ Attention Assertions """
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
        assert torch.allclose(row_sums, ones, atol=1e-6), ( #TO DO : Move into test for benchmarking
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
        
        
    def forward(self,embeddings:torch.tensor):

        """ I/ Q, K, V Projection """
        batch_size,seq_length,_=embeddings.shape
        multi_head_proj=self._qkv_projection_helper(embeddings=embeddings,batch_size=batch_size,seq_length=seq_length)
        
        """ II/ Attention Compute """
        causal_attention=self._causal_attention_helper(multi_head_proj,batch_size=batch_size,seq_length=seq_length)
        
        return causal_attention
        
"""
Input : Merged heads of shape => [B, T, d_model]
"""
class FeedForward(nn.Module):
    def __init__(self,d_model,d_expansion,gpt2_up_proj_wgt,gpt2_up_proj_bias,gpt2_down_proj_wgt,gpt2_down_proj_bias,new_gelu):
        super().__init__()
        self.d_expansion=d_expansion
        self.up_proj=nn.Linear(in_features=d_model,out_features=self.d_expansion)


        """Setting GPT-2 parameters to our up_proj linear layer"""
        with torch.no_grad():
            self.up_proj.weight.copy_(gpt2_up_proj_wgt.T)
            self.up_proj.bias.copy_(gpt2_up_proj_bias)
        
        """Activation"""
        self.activation=new_gelu
        
        """Setting parameters """
        self.down_proj=nn.Linear(in_features=d_expansion,out_features=d_model)

        
        """Setting GPT-2 parameters to our down_proj linear layer"""
        with torch.no_grad():
            self.down_proj.weight.copy_(gpt2_down_proj_wgt.T)
            self.down_proj.bias.copy_(gpt2_down_proj_bias)
        
        
    
    def forward(self,x):
        mlp_input=x
        x=self.up_proj(x) #[1, 7, 3072]
        """ Expansion Shape Assertion """
        assert x.shape[-1] == self.d_expansion  
        x=self.activation(x)
        x=self.down_proj(x)
        """ Output Shape Assertion """
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

    def __init__(self,config,layer_id):
        super().__init__()
        
        """Retreiving parameters from GPT-2 model"""
        self.config=config
        self.layer_id=None
        
        up_proj_wgt,up_proj_bias,down_proj_wgt,down_proj_bias,qkv_proj_wgt,qkv_proj_bias,l_norm_wgt,l_norm_bias,l_norm2_wgt,l_norm2_bias,qkv_final_proj_wgt,qkv_final_proj_bias,new_gelu,_,_=self.config.gpt2_params
        
        self.d_model=self.config.d_model
        self.n_heads=self.config.n_heads
        self.head_dim=self.config.d_model//self.n_heads
        self.d_expansion=self.config.d_expansion
        
        self.layer_norm_1=nn.LayerNorm(normalized_shape=self.d_model,eps=self.config.layer_norm_epsilon)
        
        with torch.no_grad():
            self.layer_norm_1.weight.copy_(l_norm_wgt)
            self.layer_norm_1.bias.copy_(l_norm_bias)
        
        self.layer_norm_2=nn.LayerNorm(normalized_shape=self.d_model,eps=self.config.layer_norm_epsilon)
        with torch.no_grad():
            self.layer_norm_2.weight.copy_(l_norm2_wgt)
            self.layer_norm_2.bias.copy_(l_norm2_bias)
        
        self.attention=CausalSelfAttention(d_model=self.d_model,
                                           n_heads=self.n_heads,
                                           head_dim=self.head_dim,
                                           qkv_proj_wgt=qkv_proj_wgt,
                                           qkv_proj_bias=qkv_proj_bias,
                                           qkv_final_proj_wgt=qkv_final_proj_wgt,
                                           qkv_final_proj_bias=qkv_final_proj_bias,
                                           )
        
        self.mlp=FeedForward(d_model=self.d_model,
                             d_expansion=self.d_expansion,
                             gpt2_up_proj_wgt=up_proj_wgt,
                             gpt2_up_proj_bias=up_proj_bias,
                             gpt2_down_proj_wgt=down_proj_wgt,
                             gpt2_down_proj_bias=down_proj_bias,
                             new_gelu=new_gelu
                             )        
        
    def forward(self,embeddings):
        """Computing Attention | Contract : [B,T,d_model] => Instance => [B,T,d_model] """
        
        pre_attention_residual=embeddings
        embeddings=self.layer_norm_1(embeddings)
        attention=self.attention(embeddings=embeddings)
        pre_mlp_residual=attention+pre_attention_residual
        ln2_output=self.layer_norm_2(pre_mlp_residual)
        post_mlp=self.mlp(ln2_output)
        output=pre_mlp_residual+post_mlp
        return output
        
        

class TinyModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.h = nn.ModuleList([TinyDecoderBlock(config=config,layer_id=i) for i in range(config.num_layers)])

    def foward(self,hidden_state):
        for layer in self.h:
            hidden_state=layer(hidden_state)
        return hidden_state

 

    


    



    
