""" Glossary 
Decoder (GPT STYLE)
    Shapes:
        n_layers = 4 : Number of transformer blocks.
        d_model = 768 : Correponds to the length of our embeddings.
        n_heads = 12
        head_dim = 64
        B = 1 : Batch size
        T = 5 : Number of tokens extracted from the sequence 
        context_length = 1024
        vocab_size = 50257
    Euristics:
        dropout probability = 0.1
"""

from . import data_loader
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings

_N_HEADS=12
_HEAD_DIM=64
_D_EXPANSION=3072

def tokenize_text(INPUT_TEXT):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt")['input_ids'] #=> Converts text into token IDs.
    return token_ids #returns token ids + shape(batch_size,sequence_length) /!\ Device = CPU

def ids_to_gpt2_input_embeddings(token_ids,model):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            token_ids =>  dtype: torch.Tensor | shape:[B, T] :
                meaning: vocabulary indices
                
        Returns:
            Type: `torch.Tensor` | shape:[B,T,d_model] :
                List of embeddings corresponding to each of our sequence of tokens.
    """
    #### Model Handle
    model.eval()
    B, T = token_ids.shape
    token_embedding_module = model.wte
    position_embedding_module = model.wpe
    
    #### Token Embeddings ####
    tok_embeddings=token_embedding_module(token_ids)  # Generates word embeddings for our sequence
    
    #### Positional Embeddings ####
    seq_offset=torch.arange(start=0,end=T)
    seq_offset=seq_offset.to(tok_embeddings.device)
    pos_embeddings=position_embedding_module(seq_offset) # Generates positional embeddings for our sequence
    
    d_model=tok_embeddings.shape[-1]
    
    #### Input Embedding ####
    x=tok_embeddings+pos_embeddings  #shape=([B,T, d_model]) | type:Torch.Tensor 
    assert x.shape == torch.Size([B, T, d_model]), f'Shape is {B, T, d_model}'
    return x 

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
    
""" Takes one embedding projection to turn it into a multi-head tensor
    input: [B,T,d_model]
    output: QKV projection of shape : ([B,T,(h),head_dim]) (h) being the number of heads, residual of shape : ([B,T,d_model])  
"""
def multi_head_proj(embedding_projection,n_heads=_N_HEADS,head_dim=_HEAD_DIM):
    B,T,d_model=embedding_projection.shape[0],embedding_projection.shape[1],embedding_projection.shape[2]
    assert n_heads*head_dim==d_model,f"Can't reshape model dimension, model dimension = {n_heads*head_dim} => n_head x head_dim must be equal to d_model"
    multi_head_proj=torch.reshape(embedding_projection,(B,T,n_heads,head_dim))
    return multi_head_proj

def multi_head_qkv_proj(proj):
    proj_reshape=[]
    for x_proj in proj:
        multi_head_projection=multi_head_proj(x_proj)
        proj_reshape.append(multi_head_projection) 
    assert len(proj_reshape)==3, f"Tuple must contain 3 tensors not {len(proj_reshape)}"
    assert proj_reshape[0].shape==proj_reshape[1].shape==proj_reshape[2].shape
    return proj_reshape


""" 
Input : Q,K,V heads of shape => [B, T, (h), d_head]
Output : Merged heads of shape => [B, T, d_model]
"""
def head_wise_attention_compute(qkv_proj):
    Q = qkv_proj[0]
    K = qkv_proj[1]
    V = qkv_proj[2]

    batch_size = Q.shape[0]
    seq_length = Q.shape[1]
    n_heads = Q.shape[-2]
    head_dim = Q.shape[-1]
    d_model = n_heads * head_dim

    m = nn.Softmax(dim=-1)

    # Q, K, V initially: [B, T, H, Dh]
    # Move to attention-friendly layout.
    Q = torch.movedim(Q, (1, 2), (2, 1))  # [B, H, T, Dh]
    K = torch.movedim(K, (1, 2), (2, 1))  # [B, H, T, Dh]
    V = torch.movedim(V, (1, 2), (2, 1))  # [B, H, T, Dh]

    assert Q.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])
    assert K.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])
    assert V.shape == torch.Size([batch_size, n_heads, seq_length, head_dim])

    # Attention scores: [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
    scores = Q @ K.transpose(-2, -1)

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
    attention_matrix = softmax_scores @ V

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
    def __init__(self,residual,d_model,d_expansion):
        super().__init__()
        self.residual=residual
        self.dropout=nn.Dropout(p=0.1)
        self.augmented=nn.Linear(in_features=d_model,out_features=d_expansion)
        self.activation=nn.GELU()
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

    
if __name__=="__main__":
    
    if torch.cuda.is_available():
        DEVICE="cuda"
    else:
        DEVICE="cpu"

    INPUT_TEXT = data_loader.return_text("data/text.txt")
    global_model=AutoModel.from_pretrained("openai-community/gpt2",output_hidden_states=True)
    global_model=global_model.to(DEVICE)
    
    """ Retreive embedding for the sequence | Output shape => [B,T,d_model] | Device = cuda:0 """
    token_ids=tokenize_text(INPUT_TEXT=INPUT_TEXT) #Retreive token IDs
    token_ids=token_ids.to(DEVICE) 
    embeddings=ids_to_gpt2_input_embeddings(token_ids=token_ids,model=global_model)
    
    """ Perform Q,K,V projection and output each Q,K,V matrices """
    d_model=embeddings.shape[2]
    proj_obj=QKVProjection(d_model).to(DEVICE)
    x_proj,residual=proj_obj(x=embeddings) 
    print(f"residual of shape : {residual.shape}")
    # OUT: tuple([B,T,d_model],[B,T,d_model],[B,T,d_model]) | 1 head
    
    """ Turns Q,K,V matrices into a multi-head paradigm"""
    # IN: tuple([B,T,d_model],[B,T,d_model],[B,T,d_model]) | 1 head  
    qkv_proj=multi_head_qkv_proj(x_proj)   
    qkv_attention=head_wise_attention_compute(qkv_proj)

    ff=FeedForward(residual=residual,d_model=d_model,d_expansion=_D_EXPANSION).to(DEVICE)
    final_mlp,residual=ff(qkv_attention)

    

    
    
    


    



    
