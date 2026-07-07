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
"""

from . import data_loader
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from boilerplates.similarity_test import compare_tensor_pair
import math

_N_HEADS=12
_HEAD_DIM=64

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
        self.Qw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Kw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.Vw=nn.Linear(in_features=self.d_model,out_features=self.d_model)
        
    def forward(self,x:torch.tensor):
        Q=self.Qw(x)
        K=self.Kw(x)
        V=self.Vw(x)
        
        return Q,K,V
    
""" Takes one embedding projection to turn it into a multi-head tensor
    input: [B,T,d_model]
    output: [B,T,n_heads,head_dim]
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
Takes 3 inputs (each corresponding to one Q,K,V head) 
of shape [B, T, (h), d_head] "(h) being the head index"
and output a computed attention for head (h).
"""
def head_wise_attention_compute(qkv_proj):
    Q=qkv_proj[0]
    K=qkv_proj[1]
    V=qkv_proj[2]
    
    batch_size=Q.shape[0]
    seq_length=Q.shape[1]
    d_model=Q.shape[-1]*Q.shape[-2]
    n_heads=Q.shape[-2]
    head_dim=Q.shape[-1]
    
    m = nn.Softmax(dim=-1)
    
    print(f'Q : {Q.shape} * K : {K.shape} ')
    
    Q=torch.movedim(Q,(1,2),(2,1))
    K=torch.movedim(K,(1,2,3),(3,1,2))
    Q_K=Q@K

    print(f"Q K shape : {Q_K.shape}")
    
    for i,j in enumerate(range(seq_length)):
        Q_K[:,:,i,j+1:seq_length]=float("-inf")
    
    print(f"Positional causal mask test :{Q_K}")
    
    Q_K=torch.div(Q_K,math.sqrt(head_dim))
    Q_K=m(Q_K)    
    print(f'V shape:{V.shape} ')
    V=torch.movedim(V,(1,2),(2,1))
    attention_matrix=Q_K@V
    #Merging heads:
        #Shape: [1, 12, 5, 64] => [1, 5, 768]
    attention_matrix=torch.reshape(torch.movedim(attention_matrix,(1,2),(2,1)),(batch_size, seq_length,n_heads*head_dim))
    
    return attention_matrix
    
    

def LayerNorm(x,residual_x):
    pass
    #TO DO : LayerNorm(x + Sublayer(x))

    
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
    d_model=embeddings.shape[2]

    """ Perform Q,K,V projection and output each Q,K,V matrices """
    proj_obj=QKVProjection(d_model).to(DEVICE)
    x_proj=proj_obj(x=embeddings) 
    # OUT: tuple([B,T,d_model],[B,T,d_model],[B,T,d_model]) | 1 head
    
    """ Turns Q,K,V matrices into a multi-head paradigm"""
    # IN: tuple([B,T,d_model],[B,T,d_model],[B,T,d_model]) | 1 head  
    qkv_proj=multi_head_qkv_proj(x_proj)
    
   
    qkv_attention=head_wise_attention_compute(qkv_proj) #TO DO
    # OUT: tuple([B,T,h,d_model],[B,T,h,d_model],[B,T,h,d_model])
   

    
    
    


    



    
