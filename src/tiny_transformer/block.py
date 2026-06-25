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

def tokenize_text(INPUT_TEXT):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt")['input_ids'] #=> Converts text into token IDs.
    token_ids=token_ids.to(DEVICE)
    return token_ids #shape [B,T]

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

class q_k_v_proj(nn.Module):
    def __init__(self,d_model,device="cuda:0"):
        super().__init__()
        self.d_model=d_model
        self.Qw=nn.Linear(in_features=self.d_model,out_features=self.d_model).to(device)
        self.Kw=nn.Linear(in_features=self.d_model,out_features=self.d_model).to(device)
        self.Vw=nn.Linear(in_features=self.d_model,out_features=self.d_model).to(device)
        
    def forward(self,x:torch.tensor):
        Q=self.Qw(x)
        K=self.Kw(x)
        V=self.Vw(x)
        
        return Q,K,V
    
""" Takes one embedding projection to turn it into a multi-head tensor
    input: [B,T,d_model]
    output: [B,T,n_heads,head_dim]
"""
def multi_head_proj(embedding_projection,n_heads=12,head_dim=64):
    B,T,d_model=embedding_projection.shape[0],embedding_projection.shape[1],embedding_projection.shape[2]
    assert multi_head_proj.shape[-2]*multi_head_proj.shape[-1]==d_model,f"Can't reshape model dimension, model dimension = {d_model} | n_head x head_dim = {multi_head_proj.shape[-2]*multi_head_proj.shape[-1]} => n_head x head_dim must be equal to d_model"
    multi_head_proj=torch.reshape(embedding_projection,(B,T,n_heads,head_dim))
    return multi_head_proj

""" 
Takes 3 inputs (each corresponding to one Q,K,V head) 
of shape [B, T, (h), d_head] "(h) being the head index"
and output a computed attention for head (h).
"""
def head_wise_attention_compute(head_q,head_k,head_v):
    attention_scores=nn.softmax((head_q@head_k.T)/head_q.shape[-1])
    return attention_scores
        
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
    proj_obj=q_k_v_proj(d_model)
    x_proj=proj_obj(x=embeddings) # Output a tuple containing Q,K,V => tuple([B,T,d_model],[B,T,d_model],[B,T,d_model]) | n_heads = 1
    
    """ Turn it into a multi-head paradigm"""
    proj_reshape=[]
    for proj in x_proj:
        multi_head_projection=multi_head_proj(proj)
        proj_reshape.append(multi_head_projection) 

    Q_reshape,K_reshape,V_reshape=proj_reshape    
    Q_heads,K_heads,V_heads=Q_reshape[0][1],K_reshape[0][1],V_reshape[0][1]
    
    
    """# TO DO , here's how you access a head for one specific projection 
    we should find a way to call the "head_wise_attention_compute" function
    for head,heads in enumerate(Q_heads):
        Q_head_h=heads[head]
    """
    
    #head_wise_attention_compute(head)
    
    
    


    



    
