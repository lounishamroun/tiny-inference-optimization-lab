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

def q_k_v_proj(embeddings):
    d_model=embeddings.shape[-1]
    
    Qw=nn.Linear(in_features=d_model,out_features=d_model)
    Kw=nn.Linear(in_features=d_model,out_features=d_model)
    Vw=nn.Linear(in_features=d_model,out_features=d_model)
    
    
    Qw,Kw,Vw=Qw.to(embeddings.device),Kw.to(embeddings.device),Vw.to(embeddings.device)
    
    Q=Qw(embeddings)
    K=Kw(embeddings)
    V=Vw(embeddings)
    
    return Q,K,V

if __name__=="__main__":
    
    if torch.cuda.is_available():
        DEVICE="cuda"
    else:
        DEVICE="cpu"

    INPUT_TEXT = data_loader.return_text("data/text.txt")
    global_model=AutoModel.from_pretrained("openai-community/gpt2",output_hidden_states=True)
    global_model=global_model.to(DEVICE)
    ### Retreive embedding for the sequence | Output shape => [B,T,d_model]
    token_ids=tokenize_text(INPUT_TEXT=INPUT_TEXT) #Retreive token IDs
    token_ids=token_ids.to(DEVICE) 
    embeddings=ids_to_gpt2_input_embeddings(token_ids=token_ids,model=global_model)
    
    ### Perform Q,K,V projection
    x_proj=q_k_v_proj(embeddings=embeddings)
    
    for x in x_proj:
        print(f"Projection type: {type(x)} | Shape proj: {x.shape if isinstance(x, torch.Tensor) else 'not a tensor'}")

    
    


    



    
