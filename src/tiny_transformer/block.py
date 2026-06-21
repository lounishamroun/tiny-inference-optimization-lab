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
from transformers import AutoTokenizer, AutoModel


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"

INPUT_TEXT = data_loader.return_text("data/text.txt")


global_model=AutoModel.from_pretrained("openai-community/gpt2",output_hidden_states=True)
global_model=global_model.to(DEVICE)

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
    token_embedding_module = model.pte
    position_embedding_module = model.wpe
    
    #### Token Embeddings ####
    tok_embeddings=token_embedding_module(token_ids)
    print(f'tok embedding shape {tok_embeddings.shape}')
    
    #### Positional Embeddings ####
    seq_offset=torch.arange(start=0,end=T)
    seq_offset=seq_offset.to(tok_embeddings.device)
    pos_embeddings=position_embedding_module(seq_offset)
    print(f'pos_embeddings shape {pos_embeddings.shape}')
    
    d_model=tok_embeddings.shape[-1]
    print(f'd model {d_model}')
    
    #### Input Embedding ####
    x=tok_embeddings+pos_embeddings  #shape=([B,T, d_model]) | type:Torch.Tensor 
    print(f'final embedding {x.shape}')
    assert x.shape == torch.Size([B, T, d_model]), f'Shape is {B, T, d_model}'
    return x 

token_ids_test=tokenize_text(INPUT_TEXT=INPUT_TEXT)
token_ids_test=token_ids_test.to(DEVICE)
embedding_for_seq=ids_to_gpt2_input_embeddings(token_ids=token_ids_test,model=global_model)
print(embedding_for_seq)



    
