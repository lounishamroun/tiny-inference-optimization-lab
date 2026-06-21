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


model=AutoModel.from_pretrained("openai-community/gpt2")
model=model.to(DEVICE)

def tokenize_text(INPUT_TEXT):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt")['input_ids'] #=> Converts text into token IDs.
    token_ids=token_ids.to(DEVICE)
    return token_ids #shape [B,T]

def ids_to_gpt2_input_embeddings(token_ids):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            word_ids =>  dtype: torch.Tensor | shape:[B, T] :
                meaning: vocabulary indices
                
        Returns:
            Type: `torch.Tensor` | shape:[B,T,d_model] :
                List of embeddings corresponding to each of our sequence of tokens.
    """
    B, T = token_ids.shape
    token_embedding_module = model.get_input_embeddings()
    position_embedding_module = model.wpe
    
    #### Token Embeddings ####
    tok_embeddings=token_embedding_module(token_ids)
    tok_embeddings=tok_embeddings.to(DEVICE)
    print(f'tok embedding shape {tok_embeddings.shape}')
    
    #### Positional Embeddings ####
    pos_embeddings=position_embedding_module(torch.arange(start=0,end=T).to(DEVICE))
    pos_embeddings=pos_embeddings.to(DEVICE)
    print(f'pos_embeddings shape {pos_embeddings.shape}')
    
    d_model=tok_embeddings.shape[-1]
    print(f'd model {d_model}')
    
    #### Final Embedding ####
    final_embedding=tok_embeddings+pos_embeddings  #shape=([B,T, d_model]) | type:Torch.Tensor 
    print(f'final embedding {final_embedding.shape}')
    assert final_embedding.shape == torch.Size([B, T, d_model]), f'Shape is {B, T, d_model}'
    return final_embedding 

token_ids_test=tokenize_text(INPUT_TEXT=INPUT_TEXT)
embedding_for_seq=ids_to_gpt2_input_embeddings(token_ids=token_ids_test)
print(embedding_for_seq)



    
