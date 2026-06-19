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
from transformers import AutoTokenizer, PreTrainedConfig, AutoModel


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"

INPUT_TEXT = data_loader.return_text("data/text.txt")

tokenizer_config=PreTrainedConfig()
model=AutoModel.from_pretrained("openai-community/gpt2")
model=model.to(DEVICE)

def tokenize_text(INPUT_TEXT):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt")['input_ids'] #=> Tokenizes each word
    token_ids=token_ids.to(DEVICE)
    return token_ids #shape [B,T]

def ids_to_embeddings(token_ids):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            word_ids => (`torch.Tensor`) | shape:[B,T,d_model] :
                Token IDs corresponding to each tokens of our text sequence.
            
        Returns:
            Type: `torch.Tensor` | shape:[B,T,d_model] :
                List of embeddings corresponding to each of our sequence of tokens.
    """
    embedding_obj=model.get_input_embeddings()
    embedding_obj=embedding_obj.to(DEVICE)
    d_model=embedding_obj.embedding_dim
    T=len(token_ids[0]) 
    final_embedding=embedding_obj(token_ids) #shape=([B,T, d_model]) | type:Torch.Tensor 
    assert final_embedding.shape[-2] == T
    assert final_embedding.shape[-1] == d_model
    assert final_embedding.shape[0] == 1
    return final_embedding #TO DO : Dynamically adapt batch size


token_ids_test=tokenize_text(INPUT_TEXT=INPUT_TEXT)
embedding_for_seq=ids_to_embeddings(token_ids=token_ids_test)
print(type(embedding_for_seq))



    
