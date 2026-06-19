""" Glossary 
Decoder (GPT STYLE)
    Shapes:
        n_layers = 4 : Number of transformer blocks.
        d_model = 768 : Correponds to the length of our embeddings.
        n_heads = 4
        head_dim = 64
        B = 1 : Batch size
        T = 5 : Sequence length (in this case 5 words)
        context_length = 512
        vocab_size = 50257
"""

from . import data_loader  
import transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer,PreTrainedConfig,PythonBackend,AutoModel
import torch 
from torch import nn
import json
import requests as r
import numpy as np
import pandas as pd

INPUT_TEXT = data_loader.return_text("data/text.txt")

tokenizer_config=PreTrainedConfig(output_hidden_states=True,output_attentions=True)
model=AutoModel.from_pretrained("openai-community/gpt2")


def tokenize_text(INPUT_TEXT,tokenizer_config=None):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt") #=> Tokenizes each word
    return token_ids['input_ids'][0]

def ids_to_embeddings(token_ids):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            word_ids (`tensor[int]`):
                Token IDs corresponding to each tokens of our text sequence.
            
        Returns => shape:[s,d_model] :
            `?`: Returns the list of embeddings corresponding to each of our sequence of tokens.
    """
    T=len(token_ids)
    
    token_embedding_weights=next(model.named_parameters("wte"))[1]
    token_embedding_obj=nn.Embedding.from_pretrained(token_embedding_weights)
    positional_weights=next(model.named_parameters("wpe"))[1]
    positional_embedding_obj=nn.Embedding.from_pretrained(positional_weights) 

    
    tok_embed=token_embedding_obj(token_ids) #shape=([T, d_model]) | type:Torch.Tensor
    pos_embed=positional_embedding_obj(torch.arange(start=0,end=T)) #shape=([T, d_model]) | type:Torch.Tensor
    full_embed=tok_embed+pos_embed #shape=([T, d_model]) | type:Torch.Tensor
       
    return full_embed.unsqueeze(0)


token_ids_test=tokenize_text(INPUT_TEXT=INPUT_TEXT)
embedding_for_seq=ids_to_embeddings(token_ids=token_ids_test)
print(embedding_for_seq.shape)



    
