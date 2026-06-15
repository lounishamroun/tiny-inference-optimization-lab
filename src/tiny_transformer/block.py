""" Glossary 
Decoder (GPT STYLE)
    Shapes:
        s=text sequence length
        n_layers = 4 : How much neurons our model has.
        d_model = 256 : Correponds to the length of our embeddings.
        n_heads = 4
        head_dim = 4
        B = 1 : Batch sizee
        T = 5 : Sequence length (in this case 5 words)
        context_length = 512
vocab_size =  8k–32k
"""

from . import data_loader  
from tokenizers import Tokenizer
from transformers import AutoTokenizer,PreTrainedConfig,PythonBackend
import torch 
from torch import nn
import json
import requests as r

INPUT_TEXT = data_loader.return_text("data/text.txt")

tokenizer_config=PreTrainedConfig(output_hidden_states=True,output_attentions=True)

def tokenize_text(INPUT_TEXT,tokenizer_config=None):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt") #=> Tokenizes each word
    return token_ids #=> return token's hidden state's IDs.

def dict_map(token_id,dictionary="https://huggingface.co/gpt2/resolve/main/vocab.json"):
    body=r.get(dictionary)
    json_dictionary=body.json()
    #TO DO : find key for value 'token_id"

def ids_to_embeddings(word_ids):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            word_ids (`tensor[int]`):
                Token IDs corresponding to each tokens of our text sequence.
            
        Returns => shape:[s,d_model] :
            `?`: Returns the list of embeddings corresponding to each of our sequence of tokens.
    """
    embedding_list=[]
    word_ids=word_ids['input_ids'].squeeze(0)
    for id_ in word_ids: 
        embedding=id_
        embedding_list.append(embedding)
    return embedding


hidden_state_token_id=tokenize_text(INPUT_TEXT=INPUT_TEXT,tokenizer_config=tokenizer_config)
ids_to_embeddings(hidden_state_token_id)

    
