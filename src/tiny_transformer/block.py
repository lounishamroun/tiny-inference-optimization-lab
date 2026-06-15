""" Transformer attributes 
Decoder (GPT STYLE)
n_layers = 4
d_model = 256
n_heads = 4
context_length = 512
vocab_size =  8k–32k
"""

from . import data_loader  
from tokenizers import Tokenizer
from transformers import AutoTokenizer,PreTrainedConfig,PythonBackend
import torch 
from torch import nn

INPUT_TEXT = data_loader.return_text("data/text.txt")

tokenizer_config=PreTrainedConfig(output_hidden_states=True,output_attentions=True)

def tokenize_text(INPUT_TEXT,tokenizer_config=None):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    tokenized_text=tokenizer(INPUT_TEXT, return_tensors="pt") #=> Tokenizes each word
    return tokenizer,tokenized_text #=> return token's hidden state's IDs.

tokenizer,hidden_state_token_id=tokenize_text(INPUT_TEXT,True)
word_ids=hidden_state_token_id['input_ids']

def decode_ids(tokenizer,word_ids):
    word_ids=word_ids.squeeze(0)
    for id_ in word_ids: 
        embedding=tokenizer.decode(id_.item())
        print(embedding)
        
decode_ids(tokenizer,word_ids)
""" #TO DO : Embedding lookup"""
        
    


""" 
hidden_dim
num_heads
mlp_ratio
dropout=0.0
bias=True
"""