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
from transformers import BertTokenizer

INPUT_TEXT = data_loader.return_text("data/text.txt")

# We'll use a pre-trained tokenizer since we'll use quite generic data
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
tokenized_text=tokenizer.tokenize(INPUT_TEXT )
print(tokenized_text)





    
