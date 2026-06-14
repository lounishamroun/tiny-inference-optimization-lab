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
from transformers import AutoTokenizer

INPUT_TEXT = data_loader.return_text("data/text.txt")


def tokenize_text(INPUT_TEXT):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenized_text=tokenizer(INPUT_TEXT, return_tensors="pt") #tensor([[ 464, 3616,  286, 1204,  318]]) => Each word has been tokenized
    return tokenized_text


""" 
hidden_dim
num_heads
mlp_ratio
dropout=0.0
bias=True
"""