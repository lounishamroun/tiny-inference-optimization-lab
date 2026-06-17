""" Glossary 
Decoder (GPT STYLE)
    Shapes:
        n_layers = 4 : Number of transformer blocks.
        d_model = 256 : Correponds to the length of our embeddings.
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

INPUT_TEXT = data_loader.return_text("data/text.txt")

tokenizer_config=PreTrainedConfig(output_hidden_states=True,output_attentions=True)
model=AutoModel.from_pretrained("openai-community/gpt2")

print(model.wpe.weight)


def tokenize_text(INPUT_TEXT,tokenizer_config=None):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt") #=> Tokenizes each word
    return len(token_ids['input_ids'].squeeze(0)),token_ids['input_ids'].squeeze(0) #=> return token's hidden state's IDs.


def ids_to_embeddings(word_ids,l_text):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            word_ids (`tensor[int]`):
                Token IDs corresponding to each tokens of our text sequence.
            
        Returns => shape:[s,d_model] :
            `?`: Returns the list of embeddings corresponding to each of our sequence of tokens.
    """
    embedding_list=[]
    word_ids=word_ids
    s=l_text
    d_model=next(model.named_parameters("(wte)"))[1][0,:].shape[0]
    
    assert s==5, f"Wrong sequence dimension => {s}"
    
    embedding_matrix=next(model.named_parameters("(wte)"))[1]
    for id_ in word_ids: 
        token_id=id_.item() #type int
        embedding=embedding_matrix[token_id,:]
        embedding_list.append(embedding)
    
    embedding_matrix=torch.empty([s,d_model])
    

    for idx in range(len(embedding_list)):
        embedding_matrix[idx,:]=embedding_list[idx]
    
    assert embedding_matrix.shape==torch.Size([5, 768]),f"Shape is {embedding_matrix.shape}"
    
    return embedding_matrix #return tensor containing the embeddings


l_text,hidden_state_token_id=tokenize_text(INPUT_TEXT=INPUT_TEXT,tokenizer_config=tokenizer_config)
embedding_matrix=ids_to_embeddings(hidden_state_token_id,l_text)


    
