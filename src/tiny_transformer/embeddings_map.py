from . import data_loader
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings

class TokenToEmbedding():
    def __init__(self,INPUT_TEXT,model,device=None) -> None:
        self.device=device
        self.model=model
    # We'll use a pre-trained tokenizer since we'll use quite generic data
        """ Text to Token ids"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="openai-community/gpt2",
            )
        token_ids=self.tokenizer(INPUT_TEXT, return_tensors="pt")['input_ids'] #=> Converts text into token IDs.
        self.token_ids=token_ids.to(self.device)
        
        """ Token ids to GPT-2 compatible embedding"""
        self.model=AutoModel.from_pretrained("openai-community/gpt2",output_hidden_states=True)
        self.model=self.model.to(self.device)
        
    def map_embeddings(self,include_positional=True):
        self.model.eval()
        batch_size, seq_length = self.token_ids.shape 
        token_embedding_module = self.model.wte
        position_embedding_module = self.model.wpe
        
        #### Token Embeddings ####
        tok_embeddings=token_embedding_module(self.token_ids)  # Generates word embeddings for our sequence
        
        #### Positional Embeddings ####
        seq_offset=torch.arange(start=0,end=seq_length)
        seq_offset=seq_offset.to(self.device)
        pos_embeddings=position_embedding_module(seq_offset) # Generates positional embeddings for our sequence
        
        d_model=tok_embeddings.shape[-1]
        
        #### Input Embedding ####
        if include_positional==True:
            full_embeddings=tok_embeddings+pos_embeddings  #shape=([B,T, d_model]) | type:Torch.Tensor
        else:
            full_embeddings=tok_embeddings
        
        assert full_embeddings.shape == torch.Size([batch_size, seq_length, d_model]), f'Shape is {batch_size, seq_length, d_model}'
        return full_embeddings

    def decode(self,token_id):
        token=self.tokenizer.decode(token_id)
        return token
