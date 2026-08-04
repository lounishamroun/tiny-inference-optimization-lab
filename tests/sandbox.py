import src.tiny_transformer
from src.tiny_transformer import data_loader,embeddings_map,get_model_param
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
        

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model=model.to(DEVICE)


        
""" Retreiving embeddings of our text sequence"""
with torch.no_grad():
    tokenizer=embeddings_map.TokenToEmbedding("My favourite Italian food is",model,device="cuda:0")
    source_input_embeddings=tokenizer.map_embeddings().detach()

    attention_module=model.transformer.h[0].attn #droping to the attention class level
    #query, key, value = attention_module.c_attn(source_input_embeddings).split(attention_module.split_size, dim=2)
    
    query_states, key_states, value_states = attention_module.c_attn(source_input_embeddings).split(attention_module.split_size, dim=2)
    
    shape_kv = (*key_states.shape[:-1], -1, attention_module.head_dim)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)
    shape_q = (*query_states.shape[:-1], -1, attention_module.head_dim)
    query_states = query_states.view(shape_q).transpose(1, 2)
    
    print(f'Q:{query_states.shape}|K:{key_states.shape}| V:{value_states.shape}')

    