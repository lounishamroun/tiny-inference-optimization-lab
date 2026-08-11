import src.tiny_transformer
from src.tiny_transformer import data_loader,embeddings_map,get_model_param,config
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
        

print(dir(config.GPT2Config))

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model=model.to(DEVICE)

print(model)

print(
    f'{type(model.get_submodule("transformer.h.0.mlp.act"))} '
    f'VS {type(model.transformer.h[0].mlp.act)}'
)
        
""" Retreiving embeddings of our text sequence"""
with torch.no_grad():
    tokenizer=embeddings_map.TokenToEmbedding("My favourite Italian food is",model,device="cuda:0")
    source_input_embeddings=tokenizer.map_embeddings().detach()

    attention_module=model.transformer.h[0].attn #droping to the attention class level
    #query, key, value = attention_module.c_attn(source_input_embeddings).split(attention_module.split_size, dim=2)
    


    