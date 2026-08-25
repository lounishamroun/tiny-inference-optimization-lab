import src.tiny_transformer
from src.tiny_transformer import data_loader,embeddings_map,config,block

from tiny_transformer import transfer_model_param
from transformers import GPT2Config, GPT2Model,initialization 
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM,GPT2PreTrainedModel
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
    
"""

for var in (block.TinyModel(conf)).state_dict():
    print(var)
"""


"""REF"""
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model=model.to(DEVICE)

custom_model=block.TinyModel(config.GPT2CustomConfig()).to(DEVICE)

"""CUSTOM"""

weight_tying = {
    "ln_1.weight": "layer_norm_1.weight",
    "ln_1.bias": "layer_norm_1.bias",
    "attn.c_attn.weight": "attention.qkv_proj.weight",
    "attn.c_attn.bias": "attention.qkv_proj.bias",
    "attn.c_proj.weight": "attention.final_projection.weight",
    "attn.c_proj.bias": "attention.final_projection.bias",
    "ln_2.weight": "layer_norm_2.weight",
    "ln_2.bias": "layer_norm_2.bias",
    "mlp.c_fc.weight": "mlp.up_proj.weight",
    "mlp.c_fc.bias": "mlp.up_proj.bias",
    "mlp.c_proj.weight": "mlp.down_proj.weight",
    "mlp.c_proj.bias": "mlp.down_proj.bias",
}




with torch.no_grad():
    for i in range(4):
        current_model_block=model.transformer.h[i]
        current_custom_block=custom_model.h[i]
        for key,value in weight_tying.items():
            if current_custom_block.get_parameter(value).shape!= current_model_block.get_parameter(key).shape:
                if current_custom_block.get_parameter(value).shape[0] != current_model_block.get_parameter(key).shape[0]:
                    reshaped_param=current_model_block.get_parameter(key).T
                    current_custom_block.get_parameter(value).copy_(reshaped_param)
            else:
                current_custom_block.get_parameter(value).copy_(current_model_block.get_parameter(key))
            
            assert torch.allclose(current_custom_block.get_parameter(value),current_model_block.get_parameter(key))
    

    