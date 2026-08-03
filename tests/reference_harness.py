import torch


def source_qkv_proj(model):
    init_proj_weights=model.transformer.h[0].attn.c_proj.weight
    init_proj_bias=model.transformer.h[0].attn.c_proj.bias
    return [init_proj_weights,init_proj_bias]


def custom_qkv_proj(model):
    init_proj_weights=model.qkv_final_proj_wgt
    init_proj_bias=model.qkv_final_proj_bias
    return [init_proj_weights,init_proj_bias]

# embeddings => expansion

def source_output(model,input):
    output=model(input)
    return output

def custom_output(model,input):
    output=model(input)
    return output