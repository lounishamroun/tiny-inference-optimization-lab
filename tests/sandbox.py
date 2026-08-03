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

custom_model=block.TinyDecoderBlock(
                    d_expansion=3072,
                    d_model=reference_input_embeddings.shape[-1],
                    n_heads=12,
                    gpt2_params=reference_param,
                    ).to(device)

print(type(model.transformer.h[0].attn.c_attn.weight))


custom_model.attention.qkv_proj.bias