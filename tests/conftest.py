import pytest

from src.tiny_transformer import data_loader,embeddings_map,block,get_model_param,config
from src.tiny_transformer.config  import GPT2CustomConfig 
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM,GPT2Config,GPT2Model
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings
    

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")    

@pytest.fixture(scope="session")
def reference_model(device):
        
    """ 
    Retreiving embeddings for an input text sequence
    Output shape => [B,T,d_model] 
    """

    reference_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    reference_model = reference_model.to(device)
    reference_model.eval() #to ensure deterministic results
    
    return reference_model

@pytest.fixture(scope="session")
def reference_param(reference_model):
    """ Retreiving essential parameters from our source model """
    gpt_2_params=get_model_param.gpt2_parameter_load_helper(reference_model)
    
    return gpt_2_params


@pytest.fixture(scope="session")
def input_text():
    return "My favourite Italian food is"

@pytest.fixture(scope="session")
def reference_input_embeddings(reference_model,input_text,device):
        
    """ Retreiving embeddings of our text sequence"""
    with torch.no_grad():
        tokenizer=embeddings_map.TokenToEmbedding(input_text,reference_model,device=device)
        source_input_embeddings=tokenizer.map_embeddings().detach()
    
    return source_input_embeddings

@pytest.fixture(scope="session")
def reference_block(reference_model):
    reference_block = reference_model.transformer.h[0]
    reference_block.eval()
    return reference_block


@pytest.fixture(scope="session")
def get_batch_seq_dim(reference_input_embeddings):
    batch_size,seq_length,_=reference_input_embeddings.shape
    return [batch_size,seq_length]



GPT2CustomConfig
@pytest.fixture(scope="session")
def conf():
    conf=GPT2CustomConfig()
    return conf


@pytest.fixture(scope="session")
def custom_block(device,conf):
    custom_block=block.TinyDecoderBlock(conf,layer_id=0).to(device)
    custom_block.eval()
    return custom_block
    
@pytest.fixture(scope="session")
def custom_attention(reference_input_embeddings,custom_block,reference_block,device):
    with torch.inference_mode():
        
        l1_norm_embeddings=reference_block.ln_1(
                reference_input_embeddings
            )

        custom_attention=custom_block.attention(l1_norm_embeddings)
        
    return custom_attention

@pytest.fixture(scope="session")
def custom_attention():
    parameter_mapping = {
        "ln_1.weight": ("layer_norm_1.weight", False),
        "ln_1.bias": ("layer_norm_1.bias", False),

        "attn.c_attn.weight": ("attention.qkv_proj.weight", True),
        "attn.c_attn.bias": ("attention.qkv_proj.bias", False),

        "attn.c_proj.weight": ("attention.final_projection.weight", True),
        "attn.c_proj.bias": ("attention.final_projection.bias", False),

        "ln_2.weight": ("layer_norm_2.weight", False),
        "ln_2.bias": ("layer_norm_2.bias", False),

        "mlp.c_fc.weight": ("mlp.up_proj.weight", True),
        "mlp.c_fc.bias": ("mlp.up_proj.bias", False),

        "mlp.c_proj.weight": ("mlp.down_proj.weight", True),
        "mlp.c_proj.bias": ("mlp.down_proj.bias", False),
    }
    return parameter_mapping

    