import pytest

from src.tiny_transformer import data_loader,embeddings_map,block,get_model_param,config
from src.tiny_transformer.config  import GPT2Config 
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
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



GPT2Config
@pytest.fixture(scope="session")
def conf():
    conf=GPT2Config()
    return conf


@pytest.fixture(scope="session")
def custom_model(device,conf):
    custom_model=block.TinyDecoderBlock(conf,layer_id=0).to(device)
    custom_model.eval()
    return custom_model
    
@pytest.fixture(scope="session")
def custom_attention(reference_input_embeddings,custom_model,reference_block,device):
    with torch.inference_mode():
        
        l1_norm_embeddings=reference_block.ln_1(
                reference_input_embeddings
            )

        custom_attention=custom_model.attention(l1_norm_embeddings)
        
    return custom_attention
   
    