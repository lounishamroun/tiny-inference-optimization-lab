import pytest
import torch
from src.tiny_transformer import data_loader,embeddings_map
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.tiny_transformer.block import gpt2_parameter_load_helper,CausalSelfAttention
from .reference_harness import *


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"


class TestAttention():
    
    def test_qkv_proj_params(self,custom_model,reference_model):
        src_init_proj_weights=reference_model.transformer.h[0].attn.c_attn.weight
        src_init_proj_bias=reference_model.transformer.h[0].attn.c_attn.bias

        cus_init_proj_weights=custom_model.attention.qkv_proj.weight
        cus_init_proj_bias=custom_model.attention.qkv_proj.bias
    
        assert torch.allclose(src_init_proj_weights.T,cus_init_proj_weights)
        assert torch.allclose(src_init_proj_bias.T,cus_init_proj_bias)
    
        
    def test_qkv_output(self,custom_model,reference_input_embeddings,reference_model):
        """Extract QKV from our custom model"""
        batch_size,seq_length,_=reference_input_embeddings.shape
        with torch.no_grad():
         custom_attention=custom_model.attention._qkv_projection_helper(reference_input_embeddings,batch_size,seq_length)[0]
        #torch.Size([1, 5, 3, 768]) shape
        
        """Extract QKV from our reference model"""
        
        
    
    def test_attention(self):
        pass
    
