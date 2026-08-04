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
    
        
    def test_fused_qkv_output(self,
    custom_model,
    reference_model,
    reference_input_embeddings,
    ):
        reference_block = reference_model.transformer.h[0]

        with torch.inference_mode():
            # Give both projections the exact same input.
            normalized_input = reference_block.ln_1(
                reference_input_embeddings
            )

            expected_qkv = reference_block.attn.c_attn(
                normalized_input
            )

            actual_qkv = custom_model.attention.qkv_proj(
                normalized_input
            )

        assert expected_qkv.shape == actual_qkv.shape

        torch.testing.assert_close(
            actual_qkv,
            expected_qkv,
            rtol=1e-5,
            atol=1e-6,
        )
            
        
    
    def test_attention(self):
        pass
    
