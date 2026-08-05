import pytest
import torch
from src.tiny_transformer import data_loader,embeddings_map
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.tiny_transformer.block import gpt2_parameter_load_helper,CausalSelfAttention
from .reference_harness import *


class TestMlp():
    
    @torch.inference_mode()
    def test_mlp_params(self,custom_model,reference_model):
            """Reference Parameters"""
            ref_up_proj_wgt=reference_model.transformer.h[0].mlp.c_fc.weight
            ref_up_proj_bias=reference_model.transformer.h[0].mlp.c_fc.bias
            ref_act=reference_model.transformer.h[0].mlp.act
            ref_down_proj_wgt=reference_model.transformer.h[0].mlp.c_proj.weight
            ref_down_proj_bias=reference_model.transformer.h[0].mlp.c_proj.bias
            
            """Custom Block's Parameters"""
            cus_up_proj_wgt=custom_model.mlp.up_proj.weight
            cus_up_proj_bias=custom_model.mlp.up_proj.bias
            cus_act=custom_model.mlp.activation
            cus_down_proj_wgt=custom_model.mlp.down_proj.weight
            cus_down_proj_bias=custom_model.mlp.down_proj.bias
            
            """Assertions""" #We should not forget to transpose our parameters since I'm using nn.Linear while the HuggingFace's implementation uses Conv1D
            assert torch.allclose(ref_up_proj_wgt.T,cus_up_proj_wgt)
            assert torch.allclose(ref_up_proj_bias,cus_up_proj_bias)
            assert type(ref_act)==type(cus_act)
            assert torch.allclose(ref_down_proj_wgt.T,cus_down_proj_wgt)
            assert torch.allclose(ref_down_proj_bias,cus_down_proj_bias)

    
    @torch.inference_mode()
    def test_mlp_forward(
        self,
        reference_block,
        reference_input_embeddings,
        custom_model,
        custom_attention
    ):
        normalized_input=reference_block.ln_2(custom_attention+reference_input_embeddings)
        
        expected_output=custom_model.mlp(normalized_input)
        actual_output=reference_block.mlp(normalized_input)
        
        
        torch.testing.assert_close(
                    expected_output,
                    actual_output,
                    rtol=1e-5,
                    atol=1e-6,
        )

    