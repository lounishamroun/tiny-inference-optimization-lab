import pytest
import torch
from src.tiny_transformer import data_loader,embeddings_map
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.tiny_transformer.block import gpt2_parameter_load_helper,CausalSelfAttention


class TestMlp():
    
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
           
            
            assert type(ref_act)==type(cus_act)
            
            torch.testing.assert_close(ref_up_proj_wgt.T,cus_up_proj_wgt,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(ref_up_proj_bias,cus_up_proj_bias,rtol=1e-5,atol=1e-6)
            
            torch.testing.assert_close(ref_down_proj_wgt.T,cus_down_proj_wgt,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(ref_down_proj_bias,cus_down_proj_bias,rtol=1e-5,atol=1e-6)

    
    @torch.inference_mode()
    def test_mlp_forward(
        self,
        reference_block,
        reference_input_embeddings,
        custom_model,
    ):
        
        actual_output=custom_model.mlp(reference_input_embeddings)
        
        expected_output=reference_block.mlp(reference_input_embeddings) 
        
        
        print(expected_output.dtype, actual_output.dtype)
        print(expected_output.device, actual_output.device)
        
        assert_close_out=torch.testing.assert_close(expected_output,actual_output,rtol=1e-5,atol=1e-6)
        #close_all_out=torch.allclose(expected_output,actual_output,rtol=1e-5,atol=1e-6)
        
        print(f' assert close vs all out : {assert_close_out}')

    