import pytest
import torch
from src.tiny_transformer import data_loader,embeddings_map
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.tiny_transformer.block import gpt2_parameter_load_helper,CausalSelfAttention


class TestMlp():
    
    def test_mlp_params(self,custom_model,reference_block):
            """Reference Parameters"""
            ref_up_proj_wgt=reference_block.mlp.c_fc.weight
            ref_up_proj_bias=reference_block.mlp.c_fc.bias
            ref_down_proj_wgt=reference_block.mlp.c_proj.weight
            ref_down_proj_bias=reference_block.mlp.c_proj.bias
            
            """Custom Block's Parameters"""
            cus_up_proj_wgt=custom_model.mlp.up_proj.weight
            cus_up_proj_bias=custom_model.mlp.up_proj.bias
            cus_down_proj_wgt=custom_model.mlp.down_proj.weight
            cus_down_proj_bias=custom_model.mlp.down_proj.bias
            
            """Assertions""" #We should not forget to transpose our parameters since I'm using nn.Linear while the HuggingFace's implementation uses Conv1D
           #We apply "0 tolerance" assertions since the copied parameters should be an exact match with our reference model.
            
            
            torch.testing.assert_close(
                        cus_up_proj_wgt,
                        ref_up_proj_wgt.T,
                        rtol=0,
                        atol=0,
                        )
        
            torch.testing.assert_close(ref_up_proj_bias,cus_up_proj_bias,rtol=0,atol=0)
            
            torch.testing.assert_close(
                        cus_down_proj_wgt,
                        ref_down_proj_wgt.T,
                        rtol=0,
                        atol=0)
            
            torch.testing.assert_close(ref_down_proj_bias,cus_down_proj_bias,rtol=0,atol=0)
            
    def test_activation(self,custom_model,reference_block,reference_input_embeddings):
        upscaled_embeddings=reference_block.mlp.c_fc(reference_input_embeddings)
        
        ref_activation=reference_block.mlp.act(upscaled_embeddings)
        custom_activation=custom_model.mlp.activation(upscaled_embeddings)
        
        torch.testing.assert_close(
                                ref_activation,
                                custom_activation,
                                rtol=1e-5,
                                atol=1e-6)
            
    
    def test_projection(self,custom_model,reference_block,reference_input_embeddings):
        
        #Maximum absolute difference: approximately 3e-6 in float32
        #Cause: GPT-2 Conv1D uses addmm, while nn.Linear dispatches through F.linear

        reference_fc = reference_block.mlp.c_fc(reference_input_embeddings)
        custom_fc = custom_model.mlp.up_proj(reference_input_embeddings)
    
        
        reference_proj = reference_block.mlp.c_proj(reference_fc)
        custom_proj = custom_model.mlp.down_proj(reference_fc) #error here
        
        torch.testing.assert_close(reference_fc,custom_fc,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(reference_proj,custom_proj,rtol=1e-5,atol=1e-5)

    
    @torch.inference_mode()
    def test_mlp_forward(
        self,
        reference_block,
        reference_input_embeddings,
        custom_model,
    ):
     

        actual_output=custom_model.mlp(reference_input_embeddings)
        expected_output=reference_block.mlp(reference_input_embeddings) 
        torch.testing.assert_close(expected_output,actual_output,rtol=1e-5,atol=1e-5)
         


    