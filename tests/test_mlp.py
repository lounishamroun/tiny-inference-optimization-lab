import pytest
import torch



class TestMlp():
    
    def test_mlp_params(self,custom_block,reference_block):
            """Reference Parameters"""
            ref_up_proj_wgt=reference_block.mlp.c_fc.weight
            ref_up_proj_bias=reference_block.mlp.c_fc.bias
            ref_down_proj_wgt=reference_block.mlp.c_proj.weight
            ref_down_proj_bias=reference_block.mlp.c_proj.bias
            
            """Custom Block's Parameters"""
            cus_up_proj_wgt=custom_block.mlp.up_proj.weight
            cus_up_proj_bias=custom_block.mlp.up_proj.bias
            cus_down_proj_wgt=custom_block.mlp.down_proj.weight
            cus_down_proj_bias=custom_block.mlp.down_proj.bias
            
            """Assertions""" #We should not forget to transpose our parameters since I'm using nn.Linear while the HuggingFace's implementation uses Conv1D
           #We apply "0 tolerance" assertions since the copied parameters should be an exact match with our reference model.
            
            
            torch.testing.assert_close(
                        cus_up_proj_wgt,
                        ref_up_proj_wgt.T,
                        rtol=0,
                        atol=0,
                        )
        
            torch.testing.assert_close(cus_up_proj_bias,ref_up_proj_bias,rtol=0,atol=0)
            
            torch.testing.assert_close(
                        cus_down_proj_wgt,
                        ref_down_proj_wgt.T,
                        rtol=0,
                        atol=0)
            
            torch.testing.assert_close(cus_down_proj_bias,ref_down_proj_bias,rtol=0,atol=0)
            
    @torch.inference_mode()
    def test_activation(self,custom_block,reference_block,reference_input_embeddings):
        upscaled_embeddings=reference_block.mlp.c_fc(reference_input_embeddings)
        
        ref_activation=reference_block.mlp.act(upscaled_embeddings)
        custom_activation=custom_block.mlp.activation(upscaled_embeddings)
        
        torch.testing.assert_close(
                                custom_activation,
                                ref_activation,
                                rtol=1e-5,
                                atol=1e-6)
        
    @torch.inference_mode()
    def test_projection(self,custom_block,reference_block,reference_input_embeddings):
        
        #Maximum absolute difference: approximately 3e-6 in float32
        #Cause: GPT-2 Conv1D uses addmm, while nn.Linear dispatches through F.linear

        reference_fc = reference_block.mlp.c_fc(reference_input_embeddings)
        custom_fc = custom_block.mlp.up_proj(reference_input_embeddings)
    
        
        reference_proj = reference_block.mlp.c_proj(reference_fc)
        custom_proj = custom_block.mlp.down_proj(reference_fc) 
        
        torch.testing.assert_close(custom_fc,reference_fc,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(custom_proj,reference_proj,rtol=1e-5,atol=1e-5)

    
    @torch.inference_mode()
    def test_mlp_forward(
        self,
        reference_block,
        reference_input_embeddings,
        custom_block,
    ):
    
        actual_output=custom_block.mlp(reference_input_embeddings)
        expected_output=reference_block.mlp(reference_input_embeddings) 
        torch.testing.assert_close(expected_output,actual_output,rtol=1e-5,atol=1e-5)
         


    