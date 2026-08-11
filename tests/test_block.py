import torch
import pytest


class TestBlock():
    @torch.inference_mode()
    def test_decoder_single_block(self,reference_block,custom_block,reference_input_embeddings):
        reference_model_output=reference_block(reference_input_embeddings)
        custom_model_output=custom_block(reference_input_embeddings)
        torch.testing.assert_close(custom_model_output,reference_model_output,rtol=1e-5,atol=1e-6)
    
    def test_layer_norm(self,reference_block,custom_block,reference_input_embeddings):
        custom_ln_1_output=custom_block.layer_norm_1(reference_input_embeddings)
        custom_ln_2_output=custom_block.layer_norm_2(reference_input_embeddings)
        reference_ln_1_output=reference_block.ln_1(reference_input_embeddings)
        reference_ln_2_output=reference_block.ln_2(reference_input_embeddings)
        
        torch.testing.assert_close(custom_ln_1_output,reference_ln_1_output,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(custom_ln_2_output,reference_ln_2_output,rtol=1e-5,atol=1e-6)
        
        