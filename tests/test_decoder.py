import torch
import pytest


class TestDecoder():
    @torch.inference_mode()
    def test_decoder_single_block(self,reference_block,custom_model,reference_input_embeddings):
        reference_model_output=reference_block(reference_input_embeddings)
        custom_model_output=custom_model(reference_input_embeddings)
        torch.testing.assert_close(custom_model_output,reference_model_output,rtol=1e-5,atol=1e-5)
        