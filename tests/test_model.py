import pytest 
import torch
from torch import nn
from .model_helper import get_output



@torch.inference_mode()
def test_decoder_model(custom_block,reference_block,reference_input_embeddings,conf):
    reference_model_output=get_output(reference_block,reference_input_embeddings,conf)
    custom_model_output=get_output(custom_block,reference_input_embeddings,conf)
    
    torch.testing.assert_close(reference_model_output, custom_model_output, rtol=1e-5, atol=1e-5)