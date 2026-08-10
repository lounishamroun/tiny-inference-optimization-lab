from typing import Any

from transformers import AutoModelForCausalLM
from .get_model_param import gpt2_parameter_load_helper
from dataclasses import dataclass


@dataclass
class GPT2Config:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.gpt2_params=gpt2_parameter_load_helper(self.model) 
        self.d_expansion=3072
        self.d_model= 768
        self.n_heads= 12
        self.num_layers= 4
        
print(GPT2Config().gpt2_params)

__all__ = ["GPT2Config"]

