from typing import Any

from dataclasses import dataclass
from .activations import NewGELUActivation


@dataclass
class GPT2CustomConfig:
    def __init__(self):
        self.d_expansion=3072
        self.d_model= 768
        self.n_heads= 12
        self.num_layers= 12
        self.vocab_size = 50257
        self.layer_norm_epsilon = 1e-5
        self.initializer_range = 0.02
        self.activation=NewGELUActivation()
        self.context_length = 1024
        



