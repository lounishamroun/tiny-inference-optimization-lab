import torch
from torch import nn
import math


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in the HuggingFace official repo : https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L70
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))