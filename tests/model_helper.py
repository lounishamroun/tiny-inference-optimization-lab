import torch
from torch import nn


class MultiLayer(nn.Module):
    def __init__(self,custom_model,reference_model,config) -> None:
        super().__init__()
        self.config=config
        self.custom_model=custom_model
        self.reference_model=reference_model
    
    def forward(self,reference_input_embeddings):
        for layer_idx in range(self.config.num_layers):
            reference_input_embeddings = self.reference_model.transformer.h[layer_idx](reference_input_embeddings)
        reference_hidden_state = self.reference_model.transformer.ln_f(reference_input_embeddings)
        
        custom_hidden_state = self.custom_model(reference_input_embeddings)
        
        return[custom_hidden_state,reference_hidden_state]
        
