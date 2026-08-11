import torch
from torch import nn


class MultiLayer(nn.Module):
    def __init__(self,block,config) -> None:
        super().__init__()
        self.conf=config
        self.block=block
        self.layers=nn.ModuleList([block for i in range(self.conf.num_layers)]) 
        self.ln_f=nn.LayerNorm(self.conf.d_model,self.conf.layer_norm_epsilon)
    
    def forward(self,hidden_states):
        for layer in self.layers:
            hidden_states=layer(hidden_states)
        
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states

def get_output(block,hidden_states,config):
    model=MultiLayer(block,config=config).to(hidden_states.device)
    model_output=model(hidden_states)
    return model_output

__all__=['get_output']