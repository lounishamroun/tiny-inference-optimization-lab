import torch


class GPT2WeightLoader():
    def __init__(self,reference_model,custom_model,single_block) -> None:
        self.reference_model=reference_model
        self.custom_model=custom_model
        self.single_block=single_block
        
        if single_block is not None and custom_model is not None:                
            raise Exception("Either a single transformer block or model object should be passed not both")
        elif single_block is None and custom_model is None:  
            raise Exception("Either a single transformer block or model object should be passed")

    def cp_block_level_params(self,layer_idx):
        
        """Up Linear projection"""
        up_proj_wgt=self.reference_model.transformer.h[layer_idx].mlp.c_fc.weight 
        up_proj_bias=self.reference_model.transformer.h[layer_idx].mlp.c_fc.bias      
               
        
        """Down Linear projection"""
        down_proj_wgt=self.reference_model.transformer.h[layer_idx].mlp.c_proj.weight 
        down_proj_bias=self.reference_model.transformer.h[layer_idx].mlp.c_proj.bias
        
        """QKV projection"""
        qkv_proj_wgt=self.reference_model.transformer.h[layer_idx].attn.c_attn.weight
        qkv_proj_bias=self.reference_model.transformer.h[layer_idx].attn.c_attn.bias
        
        """QKV Final Projection"""
        qkv_final_proj_wgt=self.reference_model.transformer.h[layer_idx].attn.c_proj.weight
        qkv_final_proj_bias=self.reference_model.transformer.h[layer_idx].attn.c_proj.bias 
        
        """Layer Norm 1"""
        l_norm_wgt=self.reference_model.transformer.h[layer_idx].ln_1.weight
        l_norm_bias=self.reference_model.transformer.h[layer_idx].ln_1.bias
        
        """Layer Norm 2"""
        l_norm2_wgt=self.reference_model.transformer.h[layer_idx].ln_2.weight
        l_norm2_bias=self.reference_model.transformer.h[layer_idx].ln_2.bias


        with torch.no_grad():
            
            """Attention"""
            
            if self.single_block is not None:
                self.single_block.attention.qkv_proj.weight.copy_(qkv_proj_wgt.T) #keep the same identity
                self.single_block.attention.qkv_proj.bias.copy_(qkv_proj_bias) 
                
                self.single_block.attention.final_projection.weight.copy_(qkv_final_proj_wgt.T)
                self.single_block.attention.final_projection.bias.copy_(qkv_final_proj_bias)
        
            else:
                self.custom_model.h[layer_idx].attention.qkv_proj.weight.copy_(qkv_proj_wgt.T) #keep the same identity
                self.custom_model.h[layer_idx].attention.qkv_proj.bias.copy_(qkv_proj_bias) 
                
                self.custom_model.h[layer_idx].attention.final_projection.weight.copy_(qkv_final_proj_wgt.T)
                self.custom_model.h[layer_idx].attention.final_projection.bias.copy_(qkv_final_proj_bias)

            """FeedForward"""
            if self.single_block is not None:
                
                self.single_block.mlp.up_proj.weight.copy_(up_proj_wgt.T)
                self.single_block.mlp.up_proj.bias.copy_(up_proj_bias)

                self.single_block.mlp.down_proj.weight.copy_(down_proj_wgt.T)
                self.single_block.mlp.down_proj.bias.copy_(down_proj_bias)

            else:

                self.custom_model.h[layer_idx].mlp.up_proj.weight.copy_(up_proj_wgt.T)
                self.custom_model.h[layer_idx].mlp.up_proj.bias.copy_(up_proj_bias)

                self.custom_model.h[layer_idx].mlp.down_proj.weight.copy_(down_proj_wgt.T)
                self.custom_model.h[layer_idx].mlp.down_proj.bias.copy_(down_proj_bias)

            """Decoder block"""

            if self.single_block is not None:
                self.single_block.layer_norm_1.weight.copy_(l_norm_wgt)
                self.single_block.layer_norm_1.bias.copy_(l_norm_bias)

                self.single_block.layer_norm_2.weight.copy_(l_norm2_wgt)
                self.single_block.layer_norm_2.bias.copy_(l_norm2_bias)
            else:
                self.custom_model.h[layer_idx].layer_norm_1.weight.copy_(l_norm_wgt)
                self.custom_model.h[layer_idx].layer_norm_1.bias.copy_(l_norm_bias)

                self.custom_model.h[layer_idx].layer_norm_2.weight.copy_(l_norm2_wgt)
                self.custom_model.h[layer_idx].layer_norm_2.bias.copy_(l_norm2_bias)
                
    def cp_model_level_params(self):
        with torch.no_grad():
            self.custom_model.wte.weight.copy_(self.reference_model.transformer.wte.weight)
            self.custom_model.wpe.weight.copy_(self.reference_model.transformer.wpe.weight)
            self.custom_model.final_ln.weight.copy_(self.reference_model.transformer.ln_f.weight)
            self.custom_model.final_ln.bias.copy_(self.reference_model.transformer.ln_f.bias)

    def transfer_all(self):
        for layer in range(self.custom_model.config.num_layers):
            self.cp_block_level_params(layer_idx=layer)
        self.cp_model_level_params()
                
    
        