def gpt2_parameter_load_helper(model):


    new_gelu=model.get_submodule("transformer.h.0.mlp.act")
    
    """Up Linear projection"""
    up_proj_wgt=model.get_parameter("transformer.h.0.mlp.c_fc.weight") 
    up_proj_bias=model.get_parameter("transformer.h.0.mlp.c_fc.bias")     
    
    """Down Linear projection"""
    down_proj_wgt=model.get_parameter("transformer.h.0.mlp.c_proj.weight") 
    down_proj_bias=model.get_parameter("transformer.h.0.mlp.c_proj.bias")
    
    """QKV projection"""
    qkv_proj_wgt=model.get_parameter("transformer.h.0.attn.c_attn.weight")
    qkv_proj_bias=model.get_parameter("transformer.h.0.attn.c_attn.bias")
    
    """QKV Final Projection"""
    qkv_final_proj_wgt=model.get_parameter("transformer.h.0.attn.c_proj.weight")
    qkv_final_proj_bias=model.get_parameter("transformer.h.0.attn.c_proj.bias") 
    
    """Layer Norm 1"""
    l_norm_wgt=model.get_parameter("transformer.h.0.ln_1.weight")
    l_norm_bias=model.get_parameter("transformer.h.0.ln_1.bias")
    
    """Layer Norm 2"""
    l_norm2_wgt=model.get_parameter("transformer.h.0.ln_2.weight")
    l_norm2_bias=model.get_parameter("transformer.h.0.ln_2.bias")
    
    """Final Layer Norm"""
    ln_f_wgt=model.get_parameter("transformer.ln_f.weight")
    ln_f_bias=model.get_parameter("transformer.ln_f.bias") 
     
    return [up_proj_wgt,up_proj_bias,down_proj_wgt,down_proj_bias,qkv_proj_wgt,qkv_proj_bias,l_norm_wgt,l_norm_bias,l_norm2_wgt,l_norm2_bias,qkv_final_proj_wgt,qkv_final_proj_bias,new_gelu,ln_f_wgt,ln_f_bias]