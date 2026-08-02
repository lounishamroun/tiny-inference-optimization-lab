from .conftest import reference_model
import pytest




def source_qkv_proj(reference_model):
    return reference_model[0].c_attn

