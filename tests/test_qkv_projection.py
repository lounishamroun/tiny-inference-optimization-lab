import pytest
import torch
from src.tiny_transformer import data_loader,embeddings_map
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import src.tiny_transformer.block


if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"


class TestAttention():
    
    def test_qkv_proj_params(self,custom_block,reference_model):
        src_init_proj_weights=reference_model.transformer.h[0].attn.c_attn.weight
        src_init_proj_bias=reference_model.transformer.h[0].attn.c_attn.bias

        cus_init_proj_weights=custom_block.attention.qkv_proj.weight
        cus_init_proj_bias=custom_block.attention.qkv_proj.bias
    
        assert torch.allclose(src_init_proj_weights.T,cus_init_proj_weights)
        assert torch.allclose(src_init_proj_bias,cus_init_proj_bias)
        
    def test_fused_qkv_output(self,
    custom_block,
    reference_input_embeddings,
    reference_block
    ):

        with torch.inference_mode():
            # Give both projections the exact same input.
            normalized_input = reference_block.ln_1(
                reference_input_embeddings
            )

            expected_qkv = reference_block.attn.c_attn(
                normalized_input
            )

            actual_qkv = custom_block.attention.qkv_proj(
                normalized_input
            )
        assert expected_qkv.shape == actual_qkv.shape #torch.Size([batch_size, seq_length, d_model*3])

        torch.testing.assert_close(
            actual_qkv,
            expected_qkv,
            rtol=1e-5,
            atol=1e-6,
        )
            
    def test_split_head_parity(self,custom_block,
        reference_input_embeddings,
        reference_block):
        batch_size,seq_length,_=reference_input_embeddings.shape
        
        
        with torch.inference_mode():
            normalized_input = reference_block.ln_1(reference_input_embeddings)
            mQ,mK,mV=custom_block.attention._qkv_projection_helper(embeddings=normalized_input,batch_size=batch_size,seq_length=seq_length)
            mQ=mQ.movedim(1,2)
            mK=mK.movedim(1,2)
            mV=mV.movedim(1,2)
            
            query_states, key_states, value_states = reference_block.attn.c_attn(normalized_input).split(reference_block.attn.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, reference_block.attn.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)
            shape_q = (*query_states.shape[:-1], -1, reference_block.attn.head_dim)
            query_states = query_states.view(shape_q).transpose(1, 2)
        
            assert mQ.shape==query_states.shape
            assert mK.shape==key_states.shape
            assert mV.shape==value_states.shape
          

            torch.testing.assert_close(mQ, query_states, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(mK, key_states, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(mV, value_states, rtol=1e-5, atol=1e-6)

        
    def test_attention(
        self,
        reference_block,
        reference_input_embeddings,
        custom_block,
    ):
        with torch.inference_mode():
            normalized_input = reference_block.ln_1(
                reference_input_embeddings
            )

            expected_attention = reference_block.attn(
                normalized_input
            )[0]

            actual_attention = custom_block.attention(
                normalized_input
            )

        assert actual_attention.shape == expected_attention.shape

        torch.testing.assert_close(
            actual_attention,
            expected_attention,
            rtol=1e-5,
            atol=1e-6,
        )
        

