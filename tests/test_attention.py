import pytest
import torch



class TestAttention:

    def test_qkv_proj_params(self, custom_block, reference_model):
        src_init_proj_weights = (
            reference_model.transformer.h[0].attn.c_attn.weight
        )
        src_init_proj_bias = (
            reference_model.transformer.h[0].attn.c_attn.bias
        )

        cus_init_proj_weights = custom_block.attention.qkv_proj.weight
        cus_init_proj_bias = custom_block.attention.qkv_proj.bias

        torch.testing.assert_close(
            cus_init_proj_weights,
            src_init_proj_weights.T,
            rtol=0,
            atol=0,
        )

        torch.testing.assert_close(
            cus_init_proj_bias,
            src_init_proj_bias,
            rtol=0,
            atol=0,
        )

    @torch.inference_mode()
    def test_fused_qkv_output(
        self,
        custom_block,
        reference_input_embeddings,
        reference_block,
    ):
        # Give both implementations exactly the same input.
        normalized_input = reference_block.ln_1(
            reference_input_embeddings
        )

        expected_qkv = reference_block.attn.c_attn(
            normalized_input
        )

        actual_qkv = custom_block.attention.qkv_proj(
            normalized_input
        )

        assert expected_qkv.shape == actual_qkv.shape

        torch.testing.assert_close(
            actual_qkv,
            expected_qkv,
            rtol=1e-5,
            atol=1e-6,
        )

    @torch.inference_mode()
    def test_split_head_parity(
        self,
        custom_block,
        reference_input_embeddings,
        reference_block,
    ):
        batch_size, seq_length, _ = (
            reference_input_embeddings.shape
        )

        normalized_input = reference_block.ln_1(
            reference_input_embeddings
        )

        mQ, mK, mV = (
            custom_block.attention._qkv_projection_helper(
                embeddings=normalized_input,
                batch_size=batch_size,
                seq_length=seq_length,
            )
        )

        mQ = mQ.movedim(1, 2)
        mK = mK.movedim(1, 2)
        mV = mV.movedim(1, 2)

        query_states, key_states, value_states = (
            reference_block.attn.c_attn(
                normalized_input
            ).split(
                reference_block.attn.split_size,
                dim=2,
            )
        )

        shape_kv = (
            *key_states.shape[:-1],
            -1,
            reference_block.attn.head_dim,
        )

        key_states = (
            key_states
            .view(shape_kv)
            .transpose(1, 2)
        )

        value_states = (
            value_states
            .view(shape_kv)
            .transpose(1, 2)
        )

        shape_q = (
            *query_states.shape[:-1],
            -1,
            reference_block.attn.head_dim,
        )

        query_states = (
            query_states
            .view(shape_q)
            .transpose(1, 2)
        )

        assert mQ.shape == query_states.shape
        assert mK.shape == key_states.shape
        assert mV.shape == value_states.shape

        torch.testing.assert_close(
            mQ,
            query_states,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            mK,
            key_states,
            rtol=1e-5,
            atol=1e-6,
        )

        torch.testing.assert_close(
            mV,
            value_states,
            rtol=1e-5,
            atol=1e-6,
        )

    @torch.inference_mode()
    def test_attention(
    self,
    reference_block,
    reference_input_embeddings,
    custom_attention_output,
    ):
        normalized_input = reference_block.ln_1(
            reference_input_embeddings
        )

        expected_attention = reference_block.attn(
            normalized_input
        )[0]

        actual_attention = custom_attention_output

        torch.testing.assert_close(
            actual_attention,
            expected_attention,
            rtol=1e-5,
            atol=1e-6,
        )

    @torch.inference_mode()
    def test_attention_probabilities_sum_to_one(
        self,
        custom_block,
        reference_input_embeddings,
    ):
        """
        Property:
        after causal masking + softmax, every attention
        distribution over key positions must sum to 1.
        """

        batch_size, seq_length, _ = reference_input_embeddings.shape

        Q, K, _ = custom_block.attention._qkv_projection_helper(
            embeddings=reference_input_embeddings,
            batch_size=batch_size,
            seq_length=seq_length,
        )

        Q = Q.movedim(1, 2)
        K = K.movedim(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (custom_block.attention.head_dim ** 0.5)

        mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )

        masked_scores = scores.masked_fill(mask, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)

        row_sums = attention_weights.sum(dim=-1)

        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            rtol=0,
            atol=1e-6,
        )

    @torch.inference_mode()
    def test_decoder_block_does_not_see_future_tokens(
        self,
        custom_block,
        reference_input_embeddings,
    ):
        """
        Strong causal property test:

        Changing future tokens must not change the
        output corresponding to earlier positions.
        """

        original = reference_input_embeddings.clone()
        modified = reference_input_embeddings.clone()

        seq_length = original.shape[1]

        assert seq_length >= 2

        # Choose a token somewhere before the end.
        cutoff = seq_length // 2

        # Completely alter every token AFTER the cutoff.
        modified[:, cutoff + 1:, :] = torch.randn_like(
            modified[:, cutoff + 1:, :]
        )

        original_output = custom_block(original)
        modified_output = custom_block(modified)

        # Positions <= cutoff must be identical because
        # they are forbidden from attending to later tokens.
        torch.testing.assert_close(
            original_output[:, :cutoff + 1, :],
            modified_output[:, :cutoff + 1, :],
            rtol=1e-5,
            atol=1e-6,
        )