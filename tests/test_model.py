import pytest 
import torch


@torch.inference_mode()
def test_all_parameters_match(custom_model, reference_model, conf):
    for layer_idx in range(conf.num_layers):

        """Attention - QKV projection"""
        assert torch.allclose(
            custom_model.h[layer_idx].attention.qkv_proj.weight,
            reference_model.transformer.h[layer_idx].attn.c_attn.weight.T
        ), f"Layer {layer_idx}: QKV projection weight mismatch"

        assert torch.allclose(
            custom_model.h[layer_idx].attention.qkv_proj.bias,
            reference_model.transformer.h[layer_idx].attn.c_attn.bias
        ), f"Layer {layer_idx}: QKV projection bias mismatch"

        """Attention - Final projection"""
        assert torch.allclose(
            custom_model.h[layer_idx].attention.final_projection.weight,
            reference_model.transformer.h[layer_idx].attn.c_proj.weight.T
        ), f"Layer {layer_idx}: attention final projection weight mismatch"

        assert torch.allclose(
            custom_model.h[layer_idx].attention.final_projection.bias,
            reference_model.transformer.h[layer_idx].attn.c_proj.bias
        ), f"Layer {layer_idx}: attention final projection bias mismatch"

        """Up Linear projection"""
        assert torch.allclose(
            custom_model.h[layer_idx].mlp.up_proj.weight,
            reference_model.transformer.h[layer_idx].mlp.c_fc.weight.T
        ), f"Layer {layer_idx}: MLP up projection weight mismatch"

        assert torch.allclose(
            custom_model.h[layer_idx].mlp.up_proj.bias,
            reference_model.transformer.h[layer_idx].mlp.c_fc.bias
        ), f"Layer {layer_idx}: MLP up projection bias mismatch"

        """Down Linear projection"""
        assert torch.allclose(
            custom_model.h[layer_idx].mlp.down_proj.weight,
            reference_model.transformer.h[layer_idx].mlp.c_proj.weight.T
        ), f"Layer {layer_idx}: MLP down projection weight mismatch"

        assert torch.allclose(
            custom_model.h[layer_idx].mlp.down_proj.bias,
            reference_model.transformer.h[layer_idx].mlp.c_proj.bias
        ), f"Layer {layer_idx}: MLP down projection bias mismatch"

        """Layer Norm 1"""
        assert torch.allclose(
            custom_model.h[layer_idx].layer_norm_1.weight,
            reference_model.transformer.h[layer_idx].ln_1.weight
        ), f"Layer {layer_idx}: LayerNorm 1 weight mismatch"

        assert torch.allclose(
            custom_model.h[layer_idx].layer_norm_1.bias,
            reference_model.transformer.h[layer_idx].ln_1.bias
        ), f"Layer {layer_idx}: LayerNorm 1 bias mismatch"

        """Layer Norm 2"""
        assert torch.allclose(
            custom_model.h[layer_idx].layer_norm_2.weight,
            reference_model.transformer.h[layer_idx].ln_2.weight
        ), f"Layer {layer_idx}: LayerNorm 2 weight mismatch"

        assert torch.allclose(
            custom_model.h[layer_idx].layer_norm_2.bias,
            reference_model.transformer.h[layer_idx].ln_2.bias
        ), f"Layer {layer_idx}: LayerNorm 2 bias mismatch"
        
        assert torch.allclose(
                    custom_model.final_ln.weight,
                    reference_model.transformer.ln_f.weight
                ), f"Layer {layer_idx}: Final LayerNorm weight mismatch"
        
        assert torch.allclose(
                    custom_model.final_ln.bias,
                    reference_model.transformer.ln_f.bias
                ), f"Layer {layer_idx}: Final LayerNorm weight mismatch"
        
        assert torch.allclose(
                    custom_model.wte.weight,
                    reference_model.transformer.wte.weight
                ), f"Layer {layer_idx}: Wte weight mismatch"

        assert torch.allclose(
                    custom_model.wpe.weight,
                    reference_model.transformer.wpe.weight
                ), f"Layer {layer_idx}: Wpe weight mismatch"
    
@torch.inference_mode()
def test_layer_outputs_match(
    custom_model,
    reference_model,
    reference_input_embeddings,
    conf,
):
    reference_hidden = reference_input_embeddings
    custom_hidden = reference_input_embeddings.clone()

    for i in range(conf.num_layers):

        reference_hidden = reference_model.transformer.h[i](
            reference_hidden
        )

        custom_hidden = custom_model.h[i](
            custom_hidden
        )

        assert reference_hidden.shape == custom_hidden.shape

        torch.testing.assert_close(
            custom_hidden,
            reference_hidden,
            rtol=1e-4,
            atol=1e-4,
        )
        
        
@pytest.mark.parametrize(
    "batch_size,seq_length",
    [
        (1, 1),
        (1, 8),
        (2, 5),
    ],
)
@torch.inference_mode()
def test_full_model_multiple_shapes(
    custom_model,
    reference_model,
    device,
    batch_size,
    seq_length,
):
    torch.manual_seed(0)

    input_ids = torch.randint(
        0,
        custom_model.config.vocab_size,
        (batch_size, seq_length),
        device=device,
    )

    expected = reference_model(
        input_ids=input_ids,
        use_cache=False,
    ).logits

    actual = custom_model(input_ids)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


@torch.inference_mode()
def test_full_model_parity(
    custom_model,
    reference_model,
    input_ids,
):
    reference_model.eval()
    custom_model.eval()

    expected = reference_model(
        input_ids=input_ids,
        use_cache=False,
    ).logits

    actual = custom_model(input_ids)

    diff = (actual - expected).abs()

    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )