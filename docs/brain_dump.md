So I wrote the single block test and it passed:

@torch.inference_mode()
def test_decoder_single_block(reference_block,custom_model,reference_input_embeddings):
    reference_model_output=reference_block(reference_input_embeddings)
    custom_model_output=custom_model(reference_input_embeddings)
 
    torch.testing.assert_close(custom_model_output,reference_model_output,rtol=1e-5,atol=1e-5)
 
In the GPT2Model reference class I spotted this:

self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

Si there's a layer id tracking but it seem useless for us, it's like the more we advance in layers the more a scaling factor decrease

if self.scale_attn_by_inverse_layer_idx:
            self.scaling /= float(self.layer_idx + 1)


Briefly investigating it seem to scale the attention weight like here:

attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=self.scaling)

Where alpha weights the q@v matrix multiplication.

Let's ignore this because I think it's for performance matter rather than accuracy.