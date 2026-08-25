Without giving me the answer, do we agree that the goal of layer idx is to influence the scaling which influence attention weights, like there's an initial parameter initialisation on the first layer but then the more we advance in layers the more the scaling factor has an effect. However I spotted that in research the scaling thing is optional for optimization so I have to find another way to understand the parameter mechanism. 

If we check the config we can see that the scaling is false by default:

scale_attn_by_inverse_layer_idx: bool = False

If I compare the weights across layers it's true that there's a discrepancy:

for i in range(12):
    p_proj_wgt=model.get_parameter(f'transformer.h.{i}.mlp.c_fc.weight')         print(p_proj_wgt.max()) 
 

tensor(4.5877, device='cuda:0', grad_fn=<MaxBackward1>)
tensor(2.2892, device='cuda:0', grad_fn=<MaxBackward1>)
tensor(10.5558, device='cuda:0', grad_fn=<MaxBackward1>)
tensor(2.2842, device='cuda:0', grad_fn=<MaxBackward1>)

When retrieving the tensors a second time these are the same weights, so the model seem to be loading the same set of weights for each layer.


One of the first clue is in the the GPT2LMHeadModel class:

# Initialize weights and apply final processing
        self.post_init()

And:

_tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

Another is that each class using the post_init() method inherits from "GPT2PreTrainedModel" so the answer must be there.
