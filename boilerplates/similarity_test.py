import torch

def compare_tensor_pair(t1,t2,allclose=True):
    assert type(t1) == torch.Tensor, "Object not a tensor" 
    assert type(t2) == torch.Tensor, "Object not a tensor"
    assert t1.device == t2.device, "Different device between both tensors"
    assert t1.shape == t2.shape, "Different shape between both tensors"
    if allclose == True:
        assert torch.allclose(t1,t2), "Important value discrepancy between both tensors"
    