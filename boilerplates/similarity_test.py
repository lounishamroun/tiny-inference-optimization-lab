import torch

def compare_tensor_pair(t1,t2,allclose=True):
    assert type(t1) == torch.Tensor, f"Object not a tensor {type(t1)}" 
    assert type(t2) == torch.Tensor, f"Object not a tensor {type(t2)}"
    assert t1.device == t2.device, f"Different device between both tensors {t1.device} VS {t2.device}"
    assert t1.shape == t2.shape, f"Different shape between both tensors {t1.shape} VS {t2.shape}"
    if allclose == True:
        assert torch.allclose(t1,t2), f"Important value discrepancy between both tensors"
    