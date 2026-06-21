import torch

def compare_tensor_pair(t1,t2,allclose=True):
    assert isinstance(t1,torch.Tensor), f"Type mismatch {type(t1)}" 
    assert isinstance(t2,torch.Tensor), f"Type mismatch {type(t2)}"
    assert t1.device == t2.device, f"Different device between both tensors {t1.device} VS {t2.device}"
    assert t1.shape == t2.shape, f"Different shape between both tensors {t1.shape} VS {t2.shape}"
    if allclose == True:
        assert torch.allclose(t1,t2), f"Important value discrepancy between both tensors | Max discrepancy: {abs(torch.max(t1)-torch.max(t2))} | Min discrepancy:{abs(torch.min(t1)-torch.min(t2))} | Avg discrepancy: {abs(torch.mean(t1)-torch.mean(t2))}" 


t1=torch.randn(size=[1,8,9])
t2=torch.randn(size=[1,8,9])

compare_tensor_pair(t1,t2)