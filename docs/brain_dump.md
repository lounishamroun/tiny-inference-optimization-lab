Ok so I spoted that here (c_attn): Conv1D(nf=2304, nx=768), we have 2304 = 3*768.

So I guess it's an inverted linear projection, like the equivalent of doing nn.Linear(768,2304) and I guess later reshape to fit my architecture.

So here they seem to be an initial linear projection before qkv projection :

Name: transformer.h.0.ln_1.weight of shape :  torch.Size([768])
Name: transformer.h.0.ln_1.bias of shape :  torch.Size([768])
Name: transformer.h.0.attn.c_attn.weight of shape :  torch.Size([768, 2304])
Name: transformer.h.0.attn.c_attn.bias of shape :  torch.Size([2304])


So checking the shape of gpt 2 weights its correct:

QKV wgth: torch.Size([768, 2304])
QKV bias: torch.Size([2304])

However we have this error : RuntimeError: The size of tensor a (768) must match the size of tensor b (2304) at non-singleton dimension 1

So here we can see that the random weights are stored in inverted, hence lets transpose our gpt 2 weights : Before copy keeping identity: torch.Size([2304, 768])

Now I should find a way to correctly replace my initial architecturee:

 """ Q,K,V projections """
        Q=self.Qw(embeddings)
        K=self.Kw(embeddings)
        V=self.Vw(embeddings)

Since we don't have anymore seperated matrices

Ok so we have output shape : 

qkv=self.qkv_proj(embeddings) => torch.Size([1, 7, 2304]), I think we should reshape in => torch.Size([1, 7, , 3 , 768])

So it becomes like that : 

qkv=self.qkv_proj(embeddings)
qkv=torch.reshape(qkv, (1,7,3,768))

By doing that we retreive our original shape for all of the 3 qkv matrices:

""" Q,K,V projections """
self.Qw,self.Kw,self.Vw=qkv[:,:,0,:],qkv[:,:,1,:],qkv[:,:,2,:]