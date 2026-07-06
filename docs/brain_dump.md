Ok so to solve the null value issue, I did this:

Q_K[:,i,:,:]=torch.tril(Q_K[:,i,:,:], diagonal=0)
mask=(Q_K[:,i,:,:] == 0)
Q_K[:,i,:,:]=Q_K[:,i,:,:].masked_fill_(mask, float("-inf"))
Q_K[:,i,:,:]=m(Q_K[:,i,:,:])

My only thought is, what if there's null elements which aren't in the diagonal but actual scores?

Regarding shapes we have the Following tensors : 

QV: [1, 12, 5, 5] | V:[1, 5, 12, 64] 

Let's remove the batch size for simplicity

QV: [12, 5, 5]  | V: [5, 12, 64] 

QV: [12, 5,.]  | V: [., 12, 64]

Output: [5, 12, 64]








