Here's my thought path, and how I reasoned about the problem, again give me constructive feedback:


So let's get back to this step, the current dimension are the following:

Q : torch.Size([1, 5, 12, 64]) * K : torch.Size([1, 5, 12, 64]) 

We want the following output dimension : [1,12,5,5]

So now I get it, basically we want to flattent the head dimension (64
) because we want to "summarize" those 64 per token dimension into one similarity indicator.

So following your technique, we need to swap sequence length in the following way:

[1,12,5,64] * [1,12,64,5]

I'm doing that:
 Q=torch.movedim(Q,(1,2),(2,1))
    K=torch.movedim(K,(1,2,3),(3,1,2))
    Q_K=Q@K

And indeed I have the correct shape output:

Now for the V we have the following initial shapes:
QK:[1, 12, 5, 5] | V:[1, 5, 12, 64]

Output => [1, 5, 12, 64]

[1, 12, 5, 5]  * [1, 12, 5, 64] => [1, 12, 5, 64]




