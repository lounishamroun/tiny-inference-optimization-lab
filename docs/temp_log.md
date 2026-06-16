Ok so here's my reasoning I need a ruthless feedback:


So I created a tokenizer which tokenises a 5 character sentence:


def tokenize_text(INPUT_TEXT):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenized_text=tokenizer(INPUT_TEXT, return_tensors="pt") 
    return tokenized_text

Return => #tensor([[ 464, 3616,  286, 1204,  318]]) => Each word has been tokenized

So I guess the next step is to create K,Q,V matrices, I will test it on a seperate file dedicted to toy tests.

So aside I read the doc about usage of "detach()" this could be useful next.

Then I started plating with pytorch to find a way to create these vector projections

lin_layer=torch.nn.Linear(in_features=1, out_features=1)

K=torch.randn(
    (embedding_vector.shape[0],
    embedding_vector.shape[0])
)

for elem in K:
    for e in elem:
        print(lin_layer.forward(e))

So here I'm realizing I'm wrong, we should have a multi dimensional hidden layer dimension.

So let's create an nn.Module class

The issue is that the embedding has "[ 464, 3616,  286, 1204,  318]" various dimension per word so Idk if I can even construct the K,Q,V matrices. 

So by inspecting the doc, I found an interesting parameter used by the tokenizer:

config (PreTrainedConfig, optional) — The configuration object used to determine the tokenizer class to instantiate.

def tokenize_text(INPUT_TEXT,tokenizer_config=None):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    tokenized_text=tokenizer(INPUT_TEXT, return_tensors="pt") #=> Tokenizes each word
    return tokenized_text

This returns the attention and hidden state.

Note that Im still confused since the hidden states doesnt have the same size hence idk how to perform the linear transformation.


