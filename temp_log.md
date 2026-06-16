Be ruthless about my reasoning and coach me/mentor me to become an elite builder, below is my detailed reasoning:

Ok so first I'm experimenting with the function I found linked to the tokenizer object:

tokenizer,hidden_state_token_id=tokenize_text(INPUT_TEXT,True)
word_ids=hidden_state_token_id['input_ids']

for index,word in enumerate(word_ids):
    print(f' w {word[index]}')
    print(tokenizer._convert_id_to_token(word[index]))

print(f"tokenizer: {type(tokenizer)} | tokenizer: {type(hidden_state_token_id)}")

It works for a specific word id

Ok so I know that private function are not meant to be used outside of the class:

def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

So I will use self.decoder.get instead

tokenizer,hidden_state_token_id=tokenize_text(INPUT_TEXT,True)
word_ids=hidden_state_token_id['input_ids']

def decode_ids(tokenizer,word_ids):
    word_ids=word_ids.squeeze(0)
    for id_ in word_ids: 
        embedding=tokenizer.decode(id_.item())
        print(embedding)
        
Let's try to create a toy exemple : 

vocab={'cat':4855,'line':4545}

for k,v in vocab.items():
    print(v)


So I tried with a known exemple and it worked:

def dict_map(dictionary="https://huggingface.co/gpt2/resolve/main/vocab.json"):
    body=r.get(dictionary)
    json_dictionary=body.json() #type dict
    for k,v in json_dictionary.items():
        if k=="ĠJav":
            print(v) => 49247 correct embedding
dict_map()

So I have the correct first attempt : 

def dict_map(key,dictionary="https://huggingface.co/gpt2/resolve/main/vocab.json"):
    body=r.get(dictionary)
    json_dictionary=body.json() #type dict
    for k,v in json_dictionary.items():
        if k==str(key):
            print(v)
            
dict_map("cat")

Thoughts: I think it would have been better to look for an existing function on hugging face doc, and also
I think it's not the most efficient way in terms of performance but I know its not the best.

""" Glossary 
Decoder (GPT STYLE)
    Shapes:
        s=text sequence length
        n_layers = 4 : How much neurons our model has.
        d_model = 256 : Correponds to the length of our embeddings.
        n_heads = 4
        head_dim = 4
        B = 1 : Batch sizee
        T = 5 : Sequence length (in this case 5 words)
        context_length = 512
vocab_size =  8k–32k
"""

from . import data_loader  
from tokenizers import Tokenizer
from transformers import AutoTokenizer,PreTrainedConfig,PythonBackend
import torch 
from torch import nn
import json
import requests as r

INPUT_TEXT = data_loader.return_text("data/text.txt")

tokenizer_config=PreTrainedConfig(output_hidden_states=True,output_attentions=True)

def tokenize_text(INPUT_TEXT,tokenizer_config=None):
# We'll use a pre-trained tokenizer since we'll use quite generic data
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="openai-community/gpt2",
        config_model_type=tokenizer_config
        )
    token_ids=tokenizer(INPUT_TEXT, return_tensors="pt") #=> Tokenizes each word
    return tokenizer,token_ids #=> return token's hidden state's IDs.

def dict_map(key,dictionary="https://huggingface.co/gpt2/resolve/main/vocab.json"):
    body=r.get(dictionary)
    json_dictionary=body.json() #type dict
    for k,v in json_dictionary.items():
        if k==str(key):
            print(v)

def ids_to_embeddings(tokenizer,word_ids):
    """
        Used to output a corresponding embedding given a token ID.

        Args:
            word_ids (`tensor[int]`):
                Token IDs corresponding to each tokens of our text sequence.
            
        Returns => shape:[s,d_model] :
            `?`: Returns the list of embeddings corresponding to each of our sequence of tokens.
    """
    embedding_list=[]
    word_ids=word_ids['input_ids'].squeeze(0)
    for id_ in word_ids: 
        word=tokenizer.decode(id_.item()).strip()
        embedding=dict_map(word)
        embedding_list.append(embedding)
    return embedding


tokenizer,hidden_state_token_id=tokenize_text(INPUT_TEXT=INPUT_TEXT,tokenizer_config=tokenizer_config)
ids_to_embeddings(tokenizer,hidden_state_token_id)


Ok so I just realized the dictionnary does not show embedding but the id so I worked in the void.

Ok so I searched for "embedding matrix huggingface" on google: https://huggingface.co/blog/getting-started-with-embeddings.


So looking at the hugginface tokenizer documentation I'm realizing that the embedding might be accessible at the model instance level, see : 

# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("google-bert/bert-base-uncased")

num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
print("We have added", num_added_toks, "tokens")
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))


So in the forward method of the gpt2 model documentation there's an interesting returned parameter:

=> hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).


Since we enabled the "return hidden state" boolean:tokenizer_config=PreTrainedConfig(output_hidden_states=True,output_attentions=True)

It might be our solution

So looking at the output of the model's layers : GPT2Model((wte): Embedding(50257, 768) (wpe): Embedding(1024, 768))

I assume this is the embedding matrix, now let's see how to lookup the correponding tokens using token ids

print(next(model.named_parameters("(wte)"))[1].shape)

torch.Size([50257, 768])

next(model.named_parameters("(wte)"))[1][0])

We can for exemple check the embedding of the first token inside the vocabulary


By using "dir" to print the function linked to the model I found a particularly interesting function :

get_input_embeddings

print(type(model.get_input_embeddings()))

So printing the type we get the following : <class 'torch.nn.modules.sparse.Embedding'>

print(dir(type(model.get_input_embeddings())))
