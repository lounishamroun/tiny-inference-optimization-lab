import block
from transformers import GPT2Config, GPT2Model,initialization,AutoTokenizer
import torch
from torch import nn
import config

if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"

initial_sentence="Dogs are a type of"

def input_ids(input_text):
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2"
    )

    ids = tokenizer(
        input_text,
        return_tensors="pt",
    )["input_ids"]
    
    return ids


def decode_id(input_id):
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2"
    )

    word = tokenizer.convert_ids_to_tokens(
        input_id,
        )
    
    return word

model=block.TinyModel(config.GPT2CustomConfig())
for i in range(10):
    ids=input_ids(initial_sentence)
    print()
    logits=model(input_ids=ids)
    #B,T,vocab_size
    soft=torch.nn.Softmax(dim=-1)
    soft_logits=soft(logits)
    max_likelihood=torch.max(soft_logits,dim=-1)
    next_id=torch.squeeze(max_likelihood.indices)[0].item()
    ids=ids.add(next_id)
    next_word=decode_id(input_id=ids[-1])
    print(f'Predicted next word: {next_word}')



    

    