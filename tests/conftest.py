import pytest

from src.tiny_transformer import block
from src.tiny_transformer.config  import GPT2CustomConfig 
import src
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from boilerplates.similarity_test import compare_tensor_pair
from src.tiny_transformer.transfer_model_param import GPT2WeightLoader

    

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")    

@pytest.fixture(scope="session")
def reference_model(device):
        
    """ 
    Retreiving embeddings for an input text sequence
    Output shape => [B,T,d_model] 
    """

    reference_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    reference_model = reference_model.to(device)
    reference_model.eval() #to ensure deterministic results
    
    return reference_model


@pytest.fixture(scope="session")
def input_text():
    return "My favourite Italian food is"


@pytest.fixture(scope="session")
def input_ids(reference_model,input_text):
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2"
    )

    ids = tokenizer(
        input_text,
        return_tensors="pt",
    )["input_ids"]

    return ids.to(next(reference_model.parameters()).device)

@pytest.fixture(scope="session")
def reference_input_embeddings(input_ids, reference_model):
    with torch.no_grad():
        seq_length = input_ids.shape[1]

        position_ids = torch.arange(
            seq_length,
            device=input_ids.device,
        )

        token_embeddings = reference_model.transformer.wte(input_ids)
        position_embeddings = reference_model.transformer.wpe(position_ids)

        embeddings = token_embeddings + position_embeddings

    return embeddings
        

@pytest.fixture(scope="session")
def reference_block(reference_model):
    reference_block = reference_model.transformer.h[0]
    reference_block.eval()
    return reference_block


@pytest.fixture(scope="session")
def get_batch_seq_dim(reference_input_embeddings):
    batch_size,seq_length,_=reference_input_embeddings.shape
    return [batch_size,seq_length]


@pytest.fixture(scope="session")
def conf():
    conf=GPT2CustomConfig()
    return conf


@pytest.fixture(scope="session")
def custom_block(device,conf,reference_model):
    custom_block=block.TinyDecoderBlock(conf,layer_idx=0).to(device)
    GPT2WeightLoader(reference_model=reference_model,custom_model=None,single_block=custom_block).cp_block_level_params(layer_idx=0)
    custom_block.eval()
    return custom_block


@pytest.fixture(scope="session")
def custom_model(conf, reference_model,device):
    model = src.tiny_transformer.block.TinyModel(conf).to(device)

    loader = GPT2WeightLoader(
        reference_model=reference_model,
        custom_model=model,
        single_block=None,
    )

    loader.transfer_all()

    model.eval()

    return model

    

@pytest.fixture(scope="session")
def custom_attention_output(reference_input_embeddings,custom_block,reference_block):
    with torch.inference_mode():
        
        normalized_input=reference_block.ln_1(
                reference_input_embeddings
            )

        custom_attention=custom_block.attention(normalized_input)
        
    return custom_attention
