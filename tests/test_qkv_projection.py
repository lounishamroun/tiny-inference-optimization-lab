import pytest
import torch
from src.tiny_transformer import data_loader,embeddings_map
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.tiny_transformer.block import gpt2_parameter_load_helper

if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
    
INPUT_TEXT = data_loader.return_text("data/text.txt")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
embedding_lookup_table=model.get_input_embeddings().weight.detach().clone().T.to(DEVICE)
tokenizer=embeddings_map.TokenToEmbedding(INPUT_TEXT,device=DEVICE)
embeddings=tokenizer.map_embeddings().detach()
gpt_2_params=gpt2_parameter_load_helper(model)


tensor_size=[
    (5,23),
    (3,45)
]

@pytest.mark.parametrize("size_a,size_b",tensor_size)
def create_2_tensors(size_a,size_b):
    a=torch.randn((size_a,size_b))
    print(a.shape)

