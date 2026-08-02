from src.tiny_transformer import data_loader,embeddings_map,get_model_param
from src.tiny_transformer.get_model_param import gpt2_parameter_load_helper
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from boilerplates.similarity_test import compare_tensor_pair
import math
import warnings

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
print(model)