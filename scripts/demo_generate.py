import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.tiny_transformer.block import TinyModel
from src.tiny_transformer.config import GPT2CustomConfig
from src.tiny_transformer.generation import greedy_generate
from src.tiny_transformer.transfer_model_param import GPT2WeightLoader


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_NAME = "openai-community/gpt2"
PROMPT = "My favourite Italian food is"
MAX_NEW_TOKENS = 50


def main():

    # -------------------------
    # Tokenizer
    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)


    # -------------------------
    # Hugging Face reference
    # -------------------------

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(DEVICE)

    reference_model.eval()


    # -------------------------
    # Custom GPT-2
    # -------------------------

    config = GPT2CustomConfig()

    model = TinyModel(
        config
    ).to(DEVICE)

    loader = GPT2WeightLoader(
        reference_model=reference_model,
        custom_model=model,
        single_block=None,
    )

    loader.transfer_all()

    model.eval()


    # -------------------------
    # Generation
    # -------------------------

    generated_ids = greedy_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
    )


    # -------------------------
    # Decode tokens → text
    # -------------------------

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print("\nPrompt:")
    print(PROMPT)

    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main()