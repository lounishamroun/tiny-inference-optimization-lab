import torch


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Greedy autoregressive generation.

    Args:
        model:
            GPT-style model returning logits shaped [B, T, vocab_size].

        input_ids:
            Token IDs shaped [B, T].

        max_new_tokens:
            Number of tokens to generate.

    Returns:
        Token IDs shaped [B, T + max_new_tokens].
    """

    generated_ids = input_ids

    for _ in range(max_new_tokens):

        # [B, T] -> [B, T, vocab_size]
        logits = model(generated_ids)

        # We only need the prediction after the last token.
        # [B, T, vocab_size] -> [B, vocab_size]
        next_token_logits = logits[:, -1, :]

        # Greedy decoding:
        # select the highest-logit token.
        # [B, vocab_size] -> [B, 1]
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # Append the new token to the sequence.
        # [B, T] + [B, 1] -> [B, T + 1]
        generated_ids = torch.cat(
            [generated_ids, next_token_id],
            dim=1,
        )

    return generated_ids