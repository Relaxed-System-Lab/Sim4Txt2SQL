"""
imagine a conversation:
user: 2 tokens
bot: 3 tokens
user: 4 tokens
bot: 5 tokens

total_prompt_tokens = 2 + (2+3+4) = 11
cachable_prompt_tokens = 2+3 = 5
cachable_ratio = 5/11

[2,3,4,5,6,7]
total_prompt_tokens = 2 + (2+3+4) + (2+3+4+5+6) = 31
cachable_tokens = (2+3) + (2+3+4+5) = 19
"""


def calculate_tokens(conversation_tokens):
    """
    Calculate the number of generated tokens and prefill tokens.

    Args:
    conversation_tokens (list): A list of lists, where each sublist represents the number of tokens in a turn of the conversation.

    Returns:
    dict: A dictionary with keys 'generated_tokens' and 'prefill_tokens'.
    """
    prefill_tokens = 0
    cachable_tokens = 0
    for i, turn in enumerate(conversation_tokens):
        if i % 2 != 0:
            # bot's turn
            prefill_tokens += sum(conversation_tokens[:i])
            # cacheable tokens
            if i > 1:
                cachable_tokens += sum(conversation_tokens[: i - 1])
    return prefill_tokens, cachable_tokens


convs = [2, 3, 4, 5, 6, 7]
print(calculate_tokens(convs))  # (11, 5)
