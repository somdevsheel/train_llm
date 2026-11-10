"""Chat template shared by SFT data prep and generation.

    <|system|>\n...\n<|user|>\n...\n<|assistant|>\n...<|endoftext|>\n<|user|>\n...

Role tags are plain text (GPT-2 BPE has no special tokens for them); each assistant turn
ends with <|endoftext|> so generation stops there. Only assistant content + EOS is trained on.
"""
EOS = 50256
ROLE_TAG = {"system": "<|system|>\n", "user": "<|user|>\n", "assistant": "<|assistant|>\n"}


def encode_turns(enc, messages):
    """-> list of (ids, target_mask) per message; mask=1 marks tokens the model learns to produce."""
    out = []
    for m in messages:
        role, text = m["role"], m["content"].strip()
        if role == "assistant":
            head = enc.encode_ordinary(ROLE_TAG["assistant"])
            body = enc.encode_ordinary(text) + [EOS]
            tail = enc.encode_ordinary("\n")
            out.append((head + body + tail, [0] * len(head) + [1] * len(body) + [0] * len(tail)))
        else:
            ids = enc.encode_ordinary(ROLE_TAG.get(role, ROLE_TAG["user"]) + text + "\n")
            out.append((ids, [0] * len(ids)))
    return out


def prompt_ids(enc, messages):
    """Encode a conversation so far, ending with the assistant tag, ready for generation."""
    ids = []
    for t, _ in encode_turns(enc, messages):
        ids += t
    return ids + enc.encode_ordinary(ROLE_TAG["assistant"])
