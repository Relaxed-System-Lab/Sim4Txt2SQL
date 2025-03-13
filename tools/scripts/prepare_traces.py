import os
import json

from datasets import load_dataset
import tiktoken
import pandas as pd
import multiprocessing as mp
import uuid
from typing import List

# utilities
cl100k_base = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
        "<|endofprompt|>": 100266,
    },
)
special_tokens = {"<|endofprompt|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"}


def encode(x):
    return enc.encode(x, allowed_special=special_tokens)


def generate_session_id():
    return str(uuid.uuid4())


def flatten(x: List[List]):
    return [item for sublist in x for item in sublist]


"""
data structure:
jsonl per line: {"partition":"data source" ,"session id": "", "model": "","timestamp":null, "input": [], "output": [], "meta": {}}

"""

### LM-Sys-chat-1m
print("Processing LM-Sys-chat-1m...")

ds = load_dataset("lmsys/lmsys-chat-1m")["train"]


def parse_lmsys_row(row):
    result = []
    session_id = f"lmsys-{generate_session_id()}"
    conversations = row["conversation"]
    for i in range(row["turn"]):
        output = conversations[2 * i + 1]["content"]
        context = conversations[: 2 * i + 1]
        context = [x["content"] for x in context]
        context = "\n\n".join(context)
        data = {
            "partition": "lmsys-chat-1m",
            "session_id": session_id,
            "timestamp": None,
            "model": row["model"],
            "input": encode(context),
            "output": encode(output),
        }
        result.append(data)

    return result


with mp.Pool(32) as pool:
    results = pool.map(parse_lmsys_row, ds)

results = flatten(results)

with open(".cache/replay_traces/lmsys-chat-1m.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
