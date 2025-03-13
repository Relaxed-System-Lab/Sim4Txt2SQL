# [0, 4B]
tiny_model_lists = [
    {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "chat": True,
    },
    {
        "name": "TinyLlama/TinyLlama_v1.1",
        "chat": False,
    },
]
# [4B - 10B]
small_model_lists = [
    {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "chat": True,
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-8B",
        "chat": False,
    },
    {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "chat": True,
    },
    {
        "name": "akjindal53244/Llama-3.1-Storm-8B",
        "chat": True,
    },
    {
        "name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "chat": True,
    },
]

# > 10B
large_model_lists = [
    {
        "name": "meta-llama/Llama-2-13b-hf",
        "chat": False,
    }
]
compress_configs = ["W8A8", "W4A16", "W8A16"]

tasks = [
    "leaderboard_bbh",
    "leaderboard_gpqa",
    "leaderboard_ifeval",
    "leaderboard_math_hard",
    "leaderboard_musr",
]
# convert tasks to comma-separated string
tasks = ",".join(tasks)
