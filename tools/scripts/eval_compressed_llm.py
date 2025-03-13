import os
import json
import pandas as pd

MODELS_PATH = ".cache/models/"
models = os.listdir(MODELS_PATH)
jobs = []
missed = pd.read_csv(".cache/missed.csv")
models = []

for row in missed.iterrows():
    row = row[1]
    model_name = row["model"]
    model_id = model_name.replace("/", "-")
    model_path = f".cache/models/{model_id}---{row['config']}"
    models.append(model_path)

for model_path in models:

    # model_path = os.path.join(MODELS_PATH, model)
    # check if tokenizer_config.json exists
    if not os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        apply_chat_template = False
    else:
        with open(os.path.join(model_path, "tokenizer_config.json"), "r") as f:
            config = json.load(f)
            if "chat_template" in config:
                apply_chat_template = True
            else:
                apply_chat_template = False
    job = f"ts -G 1 lm_eval --model vllm --model_args pretrained={model_path},tensor_parallel_size=1 --tasks mmlu --use_cache .cache/eval/cache/{model_path} --batch_size 1 --output_path .cache/eval/results/{model_path} --wandb_args project=adaptiveML{' --apply_chat_template' if apply_chat_template else ''}"
    jobs.append(job)

for job in jobs:
    os.system(job)
