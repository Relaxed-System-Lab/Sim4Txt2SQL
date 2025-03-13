import os
import json
import pandas as pd
from scripts.configs import small_model_lists, tiny_model_lists, large_model_lists

models = small_model_lists + tiny_model_lists + large_model_lists
from scripts.configs import compress_configs
import difflib
from adaml.internal.analyzer.api import analyze_perf

BASE_PATH = ".cache/eval/results"
METRIC = "acc,none"
model_names = [x["name"] for x in models]

data = []
eval_results = os.listdir(".cache/eval/results")
for eval in eval_results:
    result_file = os.listdir(f"{BASE_PATH}/{eval}")
    if len(result_file) > 0:
        with open(f"{BASE_PATH}/{eval}/{result_file[0]}", "r") as f:
            datum = json.load(f)
            quality = datum["results"]["mmlu"][METRIC]
            model_name = eval.replace(".cache__models__", "")
            model_params = model_name.split("---")
            if len(model_params) == 2:
                model_name, params = model_params
            else:
                params = "W16A16"
            parsed_params = params.split("W")[1].split("A")
            wbit, abit = int(parsed_params[0]), int(parsed_params[1])
            perf = {"prefill": 0, "decode": 0}
            if "llama" in model_name.lower():
                model_id = difflib.get_close_matches(model_name, model_names)[0]
                perf = analyze_perf(
                    model_id, "nvidia_A100", 16, 2048, 512, wbit, abit, 16
                )
            data.append(
                {
                    "model": difflib.get_close_matches(model_name, model_names)[0],
                    "quality": quality,
                    "params": params,
                    "prefill": perf["prefill"],
                    "decode": perf["decode"],
                }
            )

df = pd.DataFrame(data)
print(df)
df.to_csv(".cache/results/agg_eval.csv", index=False)
missing_configs = []
# find missing models
for model in models:
    for config in compress_configs:
        # if the model and config are not in the dataframe
        if df[(df["model"] == model["name"]) & (df["params"] == config)].empty:
            missing_configs.append(
                {"model": model["name"], "chat": model["chat"], "config": config}
            )
missed = pd.DataFrame(missing_configs)
missed.to_csv(".cache/missed.csv", index=False)
