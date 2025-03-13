import os
from scripts.configs import (
    tiny_model_lists,
    small_model_lists,
    large_model_lists,
    compress_configs,
)

models = tiny_model_lists + small_model_lists + large_model_lists

for model in models:
    for config in compress_configs:
        model_id = model["name"].replace("/", "-")
        output_dir = f".cache/models/{model_id}---{config}"
        os.system(
            f"ts -G 1 python adaml/cli/compressor.py --model {model['name']} --scheme {config} --output_dir {output_dir}"
        )
