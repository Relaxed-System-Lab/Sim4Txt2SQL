import os
from scripts.configs import small_model_lists, tiny_model_lists, large_model_lists

models = large_model_lists

model_configs = [
    "dtype=bfloat16",
]
for model in models:
    apply_chat_template = model["chat"]
    custom_template = False
    if apply_chat_template:
        if "template" in model:
            template = model["template"]
            custom_template = True
    model = model["name"]
    for config in model_configs:
        description = f"model={model}---config={config}"
        description = description.replace("=", "--")
        template_command = ""
        if apply_chat_template and custom_template:
            template_command = f" --apply_chat_template {template}"
        elif apply_chat_template and not custom_template:
            template_command = f" --apply_chat_template"
        else:
            template_command = ""
        os.system(
            f"ts -G 1 lm_eval --model hf --model_args pretrained={model},dtype=bfloat16 --use_cache .cache/eval/cache/{model} --tasks mmlu --batch_size auto --device cuda:0 --output_path .cache/eval/results --wandb_args project=adaptiveML,name={model},notes={description}{template_command}"
        )
