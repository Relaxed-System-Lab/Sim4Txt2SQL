from adaml.internal.analyzer.model_analyzer import ModelAnalyzer

model_id = 'meta-llama/Llama-2-7b-hf'

seq_len = 512
analyzer = ModelAnalyzer(
    model_id=model_id,
    hardware="nvidia_A100",
    config_file="adaml/internal/configs/llama.py",
    source="huggingface",
)

res = analyzer.analyze(seqlen=seq_len, batchsize=1, w_bit=16, a_bit=16, kv_bit=16, use_flashattention=False)

print(res)