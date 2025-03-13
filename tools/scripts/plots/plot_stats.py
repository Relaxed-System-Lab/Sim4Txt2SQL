import os
import json
import pandas as pd

def extract_engine_name(engine_config):
    engine_ids = []
    for model in engine_config.keys():
        for engine in engine_config[model]["engines"]:
            quant_config = f"w{engine['w_bit']}a{engine['a_bit']}k{engine['kv_bit']}"
            hardware_name = engine['hardware'].replace("nvidia_", "")
            engine_id = f"{hardware_name}_{quant_config}"
            engine_ids.append(engine_id)
    # convert engine_ids to a string, e.g., 'n'_a100_w16a16k16 where n is the number of engines
    engine_configs = set(engine_ids)
    engine_strs = []
    for config in engine_configs:
        n_engines = engine_ids.count(config)
        engine_strs.append(f"{n_engines}x_{config}")
    return ','.join(engine_strs)
    
def process_requests_stats(request_stats):
    stats = []
    for req in request_stats:
        stat = {}
        stat['TTFT'] = req['prefill_finished_at'] - req['arrive_at']
        stat['E2E Latency'] = req['generation_finished_at'] - req['arrive_at']
        stats.append(stat)
    return stats

def plot_stats(files):
    stats = []
    dfs = []
    for file in files:
        with open(file, "r") as f:
            stats.append(json.load(f))
    for stat in stats:
        config = stat["config"]
        engine_name = extract_engine_name(config)
        request_stats = process_requests_stats(stat["summary"])
        df = pd.DataFrame(request_stats)
        df['Engine'] = engine_name
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv("notebook/stats.csv", index=False)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input directory")
    args = parser.parse_args()
    stats_files = [os.path.join(args.input, x) for x in os.listdir(args.input) if x.endswith(".json") and "stats" in x]
    plot_stats(stats_files)