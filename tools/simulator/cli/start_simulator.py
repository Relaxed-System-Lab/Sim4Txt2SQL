import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import json
from dataclasses import asdict
from simulator.core.global_engine import LLMGlobalEngine
from simulator.core.global_engine_optimized import OPGlobalEngine
from simulator.core.utils import load_trace
from simulator.ui import make_table
from rich.console import Console
from huggingface_hub import login
import matplotlib.pyplot as plt  # Add this import for plotting


console = Console()

def run_simulation(args):
    # print(args)
    server = LLMGlobalEngine(args.input, float(args.arrival_rate))

    for i in range(args.n_engines):
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        # server.add_engine("meta-llama/Llama-2-7b-hf", "nvidia_A100", 4,4,4)

    server.start()
    server.save_results("./result/baseline_output.json")

    with open(args.trace_output, "w") as f:
        data = {"traceEvents": [asdict(x) for x in server.trace]}
        f.write(json.dumps(data, indent=4))
    stats = {
        "summary": server.requests_stats,
        "failed": server.failed_requests,
        "config": server.config,
    }
    with open(args.stats_output, "w") as f:
        f.write(json.dumps(stats, indent=4))

    print(end="\n")
    print(f"--" * 10 + " Simulation Done " + "--" * 10)

    console.print(make_table("Summary", server.summary))
    # print(f"Pass rate: {server.SLO_pass_rate(float(args.SLO))}")
    slo_scales = [round(x, 2) for x in [0.3 + 0.1 * i for i in range(18)]]  # 0.3 to 2.0
    pass_rates = [server.SLO_pass_rate(args.SLO * scale) for scale in slo_scales]
    return pass_rates

def run_simulation_optimized(args, w1=0.6, w2=0.4, index=0):
    # print(args)
    pass_rates = []
    slo_scales = [round(x, 2) for x in [0.3 + 0.1 * i for i in range(18)]]
    for scale in slo_scales:
        server = OPGlobalEngine(args.input, float(args.arrival_rate), args.SLO * scale)

        for i in range(args.n_engines):
            server.add_engine(w1, w2, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
            # server.add_engine("meta-llama/Llama-2-7b-hf", "nvidia_A100", 4,4,4)

        server.start()
        server.save_results(f"./result/optimized_output{index}.json")

        with open(f"./result/optimized_trace_output{index}.json", "w") as f:
            data = {"traceEvents": [asdict(x) for x in server.trace]}
            f.write(json.dumps(data, indent=4))
        stats = {
            "summary": server.requests_stats,
            "failed": server.failed_requests,
            "config": server.config,
        }
        with open(f"./result/optimized_stats_output{index}.json", "w") as f:
            f.write(json.dumps(stats, indent=4))

        print(end="\n")
        print(f"--" * 10 + " Simulation Done " + "--" * 10)

        console.print(make_table("Summary", server.summary))

        pass_rates.append(server.SLO_pass_rate(args.SLO * scale))
    return pass_rates

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--n-engines", type=int, help="Number of engines")
    parser.add_argument("--arrival-rate", help="Arrival rate", default=None)
    parser.add_argument("--SLO", help="Text2SQL Request SLO", default=35.28)

    parser.add_argument(
        "--trace-output",
        type=str,
        help="Trace file",
        default=".cache/replay_results/trace.json",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        help="Stats file",
        default=".cache/replay_results/stats.json",
    )
    args = parser.parse_args()
    pass_rates_baseline = run_simulation(args)
    i = 0
    for w1 in range(0, 1, 0.1):
        w2 = 1 - w1
        pass_rates_optimized = run_simulation_optimized(args, w1, w2, i)
        # Plotting the pass rates for both baseline and optimized
        plt.figure(figsize=(10, 6))
        plt.plot([round(x, 2) for x in [0.3 + 0.1 * i for i in range(18)]], pass_rates_baseline, marker='o', label="Baseline SLO Pass Rate")
        plt.plot([round(x, 2) for x in [0.3 + 0.1 * i for i in range(18)]], pass_rates_optimized, marker='o', label="Optimized SLO Pass Rate")
        plt.title(f"SLO Pass Rate vs SLO Scale (w1={w1}, w2={w2})")
        plt.xlabel("SLO Scale")
        plt.ylabel("Pass Rate")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"./result/slo_pass_plot_{i}.png")
        plt.close()
        # Save the pass rates to a JSON file
        with open(f"./result/pass_rates_{i}.json", "w") as f:
            json.dump({
                "baseline": pass_rates_baseline,
                "optimized": pass_rates_optimized
            }, f, indent=4)
        i += 1
