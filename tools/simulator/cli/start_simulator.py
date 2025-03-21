import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import json
from dataclasses import asdict
from simulator.core.global_engine import LLMGlobalEngine
from simulator.core.utils import load_trace
from simulator.ui import make_table
from rich.console import Console
from huggingface_hub import login
import matplotlib.pyplot as plt  # Add this import for plotting


console = Console()

def plot_slo_pass_rate(server, slo, output_path):
    slo_scales = [round(x, 2) for x in [0.3 + 0.1 * i for i in range(18)]]  # 0.3 to 2.0
    pass_rates = [server.SLO_pass_rate(slo * scale) for scale in slo_scales]

    plt.figure(figsize=(10, 6))
    plt.plot(slo_scales, pass_rates, marker='o', label="SLO Pass Rate")
    plt.title("SLO Pass Rate vs SLO Scale")
    plt.xlabel("SLO Scale")
    plt.ylabel("Pass Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def run_simulation(args):
    print(args)
    server = LLMGlobalEngine(args.input, float(args.arrival_rate))

    for i in range(args.n_engines):
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        # server.add_engine("meta-llama/Llama-2-7b-hf", "nvidia_A100", 4,4,4)

    server.start()
    server.save_results("./result/output.json")

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

    # Generate and save the SLO pass plot
    plot_slo_pass_rate(server, float(args.SLO), args.slo_plot_output)

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
    parser.add_argument(
        "--slo-plot-output",
        type=str,
        help="Output path for SLO pass plot",
        default="./result/slo_pass_plot.png",
    )
    args = parser.parse_args()
    run_simulation(args)
