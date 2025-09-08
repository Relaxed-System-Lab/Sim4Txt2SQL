import json
from collections import Counter
from typing import List, Dict

def load_trace_file(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def count_steps_in_trace(file_path: str) -> Counter:
    step_counter = Counter()
    traces = load_trace_file(file_path)
    
    for trace in traces:
        # Get the number of steps in each Text2SQLRequest
        num_steps = len(trace.get('Text2SQLRequest', []))
        if num_steps >= 12:  # Only count if it has 12 or more steps
            step_counter[num_steps] += 1
    
    return step_counter

def write_results_to_markdown(results_file: str, trace_files: List[str]):
    with open(results_file, 'w') as f:
        f.write("# Text2SQL Request Step Analysis\n\n")
        
        for file_path in trace_files:
            step_counter = count_steps_in_trace(file_path)
            
            # Write header for this trace file
            f.write(f"## Analysis of {file_path}\n\n")
            f.write("| Number of steps | Count |\n")
            f.write("|----------------|-------|\n")
            
            # Write data rows
            for num_steps in sorted(step_counter.keys()):
                f.write(f"| {num_steps:14d} | {step_counter[num_steps]:5d} |\n")
            
            # Write total for this file
            total_requests = sum(step_counter.values())
            f.write(f"\n**Total requests in {file_path}:** {total_requests}\n\n")
            
            # Add some statistics
            if step_counter:
                min_steps = min(step_counter.keys())
                max_steps = max(step_counter.keys())
                avg_steps = sum(k * v for k, v in step_counter.items()) / total_requests
                f.write(f"**Statistics:**\n")
                f.write(f"- Minimum steps: {min_steps}\n")
                f.write(f"- Maximum steps: {max_steps}\n")
                f.write(f"- Average steps: {avg_steps:.2f}\n\n")
            f.write("---\n\n")

def main():
    trace_files = [
        'input_file_trace1.json',
        'input_file_trace2.json',
        'input_file_trace3.json'
    ]
    
    results_file = 'trace_analysis_results.md'
    write_results_to_markdown(results_file, trace_files)
    print(f"Analysis results have been saved to {results_file}")

if __name__ == "__main__":
    main()
