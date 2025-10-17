import json
import random
import copy

def reorder_steps(request):
    """
    Reorder steps in a Text2SQLRequest so that 'Decomposer' becomes the last step.
    This swaps the order of 'Refiner' and 'Decomposer' if both are present.
    """
    steps = request["Text2SQLRequest"]
    
    # Find indices of Refiner and Decomposer steps
    refiner_idx = None
    decomposer_idx = None
    
    for i, step in enumerate(steps):
        if step["step"] == "Refiner":
            refiner_idx = i
        elif step["step"] == "Decomposer":
            decomposer_idx = i
    
    # If both Refiner and Decomposer exist, swap their positions
    if refiner_idx is not None and decomposer_idx is not None:
        # Create a copy of the steps list
        new_steps = steps.copy()
        # Swap the steps
        new_steps[refiner_idx], new_steps[decomposer_idx] = new_steps[decomposer_idx], new_steps[refiner_idx]
        return {"Text2SQLRequest": new_steps}
    
    # If no swap needed, return original request
    return request

def count_steps(request):
    return len(request["Text2SQLRequest"])

def get_step_names(request):
    """Get the step names in order for debugging purposes."""
    return [step["step"] for step in request["Text2SQLRequest"]]

def count_refiner_steps(request):
    """Count the number of Refiner steps in a request."""
    return sum(1 for step in request["Text2SQLRequest"] if step["step"] == "Refiner")

def add_refiner_steps(request, target_count):
    """Add Refiner steps at the end to reach the target count."""
    current_count = count_refiner_steps(request)
    if current_count >= target_count:
        return request
    
    # Get a sample Refiner step to use as template
    refiner_template = {
        "step": "Refiner",
        "input_length": random.choice([928, 1729, 1870]),
        "output_length": random.choice([389, 492, 555])
    }
    
    steps = request["Text2SQLRequest"]
    # Add new Refiner steps at the end
    for _ in range(target_count - current_count):
        steps.append(refiner_template.copy())
    
    return {"Text2SQLRequest": steps}

# Read the input file
input_file = "./input_file_trace4.json"
with open(input_file, 'r') as f:
    data = json.load(f)

print(f"Processing {len(data)} requests...")

# Group requests by current Refiner count
requests_by_refiner_count = {}
for request in data:
    count = count_refiner_steps(request)
    requests_by_refiner_count.setdefault(count, []).append(request)


# Create balanced distribution: ensure exactly 30 requests for each Refiner count 0..4
final_data = []

for target_count in range(5):
    existing = requests_by_refiner_count.get(target_count, []).copy()
    # If we have enough existing requests, sample 30
    if len(existing) >= 30:
        selected = random.sample(existing, 30)
    else:
        # Start with all existing
        selected = [copy.deepcopy(r) for r in existing]
        # Build a pool of candidate source requests we can modify to reach target_count.
        # Prefer requests that already have <= target_count refiners so add_refiner_steps is natural.
        pool = []
        for c in range(target_count + 1):
            pool.extend(requests_by_refiner_count.get(c, []))
        # Fallback to all requests if pool is empty
        if not pool:
            pool = data.copy()

        # Create additional requests by deep-copying and adding Refiner steps until we have 30
        while len(selected) < 30:
            src = random.choice(pool)
            newreq = copy.deepcopy(src)
            newreq = add_refiner_steps(newreq, target_count)
            selected.append(newreq)

    # Add the selected (deep-copied) items to final_data
    final_data.extend([copy.deepcopy(r) for r in selected])

# Verify final distribution
print("\nFinal distribution:")
for count in range(5):
    actual_count = sum(1 for r in final_data if count_refiner_steps(r) == count)
    print(f"Requests with {count} Refiner steps: {actual_count}")

# Shuffle the final data
random.shuffle(final_data)

# Save the modified data
output_file = "./input_file_trace4_modified2.json"
with open(output_file, 'w') as f:
    json.dump(final_data, f, indent=2)

print(f"\nModified data saved to: {output_file}")