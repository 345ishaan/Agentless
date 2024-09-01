import json

# Load the results file
with open('results_gpt-4o_reflect_claude-3-5-sonnet_lite_all_preds.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# Create a mapping of instance_id to model_patch
patch_mapping = {entry['instance_id']: entry['model_patch'] for entry in results if entry['model_patch']}

# Load the merged predictions file
with open('merged_preds.jsonl', 'r') as f:
    merged_preds = [json.loads(line) for line in f]

# Update empty model_patches
for entry in merged_preds:
    if entry['model_patch'] == "":
        instance_id = entry['instance_id']
        if instance_id in patch_mapping:
            entry['model_patch'] = patch_mapping[instance_id]

# Save the updated merged predictions back to the file
with open('merged_preds_updated.jsonl', 'w') as f:
    for entry in merged_preds:
        f.write(json.dumps(entry) + '\n')