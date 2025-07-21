import json

INPUT_FILE = 'combined_tatqa_pred.json'
OUTPUT_FILE = 'fixed_combined_tatqa_pred.json'

def fix_scale_field(predictions):
    for k, v in predictions.items():
        if isinstance(v, list) and len(v) > 1:
            # If the second element is a list, convert to string (join with ', ')
            if isinstance(v[1], list):
                v[1] = ', '.join(str(x) for x in v[1])
            else:
                v[1] = str(v[1])
    return predictions

def main():
    with open(INPUT_FILE, 'r') as f:
        predictions = json.load(f)
    fixed_predictions = fix_scale_field(predictions)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(fixed_predictions, f, indent=2)
    print(f"Fixed predictions written to {OUTPUT_FILE}")

if __name__ == '__main__':
    main() 