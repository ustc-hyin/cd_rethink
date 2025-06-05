import os
import json
import argparse
from typing import List, Tuple, Dict
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute binary classification metrics (yes/no).")
    parser.add_argument("--ref-files", type=str, required=True, help="Path to reference (GT) file (one JSON per line).")
    parser.add_argument("--res-files", type=str, required=True, help="Path to result (generated) file (one JSON per line).")
    return parser.parse_args()


def load_json_lines(file_path: str) -> List[Dict]:
    """
    Read a file where each line is a JSON object, return a list of dicts.
    """
    path = os.path.expanduser(file_path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def evaluate_binary(
    ref_data: List[Dict], res_data: List[Dict]
) -> Tuple[int, int, int, int, int]:
    """
    Compare reference and result entries one by one, counting true positives,
    true negatives, false positives, false negatives, and unknown labels.
    Returns (tp, tn, fp, fn, unknown_count).
    """
    true_pos = true_neg = false_pos = false_neg = unknown = 0

    for ref_item, res_item in tqdm(zip(ref_data, res_data), total=len(ref_data), desc="Evaluating"):
        # Ensure question_id matches between ref and res
        ref_id, res_id = ref_item["question_id"], res_item["question_id"]
        if ref_id != res_id:
            raise ValueError(f"ID mismatch: REF={ref_id}, RES={res_id}")

        ref_ans = ref_item["label"].strip().lower()
        res_ans = res_item["text"].strip().lower()

        if ref_ans == "yes":
            if "yes" in res_ans:
                true_pos += 1
            else:
                false_neg += 1
        elif ref_ans == "no":
            if "no" in res_ans:
                true_neg += 1
            else:
                false_pos += 1
        else:
            # Count labels other than "yes" or "no" as unknown
            unknown += 1

    return true_pos, true_neg, false_pos, false_neg, unknown


def compute_metrics(
    tp: int, tn: int, fp: int, fn: int, total: int, unknown: int
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, accuracy, yes_proportion, and unknown_proportion.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    yes_count = tp + fp  # total predicted "yes"
    yes_proportion = yes_count / total if total > 0 else 0.0
    unknown_prop = unknown / total if total > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Yes_Proportion": yes_proportion,
        "Unknown_Proportion": unknown_prop,
    }


def main():
    args = parse_args()
    ref_data = load_json_lines(args.ref_files)
    res_data = load_json_lines(args.res_files)

    if len(ref_data) != len(res_data):
        raise ValueError(f"REF length ({len(ref_data)}) and RES length ({len(res_data)}) do not match")

    tp, tn, fp, fn, unknown = evaluate_binary(ref_data, res_data)
    total = len(ref_data)

    metrics = compute_metrics(tp, tn, fp, fn, total, unknown)

    # Print summary
    print(f"Total Questions: {total}")
    print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, Unknown: {unknown}")
    print("Metrics:")
    print(f"  Precision         : {metrics['Precision']:.4f}")
    print(f"  Recall            : {metrics['Recall']:.4f}")
    print(f"  F1 Score          : {metrics['F1']:.4f}")
    print(f"  Accuracy          : {metrics['Accuracy']:.4f}")
    print(f"  Yes Proportion    : {metrics['Yes_Proportion']:.4f}")
    print(f"  Unknown Proportion: {metrics['Unknown_Proportion']:.4f}")


if __name__ == "__main__":
    main()
