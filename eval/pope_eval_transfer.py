import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate RG vs CD results against reference labels")
    parser.add_argument("--ref-files", type=str, required=True, help="Path to the reference (ground truth) JSONL file")
    parser.add_argument("--res-rg-files", type=str, required=True, help="Path to the RG results JSONL file")
    parser.add_argument("--res-cd-files", type=str, required=True, help="Path to the CD results JSONL file")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dictionaries.
    Each line in the file should be a valid JSON object.
    """
    expanded = os.path.expanduser(path)
    data = []
    with open(expanded, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def normalize_answer(ans: str) -> str:
    """
    Lowercase and strip whitespace from an answer string.
    """
    return ans.lower().strip()


def evaluate_metrics(
    refs: List[Dict[str, Any]],
    res_rg: List[Dict[str, Any]],
    res_cd: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Compare reference labels with RG and CD results, computing transition counts.

    Returns a dictionary with the following keys:
      - TP2TP, TP2FN, FN2TP, FN2FN,
      - TN2TN, TN2FP, FP2TN, FP2FP
    """
    # Initialize all counters to zero
    counters = {
        "TP2TP": 0,
        "TP2FN": 0,
        "FN2TP": 0,
        "FN2FN": 0,
        "TN2TN": 0,
        "TN2FP": 0,
        "FP2TN": 0,
        "FP2FP": 0,
    }

    # Iterate over all examples (assume same length and aligned order)
    for idx in tqdm(range(len(refs)), desc="Evaluating"):
        ref_item = refs[idx]
        rg_item = res_rg[idx]
        cd_item = res_cd[idx]

        # Sanity checks: question_id must match across all files
        qid = ref_item["question_id"]
        assert qid == rg_item["question_id"] == cd_item["question_id"], \
            f"Mismatch in question_id at index {idx}: {qid}, {rg_item['question_id']}, {cd_item['question_id']}"

        # Normalize text
        ref_label = normalize_answer(ref_item["label"])
        rg_ans = normalize_answer(rg_item["text"])
        cd_ans = normalize_answer(cd_item["text"])

        # Determine RG prediction outcome vs. reference
        if ref_label == "yes":
            # RG predicted positive (TP) if it contains "yes", else FN
            rg_positive = ("yes" in rg_ans)
            # Then check CD prediction
            cd_positive = ("yes" in cd_ans)

            if rg_positive:
                # TP case
                if cd_positive:
                    counters["TP2TP"] += 1
                else:
                    counters["TP2FN"] += 1
            else:
                # FN case
                if cd_positive:
                    counters["FN2TP"] += 1
                else:
                    counters["FN2FN"] += 1

        else:
            # ref_label != "yes" → negative reference
            # RG predicted negative (TN) if it contains "no", else FP
            rg_negative = ("no" in rg_ans)
            cd_negative = ("no" in cd_ans)

            if rg_negative:
                # TN case
                if cd_negative:
                    counters["TN2TN"] += 1
                else:
                    counters["TN2FP"] += 1
            else:
                # FP case
                if cd_negative:
                    counters["FP2TN"] += 1
                else:
                    counters["FP2FP"] += 1

    return counters


def print_transition_summary(c: Dict[str, int]) -> None:
    """
    Print summary of transitions between RG and CD for both positive/negative flips.
    """
    # P2N: RG positive → CD negative = TP2FN + FP2TN
    p2n = c["TP2FN"] + c["FP2TN"]
    # N2P: RG negative → CD positive = FN2TP + TN2FP
    n2p = c["FN2TP"] + c["TN2FP"]

    print(f"P2N (Positive→Negative): {p2n}")
    print(f"N2P (Negative→Positive): {n2p}")
    print("--------------")
    print(f"TP2TP: {c['TP2TP']}")
    print(f"TP2FN: {c['TP2FN']}")
    print("--------------")
    print(f"FP2FP: {c['FP2FP']}")
    print(f"FP2TN: {c['FP2TN']}")
    print("--------------")
    print(f"TN2TN: {c['TN2TN']}")
    print(f"TN2FP: {c['TN2FP']}")
    print("--------------")
    print(f"FN2FN: {c['FN2FN']}")
    print(f"FN2TP: {c['FN2TP']}")
    print("--------------")


def main():
    args = parse_args()

    # Load files
    ref_list = load_jsonl(args.ref_files)
    rg_list = load_jsonl(args.res_rg_files)
    cd_list = load_jsonl(args.res_cd_files)

    # Ensure all lengths match
    assert len(ref_list) == len(rg_list) == len(cd_list), "Input files must have the same number of lines."

    # Compute transition counters
    counters = evaluate_metrics(ref_list, rg_list, cd_list)

    # Print out summary
    print_transition_summary(counters)


if __name__ == "__main__":
    main()
