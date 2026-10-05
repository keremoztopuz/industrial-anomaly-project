"""Compare is_anomaly threshold rules on a validation half of the test set.

Protocol, fixed before looking at any result:
- Every rule turns a category's held-out normal training scores into a threshold.
  Candidates: "max" (the current rule) and "median + k * MAD" for k in 2..6, where
  MAD is scaled by 1.4826 so it matches the standard deviation of normal data.
- Each category's test images are split in half, separately for normal and
  defective images (seed 42). Rules compete on the validation half only.
- The winner has the highest mean balanced accuracy over the 15 categories.
  Ties go to the more conservative rule: larger k, with "max" the most conservative.
- The winner and the current rule are then scored once on the final half.

Every candidate is logged as a run in the MLflow experiment "threshold-rules".

Run from the repository root: python -m scripts.modeling.compare_threshold_rules
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from mlflow.tracking import MlflowClient

from anomaly.evaluation.scoring import balanced_accuracy, confusion
from anomaly.evaluation.splits import split_halves
from anomaly.utils import write_json

EXPERIMENT = "threshold-rules"
MAD_SCALE = 1.4826
K_VALUES = (2, 3, 4, 5, 6)


def max_rule(scores):
    return max(scores)


def mad_rule(scores, k):
    median = statistics.median(scores)
    mad = statistics.median(abs(score - median) for score in scores)
    return median + k * MAD_SCALE * mad


def candidates():
    """Return (name, k, rule) with k=inf for max, the most conservative candidate."""
    rules = [("max", math.inf, max_rule)]
    rules += [(f"median+{k}mad", k, lambda scores, k=k: mad_rule(scores, k)) for k in K_VALUES]
    return rules


def evaluate(thresholds, scores, splits, part):
    """Score every category's chosen half; return per-category results and pooled counts."""
    per_category, pooled = {}, {"true_positive": 0, "false_positive": 0,
                                "true_negative": 0, "false_negative": 0}
    for category, threshold in thresholds.items():
        indices = splits[category][part]
        values = torch.tensor([scores[category]["scores"][i] for i in indices])
        labels = torch.tensor([scores[category]["labels"][i] for i in indices])
        result = confusion(values, labels, threshold)
        result["balanced_accuracy"] = balanced_accuracy(result)
        per_category[category] = result
        for key in pooled:
            pooled[key] += result[key]
    return per_category, pooled


def summary_metrics(prefix, per_category, pooled):
    metrics = {f"{prefix}_balanced_accuracy_mean": statistics.mean(
        result["balanced_accuracy"] for result in per_category.values())}
    positives = pooled["true_positive"] + pooled["false_negative"]
    negatives = pooled["false_positive"] + pooled["true_negative"]
    metrics[f"{prefix}_recall"] = pooled["true_positive"] / positives
    metrics[f"{prefix}_false_positive_rate"] = pooled["false_positive"] / negatives
    for category, result in per_category.items():
        metrics[f"{prefix}_balanced_accuracy/{category}"] = result["balanced_accuracy"]
    return metrics


def pick_winner(results):
    """results: list of (name, k, validation mean balanced accuracy)."""
    return max(results, key=lambda item: (item[2], item[1]))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action="store_true",
                        help="Write the winner's thresholds to thresholds.json")
    args = parser.parse_args()

    calibration = json.loads((args.model_dir / "thresholds.json").read_text(encoding="utf-8"))
    holdout = {category: values["holdout_scores"]
               for category, values in calibration["categories"].items()}
    scores = json.loads((args.model_dir / "test_scores.json").read_text(encoding="utf-8"))
    splits = {}
    for category, data in scores.items():
        validation, final = split_halves(data["labels"], args.seed)
        splits[category] = {"validation": validation, "final": final}

    client = MlflowClient(args.tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (experiment.experiment_id if experiment
                     else client.create_experiment(EXPERIMENT))
    rules, runs, results = {}, {}, []
    for name, k, rule in candidates():
        thresholds = {category: rule(values) for category, values in holdout.items()}
        per_category, pooled = evaluate(thresholds, scores, splits, "validation")
        metrics = summary_metrics("val", per_category, pooled)
        run = client.create_run(experiment_id, run_name=name)
        runs[name] = run.info.run_id
        rules[name] = thresholds
        for key, value in {"rule": name.split("+")[0], "k": k, "split_seed": args.seed,
                           "calibration_method": calibration["method"]}.items():
            client.log_param(run.info.run_id, key, value)
        for key, value in metrics.items():
            client.log_metric(run.info.run_id, key, value)
        results.append((name, k, metrics["val_balanced_accuracy_mean"]))
        print(f"{name}: validation balanced accuracy {metrics['val_balanced_accuracy_mean']:.4f}, "
              f"recall {metrics['val_recall']:.3f}, false positives "
              f"{metrics['val_false_positive_rate']:.3f}")

    winner = pick_winner(results)
    for name in dict.fromkeys((winner, "max")):
        per_category, pooled = evaluate(rules[name], scores, splits, "final")
        final = summary_metrics("final", per_category, pooled)
        for key, value in final.items():
            client.log_metric(runs[name], key, value)
        print(f"final half, {name}: balanced accuracy "
              f"{final['final_balanced_accuracy_mean']:.4f}, "
              f"recall {final['final_recall']:.3f}, "
              f"false positives {final['final_false_positive_rate']:.3f}")
    client.set_tag(runs[winner], "selected", "true")
    for run_id in runs.values():
        client.set_terminated(run_id)
    print(f"winner: {winner}")

    if args.write and winner != "max":
        for category, values in calibration["categories"].items():
            values["threshold"] = rules[winner][category]
        calibration["method"] = winner
        calibration["selected_by"] = (f"mean validation balanced accuracy on half of the "
                                      f"test set, split seed {args.seed}")
        write_json(args.model_dir / "thresholds.json", calibration)
        print(f"wrote {winner} thresholds to {args.model_dir / 'thresholds.json'}")


if __name__ == "__main__":
    main()
