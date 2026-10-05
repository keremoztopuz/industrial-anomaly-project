"""Pick a category's input resolution on a validation half of its test set.

Protocol, fixed before looking at any result:
- Candidates are banks for one category fit at different square input sizes, all with the
  deployed settings otherwise. The deployed 256 bank is the baseline.
- The category's test images are split in half, separately for normal and defective
  images (seed 42, compare_threshold_rules.split_halves).
- The winner has the highest image AUROC on the validation half; ties go to the smaller
  size, which is cheaper to serve.
- The winner and the baseline are then scored once on the final half.

Each candidate is logged as a run in the MLflow experiment "<category>-resolution".

Run from the repository root:
python -m scripts.modeling.compare_resolutions --category pill \
    --bank 256=path/pill.pt --bank 320=path/pill.pt
"""

import argparse
from pathlib import Path

import torch
from mlflow.tracking import MlflowClient

from anomaly.data.mvtec_dataset import MVTecDataset
from anomaly.evaluation.metrics import binary_auroc
from anomaly.evaluation.scoring import score
from anomaly.evaluation.splits import split_halves
from anomaly.model.patchcore import PatchCore
from anomaly.utils import default_device


def auroc(scores, labels, indices):
    return binary_auroc(scores[indices], labels[indices])


def pick_winner(validation):
    """validation: {size: AUROC}. Highest AUROC wins, ties go to the smaller size."""
    return max(validation, key=lambda size: (validation[size], -size))


@torch.inference_mode()
def score_bank(path, dataset_root, category, device, batch_size):
    model = PatchCore.load(path, device)
    scores, labels = score(model, MVTecDataset(dataset_root, category, "test",
                                               model.image_size), batch_size)
    return model.image_size, scores, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--bank", action="append", required=True, metavar="SIZE=PATH",
                        help="Candidate bank; the first one is the deployed baseline")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    banks = {int(size): Path(path) for size, path in (bank.split("=", 1) for bank in args.bank)}
    baseline = next(iter(banks))
    results, split = {}, None
    for size, path in banks.items():
        stored_size, scores, labels = score_bank(path, args.dataset_root, args.category,
                                                 args.device, args.batch_size)
        if stored_size != size:
            parser.error(f"{path} was fit at {stored_size}, not {size}")
        if split is None:
            split = split_halves(labels.tolist(), args.seed)
        results[size] = (scores, labels)
    validation_indices, final_indices = (torch.tensor(part) for part in split)

    client = MlflowClient(args.tracking_uri)
    name = f"{args.category}-resolution"
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
    validation, runs = {}, {}
    for size, (scores, labels) in results.items():
        validation[size] = auroc(scores, labels, validation_indices)
        run = client.create_run(experiment_id, run_name=f"{args.category}-{size}")
        runs[size] = run.info.run_id
        client.log_param(run.info.run_id, "image_size", size)
        client.log_param(run.info.run_id, "split_seed", args.seed)
        client.log_param(run.info.run_id, "bank", str(banks[size]))
        client.log_metric(run.info.run_id, "val_image_auroc", validation[size])
        print(f"{args.category} {size}: validation image AUROC {validation[size]:.4f}")

    winner = pick_winner(validation)
    for size in dict.fromkeys((winner, baseline)):
        scores, labels = results[size]
        final = auroc(scores, labels, final_indices)
        client.log_metric(runs[size], "final_image_auroc", final)
        print(f"final half, {size}: image AUROC {final:.4f}")
    client.set_tag(runs[winner], "selected", "true")
    for run_id in runs.values():
        client.set_terminated(run_id)
    print(f"winner: {winner}")


if __name__ == "__main__":
    main()
