import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from mlflow.tracking import MlflowClient

from scripts.modeling import compare_threshold_rules as rules


class RuleTests(unittest.TestCase):
    def test_mad_rule_ignores_one_outlier(self):
        normal = [10.0, 10.5, 11.0, 11.5, 12.0]
        self.assertAlmostEqual(rules.mad_rule(normal, 3), 11.0 + 3 * 1.4826 * 0.5)
        self.assertAlmostEqual(rules.mad_rule(normal + [30.0], 3),
                               rules.mad_rule(normal + [12.5], 3))
        self.assertEqual(rules.max_rule(normal + [30.0]), 30.0)

    def test_split_is_stratified_complete_and_seeded(self):
        labels = [0] * 5 + [1] * 8
        validation, final = rules.split_halves(labels, seed=42)
        self.assertEqual(sorted(validation + final), list(range(13)))
        self.assertEqual(sum(labels[i] == 0 for i in validation), 3)
        self.assertEqual(sum(labels[i] == 1 for i in validation), 4)
        self.assertEqual(rules.split_halves(labels, seed=42), (validation, final))

    def test_winner_is_best_then_most_conservative(self):
        self.assertEqual(rules.pick_winner([("max", math.inf, 0.8), ("median+3mad", 3, 0.9)]),
                         "median+3mad")
        self.assertEqual(rules.pick_winner([("median+2mad", 2, 0.9), ("median+5mad", 5, 0.9)]),
                         "median+5mad")
        self.assertEqual(rules.pick_winner([("max", math.inf, 0.9), ("median+6mad", 6, 0.9)]),
                         "max")


class EndToEndTests(unittest.TestCase):
    def test_logs_every_candidate_and_writes_winner(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            model_dir = root / "banks"
            model_dir.mkdir()
            holdout = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 25.0]
            (model_dir / "thresholds.json").write_text(json.dumps({
                "method": "max held-out normal score",
                "categories": {"pill": {"threshold": 25.0, "holdout_scores": holdout}},
            }), encoding="utf-8")
            test_scores = [10.1, 10.5, 10.9, 11.1] + [14.0, 16.0, 18.0, 20.0]
            (model_dir / "test_scores.json").write_text(json.dumps({
                "pill": {"scores": test_scores, "labels": [0] * 4 + [1] * 4}}), encoding="utf-8")
            uri = f"sqlite:///{root / 'mlflow.db'}"
            client = MlflowClient(uri)
            experiment_id = client.create_experiment(
                "threshold-rules", artifact_location=(root / "store").as_uri())

            argv = ["compare", "--model-dir", str(model_dir), "--tracking-uri", uri, "--write"]
            with mock.patch("sys.argv", argv), redirect_stdout(StringIO()):
                rules.main()

            runs = {run.info.run_name: run for run in client.search_runs([experiment_id])}
            self.assertEqual(set(runs), {"max", "median+2mad", "median+3mad", "median+4mad",
                                         "median+5mad", "median+6mad"})
            [winner] = [name for name, run in runs.items() if run.data.tags.get("selected")]
            self.assertNotEqual(winner, "max")
            self.assertIn("final_balanced_accuracy_mean", runs[winner].data.metrics)
            self.assertIn("final_balanced_accuracy_mean", runs["max"].data.metrics)
            for name in set(runs) - {winner, "max"}:
                self.assertNotIn("final_balanced_accuracy_mean", runs[name].data.metrics)
            written = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
            self.assertEqual(written["method"], winner)
            self.assertLess(written["categories"]["pill"]["threshold"], 25.0)
            self.assertEqual(written["categories"]["pill"]["holdout_scores"], holdout)


if __name__ == "__main__":
    unittest.main()
