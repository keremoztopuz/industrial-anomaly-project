import unittest

from PIL import Image, ImageStat

from scripts.monitoring import simulate_drift


def dataset_with(good, defective):
    class Dataset:
        samples = [
            {"defect": {"label": "good"}, "filepath": f"g{i}.png"}
            for i in range(good)
        ] + [
            {"defect": {"label": "cut"}, "filepath": f"d{i}.png"}
            for i in range(defective)
        ]

    return Dataset()


class PickSamplesTests(unittest.TestCase):
    def test_lighting_scenarios_use_only_good_images(self):
        chosen = simulate_drift.pick_samples(
            dataset_with(60, 40), "dark", 50, seed=42
        )
        self.assertEqual(len(chosen), 50)
        self.assertTrue(all(s["defect"]["label"] == "good" for s in chosen))
        self.assertEqual(len({s["filepath"] for s in chosen}), 50)

    def test_defect_wave_is_mostly_defective(self):
        chosen = simulate_drift.pick_samples(
            dataset_with(60, 40), "defects", 50, seed=42
        )
        self.assertEqual(
            sum(s["defect"]["label"] != "good" for s in chosen), 40
        )

    def test_same_seed_same_images(self):
        first = simulate_drift.pick_samples(
            dataset_with(60, 40), "normal", 50, seed=1
        )
        second = simulate_drift.pick_samples(
            dataset_with(60, 40), "normal", 50, seed=1
        )
        self.assertEqual(first, second)

    def test_too_few_images_is_an_error(self):
        dataset = dataset_with(20, 40)
        with self.assertRaises(ValueError):
            simulate_drift.pick_samples(dataset, "normal", 50, seed=42)


class TargetTests(unittest.TestCase):
    def test_only_fixed_targets_exist(self):
        self.assertEqual(sorted(simulate_drift.TARGETS), ["live", "local"])
        self.assertTrue(simulate_drift.TARGETS["live"].endswith(".run.app"))


class ApplyScenarioTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (64, 64), (200, 200, 200))
        self.image.paste((20, 20, 20), (0, 0, 32, 64))

    def stats(self, image):
        return ImageStat.Stat(image.convert("L"))

    def test_dark_lowers_brightness(self):
        dark = simulate_drift.apply_scenario(self.image, "dark")
        self.assertAlmostEqual(
            self.stats(dark).mean[0],
            self.stats(self.image).mean[0] * 0.6,
            delta=1,
        )

    def test_blur_lowers_contrast(self):
        blurred = simulate_drift.apply_scenario(self.image, "blur")
        self.assertLess(
            self.stats(blurred).stddev[0], self.stats(self.image).stddev[0]
        )

    def test_normal_and_defects_leave_images_unchanged(self):
        for scenario in ("normal", "defects"):
            self.assertIs(
                simulate_drift.apply_scenario(self.image, scenario), self.image
            )


if __name__ == "__main__":
    unittest.main()
