import unittest

from scripts.plot_threshold_results import THEMES, luminance, readable_ink


class ReadableInkTests(unittest.TestCase):
    def test_dark_text_on_light_fill_and_white_on_dark_fill(self):
        self.assertEqual(readable_ink("#cde2fb"), "#0b0b0b")
        self.assertEqual(readable_ink("#184f95"), "#ffffff")

    def test_every_ramp_step_gets_text_with_enough_contrast(self):
        for theme in THEMES.values():
            for fill in theme["ramp"]:
                with self.subTest(fill=fill):
                    ink = readable_ink(fill)
                    lighter, darker = sorted((luminance(fill), luminance(ink)), reverse=True)
                    self.assertGreaterEqual((lighter + 0.05) / (darker + 0.05), 4.5)


if __name__ == "__main__":
    unittest.main()
