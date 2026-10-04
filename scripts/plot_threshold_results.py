"""Draw the threshold results as light and dark SVG figures for the README.

Scores every test image with the deployed banks (cached in test_scores.json next to
them), applies thresholds.json, and writes three figures per theme:
a confusion matrix over all categories, recall and false positive rate per category,
and the test score distributions with each category's threshold.

Run from the repository root: python -m scripts.plot_threshold_results
"""

import argparse
import json
from html import escape
from pathlib import Path

import torch

from mvtec_dataset import MVTecDataset
from patchcore import PatchCore
from scripts.calibrate_thresholds import confusion, default_device, score, write_json

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "normal": "#2a78d6", "defect": "#eb6834",
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "normal": "#3987e5", "defect": "#d95926",
        "ramp": ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"],
    },
}
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'
GOLDEN = 0.6180339887498949  # spreads vertical jitter evenly without randomness


def load_scores(model_dir, dataset_root, device, batch_size, rescore):
    cache = model_dir / "test_scores.json"
    if cache.is_file() and not rescore:
        return json.loads(cache.read_text(encoding="utf-8"))
    results, backbone = {}, None
    with torch.inference_mode():
        for path in sorted(model_dir.glob("*.pt")):
            model = PatchCore.load(path, device, backbone)
            backbone = model.backbone
            scores, labels = score(model, MVTecDataset(dataset_root, path.stem, "test"), batch_size)
            results[path.stem] = {"scores": scores.tolist(), "labels": labels.tolist()}
            print(f"{path.stem}: {len(labels)} test images", flush=True)
    write_json(cache, results)
    return results


class Svg:
    def __init__(self, width, height, theme):
        self.width, self.height, self.t, self.parts = width, height, theme, []

    def text(self, x, y, value, size=13, color="ink", anchor="start", weight=400):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{self.t[color]}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill, radius=0):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
                          f'height="{h:.1f}" rx="{radius}" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, color, width=1, dash=None):
        dashes = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{color}" stroke-width="{width}"{dashes}/>')

    def dot(self, x, y, r, fill):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
                          f'fill-opacity="0.75" stroke="{self.t["surface"]}" stroke-width="0.8"/>')

    def save(self, path, title):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" {FONT} role="img">'
            f'<title>{escape(title)}</title>'
            f'<rect width="100%" height="100%" fill="{self.t["surface"]}"/>'
            + "".join(self.parts) + "</svg>\n", encoding="utf-8")


def luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def readable_ink(fill):
    """Pick near-black or white text, whichever has more contrast on `fill`."""
    lum = luminance(fill)
    return "#0b0b0b" if (lum + 0.05) / (luminance("#0b0b0b") + 0.05) > 1.05 / (lum + 0.05) else "#ffffff"


def legend(svg, x, y, items):
    for label, color in items:
        if color == "threshold":
            svg.line(x, y - 4, x + 14, y - 4, svg.t["ink2"], 1.5, "4 3")
        else:
            svg.dot(x + 6, y - 4, 5, svg.t[color])
        svg.text(x + 20, y, label, 12, "ink2")
        x += 28 + 7 * len(label)


def draw_confusion(totals, theme, path):
    svg = Svg(450, 360, theme)
    count = sum(totals.values())
    svg.text(24, 32, "Confusion matrix, all 15 categories", 16, weight=600)
    svg.text(24, 52, f"{count:,} test images; thresholds chosen without the test set", 12, "ink2")
    rows = [("Normal", "true_negative", "false_positive"),
            ("Defective", "false_negative", "true_positive")]
    left, top, cell = 150, 110, 120
    svg.text(left + cell, top - 30, "Predicted", 12, "muted", "middle")
    svg.text(left + cell / 2, top - 10, "Normal", 13, "ink2", "middle")
    svg.text(left + 1.5 * cell, top - 10, "Defective", 13, "ink2", "middle")
    svg.text(36, top + cell + 4, "Actual", 12, "muted", "middle")
    for r, (label, *keys) in enumerate(rows):
        svg.text(left - 12, top + r * cell + cell / 2 + 5, label, 13, "ink2", "end")
        row_total = sum(totals[key] for key in keys)
        for c, key in enumerate(keys):
            share = totals[key] / row_total
            step = min(len(theme["ramp"]) - 1, int(share * len(theme["ramp"])))
            fill = theme["ramp"][step]
            ink = readable_ink(fill)
            x, y = left + c * cell, top + r * cell
            svg.rect(x + 1, y + 1, cell - 2, cell - 2, fill, 4)
            svg.parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2:.1f}" font-size="22" '
                             f'font-weight="600" fill="{ink}" text-anchor="middle">{totals[key]:,}</text>')
            svg.parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 22:.1f}" font-size="12" '
                             f'fill="{ink}" text-anchor="middle">{share:.1%} of {label.lower()}</text>')
    svg.save(path, "Confusion matrix over all test images")


def draw_rates(per_category, theme, path):
    order = sorted(per_category, key=lambda c: (-per_category[c]["recall"], c))
    row, top, label_w, panel_w, gap = 26, 96, 110, 300, 60
    svg = Svg(label_w + 2 * panel_w + gap + 70, top + row * len(order) + 40, theme)
    svg.text(24, 32, "Recall and false alarms per category", 16, weight=600)
    svg.text(24, 52, "Share of test images on the wrong or right side of each category's threshold",
             12, "ink2")
    panels = [("Defects caught (recall)", "recall", "defect"),
              ("Normal parts flagged (false positive rate)", "false_positive_rate", "normal")]
    for p, (title, key, color) in enumerate(panels):
        x0 = label_w + p * (panel_w + gap)
        svg.text(x0, top - 22, title, 13, "ink2", weight=600)
        for tick in (0, 0.25, 0.5, 0.75, 1):
            x = x0 + tick * panel_w
            svg.line(x, top - 8, x, top + row * len(order), theme["grid"])
            svg.text(x, top + row * len(order) + 18, f"{tick:.0%}", 11, "muted", "middle")
        for i, category in enumerate(order):
            value = per_category[category][key]
            y = top + i * row
            svg.rect(x0, y + 5, value * panel_w, row - 10, theme[color], 3)
            svg.text(x0 + value * panel_w + 6, y + row / 2 + 4, f"{value:.0%}", 11, "ink2")
        svg.line(x0, top - 8, x0, top + row * len(order), theme["axis"])
    for i, category in enumerate(order):
        svg.text(label_w - 12, top + i * row + row / 2 + 4, category, 12, "ink2", "end")
    svg.save(path, "Recall and false positive rate per category")


def draw_distributions(scores, thresholds, per_category, theme, path):
    categories, cols = sorted(scores), 3
    panel_w, panel_h, gap_x, gap_y, top = 290, 128, 24, 44, 96
    rows = -(-len(categories) // cols)
    svg = Svg(24 + cols * panel_w + (cols - 1) * gap_x + 24, top + rows * (panel_h + gap_y), theme)
    svg.text(24, 32, "Test score distributions and thresholds", 16, weight=600)
    svg.text(24, 52, "Each dot is one test image; dots right of the dashed line are flagged. "
             "Scales differ per category.", 12, "ink2")
    legend(svg, 24, 76, [("Normal", "normal"), ("Defective", "defect"), ("Threshold", "threshold")])
    for index, category in enumerate(categories):
        x0 = 24 + (index % cols) * (panel_w + gap_x)
        y0 = top + (index // cols) * (panel_h + gap_y)
        values, labels = scores[category]["scores"], scores[category]["labels"]
        threshold = thresholds[category]
        low, high = min(values + [threshold]), max(values + [threshold])
        pad = (high - low) * 0.05
        low, high = low - pad, high + pad

        def scale(value, x0=x0, low=low, high=high):
            return x0 + (value - low) / (high - low) * panel_w

        result = per_category[category]
        svg.text(x0, y0 + 14, category, 13, weight=600)
        svg.text(x0 + panel_w, y0 + 14,
                 f"recall {result['recall']:.0%} · false alarms {result['false_positive_rate']:.0%}",
                 11, "ink2", "end")
        bands = {0: (y0 + 30, "normal"), 1: (y0 + 72, "defect")}
        for label, (band_top, _) in bands.items():
            svg.rect(x0, band_top, panel_w, 34, theme["grid"], 3)
        for position, (value, label) in enumerate(zip(values, labels)):
            band_top, color = bands[label]
            offset = (position * GOLDEN) % 1
            svg.dot(scale(value), band_top + 5 + offset * 24, 2.6, theme[color])
        x = scale(threshold)
        svg.line(x, y0 + 24, x, y0 + 112, theme["ink2"], 1.5, "4 3")
        svg.text(x, y0 + 124, f"{threshold:.1f}", 10, "ink2", "middle")
    svg.save(path, "Test score distributions with per-category thresholds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec-ad"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("artifacts/border-exclusion/coreset-16384-n3-b2/patchcore"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--rescore", action="store_true", help="Ignore cached test_scores.json")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(Path.cwd().resolve()):
        parser.error("--output-dir must be inside the current directory")

    thresholds = {category: values["threshold"] for category, values in json.loads(
        (args.model_dir / "thresholds.json").read_text(encoding="utf-8"))["categories"].items()}
    scores = load_scores(args.model_dir, args.dataset_root, args.device,
                         args.batch_size, args.rescore)
    per_category = {
        category: confusion(torch.tensor(data["scores"]), torch.tensor(data["labels"]),
                            thresholds[category])
        for category, data in scores.items()
    }
    totals = {key: sum(result[key] for result in per_category.values()) for key in
              ("true_positive", "false_positive", "true_negative", "false_negative")}
    print(totals)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        draw_confusion(totals, theme, output_dir / f"threshold_confusion_{name}.svg")
        draw_rates(per_category, theme, output_dir / f"threshold_rates_{name}.svg")
        draw_distributions(scores, thresholds, per_category, theme,
                           output_dir / f"threshold_scores_{name}.svg")


if __name__ == "__main__":
    main()
