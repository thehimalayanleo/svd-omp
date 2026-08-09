"""Generate dependency-free SVG plots for the repository landing page."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures" / "latest_cross_model_summary.svg"
MODELS = ("Goodfire 67M", "Pythia-70M", "OPT-125M")
COLORS = ("#1f77b4", "#2ca02c", "#9467bd")


def load_metrics() -> tuple[list[float], list[float], list[float], list[str]]:
    foba = json.loads((ROOT / "results/svd_foba/broad_summary.json").read_text())
    foba_ratios = [
        float(row["geometric_mean_swd_error_over_foba"])
        for row in foba["models"]
    ]

    candidate_paths = (
        ROOT / "results/pruned_svd_foba/sealed_fresh_test.json",
        ROOT
        / "results/pruned_svd_foba/cross_model_EleutherAI__pythia-70m-deduped.json",
        ROOT / "results/pruned_svd_foba/cross_model_facebook__opt-125m.json",
    )
    candidates = []
    for path in candidate_paths:
        summary = json.loads(path.read_text())["summary"]
        candidates.append(summary[0] if isinstance(summary, list) else summary)
    candidate_ratios = [
        float(row["geometric_mean_swd_error_over_candidate"])
        for row in candidates
    ]
    reductions = [
        1.0 / float(row["mean_selector_read_fraction_of_full_foba"])
        for row in candidates
    ]
    wins = [f'{int(row["wins_over_swd"])}/{int(row["point_count"])}' for row in candidates]
    return foba_ratios, candidate_ratios, reductions, wins


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(
    *,
    x: int,
    title: str,
    subtitle: str,
    values: list[float],
    maximum: float,
    suffix: str,
    annotations: list[str] | None = None,
) -> list[str]:
    width = 350
    top = 108
    baseline = 390
    chart_height = baseline - top
    bar_width = 72
    gap = 32
    start = x + 38
    elements = [
        f'<text x="{x + width / 2:.0f}" y="42" text-anchor="middle" class="title">{escape(title)}</text>',
        f'<text x="{x + width / 2:.0f}" y="66" text-anchor="middle" class="subtitle">{escape(subtitle)}</text>',
        f'<line x1="{x + 25}" y1="{baseline}" x2="{x + width - 20}" y2="{baseline}" class="axis"/>',
    ]
    for tick in range(0, 5):
        value = maximum * tick / 4
        y = baseline - chart_height * tick / 4
        elements.extend(
            (
                f'<line x1="{x + 25}" y1="{y:.1f}" x2="{x + width - 20}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{x + 19}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>',
            )
        )
    for index, (model, value, color) in enumerate(zip(MODELS, values, COLORS, strict=True)):
        bar_x = start + index * (bar_width + gap)
        height = chart_height * value / maximum
        bar_y = baseline - height
        elements.extend(
            (
                f'<rect x="{bar_x}" y="{bar_y:.1f}" width="{bar_width}" height="{height:.1f}" rx="5" fill="{color}"/>',
                f'<text x="{bar_x + bar_width / 2:.1f}" y="{bar_y - 9:.1f}" text-anchor="middle" class="value">{value:.2f}{suffix}</text>',
                f'<text x="{bar_x + bar_width / 2:.1f}" y="{baseline + 22}" text-anchor="middle" class="model">{escape(model)}</text>',
            )
        )
        if annotations:
            elements.append(
                f'<text x="{bar_x + bar_width / 2:.1f}" y="{baseline + 42}" text-anchor="middle" class="annotation">{escape(annotations[index])}</text>'
            )
    return elements


def main() -> None:
    foba, candidate, reductions, wins = load_metrics()
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="470" viewBox="0 0 1200 470" role="img" aria-labelledby="plot-title plot-desc">',
        '<title id="plot-title">Cross-model fidelity and selector-cost summary</title>',
        '<desc id="plot-desc">SVD-FoBa and CP-SVD error advantages over SWD, plus CP-SVD selector reduction relative to SVD-FoBa.</desc>',
        '<style>',
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#17212b}',
        '.title{font-size:18px;font-weight:700}.subtitle{font-size:12px;fill:#536273}',
        '.axis{stroke:#657586;stroke-width:1.2}.grid{stroke:#dfe5eb;stroke-width:1}',
        '.tick{font-size:10px;fill:#657586}.value{font-size:13px;font-weight:700}',
        '.model{font-size:10px}.annotation{font-size:10px;fill:#536273}',
        '</style>',
        '<rect width="1200" height="470" fill="#ffffff"/>',
    ]
    elements.extend(
        panel(
            x=15,
            title="SVD-FoBa fidelity",
            subtitle="SWD error / SVD-FoBa error, higher is better",
            values=foba,
            maximum=2.5,
            suffix="×",
        )
    )
    elements.extend(
        panel(
            x=420,
            title="CP-SVD fidelity",
            subtitle="SWD error / CP-SVD error, higher is better",
            values=candidate,
            maximum=2.5,
            suffix="×",
            annotations=wins,
        )
    )
    elements.extend(
        panel(
            x=825,
            title="CP-SVD selector reduction",
            subtitle="SVD-FoBa scored width / CP-SVD scored width",
            values=reductions,
            maximum=10.0,
            suffix="×",
        )
    )
    elements.extend(
        (
            '<text x="600" y="452" text-anchor="middle" class="subtitle">Frozen WikiText-2 evaluations; 24 matrices and 10 selected-unit widths per model.</text>',
            '</svg>',
        )
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(elements) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
