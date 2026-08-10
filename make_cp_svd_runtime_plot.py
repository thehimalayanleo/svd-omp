"""Generate the dependency-free SVG for the direct CP-SVD runtime gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "cp_svd_direct" / "summary.json"
OUTPUT = ROOT / "figures" / "cp_svd_direct_runtime.svg"


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    runs = summary["runs"]
    maximum = 80.0
    baseline = 385.0
    chart_height = 280.0
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500" role="img" aria-labelledby="title desc">',
        '<title id="title">Direct CP-SVD end-to-end T4 latency</title>',
        '<desc id="desc">Two synchronized runs comparing dense and direct CP-SVD latency while preserving identical CP-SVD quality.</desc>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#17212b}.title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#536273}.axis{stroke:#657586;stroke-width:1.2}.grid{stroke:#dfe5eb}.tick{font-size:11px;fill:#657586}.label{font-size:13px;font-weight:600}.value{font-size:14px;font-weight:700}.callout{font-size:14px;font-weight:600;fill:#174d2c}</style>',
        '<rect width="900" height="500" fill="#ffffff"/>',
        '<text x="450" y="38" text-anchor="middle" class="title">Direct CP-SVD removes the dense matmul</text>',
        '<text x="450" y="62" text-anchor="middle" class="subtitle">Goodfire 67M, all 24 matrices replaced, Tesla T4, input 16 × 128, lower is better</text>',
    ]
    for tick in range(0, 5):
        value = maximum * tick / 4
        y = baseline - chart_height * tick / 4
        elements.extend(
            (
                f'<line x1="80" y1="{y:.1f}" x2="650" y2="{y:.1f}" class="grid"/>',
                f'<text x="70" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f} ms</text>',
            )
        )
    elements.append('<line x1="80" y1="385" x2="650" y2="385" class="axis"/>')
    colors = {"dense": "#657586", "candidate": "#2ca02c"}
    for run_index, run in enumerate(runs):
        group_x = 145 + run_index * 285
        for method_index, (method, key) in enumerate(
            (("dense", "dense_milliseconds_median"), ("candidate", "candidate_milliseconds_median"))
        ):
            value = float(run[key])
            height = chart_height * value / maximum
            x = group_x + method_index * 92
            y = baseline - height
            label = "Dense" if method == "dense" else "CP-SVD"
            elements.extend(
                (
                    f'<rect x="{x}" y="{y:.1f}" width="66" height="{height:.1f}" rx="5" fill="{colors[method]}"/>',
                    f'<text x="{x + 33}" y="{y - 9:.1f}" text-anchor="middle" class="value">{value:.2f}</text>',
                    f'<text x="{x + 33}" y="407" text-anchor="middle" class="label">{label}</text>',
                )
            )
        run_label = "Discovery" if run["run"] == "discovery" else "Fresh confirmation"
        elements.append(
            f'<text x="{group_x + 79}" y="438" text-anchor="middle" class="subtitle">{run_label}: {run["dense_over_candidate_speedup"]:.3f}× faster</text>'
        )
    elements.extend(
        (
            '<rect x="690" y="125" width="175" height="170" rx="12" fill="#eef8f1" stroke="#b7d8c1"/>',
            '<text x="777" y="164" text-anchor="middle" class="callout">Confirmed minimum</text>',
            f'<text x="777" y="202" text-anchor="middle" class="title">{summary["minimum_dense_over_candidate_speedup"]:.3f}×</text>',
            '<text x="777" y="230" text-anchor="middle" class="subtitle">end-to-end speedup</text>',
            f'<text x="777" y="263" text-anchor="middle" class="callout">{summary["replaced_dense_weight_elements_over_candidate_factors"]:.2f}× fewer factors</text>',
            '<text x="450" y="480" text-anchor="middle" class="subtitle">Cross-entropy, KL, and logit MSE exactly match the frozen CP-SVD quality artifact.</text>',
            '</svg>',
        )
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(elements) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
