"""Generate the README figures (standalone SVG, light and dark mode).

    python3 docs/figures/make_readme_figures.py
"""

from pathlib import Path

OUT = Path(__file__).parent

STYLE = """<style>
text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
.t{font-size:14px;fill:#2C2C2A}.ts{font-size:12px;fill:#5F5E5A}.th{font-size:14px;font-weight:500;fill:#2C2C2A}
.f{font-size:13px;fill:#2C2C2A;font-family:"Cambria Math","STIX Two Math","Times New Roman",serif;font-style:italic}
.arr{stroke:#888780;stroke-width:1.2;fill:none}.mk{stroke:#888780}.dash{stroke:#888780}
.bar-keep{fill:#EF9F27}.bar-drop{fill:#B4B2A9}
.c-gray,.c-gray>rect{fill:#F1EFE8;stroke:#5F5E5A}.c-gray>text{fill:#444441}
.c-purple,.c-purple>rect{fill:#EEEDFE;stroke:#534AB7}.c-purple>text{fill:#3C3489}
.c-teal,.c-teal>rect{fill:#E1F5EE;stroke:#0F6E56}.c-teal>text{fill:#085041}
.c-amber,.c-amber>rect{fill:#FAEEDA;stroke:#854F0B}.c-amber>text{fill:#633806}
@media (prefers-color-scheme:dark){
.t,.th,.f{fill:#D3D1C7}.ts{fill:#B4B2A9}.arr,.mk,.dash{stroke:#B4B2A9}.bar-drop{fill:#5F5E5A}
.c-gray,.c-gray>rect{fill:#444441;stroke:#B4B2A9}.c-gray>text{fill:#F1EFE8}
.c-purple,.c-purple>rect{fill:#3C3489;stroke:#AFA9EC}.c-purple>text{fill:#CECBF6}
.c-teal,.c-teal>rect{fill:#085041;stroke:#5DCAA5}.c-teal>text{fill:#9FE1CB}
.c-amber,.c-amber>rect{fill:#633806;stroke:#EF9F27}.c-amber>text{fill:#FAC775}}
</style>"""
DEFS = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse"><path class="mk" d="M2 1L8 5L2 9" '
        'fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fm(s: str, size: float = 13.0) -> str:
    """Tiny formula markup: x_{sub}, x^{sup}, x_c, x^T become shifted tspans."""
    out, buf, cur, i = [], "", 0.0, 0

    def flush():
        nonlocal buf, cur
        if buf:
            out.append(f'<tspan dy="{-cur:.1f}">{esc(buf)}</tspan>' if cur else esc(buf))
            cur, buf = 0.0, ""

    while i < len(s):
        ch = s[i]
        if ch in "_^" and i + 1 < len(s):
            flush()
            if s[i + 1] == "{":
                j = s.index("}", i)
                arg, i = s[i + 2:j], j + 1
            else:
                arg, i = s[i + 1], i + 2
            target = 0.3 * size if ch == "_" else -0.4 * size
            out.append(f'<tspan dy="{target - cur:.1f}" font-size="{0.72 * size:.1f}">{esc(arg)}</tspan>')
            cur = target
        else:
            buf += ch
            i += 1
    flush()
    return "".join(out)


def svg(h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{h}" viewBox="0 0 680 {h}" '
            f'role="img"><title>{title}</title><desc>{desc}</desc>{STYLE}{DEFS}{body}</svg>\n')


def box(cls, x, y, w, h, title, sub=None, formula=False):
    s = f'<g class="{cls}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" stroke-width="0.5"/>'
    cx = x + w / 2
    if sub is None:
        s += f'<text class="th" x="{cx}" y="{y + h / 2 + 5}" text-anchor="middle">{title}</text>'
    else:
        s += f'<text class="th" x="{cx}" y="{y + 23}" text-anchor="middle">{title}</text>'
        if formula:
            s += f'<text class="f" x="{cx}" y="{y + 43}" text-anchor="middle">{fm(sub, 12)}</text>'
        else:
            s += f'<text class="ts" x="{cx}" y="{y + 42}" text-anchor="middle">{esc(sub)}</text>'
    return s + "</g>"


def arrow(x1, y1, x2, y2):
    return f'<line class="arr" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)"/>'


def pipeline():
    b = '<text class="th" x="40" y="30">Once per weight matrix (offline, no training)</text>'
    b += box("c-gray", 40, 62, 150, 56, "Calibration text", "about 4k tokens, no labels")
    b += box("c-purple", 220, 44, 190, 56, "Input statistics", "G = E[φφ^T]", True)
    b += box("c-purple", 220, 112, 190, 56, "Output sensitivity", "M = E[gg^T]", True)
    b += box("c-purple", 440, 78, 200, 56, "Weighted SVD", "L_{out}^T W L_{in} = P S Q^T", True)
    b += arrow(190, 82, 218, 72) + arrow(190, 98, 218, 140)
    b += arrow(410, 72, 438, 98) + arrow(410, 140, 438, 114)
    b += '<text class="ts" x="440" y="154">atoms that sum to W exactly</text>'
    b += '<line class="dash" x1="40" y1="186" x2="640" y2="186" stroke-width="0.5" stroke-dasharray="4 4"/>'
    b += '<text class="th" x="40" y="214">Every token (online)</text>'
    b += box("c-gray", 40, 232, 120, 56, "Layer input", "φ", True)
    b += box("c-amber", 180, 232, 160, 56, "Score every atom", "a_c = s_c read_c^T φ", True)
    b += box("c-amber", 360, 232, 130, 56, "Keep top-k", "largest |a_c|", True)
    b += box("c-gray", 510, 232, 130, 56, "Output", "Σ_{kept} a_c write_c", True)
    b += arrow(160, 260, 178, 260) + arrow(340, 260, 358, 260) + arrow(490, 260, 508, 260)
    b += arrow(540, 162, 300, 230)
    b += '<text class="ts" x="40" y="318">Minimises ½ eᵀMe, where e is the change in the layer output: the second-order</text>'
    b += '<text class="ts" x="40" y="336">change in the model\'s next-token KL. Top-k is the exact optimum for every token.</text>'
    return svg(352, "How causal-metric SVD-OMP works",
               "Offline: calibration text gives input and output statistics, a weighted SVD gives the atoms. "
               "Online: score every atom on the token's input, keep the k largest, sum them.", b)


def sort_figure():
    vals = [0.9, 3.1, 0.4, 2.2, 0.3, 1.4, 0.2, 2.8, 0.6, 0.25, 1.0, 0.5]
    keep = set(sorted(range(len(vals)), key=lambda c: -vals[c])[:3])
    b = '<text class="th" x="40" y="30">One token: keep the k = 3 largest coefficients</text>'
    base, top, x0, w, gap = 250, 90, 50, 24, 8
    mx = max(vals)
    for c, v in enumerate(vals):
        h = (base - top) * v / mx
        x = x0 + c * (w + gap)
        cls = "bar-keep" if c in keep else "bar-drop"
        b += f'<rect class="{cls}" x="{x}" y="{base - h:.1f}" width="{w}" height="{h:.1f}" rx="2"/>'
        b += f'<text class="f" x="{x + w / 2}" y="{base + 18}" text-anchor="middle">{fm(f"a_{{{c + 1}}}", 12)}</text>'
    b += f'<line class="dash" x1="{x0 - 6}" y1="{base}" x2="{x0 + len(vals) * (w + gap)}" y2="{base}" stroke-width="0.8"/>'
    b += f'<text class="ts" x="44" y="66">{fm("|a_c| on this token, atoms in their fixed order", 12)}</text>'
    lx = 450
    b += f'<rect class="bar-keep" x="{lx}" y="96" width="14" height="14" rx="2"/>'
    b += f'<text class="ts" x="{lx + 22}" y="107">kept: the 3 largest</text>'
    b += f'<rect class="bar-drop" x="{lx}" y="122" width="14" height="14" rx="2"/>'
    b += f'<text class="ts" x="{lx + 22}" y="133">dropped: everything else</text>'
    b += f'<text class="ts" x="{lx}" y="168">Error left = sum of the</text>'
    b += f'<text class="ts" x="{lx}" y="186">dropped bars, squared.</text>'
    b += f'<text class="ts" x="{lx}" y="214">No other 3 atoms, and no</text>'
    b += f'<text class="ts" x="{lx}" y="232">re-weighting of the kept ones,</text>'
    b += f'<text class="ts" x="{lx}" y="250">can do better (Lean-checked).</text>'
    b += f'<text class="f" x="40" y="306">{fm("a_c(φ) = s_c · read_c^T φ")}</text>'
    b += f'<text class="f" x="300" y="306">{fm("error(S) = Σ_{c ∉ S} a_c²")}</text>'
    return svg(324, "Selection is a sort",
               "Bar chart of twelve atom coefficients on one token; the three largest are kept and the error "
               "equals the sum of squares of the dropped ones.", b)


def losses_figure():
    cards = [
        ("c-purple", "plain SVD-OMP", "weights: none", "‖(W − Ŵ)φ‖²", None,
         "this layer's output, every direction equal"),
        ("c-purple", "whitened SVD-OMP", "weights: inputs, G = E[φφ^T]", "E_φ ‖(W − Ŵ)φ‖²", None,
         "this layer's output, on the inputs it really sees"),
        ("c-purple", "causal-metric SVD-OMP", "weights: inputs G and outputs M = E[gg^T]",
         "E_φ [e^T M e] ≈ 2 ΔKL", None, "the model's next-token prediction"),
        ("c-teal", "VPD (trained)", "learned components and importance net",
         "‖W − Σ_c u_c v_c^T‖²_F + KL(f ‖ f_{masked})", "+ λ Σ_c g_c^p",
         "masked model still matches; few pieces needed"),
    ]
    b = '<text class="th" x="40" y="30">What each method minimises</text>'
    for i, (cls, name, weights, loss, loss2, words) in enumerate(cards):
        x = 40 + (i % 2) * 305
        y = 46 + (i // 2) * 150
        b += f'<g class="{cls}"><rect x="{x}" y="{y}" width="295" height="136" rx="6" stroke-width="0.5"/>'
        b += f'<text class="th" x="{x + 14}" y="{y + 24}">{name}</text></g>'
        b += f'<text class="f" x="{x + 14}" y="{y + 48}" style="font-size:12px">{fm(weights, 12)}</text>'
        b += f'<text class="f" x="{x + 14}" y="{y + 78}" style="font-size:15px">{fm(loss, 15)}</text>'
        if loss2:
            b += f'<text class="f" x="{x + 14}" y="{y + 98}" style="font-size:15px">{fm(loss2, 15)}</text>'
        b += f'<text class="ts" x="{x + 14}" y="{y + 122}">{esc(words)}</text>'
    b += ('<text class="ts" x="40" y="360">SVD family: closed form, top-k exact per token. '
          'VPD: trained against random and adversarial masks.</text>')
    return svg(376, "What each method minimises",
               "Four cards: the loss minimised by plain, whitened and causal-metric SVD-OMP, and VPD's training loss.",
               b)


if __name__ == "__main__":
    (OUT / "svd_omp_pipeline.svg").write_text(pipeline())
    (OUT / "selection_is_a_sort.svg").write_text(sort_figure())
    (OUT / "what_each_method_minimises.svg").write_text(losses_figure())
    print("wrote 3 figures")
