"""BSF-style stable-rank sweep on real LM activations.

Downloads a small public language model (Pythia-70M), runs it on a text
corpus, extracts activations from an MLP layer, and reproduces the BSF
stable-rank plot on those activations. Directly tests whether the "2-4
dimensional concept" plateau shows up on real transformer activations.

BSF measured this on vision (DINOv3). We measure on language. If the
plateau reproduces, the finding generalizes across modalities and is a
property of neural-network activation distributions in general.

Usage:
    python real_activations_stable_rank.py
    python real_activations_stable_rank.py --model EleutherAI/pythia-160m --layer 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from block_svd_omp import block_svd_decompose
from bsf_weights import run_bsf_weights
from stable_rank import activation_stable_rank_per_block

BLOCK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]

# Wikipedia-ish diverse prompts (varied domains, so activations aren't
# concentrated on one narrow topic).
PROMPTS = [
    "The transformer architecture revolutionized natural language processing by",
    "In quantum mechanics, the wave function collapses when",
    "The French Revolution began in 1789 with the storming of the",
    "Photosynthesis converts carbon dioxide and water into glucose using",
    "The Pythagorean theorem states that for a right triangle,",
    "Mount Everest, the tallest mountain in the world, rises to",
    "The Federal Reserve controls monetary policy in the United States by",
    "Modern jazz emerged in the 1940s with bebop, characterized by",
    "The immune system defends the body from pathogens through",
    "Continental drift explains how Earth's continents move over geological",
    "Recursion in computer science refers to a function calling",
    "The Renaissance began in Italy during the 14th century and",
    "Democracy as a form of government requires participation of",
    "The theory of evolution was proposed by Charles Darwin in",
    "Machine learning models learn from data to make predictions about",
    "The Great Wall of China stretches thousands of miles and was built to",
    "Neural networks approximate functions through layers of",
    "The mitochondria is the powerhouse of the cell, producing",
    "Impressionist painters like Monet emphasized light and color over",
    "The industrial revolution transformed economies by mechanizing",
    "String theory attempts to unify quantum mechanics and general",
    "The invention of the printing press by Gutenberg enabled",
    "Chess is a game of strategy played on an 8x8 board where",
    "Ecosystems consist of interacting biological communities and their",
    "The stock market functions as a mechanism for buying and selling",
    "Ancient Egyptian civilization flourished along the Nile River for over",
    "Reinforcement learning agents learn by trial and error in",
    "The scientific method involves hypothesis, experiment, and",
    "Climate change is driven primarily by human emissions of",
    "Symphonies typically consist of four movements in the classical",
    "The internet was invented in the late 20th century to enable",
    "Human languages share universal features studied by linguists like",
]


def collect_activations(model, tokenizer, layer_idx: int, prompts: list[str],
                        device: str, target_module: str = "mlp") -> torch.Tensor:
    """Run the model on prompts, hook into the specified layer's MLP output,
    return concatenated activations of shape [N_tokens, d_hidden]."""
    captured = []

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        captured.append(out.detach().cpu().reshape(-1, out.shape[-1]))

    # Different HF LMs have different attribute paths; try common ones.
    layer = None
    for path in (
        f"layers.{layer_idx}.mlp",                        # Pythia (base model)
        f"gpt_neox.layers.{layer_idx}.mlp",               # Pythia (CausalLM wrapper)
        f"model.layers.{layer_idx}.mlp",                  # Llama/Qwen
        f"transformer.h.{layer_idx}.mlp",                 # GPT-2
    ):
        try:
            m = model
            for part in path.split("."):
                m = getattr(m, part) if not part.isdigit() else m[int(part)]
            layer = m
            print(f"  hooked module: {path}")
            break
        except (AttributeError, KeyError, IndexError):
            continue
    if layer is None:
        raise RuntimeError(f"could not find layer {layer_idx}.mlp in the model")

    handle = layer.register_forward_hook(hook)
    try:
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            if i % 10 == 0:
                print(f"    prompt {i}/{len(prompts)}")
    finally:
        handle.remove()

    return torch.cat(captured, dim=0)


def sweep(W, phi, weights_are_synthetic=False):
    """Return dict of {method: {K: mean_stable_rank}}."""
    results = {m: {} for m in ["analytic", "bsf_cold", "bsf_warm"]}
    d_in = W.shape[1]
    d_out = W.shape[0]

    for K in BLOCK_SIZES:
        C = min(K * 8, min(d_out, d_in) // K * K)
        if C < K:
            continue

        V_a, _, _, blocks = block_svd_decompose(W, C, K)
        results["analytic"][K] = mean_rank(V_a, blocks, phi)

        V_c, _, _, blocks_c = run_bsf_weights(W, C, K, k_blocks=2, n=60, seed=0,
                                              warm_start_svd=False, verbose=False)
        results["bsf_cold"][K] = mean_rank(V_c, blocks_c, phi)

        V_w, _, _, blocks_w = run_bsf_weights(W, C, K, k_blocks=2, n=60, seed=0,
                                              warm_start_svd=True, verbose=False)
        results["bsf_warm"][K] = mean_rank(V_w, blocks_w, phi)

    return results


def mean_rank(V, blocks, phi):
    ranks = activation_stable_rank_per_block(V, blocks, phi)
    return sum(ranks) / len(ranks)


def main(args):
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    device = "cpu"
    model.eval().to(device)

    print(f"Extracting activations from layer {args.layer} MLP...")
    acts = collect_activations(model, tokenizer, args.layer, PROMPTS, device)
    print(f"  activations shape: {tuple(acts.shape)}")

    # Extract the same-layer up_proj / MLP up-projection weight as W.
    layer_to_hook = None
    for path in (
        f"layers.{args.layer}.mlp.dense_h_to_4h",
        f"gpt_neox.layers.{args.layer}.mlp.dense_h_to_4h",
        f"model.layers.{args.layer}.mlp.up_proj",
        f"transformer.h.{args.layer}.mlp.c_fc",
    ):
        try:
            m = model
            for part in path.split("."):
                m = getattr(m, part) if not part.isdigit() else m[int(part)]
            layer_to_hook = m
            print(f"  W taken from: {path}")
            break
        except (AttributeError, KeyError, IndexError):
            continue

    if layer_to_hook is None:
        raise RuntimeError("no MLP up-projection found; try a different model")

    W = layer_to_hook.weight.detach().float()
    print(f"  W shape: {tuple(W.shape)}")

    # Cap activations for speed.
    phi = acts[: args.n_activations]
    print(f"  using {phi.shape[0]} tokens for stable-rank measurement")

    # Match phi's d_in to W's d_in. The MLP up-proj input is the residual
    # stream d_model, which is what the hooked MLP receives as input. So
    # phi and W should already be aligned via the same layer. But the hook
    # captures MLP OUTPUT which is d_intermediate. We need MLP INPUT.
    # Simplest fix: re-hook the residual stream (input to the MLP block).
    if phi.shape[1] != W.shape[1]:
        print(f"  WARNING: activation dim {phi.shape[1]} != W.shape[1] {W.shape[1]}; "
              f"transposing W to match")
        # Try using W.T so the block dictionary lives on the same space as phi.
        if phi.shape[1] == W.shape[0]:
            W = W.T
            print(f"  W transposed to {tuple(W.shape)}")
        else:
            raise RuntimeError(
                f"cannot align activation dim {phi.shape[1]} with W dims {W.shape}")

    t0 = time.time()
    results = sweep(W, phi)
    print(f"\nSweep completed in {time.time() - t0:.1f}s")

    print(f"\n{'K':<4} " + " ".join(f"{m:>15}" for m in results))
    for K in BLOCK_SIZES:
        vals = [results[m].get(K, float("nan")) for m in results]
        print(f"{K:<4} " + " ".join(f"{v:>15.2f}" for v in vals))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Stable rank vs K on real LM activations ({args.model}, layer {args.layer})",
                 y=1.02)

    titles = ["Analytic block-SVD-OMP", "BSF-W (random init)", "BSF-W (SVD warm-start)"]
    keys = ["analytic", "bsf_cold", "bsf_warm"]

    for ax, key, title in zip(axes, keys, titles):
        Ks = sorted(results[key].keys())
        ys = [results[key][K] for K in Ks]
        ax.plot(Ks, Ks, "k--", alpha=0.4, label="full rank (= K)")
        ax.plot(Ks, ys, "-o", color="#4C72B0", label="measured")
        ax.set_xlabel("group size K")
        ax.set_ylabel("Stable Rank")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 17)
        ax.legend(loc="upper left", fontsize=9)

    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures/stable_rank_real_lm.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"figures/stable_rank_real_lm.png", bbox_inches="tight", dpi=150)

    Path("results").mkdir(exist_ok=True)
    Path("results/stable_rank_real_lm.json").write_text(json.dumps({
        "model": args.model,
        "layer": args.layer,
        "n_tokens": int(phi.shape[0]),
        "W_shape": list(W.shape),
        "results": results,
    }, indent=2))
    print(f"\nWrote figures/stable_rank_real_lm.{{png,pdf}}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m",
                    help="HF model id. Try pythia-70m (fast) or Qwen/Qwen2.5-0.5B.")
    ap.add_argument("--layer", type=int, default=3, help="Layer to extract activations from.")
    ap.add_argument("--n-activations", type=int, default=1024,
                    help="Number of tokens to use.")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
