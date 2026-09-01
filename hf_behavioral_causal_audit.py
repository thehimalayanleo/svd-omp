"""Held-out behavioral causality audit for post-training SVD-OMP atoms.

The default model pair is Qwen3-1.7B-Base and Qwen3-1.7B.  For each selected
linear module we decompose ``W_post - W_base`` and test two complementary
interventions:

* insertion: add selected delta atoms while running the base model;
* ablation: remove the same kind of atoms while running the post-trained model.

Behavior is measured as a paired-completion log-probability margin supplied by
the dataset, never by reconstruction error.  The primary comparison is
input-conditioned SVD-OMP against a token-wise norm-matched random component
control.  Run the discovery and test splits separately and do not change the
module list, k, or scoring completions after seeing test results.

Dataset JSONL schema::

    {"id":"h1", "split":"test", "group":"target", "prompt":"...",
     "positive_completion":"I cannot help with that request.",
     "negative_completion":"Here are the steps:"}

``group`` must be ``target`` or ``control``.  Larger margins mean greater
preference for ``positive_completion``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from behavioral_causal_audit import (
    LinearComponentIntervention,
    LinearDeltaIntervention,
    SVDAtoms,
    atoms_from_lowrank,
    summarize_causal_audit,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_POST_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_MODULES = ",".join(
    f"model.layers.{layer}.self_attn.o_proj" for layer in range(28)
)


def load_hf_model(
    model_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    revision: str | None = None,
) -> nn.Module:
    """Load either a text-only or multimodal causal language model."""

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=False
    )
    architectures = getattr(config, "architectures", None) or []
    is_multimodal = hasattr(config, "text_config") and any(
        "ConditionalGeneration" in name for name in architectures
    )
    if is_multimodal:
        from transformers import AutoModelForImageTextToText
        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    return model_class.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def load_hf_tokenizer(model_id: str, *, revision: str | None = None) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": False,
    }
    if "mistral" in model_id.lower():
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def model_vocab_size(model: nn.Module) -> int:
    config = model.config
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
        return int(config.text_config.vocab_size)
    raise ValueError("model configuration does not expose a vocabulary size")


def override_completion_pair(
    examples: list[dict[str, str]],
    positive: str | None,
    negative: str | None,
) -> list[dict[str, str]]:
    if positive is None and negative is None:
        return examples
    if not positive or not negative:
        raise ValueError("positive and negative completion overrides must be provided together")
    return [
        {**row, "positive_completion": positive, "negative_completion": negative}
        for row in examples
    ]


def load_examples(path: str | Path, split: str) -> tuple[list[dict[str, str]], str]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    examples: list[dict[str, str]] = []
    required = {
        "id",
        "split",
        "group",
        "prompt",
        "positive_completion",
        "negative_completion",
    }
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = required - row.keys()
        if missing:
            raise ValueError(f"line {line_number} is missing {sorted(missing)}")
        if row["group"] not in {"target", "control"}:
            raise ValueError(f"line {line_number} has invalid group {row['group']!r}")
        if row["split"] == split:
            examples.append({key: str(row[key]) for key in required})
    if not examples:
        raise ValueError(f"dataset contains no examples for split {split!r}")
    groups = {row["group"] for row in examples}
    if groups != {"target", "control"}:
        raise ValueError(f"split {split!r} must contain target and control examples")
    ids = [row["id"] for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"split {split!r} contains duplicate ids")
    return examples, digest


def resolve_module(model: nn.Module, name: str) -> nn.Linear:
    modules = dict(model.named_modules())
    if name not in modules:
        nearby = [candidate for candidate in modules if candidate.endswith(name)]
        hint = f"; suffix matches: {nearby[:5]}" if nearby else ""
        raise KeyError(f"module {name!r} does not exist{hint}")
    module = modules[name]
    if not isinstance(module, nn.Linear):
        raise TypeError(f"module {name!r} is {type(module).__name__}, not nn.Linear")
    return module


def build_delta_atoms(
    base_model: nn.Module,
    post_model: nn.Module,
    module_names: Iterable[str],
    *,
    n_components: int,
    oversample: int,
    niter: int,
    atom_dtype: torch.dtype,
    seed: int,
) -> tuple[dict[str, SVDAtoms], dict[str, Tensor], dict[str, dict[str, float]]]:
    atoms: dict[str, SVDAtoms] = {}
    deltas: dict[str, Tensor] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for module_index, name in enumerate(module_names):
        base_module = resolve_module(base_model, name)
        post_module = resolve_module(post_model, name)
        if base_module.weight.shape != post_module.weight.shape:
            raise ValueError(f"base/post shapes differ for {name}")
        delta = post_module.weight.detach().float() - base_module.weight.detach().float()
        delta_norm = delta.norm().item()
        t0 = time.time()
        dictionary = atoms_from_lowrank(
            delta,
            n_components=n_components,
            oversample=oversample,
            niter=niter,
            seed=seed + module_index,
        )
        captured_frobenius = dictionary.S.float().square().sum().sqrt().item()
        atoms[name] = dictionary.to(dtype=atom_dtype)
        deltas[name] = delta.to(dtype=atom_dtype)
        diagnostics[name] = {
            "delta_frobenius": delta_norm,
            "captured_frobenius": captured_frobenius,
            "captured_fraction": captured_frobenius / max(delta_norm, 1e-12),
            "svd_seconds": time.time() - t0,
            "svd_seed": seed + module_index,
        }
        del delta, dictionary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return atoms, deltas, diagnostics


def format_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def completion_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    completion: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt and completion must each produce at least one token")
    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    start = len(prompt_ids) - 1
    stop = start + len(completion_ids)
    continuation_logits = logits[:, start:stop, :].float()
    labels = torch.tensor(completion_ids, device=device).view(1, -1, 1)
    token_logprobs = continuation_logits.log_softmax(dim=-1).gather(-1, labels).squeeze(-1)
    return token_logprobs.mean().item()


def score_examples(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    *,
    device: torch.device,
    use_chat_template: bool,
) -> list[float]:
    margins = []
    for row in examples:
        prompt = format_prompt(tokenizer, row["prompt"], use_chat_template)
        positive = completion_logprob(
            model, tokenizer, prompt, row["positive_completion"], device
        )
        negative = completion_logprob(
            model, tokenizer, prompt, row["negative_completion"], device
        )
        margins.append(positive - negative)
    return margins


def score_with_interventions(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    atoms: dict[str, SVDAtoms],
    *,
    device: torch.device,
    use_chat_template: bool,
    policy: str,
    k: int,
    mode: str,
    seed: int,
    pool_factor: int,
) -> tuple[list[float], dict[str, float]]:
    interventions: list[LinearComponentIntervention] = []
    with ExitStack() as stack:
        for offset, (name, dictionary) in enumerate(atoms.items()):
            intervention = LinearComponentIntervention(
                resolve_module(model, name),
                dictionary,
                policy=policy,
                k=k,
                mode=mode,
                seed=seed + offset * 1009,
                pool_factor=pool_factor,
                match_reference_norm=True,
            )
            interventions.append(stack.enter_context(intervention))
        margins = score_examples(
            model,
            tokenizer,
            examples,
            device=device,
            use_chat_template=use_chat_template,
        )
    rms = {
        name: intervention.perturbation_rms
        for name, intervention in zip(atoms, interventions)
    }
    return margins, rms


def score_with_paired_dose_interventions(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    atoms: dict[str, SVDAtoms],
    *,
    device: torch.device,
    use_chat_template: bool,
    k: int,
    mode: str,
    seed: int,
    pool_factor: int,
) -> tuple[list[float], dict[str, float], list[float], dict[str, float]]:
    """Score OMP and random with the exact same per-example, per-token dose.

    The OMP pass records the realized perturbation norm at every module and
    hook call. The random pass replays those norms while retaining its own
    randomly selected atom directions. This is necessary for simultaneous
    multi-layer interventions, where earlier interventions otherwise move OMP
    and random onto different downstream activation trajectories.
    """

    candidate_interventions: list[LinearComponentIntervention] = []
    with ExitStack() as stack:
        for offset, (name, dictionary) in enumerate(atoms.items()):
            intervention = LinearComponentIntervention(
                resolve_module(model, name),
                dictionary,
                policy="input_omp",
                k=k,
                mode=mode,
                seed=seed + offset * 1009,
                pool_factor=pool_factor,
                match_reference_norm=True,
                record_token_norms=True,
            )
            candidate_interventions.append(stack.enter_context(intervention))
        candidate_margin = score_examples(
            model,
            tokenizer,
            examples,
            device=device,
            use_chat_template=use_chat_template,
        )
    candidate_rms = {
        name: intervention.perturbation_rms
        for name, intervention in zip(atoms, candidate_interventions)
    }
    norm_traces = {
        name: intervention.token_norm_trace
        for name, intervention in zip(atoms, candidate_interventions)
    }

    random_interventions: list[LinearComponentIntervention] = []
    with ExitStack() as stack:
        for offset, (name, dictionary) in enumerate(atoms.items()):
            intervention = LinearComponentIntervention(
                resolve_module(model, name),
                dictionary,
                policy="matched_random",
                k=k,
                mode=mode,
                seed=seed + offset * 1009,
                pool_factor=pool_factor,
                match_reference_norm=False,
                replay_token_norms=norm_traces[name],
            )
            random_interventions.append(stack.enter_context(intervention))
        random_margin = score_examples(
            model,
            tokenizer,
            examples,
            device=device,
            use_chat_template=use_chat_template,
        )
    random_rms = {
        name: intervention.perturbation_rms
        for name, intervention in zip(atoms, random_interventions)
    }
    return candidate_margin, candidate_rms, random_margin, random_rms


def score_with_full_deltas(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    deltas: dict[str, Tensor],
    *,
    device: torch.device,
    use_chat_template: bool,
    mode: str,
) -> tuple[list[float], dict[str, float]]:
    interventions: list[LinearDeltaIntervention] = []
    with ExitStack() as stack:
        for name, delta in deltas.items():
            intervention = LinearDeltaIntervention(
                resolve_module(model, name), delta, mode=mode
            )
            interventions.append(stack.enter_context(intervention))
        margins = score_examples(
            model,
            tokenizer,
            examples,
            device=device,
            use_chat_template=use_chat_template,
        )
    rms = {
        name: intervention.perturbation_rms
        for name, intervention in zip(deltas, interventions)
    }
    return margins, rms


def _tensor(values: list[float]) -> Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _summary_dict(summary: Any) -> dict[str, Any]:
    result = asdict(summary)
    result["passes"] = summary.passes
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install transformers>=4.51 and accelerate for the real-model audit"
        ) from exc

    examples, dataset_sha256 = load_examples(args.dataset, args.split)
    examples = override_completion_pair(
        examples,
        getattr(args, "score_positive", None),
        getattr(args, "score_negative", None),
    )
    module_names = [name.strip() for name in args.modules.split(",") if name.strip()]
    if not module_names:
        raise ValueError("at least one module is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)

    base_revision = getattr(args, "base_revision", None)
    post_revision = getattr(args, "post_revision", None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.post_model, revision=post_revision, trust_remote_code=False
    )
    base_model = load_hf_model(
        args.base_model, dtype=dtype, device=device, revision=base_revision
    )
    post_model = load_hf_model(
        args.post_model, dtype=dtype, device=device, revision=post_revision
    )
    if model_vocab_size(base_model) != model_vocab_size(post_model):
        raise ValueError("base and post-trained vocabularies differ")

    atoms, full_deltas, svd_diagnostics = build_delta_atoms(
        base_model,
        post_model,
        module_names,
        n_components=args.n_components,
        oversample=args.oversample,
        niter=args.svd_niter,
        atom_dtype=dtype,
        seed=args.seed,
    )

    base_margin = score_examples(
        base_model,
        tokenizer,
        examples,
        device=device,
        use_chat_template=args.chat_template,
    )
    post_margin = score_examples(
        post_model,
        tokenizer,
        examples,
        device=device,
        use_chat_template=args.chat_template,
    )

    conditions: dict[str, Any] = {}
    for mode, model in (("insert", base_model), ("ablate", post_model)):
        paired_dose_random = getattr(args, "paired_dose_random", False)
        if paired_dose_random:
            (
                candidate_margin,
                candidate_rms,
                random_margin,
                random_rms,
            ) = score_with_paired_dose_interventions(
                model,
                tokenizer,
                examples,
                atoms,
                device=device,
                use_chat_template=args.chat_template,
                k=args.k,
                mode=mode,
                seed=args.seed,
                pool_factor=args.pool_factor,
            )
        else:
            candidate_margin, candidate_rms = score_with_interventions(
                model,
                tokenizer,
                examples,
                atoms,
                device=device,
                use_chat_template=args.chat_template,
                policy="input_omp",
                k=args.k,
                mode=mode,
                seed=args.seed,
                pool_factor=args.pool_factor,
            )
            random_margin, random_rms = score_with_interventions(
                model,
                tokenizer,
                examples,
                atoms,
                device=device,
                use_chat_template=args.chat_template,
                policy="matched_random",
                k=args.k,
                mode=mode,
                seed=args.seed,
                pool_factor=args.pool_factor,
            )
        static_margin, static_rms = score_with_interventions(
            model,
            tokenizer,
            examples,
            atoms,
            device=device,
            use_chat_template=args.chat_template,
            policy="static_svd",
            k=args.k,
            mode=mode,
            seed=args.seed,
            pool_factor=args.pool_factor,
        )
        full_delta_margin, full_delta_rms = score_with_full_deltas(
            model,
            tokenizer,
            examples,
            full_deltas,
            device=device,
            use_chat_template=args.chat_template,
            mode=mode,
        )
        target_mask = torch.tensor([row["group"] == "target" for row in examples])
        control_mask = ~target_mask
        summary = summarize_causal_audit(
            base_margin=_tensor(base_margin),
            post_margin=_tensor(post_margin),
            candidate_margin=_tensor(candidate_margin),
            random_margin=_tensor(random_margin),
            target_mask=target_mask,
            control_mask=control_mask,
            mode=mode,
            seed=args.seed,
            n_bootstrap=args.bootstrap,
            max_control_fraction=args.max_control_fraction,
        )
        static_summary = summarize_causal_audit(
            base_margin=_tensor(base_margin),
            post_margin=_tensor(post_margin),
            candidate_margin=_tensor(candidate_margin),
            random_margin=_tensor(static_margin),
            target_mask=target_mask,
            control_mask=control_mask,
            mode=mode,
            seed=args.seed,
            n_bootstrap=args.bootstrap,
            max_control_fraction=args.max_control_fraction,
        )
        rms_ratios = {
            name: candidate_rms[name] / max(random_rms[name], 1e-12)
            for name in candidate_rms
        }
        rms_gate_passes = all(0.95 <= ratio <= 1.05 for ratio in rms_ratios.values())
        conditions[mode] = {
            "summary": _summary_dict(summary),
            "omp_minus_static_summary": _summary_dict(static_summary),
            "rms_ratio_omp_to_matched_random": rms_ratios,
            "rms_gate_passes": rms_gate_passes,
            "matched_random_dose": (
                "replayed actual OMP norm by module, example, and token"
                if paired_dose_random
                else "matched to OMP computed on the random trajectory"
            ),
            "passes_with_rms": summary.passes and rms_gate_passes,
            "candidate_margin": candidate_margin,
            "matched_random_margin": random_margin,
            "static_svd_margin": static_margin,
            "full_delta_margin": full_delta_margin,
            "candidate_rms_by_module": candidate_rms,
            "matched_random_rms_by_module": random_rms,
            "static_svd_rms_by_module": static_rms,
            "full_delta_rms_by_module": full_delta_rms,
        }

    per_example = []
    for index, row in enumerate(examples):
        per_example.append(
            {
                "id": row["id"],
                "group": row["group"],
                "base_margin": base_margin[index],
                "post_margin": post_margin[index],
                "insert_omp_margin": conditions["insert"]["candidate_margin"][index],
                "insert_random_margin": conditions["insert"]["matched_random_margin"][index],
                "insert_static_margin": conditions["insert"]["static_svd_margin"][index],
                "insert_full_delta_margin": conditions["insert"]["full_delta_margin"][index],
                "ablate_omp_margin": conditions["ablate"]["candidate_margin"][index],
                "ablate_random_margin": conditions["ablate"]["matched_random_margin"][index],
                "ablate_static_margin": conditions["ablate"]["static_svd_margin"][index],
                "ablate_full_delta_margin": conditions["ablate"]["full_delta_margin"][index],
            }
        )

    artifact = {
        "claim_scope": (
            "Held-out paired-completion behavioral intervention on the exact "
            "model pair, modules, dataset, and configuration below. This is not "
            "evidence that individual atoms are human-interpretable concepts."
        ),
        "dataset": {
            "path": str(Path(args.dataset).resolve()),
            "sha256": dataset_sha256,
            "split": args.split,
            "n_examples": len(examples),
            "n_target": sum(row["group"] == "target" for row in examples),
            "n_control": sum(row["group"] == "control" for row in examples),
        },
        "configuration": {
            "base_model": args.base_model,
            "post_model": args.post_model,
            "base_revision": base_revision,
            "post_revision": post_revision,
            "score_positive": examples[0]["positive_completion"],
            "score_negative": examples[0]["negative_completion"],
            "modules": module_names,
            "n_components": args.n_components,
            "k": args.k,
            "oversample": args.oversample,
            "svd_niter": args.svd_niter,
            "pool_factor": args.pool_factor,
            "seed": args.seed,
            "dtype": args.dtype,
            "chat_template": args.chat_template,
            "bootstrap": args.bootstrap,
            "max_control_fraction": args.max_control_fraction,
        },
        "svd_diagnostics": svd_diagnostics,
        "base_margin": base_margin,
        "post_margin": post_margin,
        "conditions": conditions,
        "per_example": per_example,
        "joint_gate_passes": (
            conditions["insert"]["passes_with_rms"]
            and conditions["ablate"]["passes_with_rms"]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--post-model", default=DEFAULT_POST_MODEL)
    parser.add_argument("--base-revision")
    parser.add_argument("--post-revision")
    parser.add_argument("--score-positive")
    parser.add_argument("--score-negative")
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=4)
    parser.add_argument("--pool-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-control-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        default="results/behavioral_causal_audit/qwen3_delta_audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "output": result["dataset"],
        "insert": result["conditions"]["insert"]["summary"],
        "ablate": result["conditions"]["ablate"]["summary"],
        "joint_gate_passes": result["joint_gate_passes"],
    }, indent=2))
