"""Shared configuration for the Goodfire 67M LlamaSimpleMLP comparison.

Lists the 24 target weight matrices and the per-module (C, k) settings used in the
VPD paper. Keeping these here means every script imports the same constants.
"""

TARGET_MODULES = [
    "h.0.attn.q_proj", "h.0.attn.k_proj", "h.0.attn.v_proj", "h.0.attn.o_proj",
    "h.0.mlp.c_fc",    "h.0.mlp.down_proj",
    "h.1.attn.q_proj", "h.1.attn.k_proj", "h.1.attn.v_proj", "h.1.attn.o_proj",
    "h.1.mlp.c_fc",    "h.1.mlp.down_proj",
    "h.2.attn.q_proj", "h.2.attn.k_proj", "h.2.attn.v_proj", "h.2.attn.o_proj",
    "h.2.mlp.c_fc",    "h.2.mlp.down_proj",
    "h.3.attn.q_proj", "h.3.attn.k_proj", "h.3.attn.v_proj", "h.3.attn.o_proj",
    "h.3.mlp.c_fc",    "h.3.mlp.down_proj",
]

# Dictionary size per module type (matches `C_PER_MODULE_4L` in the VPD paper).
C_PER_MODULE = {
    "attn.q_proj": 512,  "attn.k_proj": 512,
    "attn.v_proj": 1024, "attn.o_proj": 1024,
    "mlp.c_fc":    3072, "mlp.down_proj": 3584,
}

# Sparsity per module type.
K_PER_MODULE = {
    "q_proj": 8,  "k_proj": 8,
    "v_proj": 10, "o_proj": 10,
    "c_fc": 12,   "down_proj": 12,
}


def get_C(module_path: str) -> int:
    for suffix, C in C_PER_MODULE.items():
        if module_path.endswith(suffix):
            return C
    raise ValueError(f"Unknown module: {module_path}")


def get_k(module_path: str) -> int:
    for suffix, k in K_PER_MODULE.items():
        if module_path.endswith(suffix):
            return k
    raise ValueError(f"Unknown module: {module_path}")
