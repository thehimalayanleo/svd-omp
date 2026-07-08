"""Run this ONCE inside your Colab notebook (after the install cell that
clones goodfire-ai/param-decomp + sys.path.insert) to save the 24 target
weight matrices to a single file for local reuse.

Cell content to paste in Colab:
--------------------------------

import torch
from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import LlamaSimpleMLP

target_model = LlamaSimpleMLP.from_pretrained('goodfire/spd/runs/t-9d2b8f02')
target_model.eval()

TARGET_MODULES = [
    'h.0.attn.q_proj', 'h.0.attn.k_proj', 'h.0.attn.v_proj', 'h.0.attn.o_proj',
    'h.0.mlp.c_fc',    'h.0.mlp.down_proj',
    'h.1.attn.q_proj', 'h.1.attn.k_proj', 'h.1.attn.v_proj', 'h.1.attn.o_proj',
    'h.1.mlp.c_fc',    'h.1.mlp.down_proj',
    'h.2.attn.q_proj', 'h.2.attn.k_proj', 'h.2.attn.v_proj', 'h.2.attn.o_proj',
    'h.2.mlp.c_fc',    'h.2.mlp.down_proj',
    'h.3.attn.q_proj', 'h.3.attn.k_proj', 'h.3.attn.v_proj', 'h.3.attn.o_proj',
    'h.3.mlp.c_fc',    'h.3.mlp.down_proj',
]

weights = {
    p: target_model.get_submodule(p).weight.detach().float().cpu()
    for p in TARGET_MODULES
}
torch.save(weights, 'goodfire_67m_weights.pt')

# Also save one calibration batch of activations from a mid layer so we can
# do the stable-rank sweep on Goodfire's activations directly.
import numpy as np
d_in = weights['h.0.attn.q_proj'].shape[1]
torch.manual_seed(0)
sample_ids = torch.randint(0, target_model.config.vocab_size, (16, 128))
with torch.no_grad():
    # Hook the first MLP's input.
    captured = []
    def hook(mod, inp, out):
        captured.append(inp[0].detach().cpu())  # residual stream input
    h = target_model.h[2].mlp.c_fc.register_forward_hook(hook)
    _ = target_model(sample_ids.to(next(target_model.parameters()).device))
    h.remove()
activations = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured], dim=0)
torch.save(activations, 'goodfire_67m_activations.pt')

from google.colab import files
files.download('goodfire_67m_weights.pt')
files.download('goodfire_67m_activations.pt')
print('Downloaded 2 files. Move to your svd-omp/weights/ directory.')

--------------------------------

After download, move both files to:
    ~/github-repos/svd-omp/weights/

then run:
    python run_on_real_goodfire.py
"""
