# References and citation guide

## Citing this repository

Use GitHub's **Cite this repository** menu, which reads `CITATION.cff`, or use:

```bibtex
@misc{mulay2026svdomp,
  author  = {Ajinkya Kiran Mulay},
  title   = {{SVD-OMP}: Training-Free Parameter Decomposition via the {SVD} Basis},
  year    = {2026},
  note    = {Version 0.2.0},
  url     = {https://github.com/thehimalayanleo/svd-omp}
}
```

Until the accompanying paper receives a stable archival identifier, cite the
software repository and the exact commit used. For a reproducible reference,
replace `<commit>` in this URL:

```text
https://github.com/thehimalayanleo/svd-omp/tree/<commit>
```

GitHub also provides a permanent link to any file or line range. Open the file,
press `y` to replace the branch name with the commit hash, select the relevant
lines, and copy the resulting permalink.

## Citing benchmark claims

Reference both this repository and the original baseline when discussing a
comparison:

- SVD-OMP versus VPD: cite this repository and Bushnaq et al. (2026).
- SVD-OMP, SVD-FoBa, or CP-SVD versus SWD: cite this repository and Yan et al.
  (2026).
- Include the benchmark artifact or commit because selected-unit fidelity,
  active-edge cost, storage, and latency are different comparison axes.

## VPD

VPD is introduced in Goodfire's research report:

Lucius Bushnaq, Dan Braun, Oliver Clive-Griffin, Bart Bussmann, Nathan Hu,
Michael Ivanitskiy, Linda Linsefors, and Lee Sharkey. “Interpreting Language
Model Parameters.” Goodfire, 2026.

- Research report: https://www.goodfire.ai/research/interpreting-lm-parameters

```bibtex
@misc{bushnaq2026interpreting,
  author       = {Bushnaq, Lucius and Braun, Dan and Clive-Griffin, Oliver and
                  Bussmann, Bart and Hu, Nathan and Ivanitskiy, Michael and
                  Linsefors, Linda and Sharkey, Lee},
  title        = {Interpreting Language Model Parameters},
  howpublished = {Goodfire Research},
  year         = {2026},
  url          = {https://www.goodfire.ai/research/interpreting-lm-parameters}
}
```

## SWD

Chuanhao Yan, Xuhan Huang, Yawen Duan, Zhenfei Yin, Hang Zhao, Bryan Dai, and
Jie Fu. “Sparse Weight Decomposition for Efficient Circuit Extraction.”
arXiv:2608.03913, 2026.

- Paper: https://arxiv.org/abs/2608.03913
- Reference implementation: https://github.com/Veri-Safe/SWD
- Revision used by this repository's strengthened control:
  `4c44b7281bc7c78f80e431dac3aa75f397dd3043`

```bibtex
@article{yan2026swd,
  author  = {Yan, Chuanhao and Huang, Xuhan and Duan, Yawen and Yin, Zhenfei and
             Zhao, Hang and Dai, Bryan and Fu, Jie},
  title   = {Sparse Weight Decomposition for Efficient Circuit Extraction},
  journal = {arXiv preprint arXiv:2608.03913},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03913}
}
```
