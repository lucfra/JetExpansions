# JetExpansions

Interpretability with jet expansions of residual networks and transformers.

Code for the paper [Decomposing LLM Computation with Jets](https://openreview.net/pdf?id=u6JLh0BO5h) (ICLR 2026).

## Installation

Requires Python 3.11.

```bash
# Standard install
uv sync

# Development install (includes pytest)
uv sync --group dev

# Activate the environment, or prefix commands with `uv run` (e.g. `uv run pytest`)
source .venv/bin/activate
```

## Citation

```bibtex
@inproceedings{chen2026decomposing,
  title     = {Decomposing {LLM} Computation with Jets},
  author    = {Chen, Yihong and Xu, Xiangxiang and Stenetorp, Pontus and Riedel, Sebastian and Franceschi, Luca},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/pdf?id=u6JLh0BO5h}
}
```
