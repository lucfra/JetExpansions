# JetExpansions

Interpretability with jet expansions of residual networks and transformers.

Code for the paper [Decomposing LLM Computation with Jets](https://openreview.net/pdf?id=u6JLh0BO5h) (ICLR 2026).

## Installation

```bash
# Standard install
uv sync

# Development install (includes pytest)
uv sync --group dev

# Examples install (includes jupyter, matplotlib, seaborn)
uv sync --group examples

# Activate the environment, or prefix commands with `uv run` (e.g. `uv run pytest`)
source .venv/bin/activate
```

## What's inside

- Pytorch implementation of jet operator via `jvp` (Jacobian vector product);
- composable `jet_expansion` algorithm. This comes in two versions:
  - a generic version for any functional callable;
  - a specialised version for residual nets/transformers and expansions around block non-linearities, closely following Algorithm 1 from the paper;
- loaders and abstractions for some HF models (gpt2, gpt neo, llama, ...); extensible to other models;
- iterative and joint jet lenses;
- jet bi-grams: embedding -> unembedding and paths through one mlp.


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
