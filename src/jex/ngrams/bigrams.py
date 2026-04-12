from typing import Callable

from torch import Tensor

from jex.jet_expand import JetExpansionOut, jet_expand_lm, jet_expand
from jex.models import LM


def embedding_unembedding(lm: LM) -> JetExpansionOut:
    # one here might also consider to expand the encoder (including pos embeddings)....
    center = [lm.embedding]
    # note, here there's one weight only, so totally fine to skip declaring it
    return jet_expand_lm(lm, lm.depth + 1, center, 0)


def embedding_mlp_unembedding(lm: LM, layer: int) -> JetExpansionOut:
    """Embedding -> LN+ MLP -> final LN + unembedding"""
    # also in these expansions, taken alone, there is only one weight
    center1 = [lm.embedding]
    def ln_mlp(z: Tensor):
        x = lm.pre_mlp_norms[layer](z)
        return lm.mlps[layer](x)
    variate = lm.residual_stream(layer)
    jet_out_intermediate = jet_expand(ln_mlp, center1, variate, 0)
    assert len(jet_out_intermediate.terms) == 1
    center2 = [jet_out_intermediate.terms[0]]
    return jet_expand_lm(lm, lm.depth + 1, center2, 0)