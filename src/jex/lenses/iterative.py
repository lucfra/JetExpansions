import torch
from torch import Tensor
from torch.nn import CosineSimilarity

from jex.jet_expand import JetExpansionOut, jet_expand_lm
from jex.models import LM


class IterativeJetLenses:
    """Iterative jet lens: one independent jet expansion per layer.

    Each center is a layer output (absolute residual stream value), so each
    term independently approximates the full decoder output — compatible with
    the logit lens interpretation.

    Note:  Iterative jet lenses are separate by layer, they are not meant to form a 
    convex combination, unlike the joint lens.
    That's why in the implementation we set all weights to 1. 
    """

    def __init__(self, lm: LM, layers: list[int], order: int):
        centers = [lm.residual_stream(l) for l in layers]
        weight = torch.ones(len(centers))
        self._jet_out: JetExpansionOut = jet_expand_lm(lm, lm.depth + 1, centers, order, weights=weight)
        self._lm = lm

    def __call__(self, z: Tensor) -> tuple[list[Tensor], Tensor]:
        """Evaluate expansion on z. Returns (expansions, remainder) in logit space."""
        return self._jet_out.expansions_and_remainder_with_unembedding(z)

    def cosine_similarity(self, z: Tensor) -> Tensor:
        """Cosine similarity between each jet expansion and the true logits.

        Returns a tensor of shape (n_layers, seq_len).
        """
        exps, _ = self(z)
        true_logits = self._lm.decoder(self._lm.residual_stream(self._lm.depth)(z))
        cc = CosineSimilarity(dim=-1)
        true_logits = true_logits.float()
        return torch.stack([cc(e.float(), true_logits) for e in exps]).clamp(-1, 1)
