import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CosineSimilarity

from jex.jet_expand import JetExpansionOut, jet_expand_lm
from jex.models import LM


class JointJetLens:
    """Additive jet lens: centers are layer differentials h_l - h_{l-1}.

    The terms sum to reconstruct the full decoder output, each isolating the
    contribution of one residual block's nonlinearity (γ_l = h_l - h_{l-1}).

    Uniform weights 1/n are used by default so the terms form a convex
    combination. Weights can be optimized via optimize_weights(), which learns
    per-token scalings that minimise the reconstruction error in logit space.

    Note: unlike JetLensIterative (where each term stands alone and weights=1),
    here the terms should sum. Hence the weights should form a convex combination.
    """

    def __init__(self, lm: LM, layers: list[int], order: int):
        centers = [lm.residual_stream(layers[0])] + [
            (
                lambda a, b: (
                    lambda z: lm.residual_stream(b)(z) - lm.residual_stream(a)(z)
                )
            )(layers[i - 1], layers[i])
            for i in range(1, len(layers))
        ]
        self._jet_out: JetExpansionOut = jet_expand_lm(lm, lm.depth + 1, centers, order)
        self._lm = lm
        self._n = len(centers)
        self._log_weights: nn.Parameter | None = None
        self._metric: Tensor | None = None

    @property
    def metric(self) -> Tensor:
        """U^T U, precomputed lazily and cached. Shape (d_model, d_model)."""
        if self._metric is None:
            U = self._lm.unembedding.weight  # (vocab, d_model)
            self._metric = (U.T @ U).detach()
        return self._metric

    @property
    def weights(self) -> Tensor:
        """Per-token weights, shape (n_centers, seq_len), summing to 1 over centers."""
        assert self._log_weights is not None, "Call optimize_weights first."
        return self._log_weights.softmax(dim=0)

    def jet_logits(self, z: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Returns (combined_logits, per_center_logits), applying current weights."""
        exps = self._jet_out.expansions(z, with_unembedding=True)
        if self._log_weights is None:
            w = torch.full(
                (self._n,), 1.0 / self._n, device=exps[0].device, dtype=exps[0].dtype
            )
            weighted = [w[i] * e for i, e in enumerate(exps)]
        else:
            weighted = [self.weights[i].unsqueeze(-1) * e for i, e in enumerate(exps)]
        combined: Tensor = sum(weighted[1:], weighted[0])  # type: ignore[arg-type]
        return combined, weighted

    def remainder_logits(self, z: Tensor) -> Tensor:
        combined, _ = self.jet_logits(z)
        true_logits = self._lm.decoder(self._lm.residual_stream(self._lm.depth)(z))
        return true_logits - combined

    def _loss(self, z: Tensor) -> Tensor:
        """Reconstruction loss ||U(ln_h - weighted_combo)||_F^2 / numel, using U^T U metric."""
        exps = self._jet_out.expansions(z)  # hidden (ln) space
        combo = sum(
            (self.weights[i].unsqueeze(-1) * e for i, e in enumerate(exps[1:])),
            self.weights[0].unsqueeze(-1) * exps[0],
        )
        target = self._jet_out.f(z).detach()
        diff = target - combo
        metric = self.metric.to(diff.dtype)
        return torch.einsum("...i,ij,...j->", diff, metric, diff) / diff.numel()

    def optimize_weights(
        self, z: Tensor, lr: float = 1e-3, iters: int = 100
    ) -> list[float]:
        """Learn per-token weights minimising logit-space reconstruction error."""
        seq_len = z.shape[-1]
        self._log_weights = nn.Parameter(torch.zeros(self._n, seq_len, device=z.device))
        optim = torch.optim.Adam([self._log_weights], lr=lr)
        losses = []
        for _ in range(iters):
            optim.zero_grad()
            loss = self._loss(z)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        return losses

    def cosine_similarity(self, z: Tensor) -> Tensor:
        """Cosine similarity between the combined jet logits and the true logits."""
        combined, _ = self.jet_logits(z)
        true_logits = self._lm.decoder(self._lm.residual_stream(self._lm.depth)(z))
        return CosineSimilarity(dim=-1)(combined, true_logits).mean()
