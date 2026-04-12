from dataclasses import dataclass
from typing import Callable

from torch import Tensor, nn
import torch

from jex.jet import jet
from jex.models import LM
from jex.utils import CachedF


class JetExpandedTerm(nn.Module):
    
    def __init__(self, f: Callable[[Tensor], Tensor], center: Callable[[Tensor], Tensor], variate: Callable[[Tensor], Tensor], order: int, weight: Tensor):
        super().__init__()
        assert order >= 0, "Order should be non negative"
        self._f = f
        self._center = center
        self._variate = variate
        self._order = order
        self._weight = weight
        
    def forward(self, z: Tensor) -> Tensor:
        x0 = self._center(z)
        if self._order > 0:
            with torch.no_grad():
                y = self._variate(z)
        else:
            # no need to actually compute the variate since it's not used for zeroth-order exps
            y = torch.zeros_like(x0)
        jet_out = jet(self._f, x0, y, self._order)
        return self._weight * jet_out
    

@dataclass
class JetExpansionOut:
    terms: list[JetExpandedTerm]
    f: Callable[[Tensor], Tensor]
    unembedding: nn.Linear | None = None
    
    def expansions_and_remainder(self, z: Tensor) -> tuple[list[Tensor], Tensor]:
        """Compute expansions and remainder in one pass, reusing term evaluations."""
        exps = [term(z) for term in self.terms]
        return exps, self.f(z) - sum(exps)
    
    def expansions_and_remainder_with_unembedding(self, z: Tensor) -> tuple[list[Tensor], Tensor]:
        if self.unembedding is None:
            raise ValueError("unembedding is None")
        exps, remainder = self.expansions_and_remainder(z)
        exps = [self.unembedding(exp) for exp in exps]
        return exps, self.unembedding(remainder)
        
    
def jet_expand(f: Callable[[Tensor], Tensor], centers: list[Callable[[Tensor], Tensor]], variate: Callable[[Tensor], Tensor], order: int, weights: Tensor | None = None) -> JetExpansionOut:
    assert order >= 0, "Order should be non negative"
    n = len(centers)
    if weights is None:
        weights = torch.ones((n,)) / n
    assert weights.shape == (n,), f"Weight should be a vector of length n: {n}; got\n{weights}"
    if not isinstance(variate, CachedF):
        variate = CachedF(variate)
    terms = [JetExpandedTerm(f, centers[i], variate, order, weights[i]) for i in range(n)]
    return JetExpansionOut(terms, f)


def jet_expand_lm(lm: LM, layer: int, centers: list[Callable[[Tensor], Tensor]], order: int, weights: Tensor | None = None) -> JetExpansionOut:
    """
    Implements algorithm 1 from the [paper](https://openreview.net/pdf?id=u6JLh0BO5h), 
    except: 
    1) weight tensors are passed at construction, (if None, then they're set to uniform weights) 
    2) on layer = L+1, the jet expansion is applied to the final layer norm, instead of the full decoder, so one needs to call unembedding."""
    assert 1 <= layer <= lm.depth + 1, f"layer must be in [1, depth+1={lm.depth + 1}], got {layer}"
    variate = lm.residual_stream(layer - 1)
    if layer <= lm.depth:
        gamma = lm.layer_fn(layer - 1)
        jet_out = jet_expand(gamma, centers, variate, order, weights)
        jet_out_id = jet_expand(torch.nn.Identity(), centers, variate, order, weights)
        jet_out.terms.extend(jet_out_id.terms)
        jet_out.f = lm.residual_stream(layer)
    else:
        jet_out = jet_expand(lm.ln, centers, variate, order, weights)
        jet_out.f = lambda z: lm.ln(variate(z))
        jet_out.unembedding = lm.unembedding
    return jet_out
        
    