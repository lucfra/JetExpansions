import torch

from jex.jet_expand import expand_lm
from jex.models import toy_two_layer_rn


def test_quickstart_two_layer_rn():
    """Replicates the README quickstart: two-step jet carving of a two-block RN."""
    torch.manual_seed(0)
    lm = toy_two_layer_rn(d=32)
    z = torch.randint(0, lm.vocab_size, (1, 8))

    x0 = lm.residual_stream(0)
    x1 = lm.layer_gamma(0)

    with torch.no_grad():
        inner = expand_lm(lm, layer=2, centers=[x0, x1], order=1)
        outer = expand_lm(lm, layer=lm.depth + 1, centers=inner.terms, order=1)
        paths, remainder = outer.expansions_and_remainder(z, with_unembedding=True)

    # correct number of paths
    assert len(paths) == len(inner.terms) == 4

    # paths + remainder should reconstruct the true logits
    true_logits = lm.decoder(lm.residual_stream(lm.depth)(z))
    reconstructed = sum(paths[1:], paths[0]) + remainder
    assert torch.allclose(reconstructed, true_logits, atol=1e-4)
