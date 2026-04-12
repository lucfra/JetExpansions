import pytest
import torch

from jex.lenses.iterative import IterativeJetLenses
from jex.lenses.joint import JointJetLens
from jex.models import LM


@pytest.fixture
def layers(lm: LM) -> list[int]:
    d = lm.depth
    lys = [d // 4, d // 2, 3 * d // 4, d]
    # remove duplicates, if any (which would make the joint lens test fail)
    return list(set(lys))


@pytest.mark.parametrize("k", [0, 1])
def test_iterative_lens_shapes(lm: LM, tokens, layers, k):
    lens = IterativeJetLenses(lm, layers, k)
    exps, remainder = lens(tokens)
    assert len(exps) == len(layers)
    for e in exps:
        assert e.shape == remainder.shape
        assert e.shape == (*tokens.shape, lm.vocab_size)


@pytest.mark.parametrize("k", [0, 1])
def test_iterative_lens_cosine_similarity(lm: LM, tokens, layers, k):
    lens = IterativeJetLenses(lm, layers, k)
    cos = lens.cosine_similarity(tokens)
    assert cos.shape == (len(layers), *tokens.shape)
    assert (cos >= -1).all() and (cos <= 1).all()


@pytest.mark.parametrize("k", [0, 1])
def test_additive_lens_shapes(lm: LM, tokens, layers, k):
    lens = JointJetLens(lm, layers, k)
    combined, per_center = lens.jet_logits(tokens)
    assert len(per_center) == len(layers)
    assert combined.shape == (*tokens.shape, lm.vocab_size)
    for e in per_center:
        assert e.shape == combined.shape


@pytest.mark.parametrize("k", [0, 1])
def test_additive_lens_reconstruction(lm: LM, tokens, layers, k):
    """With uniform weights, combined + remainder should equal the true decoder output."""
    lens = JointJetLens(lm, layers, k)
    combined, _ = lens.jet_logits(tokens)
    remainder = lens.remainder_logits(tokens)
    true_logits = lm.decoder(lm.residual_stream(lm.depth)(tokens))
    tol = 1e-1 if true_logits.dtype == torch.float16 else 1e-4
    assert torch.allclose(combined + remainder, true_logits, atol=tol, rtol=tol)


def test_additive_lens_optimize_weights(lm: LM, tokens, layers, steps=5):
    if lm.name != "gpt2":
        pytest.skip("Keep it short for tests :)")
    lens = JointJetLens(lm, layers, order=0)
    losses = lens.optimize_weights(tokens, lr=1e-2, iters=steps)
    assert len(losses) == steps
    assert losses[-1] < losses[0], "Loss should decrease during optimization"
    # weights should be valid after optimization
    w = lens.weights
    assert w.shape == (len(layers), tokens.shape[1])
    assert torch.allclose(w.sum(dim=0), torch.ones(tokens.shape[1]), atol=1e-5)
