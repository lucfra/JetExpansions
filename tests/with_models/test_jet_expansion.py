import pytest
import torch

from jex.jet_expand import expand_lm
from jex.models import LM


@pytest.fixture()
def centers(lm: LM):
    centers = [
        lm.embedding,
        lm.residual_stream(1),  # = h_1
        lambda z: lm.residual_stream(4)(z) - lm.residual_stream(3)(z),  # = gamma_4(h_3)
    ]
    return centers


@pytest.mark.parametrize("k", [0, 1])
def test_decoder_expansion(lm: LM, tokens, centers, k):
    exp_out = expand_lm(lm, lm.depth + 1, centers, k)
    expansions, reminder = exp_out.expansions_and_remainder(tokens)
    assert len(expansions) == len(centers)
    assert expansions[0].shape == reminder.shape

    f_from_expansions = sum(expansions) + reminder
    # compute output from the "standard" forward
    if hasattr(lm.model, "transformer"):
        # gpt2 - style
        transformer = lm.model.transformer
    elif hasattr(lm.model, "gpt_neox"):
        transformer = lm.model.gpt_neox
    else:
        pytest.skip("Don't know the transformer structure")
    original_out = transformer(tokens)[0]  # type: ignore

    # with float16 we are a bit off... but still should be
    tol = 1e-1 if original_out.dtype == torch.float16 else 1e-4
    assert torch.allclose(f_from_expansions, original_out, atol=tol, rtol=tol)


@pytest.mark.parametrize("k", [0, 1])
def test_jet_expansions_are_diff_wrt_weights(lm: LM, tokens, centers, k):
    log_weights = torch.nn.Parameter(torch.zeros(len(centers)))
    weights = log_weights.softmax(dim=0)
    expansion_out = expand_lm(lm, lm.depth + 1, centers, k, weights=weights)
    _, r = expansion_out.expansions_and_remainder(tokens)
    # a scalar loss from the reminder
    loss = torch.sum(r**2)
    assert loss
    grad = torch.autograd.grad(loss, log_weights)
    assert len(grad) == 1
    grad = grad[0]
    assert grad.shape == log_weights.shape
    assert not torch.allclose(grad, torch.zeros_like(grad))
