import pytest
import torch

from jex.jet_expand import jet_expand_lm
from jex.models import LM


@pytest.mark.parametrize("k", [0, 1])
def test_decoder_expansion(lm: LM, tokens, k):
    centers = [
        lm.embedding, 
        lm.residual_stream(1),  # = h_1
        lambda z: lm.residual_stream(4)(z) - lm.residual_stream(3)(z)   # = gamma_4(h_3)
    ]
    exp_out = jet_expand_lm(lm, lm.depth + 1, centers, k)
    expansions, reminder = exp_out.expansions_and_remainder(tokens)
    assert len(expansions) == len(centers)
    assert expansions[0].shape == reminder.shape

    f_from_expansions = sum(expansions) + reminder
    # compute output from the "standard" forward
    if hasattr(lm.model, 'transformer'):
        # gpt2 - style
        transformer = lm.model.transformer
    elif hasattr(lm.model, 'gpt_neox'):
        transformer = lm.model.gpt_neox
    else:
        pytest.skip("Don't know the transformer structure")
    original_out = transformer(tokens)[0]  # type: ignore
    
    # with float16 we are a bit off... but still should be
    tol = 1e-1 if original_out.dtype == torch.float16 else 1e-4
    assert torch.allclose(f_from_expansions, original_out, atol=tol, rtol=tol)