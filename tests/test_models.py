import pytest
from pytest import fixture
import torch
import torch.nn as nn
from jex.models import LM, from_pretrained


@fixture(scope="module")
def gpt2() -> LM:
    return from_pretrained("gpt2")


@fixture(scope="module")
def pythia() -> LM:
    return from_pretrained("EleutherAI/pythia-70m")


@fixture(params=["gpt2", "pythia"])
def lm(request) -> LM:
    return request.getfixturevalue(request.param)


@fixture
def tokens(lm):
    return torch.tensor(lm.tokenizer.encode("Jet expansions?")).unsqueeze(0)


def test_loads(lm):
    assert isinstance(lm, LM)
    assert isinstance(lm.layers, list)
    assert isinstance(lm.ln, nn.Module)
    assert isinstance(lm.lm_head, nn.Module)
    assert callable(lm.getter)


def test_forward(lm, tokens):
    with torch.no_grad():
        logits = lm.model(tokens).logits

    assert logits.shape == (1, tokens.shape[1], lm.vocab_size)


def test_layer_injection(lm, tokens):
    if lm.pos_emb is None:
        pytest.skip("layer injection not supported for RoPE models (position embeddings are not standalone modules)")

    with torch.no_grad():
        embeddings = lm.emb(tokens)
        assert embeddings.shape == (1, tokens.shape[1], lm.model.config.hidden_size)
        l4 = lm.getter(lm.layers[4](embeddings))
        assert l4.shape == (1, tokens.shape[1], lm.model.config.hidden_size)
        ln_out = lm.ln(l4)
        assert ln_out.shape == (1, tokens.shape[1], lm.model.config.hidden_size)
        out = lm.lm_head(ln_out)
        assert out.shape == (1, tokens.shape[1], lm.vocab_size)