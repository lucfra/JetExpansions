import pytest
import torch
import torch.nn as nn
from jex.models import LM


def test_loads(lm):
    assert isinstance(lm, LM)
    assert isinstance(lm.layers, list)
    assert isinstance(lm.ln, nn.Module)
    assert isinstance(lm.unembedding, nn.Module)
    assert callable(lm.getter)


def test_forward(lm, tokens):
    with torch.no_grad():
        logits = lm.model(tokens).logits

    assert logits.shape == (1, tokens.shape[1], lm.vocab_size)


def test_layer_injection(lm: LM, tokens):
    if lm.pos_emb is None:
        pytest.skip(
            "layer injection not supported for RoPE models (position embeddings are not standalone modules)"
        )

    d = lm.model.config.hidden_size  # type: ignore

    with torch.no_grad():
        embeddings = lm.embedding(tokens)
        assert embeddings.shape == (1, tokens.shape[1], d)
        l4 = lm.getter(lm.layers[4](embeddings))
        assert l4.shape == (1, tokens.shape[1], d)
        ln_out = lm.ln(l4)
        assert ln_out.shape == (1, tokens.shape[1], d)
        out = lm.unembedding(ln_out)
        assert out.shape == (1, tokens.shape[1], lm.vocab_size)


def test_residual_stream(lm: LM, tokens):
    hidden_size = lm.model.config.hidden_size  # type: ignore
    seq_len = tokens.shape[1]
    expected_shape = (1, seq_len, hidden_size)

    # layer=0 is embedding output
    h0 = lm.residual_stream(0)(tokens)
    assert h0.shape == expected_shape

    # mid-layer
    mid = lm.depth // 2
    hm = lm.residual_stream(mid)(tokens)
    assert hm.shape == expected_shape

    # layer=depth is full residual stream (after last block)
    hd = lm.residual_stream(lm.depth)(tokens)
    assert hd.shape == expected_shape

    # successive layers produce different values
    assert not torch.allclose(h0, hm)
    assert not torch.allclose(hm, hd)

    # shared forward cache: calling two residual_stream CachedFs with the same tensor
    # should not trigger two full forward passes — verify outputs are consistent
    h_mid_a = lm.residual_stream(mid)(tokens)
    h_mid_b = lm.residual_stream(mid)(tokens)
    assert torch.allclose(h_mid_a, h_mid_b)

    # last layer coincides with the transformer output (before the lm head)
    h_L = lm.residual_stream(lm.depth)(tokens)
    h_L_after_global_norm = lm.ln(h_L)
    if hasattr(lm.model, "transformer"):
        # gpt2 - style
        transformer = lm.model.transformer
    elif hasattr(lm.model, "gpt_neox"):
        transformer = lm.model.gpt_neox
    else:
        pytest.skip("Don't know the transformer structure")
    if transformer:
        hL_transformer = transformer(tokens)  # type: ignore
        hL_transformer = lm.getter(hL_transformer)
        assert torch.allclose(hL_transformer, h_L_after_global_norm)

    # out-of-range raises
    with pytest.raises(AssertionError):
        lm.residual_stream(lm.depth + 1)


def test_layer_fn(lm, tokens):
    hidden_size = lm.model.config.hidden_size
    seq_len = tokens.shape[1]
    expected_shape = (1, seq_len, hidden_size)

    # layer_fn at index 0 and mid-network
    for idx in (0, lm.depth // 2):
        h_in = lm.residual_stream(idx)(tokens)
        h_out = lm.layer_fn(idx)(h_in)
        assert h_out.shape == expected_shape

    # output should differ from input (block is not identity)
    h_in = lm.residual_stream(0)(tokens)
    h_out = lm.layer_fn(0)(h_in)
    assert not torch.allclose(h_in, h_out)
