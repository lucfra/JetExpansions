import pytest
import torch

from jex.models import LM
from jex.ngrams.bigrams import embedding_unembedding, embedding_mlp_unembedding
from jex.ngrams.utils import make_vocabulary


@pytest.mark.parametrize('bs', [2, 13])
def test_eu(lm: LM, bs):
    eu_expansion = embedding_unembedding(lm)
    one_grams = make_vocabulary(lm, 1, batch_size=bs)
    exps, remainder = eu_expansion.expansions_and_remainder_with_unembedding(next(one_grams))
    assert remainder.shape == (bs, 1, lm.vocab_size)
    assert len(exps) == 1
    eu_out = exps[0]
    assert eu_out.shape == (bs, 1, lm.vocab_size)


def test_embedding_mlp_unembedding_path(gpt2: LM):
    """Verify that embedding_mlp_unembedding computes U(ln_f(mlp_l(pre_mlp_norm_l(embedding(z))))).

    Manually build the path and check it matches the jet expansion output exactly.
    """
    layer = 0
    tokens = torch.tensor([[42, 17, 3]])  # (1, 3)

    jet_out = embedding_mlp_unembedding(gpt2, layer)
    exps, _ = jet_out.expansions_and_remainder_with_unembedding(tokens)
    result = exps[0]  # (1, 3, vocab)

    with torch.no_grad():
        emb = gpt2.embedding(tokens)                          # (1, 3, d)
        normed = gpt2.pre_mlp_norms[layer](emb)              # (1, 3, d)
        after_mlp = gpt2.mlps[layer](normed)                 # (1, 3, d)
        after_ln = gpt2.ln(after_mlp)                        # (1, 3, d)
        expected = gpt2.unembedding(after_ln)                 # (1, 3, vocab)

    assert torch.allclose(result.float(), expected.float(), atol=1e-4)
