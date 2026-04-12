import pytest

from jex.models import LM
from jex.ngrams.bigrams import embedding_unembedding
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
    
    
    