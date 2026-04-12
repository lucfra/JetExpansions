from collections.abc import Iterator

import torch
from torch import Tensor

from jex.models import LM


def make_vocabulary(lm: LM, ngrams: int = 1, batch_size: int | None = None) -> Iterator[Tensor]:
    """Iterate over all n-gram batches of the vocabulary.

    Yields batches of shape (batch_size, ngrams). If batch_size is None, a single
    batch covering all sequences is yielded.
    For ngrams=1 there are vocab_size sequences total.
    For ngrams=k there are vocab_size^k sequences, yielded in chunks of batch_size.
    """
    vocab = torch.arange(lm.vocab_size)
    if ngrams == 1:
        all_seqs = vocab.unsqueeze(1)  # (V, 1)
    else:
        all_seqs = torch.cartesian_prod(*[vocab] * ngrams)  # (V^k, k)

    if batch_size is None:
        yield all_seqs
        return

    for i in range(0, len(all_seqs), batch_size):
        yield all_seqs[i : i + batch_size]
