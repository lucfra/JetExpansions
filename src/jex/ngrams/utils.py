from collections.abc import Iterator

import torch
from torch import Tensor

from jex.jet_expand import JetExpansionOut
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


def eval_over_vocab(jet_out: JetExpansionOut, lm: LM, batch_size: int = 10000) -> Tensor:
    """Evaluate a JetExpansionOut over the full vocabulary.

    Returns logits of shape (vocab_size, vocab_size).
    """
    all_logits = []
    for batch in make_vocabulary(lm, ngrams=1, batch_size=batch_size):
        exps, _ = jet_out.expansions_and_remainder_with_unembedding(batch)
        logits: Tensor = sum(exps[1:], exps[0])  # type: ignore[arg-type]
        all_logits.append(logits[:, 0, :].detach())  # (batch_size, vocab)
    return torch.cat(all_logits, dim=0)  # (vocab_size, vocab)


def top_bigrams(logits: Tensor, lm: LM, n: int = 50) -> list[tuple[str, str, float]]:
    """Return the global top-n (input_token, next_token, probability) bigram pairs.

    Softmax is applied row-wise (over output tokens) before ranking, so scores are
    comparable across paths and inputs.
    """
    vocab_size = logits.shape[1]
    probs = logits.float().softmax(dim=1)  # (vocab, vocab)
    best = probs.flatten().topk(n)
    rows = []
    for score, idx in zip(best.values.tolist(), best.indices.tolist()):
        src = lm.tokenizer.decode([idx // vocab_size])
        tgt = lm.tokenizer.decode([idx % vocab_size])
        rows.append((src, tgt, score))
    return rows


def print_bigram_table(rows: list[tuple[str, str, float]], title: str) -> None:
    """Print a formatted bigram table using tabulate."""
    from tabulate import tabulate
    fmt = [(repr(src), repr(tgt), f"{p:.2%}") for src, tgt, p in rows]
    print(f"\n{title}")
    print(tabulate(fmt, headers=["Token", "Next token", "Prob"], tablefmt="simple_outline"))


def print_bigram_tables_side_by_side(tables: list[tuple[str, list[tuple[str, str, float]]]]) -> None:
    """Print multiple bigram tables side by side using tabulate.

    Args:
        tables: List of (title, rows) where rows come from top_bigrams().
    """
    from tabulate import tabulate
    n = max(len(rows) for _, rows in tables)
    combined = []
    for i in range(n):
        row = []
        for _, rows in tables:
            if i < len(rows):
                src, tgt, p = rows[i]
                row += [repr(src), repr(tgt), f"{p:.2%}"]
            else:
                row += ["", "", ""]
        combined.append(row)

    headers = []
    for title, _ in tables:
        headers += [title, "", ""]

    print(tabulate(combined, headers=headers, tablefmt="simple_outline"))
