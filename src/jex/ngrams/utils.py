from collections.abc import Iterator
from typing import overload

import torch
from torch import Tensor

from jex.jet_expand import JetExpansionOut
from jex.models import LM


def make_vocabulary(
    lm: LM, ngrams: int = 1, batch_size: int | None = None
) -> Iterator[Tensor]:
    """Iterate over all n-gram batches of the vocabulary.

    Yields batches of shape (batch_size, ngrams). If batch_size is None, a single
    batch covering all sequences is yielded.
    For ngrams=1 there are v sequences total.
    For ngrams=k there are v^k sequences, yielded in chunks of batch_size.
    """
    v = torch.arange(lm.vocab_size)
    if ngrams == 1:
        all_seqs = v.unsqueeze(1)  # (V, 1)
    else:
        all_seqs = torch.cartesian_prod(*[v] * ngrams)  # (V^k, k)

    if batch_size is None:
        yield all_seqs
        return

    for i in range(0, len(all_seqs), batch_size):
        yield all_seqs[i : i + batch_size]


@overload
def eval_over_vocab(
    jet_out: JetExpansionOut,
    lm: LM,
    batch_size: int = ...,
    *,
    prob_space: bool = ...,
    row_topk: int,
    global_topk: None = ...,
) -> tuple[Tensor, Tensor]: ...
@overload
def eval_over_vocab(
    jet_out: JetExpansionOut,
    lm: LM,
    batch_size: int = ...,
    *,
    prob_space: bool = ...,
    row_topk: None = ...,
    global_topk: int,
) -> tuple[Tensor, Tensor]: ...
@overload
def eval_over_vocab(
    jet_out: JetExpansionOut,
    lm: LM,
    batch_size: int = ...,
    *,
    prob_space: bool = ...,
    row_topk: None = ...,
    global_topk: None = ...,
) -> Tensor: ...
def eval_over_vocab(
    jet_out: JetExpansionOut,
    lm: LM,
    batch_size: int = 2000,
    *,
    prob_space: bool = True,
    row_topk: int | None = None,
    global_topk: int | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Evaluate a JetExpansionOut over the full vocabulary.

    - Default: returns a (vocab, vocab) matrix of logits or row-wise probabilities.
    - row_topk: returns (values, indices) each of shape (vocab, row_topk).
    - global_topk: returns (values, indices) where values has shape (global_topk,)
      and indices has shape (global_topk, 2) with columns [input_token, output_token].
    """
    assert row_topk is None or 0 < row_topk <= lm.vocab_size
    assert global_topk is None or global_topk > 0
    assert jet_out.unembedding is not None, "JetExpansionOut must have an unembedding"
    device = jet_out.unembedding.weight.device
    all_logits = []
    all_values, all_indices = [], []
    k = global_topk if global_topk is not None else row_topk
    for batch in make_vocabulary(lm, ngrams=1, batch_size=batch_size):
        exps = jet_out.expansions(batch.to(device))
        hidden: Tensor = sum(exps[1:], exps[0])  # type: ignore[arg-type]
        out = jet_out.unembedding(hidden)
        if prob_space:
            out = out.softmax(dim=-1)
        if k:
            values, indices = torch.topk(out, k=k, dim=-1)
            all_values.append(values)
            all_indices.append(indices)
        else:
            all_logits.append(out[:, 0, :])  # (batch_size, v)

    if k:
        values = torch.cat(all_values, dim=0)  # (V, k)
        indices = torch.cat(all_indices, dim=0)  # (V, k)
        if global_topk is not None:
            flat_values = values.flatten()
            best = flat_values.topk(global_topk)
            flat_idx = best.indices
            in_tokens = flat_idx // global_topk
            out_tokens = indices.flatten()[flat_idx]
            return best.values, torch.stack(
                [in_tokens, out_tokens], dim=-1
            )  # (n,), (n, 2)
        return values, indices  # (V, row_topk), (V, row_topk)

    return torch.cat(all_logits, dim=0)  # (v, v)


def decode_topk(
    values: Tensor, indices: Tensor, lm: LM
) -> list[tuple[str, str, float]]:
    """Decode topk results from eval_over_vocab into (input_token, output_token, score) triples.

    Accepts both formats returned by eval_over_vocab:
    - row_topk: values/indices of shape (V, k) — input token is the row index.
    - global_topk: values of shape (n,), indices of shape (n, 2) with [input, output].
    """
    rows = []
    if indices.dim() == 2 and indices.shape[-1] == 2:
        # global_topk format: indices is (n, 2)
        for score, (src_id, tgt_id) in zip(values.tolist(), indices.tolist()):
            src = lm.tokenizer.decode([src_id])
            tgt = lm.tokenizer.decode([tgt_id])
            rows.append((src, tgt, score))
    elif indices.dim() == 1:
        # row_topk format: values/indices are (V, k)
        for src_id, (row_scores, row_indices) in enumerate(
            zip(values.tolist(), indices.tolist())
        ):
            src = lm.tokenizer.decode([src_id])
            for score, tgt_id in zip(row_scores, row_indices):
                tgt = lm.tokenizer.decode([tgt_id])
                rows.append((src, tgt, score))
    else:
        raise NotImplementedError()
    return rows


def print_bigram_table(rows: list[tuple[str, str, float]], title: str) -> None:
    """Print a formatted bigram table using tabulate."""
    from tabulate import tabulate

    fmt = [(repr(src), repr(tgt), f"{p:.2%}") for src, tgt, p in rows]
    print(f"\n{title}")
    print(
        tabulate(
            fmt, headers=["Token", "Next token", "Prob"], tablefmt="simple_outline"
        )
    )


def print_bigram_tables_side_by_side(
    tables: list[tuple[str, list[tuple[str, str, float]]]],
) -> None:
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
