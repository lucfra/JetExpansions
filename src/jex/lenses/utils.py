from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from torch import Tensor

from jex.models import LM


def top_token_table(
    expansions: list[Tensor],
    lm: LM,
) -> tuple[list[list[str]], list[list[float]]]:
    """Extract the top-1 predicted token and its logit score for each (layer, position).

    Args:
        expansions: List of logit tensors, one per layer, each shape (1, seq_len, vocab).
        lm: The language model (used for decoding token ids).

    Returns:
        (data, values): both shape (n_layers, seq_len).
        data[i][j]  — decoded string of the top-1 token at layer i, position j.
        values[i][j] — its raw logit score.
    """
    data, values = [], []
    for exp in expansions:
        logits = exp[0]  # (seq_len, vocab)
        top_vals, top_ids = logits.float().topk(1, dim=-1)
        top_vals = top_vals[:, 0].tolist()
        top_ids = top_ids[:, 0].tolist()
        tok_strs = lm.tokenizer.convert_ids_to_tokens(top_ids)
        data.append([t.replace("Ġ", "_") for t in tok_strs])
        values.append(top_vals)
    return data, values


def plot_table(
    ax: Any,
    data: list[list[str]],
    values: list[list[float]],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    extra_values: list[list[float]] | None = None,
) -> None:
    """Draw a color-coded token table on `ax`.

    Cells are colored with a diverging RdBu_r colormap centered at 0.
    Pass extra_values to include them in the normalization range without plotting them.
    """
    cmap = plt.get_cmap("RdBu_r")
    arr = np.array(values)
    all_vals = arr if extra_values is None else np.concatenate([arr, np.array(extra_values)], axis=0)
    norm = colors.TwoSlopeNorm(
        vmin=min(float(all_vals.min()) * 1.3, -0.1),
        vcenter=0.0,
        vmax=max(float(all_vals.max()) * 1.3, 0.1),
    )
    ax.set_axis_off()
    ax.table(
        cellText=data,
        cellColours=cmap(norm(arr)),
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="right",
    ).auto_set_font_size(True)
    plt.tight_layout()


def plot_lens_table(
    expansions: list[Tensor],
    true_logits: Tensor,
    tokens: Tensor,
    lm: LM,
    title: str | None = None,
    layer_indices: list[int] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot the full jet lens table: one row per layer, one column per input token.

    Each cell shows the top-1 predicted next token at that layer and position,
    colored by raw logit score (diverging RdBu_r, centered at 0).
    A bottom row shows the true model output for reference.

    Args:
        expansions: Output of lens(tokens)[0] — list of (1, seq_len, vocab) tensors.
        true_logits: Shape (1, seq_len, vocab) — the actual model output logits.
        tokens: Shape (1, seq_len) — input token ids (used for column labels).
        lm: The language model.
        title: Optional figure title.
        layer_indices: Layer numbers for row labels; defaults to 0, 1, 2, ...
        save_path: If given, save the figure to this path.
    """
    n_layers = len(expansions)
    seq_len = tokens.shape[-1]

    col_labels = [lm.tokenizer.convert_ids_to_tokens([t.item()])[0].replace("Ġ", "_") for t in tokens[0]]
    row_labels = [f"Layer {i}" for i in (layer_indices or range(n_layers))]

    data, values = top_token_table(expansions, lm)

    tl = true_logits[0].float()
    true_vals, true_ids = tl.topk(1, dim=-1)
    true_data = [[t.replace("Ġ", "_") for t in lm.tokenizer.convert_ids_to_tokens(true_ids[:, 0].tolist())]]
    true_values = [true_vals[:, 0].tolist()]

    ratio = 1.5
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [n_layers + 1, 1], "hspace": 0.01},
        figsize=((seq_len + 2) * 0.9 * ratio, (n_layers + 2) * 0.18 * ratio),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plot_table(axes[0, 0], data, values, row_labels=row_labels, col_labels=col_labels, extra_values=true_values)
    plot_table(axes[1, 0], true_data, true_values, row_labels=["True logits"], extra_values=values)

    plt.subplots_adjust(left=0.1, right=0.9)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_joint_lens_table(
    per_center_logits: list[Tensor],
    combined_logits: Tensor,
    true_logits: Tensor,
    tokens: Tensor,
    lm: LM,
    weights: Tensor,
    cos_sim: float,
    layer_indices: list[int] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Plot the joint jet lens table: one row per center, plus reconstruction and true logit rows.

    Each cell shows the top-1 token and its per-position weight percentage.
    Row labels show the layer index and mean weight across positions.
    Bottom section has two rows: true logits and the combined reconstruction (with cosine sim).

    Args:
        per_center_logits: lens.jet_logits(z)[1] — list of (1, seq_len, vocab) weighted tensors.
        combined_logits: lens.jet_logits(z)[0] — shape (1, seq_len, vocab).
        true_logits: Shape (1, seq_len, vocab) — the actual model output logits.
        tokens: Shape (1, seq_len) — input token ids (used for column labels).
        lm: The language model.
        weights: lens.weights — shape (n_centers, seq_len), softmaxed per-token weights.
        cos_sim: Cosine similarity of the reconstruction (shown in the reconstruction row label).
        layer_indices: Layer numbers for row labels; defaults to 0, 1, 2, ...
        title: Optional figure title.
        save_path: If given, save the figure to this path.
    """
    n_centers = len(per_center_logits)
    seq_len = tokens.shape[-1]
    w = weights.detach().float()  # (n_centers, seq_len)

    col_labels = [lm.tokenizer.convert_ids_to_tokens([t.item()])[0].replace("Ġ", "_") for t in tokens[0]]

    indices = layer_indices or list(range(n_centers))
    mean_w = w.mean(dim=1)  # (n_centers,)
    block_names = ["Embedding"] + [f"Block {i}" for i in indices[1:]]
    row_labels = [f"{name} ({mean_w[k].item() * 100:.2f}%)" for k, name in enumerate(block_names)]

    # cell text: "token (w%)" where w is the per-position weight
    raw_data, values = top_token_table(per_center_logits, lm)
    data = [
        [f"{tok} ({w[i, j].item() * 100:.2f}%)" for j, tok in enumerate(row)]
        for i, row in enumerate(raw_data)
    ]

    # bottom rows: true logits + reconstruction
    tl = true_logits[0].float()
    true_vals, true_ids = tl.topk(1, dim=-1)
    true_data = [[t.replace("Ġ", "_") for t in lm.tokenizer.convert_ids_to_tokens(true_ids[:, 0].tolist())]]
    true_values = [true_vals[:, 0].tolist()]

    combined_data, combined_values = top_token_table([combined_logits], lm)
    bottom_data = true_data + combined_data
    bottom_values = true_values + combined_values
    bottom_row_labels = ["Logits", f"Expan. ({cos_sim:.3f})"]

    ratio = 1.5
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": [n_centers + 1, 2], "hspace": 0.4},
        figsize=((seq_len + 2) * 0.9 * ratio, (n_centers + 3) * 0.18 * ratio),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    plot_table(axes[0, 0], data, values, row_labels=row_labels, col_labels=col_labels, extra_values=bottom_values)
    plot_table(axes[1, 0], bottom_data, bottom_values, row_labels=bottom_row_labels, extra_values=values)

    plt.subplots_adjust(left=0.15, right=0.9)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
