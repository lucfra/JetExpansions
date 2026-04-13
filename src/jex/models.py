from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from jex.utils import CachedF


@dataclass
class LM:
    """Minimal adapter exposing the components needed for jet expansions.

    Attributes:
        model: The underlying model for forward passes (HuggingFace model, but should accommodate other models as well).
        tokenizer: The associated tokenizer.
        layers: Ordered list of transformer blocks.
        attentions: Ordered list of attention sub-modules (one per block).
        mlps: Ordered list of MLP sub-modules (one per block).
        pre_attn_norms: Layer norms applied before attention (one per block).
        pre_mlp_norms: Layer norms applied before the MLP (one per block).
        ln: Final layer norm applied before the lm_head.
        lm_head: Unembedding projection (hidden dim → vocab).
        encoder: Embedding function (vocab -> hidden dim).
        pos_emb: Positional embedding module, or None for models that use
            rotary/relative encodings (e.g. Llama, Mistral, Gemma).
        getter: Extracts the hidden-state tensor from a block's forward output.
                This is needed since different architectures return tuples, tensors, or dataclasses.
                The getter abstracts these differences away.
        layer_fn: Factory returning a callable h -> h' for layer[idx], handling positional
                  encodings for each architecture (absolute, per-layer RoPE, model-level RoPE).
                  Signature: (idx: int) -> Callable[[Tensor], Tensor].
        name: Optional identifier for the model (set automatically to the HuggingFace model_id
              by from_pretrained; can be set manually for custom models).
    """

    model: nn.Module
    tokenizer: Any
    layers: list[nn.Module]
    attentions: list[nn.Module]
    mlps: list[nn.Module]
    pre_attn_norms: list[nn.Module]
    pre_mlp_norms: list[nn.Module]
    ln: nn.Module
    unembedding: nn.Linear
    embedding: nn.Module
    pos_emb: nn.Module | None
    getter: Callable[[Any], Tensor]
    layer_fn: Callable[[int], Callable[[Tensor], Tensor]]
    name: str | None = None

    def __post_init__(self):
        # one CachedF capturing ALL layer hidden states — shared across residual_stream calls
        def _capture_all(z: Tensor) -> list[Tensor]:
            captured = []
            handles = [
                layer.register_forward_hook(
                    lambda _m, _inp, out, _c=captured: _c.append(self.getter(out))
                )
                for layer in self.layers
            ]
            with torch.no_grad():
                self.model(z)
            for h in handles:
                h.remove()
            return captured

        self._forward_cache = CachedF(_capture_all)  # type: ignore

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def decoder(self) -> Callable[[Tensor], Tensor]:
        """Dec = ln ∘ lm_head — the full decoder as a single callable."""
        return lambda x: self.unembedding(self.ln(x))

    @property
    def encoder(self) -> Callable[[Tensor], Tensor]:
        """Enc = embedding (+ pos_emb if applicable) — maps token ids to hidden states."""

        def _encode(z: Tensor) -> Tensor:
            h = self.embedding(z)
            if self.pos_emb is not None:
                h = h + self.pos_emb(torch.arange(z.shape[-1], device=z.device))
            return h

        return _encode

    @property
    def vocab_size(self) -> int:
        """Actual output vocabulary size from the unembedding weight (may differ from tokenizer.vocab_size due to padding)."""
        return int(self.unembedding.weight.shape[0])

    def residual_stream(self, layer: int) -> Callable[[Tensor], Tensor]:
        """Return a callable computing the residual stream after `layer` blocks.

        layer=0 returns the embedding output (before any transformer block).
        layer=l returns the hidden state after block l-1.

        A single forward pass is shared across all residual_stream calls for the same z.
        """
        assert layer <= self.depth, (
            f"The model has {self.depth} layers, but you requested layer {layer}."
        )
        if layer == 0:
            return CachedF(lambda z: self.encoder(z))

        return CachedF(lambda z: self._forward_cache(z)[layer - 1])

    def layer_gamma(self, layer: int) -> Callable[[Tensor], Tensor]:
        """Return a callable for γ_layer(h_{layer-1}(z)) — the block nonlinearity applied to the preceding residual stream.

        Useful as a jet center representing the contribution of block `layer`.
        """
        assert 0 <= layer < self.depth, (
            f"layer must be in [0, depth={self.depth}), got {layer}."
        )
        h_prev = self.residual_stream(layer)
        gamma = self.layer_fn(layer)
        return CachedF(lambda z: gamma(h_prev(z)))


class _ResBlock(nn.Module):
    """Residual block: h → h + nonlin(h)."""

    def __init__(self, d: int):
        super().__init__()
        self.nonlin = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h: Tensor) -> Tensor:
        return h + self.nonlin(h)


class TwoLayersToyRN(nn.Module):
    """Two-block residual network for illustrating jet expansions.

    Token ids → embedding → two residual blocks → layer norm → unembedding.
    Use toy_two_layer_rn() to get an LM-wrapped version ready for jet_expand_lm.
    """

    VOCAB_SIZE: int = 100

    def __init__(self, d: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(self.VOCAB_SIZE, d)
        self.blocks = nn.ModuleList([_ResBlock(d), _ResBlock(d)])
        self.ln = nn.LayerNorm(d)
        self.unembedding = nn.Linear(d, self.VOCAB_SIZE, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        h = self.embedding(z)
        for block in self.blocks:
            h = block(h)
        return self.unembedding(self.ln(h))


def toy_two_layer_rn(d: int = 32) -> "LM":
    """Create a TwoLayersToyRN wrapped as an LM, ready for jet_expand_lm."""
    model = TwoLayersToyRN(d)
    blocks: list[_ResBlock] = list(model.blocks)  # type: ignore[assignment]

    def layer_fn(idx: int) -> Callable[[Tensor], Tensor]:
        return blocks[idx].nonlin

    return LM(
        model=model,
        tokenizer=None,
        layers=list(model.blocks),
        attentions=[],
        mlps=[b.nonlin for b in blocks],
        pre_attn_norms=[],
        pre_mlp_norms=[],
        ln=model.ln,
        unembedding=model.unembedding,
        embedding=model.embedding,
        pos_emb=None,
        getter=_get_hidden_state,
        layer_fn=layer_fn,
        name="TwoLayersToyRN",
    )


def _causal_mask(seq_len: int, device, dtype) -> Tensor:
    """Additive causal attention mask: 0 for positions that can attend, -inf for future positions."""
    mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    future = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(
        diagonal=1
    )
    return mask.masked_fill(future, float("-inf"))


def _get_hidden_state(x) -> Tensor:
    """Extract the hidden-state tensor from whatever a transformer block returns."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return x[0]
    # HF ModelOutput: this extraction is not super safe... but should work
    return x[0]


def _gpt2_adapter(model, tokenizer) -> LM:
    def layer_fn(idx: int) -> Callable[[Tensor], Tensor]:
        return lambda h: _get_hidden_state(model.transformer.h[idx](h))

    blocks = list(model.transformer.h)
    return LM(
        model=model,
        tokenizer=tokenizer,
        layers=blocks,
        attentions=[b.attn for b in blocks],
        mlps=[b.mlp for b in blocks],
        pre_attn_norms=[b.ln_1 for b in blocks],
        pre_mlp_norms=[b.ln_2 for b in blocks],
        ln=model.transformer.ln_f,
        unembedding=model.lm_head,
        embedding=model.transformer.wte,
        pos_emb=model.transformer.wpe,
        getter=_get_hidden_state,
        layer_fn=layer_fn,
    )


def _llama_adapter(model, tokenizer) -> LM:
    def layer_fn(idx: int) -> Callable[[Tensor], Tensor]:
        def call(h: Tensor) -> Tensor:
            seq_len = h.shape[-2]
            position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
            cos, sin = model.model.rotary_emb(h, position_ids)
            mask = _causal_mask(seq_len, h.device, h.dtype)
            return _get_hidden_state(
                model.model.layers[idx](
                    h, attention_mask=mask, position_embeddings=(cos, sin)
                )
            )

        return call

    blocks = list(model.model.layers)
    return LM(
        model=model,
        tokenizer=tokenizer,
        layers=blocks,
        attentions=[b.self_attn for b in blocks],
        mlps=[b.mlp for b in blocks],
        pre_attn_norms=[b.input_layernorm for b in blocks],
        pre_mlp_norms=[b.post_attention_layernorm for b in blocks],
        ln=model.model.norm,
        unembedding=model.lm_head,
        embedding=model.model.embed_tokens,
        pos_emb=None,  # uses RoPE, computed at model level
        getter=_get_hidden_state,
        layer_fn=layer_fn,
    )


def _gpt_neox_adapter(model, tokenizer) -> LM:
    # GPT-NeoX (Pythia): model.gpt_neox.{embed_in, layers, final_layer_norm}
    def layer_fn(idx: int) -> Callable[[Tensor], Tensor]:
        def call(h: Tensor) -> Tensor:
            seq_len = h.shape[-2]
            position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
            cos, sin = model.gpt_neox.rotary_emb(h, position_ids)
            mask = _causal_mask(seq_len, h.device, h.dtype)
            return _get_hidden_state(
                model.gpt_neox.layers[idx](
                    h, attention_mask=mask, position_embeddings=(cos, sin)
                )
            )

        return call

    blocks = list(model.gpt_neox.layers)
    return LM(
        model=model,
        tokenizer=tokenizer,
        layers=blocks,
        attentions=[b.attention for b in blocks],
        mlps=[b.mlp for b in blocks],
        pre_attn_norms=[b.input_layernorm for b in blocks],
        pre_mlp_norms=[b.post_attention_layernorm for b in blocks],
        ln=model.gpt_neox.final_layer_norm,
        unembedding=model.embed_out,
        embedding=model.gpt_neox.embed_in,
        pos_emb=None,  # uses RoPE per layer
        getter=_get_hidden_state,
        layer_fn=layer_fn,
    )


_ADAPTERS: dict[str, Callable] = {
    "GPT2LMHeadModel": _gpt2_adapter,
    "GPTNeoForCausalLM": _gpt2_adapter,
    "GPTNeoXForCausalLM": _gpt_neox_adapter,
    "LlamaForCausalLM": _llama_adapter,
    "MistralForCausalLM": _llama_adapter,
    "GemmaForCausalLM": _llama_adapter,
}


def from_pretrained(model_id: str, **kwargs) -> LM:
    """Load a model and tokenizer from HuggingFace Hub and wrap it as an LM.

    Args:
        model_id: HuggingFace model identifier (e.g. "gpt2", "meta-llama/Llama-2-7b-hf").
        **kwargs: Forwarded to AutoModelForCausalLM.from_pretrained
            (e.g. torch_dtype, device_map, token).

    Returns:
        An LM adapter ready for jet-based analysis.

    Raises:
        ValueError: If the model architecture is not supported.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    arch = type(model).__name__
    if arch not in _ADAPTERS:
        raise ValueError(
            f"Unsupported architecture '{arch}'. "
            f"Supported: {sorted(_ADAPTERS)}. "
            "You can build an LM adapter manually with jex.models.LM(...)."
        )
    lm = _ADAPTERS[arch](model, tokenizer)
    lm.name = model_id
    return lm
