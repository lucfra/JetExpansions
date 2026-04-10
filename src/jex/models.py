from dataclasses import dataclass
from typing import Any, Callable

from torch import Tensor
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LM:
    """Minimal adapter exposing the components needed for jet expansions.

    Attributes:
        model: The underlying model for forward passes (HuggingFace model, but should accommodate other models as well).
        tokenizer: The associated tokenizer.
        layers: Ordered list of transformer blocks.
        ln: Final layer norm applied before the lm_head.
        lm_head: Unembedding projection (hidden dim → vocab).
        emb: Token embedding table.
        pos_emb: Positional embedding module, or None for models that use
            rotary/relative encodings (e.g. Llama, Mistral, Gemma).
        getter: Extracts the hidden-state tensor from a block's forward output.
                This is needed since different architectures return tuples, tensors, or dataclasses.
                The getter abstracts these differences away.
    """
    model: nn.Module
    tokenizer: Any
    layers: list[nn.Module]
    ln: nn.Module
    lm_head: nn.Linear
    emb: nn.Module
    pos_emb: nn.Module | None
    getter: Callable[[Any], Tensor]

    @property
    def vocab_size(self) -> int:
        """Actual output vocabulary size from the lm_head weight (may differ from tokenizer.vocab_size due to padding)."""
        return int(self.lm_head.weight.shape[0])


def _get_hidden_state(x) -> Tensor:
    """Extract the hidden-state tensor from whatever a transformer block returns."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (tuple, list)):
        return x[0]
    # HF ModelOutput: __iter__ yields values (not keys), first is always hidden state
    return next(iter(x))


def _gpt2_adapter(model, tokenizer) -> LM:
    return LM(
        model=model,
        tokenizer=tokenizer,
        layers=list(model.transformer.h),
        ln=model.transformer.ln_f,
        lm_head=model.lm_head,
        emb=model.transformer.wte,
        pos_emb=model.transformer.wpe,
        getter=_get_hidden_state,
    )


def _llama_adapter(model, tokenizer) -> LM:
    return LM(
        model=model,
        tokenizer=tokenizer,
        layers=list(model.model.layers),
        ln=model.model.norm,
        lm_head=model.lm_head,
        emb=model.model.embed_tokens,
        pos_emb=None,  # uses RoPE, computed inside attention
        getter=_get_hidden_state,
    )


def _gpt_neox_adapter(model, tokenizer) -> LM:
    # GPT-NeoX (Pythia): model.gpt_neox.{embed_in, layers, final_layer_norm}
    return LM(
        model=model,
        tokenizer=tokenizer,
        layers=list(model.gpt_neox.layers),
        ln=model.gpt_neox.final_layer_norm,
        lm_head=model.embed_out,
        emb=model.gpt_neox.embed_in,
        pos_emb=None,  # uses RoPE
        getter=_get_hidden_state,
    )


_ADAPTERS: dict[str, Callable] = {
    "GPT2LMHeadModel": _gpt2_adapter,
    "GPTNeoForCausalLM": _gpt2_adapter,
    "GPTNeoXForCausalLM": _gpt_neox_adapter,
    "LlamaForCausalLM": _llama_adapter,
    "MistralForCausalLM": _llama_adapter,
    "GemmaForCausalLM": _llama_adapter,
    "Gemma2ForCausalLM": _llama_adapter,
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
    return _ADAPTERS[arch](model, tokenizer)
