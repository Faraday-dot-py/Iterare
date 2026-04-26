"""Activation capture and residual-stream steering hooks for transformers models.

Designed for GPT-2-family models but works for any model where decoder blocks
expose their output as the first tensor in a tuple.

Usage
-----
with ActivationCapture(model, layers=[6, 12, 18]) as cap:
    outputs = model(**inputs)
vectors = cap.get_mean_vectors()   # dict[layer -> (hidden,)]

with SteeringIntervention(model, layers=[6, 12, 18], alpha=0.5, vector=v) as _:
    outputs = model.generate(**inputs, max_new_tokens=200)
"""

from __future__ import annotations
import torch
from typing import Callable


class ActivationCapture:
    """Capture mean residual-stream activations across a forward pass."""

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self._handles: list = []
        self._store: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    def __enter__(self):
        for layer_idx in self.layers:
            block = _get_block(self.model, layer_idx)
            handle = block.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # mean over sequence dimension → (hidden,)
            self._store[layer_idx].append(hidden.mean(dim=1).squeeze(0).detach().cpu())
        return hook

    def get_mean_vectors(self) -> dict[int, torch.Tensor]:
        """Mean across all captured forward passes for each layer."""
        return {
            l: torch.stack(vecs).mean(dim=0)
            for l, vecs in self._store.items()
            if vecs
        }

    def reset(self):
        self._store = {l: [] for l in self.layers}


class SteeringIntervention:
    """Subtract a steering vector from the residual stream during generation."""

    def __init__(
        self,
        model,
        layers: list[int],
        alpha: float,
        vectors: dict[int, torch.Tensor],
    ):
        """
        Parameters
        ----------
        model:    The causal LM.
        layers:   Which layer outputs to modify.
        alpha:    Scalar multiplier for the subtracted vector.
        vectors:  dict mapping layer_idx -> steering vector (hidden,).
        """
        self.model = model
        self.layers = layers
        self.alpha = alpha
        self.vectors = vectors
        self._handles: list = []

    def __enter__(self):
        for layer_idx in self.layers:
            if layer_idx not in self.vectors:
                continue
            block = _get_block(self.model, layer_idx)
            handle = block.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int) -> Callable:
        vec = self.vectors[layer_idx].to(dtype=torch.float32)

        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            device = hidden.device
            v = vec.to(device)
            # broadcast subtract: hidden is (batch, seq, hidden)
            hidden = hidden - self.alpha * v.unsqueeze(0).unsqueeze(0)
            if is_tuple:
                return (hidden,) + output[1:]
            return hidden
        return hook


def _get_block(model, layer_idx: int):
    """Return the i-th decoder block for GPT-2 or Llama-family models."""
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    # Llama / Qwen / Mistral
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # Fallback: assume model.layers
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise AttributeError(f"Cannot find decoder blocks in {type(model).__name__}")


def num_layers(model) -> int:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise AttributeError(f"Cannot determine num layers for {type(model).__name__}")
