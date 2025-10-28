"""Utility bridge that wires 3D-R1 inference into lmms-eval.

The bridge defers model construction to an entry point specified via the
``THREE_DR1_ENTRYPOINT`` environment variable so that downstream users can
plug in their preferred 3D-R1 loader without editing this module.
"""

from __future__ import annotations

import importlib
import os
import re
from typing import Any, Callable, Iterable, List, Optional

import torch

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model: Any | None = None
_inference_fn: Callable[..., Any] | None = None

_DEFAULT_ERROR = (
    "No THREE_DR1_ENTRYPOINT specified and the default 3D-R1 implementation "
    "is not available. Install your 3D-R1 package and set the environment "
    "variable THREE_DR1_ENTRYPOINT to a 'module:function' loader (e.g. "
    "'my_pkg.pipeline:load_pipeline')."
)


def _load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    if ":" not in entrypoint:
        raise ValueError(
            "THREE_DR1_ENTRYPOINT must be of the form 'package.module:function'"
        )
    module_name, func_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Unable to import '{module_name}' from THREE_DR1_ENTRYPOINT='{entrypoint}'. "
            "Ensure your 3D-R1 package is installed and the entry point is correct."
        ) from exc
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(f"Entry point '{entrypoint}' not found") from exc


def _default_loader(**_: Any) -> Any:
    class _StubModel:
        def __init__(self, message: str) -> None:
            self._message = message

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - informational
            raise RuntimeError(self._message)

        def generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - informational
            raise RuntimeError(self._message)

    return _StubModel(_DEFAULT_ERROR)


def _lazy_init() -> None:
    """Initialise the underlying 3D-R1 model lazily on first use."""
    global _model, _inference_fn
    if _model is not None:
        return

    entrypoint = os.environ.get("THREE_DR1_ENTRYPOINT")
    loader = _load_entrypoint(entrypoint) if entrypoint else _default_loader

    loader_kwargs: dict[str, Any] = {}
    model_path = os.environ.get("THREE_DR1_MODEL_PATH")
    if model_path:
        loader_kwargs["model_path"] = model_path

    torch_dtype = os.environ.get("THREE_DR1_DTYPE", "float16").lower()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    loader_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.float16)

    _model = loader(**loader_kwargs)

    inference_attr = os.environ.get("THREE_DR1_INFERENCE_ATTR", "generate")
    if inference_attr == "call":
        if callable(_model):
            _inference_fn = _model  # type: ignore[assignment]
        else:
            raise TypeError("THREE_DR1_INFERENCE_ATTR='call' but model is not callable")
    else:
        try:
            _inference_fn = getattr(_model, inference_attr)
        except AttributeError as exc:
            raise AttributeError(
                f"Model missing inference attribute '{inference_attr}'. "
                "Override via THREE_DR1_INFERENCE_ATTR."
            ) from exc

    if hasattr(_model, "to"):
        _model = _model.to(_device)  # type: ignore[assignment]

    if hasattr(_model, "eval"):
        _model.eval()


def _postprocess_answer(text: Any) -> str:
    """Normalize the raw generation into a compact final answer string."""
    if not isinstance(text, str):
        text = str(text)

    patterns = [
        r"(?:Final\s*answer|Answer)\s*[:\-]\s*([A-D]|Yes|No|True|False)\b",
        r"\b([A-D])\b(?!.*\b[A-D]\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return text.strip()


def _prepare_kwargs(**kwargs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is not None:
            payload[key] = value
    return payload


def run_3dr1(prompt: str, images: List[str], device: Optional[str] = None, **kwargs: Any) -> str:
    """Execute a 3D-R1 forward pass using the configured entrypoint."""
    if not isinstance(images, Iterable):
        raise TypeError("images must be an iterable of file paths")

    _lazy_init()

    chosen_device = device or _device
    extra_kwargs = _prepare_kwargs(**kwargs)

    if _inference_fn is None:
        raise RuntimeError("3D-R1 inference function not initialised")

    result = _inference_fn(prompt=prompt, image_paths=list(images), device=chosen_device, **extra_kwargs)
    return _postprocess_answer(result)
