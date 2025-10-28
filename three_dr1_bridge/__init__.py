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
_STUB_NOTICE_SHOWN = False

_DEFAULT_ENTRYPOINT = "three_dr1.pipeline:load_pipeline"

_DEFAULT_ERROR = (
    "No THREE_DR1_ENTRYPOINT specified and the official 3D-R1 loader "
    "'three_dr1.pipeline:load_pipeline' could not be imported. Install the "
    "upstream 3D-R1 package (which exposes that entry point), or set the "
    "environment variable THREE_DR1_ENTRYPOINT to your own 'module:function' "
    "loader (e.g. 'my_pkg.pipeline:load_pipeline'). If you only need the stub "
    "predictions for smoke tests, export THREE_DR1_ALLOW_STUB=1 instead."
)

NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


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
            return self._respond()

        def generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - informational
            return self._respond()

        def _respond(self) -> str:
            global _STUB_NOTICE_SHOWN
            if not _STUB_NOTICE_SHOWN:
                print(
                    "[three_d_r1][warning] Running with the built-in stub. "
                    "Configure THREE_DR1_ENTRYPOINT for real predictions.",
                    flush=True,
                )
                _STUB_NOTICE_SHOWN = True
            return (
                f"{_DEFAULT_ERROR} Final answer: 0"
            )

    return _StubModel(_DEFAULT_ERROR)


def _resolve_loader() -> Callable[..., Any]:
    entrypoint = os.environ.get("THREE_DR1_ENTRYPOINT")
    if entrypoint:
        return _load_entrypoint(entrypoint)

    try:
        return _load_entrypoint(_DEFAULT_ENTRYPOINT)
    except (ModuleNotFoundError, AttributeError):
        if os.environ.get("THREE_DR1_ALLOW_STUB", "0") == "1":
            return _default_loader
        raise RuntimeError(_DEFAULT_ERROR)


def _lazy_init() -> None:
    """Initialise the underlying 3D-R1 model lazily on first use."""
    global _model, _inference_fn
    if _model is not None:
        return

    loader = _resolve_loader()

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


def _coerce_numeric(fragment: str) -> Optional[str]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", fragment)
    if match:
        token = match.group(0)
        if re.fullmatch(r"[-+]?\d+", token):
            return str(int(token))
        if re.fullmatch(r"[-+]?\d+\.0+", token):
            return str(int(float(token)))
        return token

    for word, value in NUM_WORDS.items():
        if re.search(rf"\b{word}\b", fragment, flags=re.IGNORECASE):
            return str(value)

    return None


def _postprocess_answer(text: Any) -> str:
    """Normalise the raw generation into a compact final answer string."""
    if not isinstance(text, str):
        text = str(text)

    explicit = re.search(r"(?:Final\s*answer|Answer)\s*[:\-]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if explicit:
        candidate = explicit.group(1).strip()
        number = _coerce_numeric(candidate)
        if number is not None:
            return number
        letter = re.search(r"\b([A-D])\b", candidate, flags=re.IGNORECASE)
        if letter:
            return letter.group(1).upper()
        return candidate

    number = _coerce_numeric(text)
    if number is not None:
        return number

    letter = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", text)
    if letter:
        return letter.group(1)

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

    image_list = list(images)
    debug_enabled = os.environ.get("LMMS_EVAL_DEBUG", "0") == "1"
    if debug_enabled:
        truncated = prompt.replace("\n", " ")[:120]
        print(
            f"[three_d_r1] device={chosen_device} views={len(image_list)} prompt[:120]={truncated}..."
        )

    if _inference_fn is None:
        raise RuntimeError("3D-R1 inference function not initialised")

    result = _inference_fn(
        prompt=prompt,
        image_paths=image_list,
        device=chosen_device,
        **extra_kwargs,
    )
    answer = _postprocess_answer(result)

    if debug_enabled:
        print(f"[three_d_r1] normalised output: {answer}")

    return answer
