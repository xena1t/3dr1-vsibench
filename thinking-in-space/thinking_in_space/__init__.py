"""Namespace helpers for the thinking_in_space package."""

from __future__ import annotations

import importlib
import sys

_LMMS_EVAL_MODULE = "lmms_eval"
_ALIAS = __name__ + ".lmms_eval"

if _LMMS_EVAL_MODULE in sys.modules:
    sys.modules[_ALIAS] = sys.modules[_LMMS_EVAL_MODULE]
else:
    lmms_eval = importlib.import_module(_LMMS_EVAL_MODULE)
    sys.modules[_ALIAS] = lmms_eval

__all__ = ["_ALIAS"]
