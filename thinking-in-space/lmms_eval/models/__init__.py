import importlib
import logging
import os
import sys

try:
    import hf_transfer  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hf_transfer = None

try:
    from loguru import logger  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    logger = logging.getLogger("lmms_eval.models")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    _HAS_LOGURU = False
else:
    _HAS_LOGURU = True

try:
    from loguru import logger  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    logger = logging.getLogger("lmms_eval.models")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    _HAS_LOGURU = False
else:
    _HAS_LOGURU = True

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

if _HAS_LOGURU:
    logger.remove()
    logger.add(sys.stdout, level="WARNING")

# --- Baseline models ---
AVAILABLE_MODELS = {
    "batch_gpt4": "BatchGPT4",
    "claude": "Claude",
    "from_log": "FromLog",
    "fuyu": "Fuyu",
    "gemini_api": "GeminiAPI",
    "gpt4v": "GPT4V",
    "idefics2": "Idefics2",
    "instructblip": "InstructBLIP",
    "internvl": "InternVLChat",
    "internvl2": "InternVL2",
    "llama_vid": "LLaMAVid",
    "llava": "Llava",
    "llava_hf": "LlavaHf",
    "llava_onevision": "Llava_OneVision",
    "llava_sglang": "LlavaSglang",
    "llava_vid": "LlavaVid",
    "longva": "LongVA",
    "mantis": "Mantis",
    "minicpm_v": "MiniCPM_V",
    "mplug_owl_video": "mplug_Owl",
    "phi3v": "Phi3v",
    "qwen_vl": "Qwen_VL",
    "qwen_vl_api": "Qwen_VL_API",
    "reka": "Reka",
    "srt_api": "SRT_API",
    "tinyllava": "TinyLlava",
    "videoChatGPT": "VideoChatGPT",
    "video_llava": "VideoLLaVA",
    "vila": "VILA",
    "xcomposer2_4KHD": "XComposer2_4KHD",
    "xcomposer2d5": "XComposer2D5",
    "qwen2vl": "Qwen2VL",
    "qwen3vl": "Qwen3VL",
    "three_d_r1": "ThreeDR1",
}

# --- Extendable plugin loader ---
if os.environ.get("LMMS_EVAL_PLUGINS"):
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        try:
            m = importlib.import_module(f"{plugin}.models")
            for model_name, model_class in getattr(m, "AVAILABLE_MODELS", {}).items():
                AVAILABLE_MODELS[model_name] = model_class
            print(f"[INFO] Loaded plugin models from: {plugin}")
        except Exception as e:
            logger.warning(f"Failed to import plugin {plugin}: {e}")

# --- Force import of three_d_r1 adapter (ensures registration decorator runs) ---
try:
    from lmms_eval.models import three_d_r1  # triggers @register_model("three_d_r1")
    AVAILABLE_MODELS["three_d_r1"] = "ThreeDR1"
    print("[INFO] three_d_r1 imported successfully and added to AVAILABLE_MODELS")
except Exception as e:
    logger.warning(f"Could not import three_d_r1: {e}")

from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


from lmms_eval.api.registry import MODEL_REGISTRY


def get_model(model_name):
    """
    Resolve a model class by name.
    Checks both AVAILABLE_MODELS and the global MODEL_REGISTRY (for plugin-registered models).
    """
    # 1️⃣ Direct lookup in AVAILABLE_MODELS
    if model_name in AVAILABLE_MODELS:
        model_class = AVAILABLE_MODELS[model_name]
        try:
            module = __import__(f"lmms_eval.models.{model_name}", fromlist=[model_class])
            return getattr(module, model_class)
        except Exception as e:
            logger.error(f"Failed to import {model_class} from {model_name}: {e}")
            raise

    # 2️⃣ Fallback: check MODEL_REGISTRY (populated by @register_model)
    if model_name in MODEL_REGISTRY:
        print(f"[INFO] Found {model_name} in MODEL_REGISTRY")
        return MODEL_REGISTRY[model_name]

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS or MODEL_REGISTRY.")

    # 2️⃣ Fallback to MODEL_REGISTRY
    if model_name in MODEL_REGISTRY:
        print(f"[INFO] Found {model_name} in MODEL_REGISTRY")
        return MODEL_REGISTRY[model_name]

if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError as e:
                logger.debug(f"Failed to import {model_class} from {model_name}: {e}")

from lmms_eval.models import three_d_r1  # ensure 3D-R1 auto-registers on import
