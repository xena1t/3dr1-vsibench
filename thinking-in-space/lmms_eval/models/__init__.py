import importlib
import os
import sys

import hf_transfer
from loguru import logger

# --- Patch sys.modules to unify namespace ---
import thinking_in_space.lmms_eval as _ts_eval
sys.modules["lmms_eval"] = _ts_eval  # ensure registry & imports share namespace

# --- Configure logging ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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
}

# --- Extendable plugin loader ---
if os.environ.get("LMMS_EVAL_PLUGINS"):
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        try:
            m = importlib.import_module(f"{plugin}.models")
            for model_name, model_class in getattr(m, "AVAILABLE_MODELS", {}).items():
                AVAILABLE_MODELS[model_name] = model_class
        except Exception as e:
            logger.warning(f"Failed to import plugin {plugin}: {e}")

# --- Force import of three_d_r1 adapter ---
try:
    from lmms_eval.models import three_d_r1  # ensure 3D-R1 auto-registers
    AVAILABLE_MODELS["three_d_r1"] = "ThreeDR1"
except Exception as e:
    logger.warning(f"Could not import three_d_r1: {e}")

# --- get_model API ---
def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    try:
        module = __import__(f"lmms_eval.models.{model_name}", fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise

# --- Ensure 3D-R1 is discoverable ---
try:
    from lmms_eval.models import three_d_r1 as _three_d_r1
    if "three_d_r1" not in AVAILABLE_MODELS:
        AVAILABLE_MODELS["three_d_r1"] = "ThreeDR1"
    print("[INFO] Registered model three_d_r1 in AVAILABLE_MODELS")
except Exception as e:
    print(f"[WARN] Failed to auto-register three_d_r1: {e}")

