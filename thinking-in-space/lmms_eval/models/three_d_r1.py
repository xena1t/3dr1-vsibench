"""lmms-eval adapter that allows VSI-Bench to call into 3D-R1."""

from __future__ import annotations

import json
import os
from pathlib import Path
from subprocess import check_output
from typing import Any, Dict, Iterable, List, Sequence, Union
from collections.abc import Mapping

from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import MODEL_REGISTRY, register_model

from three_dr1_bridge import run_3dr1

REPO_ROOT = Path(__file__).resolve().parents[3]


class ThreeDR1(lmms):
    """Minimal wrapper translating lmms-eval requests into 3D-R1 calls."""

    def __init__(self, device: str = "cuda", views: int = 24, **kwargs) -> None:
        super().__init__()
        self.device = device
        self.views = views
        self.kwargs = kwargs

    def loglikelihood(self, requests):
        """Not implemented for the 3D-R1 adapter."""
        raise NotImplementedError("three_d_r1 does not support loglikelihood")

    def _video_to_multiview(self, video_path: str, tmp_dir: str) -> List[str]:
        cmd = [
            "python",
            "tools/video_to_multiview.py",
            "--video",
            video_path,
            "--out",
            tmp_dir,
            "--views",
            str(self.views),
        ]
        output = check_output(cmd, cwd=str(REPO_ROOT))
        return json.loads(output.decode("utf-8"))

    def _save_image_like(self, item, index: int, image_idx: int) -> str:
        tmp_path = os.path.join(
            "/tmp",
            f"three_dr1_frame_{os.getpid()}_{index}_{image_idx}.png",
        )
        if hasattr(item, "save"):
            item.save(tmp_path)
            return tmp_path
        raise TypeError(
            "three_d_r1 received an unsupported visual type; expected str, dict with path, or PIL.Image"
        )

    def _materialize_visuals(self, visuals: Iterable, index: int) -> List[str]:
        images: List[str] = []
        video_paths: List[str] = []
        for image_idx, visual in enumerate(visuals):
            if visual is None:
                continue
            if isinstance(visual, str):
                lower = visual.lower()
                if lower.endswith((".mp4", ".mov", ".avi", ".mkv")):
                    video_paths.append(visual)
                else:
                    images.append(visual)
            elif isinstance(visual, dict) and "path" in visual:
                path_value = visual["path"]
                if isinstance(path_value, str):
                    lower = path_value.lower()
                    if lower.endswith((".mp4", ".mov", ".avi", ".mkv")):
                        video_paths.append(path_value)
                    else:
                        images.append(path_value)
                else:
                    raise TypeError("Expected 'path' entry in visual dict to be a string")
            else:
                images.append(self._save_image_like(visual, index, image_idx))

        for vid_idx, video_path in enumerate(video_paths):
            tmp_dir = os.path.join("/tmp", f"mv_{os.getpid()}_{index}_{vid_idx}")
            images.extend(self._video_to_multiview(video_path, tmp_dir))

        return images

    def _is_instance_request(self, request: object) -> bool:
        if isinstance(request, Instance):
            return True
        return all(
            hasattr(request, attr) for attr in ("args", "arguments", "request_type", "metadata")
        )

    def _get_field(self, request: object, key: str, default: Any = None) -> Any:
        if isinstance(request, Mapping):
            return request.get(key, default)

        getter = getattr(request, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                pass

        return getattr(request, key, default)

    def _ensure_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _resolve_from_instance(self, instance: Instance, index: int) -> Dict[str, Union[str, Sequence[str]]]:
        context, _, doc_to_visual, doc_id, task, split = instance.args
        prompt = context
        doc = self.task_dict[task][split][doc_id]
        visuals = doc_to_visual(doc)
        images = self._materialize_visuals(visuals or [], index)
        return {"prompt": prompt, "images": images}

    def _resolve_images(self, request: Union[Dict, Instance], index: int) -> Dict[str, Union[str, Sequence[str]]]:
        if self._is_instance_request(request):
            # cast for type checking; runtime already validated attributes
            request = request  # type: ignore
            return self._resolve_from_instance(request, index)

        prompt = self._get_field(request, "prompt", "")

        visuals: List[Any] = []
        visuals.extend(self._ensure_list(self._get_field(request, "images", [])))
        visuals.extend(self._ensure_list(self._get_field(request, "videos", [])))

        images = self._materialize_visuals(visuals, index)
        return {"prompt": prompt, "images": images}

    def generate_until(self, requests: List[Union[Dict, Instance]]) -> List[str]:
        outputs: List[str] = []
        for idx, request in enumerate(requests):
            payload = self._resolve_images(request, idx)
            prompt = payload.get("prompt", "")
            images = payload.get("images", []) or []
            answer = run_3dr1(
                prompt=prompt,
                images=list(images),
                device=self.device,
                **self.kwargs,
            )
            if os.environ.get("LMMS_EVAL_DEBUG", "0") == "1":
                print(
                    f"[three_d_r1] sample={idx} views={len(images)} answer={answer}"
                )
            outputs.append(answer)
        return outputs


if "three_d_r1" not in MODEL_REGISTRY:
    ThreeDR1 = register_model("three_d_r1")(ThreeDR1)
else:
    ThreeDR1 = MODEL_REGISTRY["three_d_r1"]


__all__ = ["ThreeDR1"]
