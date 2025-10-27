"""lmms-eval adapter that allows VSI-Bench to call into 3D-R1."""

from __future__ import annotations

import json
import os
from pathlib import Path
from subprocess import check_output
from typing import Dict, List

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import MODEL_REGISTRY, register_model

from three_dr1_bridge import run_3dr1

REPO_ROOT = Path(__file__).resolve().parents[3]


def _register_three_d_r1() -> None:
    if "three_d_r1" in MODEL_REGISTRY:
        return

    @register_model("three_d_r1")
    class ThreeDR1(lmms):
        """Minimal wrapper translating lmms-eval requests into 3D-R1 calls."""

        def __init__(self, device: str = "cuda", views: int = 24, **kwargs) -> None:
            self.device = device
            self.views = views
            self.kwargs = kwargs

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

        def _resolve_images(self, request: Dict, index: int) -> List[str]:
            if request.get("videos"):
                video = request["videos"][0]
                video_path = video if isinstance(video, str) else video.get("path")
                if not video_path:
                    raise RuntimeError("three_d_r1: missing video path in request")
                tmp_dir = os.path.join("/tmp", f"mv_{os.getpid()}_{index}")
                return self._video_to_multiview(video_path, tmp_dir)

            if request.get("images"):
                images = request["images"]
                if isinstance(images, list):
                    return images
                raise TypeError("three_d_r1 expects images to be a list of paths")

            return []

        def generate_until(self, requests: List[Dict]) -> List[str]:
            outputs: List[str] = []
            for idx, request in enumerate(requests):
                prompt = request["prompt"]
                images = self._resolve_images(request, idx)
                answer = run_3dr1(
                    prompt=prompt,
                    images=images,
                    device=self.device,
                    **self.kwargs,
                )
                outputs.append(answer)
            return outputs


_register_three_d_r1()
