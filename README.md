# 3D-R1 × VSI-Bench Integration Guide

This repository stitches together the [AIGeeksGroup/3D-R1](https://github.com/AIGeeksGroup/3D-R1) project and the VSI-Bench evaluation suite (via `thinking-in-space/lmms-eval`). Follow the instructions below to reproduce the exact integration used in this workspace and run VSI-Bench with 3D-R1 acting as the model backend.

> **Looking for the upstream README?**
> The official 3D-R1 documentation—including the paper links, citation block, training/data preparation walkthroughs, RL recipes, and case studies—is vendored in this repository at [`3D-R1/README.md`](./3D-R1/README.md). The summary below highlights the most commonly referenced sections so you can quickly navigate the upstream guide while setting up VSI-Bench.

### Upstream 3D-R1 highlights

* **Paper & assets:** [`3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding`](https://arxiv.org/abs/2507.23478) with project site, datasets, and released checkpoints linked at the top of the upstream README.
* **Environment expectations:** CUDA 11.8, Python 3.9.16, and dependencies such as `torch==2.0.1+cu118`, `trimesh`, `Depth-Anything`, and `google-generativeai`. After installing Python packages, build the bundled `pointnet2` and accelerated `giou` extensions from source (see "Environment Setup" in the upstream README).
* **Data preparation:** Download ScanNet-derived point clouds plus language annotations from ScanRefer, Nr3D, ScanQA, and the ScanNet split of 3D-LLM. Scene-30K can be synthesised locally via `script/synthesize_scene30K.sh` or downloaded from Hugging Face.
* **Training scripts:** Supervised fine-tuning and RLHF entry points live under `script/train.generalist.sh` and `script/train.rl.sh`.
* **Case studies:** The README showcases dense captioning, grounding, QA, dialogue, reasoning, and planning demos, along with news updates and a TODO list that tracks upcoming upstream releases (visualisation tutorial, Hugging Face demo, etc.).

Use that upstream README when you need deeper information about the model architecture or training pipeline; keep reading below for the steps specific to VSI-Bench integration.

> **Note**
> The codebase currently wires 3D-R1 through environment-driven entry points defined in `three_dr1_bridge/__init__.py`. Point those hooks at your actual model loader and inference function before running evaluations. Runs abort with a helpful error if no entry point is configured; set `THREE_DR1_ALLOW_STUB=1` only when you deliberately want the placeholder predictions for smoke-tests (they return `0` for every example and will score `0.0`).

---

## 1. Repository Layout

```
/workspace/
├─ 3dr1-vsibench/              # this repo
│  ├─ README.md                # integration playbook (this file)
│  ├─ pyproject.toml           # optional editable install for the helper package
│  ├─ three_dr1_bridge/        # lightweight helpers that expose 3D-R1 inference
│  ├─ thinking-in-space/       # VSI-Bench repo that ships lmms-eval
│  └─ tools/
│     └─ video_to_multiview.py # utility that extracts view frames from videos
```

Install the helper package in editable mode alongside the upstream repos so local code modifications are picked up immediately.

---

## 2. Environment Setup

1. **Create and activate the Conda environment** (Python 3.10 recommended):

   ```bash
   conda create -y -n vsir1 python=3.10
   conda activate vsir1
   ```

2. **Install a CUDA-enabled PyTorch build** (adjust wheel to match your CUDA runtime):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install shared Python dependencies** required by both repos:

   ```bash
   pip install \
       transformers accelerate pillow numpy scipy tqdm decord imageio \
       einops opencv-python pyyaml omegaconf matplotlib loguru pandas datasets
   ```

4. *(Optional)* If your 3D-R1 variant relies on vLLM or PEFT, install them:

   ```bash
   pip install vllm peft
   ```

5. **Install the local repositories in editable mode** so modifications are immediately available:

   ```bash
   pip install -e /workspace/3dr1-vsibench
   pip install -e /workspace/3dr1-vsibench/thinking-in-space
   ```

---

## 3. Validate the GPU Environment

After installing PyTorch, confirm CUDA visibility:

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA build version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU 0 name:', torch.cuda.get_device_name(0))
PY
```

You should also verify `nvidia-smi` works in the same shell. If CUDA reports unavailable, reinstall the correct PyTorch wheel or fix the container runtime before proceeding.

---

## 4. Supply 3D-R1 Entry Points

Edit `three_dr1_bridge/__init__.py` and replace the placeholder logic with the actual 3D-R1 model loading and inference calls. The adapter expects a callable with the signature:

```python
run_3dr1(prompt: str, images: List[str], device: Optional[str] = None, **kwargs) -> str
```

Return a concise answer string (e.g., `"A"`, `"Yes"`). The included `_postprocess_answer` helper converts common patterns into that format, but you can customize it to match the evaluation requirements.

---

## 5. (Optional) Pre-extract Multi-view Frames Manually

VSI-Bench ships videos. The adapter automatically extracts frames on the fly, but you can generate them yourself using the bundled tool:

```bash
python tools/video_to_multiview.py \
    --video /path/to/video.mp4 \
    --out /tmp/scene01/views \
    --views 24 \
    --min_delta 150 \
    --max_side 896 \
    --fps_cap 8
```

Modify parameters to balance coverage vs. VRAM usage:

* `--views`: number of frames to keep (increase for more coverage; decrease for speed).
* `--min_delta`: smaller values allow more similar frames through (useful for static scenes).
* `--max_side`: maximum resolution of the longer image edge.
* `--fps_cap`: upper bound on sampled FPS to skip redundant frames in high-FPS videos.

---

## 6. Running VSI-Bench with 3D-R1

All commands below assume `conda activate vsir1` and `cd /workspace/3dr1-vsibench/thinking-in-space`.

### 6.1 Smoke Test (limited samples)

This run validates the plumbing on a handful of samples to ensure the bridge, extractor, and adapter work together:

```bash
python -m lmms_eval \
  --model three_d_r1 \
  --tasks vsibench \
  --batch_size 1 \
  --output_path logs/$(date +%Y%m%d)/3dr1_vsibench_test \
  --log_samples \
  --limit 5 \
  --model_args "device=cuda,views=24"
```

### 6.2 Full Evaluation Run

Remove the `--limit` flag to evaluate the entire benchmark. Adjust `--batch_size` and `views` if you hit memory limits.

```bash
python -m lmms_eval \
  --model three_d_r1 \
  --tasks vsibench \
  --batch_size 1 \
  --output_path logs/$(date +%Y%m%d)/3dr1_vsibench_full \
  --log_samples \
  --model_args "device=cuda,views=24"
```

The `output_path` directory will contain evaluation metrics and the serialized prompts/answers if `--log_samples` is enabled.

---

### 6.3 Diagnosing `0.0` Results

An overall score of `0.0` almost always indicates that the bridge returned
placeholder text (for example, when the fallback stub is still active) or that
the numeric answers cannot be parsed. Use the following checks to track down the
issue:

1. **Configure a real entry point.** The bridge now raises an error if
   `THREE_DR1_ENTRYPOINT` is missing so that accidental runs do not silently
   produce all-zero scores. Override this behaviour only when intentionally
   setting `THREE_DR1_ALLOW_STUB=1` for smoke tests.
2. **Inspect the logged samples.** When `--log_samples` is enabled, review the
   generated `vsibench.json` file with the helper script:

   ```bash
   python tools/vsibench_debug.py \
       --log logs/$(date +%Y%m%d)/3dr1_vsibench_test/<run_id>/vsibench.json \
       --limit 5 --summary
   ```

   The script prints the question type, model prediction, ground truth, and for
   numeric tasks the absolute error. A prediction distribution summary quickly
   reveals when the model is outputting the same constant for every example.
3. **Enable verbose adapter logs** by re-running with `LMMS_EVAL_DEBUG=1` to
   confirm that prompts, extracted view counts, and normalised answers look as
   expected.

---

## 6.4 Handling Missing Optional Dependencies

The integration purposely keeps heavyweight dependencies optional so that the
Python modules can still be imported in lightweight environments. When a
dependency such as PyTorch or Jinja2 is missing, the CLI now surfaces explicit
runtime guidance instead of failing during import. Install the requirement when
you encounter one of these messages:

```bash
# Example: PyTorch requested by lmms-eval at runtime
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Example: Jinja2 needed for template-based prompt construction
pip install jinja2
```

If you only intend to run quick smoke tests without GPU support, you may set
`THREE_DR1_ALLOW_STUB=1` to bypass the PyTorch requirement temporarily. Real
evaluations should always install the dependencies so the adapter can execute
the actual 3D-R1 model.

---

## 7. Optional: Pose-aware Pipeline with COLMAP

If your 3D-R1 model requires camera poses or intrinsics, integrate COLMAP:

1. Install COLMAP in the execution environment (platform-specific).
2. Extract a denser frame set (e.g., 96 views) to aid structure-from-motion:

   ```bash
   VIDEO=/path/to/video.mp4
   SCENE_DIR=/workspace/work/scene01
   mkdir -p "$SCENE_DIR/frames"

   python tools/video_to_multiview.py \
       --video "$VIDEO" \
       --out "$SCENE_DIR/frames" \
       --views 96 \
       --min_delta 100
   ```

3. Run COLMAP to estimate poses:

   ```bash
   colmap feature_extractor --database_path "$SCENE_DIR/db.db" --image_path "$SCENE_DIR/frames"
   colmap exhaustive_matcher --database_path "$SCENE_DIR/db.db"
   mkdir -p "$SCENE_DIR/sparse"
   colmap mapper --database_path "$SCENE_DIR/db.db" --image_path "$SCENE_DIR/frames" --output_path "$SCENE_DIR/sparse"
   ```

4. Convert COLMAP outputs (`cameras.txt`, `images.txt`) into the pose format that your 3D-R1 inference expects and modify `three_dr1_bridge/__init__.py` accordingly to forward both images and pose metadata.

---

## 8. Troubleshooting Checklist

| Symptom | Suggested Fix |
| --- | --- |
| `ModuleNotFoundError: three_dr1_bridge` | Ensure `pip install -e /workspace/3dr1-vsibench` was executed inside the active environment. |
| `CUDA available: False` | Reinstall a CUDA wheel, check driver/container setup, and confirm `nvidia-smi` output. |
| OOM during evaluation | Reduce `--views` or `--batch_size`, lower `--max_side` in frame extraction, or enable mixed precision in your model. |
| Answers scored incorrectly | Update `_postprocess_answer` or return exact tokens (`A/B/C/D`, `Yes/No`, etc.). |

---

## 9. Next Steps

* Implement the 3D-R1 model initialization and inference logic inside `three_dr1_bridge/__init__.py`.
* Run the smoke test to ensure the adapter and extractor function correctly.
* Scale up to the full VSI-Bench evaluation and analyze the generated logs.

Happy benchmarking!
