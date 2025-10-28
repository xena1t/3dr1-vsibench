"""Python fallback configuration for the VSI-Bench task."""

from __future__ import annotations

from . import utils

VSIBENCH_CONFIG = {
    "dataset_path": "nyu-visionx/VSI-Bench",
    "dataset_kwargs": {
        "token": True,
        "cache_dir": "vsibench",
        "video": True,
    },
    "task": "vsibench",
    "test_split": "test",
    "output_type": "generate_until",
    "process_docs": utils.process_docs,
    "doc_to_visual": utils.vsibench_doc_to_visual,
    "doc_to_text": utils.vsibench_doc_to_text,
    "doc_to_target": "ground_truth",
    "generation_kwargs": {
        "max_new_tokens": 16,
        "temperature": 0,
        "top_p": 1.0,
        "num_beams": 1,
        "do_sample": False,
    },
    "process_results": utils.vsibench_process_results,
    "metric_list": [
        {
            "metric": "vsibench_score",
            "aggregation": utils.vsibench_aggregate_results,
            "higher_is_better": True,
        }
    ],
    "lmms_eval_specific_kwargs": {
        "default": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Please answer the question using a single word or phrase.",
        },
        "gemini_api": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Do not response anything other than a single number!",
        },
        "gpt4v": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Do not response anything other than a single number!",
        },
    },
    "metadata": [
        {"version": 0.0},
    ],
}
