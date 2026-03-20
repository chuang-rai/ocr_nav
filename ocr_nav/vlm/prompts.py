"""Centralized VLM prompt definitions for annotation and scene understanding.

Usage:
    from ocr_nav.vlm.prompts import SCENE_ANNOTATION_PROMPT

    response = qwen3_vl.query(SCENE_ANNOTATION_PROMPT, image_bytes)
"""

import json

# ---------------------------------------------------------------------------
# JSON output schema (kept as a Python dict so it can be reused programmatically
# for validation and easily serialized into prompts)
# ---------------------------------------------------------------------------
SCENE_ANNOTATION_SCHEMA = {
    "description": "A brief description of the scene.",
    "objects": [
        {
            "label": "object label",
            "bounding_box": ["x_min", "y_min", "x_max", "y_max"],
            "attributes": {
                "color": "color of the object",
                "material": "material of the object",
                "function": "function or use of the object",
            },
        },
    ],
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SCENE_ANNOTATION_PROMPT = (
    "Describe the scene in the image based on the highlighted areas in the mask. "
    "Also draw bounding boxes around any objects of interest. "
    "Please normalize the coordinates to be between 0 and 1000. "
    "The output format should be in JSON:\n"
    f"{json.dumps(SCENE_ANNOTATION_SCHEMA, indent=4)}"
)
