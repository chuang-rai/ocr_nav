import re
import numpy as np
from collections import defaultdict
from typing import List, Tuple


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match)
        else:
            mathes_other.append(a_match)
    return matches, mathes_image, mathes_other


def compute_text_freq(text_tuple_list: List[Tuple[str]]):
    text_freq_dict = defaultdict(int)
    for text_tuple in text_tuple_list:
        text_freq_dict[text_tuple[1]] += 1
    return text_freq_dict


def select_points_in_bbox(pc_image_2d: np.ndarray, bbox: Tuple[int], pc_3d: np.ndarray) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    selected_indices = np.where(
        (pc_image_2d[:, 0] >= x_min)
        & (pc_image_2d[:, 0] <= x_max)
        & (pc_image_2d[:, 1] >= y_min)
        & (pc_image_2d[:, 1] <= y_max)
    )[0]
    selected_points_3d = pc_3d[selected_indices]
    return selected_points_3d
