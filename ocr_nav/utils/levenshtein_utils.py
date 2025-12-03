import numpy as np
from typing import List


def levenshtein_distance(str1: str, str2: str) -> int:
    assert isinstance(str1, str)
    assert isinstance(str2, str)
    len1 = len(str1)
    len2 = len(str2)
    dist = np.zeros((len1 + 1, len2 + 1), dtype=int)

    dist[0, :] = np.arange(len2 + 1)
    dist[:, 0] = np.arange(len1 + 1)
    # str1 as column, str2 as row
    for r in range(1, len1 + 1):
        for c in range(1, len2 + 1):
            if str1[r - 1] == str2[c - 1]:
                substitution = 0
            else:
                substitution = 1
            dist[r][c] = np.min([dist[r - 1, c] + 1, dist[r, c - 1] + 1, dist[r - 1, c - 1] + substitution])
    return dist[len1][len2]


def levenshtein_distance_batch(str_list1: List[str], str_list2: List[str]) -> List[int]:
    pass
