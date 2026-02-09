import cv2
import numpy as np


def get_largest_region(binary_map: np.ndarray) -> np.ndarray:
    """Get the largest disconnected island region in the binary map.

    Args:
        binary_map (np.ndarray): The binary map.

    Returns:
        np.ndarray: the largest region in the binary map.
    """
    # Threshold it so it becomes binary
    input = (binary_map > 0).astype(np.uint8)
    output = cv2.connectedComponentsWithStats(input, 8, cv2.CV_8UC1)
    # the first region is always the background (0), so we find the largest from the rest
    areas = output[2][1:, cv2.CC_STAT_AREA]
    print(areas)
    print(np.argsort(areas))
    id = np.argsort(areas)[::-1][0] + 1
    print(id)
    return output[1] == id


canvas = np.zeros((600, 600), dtype=np.uint8)

cv2.circle(canvas, (150, 150), 150, 255, -1)
cv2.rectangle(canvas, (400, 400), (500, 500), 255, -1)

cv2.imshow("Canvas", canvas)
cv2.waitKey()

# output = cv2.connectedComponentsWithStats(canvas, 8, cv2.CV_8UC1)
# print(output[0])
# areas = output[2][1:, cv2.CC_STAT_AREA]
# ids = np.argsort(areas)[::-1]
# print(ids)
# print(areas)
mask = get_largest_region(canvas)
print(np.sum(mask))
