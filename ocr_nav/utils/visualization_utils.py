import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d
from typing import List, Tuple


def extract_coordinates_and_label(ref_text) -> Tuple[str, List[List[float]]]:

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        raise

    return (label_type, cor_list)


def draw_bounding_boxes(image: Image.Image, refs: List[Tuple[str, str, str]], output_dir: str) -> Image.Image:

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_dir}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30)
                        )

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def draw_cube(center: np.ndarray, size: float = 0.05, color: List[float] = [0, 1, 0]) -> o3d.geometry.TriangleMesh:
    cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    cube.paint_uniform_color(color)
    cube.compute_vertex_normals()
    cube.translate(center - np.array([size / 2, size / 2, size / 2]))
    return cube


def draw_coordinate(origin: np.ndarray, size: float = 0.1) -> o3d.geometry.TriangleMesh:
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axis.translate(origin)
    return axis


def draw_line(
    pos1: np.ndarray,
    pos2: np.ndarray,
    color: List[float] = [1, 0, 0],
) -> o3d.geometry.LineSet:
    # draw line between two points
    points = [pos1, pos2]
    lines = [[0, 1]]
    colors = [color]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_bounding_boxes_on_image_np(
    image: np.ndarray, boxes: list, normalize_max: float | None = None, index: int = "rowcol"
) -> np.ndarray:
    image_with_boxes = image.copy()
    height, width = image.shape[:2]
    for box_info in boxes:
        box = box_info["bounding_box"]
        x_min, y_min, x_max, y_max = box
        if normalize_max is None:
            top_left = (int(x_min), int(y_min))
            bottom_right = (int(x_max), int(y_max))
        else:
            top_left = (int(x_min / normalize_max * width), int(y_min / normalize_max * height))
            bottom_right = (int(x_max / normalize_max * width), int(y_max / normalize_max * height))

        if index == "xy":
            top_left = (int(y_min / normalize_max * width), int(x_min / normalize_max * height))
            bottom_right = (int(y_max / normalize_max * width), int(x_max / normalize_max * height))
        cv2.rectangle(
            image_with_boxes,
            top_left,
            bottom_right,
            color=(0, 255, 0),
            thickness=2,
        )
        label = box_info["label"]
        cv2.putText(
            image_with_boxes,
            label,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return image_with_boxes


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2_with_corners, keypoints2, matches, None, matchColor=(0, 255, 0), flags=2
    )

    return img_matches


def visualize_masks_on_image(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # random list of colors for each mask
    color_list = [
        (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(masks.shape[0])
    ]
    color_mask = np.zeros_like(image)
    for i, mask in enumerate(masks):
        color_mask[mask > 0] = color_list[i % len(color_list)]
    blended_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return blended_image
