import numpy as np
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
