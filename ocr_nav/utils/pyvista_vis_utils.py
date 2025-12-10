import pyvista as pv
import numpy as np
from typing import List


def plot_text_at_location(text: str, location: List[float], plotter: pv.Plotter):
    text_actor = pv.Text3D(text, depth=0.01, font_size=12)
    text_actor.translate(location)
    plotter.add_mesh(text_actor, color="black")


def draw_cube(center: np.ndarray, size: float = 0.05, color: str = "green") -> pv.PolyData:
    cube = pv.Cube(center=center, x_length=size, y_length=size, z_length=size)
    return cube


def draw_sphere(center: np.ndarray, radius: float = 0.03, color: str = "red") -> pv.PolyData:
    sphere = pv.Sphere(radius=radius, center=center)
    return sphere


def draw_text(
    text: str,
    position: np.ndarray,
    depth: float = None,
    width: float = None,
    height: float = None,
    normal: np.ndarray = None,
) -> pv.PolyData:
    normal = (normal[0], normal[1], normal[2]) if normal is not None else None
    text_actor = pv.Text3D(
        text, depth=depth, width=width, height=height, center=(position[0], position[1], position[2]), normal=normal
    )
    # text_actor.translate(position)
    return text_actor


def draw_line(
    point1: np.ndarray,
    point2: np.ndarray,
    color: str = "red",
    line_width: float = 2.0,
) -> pv.PolyData:
    line = pv.Line(point1, point2)
    return line


def draw_coordinate(origin: np.ndarray, size: float = 0.1) -> pv.PolyData:
    # Create coordinate axes
    axes = pv.Axes(show_actor=True, line_width=2.0)
    axes_actor = axes.actor
    axes_actor.SetScale(size, size, size)
    axes_actor.SetPosition(origin)
    return axes_actor


def create_plotter() -> pv.Plotter:
    plotter = pv.Plotter()
    plotter.set_background("white")
    return plotter


def draw_point_cloud(
    plotter: pv.Plotter, points: np.ndarray, color: np.ndarray = None, point_size: float = 5.0
) -> pv.PolyData:
    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = color if color is not None else np.array([[0.0, 0.0, 1.0]] * points.shape[0])
    plotter.add_points(point_cloud, scalars="colors", rgb=True, point_size=point_size)
    return plotter
