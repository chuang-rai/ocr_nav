import pyvista as pv
import open3d as o3d
import numpy as np
from typing import List, Optional


def plot_text_at_location(text: str, location: List[float], plotter: pv.Plotter):
    text_actor = pv.Text3D(text, depth=0.01)
    text_actor.translate(location)
    plotter.add_mesh(text_actor, color="black")


def draw_cube(center: np.ndarray, size: float = 0.05, color: str = "green") -> pv.PolyData:
    cube = pv.Cube(center=center, x_length=size, y_length=size, z_length=size)
    return cube


def draw_sphere(center: np.ndarray, radius: float = 0.03) -> pv.PolyData:
    sphere = pv.Sphere(radius=radius, center=center)
    return sphere


def draw_text(
    text: str,
    position: np.ndarray,
    depth: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    normal: Optional[np.ndarray] = None,
) -> pv.PolyData:
    normal_list = [normal[0], normal[1], normal[2]] if normal is not None else None
    text_actor = pv.Text3D(
        text,
        depth=depth,
        width=width,
        height=height,
        center=(position[0], position[1], position[2]),
        normal=normal_list,  # type: ignore
    )
    # text_actor.translate(position)
    return text_actor


def draw_line(
    point1: np.ndarray,
    point2: np.ndarray,
) -> pv.PolyData:
    line = pv.Line(point1, point2)
    return line


def draw_coordinate(origin: np.ndarray, size: float = 0.1) -> pv.Actor:
    # Create coordinate axes
    axes = pv.Axes(show_actor=True, line_width=2.0)
    axes_actor = axes.actor
    assert axes_actor is not None
    axes_actor.SetScale(size, size, size)
    axes_actor.SetPosition(origin.tolist())
    return axes_actor


def create_plotter() -> pv.Plotter:
    plotter = pv.Plotter()
    plotter.set_background((1.0, 1.0, 1.0))  # type: ignore
    return plotter


def draw_point_cloud(
    plotter: pv.Plotter, points: np.ndarray, color: Optional[np.ndarray] = None, point_size: float = 5.0
) -> pv.Plotter:
    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = color if color is not None else np.array([[0.0, 0.0, 1.0]] * points.shape[0])
    plotter.add_points(point_cloud, scalars="colors", rgb=True, point_size=point_size)
    return plotter


def convert_open3d_mesh_to_pyvista(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(faces)
    print(faces.shape)
    # PyVista expects faces in a specific format: [N, v0, v1, v2, N, v0, v1, v2, ...]
    faces_pv = np.hstack((np.full((faces.shape[0], 1), 3), faces)).flatten()
    pv_mesh = pv.PolyData(vertices, faces_pv)
    return pv_mesh
