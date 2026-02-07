import os
import pyvista as pv
import numpy as np
from ocr_nav.utils.io_utils import load_ply_point_cloud
from ocr_nav.utils.pyvista_vis_utils import create_plotter, PointCloudBoxSelector


def main():
    file_path = "/control_suite/temp/lab_downstairs_test/glim_map.ply"

    point_cloud = load_ply_point_cloud(file_path)
    plotter = create_plotter()
    selector = PointCloudBoxSelector(point_cloud, plotter)


if __name__ == "__main__":
    main()
