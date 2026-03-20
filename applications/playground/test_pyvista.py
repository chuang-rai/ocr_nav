import numpy as np

from ocr_nav.utils.pyvista_vis_utils import create_plotter, draw_cube

plotter = create_plotter()
cube = draw_cube(center=np.array([0, 0, 0]), size=0.1, color="green")
plotter.add_mesh(cube)
plotter.show()
