import pyvista as pv

mesh = pv.Sphere()
plotter = pv.Plotter()

# This adds a box you can resize/rotate directly with your mouse
plotter.add_box_widget(callback=lambda b: print(f"New bounds: {b}"))
plotter.add_mesh(mesh)
plotter.show()
