"""
Test if CurveCWSFourier object is being created.
"""

from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
import matplotlib.pyplot as plt



surface = SurfaceRZFourier.from_nphi_ntheta(256, 256, "full torus", 1)
curve = CurveCWSFourier(surface.mpol, surface.ntor, surface.x, 250, 1, surface.nfp, surface.stellsym)

curve.set_dofs([1, 10, 10, 10, 1, 5, 0, 0])

curve.x
curve.dof_names
curve.fix("x0")
curve.fix("x4")

curve.dofs_free_status

print("test_curve.py - sucess")

#curve.plot(close=False)