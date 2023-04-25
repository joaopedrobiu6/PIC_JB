from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
import matplotlib.pyplot as plt

print("\n\n*************************************************************************************")
print("Creates a CurveCWSFourier object using a SurfaceRZFourier as the CWS; prints the dofs of the curve, their names and their free status")
print("*************************************************************************************\n\n")


surface = SurfaceRZFourier.from_nphi_ntheta(256, 256, "full torus", 1)
curve = CurveCWSFourier(surface.mpol, surface.ntor, surface.x, 250, 1, surface.nfp, surface.stellsym)

curve.set_dofs([1, 10, 10, 10, 1, 5, 0, 0])
print(curve.x)
print(curve.dof_names)
print(curve.fix("x0"))
print(curve.fix("x4"))

print(curve.dofs_free_status)

#curve.plot(close=False)