from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
import numpy as np

# CREATE CWS
dofs = [1, 0.1, 0.1] # RBC00, RBC10, ZBS10
quadpoints_phi = np.arange(0, 64, 1)
quadpoints_theta = quadpoints_phi.copy()
surf = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=0, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
surf.set_dofs(dofs)

curve_cws = CurveCWSFourier(surf.mpol, surf.ntor, surf.x, 64, 1, surf.nfp, surf.stellsym)
curve_cws.set_dofs([1, 0, 0.1, 0, 1, 0, 0.1, 0])

print(curve_cws.dgammadash_by_dcoeff()[0])