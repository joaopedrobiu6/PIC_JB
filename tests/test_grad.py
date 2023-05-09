from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
import numpy as np
import pprint

# CREATE CWS
dofs = [1, 0.1, 0.1] # RBC00, RBC10, ZBS10
quadpoints_phi = np.arange(0, 64, 1)
quadpoints_theta = quadpoints_phi.copy()
surf = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=0, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
surf.set_dofs(dofs)

curve_cws = CurveCWSFourier(surf.mpol, surf.ntor, surf.x, 64, 1, surf.nfp, surf.stellsym)
curve_cws.set_dofs([1, 0, 0.1, 0, 1, 0, 0.1, 0])

'''
file1 = open("python_cws_dgamma.txt", "w")
for i in range(0, len(quadpoints_phi)):
    value = curve_cws.dgamma_by_dcoeff()[i][0]
    file1.write(f"{value}\n")
file1.close()

file1 = open("python_cws_dgammadash.txt", "w")
for i in range(0, len(quadpoints_phi)):
    value = curve_cws.dgammadash_by_dcoeff()[i]
    file1.write(f"{value}\n")
file1.close()

file1 = open("python_cws_dgammadashdash.txt", "w")
for i in range(0, len(quadpoints_phi)):
    value = curve_cws.dgammadashdash_by_dcoeff()[i]
    file1.write(f"{value}\n")
file1.close()
'''

print(curve_cws.dgammadashdash_by_dcoeff()[0])