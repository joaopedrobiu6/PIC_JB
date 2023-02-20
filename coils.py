from simsopt.geo import CurveCWS, SurfaceRZFourier
import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure()
ax = fig.add_subplot(projection="3d")


filename_stell = "/home/joaobiu/simsopt_curvecws/tests/test_files/input.LandremanPaul2021_QA"

surf = SurfaceRZFourier.from_vmec_input(filename_stell)

R0 = surf.get_rc(0, 0)

cws = SurfaceRZFourier.from_nphi_ntheta(nphi=128, ntheta=128, range="full torus", nfp=1)
cws.set_dofs([R0, 0.5, 0.6])




cs = CurveCWS(mpol=cws.mpol, ntor=cws.ntor, idofs=cws.x, numquadpoints=250, order=1, nfp=cws.nfp, stellsym=cws.stellsym)

phi_array = np.linspace(0, 2 * np.pi, 11)

base_curves = []

for i in range(11):

    curve_cws = CurveCWS(
        mpol=cws.mpol,
        ntor=cws.ntor,
        idofs=cws.x,
        numquadpoints=100,
        order=1,
        nfp=cws.nfp,
        stellsym=cws.stellsym,
    )
    curve_cws.set_dofs([1, 0, 0, 0, 0, phi_array[i], 0, 0])
    gamma = curve_cws.gamma()
    x = gamma[:, 0]
    y = gamma[:, 1]
    z = gamma[:, 2]
    ax.plot(x, y, z)
    
surf.plot(ax=ax, show=False, alpha=0.6)

cws.plot(ax=ax, show=False, alpha=0.1)
plt.title("Coil Curves and Plasma Surface before optimization")

plt.savefig("coils.png", dpi=400)

plt.show()