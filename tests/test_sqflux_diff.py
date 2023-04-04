'''
Compares the SquaredFlux() of a coil made with CurveCWSFourier and a coil made with CurveXYZFourier and its dependance
in the number of quadpoints

Both Curves have the same dimensions and positions.

The surface used is a circular Tokamak and the CWS is a circular torus
'''

from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, CurveLength, CurveXYZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import matplotlib.pyplot as plt
import numpy as np

# SURFACE INPUT FILES FOR TESTING
circular_tokamak = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_circular_tokamak_reference.nc"
w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
filename = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_n3are_R7.75B5.7.nc"

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(circular_tokamak, range="half period", ntheta=256, nphi=256)
surf = SurfaceRZFourier.from_nphi_ntheta(255, 256, "half period", 1)

R = s.get_rc(0, 0)
surf.set_dofs([R, 4, 4])

quad = []
sf_xyz = []
sf_cws = []

# CREATE A CURVE ON A CWS
for i in range(40, 400, 10): 
    c = CurveCWSFourier(surf.mpol, surf.ntor, surf.x, i, 1, surf.nfp, surf.stellsym)
    c.set_dofs([1, 0, 0, 0, 0, 0, 0, 0])

    c_xyz = CurveXYZFourier(i, 10)
    c_xyz.set("xc(0)", R)
    c_xyz.set("xc(1)", 4)
    c_xyz.set("yc(0)", 0)
    c_xyz.set("yc(1)", 0)
    c_xyz.set("zs(1)", 4)


    current = Current(1E5)
    coils_cws =  [Coil(c, current)]
    coils_xys =  [Coil(c_xyz, current)]

    bs_cws = BiotSavart(coils_cws)
    Jf_cws = SquaredFlux(s, bs_cws)
    squaredflux_cws = Jf_cws.J()

    bs_xyz = BiotSavart(coils_xys)
    Jf_xyz = SquaredFlux(s, bs_xyz)
    squaredflux_xyz = Jf_cws.J()

    quad.append(i)
    sf_cws.append(squaredflux_cws)
    sf_xyz.append(squaredflux_xyz)

fig, ax = plt.subplots(1, 3)

ax[0].plot(quad, sf_cws, "-", color="red")
#ax[0].set_xlabel("num quadpoints")
ax[0].set_ylabel("Squared Flux")
ax[0].set_title("CurveCWSFourier")
#plt.savefig("quad_sf_cws.png")
ax[1].plot(quad, sf_xyz, "-", color="blue")
ax[1].set_xlabel("num quadpoints")
ax[1].set_ylabel("Squared Flux")
ax[1].set_title("CurveXYZFourier")
#plt.savefig("quad_sf_cws.png")

diff = np.asarray(sf_xyz) - np.asarray(sf_cws)
ax[2].plot(quad, diff, "-", color="green")
ax[2].set_xlabel("num quadpoints")
ax[2].set_ylabel("Squared Flux Difference")

fig.suptitle("Circular Tokamak")

fig.savefig("diff_tokamak.png")

""" 
#PLOT
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#surf.plot(ax = ax, show=False, alpha=0.2)
s.plot(ax=ax,show=False, alpha=0.2)

#c.plot(ax=ax, show=False)
#c.plot()
c_xyz.plot(ax=ax, alpha=1)

s.plot("plotly", show=True, close=True)
c.plot("mayavi", ax=ax, show=False, close=True)
c_xyz.plot("mayavi", ax=ax, show=True, close=True)
"""

plt.show()