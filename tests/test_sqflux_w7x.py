'''
Compares the SquaredFlux() of a coil made with CurveCWSFourier and a coil made with CurveXYZFourier
Both Curves have the same dimensions and positions.

The surface used is W7X's and the CWS is a circular torus
'''

from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, CurveLength, CurveXYZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import matplotlib.pyplot as plt

# SURFACE INPUT FILES FOR TESTING
w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(w7x, range= "full torus", ntheta=256, nphi=256)
surf = SurfaceRZFourier.from_nphi_ntheta(255, 256, "full torus", 1)

R = s.get_rc(0, 0)
surf.set_dofs([R, 3, 3])
surf.x

# CREATE A CURVE ON A CWS
c = CurveCWSFourier(surf.mpol, surf.ntor, surf.x, 100, 1, surf.nfp, surf.stellsym)
c.set_dofs([1, 0, 0, 0, 0, 0, 0, 0])

c_xyz = CurveXYZFourier(100, 10)
c_xyz.set("xc(0)", R)
c_xyz.set("xc(1)", 3)
c_xyz.set("yc(0)", 0)
c_xyz.set("yc(1)", 0)
c_xyz.set("zs(1)", 3)


current = Current(1E5)
coils_cws =  [Coil(c, current)]
coils_xys =  [Coil(c_xyz, current)]

bs_cws = BiotSavart(coils_cws)
Jf_cws = SquaredFlux(s, bs_cws)
squaredflux_cws = Jf_cws.J()

bs_xyz = BiotSavart(coils_xys)
Jf_xyz = SquaredFlux(s, bs_xyz)
squaredflux_xyz = Jf_cws.J()

Bcoil_cws = bs_cws.B()
Bcoil_xyz = bs_xyz.B()

'''
print("Bcoil_cws: \n", Bcoil_cws)
print("Bcoil_cws shape: ", Bcoil_cws.shape)
print("####################################################################")
print("Bcoil_xyz: \n", Bcoil_xyz)
print("Bcoil_cws shape: ", Bcoil_cws.shape)
print("####################################################################")
print("Bcoil difference: \n", Bcoil_cws - Bcoil_xyz)
'''

print("Squared Flux:")
print("With CurveXYZFourier: ", Jf_xyz.J())
print("With CurveCWSFourier: ", Jf_cws.J())

if abs(Jf_xyz.J()-Jf_cws.J()) < 1e-15:
    print("test_sqflux_w7x.py - sucess")
else:
    print("test_sqflux_w7x.py - failed")

#PLOT

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#surf.plot(ax = ax, show=False, alpha=0.2)
#s.plot(ax=ax,show=False, alpha=0.2)

#c.plot(ax=ax, show=False)
#c.plot()
#c_xyz.plot(ax=ax, alpha=1)


#s.plot("mayavi", ax=ax, show=False, close=True)
#c.plot("mayavi", ax=ax, show=True, close=True)
#c_xyz.plot("mayavi", ax=ax, show=False, close=True)

#plt.show()