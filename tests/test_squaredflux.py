from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, CurveLength, CurveXYZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import matplotlib.pyplot as plt

# SURFACE INPUT FILES FOR TESTING
circular_tokamak = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_circular_tokamak_reference.nc"
w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
filename = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_n3are_R7.75B5.7.nc"

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(circular_tokamak, range="full torus", ntheta=64, nphi=64)
surf = SurfaceRZFourier.from_nphi_ntheta(61, 62, "full torus", 1)

R = s.get_rc(0, 0)
surf.set_dofs([R, 4, 4])

# CREATE A CURVE ON A CWS
c = CurveCWSFourier(surf.mpol, surf.ntor, surf.x, 150, 1, surf.nfp, surf.stellsym)
c.set_dofs([1, 0, 0, 0, 0, 0, 0, 0])

c_xyz = CurveXYZFourier(150, 10)
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

Bcoil_cws = bs_cws.B()
Bcoil_xyz = bs_xyz.B()

print("Bcoil_cws: \n", Bcoil_cws)
print("Bcoil_cws shape: ", Bcoil_cws.shape)
print("####################################################################")
print("Bcoil_xyz: \n", Bcoil_xyz)
print("Bcoil_cws shape: ", Bcoil_cws.shape)
print("####################################################################")
print("Bcoil difference: \n", Bcoil_cws - Bcoil_xyz)



#PLOT
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf.plot(ax = ax, show=False, alpha=0.2)
s.plot(ax=ax, show=False, alpha=1)

c.plot(ax=ax, show=False)
c_xyz.plot(ax=ax)
#coils_cws[0].plot(ax=ax)
#coils_xyz[0].plot(ax=ax)

plt.show()