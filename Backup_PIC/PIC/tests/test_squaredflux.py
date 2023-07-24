from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, CurveLength
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import matplotlib.pyplot as plt


#########################################################################################################
#########################################################################################################
################################################DEF######################################################
#########################################################################################################
#########################################################################################################
def showplots(surf1, surf2, curve):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    R = s.get_rc(0, 0)
    surf1.set_dofs([R, 4, 4])

    surf1.plot(ax = ax, show=False, alpha=0.2)
    surf2.plot(ax=ax, show=False, alpha=1)
    curve.plot(ax=ax)


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

current = Current(1E5)
coils =  [Coil(c, current)]

bs = BiotSavart(coils)
Jf = SquaredFlux(s, bs)
print(Jf.J())
c.plot()

showplots(surf, s, c)
