from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil

w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
s = SurfaceRZFourier.from_wout(w7x, range="full torus", ntheta=64, nphi=64)

print(s.mpol, s.ntor)
print(s.x.shape)

c = CurveCWSFourier(s.mpol, s.ntor, s.x, 150, 1, s.nfp, s.stellsym)

current = Current(1E5)
coils =  [Coil(c, current)]

bs = BiotSavart(coils)

Jf = SquaredFlux(s, bs)
print(Jf.dJ())

"""
c.set_dofs([1, 0, 0, 0, 2, 1, 4, 0])
auto = c.dgamma_by_dcoeff()
print(auto.shape)
print(auto)
"""