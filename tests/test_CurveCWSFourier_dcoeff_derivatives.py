"""
Test if CurveCWSFourier gradients are being computed:
dgamma_by_dcoeff, dgammadash_by_dcoeff, dgammadashdash_by_dcoeff.
"""

from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil

circular_tokamak = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_circular_tokamak_reference.nc"

s = SurfaceRZFourier.from_wout(circular_tokamak, range="half period", ntheta=64, nphi=64)
surf = SurfaceRZFourier.from_nphi_ntheta(255, 256, "half period", 1)

R = s.get_rc(0, 0)
surf.set_dofs([R, 4, 4])

quad = 250
curve = CurveCWSFourier(surf.mpol, surf.ntor, surf.x, quad, 0, surf.nfp, surf.stellsym)
curve.set_dofs([1, 0, 0, 0])


current = Current(1E5)
coils_cws =  [Coil(curve, current)]
bs_cws = BiotSavart(coils_cws)
bs_cws.set_points(s.gamma().reshape((-1, 3)))
Jf_cws = SquaredFlux(s, bs_cws)
squaredflux_cws = Jf_cws.J()
dj = Jf_cws.dJ()

grad1 = curve.dgamma_by_dcoeff()


grad2 = curve.dgammadash_by_dcoeff()


grad3 = curve.dgammadashdash_by_dcoeff()

if grad1.shape != (quad, 3, len(curve.x)) and grad2.shape != (quad, 3, len(curve.x)) and grad3.shape != (quad, 3, len(curve.x)):
    print("test_dcoeff.py - failed")
else:
    print("test_dcoeff.py - sucess")

