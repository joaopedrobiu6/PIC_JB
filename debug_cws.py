from simsopt.geo import CurveCWSFourier, SurfaceRZFourier

circular_tokamak = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_circular_tokamak_reference.nc"
w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
filename = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_n3are_R7.75B5.7.nc"

#s = SurfaceRZFourier.from_nphi_ntheta(32, 32, "full torus", 1)
s = SurfaceRZFourier.from_wout(circular_tokamak, range="full torus", ntheta=64, nphi=64)

print(s.mpol, s.ntor)
c = CurveCWSFourier(s.mpol, s.ntor, s.x, 10, 1, s.nfp, s.stellsym)
c.set_dofs([1, 0, 0, 0, 0, 0, 0, 0])

print(c.gamma())
#c.dgamma_by_dcoeff()

auto = c.dgamma_by_dcoeff()
print(auto.shape)
print(auto)