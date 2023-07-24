import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, CurveXYZFourier

###################################################
###################################################
##################CurveCWSFourier##################
###################################################
###################################################

# CWS
s = SurfaceRZFourier.from_nphi_ntheta(150, 150, "full torus", 1)
R = s.get_rc(0, 0)
s.set_dofs([R, 1, 1])

# CWS CURVE
c_cws = CurveCWSFourier(s.mpol, s.ntor, s.x, 15, 0, s.nfp, s.stellsym)
c_cws.set_dofs([1, 0, 0, 0]) # [th_l, th_c0, phi_l, phi_c0]

###################################################
###################################################
##################CurveXYZFourier##################
###################################################
###################################################

c_xyz = CurveXYZFourier(15, 10)
c_xyz.set("xc(0)", R)
c_xyz.set("xc(1)", 1)
c_xyz.set("yc(0)", 0)
c_xyz.set("yc(1)", 0)
c_xyz.set("zs(1)", 1)


gamma_diff = c_xyz.gamma()-c_cws.gamma()
gammadash_diff = c_xyz.gammadash()-c_cws.gammadash()
gammadashdash_diff = c_xyz.gammadashdash()-c_cws.gammadashdash()
#dgamma_by_dcoeff_diff = c_xyz.dgamma_by_dcoeff() - c_cws.dgamma_by_dcoeff()

print(sum(abs(gamma_diff)))
print(sum(abs(gammadash_diff)))
print(sum(abs(gammadashdash_diff)))
#print(sum(abs(dgamma_by_dcoeff_diff)))

# PLOTS
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

c_cws.plot(ax=ax,show=False, plot_derivative=True)
c_xyz.plot(ax=ax, show=True, plot_derivative=False)
