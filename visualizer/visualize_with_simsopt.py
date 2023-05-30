from simsopt.geo import SurfaceRZFourier, CurveXYZFourier
import numpy as np
import matplotlib.pyplot as plt



mpol=1
ntor=0
num_dofs = 2*(mpol+1)*(2*ntor+1) - ntor - (ntor+1)
ntheta = 500
nphi = 500
wout = '/home/joaobiu/simsopt_curvecws/examples/3_Advanced/input.axiTorus_nfp3_QA_final'
# CREATE FLUX SURFACE (BOUNDARY)
s = SurfaceRZFourier.from_vmec_input(wout, range="half period", ntheta=ntheta, nphi=nphi)
s_full = SurfaceRZFourier.from_vmec_input(wout, range="full torus", ntheta=ntheta, nphi=int(nphi*2*s.nfp))
# CREATE COIL WINDING SURFACE
cws_full = SurfaceRZFourier.from_nphi_ntheta(nphi, ntheta, "half period", s.nfp)
surface = SurfaceRZFourier.from_nphi_ntheta(int(nphi*2*s.nfp), ntheta, "full torus", s.nfp)
R = s.get_rc(0, 0)
minor_radius_factor_cws = 1 + 0.2595/s.get_zs(1, 0)
surface.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])
cws_full.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])
fig = plt.figure()
ax = plt.axes(projection='3d')
lim=1.5
ax.set_xlim3d(-lim, lim)
ax.set_ylim3d(-lim, lim)
ax.set_zlim3d(-lim, lim)
x_s = surface.gamma()[:,:,0]
y_s = surface.gamma()[:,:,1]
z_s = surface.gamma()[:,:,2]
ax.plot_surface(x_s, y_s, z_s, antialiased = True,  alpha=0.7)


c_xyz = CurveXYZFourier(50, 1)
c_xyz.set("xc(0)", R)
c_xyz.set("xc(1)", 0.7)
c_xyz.set("yc(0)", 0)
c_xyz.set("yc(1)", 0) 
c_xyz.set("zs(1)", 0.7)

ax.plot(c_xyz.gamma()[:, 0], c_xyz.gamma()[:, 1], c_xyz.gamma()[:, 2], color = "red")

print(f"surface dofs: {surface.x}")
print(f"curve dofs: {c_xyz.x}")


plt.savefig("curve_surface.pdf", bbox_inches='tight')
print("\n*******\nPDF SAVED\n*******")
