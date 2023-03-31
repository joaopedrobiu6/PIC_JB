from simsopt.geo import CurveCWSFourier, CurveRZFourier
from simsopt.geo import SurfaceRZFourier

# CREATE CURVECWSFOURIER
s = SurfaceRZFourier.from_nphi_ntheta(61, 62, "full torus", 1)
R = s.get_rc(0, 0)
s.set_dofs([R, 4, 4])

c_cws = CurveCWSFourier(s.mpol, s.ntor, s.x, 150, 0, s.nfp, s.stellsym)

c_cws.set_dofs([1, 0, 0, 0])

# CREATE CURVERZFOURIER

c_rz = CurveRZFourier(150, 1, 1, 0)
c_rz.set_dofs([0, 0, 1, 0, 0, 0])
print(c_rz.gamma())
c_rz.plot()
c_cws.plot()

print(c_rz.get_dofs())