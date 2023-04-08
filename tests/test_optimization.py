import os
import numpy as np
from scipy.optimize import minimize
from simsopt.objectives import Weight
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
from simsopt.geo import (
    CurveLength,
    CurveCurveDistance,
    MeanSquaredCurvature,
    LpCurveCurvature,
    CurveSurfaceDistance,
    CurveCWSFourier
)

OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

LENGTH_WEIGHT = Weight(1e-6)

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.01
CC_WEIGHT = 1000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 10

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# SURFACE INPUT FILES FOR TESTING
circular_tokamak = ("/home/joaobiu/simsopt_curvecws/tests/test_files/wout_circular_tokamak_reference.nc")
w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
filename = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_n3are_R7.75B5.7.nc"

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(
    w7x, range="half period", ntheta=64, nphi=64
)
cws = SurfaceRZFourier.from_nphi_ntheta(255, 256, "half period", 1)

R = s.get_rc(0, 0)
cws.set_dofs([R, 4, 4])

# CREATE A CURVE ON A CWS
# c_cws = CurveCWSFourier(cws.mpol, cws.ntor, cws.x, 300, 1, cws.nfp, cws.stellsym)
# c_cws.set_dofs([1, 0, 0, 0, 0, 0, 0, 0])

ncoils = 4

phi_array = np.linspace(0, 2 * np.pi, ncoils)
print(phi_array)

base_curves = []


for i in range(ncoils - 1):

    curve_cws = CurveCWSFourier(
        mpol=cws.mpol,
        ntor=cws.ntor,
        idofs=cws.x,
        numquadpoints=250,
        order=1,
        nfp=cws.nfp,
        stellsym=cws.stellsym,
    )
    curve_cws.set_dofs([1, 0, 0, 0, 0, phi_array[i], 0, 0])
    base_curves.append(curve_cws)

base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()

coils = []

for i in range(ncoils - 1):
    coils.append(Coil(base_curves[i], base_currents[i]))

bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))


curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {
    "B_N": np.sum(bs.B().reshape((64, 64, 3)) * s.unitnormal(), axis=2)[:, :, None]
}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

Jccdist = CurveCurveDistance(curves, 0.1, num_basecurves=ncoils)
print(Jccdist.shortest_distance())

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]


JF = (
    Jf
    + LENGTH_WEIGHT * sum(Jls)
    + CC_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * sum(Jcs)
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
)




def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((64, 64, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad

f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps * h)
    J2, _ = f(dofs - eps * h)
    print("err", (J1 - J2) / (2 * eps) - dJh)


res = minimize(
    fun,
    dofs,
    jac=True,
    method="L-BFGS-B",
    tol=1e-15,
)

curves_to_vtk(curves, OUT_DIR + f"curves_opt_short")
pointData = {"B_N": np.sum(bs.B().reshape((64, 64, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

curves[0].plot()