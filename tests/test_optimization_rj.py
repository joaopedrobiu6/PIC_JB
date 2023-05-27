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
    CurveCWSFourier
)

OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

LENGTH_WEIGHT = Weight(1e-5)

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1e5

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6

# SURFACE INPUT FILES FOR TESTING
wout = "/home/joaobiu/PIC/vmec_equilibria/NCSX/li383_1.4m/wout_li383_1.4m.nc"

MAXITER = 450 
minor_radius_factor_cws = 1.8
ncoils = 5
order = 6 # order of dofs of cws curves
numquadpoints = 200
ntheta = 290
nphi = 290

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(wout, range="half period", ntheta=ntheta, nphi=nphi)
cws = SurfaceRZFourier.from_nphi_ntheta(nphi, ntheta, "half period", s.nfp)

R = s.get_rc(0, 0)
cws.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])

phi_array = np.linspace(0.01, np.pi/s.nfp, ncoils, endpoint=False)

base_curves = []

for i in range(ncoils):
    curve_cws = CurveCWSFourier(
        mpol=cws.mpol,
        ntor=cws.ntor,
        idofs=cws.x,
        numquadpoints=numquadpoints,
        order=order,
        nfp=cws.nfp,
        stellsym=cws.stellsym,
    )
    curve_dofs = np.zeros(len(curve_cws.get_dofs()),)
    curve_dofs[0]=1
    curve_dofs[2*order+2]=0
    curve_dofs[2*order+3]=phi_array[i]
    curve_cws.set_dofs(curve_dofs)
    curve_cws.fix(0)
    curve_cws.fix(2*order+2)
    base_curves.append(curve_cws)
base_currents = [Current(1)*1e5 for i in range(ncoils)]
base_currents[0].fix_all()

coils = []

for i in range(ncoils):
    coils.append(Coil(base_curves[i], base_currents[i]))

bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
cws.to_vtk(OUT_DIR + "cws_init")

Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
print(Jccdist.shortest_distance())

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
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
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J}, Jf={jf}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    print(outstr)
    return J, grad

dofs = np.copy(JF.x)

res = minimize(
    fun,
    dofs,
    jac=True,
    method="L-BFGS-B",
    options={"maxiter": MAXITER, "maxcor": 300},
    tol=1e-15,
)

curves_to_vtk(curves, OUT_DIR + f"curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
cws.to_vtk(OUT_DIR + "cws_opt")