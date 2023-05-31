import os
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveCWSFourier, ArclengthVariation
)
import matplotlib.pyplot as plt

OUT_DIR = "./output_cws_evn/"
os.makedirs(OUT_DIR, exist_ok=True)
    
# Threshold and weight for the maximum length of each individual coil:
LENGTH_THRESHOLD = 20
#LENGTH_WEIGHT = 0.1
LENGTH_WEIGHT = 1e-8

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
#CC_WEIGHT = 1000
CC_WEIGHT = 100

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 60    
#CURVATURE_WEIGHT = 0.1
CURVATURE_WEIGHT = 1e-5

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 200
#MSC_WEIGHT = 0.01
MSC_WEIGHT = 1e-10

ARCLENGTH_WEIGHT = 3e-8
LENGTH_CON_WEIGHT = 0.1

# SURFACE INPUT FILES FOR TESTING
wout = '/home/joaobiu/simsopt_curvecws/examples/3_Advanced/input.axiTorus_nfp3_QA_final'

ncoils = 4
order = 10 # order of dofs of cws curves
quadpoints = 300 #13 * order
ntheta = 50
nphi = 42


# CREATE FLUX SURFACE
s = SurfaceRZFourier.from_vmec_input(wout, range="half period", ntheta=ntheta, nphi=nphi)
s_full = SurfaceRZFourier.from_vmec_input(wout, range="full torus", ntheta=ntheta, nphi=int(nphi*2*s.nfp))
# CREATE COIL WINDING SURFACE SURFACE


def optimize_extend_via_normal_factor(factor):
    cws = SurfaceRZFourier.from_vmec_input(wout, range="half period", ntheta=ntheta, nphi=nphi)
    cws_full = SurfaceRZFourier.from_vmec_input(wout, range="full torus", ntheta=ntheta, nphi=int(nphi*2*s.nfp))
    cws.extend_via_normal(factor)
    cws_full.extend_via_normal(factor)

    # CREATE CURVES + COILS     
    base_curves = []
    for i in range(ncoils):
        curve_cws = CurveCWSFourier(
            mpol=cws.mpol,
            ntor=cws.ntor,
            idofs=cws.x,
            quadpoints=quadpoints,
            order=order,
            nfp=cws.nfp,
            stellsym=cws.stellsym,
        )
        angle = (i + 0.5)*(2 * np.pi)/((2) * s.nfp * ncoils)
        curve_dofs = np.zeros(len(curve_cws.get_dofs()),)
        curve_dofs[0] = 1
        curve_dofs[2*order+2] = 0
        curve_dofs[2*order+3] = angle
        curve_cws.set_dofs(curve_dofs)
        curve_cws.fix(0)
        curve_cws.fix(2*order+2)
        base_curves.append(curve_cws)
    base_currents = [Current(1)*1e5 for _ in range(ncoils)]
    #base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

    bs = BiotSavart(coils)

    bs.set_points(s_full.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]

    bs.set_points(s.gamma().reshape((-1, 3)))
    return bs, base_curves, curves, cws_full

factor = np.arange(0.220, 0.260, 0.005)
j_list = []

MAXITER = 50

for n in factor:
    bs, base_curves, curves, cws_full= optimize_extend_via_normal_factor(n)
    Jf = SquaredFlux(s, bs, local=True)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]
    J_LENGTH = LENGTH_WEIGHT * sum(Jls)
    J_CC = CC_WEIGHT * Jccdist
    J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
    J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, f="max") for i, J in enumerate(Jmscs))
    J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
    J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD, f="max") for i in range(len(base_curves))])
    JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_ALS + J_MSC + J_LENGTH

    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
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
    print(f"{n:.4f}: {JF.J():.3e}")
    j_list.append(JF.J())

plt.scatter(factor, j_list)
plt.savefig("ext_via_normal.png")