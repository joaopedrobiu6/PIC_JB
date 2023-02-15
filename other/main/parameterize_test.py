#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    CurveXYZFourier,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    curves_to_vtk,
    plot,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux, Weight

##################################################################################################
######################################## INPUT PARAMETERS ########################################
##################################################################################################
# Weight on the curve lengths in the objective function.
LENGTH_WEIGHT = Weight(1e-8)
# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.3
CC_WEIGHT = 1000
# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 5.0
CURVATURE_WEIGHT = 1e-6
# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.3
CS_WEIGHT = 10
# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 5
MSC_WEIGHT = 1e-6
filename = "/home/joaobiu/PIC/vmec_equilibria/W7-X/Standard/wout.nc"
OUT_DIR = "./test_output/"
os.makedirs(OUT_DIR, exist_ok=True)
MAXITER = 200

ntheta = 32
nphi = 32
# range = 'full torus', 'field period', 'half period'
s = SurfaceRZFourier.from_wout(filename, range="half period", ntheta=ntheta, nphi=nphi)

ncoils = 4
R0 = s.get_rc(0, 0)  # get_rc(0,0) dá nos o raio grande da superfície
R1 = 1.5
order = 3

# curves and coils
base_curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128
)
base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

# Biot-Savart
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
print("Initial max|⟨B·n⟩ᵢ|:", np.max(np.abs(B_dot_n)))

# CRIAR AS CURVAS INICIAIS, CIRCULARES EM TORNO DO TORO
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {
    "B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[
        :, :, None
    ]
}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]


# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = (
    Jf
    + LENGTH_WEIGHT * sum(Jls)
    + CC_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * sum(Jcs)
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
)
# + CS_WEIGHT * Jcsdist


# OTIMIZAÇÃO

# FUNÇÃO A OTIMIZAR:
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()

    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    # print(outstr)

    return J, grad


# OTIMIZAÇÃO:
dofs = JF.x
res = minimize(fun, dofs, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "maxcor": 300}, tol=1e-15)

curves_to_vtk(curves, OUT_DIR + f"curves_opt_short")

pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}

s.to_vtk(OUT_DIR + "surf_opt_short", extra_data=pointData)

print(f"curves gamma: {coils[0].curve.gamma()}")

# PARAMETRIZAR AS CURVAS OBTIDAS:
"""
newcurves = [c.curve.x for c in coils]
x_coefs = []
y_coefs = []
z_coefs = []

for i in range(0, len(newcurves)):
    coefs = curves[i].x
    x_coefs_aux = []
    y_coefs_aux = []
    z_coefs_aux = []
    print(coefs)

    for j in range(0, len(coefs)):
        if 0 <= j < 2 * order + 1:
            x_coefs_aux.append(coefs[j])
        if 2 * order + 1 <= j < 4 * order + 2:
            y_coefs_aux.append(coefs[j])
        if 4 * order + 2 <= j <= len(coefs):
            z_coefs_aux.append(coefs[j])
    x_coefs.append(x_coefs_aux)
    y_coefs.append(y_coefs_aux)
    z_coefs.append(z_coefs_aux)

#print(f"xx: {x_coefs}\nyy: {y_coefs}\nzz: {z_coefs}")

"""
