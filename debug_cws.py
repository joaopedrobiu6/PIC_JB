from simsopt.geo import CurveCWSFourier, SurfaceRZFourier
import matplotlib.pyplot as plt
import numpy as np

circular_tokamak = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_circular_tokamak_reference.nc"
w7x = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
filename = "/home/joaobiu/simsopt_curvecws/tests/test_files/wout_n3are_R7.75B5.7.nc"

def rep(data):
    return data

#s = SurfaceRZFourier.from_nphi_ntheta(32, 32, "full torus", 1)
s = SurfaceRZFourier.from_wout(w7x, range="full torus", ntheta=64, nphi=64)

print(s.mpol, s.ntor)
print(s.x.shape)
c = CurveCWSFourier(s.mpol, s.ntor, s.x, 150, 1, s.nfp, s.stellsym)
c.set_dofs([1, 0, 0, 0, 2, 1, 4, 0])

def plot_deriv():
    cdd = c.gamma()
    print(cdd)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-8, 8)
    ax.set_ylim3d(-8, 8)
    ax.set_zlim3d(-8, 8)
    x = c.gamma()[:, 0]
    y = c.gamma()[:, 1]
    z = c.gamma()[:, 2]
    xt = c.gammadash()[:, 0]
    yt = c.gammadash()[:, 1]
    zt = c.gammadash()[:, 2]

    plt.plot(c.gamma()[:, 0], c.gamma()[:, 1], c.gamma()[:, 2])
    ax.quiver(x, y, z, 0.1 * xt, 0.1 * yt, 0.1 * zt, arrow_length_ratio=0.1, color="r")

    c.plot(plot_derivative=True)

def test_dgamma_by_dcoeff():
    auto = c.dgamma_by_dcoeff()
    print(auto.shape)
    print(auto)

test_dgamma_by_dcoeff()