from simsopt.geo import CurveCWSFourier, SurfaceRZFourier, curves_to_vtk
import matplotlib.pyplot as plt
import numpy as np
import os

filename = "../vmec_equilibria/NCSX/li383_1.4m/wout_li383_1.4m.nc"

def image_maker(filename, name1, name2):
    s = SurfaceRZFourier.from_wout(filename, range="full torus", ntheta=64, nphi=64)  # range = 'full torus', 'field period', 'half period'

    sdofs = s.get_dofs()

    cws = CurveCWSFourier(mpol=s.mpol, ntor=s.ntor, idofs=sdofs, quadpoints=250, order=1, nfp=s.nfp, stellsym=s.stellsym)

    phi_array = np.linspace(0, 2 * np.pi, 8)

    fig = plt.figure()
    #Create multiple cws curves and plot them with the surface
    ax = fig.add_subplot(projection="3d")

    for phi in phi_array:
        cws.set_dofs([1, 0, 0, 0, 0, phi, 0, 0])
        gamma = cws.gamma()
        x = gamma[:, 0]
        y = gamma[:, 1]
        z = gamma[:, 2]
        ax.plot(x, y, z)


    s.plot(ax=ax, show=False, alpha=0.2)
    #plt.title("Superimposed Coil Curves and CWS")
    #plt.axis('off')
    
    ax.set_xlabel('X', weight='bold')
    ax.set_ylabel('Y', weight='bold')
    ax.set_zlabel('Z', weight='bold')
    plt.savefig(name1, dpi=400, bbox_inches = 'tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for phi in phi_array:
        cws.set_dofs([1, 0, 0, 0, 0, phi, 0, 0])
        gamma = cws.gamma()
        x = gamma[:, 0]
        y = gamma[:, 1]
        z = gamma[:, 2]

        r = np.sqrt(x * x + y * y)
        r_coil = r
        z_coil = z
        ax.plot(r, z)

        surface_gamma = s.cross_section(phi)
        x = surface_gamma[:, 0]
        y = surface_gamma[:, 1]
        z = surface_gamma[:, 2]
        r = np.sqrt(x * x + y * y)
        ax.plot(r, z, "--")
    ax.plot(r_coil, z_coil, label=f"Curve")
    ax.plot(r, z, "--", label=f"CWS")
    ax.legend()
    #ax.legend()
    #ax.set_title(r"Coil Curve and CWS at different $\phi$ angles")
    ax.set_xlabel("r", weight='bold')
    ax.set_ylabel("z", weight='bold')

    plt.savefig(name2, dpi=400, bbox_inches = 'tight')
    plt.show()

image_maker(filename, "pic_images/plot3d_1.png", "pic_images/plot2d_1.png")