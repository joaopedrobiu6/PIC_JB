from curves import *

# theta, phi = read_data("/home/joaobiu/PIC", "thetadata.csv", "phidata.csv")

dofs = [1, 1, 1, 1, 1, 0, 0, 0]
lim = 20
R = 15
a = 5

x, y, z, toroid = torus(R, a, lim)
xc, yc, zc, curv = curve(R, a, order=1, dofs=dofs)

plt.show()

