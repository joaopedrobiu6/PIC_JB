import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
#########################################################################################3
from simsopt.field import Current, Coil
from simsopt.geo import CurveXYZFourier, plot
from simsopt.field import BiotSavart

c = CurveXYZFourier(quadpoints=30, order=1)     #FAZER UMA CURVA
dof_names = c.local_dof_names                   #Obter o nome dos dofs - degrees of freedom
print(f"dof names:\n {dof_names}")

#PARAMETROS INICIAIS

c.x = [1, 1, 1, -2, 1, 0.3, 3, 2, 0.2]          #Coeficientes de Fourier
print(f"dofs:\n {c.x}")

current = Current(10000.)                       #Corrente inicial

coil = Coil(c, current)                         #Juntando uma curve com uma current obtemos um coil
#coil.plot(engine="mayavi")

#Campo de Biot-Savart
field = BiotSavart([coil])
field.set_points(np.array([[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]]))
print(field.B())

