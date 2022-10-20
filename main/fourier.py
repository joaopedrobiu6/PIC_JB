from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import pi
import numpy as np
import pandas as pd
from scipy.signal import square
from scipy.integrate import quad
from math import *


def fourier_coefs(n):
    fp = lambda x: x * x * np.cos(i * x)
    fi = lambda x: x * x * np.sin(i * x)
    An = []
    Bn = []
    sum = 0
    for i in range(n):
        an = quad(fp, 0, 2 * np.pi)[0]
        if an < 1e-6:
            an = 0
        An.append(an)
        bn = quad(fi, 0, 2 * np.pi)[0]
        if bn < 1e-6:
            bn = 0
        Bn.append(bn)
    return An, Bn

n = 3
A, B = fourier_coefs(n)
print("A: ", A)
print("B: ", B)

t = np.linspace(0, 2 * pi, 5)
C = []

for i in range(n):
    aux = (A[i] * np.cos(i * t) + B[i] * np.sin(i * t))
    C.append(aux)

func_th = input("Rarametrização de Theta: ")
func_ph = input("Parametrização de Phi: ")

theta = eval(func_th)
phi = eval(func_ph)

theta = np.asarray(theta)
phi = np.asarray(phi)

C = np.asarray(C)
print(C[1])

C2 = 0
for i in range(n):
    C2 = C2 + C[i]
C2 = np.asarray(C2)

print(theta + C2)