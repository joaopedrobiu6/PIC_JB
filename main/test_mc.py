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


def fourier_coefs_theta(n):
    fp = lambda x: (4 * x**2 + 3) * np.cos(i * x)
    fi = lambda x: (4 * x**2 + 3) * np.sin(i * x)
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


def fourier_coefs_phi(n):
    fp = lambda x: (x) * np.cos(i * x)
    fi = lambda x: (x) * np.sin(i * x)
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


#FUNÇÕES PARA CRIAR UM TORO E UMA CURVA NO TORO
def torus(R, a, lim):
    #CÁLCULOS
    theta = np.linspace(0, 2 * pi, 1000)
    phi = np.linspace(0, 2 * pi, 1000)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + a * np.cos(theta)) * np.cos(phi)
    y = (R + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    #GRÁFICOS
    ax = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    toroid = ax.plot_surface(x, y, z, alpha=0.1, antialiased=True)
    #RETURN
    return x, y, z, toroid


def curve(R, a, n):
    #CÁLCULOS
    func_th = input("Rarametrização de Theta: ")
    func_ph = input("Parametrização de Phi: ")
    t = np.linspace(0, 2 * pi, 1000)

    #SEM SERIE DE FOURIER
    theta = eval(func_th)
    phi = eval(func_ph)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    #SERIE DE FOURIER PARA THETA
    A1, B1 = fourier_coefs_theta(n)
    C1 = []
    for i in range(n):
        aux = (A1[i] * np.cos(i * t) + B1[i] * np.sin(i * t))
        C1.append(aux)
    C1 = np.asarray(C1)
    C21 = 0
    for i in range(n):
        C21 = C21 + C1[i]

    theta = theta + C21

    #SERIE DE FOURIER PARA PHI
    A2, B2 = fourier_coefs_phi(n)
    C2 = []
    for i in range(n):
        aux = (A2[i] * np.cos(i * t) + B2[i] * np.sin(i * t))
        C2.append(aux)
    C2 = np.asarray(C2)
    C22 = 0
    for i in range(n):
        C22 = C22 + C2[i]

    phi = phi + C22

    #PARAMETRIZAÇÂO
    xc = (R + a * np.cos(theta)) * np.cos(phi)
    yc = (R + a * np.cos(theta)) * np.sin(phi)
    zc = a * np.sin(theta)
    #GRÁFICOS
    curv = plt.plot(xc, yc, zc, color="orange")
    #RETURN
    return xc, yc, zc, curv


#INPUTS
print(
    "Okay então, o limite dos eixos é int, os Raios (R, a) são floats.\nOs coeficientes de t nas parametrizações têm de ser inteiros para a curva ser fechada!\n"
)
lim = int(input("Limite dos eixos (abs): "))
R = float(input("Raio R: "))
a = float(input("Raio a: "))
n = int(input("n: "))

x, y, z, toroid = torus(R, a, lim)
xc, yc, zc, curv = curve(R, a, n)

plt.show()
