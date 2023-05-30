from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import pandas as pd
from scipy.integrate import quad
from math import *
import os

path = "/home/joaobiu/PIC/bin"
os.chdir(path)

file_theta = "thetadata.csv"
file_phi = "phidata.csv"

dataframe_theta = pd.read_csv(file_theta, index_col=False, header=None, sep=";")
dataframe_phi = pd.read_csv(file_phi, index_col=False, header=None, sep=";")

# FUNÇÕES PARA CRIAR UM TORO E UMA CURVA NO TORO
def torus(R, a, lim):
    # CÁLCULOS
    theta = np.linspace(0, 2 * pi, 1000)
    phi = np.linspace(0, 2 * pi, 1000)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + a * np.cos(theta)) * np.cos(phi)
    y = (R + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    # GRÁFICOS
    ax = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    toroid = ax.plot_surface(x, y, z, alpha=0.1, antialiased=True)
    # RETURN
    return x, y, z, toroid


def curve(R, a, dataframe_theta, dataframe_phi):
    # CÁLCULOS
    #func_th = input("Rarametrização de Theta: ")
    #func_ph = input("Parametrização de Phi: ")
    t = np.linspace(0, 2 * pi, 1000)

    # SEM SERIE DE FOURIER
    theta = t
    phi = t
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    A1 = np.asarray(dataframe_theta.iloc[0:, 0])
    B1 = np.asarray(dataframe_theta.iloc[0:, 1])
    n = np.size(A1)
    print(n)
    # SERIE DE FOURIER PARA THETA
    C1 = []
    for i in range(0, n):
        aux = A1[i] * np.cos(i * t) + B1[i] * np.sin(i * t)
        C1.append(aux)
    C1 = np.asarray(C1)
    C21 = 0
    for i in range(n):
        C21 = C21 + C1[i]

    theta = theta + C21

    # SERIE DE FOURIER PARA PHI


    A2 = np.asarray(dataframe_phi.iloc[0:, 0])
    B2 = np.asarray(dataframe_phi.iloc[0:, 1])

    C2 = []
    for i in range(n):
        aux = A2[i] * np.cos(i * t) + B2[i] * np.sin(i * t)
        C2.append(aux)
    C2 = np.asarray(C2)
    C22 = 0
    for i in range(n):
        C22 = C22 + C2[i]

    phi = phi + C22

    # PARAMETRIZAÇÂO
    xc = (R + a * np.cos(theta)) * np.cos(phi)
    yc = (R + a * np.cos(theta)) * np.sin(phi)
    zc = a * np.sin(theta)
    # GRÁFICOS
    curv = plt.plot(xc, yc, zc, color="orange")
    # RETURN
    return xc, yc, zc, curv


# INPUTS
lim = int(input("Limite dos eixos (abs): "))
R = float(input("Raio R: "))
a = float(input("Raio a: "))

x, y, z, toroid = torus(R, a, lim)
xc, yc, zc, curv = curve(R, a, dataframe_theta, dataframe_phi)

plt.show()
