from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import pi
import numpy as np
import pandas as pd


#FUNÇÕES PARA CRIAR UM TORO E UMA CURVA NO TORO
def torus(R, a):
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
    ax.set_xlim3d(-12, 12)
    ax.set_ylim3d(-12, 12)
    ax.set_zlim3d(-12, 12)
    toroid = ax.plot_surface(x, y, z, alpha = 0.1, antialiased = True)
    #RETURN
    return x, y, z, toroid


def curve(R, a):
    #CÁLCULOS
    func_th = input("Rarametrização de Theta: ")
    func_ph = input("Parametrização de Phi: ")
    t = np.linspace(0, 2 * pi, 1000)
    theta = eval(func_th)
    phi = eval(func_ph)
    xc = (R + a * np.cos(theta)) * np.cos(phi)
    yc = (R + a * np.cos(theta)) * np.sin(phi)
    zc = a * np.sin(theta)
    #GRÁFICOS
    curv = plt.plot(xc, yc, zc, color = "orange")
    #RETURN
    return xc, yc, zc, curv


#INPUTS
R = float(input("Raio R: "))
a = float(input("Raio a: "))

x, y, z, toroid = torus(R, a)
xc, yc, zc, curv = curve(R, a)

plt.show()
