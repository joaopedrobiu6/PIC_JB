import matplotlib.pyplot as plt
from math import pi
import numpy as np
import pandas as pd
from math import *
import os

def read_data(path_, th_, phi_):
    os.chdir(path_)

    dataframe_theta = pd.read_csv(th_, index_col=False, header=None, sep=";")
    dataframe_phi = pd.read_csv(phi_, index_col=False, header=None, sep=";")

    return dataframe_theta, dataframe_phi

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

def curve_fromfile(R, a, dataframe_theta, dataframe_phi):
    # CÁLCULOS
    t = np.linspace(0, 2 * pi, 1000)

    # SEM SERIE DE FOURIER
    theta = t
    phi = t
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    A1 = np.asarray(dataframe_theta.iloc[0:, 0])
    B1 = np.asarray(dataframe_theta.iloc[0:, 1])

    n = np.size(A1)
    
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

def curve(R, a, order, dofs):
    t = np.linspace(0, 2 * pi, 1000)

    theta_c = []
    theta_s = []
    phi_c = []
    phi_s = []
    counter = 0

     
    
    theta_l = dofs[counter]
    for i in range(0, order+1):
        theta_c.append(dofs[counter+1])
        counter = counter + 1
         
    for i in range(0, order):
        theta_s.append(dofs[counter+1])
        counter = counter + 1
         
    phi_l = dofs[counter+1]
    counter = counter + 1
    for i in range(0, order+1):
        phi_c.append(dofs[counter+1])
        counter = counter + 1
         
    for i in range(0, order):
        phi_s.append(dofs[counter+1])
        counter = counter + 1
         
    aux_th = 0

    for i in range(0, order):
        aux_th = aux_th + theta_c[i] * np.cos(i * t)
    for i in range(1, order + 1):
        aux_th = aux_th + theta_s[i - 1] * np.sin(i * t)
    
    aux_th = aux_th + theta_l

    print(aux_th)

    aux_ph = 0

    for i in range(0, order):
        aux_ph = aux_ph + phi_c[i] * np.cos(i * t)
    for i in range(1, order + 1):
        aux_ph = aux_ph + phi_s[i - 1] * np.sin(i * t)

    aux_ph = aux_ph + phi_l

    print(aux_ph)

    # PARAMETRIZAÇÂO
    xc = (R + a * np.cos(aux_ph)) * np.cos(aux_th)
    yc = (R + a * np.cos(aux_ph)) * np.sin(aux_th)
    zc = a * np.sin(aux_ph)
    # GRÁFICOS
    curv = plt.plot(xc, yc, zc, color="orange")
    # RETURN
    return xc, yc, zc, curv