from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import pi
import numpy as np
import pandas as pd

def torus(R, a):
    theta = np.linspace(0, 2*pi, 50)
    phi = np.linspace(0, 2*pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    x = (R+a*np.cos(theta))*np.cos(phi)
    y = (R+a*np.cos(theta))*np.sin(phi)
    z = a*np.sin(theta)
    return x, y, z

def curve_parameterization():
    theta = input("fn: ")
    phi = input("fn: ")
    return theta, phi

th, ph = curve_parameterization()

h = np.linspace(0, 2*np.pi, 50)
eval(th, h)