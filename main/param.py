from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import pi
import numpy as np
import pandas as pd

import os
path = "/home/joaobiu/PIC/bin"
os.chdir(path)
os.system("./montecarlo.exe")