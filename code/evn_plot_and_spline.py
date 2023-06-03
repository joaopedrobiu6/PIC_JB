import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

outdir = '/home/joaobiu/pic/code/evn_final_1/'

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

x, y = Read_Two_Column_File(outdir + 'data.txt')

def minimum(x, y):
    #y_index = y.index(min(y))
    min_x = x[y.index(min(y))]
    min_y = min(y)
    
    plt.scatter(x , y, color = "#2ec77d", marker = ".")
    plt.scatter(min_x, min_y, label = f"minimum value of J: ({min_x:.5}, {min(y):.3e})")
    plt.legend()
    plt.title("Extend via normal factor variation")
    plt.xlabel("extend_via_normal factor")
    plt.ylabel("JF.J()")
    plt.savefig(outdir + "opt_evn_factor.pdf", dpi = 300)

    return min_x, min_y


x_min, y_min = minimum(x, y)
print(f"{x_min} {y_min}")