import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import operator

outdir = '/home/joaobiu/pic/code/evn_final_2/'

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

def sort(x, y):
    x_sort = x

    y_sort = y

    L = sorted(zip(x,y), key=operator.itemgetter(1))

    new_x, new_y = zip(*L)
    data = np.column_stack([new_x, new_y])
    datafile_path = outdir + "sorted_values.txt"
    np.savetxt(datafile_path , data, fmt=['%f','%e'])



x_min, y_min = minimum(x, y)

data = np.column_stack([x_min, y_min])
datafile_path = outdir + "minimum.txt"
np.savetxt(datafile_path , data, fmt=['%f','%e'])

sort(x, y)

print(f"{x_min} {y_min}")
