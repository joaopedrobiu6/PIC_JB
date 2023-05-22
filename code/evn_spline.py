import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

x, y = Read_Two_Column_File('evn/data_65.txt')

spl = UnivariateSpline(x, y, k=4, s=0)
x_ = np.linspace(0.256, 0.257, 1000)
plt.plot(x_, spl(x_), 'g', lw='1')
list_spl = []
for i in range(0, len(spl(x_))):
    list_spl.append(spl(x_)[i])

min_x = x_[list_spl.index(min(spl(x_)))]
print(f"{min_x} {min(list_spl)}")

plt.show()
