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

x_, y_ = Read_Two_Column_File('/home/joaobiu/pic/code/evn_fix/data1.txt')
x = np.array(x_)
y = np.array(y_)
print(x.shape)
print(y.shape)


spl = UnivariateSpline(x, y, k=3, s=0)
x_ = np.linspace(min(x), max(x), 1000)
plt.plot(x_, spl(x_), '--g', lw='1')
list_spl = []
for i in range(0, len(spl(x_))):
    list_spl.append(spl(x_)[i])

min_x = x_[list_spl.index(min(spl(x_)))]
print(f"Minimum:\nFactor: {min_x}\nJF.J(): {min(list_spl)}")

plt.scatter(min_x , min(list_spl), color = "red", marker = "o", label = f"minimum value of J: ({min_x:.5}, {min(list_spl):.3e})")
plt.legend()
plt.title("Extend via normal factor variation")
plt.xlabel("extend_via_normal factor")
plt.ylabel("JF.J()")
plt.savefig("opt_evn_factor.png")

plt.show()     
