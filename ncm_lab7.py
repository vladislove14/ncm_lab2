import numpy as np
import matplotlib.pyplot as plt

a = -1.5
b = 1
c = -1
d = 5
k = -1
n = 1
i1 = 0
i2 = 2
N = 10
h = (i2 - i1)/N
x1 = [i1 + i*h for i in range(N + 1)]
u0 = 1 + d
y0 = a + c
z0 = a**2 + 2*b
vu = [u0]


def to_fixed(num_obj, digits=0):
    return f"{num_obj:.{digits}f}"


def f(x):
    return a**3*np.exp(a*x) + k*x*a**2*np.exp(a*x) + 2*k*b*x + (a*np.exp(a*x) + 2*b*x + c)**2 + n*np.exp(a*x) + \
           n*b*x**3 + n*c*x + n*d


def u_(x):
    return np.exp(a*x) + b*x**2 + c*x + d


def q(x, u_0, y_0, z_0):
    return f(x) - k*x*z_0 - y_0**2 - n*u_0


u = [u_(x1[i]) for i in range(N+1)]
print("{0} \t {1} \t {2} \t {3}".format(to_fixed(x1[0], 3), to_fixed(u[0], 15), to_fixed(u0, 15), abs(u[0] - u0)))

for i in range(1, N+1):
    k1 = y0
    l1 = z0
    c1 = q(x1[i-1], u0, y0, z0)
    k2 = (y0 + l1*h/2)
    l2 = (z0 + c1*h/2)
    c2 = q(x1[i-1] + h/2, u0 + k1*h/2, y0 + l1*h/2, z0 + c1*h/2)
    k3 = (y0 + l2*h/2)
    l3 = (z0 + c2*h/2)
    c3 = q(x1[i-1] + h/2, u0 + k2*h/2, y0 + l2*h/2, z0 + c2*h/2)
    k4 = (y0 + l3*h)
    l4 = (z0 + c3*h)
    c4 = q(x1[i-1] + h, u0 + k3*h, y0 + l3*h, z0 + c3*h)
    u0 += h*(k1 + 2*k2 + 2*k3 + k4)/6
    y0 += h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
    z0 += h * (c1 + 2 * c2 + 2 * c3 + c4) / 6
    vu.append(u0)
    print("{0} \t {1} \t {2} \t {3}".format(to_fixed(x1[i], 3), to_fixed(u[i], 15), to_fixed(u0, 15), abs(u[i] - u0)))

x_ = np.arange(i1, i2, 0.0001)
plt.plot(x_, u_(x_), label='u(x) точний розв.', linewidth='3.0')
plt.plot(x1, vu, label='u(x) наближений розв.', linestyle='--', color='red')
plt.legend()
plt.grid()
plt.xlabel('x label')
plt.ylabel('u(x) label')
plt.show()
