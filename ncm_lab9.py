import matplotlib.pyplot as plt
import numpy as np
from ncm_lab2 import square_roots as sr
from ncm_lab6 import rynhe as rh

# region Parameters
a = 1
b = 3
b1 = 1
b2 = 2
b3 = 0
k1 = 2
k2 = 0
c1 = 0
c2 = 0
c3 = 0
p1 = 1
p2 = 3
d1 = 2
d2 = 1
d3 = 3
q1 = 3
q2 = 2
a1 = 1
a2 = 1
a3 = 1
a4 = 25
n1 = 6
n2 = 3
n3 = 1
alpha = a1*a**n1 + a2*a**n2 + a3*a**n3 + a4
beta = a1*n1*a**(n1 - 1) + a2*n2*a**(n2 - 1) + a3*n3*a**(n3 - 1)
gamma = a1*b**n1 + a2*b**n2 + a3*b**n3 + a4
delta = -(a1*n1*b**(n1 - 1) + a2*n2*b**(n2 - 1) + a3*n3*b**(n3 - 1))
# endregion

eps = 1e-06                          # Точність обчислення інтегралів
n = 100                               # Кількість точок розбиття
h = (b - a)/n                        # Крок дискретизації
_x = [a + i*h for i in range(n + 1)]


def k(x):
    return b1*x**k1 + b2*x**k2 + b3


def q(x):
    return d1*x**q1 + d2*x**q2 + d3


def f(x):
    return -(b1*x**k1 + b2*x**k2 + b3)*(a1*n1*(n1 - 1)*x**(n1 - 2) + a2*n2*(n2 - 1)*x**(n2-2) + a3*n3*(n3-1)*x**(n3-2))\
           -(b1*k1*x**(k1 - 1) + b2*k2*x**(k2 - 1))*(a1*n1*x**(n1 - 1) + a2*n2*x**(n2 - 1) + a3*n3*x**(n3 - 1)) \
           + (c1*x**p1 + c2*x**p2 + c3)*(a1*n1*x**(n1 - 1) + a2*n2*x**(n2 - 1) + a3*n3*x**(n3 - 1)) \
           + (d1 * x**q1 + d2*x**q2 + d3)*(a1*x**n1 + a2*x**n2 + a3*x**n3 + a4)


def u(x):
    return a1*x**n1 + a2*x**n2 + a3*x**n3 + a4


def phi(x, i, j):
    if j == 0:
        if i == 0:
            if _x[0] <= x <= _x[1]:
                return (_x[1] - x)/h
            else:
                return 0
        if i == n:
            if _x[n - 1] <= x <= _x[n]:
                return (x - _x[n - 1])/h
            else:
                return 0
        if _x[i - 1] <= x <= _x[i]:
            return (x - _x[i - 1])/h
        elif _x[i] < x <= _x[i + 1]:
            return (_x[i + 1] - x)/h
        else:
            return 0
    elif j == 1:
        if i == 0:
            if _x[0] <= x <= _x[1]:
                return -1/h
            else:
                return 0
        if i == n:
            if _x[n - 1] <= x <= _x[n]:
                return 1/h
            else:
                return 0
        if _x[i - 1] <= x <= _x[i]:
            return 1/h
        elif _x[i] < x <= _x[i + 1]:
            return -1/h
        else:
            return 0
    else:
        return 0


def getSystem():
    s = np.matrix(np.zeros((n + 1, n + 1)))
    r = np.matrix(np.zeros((n + 1, 1)))
    s[0, 0] = k(a)*beta/alpha + (1/h**2)*rh(lambda x: k(x) + q(x)*(_x[1] - x)**2, _x[0], _x[1], eps)[1]
    s[0, 1] = (1/h**2)*rh(lambda x: -k(x) + q(x)*(_x[1] - x)*(x - _x[0]), _x[0], _x[1], eps)[1]
    s[n, n] = k(b)*delta/gamma + (1/h**2)*rh(lambda x: k(x) + q(x)*(x - _x[n - 1])**2, _x[n - 1], _x[n], eps)[1]
    s[n, n - 1] = (1/h**2)*rh(lambda x: -k(x) + q(x)*(x - _x[n - 1])*(_x[n] - x), _x[n - 1], _x[n], eps)[1]
    r[0, 0] = (1/h)*rh(lambda x: f(x)*(_x[1] - x), _x[0], _x[1], eps)[1]
    r[n, 0] = (1/h)*rh(lambda x: f(x)*(x - _x[n - 1]), _x[n - 1], _x[n], eps)[1]
    for i in range(1, n):
        s[i, i] = (1/h**2)*(rh(lambda x: k(x) + q(x)*(x - _x[i - 1])**2, _x[i - 1], _x[i], eps)[1] +
                            rh(lambda x: k(x) + q(x)*(_x[i + 1] - x)**2, _x[i], _x[i + 1], eps)[1])
        s[i, i - 1] = (1/h**2)*rh(lambda x: -k(x) + q(x)*(x - _x[i - 1])*(_x[i] - x), _x[i - 1], _x[i], eps)[1]
        s[i, i + 1] = (1/h**2)*rh(lambda x: -k(x) + q(x)*(_x[i + 1] - x)*(x - _x[i]), _x[i], _x[i + 1], eps)[1]
        r[i, 0] = (1/h)*(rh(lambda x: f(x)*(x - _x[i - 1]), _x[i - 1], _x[i], eps)[1] +
                         rh(lambda x: f(x)*(_x[i + 1] - x), _x[i], _x[i + 1], eps)[1])
    return [s, r]


def u_n(x, c):
    res = 0
    for i in range(n + 1):
        res += c[i, 0] * phi(x, i, 0)
    return res


def deviation():
    res = 0
    for i in range(n + 1):
        res += (u(_x[i]) - u_[i])**2
    return res**(1/2)/n


x_ = np.arange(a, b + h, h)
sys = getSystem()
vc = sr(sys[0], sys[1])
u_ = [u_n(_x[i], vc[0]) for i in range(n + 1)]
dev = deviation()
plt.title("N = {0}\nDeviation = {1}".format(n, dev))
plt.plot(x_, u_, label = 'МСЕ', linestyle = '-.', color = 'red', linewidth = 2.0)
plt.plot(x_, u(x_), label = 'u(x)', color = 'green', linewidth = 1.0, alpha = 1.0)
plt.legend()
plt.grid()
plt.xlabel('x label')
plt.ylabel('u(x) label')
plt.show()
