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
c1 = 2
c2 = 1
c3 = 1
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

n = 7                                # Кількість базисних функцій
h = (b - a)/(n + 1)
_x = [a + i*h for i in range(1, n + 1)]


def k(x):
    return b1*x**k1 + b2*x**k2 + b3


def k_derivative(x):
    return b1*k1*x**(k1 - 1) + b2*k2*x**(k2 - 1)


def p(x):
    return c1*x**p1 + c2*x**p2 + c3


def q(x):
    return d1*x**q1 + d2*x**q2 + d3


def f(x, c1, c2, c3):
    return -(b1*x**k1 + b2*x**k2 + b3)*(a1*n1*(n1 - 1)*x**(n1 - 2) + a2*n2*(n2 - 1)*x**(n2-2) + a3*n3*(n3-1)*x**(n3-2))\
           -(b1*k1*x**(k1 - 1) + b2*k2*x**(k2 - 1))*(a1*n1*x**(n1 - 1) + a2*n2*x**(n2 - 1) + a3*n3*x**(n3 - 1)) \
           + (c1*x**p1 + c2*x**p2 + c3)*(a1*n1*x**(n1 - 1) + a2*n2*x**(n2 - 1) + a3*n3*x**(n3 - 1)) \
           + (d1 * x**q1 + d2*x**q2 + d3)*(a1*x**n1 + a2*x**n2 + a3*x**n3 + a4)


def u(x):
    return a1*x**n1 + a2*x**n2 + a3*x**n3 + a4


def factorial(i):
    if i == 0:
        return 1
    return i*factorial(i - 1)


def phi(x, i, j):
    if j == i + 2:
        return factorial(i + 2)
    elif j > i + 2:
        return 0
    A = gamma*(b - a)/(gamma*(i + 1) + delta*(b - a)) + b
    return factorial(i + 1) * ((x - a)**(i - j + 1)) * (x - A)/factorial(i - j + 1) + \
           j*factorial(i + 1)*(x - a)**(i - j + 2)/factorial(i - j + 2)


def L(x, i):
    return -(k_derivative(x)*phi(x, i + 1, 1) + k(x)*phi(x, i + 1, 2)) + \
            p(x)*phi(x, i + 1, 1) + q(x)*phi(x, i + 1, 0)


def getSystem1():
    s = np.matrix(np.zeros((n, n)))
    r = np.matrix(np.zeros((n, 1)))
    for j in range(n):
        for i in range(n):
            s[j, i] = L(_x[j], i)
        r[j, 0] = f(_x[j], c1, c2, c3)
    return [s, r]


def u_n(x, c):
    res = 0
    for i in range(n):
        res += c[i, 0] * phi(x, i + 1, 0)
    return res


def G(u, v, i, j):
    return -k(b)*u(b, i + 1, 1)*v(b, j + 1, 0) + k(a)*u(a, i + 1, 1)*v(a, j + 1, 0) +   \
            rh(lambda x: k(x)*u(x, i + 1, 1)*v(x, j + 1, 1) + q(x)*v(x, j + 1, 0)*u(x, i + 1, 0), a, b, 1e-03)[1]


def l(v, i):
    return rh(lambda x: f(x, 0, 0, 0)*v(x, i + 1, 0), a, b, 1e-06)[1]


def getSystem2():
    s = np.matrix(np.zeros((n, n)))
    r = np.matrix(np.zeros((n, 1)))
    for i in range(n):
        for j in range(n):
            s[i, j] = G(phi, phi, i, j)
        r[i, 0] = l(phi, i)
    return [s, r]


x_ = np.arange(a, b + 0.001, 0.001)

sys1 = getSystem1()
sys2 = getSystem2()
vC1 = sys1[0].I*sys1[1]
vC2 = sr(sys2[0], sys2[1])[0]
plt.plot(x_, u_n(x_, vC1), label = 'м. Колокацій', linestyle = ':', linewidth = 2.0)
plt.plot(x_, u_n(x_, vC2), label = 'м. Рітца', linestyle = '-.', color = 'red')
plt.plot(x_, u(x_), label = 'u(x)', color = 'black', linewidth = 1.0, alpha = 0.9)
plt.grid()
plt.legend()
plt.xlabel('x label')
plt.ylabel('u(x) label')
plt.show()
