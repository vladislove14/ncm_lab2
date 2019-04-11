import numpy as np

N = 200
eps = 10**(-4)
a = eps**2/16
b = 2
I1 = 0.0000125


def f(x):                                             # Під інтегральна функція
    return (x**(1/2)*(np.exp(x/2)+3))**(-1)


def integral(func, step, a, b):        # Обчислення інтегралу за складеною квадратурною формулою Сімпсона
    n = int((b-a)/step)
    res = 0
    for i in range(n):
        res += func(a + i * step) + 4*func((a + (i+1) * step) - step/2) + func((a + (i+1) * step))
    return step*res/6


def r(h, a, b):
    return (16 * integral(f, h, a, b) - 17 * integral(f, 2*h, a, b) + integral(f, 4*h, a, b))/225


def rynhe(a, b):
    h = 1/2 * (b-a)/N
    while abs(r(h, a, b)) >= eps:
        h /= 2
       # print(abs(r(h, a, b)))
    return h


#s = rynhe(a,b)
#print("Крок h = {0}\nЗначення інтегралу: I = {1}".format(s,integral(f, s, a, b)+I1))
