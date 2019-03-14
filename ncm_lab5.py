import ncm_lab2 as l2
import numpy as np
import matplotlib.pyplot as plt


#region Завдання 1
a = 1
w = 2
b = 1
n = 5
N = 200
interval = [0,3]
h = (interval[1] - interval[0]) / N

f = lambda x, k=0: a*np.cos(w*x) + b*x**2

phi = lambda x, k=0: np.exp(k*x)

r = lambda x, k=0: f(x)-polynom(c, x, phi,n)

r2 = lambda x,k=0: f(x) - polynom(c2,(x - interval[0])*2/(interval[1] - interval[0]) -1,legandra,n)

f2 = lambda t,k=0: f((t+1)*(interval[1] - interval[0])/2 + interval[0])

def bilin_formL2(f,g,params,interv = interval):  #Скалярний добуток в L2[a,b]
    h = (interv[1] - interv[0]) / N
    x = np.matrix(np.zeros((N,1)))
    res = 0
    for i in range(N):
        x[i,0] = interv[0] + i*h
        res += f(x[i,0]-h/2,params[0])*g(x[i,0]-h/2,params[1])
    return res*h/(interv[1]-interv[0])

def legandra(x,n):
    if(n == 1):
        return x
    if(n == 0):
        return 1
    return ((2*n-1)*x/n)*legandra(x,n-1)-((n-1)/n)*legandra(x,n-2)

def syst(n,bilin_form):
    A = np.matrix(np.zeros((n+1,n+1)))
    b = np.matrix(np.zeros((n+1,1)))
    for i in range(n+1):
        for j in range(n+1):
            A[i,j] = bilin_form(phi, phi,[i,j])
        b[i, 0] = bilin_form(f, phi, [0, i])
    return [A,b]

def coef():
    c = np.matrix(np.zeros((n+1,1)))
    for j in range(n+1):
        c[j,0] = bilin_formL2(f2,legandra,[0,j],[-1,1])/bilin_formL2(legandra,legandra,[j,j],[-1,1])
    return c

def polynom(c,x,phi,n):
    res = 0
    for k in range(n+1):
        res += c[k,0]*phi(x,k)
    return res

system = syst(n,bilin_formL2)
c = l2.square_roots(system[0],system[1])[0]
mism = system[0]*c - system[1]   #Нев'язка
c2 = coef()
dev = (bilin_formL2(r,r,[0,0]))**(1/2)
dev2 = (bilin_formL2(r2,r2,[0,0]))**(1/2)
x = np.arange(interval[0], interval[1], 0.001)
print("Завдання 1:\n Число обумосленості матриці системи:\n cond(A) = {0}\n Нев'язка:\n {1}\n Відхилення:\n ||f(x)-Q(x)|| = {2}".format(l2.cond(system[0]),mism,dev))
print(" Відхилення(для випадку побудови наближення по поліномам Лежандри):\n ||f(x)-Q(x)|| = {0}".format(dev2))
plt.gcf().canvas.set_window_title("Завдання 1")
plt.plot(x, f(x), label = 'f(x)', linewidth = 3.0)
plt.plot(x,polynom(c,x,phi,n), label = "Q(x) по exp(kx)", linestyle = '--')
plt.plot(x, polynom(c2,((x - interval[0])*2/(interval[1] - interval[0]) -1),legandra,n), label = 'Q(x) по legandra')
plt.legend()
plt.grid()
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()
#endregion


#region Завдання 2
n = 15
m = 5
h = (interval[1] - interval[0]) / n
x = np.matrix(np.zeros((n + 1, 1)))
y = np.matrix(np.zeros((n + 1, 1)))

for i in range(n + 1):
    x[i, 0] = interval[0] + i * h
    y[i, 0] = f(x[i, 0])

phi = lambda x, k = 0: x**k

def bilin_forml2(f,g,param):
    res = 0
    for i in range(n + 1):
        res += f(x[i, 0], param[0]) * g(x[i, 0], param[1])
    return res/(n+1)

def sum(c,n,k):
    res = 0
    for i in range(n+1):
        res += (polynom(c,x[i,0],phi,k) - y[i,0])**2
    return res/(n-k)

def optim():
    system = syst(0, bilin_forml2)
    c = l2.square_roots(system[0], system[1])[0]
    sigma = (sum(c,n, 0)) ** (1 / 2)
    tmp = sigma + 1
    k = 1
    while (sigma < tmp and k < n/2 ):
        tmp = sigma
        system = syst(k, bilin_forml2)
        c = l2.square_roots(system[0], system[1])[0]
        sigma = (sum(c,n, k)) ** (1 / 2)
        k += 1
    return k - 1

m = optim()

system = syst(m, bilin_forml2)

c = l2.square_roots(system[0], system[1])[0]

mism = system[0]*c - system[1]

r = lambda x,k = 0: f(x) - polynom(c,x,phi,m)

x1 = np.arange(interval[0], interval[1]+h, h)

dev = (bilin_forml2(r,r,[0,0]))**(1/2)
print("Завдання 2:\n Число обумосленості матриці системи:\n cond(A) = {0}\n Нев'язка:\n {1}\n Відхилення:\n ||f(x)-Q(x)|| = {2}".format(l2.cond(system[0]),mism,dev))

plt.gcf().canvas.set_window_title("Завдання 2")
plt.plot(x1, f(x1), label = 'f(x)')
plt.plot(x1, polynom(c,x1,phi,m), label = 'Q(x)',linestyle = "-.",color = "r")
plt.legend()
plt.grid()
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()
#endregion


#region Завдання 3
n = 50
rho = lambda x: 1
h = (interval[1] - interval[0]) / n
x = np.matrix(np.zeros((n + 1, 1)))
y = np.matrix(np.zeros((n + 1, 1)))
vh = np.matrix(np.zeros((n, 1)))
A = np.matrix(np.zeros((n-1, n-1)))
H = np.matrix(np.zeros((n - 1, n+1)))
R = np.matrix(np.zeros((n+1, n+1)))
G = lambda z,i: m1[i,0]*((x[i+1,0] - z)**3)/(6*vh[i+1,0]) + m1[i+1,0]*((z - x[i,0])**3)/(6*vh[i+1,0]) + (mu[i,0] - (m1[i,0]*vh[i+1,0]**2)/6)*(x[i+1,0] - z)/vh[i+1,0] + (mu[i + 1,0] - (m1[i + 1,0]*vh[i+1,0]**2)/6)*(z - x[i,0])/vh[i+1,0]
for i in range(n + 1):
    x[i, 0] = interval[0] + i * h
    y[i, 0] = f(x[i, 0])
for i in range(n):
    vh[i, 0] = x[i + 1, 0] - x[i, 0]
for i in range(A.shape[0]):
    A[i,i] = (vh[i,0] + vh[i+1, 0])/3
    if(i != A.shape[0]-1 ):
        A[i+1,i] = vh[i+1, 0]/6
        A[i, i+1] = vh[i+1, 0]/6
for i in range(H.shape[0]):
    H[i,i] = 1/vh[i, 0]
    if(i != H.shape[1] - 1):
        H[i, i+1] = -((1/vh[i, 0]) + (1/vh[i+1, 0]))
        if(i != H.shape[1] - 2):
            H[i, i+2] = 1/vh[i+1, 0]
for i in range(R.shape[0]):
    R[i,i] = rho(x[i, 0])
m = l2.square_roots(A+H*R*H.T, H*y)[0]
mu = y - R.I*H.T*m
m1 = np.matrix(np.zeros((n + 1, 1)))
for i in range(1,m1.shape[0]-1):
    m1[i,0] = m[i-1,0]

for i in range(n-1):
    var = np.arange(x[i,0],x[i+1,0] + h, h)
    plt.plot(var,G(var,i))
plt.gcf().canvas.set_window_title("Завдання 3")
plt.plot(x1,f(x1))
plt.show()
#endregion