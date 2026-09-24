import numpy as np



def cost(x):
    return x**2 - 4*x + 5



def grad(x):
    return 2*x - 4



def myGD(x0, eta):
    x = [x0]

    for it in range(4):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)

    return x

x0 = 5
eta = 0.2
x = myGD(x0, eta)


# In ket qua
for i in range(len(x)):
    print("Buoc", i)
    print("x =", x[i])
    print("f(x) =", cost(x[i]))
    print()