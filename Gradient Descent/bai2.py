import numpy as np



def grad(x):
    return x**2 - 1



def cost(x):
    return (1/3)*x**3 - x


def myGD(x0, eta):
    x = [x0]

    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])

        if abs(grad(x_new)) < 1e-3:
            x.append(x_new)
            break

        x.append(x_new)

    return x, it



x0 = float(input("Nhap x0 = "))
eta = float(input("Nhap eta = "))

x, it = myGD(x0, eta)

print("\nKet qua:")
print("x =", x[-1])
print("g(x) =", cost(x[-1]))
print("So lan lap =", it + 1)