import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)

y = np.concatenate(
    (np.ones((1, N)), -1*np.ones((1, N))),
    axis=1
)

# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis=0)


def h(w, x):
    return np.sign(np.dot(w.T, x))


def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


def perceptron(X, y, w_init):

    w = [w_init]

    N = X.shape[1]
    d = X.shape[0]

    mis_points = []

    while True:

        # Tron du lieu
        mix_id = np.random.permutation(N)

        for i in range(N):

            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]

            # Neu phan loai sai
            if h(w[-1], xi)[0] != yi:

                mis_points.append(mix_id[i])

                # Cap nhat trong so
                w_new = w[-1] + yi * xi

                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break

    return (w, mis_points)


# Khoi tao trong so
d = X.shape[0]
w_init = np.random.randn(d, 1)

# Chay Perceptron
(w, m) = perceptron(X, y, w_init)


# =========================
# IN KET QUA
# =========================

w_final = w[-1]

print("Trong so cuoi cung:")
print(w_final)

print("So lan cap nhat trong so:", len(w) - 1)


# =========================
# VE HINH
# =========================

plt.figure(figsize=(8, 6))

# Ve lop 1
plt.scatter(
    X0[0, :],
    X0[1, :],
    color='blue',
    label='Lop 1'
)

# Ve lop -1
plt.scatter(
    X1[0, :],
    X1[1, :],
    color='red',
    label='Lop -1'
)


# Duong phan chia
x1 = np.linspace(0, 6, 100)

# w0 + w1*x1 + w2*x2 = 0
x2 = -(w_final[0, 0] + w_final[1, 0] * x1) / w_final[2, 0]

plt.plot(
    x1,
    x2,
    'k-',
    label='Duong phan chia'
)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Phan loai bang Perceptron')

plt.legend()
plt.grid()

plt.show()