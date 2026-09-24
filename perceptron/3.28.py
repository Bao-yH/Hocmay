import numpy as np

# Trong so
w = np.array([[-2], [1], [0]])

# Diem du lieu da them bias
x = np.array([[2], [3], [1]])

# Nhan thuc te
y = 1

# 1. Tinh w^T * x
tich = np.dot(w.T, x)

print("w^T * x =", tich[0, 0])

# Tinh nhan du doan
if tich[0, 0] >= 0:
    y_du_doan = 1
else:
    y_du_doan = -1

print("Nhan du doan =", y_du_doan)

# Kiem tra phan lop
if y_du_doan != y:
    print("Mau bi phan lop sai")

    # 2. Cap nhat Perceptron
    w = w + y * x

    print("Trong so sau khi cap nhat:")
    print(w)
else:
    print("Mau duoc phan lop dung")

# 3. Tinh lai w^T * x
tich_moi = np.dot(w.T, x)

print("w^T * x sau khi cap nhat =", tich_moi[0, 0])