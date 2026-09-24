import numpy as np

# Trong so
w = np.array([[1], [2], [-10]])

# Diem du lieu da them bias
x = np.array([[3], [4], [1]])

# Tinh w^T * x
tich = np.dot(w.T, x)

print("w^T * x =", tich[0, 0])

# Nhan du doan
if tich[0, 0] >= 0:
    y_du_doan = 1
else:
    y_du_doan = -1

print("Nhan du doan =", y_du_doan)

# Nhan thuc te
y = -1

# Kiem tra phan lop
if y_du_doan != y:
    print("Diem du lieu bi phan lop sai")
else:
    print("Diem du lieu duoc phan lop dung")

# Cap nhat Perceptron neu phan lop sai
if y_du_doan != y:
    w_moi = w + y * x
    print("Trong so moi:")
    print(w_moi)