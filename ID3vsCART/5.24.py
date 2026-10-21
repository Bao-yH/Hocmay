
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Tao 50 diem du lieu gia lap
np.random.seed(42)

X = np.linspace(0, 10, 50).reshape(-1, 1)

# Cong thuc y = 2*x + sin(x) + nhieu
nhieu = np.random.normal(0, 0.8, 50)
y = 2 * X[:, 0] + np.sin(X[:, 0]) + nhieu

# 2. Chia tap train va test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tao cac diem de ve duong du doan
X_ve = np.linspace(0, 10, 300).reshape(-1, 1)

# 3. Huan luyen cac mo hinh CART hoi quy
do_sau = [2, 5, 10]
cac_mo_hinh = {}

print("KET QUA DANH GIA MO HINH")
print("Do sau | MSE train | MSE test")

for d in do_sau:
    mo_hinh = DecisionTreeRegressor(
        max_depth=d,
        random_state=42
    )

    mo_hinh.fit(X_train, y_train)
    cac_mo_hinh[d] = mo_hinh

    # Du doan tren train va test
    du_doan_train = mo_hinh.predict(X_train)
    du_doan_test = mo_hinh.predict(X_test)

    mse_train = mean_squared_error(y_train, du_doan_train)
    mse_test = mean_squared_error(y_test, du_doan_test)

    print(f"{d:6} | {mse_train:.4f}    | {mse_test:.4f}")

# 4. Ve du lieu goc va duong du doan cua 3 mo hinh
plt.figure(figsize=(12, 7))

plt.scatter(
    X_train[:, 0], y_train,
    color="blue", label="Du lieu train"
)

plt.scatter(
    X_test[:, 0], y_test,
    color="black", marker="x", label="Du lieu test"
)

for d, mo_hinh in cac_mo_hinh.items():
    y_ve = mo_hinh.predict(X_ve)

    plt.plot(
        X_ve[:, 0], y_ve,
        label=f"max_depth = {d}"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("CART hoi quy voi cac do sau khac nhau")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

