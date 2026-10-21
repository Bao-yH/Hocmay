
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Doc du lieu Iris
iris = load_iris()
X = iris.data
y = iris.target

ten_dac_trung = iris.feature_names
ten_lop = iris.target_names

# 2. Chia du lieu thanh tap train va test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 3. Huấn luyen CART khong gioi han do sau
cay_day_du = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

cay_day_du.fit(X_train, y_train)

# Danh gia mo hinh
du_doan_train = cay_day_du.predict(X_train)
du_doan_test = cay_day_du.predict(X_test)

print("=== CAY CART KHONG GIOI HAN DO SAU ===")
print("Do sau cua cay:", cay_day_du.get_depth())
print("So la:", cay_day_du.get_n_leaves())
print("Accuracy train:",
      accuracy_score(y_train, du_doan_train))
print("Accuracy test:",
      accuracy_score(y_test, du_doan_test))

# 4. Ve cay quyet dinh
plt.figure(figsize=(18, 10))

plot_tree(
    cay_day_du,
    feature_names=ten_dac_trung,
    class_names=ten_lop,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("CART - Iris - Khong gioi han do sau")
plt.tight_layout()
plt.show()

# 5. So sanh max_depth = 2 va 3
print("\n=== SO SANH CAC MO HINH ===")
print("max_depth | Train Accuracy | Test Accuracy")

for do_sau in [None, 2, 3]:
    mo_hinh = DecisionTreeClassifier(
        criterion="gini",
        max_depth=do_sau,
        random_state=42
    )

    mo_hinh.fit(X_train, y_train)

    acc_train = mo_hinh.score(X_train, y_train)
    acc_test = mo_hinh.score(X_test, y_test)

    print(f"{str(do_sau):9} | "
          f"{acc_train:.4f}         | {acc_test:.4f}")

    # Ve cay voi do sau 2 va 3
    if do_sau in [2, 3]:
        plt.figure(figsize=(14, 7))

        plot_tree(
            mo_hinh,
            feature_names=ten_dac_trung,
            class_names=ten_lop,
            filled=True,
            rounded=True,
            fontsize=10
        )

        plt.title(f"CART - Iris - max_depth = {do_sau}")
        plt.tight_layout()
        plt.show()
