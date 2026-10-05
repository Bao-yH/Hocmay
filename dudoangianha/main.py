import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Doc du lieu
data = pd.read_csv("e:/onthipython/dudoangianha/data.csv")

print("Du lieu:")
print(data)

# Chon dau vao X va ket qua y
X = data[["dien_tich", "so_phong", "tuoi_nha"]]
y = data["gia"]

# Chia du lieu thanh tap huan luyen va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tao mo hinh hoi quy tuyen tinh
model = LinearRegression()

# Huan luyen mo hinh
model.fit(X_train, y_train)

# Du doan
y_pred = model.predict(X_test)

print("\nGia thuc te:")
print(y_test.values)

print("\nGia du doan:")
print(y_pred)

# Danh gia mo hinh
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE =", mse)
print("R2 =", r2)

# Nhap thong tin nha moi
dien_tich = float(input("\nNhap dien tich nha (m2): "))
so_phong = int(input("Nhap so phong: "))
tuoi_nha = int(input("Nhap tuoi nha: "))

# Du doan gia nha
nha_moi = [[dien_tich, so_phong, tuoi_nha]]
gia_du_doan = model.predict(nha_moi)

print("Gia nha du doan:", gia_du_doan[0], "trieu dong")