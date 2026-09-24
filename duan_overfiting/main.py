import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

# Doc du lieu
data = pd.read_csv("data.csv")

# Du lieu dau vao
X = data.drop("MedHouseVal", axis=1)

# Gia nha can du doan
y = data["MedHouseVal"]

print("So dong du lieu:", len(data))
print("So cot dau vao:", X.shape[1])

# Chia du lieu thanh train va test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nSo du lieu train:", len(X_train))
print("So du lieu test:", len(X_test))

# Tao mo hinh Linear Regression
model = LinearRegression()

# Cho mo hinh hoc du lieu
model.fit(X_train, y_train)

# Du doan gia nha
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Danh gia mo hinh
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\nKet qua mo hinh:")
print("Train R2:", train_r2)
print("Test R2:", test_r2)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)


# Tao mo hinh Polynomial bac cao
poly_model = make_pipeline(
    PolynomialFeatures(degree=5),
    LinearRegression()
)

# Cho mo hinh hoc
poly_model.fit(X_train, y_train)

# Du doan
y_train_poly = poly_model.predict(X_train)
y_test_poly = poly_model.predict(X_test)

# Danh gia
train_r2_poly = r2_score(y_train, y_train_poly)
test_r2_poly = r2_score(y_test, y_test_poly)

train_mse_poly = mean_squared_error(y_train, y_train_poly)
test_mse_poly = mean_squared_error(y_test, y_test_poly)

print("\nMo hinh Polynomial:")
print("Train R2:", train_r2_poly)
print("Test R2:", test_r2_poly)

print("Train MSE:", train_mse_poly)
print("Test MSE:", test_mse_poly)



# Tao mo hinh Decision Tree qua sau
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

# Cho mo hinh hoc
tree_model.fit(X_train, y_train)

# Du doan
y_train_tree = tree_model.predict(X_train)
y_test_tree = tree_model.predict(X_test)

# Danh gia
train_r2_tree = r2_score(y_train, y_train_tree)
test_r2_tree = r2_score(y_test, y_test_tree)

train_mse_tree = mean_squared_error(y_train, y_train_tree)
test_mse_tree = mean_squared_error(y_test, y_test_tree)

print("\nMo hinh Decision Tree:")
print("Train R2:", train_r2_tree)
print("Test R2:", test_r2_tree)

print("Train MSE:", train_mse_tree)
print("Test MSE:", test_mse_tree)