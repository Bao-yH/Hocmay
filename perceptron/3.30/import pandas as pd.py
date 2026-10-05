import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Doc du lieu
data = pd.read_csv("data (1).csv")

# Du lieu dau vao
X = data.drop("gian_lan", axis=1)

# Nhan can du doan
y = data["gian_lan"]

# Chia du lieu thanh tap train va test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Chuan hoa du lieu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tao mo hinh Perceptron
model = Perceptron(
    max_iter=1000,
    eta0=0.01,
    random_state=42
)

# Huan luyen mo hinh
model.fit(X_train, y_train)

# Du doan
y_pred = model.predict(X_test)

# Tinh cac chi so
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# In ket qua
print("Ket qua mo hinh Perceptron")
print("-------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)