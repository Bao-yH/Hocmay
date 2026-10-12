
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(2)

# Du lieu dau vao
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00,
               2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25,
               4.50, 4.75, 5.00, 5.50]]).T

# Nhan cua du lieu
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
              1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Khoi tao mo hinh Logistic Regression
model = LogisticRegression()

# Huan luyen mo hinh de tim W
model.fit(X, y)

# Lay trong so W va bias
w1 = model.coef_[0][0]
w0 = model.intercept_[0]

print("Trong so W:")
print(w1)

print("Bias:")
print(w0)

print("Vector W =")
print(np.array([w0, w1]))

# Du doan xac suat
print("Xac suat du doan:")
print(model.predict_proba(X))

# Du doan nhan
print("Nhan du doan:")
print(model.predict(X))
