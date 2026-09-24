# Du an du doan gia nha va giam overfitting

## 1. Gioi thieu

Du an su dung Machine Learning de du doan gia nha dua tren cac thong tin ve ngoi nha.

Du an duoc thuc hien bang Python va thu vien scikit-learn.

Muc tieu cua du an la:
- Xay dung mo hinh du doan gia nha.
- Tim hieu hien tuong overfitting.
- Thu nghiem cach giam overfitting bang Decision Tree.
- Su dung Git de quan ly cac phien ban cua du an.

## 2. Du lieu

Du lieu duoc luu trong file `data.csv`.

Bien can du doan la:

`MedHouseVal`

Cac cot con lai duoc su dung lam du lieu dau vao cho mo hinh.

Du lieu duoc chia thanh:
- 80% du lieu train
- 20% du lieu test

## 3. Mo hinh ban dau

Mo hinh Linear Regression duoc su dung de lam mo hinh co ban.

Ket qua:

- Train R2: 0.6126
- Test R2: 0.5758

Sau do su dung Decision Tree voi do sau khong gioi han.

Ket qua:

- Train R2: 1.0000
- Test R2: 0.6228

Train R2 dat 1.0 trong khi Test R2 thap hon dang ke.

Dieu nay cho thay mo hinh Decision Tree dang bi overfitting.

## 4. Giam overfitting

De giam overfitting, gioi han do sau cua cay:

```python
max_depth=5