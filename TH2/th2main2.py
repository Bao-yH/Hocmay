import pandas as pd
import numpy as np

# Doc file Excel
df = pd.read_excel("bang_diem_ban_dau.xlsx")

# Cong lai 1 diem
df["DiemCuoiKy_Dung"] = df["DiemCuoiKy_Sai"] + 1

# Xep lai hoc luc
def xep_loai(diem):
    if diem >= 8.5:
        return "Gioi"
    elif diem >= 7.0:
        return "Kha"
    elif diem >= 5.0:
        return "Trung binh"
    else:
        return "Yeu"

df["HocLuc_Dung"] = df["DiemCuoiKy_Dung"].apply(xep_loai)

print(df)