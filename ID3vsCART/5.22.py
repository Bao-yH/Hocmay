
import math
from collections import Counter
from pprint import pprint

# Du lieu mua may tinh
du_lieu = [
    {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Không"},
    {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
    {"Tuổi": "Trung niên", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
    {"Tuổi": "Trung niên", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
    {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Không"},
    {"Tuổi": "Trẻ", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
    {"Tuổi": "Trung niên", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
    {"Tuổi": "Trung niên", "Thu nhập": "Cao", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"}
]

# Tinh Entropy
def entropy(data, target):
    dem = Counter(mau[target] for mau in data)
    tong = len(data)
    ket_qua = 0

    for so_luong in dem.values():
        p = so_luong / tong
        ket_qua -= p * math.log2(p)

    return ket_qua


# Tim gia tri xuat hien nhieu nhat
def majority_class(data, target):
    dem = Counter(mau[target] for mau in data)
    return dem.most_common(1)[0][0]


# Chia du lieu theo gia tri cua mot thuoc tinh
def split_data(data, attribute):
    cac_nhom = {}

    for mau in data:
        gia_tri = mau[attribute]

        if gia_tri not in cac_nhom:
            cac_nhom[gia_tri] = []

        cac_nhom[gia_tri].append(mau)

    return cac_nhom


# Xay dung cay quyet dinh bang ID3
def build_tree(data, attributes, target):
    # Truong hop 1: Tat ca mau co cung nhan
    cac_nhan = [mau[target] for mau in data]

    if len(set(cac_nhan)) == 1:
        return cac_nhan[0]

    # Truong hop 2: Het thuoc tinh de chia
    if not attributes:
        return majority_class(data, target)

    # Entropy cua tap du lieu hien tai
    entropy_goc = entropy(data, target)

    # Tim thuoc tinh co Information Gain lon nhat
    gain_lon_nhat = -1
    thuoc_tinh_tot_nhat = None

    for attribute in attributes:
        cac_nhom = split_data(data, attribute)
        entropy_sau_chia = 0

        for nhom in cac_nhom.values():
            trong_so = len(nhom) / len(data)
            entropy_sau_chia += trong_so * entropy(nhom, target)

        gain = entropy_goc - entropy_sau_chia

        print(f"Gain({attribute}) = {gain:.4f}")

        if gain > gain_lon_nhat:
            gain_lon_nhat = gain
            thuoc_tinh_tot_nhat = attribute

    # Tao nut goc
    cay = {thuoc_tinh_tot_nhat: {}}

    # Loai bo thuoc tinh da su dung
    thuoc_tinh_con_lai = [
        a for a in attributes if a != thuoc_tinh_tot_nhat
    ]

    # De quy xay dung cac nhanh
    cac_nhom = split_data(data, thuoc_tinh_tot_nhat)

    for gia_tri, nhom in cac_nhom.items():
        cay[thuoc_tinh_tot_nhat][gia_tri] = build_tree(
            nhom, thuoc_tinh_con_lai, target
        )

    return cay


# Du doan nhan cua mot mau moi
def predict(cay, mau_moi, nhan_mac_dinh="Không"):
    # Neu cay la nhan la thi tra ve ket qua
    if not isinstance(cay, dict):
        return cay

    # Lay thuoc tinh tai nut hien tai
    thuoc_tinh = next(iter(cay))
    gia_tri = mau_moi.get(thuoc_tinh)

    # Neu gia tri co trong cay thi di tiep xuong nhanh
    if gia_tri in cay[thuoc_tinh]:
        return predict(
            cay[thuoc_tinh][gia_tri], mau_moi, nhan_mac_dinh
        )

    # Gia tri chua xuat hien trong du lieu huan luyen
    return nhan_mac_dinh


# Chuong trinh chinh
if __name__ == "__main__":
    thuoc_tinh = [
        "Tuổi",
        "Thu nhập",
        "Sinh viên",
        "Đánh giá tín dụng"
    ]

    nhan = "Mua máy tính"

    print("CAC GIA TRI INFORMATION GAIN:")
    cay_quyet_dinh = build_tree(du_lieu, thuoc_tinh, nhan)

    print("\nCAY QUYET DINH ID3:")
    pprint(cay_quyet_dinh, sort_dicts=False)

    # Kiem tra mot vai mau moi
    mau_thu = [
        {
            "Tuổi": "Trẻ",
            "Thu nhập": "Cao",
            "Sinh viên": "Có",
            "Đánh giá tín dụng": "Tốt"
        },
        {
            "Tuổi": "Trung niên",
            "Thu nhập": "Cao",
            "Sinh viên": "Không",
            "Đánh giá tín dụng": "Tốt"
        },
        {
            "Tuổi": "Già",
            "Thu nhập": "Thấp",
            "Sinh viên": "Có",
            "Đánh giá tín dụng": "Xuất sắc"
        }
    ]

    print("\nKET QUA DU DOAN:")
    for i, mau in enumerate(mau_thu, start=1):
        ket_qua = predict(cay_quyet_dinh, mau)
        print(f"Mau {i}: {ket_qua}")
