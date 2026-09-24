import time

# === Ham doc file do thi ===
def doc_do_thi(duong_dan):
    do_thi = {}
    with open(duong_dan, 'r') as f:
        for dong in f:
            phan = dong.strip().split()
            if len(phan) != 3:
                continue
            dinh_u, dinh_v, chi_phi = phan
            chi_phi = float(chi_phi)
            if dinh_u not in do_thi:
                do_thi[dinh_u] = {}
            do_thi[dinh_u][dinh_v] = chi_phi
            if dinh_v not in do_thi:
                do_thi[dinh_v] = {}
    return do_thi


# === Ham doc file uoc luong (heuristic) ===
def doc_uoc_luong(duong_dan):
    h = {}
    with open(duong_dan, 'r') as f:
        for dong in f:
            phan = dong.strip().split()
            if len(phan) != 2:
                continue
            dinh, gia_tri = phan
            h[dinh] = float(gia_tri)
    return h


# === Thuat toan leo doi ===
def leo_doi(do_thi, uoc_luong, bat_dau, dich):
    hien_tai = bat_dau
    duong_di = [hien_tai]
    tong_chi_phi = 0
    cac_buoc = []
    thoi_gian_bat_dau = time.time()

    while hien_tai != dich:
        lang_gieng = do_thi[hien_tai]
        if not lang_gieng:
            break

        dinh_tot_nhat = None
        h_tot_nhat = uoc_luong[hien_tai]
        chi_phi_tot_nhat = 0

        for n, c in lang_gieng.items():
            h_gia_tri = uoc_luong.get(n, float('inf'))
            cac_buoc.append((hien_tai, n, h_gia_tri))
            if h_gia_tri < h_tot_nhat:
                h_tot_nhat = h_gia_tri
                dinh_tot_nhat = n
                chi_phi_tot_nhat = c

        if dinh_tot_nhat is None:
            break

        hien_tai = dinh_tot_nhat
        duong_di.append(hien_tai)
        tong_chi_phi += chi_phi_tot_nhat

    thoi_gian_ket_thuc = time.time()
    return duong_di, tong_chi_phi, cac_buoc, thoi_gian_ket_thuc - thoi_gian_bat_dau


# === Chuong trinh chinh ===
do_thi = doc_do_thi('CANH.txt')
uoc_luong = doc_uoc_luong('UOCLUONG.txt')

dinh_bat_dau = 'A'
dinh_dich = 'B'

duong_di, chi_phi, cac_buoc, thoi_gian_chay = leo_doi(do_thi, uoc_luong, dinh_bat_dau, dinh_dich)

print("=== KET QUA THUAT TOAN LEO DOI ===")
print(f"Duong di: {' -> '.join(duong_di)}")
if duong_di[-1] == dinh_dich:
    print(f" Tim thay dich {dinh_dich}")
else:
    print(f" Dung tai dinh {duong_di[-1]} (dinh cuc bo)")
print(f"Tong chi phi duong di: {chi_phi}")
print(f"Thoi gian chay: {thoi_gian_chay:.6f} giay")

print("\n=== Qua trinh duyet ===")
for buoc in cac_buoc:
    print(f"Tu {buoc[0]} xet {buoc[1]} co h = {buoc[2]}")
