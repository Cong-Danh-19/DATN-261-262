"""
03_bim_parser.py  —  Member 2: chuẩn bị dữ liệu BIM

Đọc file OBJ (export từ Navisworks -> FBX -> Blender -> OBJ), tách từng cấu kiện theo
dòng 'g'/'o', phân loại theo tên (hỗ trợ tên viết tắt tiếng Việt như VL.BT_COT_B40),
đổi trục về Z-up, lọc theo cao độ tầng, lấy mẫu điểm và lưu ra:

    out_dir/
    ├── bim_full_mesh.ply        # mesh các cấu kiện được giữ lại (mét, Z hướng lên)
    ├── bim_full_points.ply      # point cloud của các cấu kiện "nhìn thấy được" -> dùng cho registration
    ├── elements/0001_column_....ply   # từng cấu kiện (mesh)          -> dùng cho Member 3
    └── elements_manifest.csv    # bảng thông tin cấu kiện              -> dùng cho Member 3

Ví dụ:
    # 1) Xem danh sách cao độ sàn để biết tầng nào nằm ở khoảng Z nào
    python src/03_bim_parser.py --input data/bim/building.obj --list-levels

    # 2) Chỉ lấy 1 tầng (ví dụ sàn ở Z=8.8 đến sàn trên ở Z=12.2)
    python src/03_bim_parser.py --input data/bim/building.obj --out data/bim/prepared --zmin 8.8 --zmax 12.2

Chỉ cần numpy (không phụ thuộc trimesh/open3d).
"""
import argparse
import collections
import csv
import os
import re
import unicodedata

import numpy as np

# ----------------------------------------------------------------------------- phân loại
# Mỗi luật: (loại, điều kiện trên tập token của tên). Kiểm tra theo thứ tự, luật đầu tiên khớp thắng.
RULES = [
    ("formwork",      lambda t: "coff" in t or "coffa" in t or "formwork" in t),       # cốp pha (tạm thời)
    ("pile_cap",      lambda t: {"dai", "mong"} <= t or "pilecap" in t),             # đài móng
    ("lean_concrete", lambda t: "lot" in t),                                         # bê tông lót
    ("pile",          lambda t: "coc" in t or "pile" in t),                          # cọc
    ("column",        lambda t: "cot" in t or "column" in t),                        # cột
    ("beam",          lambda t: "dam" in t or "beam" in t or "framing" in t),        # dầm
    ("slab",          lambda t: "san" in t or "slab" in t or "floor" in t),          # sàn
    ("wall",          lambda t: "vach" in t or "tuong" in t or "wall" in t),         # vách / tường
    ("stair",         lambda t: {"cau", "thang"} <= t or "stair" in t),
    ("door",          lambda t: "door" in t),
    ("window",        lambda t: "window" in t),
    ("steel",         lambda t: "thep" in t or "steel" in t),
]
# Các loại camera có thể nhìn thấy khi quay trong nhà -> dùng làm đích căn chỉnh
DEFAULT_TARGET = "column,beam,slab,wall,stair,door,window"


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def base_name(name):
    """'VL.BT_COT_B40.1079_Mesh.2380' -> 'VL.BT_COT_B40'"""
    return re.sub(r"(\.\d+)?_Mesh.*$", "", name)


def classify(name):
    tokens = set(re.split(r"[^a-z0-9]+", strip_accents(base_name(name)).lower())) - {""}
    for cat, rule in RULES:
        if rule(tokens):
            return cat
    return "other"


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", strip_accents(s))[:50]


# ----------------------------------------------------------------------------- đọc OBJ
def read_obj_groups(path):
    """Trả về (V: (N,3) float, groups: OrderedDict tên -> mảng tam giác (M,3) chỉ số 0-based)."""
    V = []
    groups = collections.OrderedDict()
    cur = "default"
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith(("g ", "o ")):
                cur = line[2:].strip() or "unnamed"
                # tên trùng -> thêm hậu tố để không bị gộp nhầm
                if cur in groups:
                    k = 2
                    while f"{cur}#{k}" in groups:
                        k += 1
                    cur = f"{cur}#{k}"
                groups[cur] = []
            elif line.startswith("f "):
                n = len(V)
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else n + i)   # hỗ trợ chỉ số âm
                for j in range(1, len(idx) - 1):            # tam giác hoá đa giác (fan)
                    groups.setdefault(cur, []).append((idx[0], idx[j], idx[j + 1]))
    V = np.asarray(V, dtype=np.float64)
    groups = collections.OrderedDict((k, np.asarray(v, dtype=np.int64)) for k, v in groups.items() if v)
    return V, groups


def compact(V, F):
    """Lấy riêng các đỉnh mà mesh dùng, đánh lại chỉ số."""
    used, inv = np.unique(F, return_inverse=True)
    return V[used], inv.reshape(F.shape)


# ----------------------------------------------------------------------------- hình học
def triangle_areas(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def sample_surface(V, F, n, rng):
    areas = triangle_areas(V, F)
    if areas.sum() <= 0:
        return np.empty((0, 3))
    tri = rng.choice(len(F), size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    a, b, c = V[F[tri, 0]], V[F[tri, 1]], V[F[tri, 2]]
    return a + u[:, None] * (b - a) + v[:, None] * (c - a)


# ----------------------------------------------------------------------------- ghi PLY
def write_ply_points(path, P):
    P = np.asarray(P, dtype="<f4")
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n").encode())
        f.write(P.tobytes())


def write_ply_mesh(path, V, F):
    V = np.asarray(V, dtype="<f4")
    face = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"] = 3
    face["i"] = F
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(V.tobytes())
        f.write(face.tobytes())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="file .obj")
    ap.add_argument("--out", help="thư mục output")
    ap.add_argument("--unit-scale", type=float, default=1.0,
                    help="hệ số đổi sang mét: m->1 (mặc định), mm->0.001, ft->0.3048")
    ap.add_argument("--up", choices=["y", "z"], default="y",
                    help="trục hướng lên trong file OBJ. Blender xuất mặc định Y-up -> script đổi về Z-up")
    ap.add_argument("--zmin", type=float, default=None, help="chỉ giữ cấu kiện có tâm Z >= zmin (mét, sau khi đổi trục)")
    ap.add_argument("--zmax", type=float, default=None, help="chỉ giữ cấu kiện có tâm Z <= zmax")
    ap.add_argument("--target-categories", default=DEFAULT_TARGET,
                    help="các loại đưa vào bim_full_points.ply để căn chỉnh")
    ap.add_argument("--density", type=float, default=400, help="điểm / m2 khi lấy mẫu (400 ~ cách nhau 5cm)")
    ap.add_argument("--max-points-per-element", type=int, default=100000)
    ap.add_argument("--min-area", type=float, default=0.01, help="bỏ cấu kiện nhỏ hơn (m2)")
    ap.add_argument("--list-levels", action="store_true", help="chỉ in cao độ sàn rồi thoát")
    args = ap.parse_args()

    print(f"[INFO] Đang đọc {args.input} ...")
    V, groups = read_obj_groups(args.input)
    V = V * args.unit_scale
    if args.up == "y":
        V = np.c_[V[:, 0], -V[:, 2], V[:, 1]]      # xoay +90° quanh X: Y-up -> Z-up
    print(f"[INFO] {len(V):,} đỉnh, {len(groups):,} cấu kiện")
    if len(groups) <= 1:
        print("[WARN] File chỉ có <=1 cấu kiện: đã bị gộp. Cần export lại hoặc cắt tay bằng CloudCompare.")

    # thông tin từng cấu kiện
    elems = []
    for name, F in groups.items():
        Ve, Fe = compact(V, F)
        elems.append(dict(name=name, base=base_name(name), cat=classify(name), V=Ve, F=Fe,
                          bmin=Ve.min(0), bmax=Ve.max(0), area=float(triangle_areas(Ve, Fe).sum())))

    ext = V.max(0) - V.min(0)
    print(f"[INFO] Kích thước toàn mô hình (m): X={ext[0]:.1f}  Y={ext[1]:.1f}  Z(cao)={ext[2]:.1f}")
    print("[INFO] Số cấu kiện theo loại:")
    by_cat = collections.Counter(e["cat"] for e in elems)
    for c, n in by_cat.most_common():
        names = sorted({e["base"] for e in elems if e["cat"] == c})
        print(f"       {c:14s} {n:6d}   vd: {', '.join(names[:4])}")

    if args.list_levels:
        lv = collections.Counter(round(float(e["bmax"][2]), 1) for e in elems if e["cat"] == "slab")
        merged = []
        for z in sorted(lv):
            if merged and z - merged[-1][1] < 0.5:
                merged[-1][1] = z
                merged[-1][2] += lv[z]
            else:
                merged.append([z, z, lv[z]])
        print("[LEVELS] Cao độ mặt sàn (Z, mét) và số tấm sàn:")
        for z0, z1, n in merged:
            rng_txt = f"{z0:6.1f}" if z0 == z1 else f"{z0:6.1f} .. {z1:.1f}"
            print(f"         Z = {rng_txt:14s} ({n} tấm)")
        print("         -> Một tầng: --zmin <sàn tầng đó - 0.5> --zmax <sàn tầng trên + 0.2>")
        return

    if not args.out:
        raise SystemExit("[ERROR] Cần --out (hoặc dùng --list-levels)")
    os.makedirs(os.path.join(args.out, "elements"), exist_ok=True)

    targets = set(args.target_categories.split(","))
    rng = np.random.default_rng(0)
    rows, keep_meshes, target_pts = [], [], []
    for i, e in enumerate(elems, start=1):
        if e["area"] < args.min_area:
            continue
        cz = 0.5 * (e["bmin"][2] + e["bmax"][2])
        if args.zmin is not None and cz < args.zmin:
            continue
        if args.zmax is not None and cz > args.zmax:
            continue
        eid = f"{i:04d}"
        fname = f"{eid}_{e['cat']}_{safe_name(e['base'])}.ply"
        write_ply_mesh(os.path.join(args.out, "elements", fname), e["V"], e["F"])
        keep_meshes.append((e["V"], e["F"]))
        in_target = e["cat"] in targets
        n = int(min(max(e["area"] * args.density, 50), args.max_points_per_element))
        if in_target:
            target_pts.append(sample_surface(e["V"], e["F"], n, rng))
        rows.append(dict(element_id=eid, name=e["name"], category=e["cat"], file=f"elements/{fname}",
                         area_m2=round(e["area"], 4), in_registration_target=int(in_target),
                         bbox_min=";".join(f"{v:.3f}" for v in e["bmin"]),
                         bbox_max=";".join(f"{v:.3f}" for v in e["bmax"])))

    if not rows:
        raise SystemExit("[ERROR] Không còn cấu kiện nào sau khi lọc. Kiểm tra --zmin/--zmax (xem --list-levels).")

    # gộp mesh
    off, Vs, Fs = 0, [], []
    for Ve, Fe in keep_meshes:
        Vs.append(Ve); Fs.append(Fe + off); off += len(Ve)
    write_ply_mesh(os.path.join(args.out, "bim_full_mesh.ply"), np.vstack(Vs), np.vstack(Fs))
    P = np.vstack(target_pts) if target_pts else np.empty((0, 3))
    write_ply_points(os.path.join(args.out, "bim_full_points.ply"), P)

    with open(os.path.join(args.out, "elements_manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    kept = collections.Counter(r["category"] for r in rows)
    print(f"[INFO] Giữ lại {len(rows):,} cấu kiện: {dict(kept)}")
    print(f"[INFO] bim_full_points.ply: {len(P):,} điểm (chỉ gồm: {', '.join(sorted(targets & set(kept)))})")
    print(f"[DONE] Output tại {args.out}")


if __name__ == "__main__":
    main()
