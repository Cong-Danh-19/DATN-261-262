"""
04_registration.py  —  Member 2: làm sạch point cloud + căn chỉnh (registration) với BIM

Point cloud từ COLMAP KHÔNG có tỷ lệ thật (unscaled) và nằm ở hệ toạ độ tuỳ ý.
Đích (--target) thường là data/bim/bim_merged.ply do 03_bim_parser.py xuất ra (mesh -> tự lấy mẫu điểm).
run_pipeline.py gọi kiểu: 04_registration.py --source <colmap.ply> --target <bim_merged.ply> --picks <json> --out <registered.ply>
Quy trình: (1) làm sạch  ->  (2) căn chỉnh thô bằng >=3 cặp điểm chọn tay
(tính cả scale)  ->  (3) ICP tinh chỉnh  ->  (4) đo fitness/RMSE trước & sau.

Có 4 chế độ (perturb dùng cho thực nghiệm có đáp án, ví dụ bộ Schependomlaan):

  1) demo  : tự sinh dữ liệu giả (một căn phòng) để test code khi chưa có dữ liệu thật
       python src/04_registration.py demo --out data/demo

  2) pick  : mở cửa sổ chọn điểm tương ứng trên point cloud và trên BIM, lưu vào JSON
       python src/04_registration.py pick --source data/pc_output/fused.ply \
              --target data/bim/bim_merged.ply --picks data/reg/picks.json

  3) run   : làm sạch + căn chỉnh thô từ picks.json + ICP + xuất kết quả
       python src/04_registration.py run --source data/pc_output/fused.ply \
              --target data/bim/bim_merged.ply --picks data/reg/picks.json \
              --out data/reg

  4) perturb : làm lệch một point cloud ĐÃ khớp sẵn với BIM (scale/xoay/dịch ngẫu nhiên) + lưu đáp án,
               để đo sai số thật của quy trình bằng 'run --gt'

Định dạng point cloud đọc được (--source):
    .ply (nhị phân/ascii, cả loại có element lạ như cameras của VisualSFM, header kiểu Windows)
    .pcd   .xyz/.txt/.csv/.pts (x y z [r g b])   .las/.laz (cần pip install laspy)
    mô hình COLMAP: thư mục sparse/0 hoặc file points3D.bin (tự lọc điểm xấu và điểm sát camera)
    mesh .obj/.stl/.ply chỉ có mặt tam giác (tự lấy mẫu 500k điểm)
Đơn vị khác mét: --source-scale (vd. mm -> 0.001).

Output của 'run':
    out/
    ├── transformation.txt      # ma trận 4x4 (đã gồm scale) đưa point cloud -> hệ BIM
    ├── registered_pc.ply       # point cloud đã làm sạch & đã căn chỉnh  -> giao cho Member 3
    ├── cleaned_pc.ply          # point cloud đã làm sạch (chưa căn chỉnh)
    └── metrics.json            # fitness, inlier RMSE trước/sau ICP, scale ước lượng

Thư viện: pip install open3d numpy   (tuỳ chọn: laspy cho .las/.laz)
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d


# ----------------------------------------------------------------------------- tiện ích
PLY_TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1", "short": "<i2", "ushort": "<u2",
             "int16": "<i2", "uint16": "<u2", "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
             "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def _ply_header(f):
    """Đọc header PLY. Trả về (format, [(tên element, số lượng, [(prop, dtype | ('list', count_t, item_t))])])."""
    fmt, els = None, []
    while True:
        line = f.readline()
        if not line:
            raise SystemExit("[ERROR] PLY hỏng: không thấy end_header")
        w = line.decode("latin-1").split()
        if not w:
            continue
        if w[0] == "format":
            fmt = w[1]
        elif w[0] == "element":
            els.append((w[1], int(w[2]), []))
        elif w[0] == "property" and els:
            if w[1] == "list":
                els[-1][2].append((w[4], ("list", PLY_TYPES[w[2]], PLY_TYPES[w[3]])))
            else:
                els[-1][2].append((w[2], PLY_TYPES[w[1]]))
        elif w[0] == "end_header":
            return fmt, els


def read_ply_vertices(path):
    """Đọc phần 'vertex' của PLY bằng numpy (ascii hoặc binary), bỏ qua các element lạ như 'cameras'
    của VisualSFM/CMVS mà Open3D có thể đọc lỗi; vertex không cần đứng đầu. Trả về (xyz, rgb 0..1 hoặc None)."""
    import itertools
    with open(path, "rb") as f:
        fmt, els = _ply_header(f)
        if fmt == "binary_big_endian":            # đổi mọi kiểu dữ liệu sang thứ tự byte big-endian
            els = [(nm, n, [(q, (t if isinstance(t, tuple) else t.replace("<", ">"))) for q, t in pr])
                   for nm, n, pr in els]
        if "vertex" not in [e[0] for e in els]:
            raise SystemExit(f"[ERROR] PLY không có element 'vertex': {path}")
        col = None
        for name, n, props in els:
            fixed = all(not isinstance(t, tuple) for _, t in props)
            if name != "vertex":                   # bỏ qua element đứng trước vertex
                if fmt == "ascii":
                    for _ in itertools.islice(f, n):
                        pass
                elif fixed:
                    f.seek(np.dtype(props).itemsize * n, 1)
                else:
                    raise SystemExit(f"[ERROR] PLY nhị phân có element '{name}' độ dài thay đổi đứng trước vertex: "
                                     "hãy lưu lại file bằng CloudCompare (File -> Save, chọn PLY)")
                continue
            if not fixed:
                raise SystemExit("[ERROR] PLY có property dạng list trong vertex, không hỗ trợ")
            vnames = [q[0] for q in props]
            if fmt == "ascii":
                A = np.loadtxt(itertools.islice(f, n), dtype=np.float64, ndmin=2)
                col = {k: A[:, i] for i, k in enumerate(vnames) if i < A.shape[1]}
            else:
                dt = np.dtype(props)
                A = np.frombuffer(f.read(dt.itemsize * n), dtype=dt, count=n)
                col = {k: A[k].astype(np.float64) for k in vnames}
            break
    xyz = np.c_[col["x"], col["y"], col["z"]]
    rgb = None
    for r, g, b in [("red", "green", "blue"), ("diffuse_red", "diffuse_green", "diffuse_blue"), ("r", "g", "b")]:
        if r in col:
            c = np.c_[col[r], col[g], col[b]]
            rgb = np.clip(c / (255.0 if c.max() > 1.0 else 1.0), 0, 1)
            break
    return xyz, rgb


def ply_face_count(path):
    try:
        with open(path, "rb") as f:
            _, els = _ply_header(f)
    except (SystemExit, KeyError):
        return 0
    return sum(n for name, n, _ in els if name == "face")


def ply_needs_numpy_reader(path):
    """True nếu PLY có element khác vertex/face, hoặc vertex không đứng đầu: để numpy đọc thay Open3D."""
    try:
        with open(path, "rb") as f:
            _, els = _ply_header(f)
    except (SystemExit, KeyError):
        return False
    names = [e[0] for e in els]
    with open(path, "rb") as f:
        crlf = b"\r\n" in f.read(512)          # header kiểu Windows: một số trình đọc C++ không chịu
    return bool(names) and (crlf or any(n not in ("vertex", "face") for n in names) or names[0] != "vertex")


def _qrot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_colmap_model(path, max_error=2.0, min_track=3, near_cam_frac=0.01):
    """Đọc thẳng mô hình COLMAP (thư mục chứa points3D.bin, hoặc chính file points3D.bin) và lọc điểm xấu:
    - sai số chiếu lại > max_error pixel
    - chỉ thấy trong < min_track ảnh
    - nằm sát camera (< near_cam_frac x kích thước cảnh): dấu hiệu chữ/logo in cố định trên video
    Trả về (xyz, rgb)."""
    import struct
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    b = open(os.path.join(folder, "points3D.bin"), "rb").read()
    n = struct.unpack_from("<Q", b, 0)[0]
    o = 8
    xyz = np.empty((n, 3)); rgb = np.empty((n, 3)); err = np.empty(n); trk = np.empty(n, dtype=np.int64)
    for i in range(n):
        xyz[i] = struct.unpack_from("<3d", b, o + 8)
        rgb[i] = struct.unpack_from("<3B", b, o + 32)
        err[i] = struct.unpack_from("<d", b, o + 35)[0]
        trk[i] = struct.unpack_from("<Q", b, o + 43)[0]
        o += 51 + 8 * trk[i]
    keep = (err <= max_error) & (trk >= min_track)
    msg = [f"sai số > {max_error}px: {int((err > max_error).sum()):,}", f"thấy < {min_track} ảnh: {int((trk < min_track).sum()):,}"]
    img = os.path.join(folder, "images.bin")
    if os.path.exists(img) and near_cam_frac > 0:
        b = open(img, "rb").read()
        m = struct.unpack_from("<Q", b, 0)[0]
        o = 8
        cams = []
        for _ in range(m):
            q = struct.unpack_from("<4d", b, o + 4); t = np.array(struct.unpack_from("<3d", b, o + 36)); o += 64
            o = b.index(b"\0", o) + 1
            k = struct.unpack_from("<Q", b, o)[0]; o += 8 + 24 * k
            cams.append(-_qrot(*q).T @ t)
        cams = np.array(cams)
        lo, hi = np.percentile(xyz[keep], [2, 98], axis=0)
        thr = near_cam_frac * float(np.linalg.norm(hi - lo))
        dmin = np.full(n, np.inf)
        step = max(500, int(4e6 // max(len(cams), 1)))   # giới hạn bộ nhớ khi có hàng nghìn ảnh
        for s in range(0, n, step):                       # khoảng cách tới camera gần nhất, theo từng khối
            d = np.linalg.norm(xyz[s:s + step, None, :] - cams[None, :, :], axis=2)
            dmin[s:s + step] = d.min(1)
        near = dmin < thr
        keep &= ~near
        msg.append(f"sát camera: {int(near.sum()):,}")
        print(f"[COLMAP] {m} ảnh đã đăng ký")
    print(f"[COLMAP] {n:,} điểm -> giữ {int(keep.sum()):,}  (bỏ: {', '.join(msg)})")
    return xyz[keep], rgb[keep] / 255.0


def read_xyz_text(path):
    """File văn bản x y z [r g b] (.txt .csv .xyz .pts .asc)."""
    import itertools
    with open(path, encoding="latin-1") as f:
        first = [l for l in itertools.islice(f, 50) if l.strip()]
    delim = "," if "," in first[-1] else None
    skip = 0
    for l in first:                                     # bỏ dòng tiêu đề / dòng đếm số điểm (.pts)
        try:
            v = [float(x) for x in l.replace(",", " ").split()]
            if len(v) >= 3:
                break
        except ValueError:
            pass
        skip += 1
    A = np.loadtxt(path, delimiter=delim, skiprows=skip, ndmin=2, encoding="latin-1")
    rgb = None
    if A.shape[1] >= 6:
        c = A[:, 3:6] if A.shape[1] == 6 else A[:, -3:]
        rgb = np.clip(c / (255.0 if c.max() > 1.0 else 1.0), 0, 1)
    return A[:, :3], rgb


def read_las(path):
    try:
        import laspy
    except ImportError:
        raise SystemExit("[ERROR] Đọc .las/.laz cần:  pip install laspy  (thêm  pip install lazrs  cho .laz)")
    las = laspy.read(path)
    xyz = np.c_[las.x, las.y, las.z]
    rgb = None
    if hasattr(las, "red"):
        c = np.c_[las.red, las.green, las.blue].astype(np.float64)
        rgb = c / (65535.0 if c.max() > 255 else 255.0)
    return xyz, rgb


def load_pc(path, scale=1.0):
    """Đọc point cloud từ nhiều định dạng; scale dùng để đổi đơn vị (vd. mm -> m: 0.001).
    Hỗ trợ: .ply (cả loại có element lạ), .pcd, .xyz/.txt/.csv/.pts, .las/.laz, mô hình COLMAP
    (thư mục sparse/0 hoặc points3D.bin), và mesh (.ply/.obj/.stl -> lấy mẫu điểm)."""
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] Không thấy file/thư mục: {path}")
    ext = os.path.splitext(path)[1].lower()
    xyz = rgb = pc = None
    if os.path.isdir(path) or os.path.basename(path).lower() == "points3d.bin":
        xyz, rgb = read_colmap_model(path)
    elif ext == ".ply" and ply_face_count(path) > 0:
        pc = o3d.geometry.PointCloud()          # PLY là mesh (vd. bim_merged.ply của 03): lấy mẫu bề mặt bên dưới
    elif ext == ".ply" and ply_needs_numpy_reader(path):
        xyz, rgb = read_ply_vertices(path)
    elif ext in (".txt", ".csv", ".asc", ".pts") or (ext == ".xyz"):
        xyz, rgb = read_xyz_text(path)
    elif ext in (".las", ".laz"):
        xyz, rgb = read_las(path)
    if xyz is not None and pc is None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64)))
        if rgb is not None:
            pc.colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64))
    elif ext in (".obj", ".stl", ".off", ".gltf", ".glb", ".fbx"):
        pc = o3d.geometry.PointCloud()          # file mesh: xử lý ở nhánh mesh bên dưới
    elif pc is None:
        pc = o3d.io.read_point_cloud(path)      # .ply thường, .pcd, .xyzn, .xyzrgb ...
    if pc.is_empty():
        # file là mesh (hoặc PLY chỉ có mặt tam giác): lấy mẫu điểm trên bề mặt
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise SystemExit(f"[ERROR] Không đọc được point cloud từ {path}")
        area = mesh.get_surface_area()          # ~400 điểm/m2 nếu đơn vị là mét, giới hạn 200k..3M
        n = int(np.clip(area * 400, 200000, 3000000))
        pc = mesh.sample_points_uniformly(n)
        print(f"[INFO] {os.path.basename(path)} là mesh -> lấy mẫu {n:,} điểm trên bề mặt")
    if scale != 1.0:
        pc.scale(scale, center=np.zeros(3))     # đổi đơn vị đúng MỘT lần
    print(f"[LOAD] {os.path.basename(path)}: {len(pc.points):,} điểm")
    return pc


def clean_point_cloud(pc, voxel, nb_neighbors=20, std_ratio=2.0):
    """Downsample + lọc nhiễu thống kê. voxel tính theo đơn vị của chính point cloud."""
    n0 = len(pc.points)
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[CLEAN] {n0:,} -> {len(pc.points):,} điểm (voxel={voxel:.4f}, std_ratio={std_ratio})")
    return pc


def robust_extent(points, lo=2, hi=98):
    """Kích thước cảnh bỏ qua điểm rác bay xa: dùng phân vị 2%-98% thay vì min-max."""
    P = np.asarray(points)
    if len(P) > 200000:
        P = P[np.random.default_rng(0).choice(len(P), 200000, replace=False)]
    a, b = np.percentile(P, [lo, hi], axis=0)
    return b - a


def auto_voxel(pc, fraction=0.005):
    """Voxel bằng một phần nhỏ đường chéo cảnh (COLMAP không có đơn vị thật). Không dùng hộp bao min-max:
    vài điểm rác ở rất xa sẽ làm voxel to hơn cả công trình và xoá sạch point cloud."""
    return max(float(np.linalg.norm(robust_extent(pc.points)) * fraction), 1e-9)


def umeyama(src, dst, with_scale=True):
    """Tìm s, R, t sao cho dst ≈ s*R*src + t (bình phương tối thiểu). src, dst: (N,3), N>=3."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, s


def evaluate(src, tgt, T, dist):
    r = o3d.pipelines.registration.evaluate_registration(src, tgt, dist, T)
    return dict(fitness=float(r.fitness), inlier_rmse=float(r.inlier_rmse))


def run_icp(src, tgt, T_init, voxels=(0.10, 0.05, 0.02), refine_scale=False):
    """ICP nhiều mức (thô -> mịn). voxels tính theo mét vì lúc này src đã về hệ BIM."""
    T = T_init
    for v in voxels:
        s_d, t_d = src.voxel_down_sample(v), tgt.voxel_down_sample(v)
        max_dist = v * 2.0
        if refine_scale:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        else:
            t_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 3, max_nn=30))
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        res = o3d.pipelines.registration.registration_icp(
            s_d, t_d, max_dist, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T = res.transformation
        print(f"[ICP] voxel={v:.3f}m  fitness={res.fitness:.3f}  inlier_rmse={res.inlier_rmse:.4f}m")
    return T


# ----------------------------------------------------------------------------- chế độ pick
def color_by_height(pc):
    """Point cloud không có màu (vd. BIM) -> tô màu theo trục Z để dễ nhận ra tầng, mái."""
    if pc.has_colors():
        return pc
    z = np.asarray(pc.points)[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)
    # thang màu xanh dương -> xanh lá -> vàng -> đỏ
    rgb = np.c_[np.clip(2 * t - 0.5, 0, 1), np.clip(1.5 - np.abs(2 * t - 1) * 1.5, 0, 1), np.clip(1 - 2 * t, 0, 1)]
    out = o3d.geometry.PointCloud(pc)
    out.colors = o3d.utility.Vector3dVector(rgb)
    return out


def check_picks(p_src, p_tgt):
    """In sai số từng cặp sau căn chỉnh thô + cảnh báo các trường hợp dễ sai. Trả về (T, s, resid)."""
    P, Q = np.asarray(p_src, float), np.asarray(p_tgt, float)
    for name, X in (("point cloud", P), ("BIM", Q)):
        sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
        if len(X) < 3 or sv[1] < 1e-6 * max(sv[0], 1e-12):
            raise SystemExit(f"[ERROR] Các điểm chọn trên {name} gần như thẳng hàng: chọn các góc trải rộng hơn.")
        if len(X) >= 4 and sv[2] < 0.02 * sv[0]:
            print(f"[WARN] Các điểm trên {name} gần như cùng một mặt phẳng: nên có ít nhất 1 điểm ở độ cao khác.")
    T, s = umeyama(P, Q, with_scale=True)
    resid = np.linalg.norm((np.c_[P, np.ones(len(P))] @ T.T)[:, :3] - Q, axis=1)
    print("[PICKS] Sai số từng cặp sau căn chỉnh thô (đơn vị của BIM):")
    for i, r in enumerate(resid, 1):
        print(f"        cặp {i}: {r:.3f}")
    if len(P) >= 4:
        # Bỏ lần lượt từng cặp, căn bằng các cặp còn lại: bỏ đúng cặp chọn sai thì phần còn lại khớp rất tốt
        score = []
        for i in range(len(P)):
            k = np.arange(len(P)) != i
            Ti, _ = umeyama(P[k], Q[k], with_scale=True)
            score.append(np.linalg.norm((np.c_[P[k], np.ones(k.sum())] @ Ti.T)[:, :3] - Q[k], axis=1).mean())
        score = np.array(score)
        j = int(score.argmin())
        if score[j] < 0.5 * np.median(score) and resid.mean() > 2 * score[j]:
            print(f"[WARN] Nghi cặp {j + 1} chọn sai: bỏ cặp này thì sai số TB các cặp còn lại giảm từ "
                  f"{resid.mean():.3f} xuống {score[j]:.3f}. Nên chọn lại điểm đó.")
    print(f"[PICKS] scale = {s:.4f}, sai số TB = {resid.mean():.3f}, lớn nhất = {resid.max():.3f}")
    return T, s, resid


def pick_points(pc, title):
    pc = color_by_height(pc)
    print(f"\n=== {title} ===")
    print("  SHIFT + Click trái : chọn điểm      SHIFT + Click phải : bỏ điểm vừa chọn")
    print("  Chọn >= 4 điểm dễ nhận ra (góc tường, chân cột, góc cửa), THEO CÙNG MỘT THỨ TỰ ở cả 2 cửa sổ.")
    print("  Phím +/- : to/nhỏ điểm.  KHÔNG bấm K hoặc C (chế độ cắt).  Xong thì nhấn Q để đóng cửa sổ.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pc)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    return np.asarray(pc.points)[idx].tolist()


def cmd_pick(a):
    src = load_pc(a.source, a.source_scale)
    tgt = load_pc(a.target)
    p_src = pick_points(src, "1/2  POINT CLOUD (COLMAP)")
    p_tgt = pick_points(tgt, "2/2  BIM")
    if len(p_src) != len(p_tgt) or len(p_src) < 3:
        raise SystemExit(f"[ERROR] Cần >=3 cặp và số điểm 2 bên bằng nhau (đang có {len(p_src)} vs {len(p_tgt)}).")
    os.makedirs(os.path.dirname(os.path.abspath(a.picks)), exist_ok=True)
    with open(a.picks, "w") as f:
        json.dump({"source_points": p_src, "target_points": p_tgt}, f, indent=2)
    _, s, _ = check_picks(p_src, p_tgt)
    print(f"[DONE] Đã lưu {len(p_src)} cặp điểm vào {a.picks}. Scale ước lượng = {s:.4f}")


# ----------------------------------------------------------------------------- chế độ run
def transform_error(T, T_gt, pts):
    """So T ước lượng với T_gt đúng. pts: các điểm nguồn để đo độ lệch trung bình sau biến đổi."""
    sa, sb = np.cbrt(np.linalg.det(T[:3, :3])), np.cbrt(np.linalg.det(T_gt[:3, :3]))
    Ra, Rb = T[:3, :3] / sa, T_gt[:3, :3] / sb
    ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
    P = np.c_[pts, np.ones(len(pts))]
    d = np.linalg.norm(P @ T.T - P @ T_gt.T, axis=1)[:, None]
    return dict(rotation_error_deg=float(ang), scale_error_pct=float(abs(sa / sb - 1) * 100),
                translation_error_m=float(np.linalg.norm(T[:3, 3] - T_gt[:3, 3])),
                point_error_mean_m=float(d.mean()), point_error_max_m=float(d.max()))


def register(src_raw, tgt, p_src, p_tgt, out, std_ratio=2.0, voxels=(0.10, 0.05, 0.02),
             refine_scale=False, show=False, crop_margin=None, T_gt=None):
    os.makedirs(out, exist_ok=True)

    # 1) làm sạch (trong đơn vị gốc của COLMAP)
    src = clean_point_cloud(src_raw, auto_voxel(src_raw, 0.002), std_ratio=std_ratio)
    o3d.io.write_point_cloud(os.path.join(out, "cleaned_pc.ply"), src)

    # 2) căn chỉnh thô bằng các cặp điểm chọn tay (có scale)
    T0, s, resid = check_picks(p_src, p_tgt)
    print(f"[INIT] scale={s:.4f}  sai số trên các điểm chọn: mean={resid.mean():.3f}m max={resid.max():.3f}m")
    if resid.max() > 0.5:
        print("[WARN] Sai số điểm chọn > 0.5m: có thể chọn sai thứ tự / sai điểm. Nên chọn lại.")

    # 3) ICP — đưa point cloud về hệ BIM (mét) trước, để các ngưỡng voxel/khoảng cách có nghĩa
    src_init = o3d.geometry.PointCloud(src)
    src_init.transform(T0)
    if crop_margin is not None:
        # bỏ cây cối / nhà bên cạnh: chỉ giữ phần nằm trong hộp bao BIM (+ lề)
        bb = tgt.get_axis_aligned_bounding_box()
        box = o3d.geometry.AxisAlignedBoundingBox(bb.min_bound - crop_margin, bb.max_bound + crop_margin)
        n0 = len(src_init.points)
        idx = box.get_point_indices_within_bounding_box(src_init.points)
        src_init = src_init.select_by_index(idx)
        src = src.select_by_index(idx)
        print(f"[CROP] giữ {len(idx):,}/{n0:,} điểm nằm trong hộp bao BIM ± {crop_margin} m")
    eval_dist = voxels[-1] * 2.5
    I = np.eye(4)
    before = evaluate(src_init, tgt, I, eval_dist)
    T_icp = run_icp(src_init, tgt, I, voxels=voxels, refine_scale=refine_scale)
    after = evaluate(src_init, tgt, T_icp, eval_dist)
    T = T_icp @ T0          # tổng hợp: COLMAP -> (thô) -> (ICP) -> BIM

    # 4) lưu kết quả
    np.savetxt(os.path.join(out, "transformation.txt"), T, fmt="%.8f")
    reg = o3d.geometry.PointCloud(src)
    reg.transform(T)
    o3d.io.write_point_cloud(os.path.join(out, "registered_pc.ply"), reg)
    scale_final = float(np.cbrt(np.linalg.det(T[:3, :3])))
    metrics = dict(scale_init=float(s), scale_final=scale_final, eval_distance_m=eval_dist,
                   picks_residual_mean_m=float(resid.mean()), picks_residual_max_m=float(resid.max()),
                   before_icp=before, after_icp=after, n_points_registered=len(reg.points))
    if T_gt is not None:
        pts = np.asarray(src.points)
        metrics["vs_ground_truth_before_icp"] = transform_error(T0, T_gt, pts)
        metrics["vs_ground_truth_after_icp"] = transform_error(T, T_gt, pts)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[RESULT] Trước ICP: fitness={before['fitness']:.3f} RMSE={before['inlier_rmse']:.4f}m")
    print(f"[RESULT] Sau  ICP: fitness={after['fitness']:.3f} RMSE={after['inlier_rmse']:.4f}m")
    if T_gt is not None:
        for k, lab in [("vs_ground_truth_before_icp", "Trước ICP"), ("vs_ground_truth_after_icp", "Sau  ICP")]:
            e = metrics[k]
            print(f"[GT]     {lab}: lệch góc {e['rotation_error_deg']:.3f}°, scale {e['scale_error_pct']:.2f}%, "
                  f"điểm lệch TB {e['point_error_mean_m']:.3f} m (max {e['point_error_max_m']:.3f} m)")
    print(f"[DONE] Kết quả tại {out}")

    if show:
        reg.paint_uniform_color([1.0, 0.3, 0.0])   # cam = point cloud
        t = o3d.geometry.PointCloud(tgt)
        t.paint_uniform_color([0.2, 0.4, 1.0])     # xanh = BIM
        o3d.visualization.draw_geometries([t, reg], window_name="Cam: point cloud | Xanh: BIM")
    return T, metrics


def cmd_run(a):
    if not os.path.exists(a.picks):
        raise SystemExit(f"[ERROR] Chưa có file cặp điểm {a.picks}. Chạy trước:\n"
                         f"  python src/04_registration.py pick --source {a.source} --target {a.target} --picks {a.picks}")
    # --out là thư mục, hoặc một file .ply (cách run_pipeline.py gọi): khi đó các file phụ vào <thư mục>/registration/
    out_file = a.out if a.out.lower().endswith(".ply") else None
    out_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "registration") if out_file else a.out
    src_raw = load_pc(a.source, a.source_scale)
    tgt = load_pc(a.target)
    with open(a.picks) as f:
        picks = json.load(f)
    T_gt = np.loadtxt(a.gt) if a.gt else None
    register(src_raw, tgt, picks["source_points"], picks["target_points"], out_dir,
             std_ratio=a.std_ratio, voxels=tuple(a.voxels), refine_scale=a.refine_scale, show=a.show,
             crop_margin=a.crop_margin, T_gt=T_gt)
    if out_file:
        import shutil
        shutil.copyfile(os.path.join(out_dir, "registered_pc.ply"), out_file)
        print(f"[DONE] Point cloud đã căn chỉnh: {out_file}  (file phụ: {out_dir})")


# ----------------------------------------------------------------------------- chế độ perturb
def cmd_perturb(a):
    """Lấy một point cloud ĐÃ KHỚP sẵn với BIM, làm lệch nó giống output COLMAP (scale/xoay/dịch ngẫu nhiên)
    và lưu đáp án gt_transform.txt (đưa cloud lệch về lại hệ BIM) để đo sai số của quy trình căn chỉnh."""
    pc = load_pc(a.source, a.source_scale)
    if a.target:
        bb = load_pc(a.target).get_axis_aligned_bounding_box()
        box = o3d.geometry.AxisAlignedBoundingBox(bb.min_bound - a.margin, bb.max_bound + a.margin)
        pc = pc.crop(box)
        print(f"[CROP] giữ {len(pc.points):,} điểm trong hộp bao BIM ± {a.margin} m")
    if a.max_points and len(pc.points) > a.max_points:
        pc = pc.random_down_sample(a.max_points / len(pc.points))
    rng = np.random.default_rng(a.seed)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    ang = np.deg2rad(a.rot_deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K           # công thức Rodrigues
    t = rng.normal(size=3); t = t / np.linalg.norm(t) * a.trans
    M = np.eye(4); M[:3, :3] = a.scale * R; M[:3, 3] = t                    # BIM -> "COLMAP"
    pc.transform(M)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    o3d.io.write_point_cloud(a.out, pc)
    gt = os.path.splitext(a.out)[0] + "_gt_transform.txt"
    np.savetxt(gt, np.linalg.inv(M), fmt="%.10f")
    print(f"[DONE] {a.out}: {len(pc.points):,} điểm, scale={a.scale}, xoay {a.rot_deg}°, dịch {a.trans}")
    print(f"       Đáp án (cloud lệch -> BIM): {gt}")


# ----------------------------------------------------------------------------- chế độ demo
def make_room_mesh():
    """Phòng 6x4x3m: sàn + 4 tường + 1 cột (đơn vị mét)."""
    parts = []
    def box(sx, sy, sz, cx, cy, cz):
        b = o3d.geometry.TriangleMesh.create_box(sx, sy, sz)
        b.translate([cx - sx / 2, cy - sy / 2, cz - sz / 2])
        return b
    parts += [box(6, 4, 0.2, 3, 2, -0.1)]                                    # sàn
    parts += [box(6, 0.2, 3, 3, 0, 1.5), box(6, 0.2, 3, 3, 4, 1.5)]           # tường dài
    parts += [box(0.2, 4, 3, 0, 2, 1.5), box(0.2, 4, 3, 6, 2, 1.5)]           # tường ngắn
    parts += [box(0.4, 0.4, 3, 2, 2, 1.5)]                                    # cột
    m = parts[0]
    for p in parts[1:]:
        m += p
    return m


def cmd_demo(a):
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(42)
    bim_mesh = make_room_mesh()
    bim = bim_mesh.sample_points_uniformly(100000)
    o3d.io.write_point_cloud(os.path.join(a.out, "demo_bim_points.ply"), bim)

    # giả lập COLMAP: chỉ thấy 1 phần phòng, có nhiễu, có outlier, bị scale + xoay + dịch
    pts = np.asarray(bim.points)
    pts = pts[pts[:, 0] < 4.5]                                    # camera chỉ quét được một phần
    pts = pts + rng.normal(0, 0.01, pts.shape)                    # nhiễu 1cm
    outl = rng.uniform(pts.min(0) - 1, pts.max(0) + 1, (500, 3))  # outlier
    pts = np.vstack([pts, outl])
    s_true = 0.37
    ax, ay, az = np.deg2rad([20, -35, 60])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    t_true = np.array([1.2, -0.7, 3.3])
    T_gt_inv = np.eye(4); T_gt_inv[:3, :3] = s_true * R; T_gt_inv[:3, 3] = t_true  # BIM -> "COLMAP"
    colmap_pts = (s_true * (R @ pts.T)).T + t_true
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(colmap_pts))
    o3d.io.write_point_cloud(os.path.join(a.out, "demo_colmap_like.ply"), src)

    # giả lập việc chọn tay 4 góc, mỗi điểm lệch ~5cm như người chọn thật
    corners_bim = np.array([[0.1, 0.1, 0.0], [4.0, 0.1, 0.0], [0.1, 3.9, 0.0], [0.1, 0.1, 3.0], [2.2, 2.2, 3.0]])
    p_tgt = corners_bim
    p_src = (s_true * (R @ (corners_bim + rng.normal(0, 0.05, corners_bim.shape)).T)).T + t_true

    T, m = register(src, bim, p_src, p_tgt, a.out, show=a.show)
    T_true = np.linalg.inv(T_gt_inv)
    err_t = np.linalg.norm(T[:3, 3] - T_true[:3, 3])
    print(f"\n[DEMO] scale đúng = {1/s_true:.4f}, ước lượng = {m['scale_final']:.4f}; sai lệch tịnh tiến = {err_t:.4f}m")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Member 2 - Registration point cloud <-> BIM")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo"); d.add_argument("--out", default="data/demo"); d.add_argument("--show", action="store_true")

    p = sub.add_parser("pick")
    p.add_argument("--source", required=True); p.add_argument("--target", required=True)
    p.add_argument("--picks", required=True)
    p.add_argument("--source-scale", type=float, default=1.0, help="đổi đơn vị point cloud, vd mm->m: 0.001")

    r = sub.add_parser("run")
    r.add_argument("--source", required=True, help="point cloud COLMAP (.ply)")
    r.add_argument("--target", required=True, help="bim_full_points.ply từ 03_bim_parser.py")
    r.add_argument("--picks", default="data/reg/picks.json", help="file JSON tạo bởi lệnh pick")
    r.add_argument("--out", required=True)
    r.add_argument("--std-ratio", type=float, default=2.0, help="nhỏ hơn = lọc nhiễu mạnh hơn")
    r.add_argument("--voxels", type=float, nargs="+", default=[0.10, 0.05, 0.02], help="các mức ICP (mét)")
    r.add_argument("--refine-scale", action="store_true", help="cho ICP chỉnh tiếp cả scale (point-to-point)")
    r.add_argument("--show", action="store_true", help="hiển thị kết quả chồng lên nhau")
    r.add_argument("--source-scale", type=float, default=1.0, help="đổi đơn vị point cloud, vd mm->m: 0.001")
    r.add_argument("--crop-margin", type=float, default=None,
                   help="sau căn chỉnh thô, chỉ giữ điểm trong hộp bao BIM ± lề (m); bỏ cây cối, nhà bên cạnh")
    r.add_argument("--gt", default=None, help="file ma trận đáp án (từ lệnh perturb) để đo sai số thật")

    q = sub.add_parser("perturb", help="làm lệch một cloud đã khớp sẵn với BIM để có đáp án kiểm tra")
    q.add_argument("--source", required=True, help="point cloud đã nằm đúng hệ BIM")
    q.add_argument("--out", required=True, help="file .ply đầu ra (cloud bị làm lệch)")
    q.add_argument("--source-scale", type=float, default=1.0, help="đổi đơn vị trước, vd mm->m: 0.001")
    q.add_argument("--target", default=None, help="bim_full_points.ply: nếu có thì cắt cloud quanh hộp bao BIM")
    q.add_argument("--margin", type=float, default=3.0, help="lề khi cắt (m)")
    q.add_argument("--max-points", type=int, default=1500000)
    q.add_argument("--scale", type=float, default=0.37, help="tỷ lệ giả lập (COLMAP không có đơn vị thật)")
    q.add_argument("--rot-deg", type=float, default=35.0)
    q.add_argument("--trans", type=float, default=5.0)
    q.add_argument("--seed", type=int, default=0)

    import sys
    argv = sys.argv[1:]
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv          # run_pipeline.py gọi không có lệnh con -> hiểu là 'run'
    a = ap.parse_args(argv)
    {"demo": cmd_demo, "pick": cmd_pick, "run": cmd_run, "perturb": cmd_perturb}[a.cmd](a)


if __name__ == "__main__":
    main()
