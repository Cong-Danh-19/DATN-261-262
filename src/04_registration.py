"""
04_registration.py  —  Member 2: làm sạch point cloud + căn chỉnh (registration) với BIM

Point cloud từ COLMAP KHÔNG có tỷ lệ thật (unscaled) và nằm ở hệ toạ độ tuỳ ý.
Quy trình: (1) làm sạch  ->  (2) căn chỉnh thô bằng >=3 cặp điểm chọn tay
(tính cả scale)  ->  (3) ICP tinh chỉnh  ->  (4) đo fitness/RMSE trước & sau.

Có 3 chế độ:

  1) demo  : tự sinh dữ liệu giả (một căn phòng) để test code khi chưa có dữ liệu thật
       python src/04_registration.py demo --out data/demo

  2) pick  : mở cửa sổ chọn điểm tương ứng trên point cloud và trên BIM, lưu vào JSON
       python src/04_registration.py pick --source data/pc_output/fused.ply \
              --target data/bim/prepared/bim_full_points.ply --picks data/reg/picks.json

  3) run   : làm sạch + căn chỉnh thô từ picks.json + ICP + xuất kết quả
       python src/04_registration.py run --source data/pc_output/fused.ply \
              --target data/bim/prepared/bim_full_points.ply --picks data/reg/picks.json \
              --out data/reg

Output của 'run':
    out/
    ├── transformation.txt      # ma trận 4x4 (đã gồm scale) đưa point cloud -> hệ BIM
    ├── registered_pc.ply       # point cloud đã làm sạch & đã căn chỉnh  -> giao cho Member 3
    ├── cleaned_pc.ply          # point cloud đã làm sạch (chưa căn chỉnh)
    └── metrics.json            # fitness, inlier RMSE trước/sau ICP, scale ước lượng

Thư viện: pip install open3d numpy
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d


# ----------------------------------------------------------------------------- tiện ích
def load_pc(path):
    pc = o3d.io.read_point_cloud(path)
    if pc.is_empty():
        # thử đọc như mesh rồi lấy mẫu điểm
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise SystemExit(f"[ERROR] Không đọc được {path}")
        pc = mesh.sample_points_uniformly(200000)
    return pc


def clean_point_cloud(pc, voxel, nb_neighbors=20, std_ratio=2.0):
    """Downsample + lọc nhiễu thống kê. voxel tính theo đơn vị của chính point cloud."""
    n0 = len(pc.points)
    pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"[CLEAN] {n0:,} -> {len(pc.points):,} điểm (voxel={voxel:.4f}, std_ratio={std_ratio})")
    return pc


def auto_voxel(pc, fraction=0.005):
    """Voxel bằng ~0.5% đường chéo bounding box (vì COLMAP không có đơn vị thật)."""
    ext = pc.get_axis_aligned_bounding_box().get_extent()
    return float(np.linalg.norm(ext) * fraction)


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
def pick_points(pc, title):
    print(f"\n=== {title} ===")
    print("  SHIFT + Click trái : chọn điểm      SHIFT + Click phải : bỏ điểm vừa chọn")
    print("  Chọn >= 4 điểm dễ nhận ra (góc tường, chân cột, góc cửa), THEO CÙNG MỘT THỨ TỰ ở cả 2 cửa sổ.")
    print("  Xong thì nhấn Q để đóng cửa sổ.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pc)
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    return np.asarray(pc.points)[idx].tolist()


def cmd_pick(a):
    src = load_pc(a.source)
    tgt = load_pc(a.target)
    p_src = pick_points(src, "1/2  POINT CLOUD (COLMAP)")
    p_tgt = pick_points(tgt, "2/2  BIM")
    if len(p_src) != len(p_tgt) or len(p_src) < 3:
        raise SystemExit(f"[ERROR] Cần >=3 cặp và số điểm 2 bên bằng nhau (đang có {len(p_src)} vs {len(p_tgt)}).")
    os.makedirs(os.path.dirname(os.path.abspath(a.picks)), exist_ok=True)
    with open(a.picks, "w") as f:
        json.dump({"source_points": p_src, "target_points": p_tgt}, f, indent=2)
    _, s = umeyama(p_src, p_tgt)
    print(f"[DONE] Đã lưu {len(p_src)} cặp điểm vào {a.picks}. Scale ước lượng = {s:.4f}")


# ----------------------------------------------------------------------------- chế độ run
def register(src_raw, tgt, p_src, p_tgt, out, std_ratio=2.0, voxels=(0.10, 0.05, 0.02),
             refine_scale=False, show=False):
    os.makedirs(out, exist_ok=True)

    # 1) làm sạch (trong đơn vị gốc của COLMAP)
    src = clean_point_cloud(src_raw, auto_voxel(src_raw, 0.002), std_ratio=std_ratio)
    o3d.io.write_point_cloud(os.path.join(out, "cleaned_pc.ply"), src)

    # 2) căn chỉnh thô bằng các cặp điểm chọn tay (có scale)
    T0, s = umeyama(p_src, p_tgt, with_scale=True)
    resid = np.linalg.norm((np.c_[p_src, np.ones(len(p_src))] @ T0.T)[:, :3] - np.asarray(p_tgt), axis=1)
    print(f"[INIT] scale={s:.4f}  sai số trên các điểm chọn: mean={resid.mean():.3f}m max={resid.max():.3f}m")
    if resid.max() > 0.5:
        print("[WARN] Sai số điểm chọn > 0.5m: có thể chọn sai thứ tự / sai điểm. Nên chọn lại.")

    # 3) ICP — đưa point cloud về hệ BIM (mét) trước, để các ngưỡng voxel/khoảng cách có nghĩa
    src_init = o3d.geometry.PointCloud(src)
    src_init.transform(T0)
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
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[RESULT] Trước ICP: fitness={before['fitness']:.3f} RMSE={before['inlier_rmse']:.4f}m")
    print(f"[RESULT] Sau  ICP: fitness={after['fitness']:.3f} RMSE={after['inlier_rmse']:.4f}m")
    print(f"[DONE] Kết quả tại {out}")

    if show:
        reg.paint_uniform_color([1.0, 0.3, 0.0])   # cam = point cloud
        t = o3d.geometry.PointCloud(tgt)
        t.paint_uniform_color([0.2, 0.4, 1.0])     # xanh = BIM
        o3d.visualization.draw_geometries([t, reg], window_name="Cam: point cloud | Xanh: BIM")
    return T, metrics


def cmd_run(a):
    src_raw = load_pc(a.source)
    tgt = load_pc(a.target)
    with open(a.picks) as f:
        picks = json.load(f)
    register(src_raw, tgt, picks["source_points"], picks["target_points"], a.out,
             std_ratio=a.std_ratio, voxels=tuple(a.voxels), refine_scale=a.refine_scale, show=a.show)


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

    r = sub.add_parser("run")
    r.add_argument("--source", required=True, help="point cloud COLMAP (.ply)")
    r.add_argument("--target", required=True, help="bim_full_points.ply từ 03_bim_parser.py")
    r.add_argument("--picks", required=True, help="file JSON tạo bởi lệnh pick")
    r.add_argument("--out", required=True)
    r.add_argument("--std-ratio", type=float, default=2.0, help="nhỏ hơn = lọc nhiễu mạnh hơn")
    r.add_argument("--voxels", type=float, nargs="+", default=[0.10, 0.05, 0.02], help="các mức ICP (mét)")
    r.add_argument("--refine-scale", action="store_true", help="cho ICP chỉnh tiếp cả scale (point-to-point)")
    r.add_argument("--show", action="store_true", help="hiển thị kết quả chồng lên nhau")

    a = ap.parse_args()
    {"demo": cmd_demo, "pick": cmd_pick, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
