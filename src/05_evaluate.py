"""05_evaluate.py - So sanh point cloud (da registration) voi BIM theo tung cau kien.

Voi moi cau kien BIM:
  * Deviation : khoang cach tu cac diem cua point cloud toi be mat mesh cau kien
                (moi diem duoc gan cho cau kien gan no nhat).
  * Coverage  : lay mau deu tren be mat cau kien (bo cac mat tiep xuc voi cau kien
                khac vi camera khong the thay), mot diem mau duoc tinh la "phu" neu
                trong ban kinh --cov-radius co diem cua point cloud.
  * Trang thai: Detected / Partial / Missing / Out of scan.

Dau vao:
    --pcd      point cloud da registration (.ply, don vi met, cung he toa do voi BIM)
    --bim-dir  thu muc do 03_bim_parser.py tao ra (elements.csv + elements/*.ply)

Dau ra (thu muc --out):
    results.csv            thong ke tung cau kien
    heatmap_deviation.ply  point cloud to mau theo sai so (xanh -> do, do = vuot --dev-threshold)
    bim_status.ply         mesh BIM to mau theo trang thai
    deviation_hist.png     histogram sai so

Vi du:
    python src/05_evaluate.py --demo                       # chay thu tren du lieu gia
    python src/05_evaluate.py --pcd data/pc_output/registered.ply --bim-dir data/bim --out results
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

STATUS_COLORS = {
    "Detected": (0.20, 0.70, 0.30),
    "Partial": (0.95, 0.70, 0.10),
    "Missing": (0.85, 0.15, 0.15),
    "Out of scan": (0.65, 0.65, 0.65),
}


# ----------------------------------------------------------------------------- I/O
def load_bim(bim_dir):
    bim_dir = Path(bim_dir)
    table = pd.read_csv(bim_dir / "elements.csv", encoding="utf-8-sig")
    meshes = []
    for gid in table["GlobalId"]:
        m = o3d.io.read_triangle_mesh(str(bim_dir / "elements" / f"{gid}.ply"))
        m.compute_triangle_normals()
        meshes.append(m)
    return table, meshes


# ----------------------------------------------------------------------- core logic
def build_scene(meshes):
    scene = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))  # id = thu tu
    return scene


def assign_points(scene, points):
    """Khoang cach tu moi diem toi BIM va id cau kien gan nhat."""
    q = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(q)
    closest = ans["points"].numpy()
    ids = ans["geometry_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(points - closest, axis=1)
    return dist, ids


def visible_samples(mesh, scene, spacing, contact_eps, max_samples):
    """Lay mau tren be mat, bo cac diem nam tren mat tiep xuc voi cau kien khac."""
    n = int(np.clip(mesh.get_surface_area() / spacing ** 2, 50, max_samples))
    pc = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=True)
    pts, nrm = np.asarray(pc.points), np.asarray(pc.normals)
    probe = o3d.core.Tensor(pts + nrm * contact_eps, dtype=o3d.core.Dtype.Float32)
    try:
        inside = scene.compute_occupancy(probe).numpy() > 0.5
    except Exception:          # mesh khong kin -> bo qua buoc loai mat tiep xuc
        inside = np.zeros(len(pts), bool)
    return pts[~inside]


def evaluate(points, table, meshes, args):
    scene = build_scene(meshes)
    dist, ids = assign_points(scene, points)
    assoc = dist <= args.max_assoc               # diem thuoc ve BIM (con lai la clutter)
    tree = cKDTree(points)

    pmin, pmax = points.min(0) - args.max_assoc, points.max(0) + args.max_assoc
    rows = []
    for i, (row, mesh) in enumerate(zip(table.itertuples(), meshes)):
        bmin = np.array([row.min_x, row.min_y, row.min_z])
        bmax = np.array([row.max_x, row.max_y, row.max_z])
        in_scan = bool(np.all(bmax >= pmin) and np.all(bmin <= pmax))

        d = dist[assoc & (ids == i)]
        rec = {
            "GlobalId": row.GlobalId, "ifc_type": row.ifc_type, "name": row.name,
            "storey": row.storey, "n_points": len(d),
            "coverage_pct": np.nan, "mean_dev_cm": np.nan, "median_dev_cm": np.nan,
            "max_dev_cm": np.nan, "rmse_cm": np.nan, "pct_over_thr": np.nan,
        }
        if not in_scan:
            rec["status"] = "Out of scan"
            rows.append(rec)
            continue

        samples = visible_samples(mesh, scene, args.sample_spacing, args.contact_eps, args.max_samples)
        if len(samples):
            nn, _ = tree.query(samples, k=1, distance_upper_bound=args.cov_radius)
            coverage = float(np.mean(np.isfinite(nn)))
        else:
            coverage = 0.0
        rec["coverage_pct"] = round(100 * coverage, 1)

        if len(d):
            rec.update(mean_dev_cm=100 * d.mean(), median_dev_cm=100 * np.median(d),
                       max_dev_cm=100 * d.max(), rmse_cm=100 * np.sqrt(np.mean(d ** 2)),
                       pct_over_thr=100 * np.mean(d > args.dev_threshold))

        mean_dev = d.mean() if len(d) else np.inf
        if coverage >= args.t_cov and mean_dev <= args.t_dev:
            rec["status"] = "Detected"
        elif coverage < args.t_missing:
            rec["status"] = "Missing"
        else:
            rec["status"] = "Partial"
        rows.append(rec)

    df = pd.DataFrame(rows).round(2)
    return df, dist, assoc


# ---------------------------------------------------------------------------- output
def deviation_colors(dist, assoc, thr):
    """Xanh (0) -> vang -> do (>= thr). Diem khong thuoc BIM: xam."""
    t = np.clip(dist / thr, 0, 1)[:, None]
    green, yellow, red = np.array([0.1, 0.7, 0.2]), np.array([1.0, 0.85, 0.1]), np.array([0.9, 0.1, 0.1])
    col = np.where(t < 0.5, green + (yellow - green) * (t / 0.5), yellow + (red - yellow) * ((t - 0.5) / 0.5))
    col[dist >= thr] = red
    col[~assoc] = 0.6
    return col


def save_outputs(out, points, dist, assoc, df, meshes, args):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(deviation_colors(dist, assoc, args.dev_threshold))
    o3d.io.write_point_cloud(str(out / "heatmap_deviation.ply"), pc)

    merged = o3d.geometry.TriangleMesh()
    for status, m in zip(df["status"], meshes):
        mc = o3d.geometry.TriangleMesh(m)
        mc.paint_uniform_color(STATUS_COLORS[status])
        merged += mc
    o3d.io.write_triangle_mesh(str(out / "bim_status.ply"), merged)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(100 * dist[assoc], bins=60, color="#3a7bd5")
    ax.axvline(100 * args.dev_threshold, color="red", ls="--", label=f"nguong {100 * args.dev_threshold:.0f} cm")
    ax.set_xlabel("Khoang cach toi BIM (cm)")
    ax.set_ylabel("So diem")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "deviation_hist.png", dpi=150)
    plt.close(fig)


def print_summary(df, dist, assoc, args):
    print("\n=== KET QUA ===")
    print(df["status"].value_counts().to_string())
    d = dist[assoc]
    if len(d):
        print(f"\nDiem thuoc BIM: {assoc.sum()}/{len(assoc)} ({100 * assoc.mean():.1f}%)")
        print(f"Sai so: mean={100 * d.mean():.2f} cm  median={100 * np.median(d):.2f} cm  "
              f"max={100 * d.max():.2f} cm  >{100 * args.dev_threshold:.0f}cm: {100 * np.mean(d > args.dev_threshold):.1f}%")
    cols = ["GlobalId", "ifc_type", "n_points", "coverage_pct", "mean_dev_cm", "status"]
    print("\n" + df[df.status != "Out of scan"][cols].to_string(index=False))


# ------------------------------------------------------------------------- demo data
def make_demo(demo_dir, rng=np.random.default_rng(0)):
    """Tao BIM gia (2 cot, 1 dam, 1 san) va point cloud gia:
       COT_A thi cong du, COT_B lech 6 cm, DAM_C moi lam mot nua, SAN_D chua thi cong."""
    elem_dir = demo_dir / "bim" / "elements"
    elem_dir.mkdir(parents=True, exist_ok=True)

    def box(size, origin):
        m = o3d.geometry.TriangleMesh.create_box(*size)
        m.translate(origin)
        m.compute_triangle_normals()
        return m

    specs = [  # GlobalId, loai, kich thuoc, goc
        ("COT_A", "IfcColumn", (0.4, 0.4, 3.0), (0.0, 0.0, 0.0)),
        ("COT_B", "IfcColumn", (0.4, 0.4, 3.0), (4.0, 0.0, 0.0)),
        ("DAM_C", "IfcBeam", (3.6, 0.3, 0.5), (0.4, 0.05, 2.5)),
        ("SAN_D", "IfcSlab", (4.4, 3.0, 0.2), (0.0, 0.0, 3.0)),
    ]
    rows, meshes = [], {}
    for gid, typ, size, origin in specs:
        m = box(size, origin)
        meshes[gid] = m
        o3d.io.write_triangle_mesh(str(elem_dir / f"{gid}.ply"), m)
        bmin, bmax = np.array(origin), np.array(origin) + np.array(size)
        rows.append(dict(GlobalId=gid, ifc_type=typ, name=gid, storey="DEMO", storey_elev_m=0.0,
                         area_m2=m.get_surface_area(), watertight=True,
                         min_x=bmin[0], min_y=bmin[1], min_z=bmin[2],
                         max_x=bmax[0], max_y=bmax[1], max_z=bmax[2]))
    pd.DataFrame(rows).to_csv(demo_dir / "bim" / "elements.csv", index=False, encoding="utf-8-sig")

    def scan(mesh, n, shift=(0, 0, 0), keep=None, noise=0.008):
        pts = np.asarray(mesh.sample_points_uniformly(n).points)
        if keep is not None:
            pts = pts[keep(pts)]
        return pts + np.array(shift) + rng.normal(0, noise, pts.shape)

    clouds = [
        scan(meshes["COT_A"], 20000),
        scan(meshes["COT_B"], 20000, shift=(0.06, 0, 0)),
        scan(meshes["DAM_C"], 20000, keep=lambda p: p[:, 0] < 2.2),
        rng.uniform([-1, -1, 0], [5.5, 4, 3.5], (800, 3)),   # nhieu / do vat trong cong truong
    ]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(clouds)))
    o3d.io.write_point_cloud(str(demo_dir / "registered_demo.ply"), pcd)
    print(f"Da tao du lieu demo trong {demo_dir}")
    return demo_dir / "registered_demo.ply", demo_dir / "bim"


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", type=Path, help="point cloud da registration (.ply, met)")
    ap.add_argument("--bim-dir", type=Path, default=Path("data/bim"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--demo", action="store_true", help="tao du lieu gia va chay thu")
    ap.add_argument("--voxel", type=float, default=0.0, help="downsample point cloud (m), 0 = khong")
    ap.add_argument("--max-assoc", type=float, default=0.15, help="diem xa BIM hon muc nay la clutter (m)")
    ap.add_argument("--dev-threshold", type=float, default=0.05, help="nguong to do tren heatmap (m)")
    ap.add_argument("--cov-radius", type=float, default=0.05, help="ban kinh tinh coverage (m)")
    ap.add_argument("--sample-spacing", type=float, default=0.03, help="khoang cach lay mau tren BIM (m)")
    ap.add_argument("--contact-eps", type=float, default=0.02, help="do day kiem tra mat tiep xuc (m)")
    ap.add_argument("--max-samples", type=int, default=200000)
    ap.add_argument("--t-cov", type=float, default=0.6, help="coverage toi thieu de Detected")
    ap.add_argument("--t-dev", type=float, default=0.03, help="sai so trung binh toi da de Detected (m)")
    ap.add_argument("--t-missing", type=float, default=0.1, help="coverage duoi muc nay la Missing")
    args = ap.parse_args()

    if args.demo:
        args.pcd, args.bim_dir = make_demo(Path("data/demo"))
        if args.out == Path("results"):
            args.out = Path("results/demo")
    if args.pcd is None:
        ap.error("can --pcd hoac --demo")

    pcd = o3d.io.read_point_cloud(str(args.pcd))
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    points = np.asarray(pcd.points, dtype=float)
    print(f"Point cloud: {len(points)} diem")

    table, meshes = load_bim(args.bim_dir)
    print(f"BIM: {len(meshes)} cau kien")

    df, dist, assoc = evaluate(points, table, meshes, args)
    save_outputs(args.out, points, dist, assoc, df, meshes, args)
    print_summary(df, dist, assoc, args)
    print(f"\nDa luu ket qua vao {args.out}")


if __name__ == "__main__":
    main()
