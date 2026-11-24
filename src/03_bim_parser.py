"""03_bim_parser.py - Doc file IFC, tach tung cau kien (cot, dam, san) thanh mesh .ply.

Dau ra (mac dinh trong data/bim/):
    elements/<GlobalId>.ply   mesh tung cau kien (don vi: met, toa do the gioi)
    elements.csv              GlobalId, loai, ten, tang, cao do, dien tich, bounding box
    bim_merged.ply            tat ca cau kien gop lai (de M2 dung cho registration)

Vi du:
    python src/03_bim_parser.py --ifc data/bim/PLM.ifc --list-storeys
    python src/03_bim_parser.py --ifc data/bim/PLM.ifc --storey "TẦNG 2"
    python src/03_bim_parser.py --ifc data/bim/PLM.ifc --types IfcColumn IfcSlab

Luu y: file IFC luu don vi mm, nhung ifcopenshell.geom tra ve hinh hoc theo met.
"""

import argparse
import csv
import multiprocessing
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ue
import ifcopenshell.util.unit
import numpy as np
import open3d as o3d

DEFAULT_TYPES = ["IfcColumn", "IfcBeam", "IfcSlab"]


def storey_of(element):
    """Tra ve (ten tang, cao do tang [m]) chua cau kien, hoac ("", nan)."""
    container = ue.get_container(element)
    while container is not None and not container.is_a("IfcBuildingStorey"):
        container = ue.get_container(container) if container.is_a("IfcElement") else None
    if container is None:
        return "", float("nan")
    elev = container.Elevation if container.Elevation is not None else float("nan")
    return container.Name or "", elev


def list_storeys(model):
    unit_scale = ifcopenshell.util.unit.calculate_unit_scale(model)
    print("Cac tang trong file IFC:")
    for s in sorted(model.by_type("IfcBuildingStorey"), key=lambda s: s.Elevation or 0):
        n = sum(1 for e in ue.get_decomposition(s) if e.is_a() in DEFAULT_TYPES)
        print(f"  {s.Name:<30} cao do = {(s.Elevation or 0) * unit_scale:8.2f} m   ({n} cot/dam/san)")


def parse(ifc_path, out_dir, types, storey_filter=None):
    model = ifcopenshell.open(str(ifc_path))
    unit_scale = ifcopenshell.util.unit.calculate_unit_scale(model)  # mm -> m = 0.001

    elements = []
    for t in types:
        for e in model.by_type(t):
            name, elev = storey_of(e)
            if storey_filter and storey_filter.lower() not in name.lower():
                continue
            elements.append((e, name, elev * unit_scale))
    if not elements:
        raise SystemExit("Khong tim thay cau kien nao. Kiem tra --types / --storey (xem --list-storeys).")
    info = {e.id(): (name, elev) for e, name, elev in elements}

    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)

    elem_dir = out_dir / "elements"
    elem_dir.mkdir(parents=True, exist_ok=True)

    rows, merged = [], o3d.geometry.TriangleMesh()
    it = ifcopenshell.geom.iterator(settings, model, multiprocessing.cpu_count(),
                                    include=[e for e, _, _ in elements])
    if not it.initialize():
        raise SystemExit("ifcopenshell khong tao duoc hinh hoc.")
    while True:
        shape = it.get()
        elem = model.by_id(shape.id)
        verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.asarray(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
        if len(faces):
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                             o3d.utility.Vector3iVector(faces))
            mesh.remove_duplicated_vertices()
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(elem_dir / f"{elem.GlobalId}.ply"), mesh)
            merged += mesh
            bmin, bmax = verts.min(0), verts.max(0)
            name, elev = info[shape.id]
            rows.append({
                "GlobalId": elem.GlobalId,
                "ifc_type": elem.is_a(),
                "name": elem.Name or "",
                "storey": name,
                "storey_elev_m": round(elev, 3),
                "area_m2": round(mesh.get_surface_area(), 4),
                "watertight": mesh.is_watertight(),
                "min_x": bmin[0], "min_y": bmin[1], "min_z": bmin[2],
                "max_x": bmax[0], "max_y": bmax[1], "max_z": bmax[2],
            })
        if not it.next():
            break

    with open(out_dir / "elements.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    o3d.io.write_triangle_mesh(str(out_dir / "bim_merged.ply"), merged)

    print(f"Don vi IFC: 1 don vi = {unit_scale} m. Hinh hoc xuat ra theo met.")
    print(f"Da xuat {len(rows)} cau kien -> {elem_dir}")
    for t in types:
        print(f"  {t:<12} {sum(r['ifc_type'] == t for r in rows)}")
    ext = merged.get_axis_aligned_bounding_box()
    print(f"Pham vi BIM (m): min={np.round(ext.min_bound, 2)}  max={np.round(ext.max_bound, 2)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ifc", default="data/bim/PLM.ifc", type=Path)
    ap.add_argument("--out", default="data/bim", type=Path)
    ap.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    ap.add_argument("--storey", help="chi lay tang co ten chua chuoi nay, vd 'TẦNG 2'")
    ap.add_argument("--list-storeys", action="store_true")
    args = ap.parse_args()

    if args.list_storeys:
        list_storeys(ifcopenshell.open(str(args.ifc)))
        return
    parse(args.ifc, args.out, args.types, args.storey)


if __name__ == "__main__":
    main()
