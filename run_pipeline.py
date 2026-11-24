"""run_pipeline.py - Chay toan bo pipeline 01 -> 05 theo config.yaml.

Moi buoc la mot script trong src/. File nay KHONG chua thuat toan, chi:
  * doc config.yaml (duong dan + tham so),
  * kiem tra buoc da duoc code chua, du file dau vao chua,
  * chay lan luot tung buoc, dung lai neu mot buoc loi,
  * in bang tong ket.

Cach dung:
    python run_pipeline.py --check          # chi xem trang thai tung buoc, khong chay gi
    python run_pipeline.py                  # chay tat ca cac buoc chay duoc
    python run_pipeline.py --steps 3 5      # chi chay buoc 3 va 5
    python run_pipeline.py --demo           # chay buoc 5 tren du lieu gia
    python run_pipeline.py --dry-run        # in lenh se chay, khong chay that
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STUB_MARK = "TODO: Implement"

# trang thai cua tung buoc
DONE, FAILED, SKIP_OFF, SKIP_TODO, SKIP_INPUT, SKIP_PREV, READY, NOT_SELECTED = (
    "OK", "LOI", "Tat", "Chua code", "Thieu dau vao", "Buoc truoc loi", "San sang", "-")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    fill = lambda s: str(s).format(**paths)
    for st in cfg["steps"]:
        st["inputs"] = [fill(p) for p in st.get("inputs", [])]
        st["outputs"] = [fill(p) for p in st.get("outputs", [])]
        st["args"] = [fill(a) for a in st.get("args", [])]
    return cfg


def is_stub(script):
    """Script chi co dong TODO (chua ai viet)?"""
    p = ROOT / script
    return not p.exists() or STUB_MARK in p.read_text(encoding="utf-8", errors="ignore")


def missing(files):
    return [f for f in files if not (ROOT / f).exists()]


def check_step(st):
    """Tra ve (trang thai, ghi chu) neu chay buoc nay bay gio."""
    if not st.get("enabled", True):
        return SKIP_OFF, "enabled: false trong config"
    if is_stub(st["script"]):
        return SKIP_TODO, f"{st['script']} van la TODO"
    miss = missing(st["inputs"])
    if miss:
        return SKIP_INPUT, "thieu: " + ", ".join(miss)
    return READY, ""


def run_step(st, dry_run):
    cmd = [sys.executable, str(ROOT / st["script"])] + st["args"]
    print("    $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    if dry_run:
        return DONE, "dry-run"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    code = subprocess.call(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    if code != 0:
        return FAILED, f"ma loi {code} sau {dt:.1f}s"
    miss = missing(st["outputs"])
    if miss:
        return FAILED, f"chay xong nhung khong thay: {', '.join(miss)}"
    return DONE, f"{dt:.1f}s"


def print_table(rows):
    print("\n" + "=" * 78)
    print(f"{'Buoc':<5}{'Ten':<14}{'Trang thai':<17}Ghi chu")
    print("-" * 78)
    for st, status, note in rows:
        print(f"{st['id']:<5}{st['name']:<14}{status:<17}{note}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--steps", nargs="+", type=int, help="chi chay cac buoc nay, vd: --steps 3 5")
    ap.add_argument("--check", action="store_true", help="chi kiem tra trang thai, khong chay")
    ap.add_argument("--dry-run", action="store_true", help="in lenh, khong chay")
    ap.add_argument("--demo", action="store_true", help="chay buoc 5 voi du lieu gia")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = cfg["steps"]

    if args.demo:
        args.steps = [5]
        for st in steps:
            if st["name"] == "evaluate":
                st["inputs"], st["args"] = [], ["--demo"]
                st["outputs"] = ["results/demo/results.csv"]

    selected = set(args.steps) if args.steps else {st["id"] for st in steps}
    rows, prev_failed = [], False

    for st in steps:
        if st["id"] not in selected:
            rows.append((st, NOT_SELECTED, "khong chon"))
            continue
        print(f"\n[{st['id']}] {st['title']}")
        if prev_failed:
            status, note = SKIP_PREV, ""
        else:
            status, note = check_step(st)
            if status == READY and not args.check:
                status, note = run_step(st, args.dry_run)
                prev_failed = status == FAILED
        print(f"    -> {status}{': ' + note if note else ''}")
        rows.append((st, status, note))

    print_table(rows)
    if any(s == FAILED for _, s, _ in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
