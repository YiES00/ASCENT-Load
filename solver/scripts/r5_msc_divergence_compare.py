# MSC DIVERG 결과(GACOMP)를 ASCENT-Load 발산 동압과 마하별로 대조한다 (r5 MC2 잔여)
"""Compare MSC Nastran divergence dynamic pressures with ASCENT-Load's.

Reads the 'D I V E R G E N C E      S U M M A R Y' blocks (one per Mach)
from the three GACOMP decks run in MSC Nastran 2016 —
  v0_smoke   : native 7-DOF restraint, M0.50 only, 3 roots
  v1_matched : native 7 DOF + the study's virtual 3-2-1 mount (13 DOF)
  v2_native  : native 7 DOF only
— takes the smallest POSITIVE divergence dynamic pressure per Mach (the
only physically meaningful one, per the MSC Aeroelastic Analysis User's
Guide) and tabulates it against the flight dynamic pressure and against
ASCENT-Load's two spline constructions (production operator, eps 1e-12,
from docs/dissertation/scripts/outputs/r4_divergence_final.csv).

MSC solves [K_ll - lambda Q_ll] u = 0 with complex Lanczos and never
inverts K, so its q_div does not depend on any regularization.

Usage:  cd solver && python scripts/r5_msc_divergence_compare.py
Data:   tests/validation/GACOMP/gacomp_msc_diverg_v{0,1,2}_*_MSC.f06
        (proprietary directory, git-ignored)
Output: r5_msc_divergence_compare.json next to this script.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
GACOMP = os.path.join(SOLVER, "tests", "validation", "GACOMP")
AL_CSV = os.path.join(SOLVER, "..", "docs", "dissertation", "scripts",
                      "outputs", "r4_divergence_final.csv")
DECKS = ["gacomp_msc_diverg_v0_smoke", "gacomp_msc_diverg_v1_matched",
         "gacomp_msc_diverg_v2_native"]

_MACH = re.compile(r"MACH NUMBER\s*=\s*([-+.\dE]+)")
_ROW = re.compile(r"^\s*(\d+)\s+([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)\s*$")


def parse_divergence(path):
    """{mach: [(root, q_div, re, im), ...]} from the DIVERGENCE SUMMARY blocks."""
    out = {}
    mach = None
    in_blk = False
    with open(path, errors="ignore") as f:
        for line in f:
            m = _MACH.search(line)
            if m and "METHOD" in line:
                mach = round(float(m.group(1)), 3)
                out[mach] = []
                in_blk = True
                continue
            if in_blk:
                r = _ROW.match(line)
                if r:
                    out[mach].append((int(r.group(1)), float(r.group(2)),
                                      float(r.group(3)), float(r.group(4))))
                elif line.startswith("1"):        # 새 페이지
                    in_blk = False
    return out


def load_ascent(path):
    """{mach: {'q_flight':..., 'rotation':..., 'surface':...}} at production eps."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("production", "").lower() != "true":
                continue
            mach = round(float(row["mach"]), 3)
            d = out.setdefault(mach, {"q_flight": float(row["q_flight"])})
            d[row["method"]] = float(row["q_div"])
    return out


def main():
    al = load_ascent(AL_CSV)
    if not al:
        print(f"경고 -- ASCENT-Load 발산 CSV 없음: {AL_CSV}")
    out = {}
    for deck in DECKS:
        path = os.path.join(GACOMP, f"{deck}_MSC.f06")
        if not os.path.isfile(path):
            print(f"없음: {path}")
            continue
        d = parse_divergence(path)
        out[deck] = {}
        print(f"\n=== {deck} ===")
        print(f"{'M':>4} {'q_flight':>10} {'MSC q_div':>12} {'여유':>9} {'#양근':>4} "
              f"{'AL 표면':>10} {'MSC/표면':>9} {'AL 회전':>10} {'MSC/회전':>10}")
        for mach, rows in sorted(d.items()):
            pos = [r for r in rows if r[1] > 0]
            rec = dict(n_roots=len(rows), n_positive=len(pos),
                       roots=[dict(root=r[0], q_div=r[1], re=r[2], im=r[3]) for r in rows])
            a = al.get(mach, {})
            qf = a.get("q_flight")
            if pos:
                qd = min(r[1] for r in pos)
                rec.update(q_div_min=qd, margin=(qd / qf if qf else None),
                           ratio_to_surface=(qd / a["surface"] if "surface" in a else None),
                           ratio_to_rotation=(qd / a["rotation"] if "rotation" in a else None))
                print(f"{mach:>4} {qf if qf else float('nan'):10.4e} {qd:12.5e} "
                      f"{rec['margin'] if rec['margin'] else float('nan'):9.2f} {len(pos):>4} "
                      f"{a.get('surface', float('nan')):10.4e} "
                      f"{rec['ratio_to_surface'] if rec['ratio_to_surface'] else float('nan'):9.1f} "
                      f"{a.get('rotation', float('nan')):10.4e} "
                      f"{rec['ratio_to_rotation'] if rec['ratio_to_rotation'] else float('nan'):10.1f}")
            else:
                print(f"{mach:>4}  양의 발산근 없음 ({len(rows)}개)")
            out[deck][str(mach)] = rec
    with open(os.path.join(HERE, "r5_msc_divergence_compare.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
