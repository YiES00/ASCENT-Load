# MSC 의 적용하중을 그대로 넣은 SOL 101 로 구조 강성만 분리 대조한다 (r5 MC5 추적)
"""Isolate K: solve ASCENT-Load SOL 101 under MSC's own applied load.

The hold-out comparison shows ASCENT-Load's elastic field is one scalar
(~1/1.215) times MSC's, while the applied nodal loads agree to 0.1 %
and the factor is independent of q (so not the aeroelastic coupling).
That leaves the structural stiffness. This script removes everything
else: it takes the MSC GPFORCE 'APP-LOAD' of one subcase verbatim as
FORCE/MOMENT cards, restrains the SUPORT DOFs with SPC1 (3-2-1, statically
determinate, so no stiffness is added to a self-balanced load) and
solves plain SOL 101 in ASCENT-Load. If the elastic field is still
~1/1.215 of MSC's, the discrepancy is in K.

Usage:  cd solver && python scripts/r5_msc_k_isolation.py [subcase=1]
Output: tests/validation/ILC8/ilc8_k_isolation_sc<N>.{bdf,f06} and a
        printed scale factor.
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)
from r3_msc_displacement_metrics import parse_displacements, remove_rigid  # noqa
from r5_msc_holdout_metrics import parse_gpforce_applied, scale_fit, REGIONS  # noqa

ILC8 = os.path.join(SOLVER, "tests", "validation", "ILC8")
STEM = "ilc8_msc_sol144_v9_holdout"
SC = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT = os.path.join(ILC8, f"ilc8_k_isolation_sc{SC}")

KEEP = {"GRID", "CQUAD4", "CBAR", "CONM2", "RBE2", "PBAR", "PSHELL",
        "MAT1", "CORD2R", "PARAM"}
DROP_PARAM = {"AUNITS"}


def build_deck():
    src = os.path.join(ILC8, f"{STEM}.bdf")
    app = parse_gpforce_applied(os.path.join(ILC8, f"{STEM}_MSC.f06"))[SC]
    L = ["$ K 분리 검정 덱 -- MSC SC%d APP-LOAD 를 그대로 FORCE/MOMENT 로 적용" % SC,
         "$ 생성: scripts/r5_msc_k_isolation.py (구조 카드는 v9 hold-out 덱 그대로)",
         "SOL     101", "CEND",
         f"TITLE    = ILC-8 K ISOLATION - MSC SC{SC} APPLIED LOAD, SOL 101",
         "ECHO     = NONE", "SPC      = 1", "LOAD     = 100", "DISP     = ALL",
         "BEGIN BULK"]
    keep = False
    with open(src) as f:
        in_bulk = False
        for line in f:
            if line.startswith("BEGIN BULK"):
                in_bulk = True
                continue
            if not in_bulk or line.startswith("ENDDATA"):
                continue
            if line.startswith("$") or not line.strip():
                continue
            name = line[:8].strip()
            if line[:1] in "+*" or name == "":
                if keep:
                    L.append(line.rstrip("\n"))
                continue
            base = name.rstrip("*")
            keep = base in KEEP
            if base == "PARAM" and line[8:16].strip() in DROP_PARAM:
                keep = False
            if keep:
                L.append(line.rstrip("\n"))
    # 3-2-1 기준 마운트 (원 덱의 SUPORT 와 동일 자유도)
    L += ["SPC1    1       123     101519",
          "SPC1    1       23      101719",
          "SPC1    1       3       101501"]
    n_f = n_m = 0
    for nid in sorted(app):
        v = app[nid]
        fm = float(np.linalg.norm(v[:3]))
        if fm > 1e-12:
            n = v[:3] / fm
            L.append("FORCE*  %16d%16d%16d%16.8E" % (100, nid, 0, fm))
            L.append("*       %16.8E%16.8E%16.8E" % tuple(n))
            n_f += 1
        mm = float(np.linalg.norm(v[3:]))
        if mm > 1e-12:
            n = v[3:] / mm
            L.append("MOMENT* %16d%16d%16d%16.8E" % (100, nid, 0, mm))
            L.append("*       %16.8E%16.8E%16.8E" % tuple(n))
            n_m += 1
    L.append("ENDDATA")
    with open(OUT + ".bdf", "w") as f:
        f.write("\n".join(L) + "\n")
    tot = np.sum([app[n][:3] for n in app], axis=0)
    print(f"덱 작성: {OUT}.bdf  (FORCE {n_f}, MOMENT {n_m}, 적용하중 합 "
          f"{tot} N)")


def main():
    build_deck()
    r = subprocess.run([sys.executable, "-m", "ascent_load", OUT + ".bdf"],
                       cwd=SOLVER, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit("SOL 101 실패")
    from ascent_load.bdf.parser import parse_bdf
    model = parse_bdf(os.path.join(ILC8, f"{STEM}.bdf"))
    d_msc = parse_displacements(os.path.join(ILC8, f"{STEM}_MSC.f06"))[SC]
    d101 = parse_displacements(OUT + ".f06")
    d101 = d101[sorted(d101)[0]]
    common = sorted(set(d_msc) & set(d101) & set(model.nodes))
    xyz = np.array([model.nodes[n].xyz_global for n in common])
    em = remove_rigid(xyz, np.array([d_msc[n] for n in common]))
    e1 = remove_rigid(xyz, np.array([d101[n] for n in common]))
    s, l0, ls = scale_fit(e1, em)
    print(f"\nSOL 101(ASCENT, MSC 하중) 대 MSC SOL 144 SC{SC} 탄성장 "
          f"(공통 {len(common)} 절점)")
    print(f"  단일 배율 s = {s:.5f}  (1/s = {1 / s:.5f});  L2 {l0 * 100:.2f}% "
          f"-> 배율 후 {ls * 100:.2f}%")
    for nm, lo, hi in REGIONS:
        ii = np.array([i for i, n in enumerate(common) if lo <= n <= hi])
        if len(ii) < 10:
            continue
        s_r, l0_r, ls_r = scale_fit(e1[ii, 2], em[ii, 2])
        print(f"  {nm:18s} n={len(ii):5d}  s(T3)={s_r:.5f}  배율 후 L2 {ls_r * 100:.2f}%")
    for k, nm in enumerate(("T1", "T2", "T3")):
        s_c, _, ls_c = scale_fit(e1[:, k], em[:, k])
        print(f"  성분 {nm}: s={s_c:.5f}  배율 후 L2 {ls_c * 100:.2f}%")


if __name__ == "__main__":
    main()
