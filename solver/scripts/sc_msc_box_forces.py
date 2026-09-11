# MSC AEROF(박스별 공력)와 ASCENT-Load aero_forces 를 박스·패널·스팬 단위로 대조한다 (스플라인 결합 추적 B)
"""Box-by-box aerodynamic force comparison against MSC Nastran.

MSC prints 'AERODYNAMIC FORCES ON THE AERODYNAMIC ELEMENTS' per subcase:
one row per box (GRID ID = box id) with T3 = force along the box normal
(MSC convention chord x span: left-hand panels point -z, the 40 deg
V-tail panels carry cos 40 = 0.766) and R2 = the box pitching moment.
T3 is projected to basic Fz with msc_normal_z before comparing. ASCENT-Load stores aero_forces (n_box, 3) and
aero_boxes (ids) in the .aload. Both codes trim to the same total lift,
so the *distribution* is what can differ — per CAERO panel and along
the span — and that distribution is where a normalwash/spline coupling
difference would show up.

Usage:  cd solver && python scripts/sc_msc_box_forces.py <deck-stem> [<deck-stem> ...]
        e.g. ilc8_msc_sol144_v9_holdout ilc8_msc_sol144_v6_rigid
Output: sc_msc_box_forces_<stem>.json next to this script.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
ILC8 = os.path.join(SOLVER, "tests", "validation", "ILC8")

_SC = re.compile(r"SUBCASE\s+(\d+)\s*$")
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+LS\s+([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)"
                  r"\s+([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)")


def parse_msc_aerof(path):
    """{subcase: {box_id: (T3, R2)}}."""
    out = {}
    sc = None
    in_blk = False
    with open(path, errors="ignore") as f:
        for line in f:
            m = _SC.search(line)
            if m:
                sc = int(m.group(1))
            if "AERODYNAMIC FORCES ON THE AERODYNAMIC ELEMENTS" in line:
                in_blk = True
                continue
            if not in_blk:
                continue
            s = line[1:] if line[:1] == "0" else line
            r = _ROW.match(s)
            if r:
                out.setdefault(sc, {})[int(r.group(2))] = (float(r.group(5)), float(r.group(7)))
            elif line.startswith("1") and "AERODYNAMIC FORCES" not in line:
                in_blk = False
    return out


def msc_normal_z(boxes):
    """MSC 관례(코드×스팬) 법선의 z 성분. ASCENT 는 법선을 +z/+y 로 정규화하므로
    normal_flipped 인 박스는 부호를 되돌린다. MSC AEROF T3 는 이 법선 방향 힘이라
    Fz(기본계) = T3 * n_z 로 환산한다 (좌측 패널 부호, V-테일 cos 40° 투영)."""
    return {int(b.box_id): float((-1.0 if b.normal_flipped else 1.0) * b.normal[2])
            for b in boxes}


def main(stems):
    from ascent_load.output.result_io import load_results
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.aero.panel import generate_all_panels

    for spec in stems:
        # 'stem' 또는 'stem:aload경로' (ASCENT 결과를 다른 실행으로 바꿔 끼울 때)
        stem, _, aload = spec.partition(":")
        msc = parse_msc_aerof(os.path.join(ILC8, f"{stem}_MSC.f06"))
        res, _ = load_results(aload or os.path.join(ILC8, f"{stem}.aload"))
        model = parse_bdf(os.path.join(ILC8, f"{stem}.bdf"))
        model.cross_reference()
        boxes = generate_all_panels(model, use_nastran_eid=True)
        geom = {}
        for b in boxes:
            c = np.asarray(b.doublet_point, float)
            geom[int(b.box_id)] = dict(y=float(c[1]), x=float(c[0]), z=float(c[2]),
                                       caero=int(b.box_id) // 100000,
                                       area=float(b.area))
        nz = msc_normal_z(boxes)
        out = {}
        print(f"\n================ {spec} ================")
        for sr in res.subcases:
            sc = sr.subcase_id
            if sc not in msc:
                continue
            ids = [int(getattr(b, 'box_id', b)) for b in sr.aero_boxes]
            fa = np.asarray(sr.aero_forces)[:, 2]
            a = dict(zip(ids, fa))
            common = sorted(set(a) & set(msc[sc]))
            fm = np.array([msc[sc][i][0] * nz.get(i, 1.0) for i in common])
            fs = np.array([a[i] for i in common])
            tot_m, tot_a = fm.sum(), fs.sum()
            # 패널(CAERO)별
            per_caero = {}
            for i, m_, s_ in zip(common, fm, fs):
                k = geom.get(i, {}).get("caero", 0)
                d = per_caero.setdefault(k, [0.0, 0.0, 0])
                d[0] += m_; d[1] += s_; d[2] += 1
            # 스팬 분포: 우익(y>0) 박스를 y 로 10 구간
            y = np.array([geom.get(i, {}).get("y", np.nan) for i in common])
            span = []
            if np.isfinite(y).any():
                ymax = np.nanmax(np.abs(y))
                edges = np.linspace(0, ymax, 11)
                for k in range(10):
                    sel = (np.abs(y) >= edges[k]) & (np.abs(y) < edges[k + 1] + (1e-9 if k == 9 else 0))
                    if sel.any():
                        span.append(dict(eta_lo=edges[k] / ymax, eta_hi=edges[k + 1] / ymax,
                                         n=int(sel.sum()), Fz_msc=float(fm[sel].sum()),
                                         Fz_al=float(fs[sel].sum())))
            l2 = float(np.linalg.norm(fs - fm) / max(np.linalg.norm(fm), 1e-12))
            out[sc] = dict(n_common=len(common), Fz_total_msc=float(tot_m), Fz_total_al=float(tot_a),
                           box_L2_rel=l2,
                           per_caero={str(k): dict(Fz_msc=v[0], Fz_al=v[1], n=v[2],
                                                    ratio=(v[1] / v[0] if abs(v[0]) > 1e-9 else None))
                                      for k, v in sorted(per_caero.items())},
                           span=span)
            print(f"SC{sc}: 공통 박스 {len(common)}  총 Fz MSC {tot_m:10.1f} / AL {tot_a:10.1f}  "
                  f"박스별 L2 편차 {l2 * 100:5.1f}%")
            for k, v in sorted(per_caero.items()):
                print(f"   CAERO {k:5d}: n={v[2]:3d}  Fz MSC {v[0]:9.1f}  AL {v[1]:9.1f}  "
                      f"비 {v[1] / v[0] if abs(v[0]) > 1e-9 else float('nan'):6.3f}")
            if span:
                print("   스팬 분포(|y| 10구간) Fz MSC / AL / 비:")
                for s_ in span:
                    print(f"     η {s_['eta_lo']:.1f}~{s_['eta_hi']:.1f}: {s_['Fz_msc']:8.1f} / "
                          f"{s_['Fz_al']:8.1f} / {s_['Fz_al'] / s_['Fz_msc'] if abs(s_['Fz_msc']) > 1e-9 else float('nan'):.3f}")
        with open(os.path.join(HERE, f"sc_msc_box_forces_{stem}.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ilc8_msc_sol144_v9_holdout"])
