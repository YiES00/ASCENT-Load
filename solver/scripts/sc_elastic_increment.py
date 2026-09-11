# 박스별 탄성 공력 증분(유연 − 강체)을 MSC 와 ASCENT-Load 에서 각각 뽑아 부호·분포를 대조한다 (스플라인 결합 추적 B4)
"""Elastic aerodynamic increment per box, MSC vs ASCENT-Load.

Both codes trim rigid (v6, AEQR=0 in MSC; E x 1e6 in ASCENT-Load) and
flexible (v8) on the same seven conditions. The difference
F_flex - F_rigid per box is the aeroelastic increment each code
produces. Comparing the two increments spanwise and chordwise shows
whether the coupling differs in sign, magnitude, or location.

Note the trim variables differ between rigid and flexible runs, so the
increment includes the alpha/elevator re-trim; both codes are treated
the same way, so the comparison stays like for like.

Usage:  cd solver && python scripts/sc_elastic_increment.py <rigid_aload_ascent>
Output: sc_elastic_increment.json next to this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)
from sc_msc_box_forces import parse_msc_aerof, msc_normal_z, ILC8  # noqa: E402


def main(rigid_aload):
    from ascent_load.output.result_io import load_results
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.aero.panel import generate_all_panels

    msc_r = parse_msc_aerof(os.path.join(ILC8, "ilc8_msc_sol144_v6_rigid_MSC.f06"))
    msc_f = parse_msc_aerof(os.path.join(ILC8, "ilc8_msc_sol144_v8_shellbend_MSC.f06"))
    al_r, _ = load_results(rigid_aload)
    al_f, _ = load_results(os.path.join(ILC8, "ilc8_msc_sol144_v8_shellbend.aload"))
    model = parse_bdf(os.path.join(ILC8, "ilc8_msc_sol144_v8_shellbend.bdf"))
    model.cross_reference()
    boxes = generate_all_panels(model, use_nastran_eid=True)
    geom = {int(b.box_id): (float(b.doublet_point[0]), float(b.doublet_point[1]),
                            float(b.chord), int(b.box_id) // 100000) for b in boxes}
    nz = msc_normal_z(boxes)   # MSC T3(법선 방향) → 기본계 Fz

    def al_map(res, sc):
        sr = next(s for s in res.subcases if s.subcase_id == sc)
        return dict(zip((int(getattr(b, 'box_id', b)) for b in sr.aero_boxes),
                        np.asarray(sr.aero_forces)[:, 2]))

    out = {}
    for sc in sorted(set(msc_r) & set(msc_f)):
        ar, af = al_map(al_r, sc), al_map(al_f, sc)
        ids = sorted(set(msc_r[sc]) & set(msc_f[sc]) & set(ar) & set(af))
        dm = np.array([(msc_f[sc][i][0] - msc_r[sc][i][0]) * nz[i] for i in ids])
        da = np.array([af[i] - ar[i] for i in ids])
        fr = np.array([msc_r[sc][i][0] * nz[i] for i in ids])
        y = np.array([geom[i][1] for i in ids])
        g = np.array([geom[i][3] for i in ids])
        rec = dict(n=len(ids), dF_total_msc=float(dm.sum()), dF_total_al=float(da.sum()),
                   F_rigid_total=float(fr.sum()))
        print(f"\nSC{sc}: 강체 총 Fz {fr.sum():9.1f}  탄성 증분 총합 MSC {dm.sum():8.1f} / AL {da.sum():8.1f}")
        per = {}
        for k in sorted(set(g)):
            sel = g == k
            per[str(k)] = dict(dF_msc=float(dm[sel].sum()), dF_al=float(da[sel].sum()),
                               F_rigid=float(fr[sel].sum()))
            print(f"   그룹 {k:3d}: 강체 {fr[sel].sum():9.1f}  ΔMSC {dm[sel].sum():8.1f}  ΔAL {da[sel].sum():8.1f}")
        rec["per_group"] = per
        # 우익(그룹 10 = CAERO 1000001) 스팬 분포와 코드 방향(전연/후연) 분포
        wing = (g == 10)
        if wing.any():
            yw = np.abs(y[wing]); ymax = yw.max(); edges = np.linspace(0, ymax, 6)
            span = []
            print("   우익 스팬 5구간 (η): 강체 / ΔMSC / ΔAL")
            for k in range(5):
                sel = wing & (np.abs(y) >= edges[k]) & (np.abs(y) <= edges[k + 1])
                span.append(dict(eta=float(edges[k + 1] / ymax), F_rigid=float(fr[sel].sum()),
                                 dF_msc=float(dm[sel].sum()), dF_al=float(da[sel].sum())))
                print(f"     η≤{edges[k + 1] / ymax:.1f}: {fr[sel].sum():8.1f} / {dm[sel].sum():7.1f} / {da[sel].sum():7.1f}")
            rec["wing_span"] = span
            # 코드 방향: 같은 스트립 안에서 x 순서 (전연 박스 vs 후연 박스)
            xs = np.array([geom[i][0] for i in ids])
            strips = {}
            for i, sel_i in enumerate(wing):
                if sel_i:
                    strips.setdefault(round(y[i], 1), []).append(i)
            le = te = 0.0; le_m = te_m = 0.0
            for yy, idx in strips.items():
                idx = sorted(idx, key=lambda j: xs[j])
                h = max(1, len(idx) // 2)
                le += da[idx[:h]].sum(); te += da[idx[h:]].sum()
                le_m += dm[idx[:h]].sum(); te_m += dm[idx[h:]].sum()
            rec["wing_chord"] = dict(dF_msc_LE=le_m, dF_msc_TE=te_m, dF_al_LE=le, dF_al_TE=te)
            print(f"   우익 코드 방향 (전연 절반 / 후연 절반): ΔMSC {le_m:7.1f} / {te_m:7.1f}   ΔAL {le:7.1f} / {te:7.1f}")
        out[sc] = rec
    with open(os.path.join(HERE, "sc_elastic_increment.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
