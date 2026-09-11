# 트림 α 편차가 SUPORT 기준계(기어 하중을 받는 킬 스킨 절점)의 국소 변형 차이인지 판정한다 — 강체 적합·기준계 무관 국소량·유효 날개 받음각 (스플라인 결합 추적 C4/D1)
"""Is the trim-alpha bias a SUPORT-reference artifact?

Both codes print displacements relative to the deck SUPORT DOFs
(101519/123, 101719/23, 101501/3 on the ILC-8 keel/side skin, 500-700 mm
baseline, gear legs attached). ANGLEA is therefore the incidence of that
local skin triad. This script

 1. fits a rigid-body motion to u_AL - u_MSC over the wing/HTP nodes
    (far field) -> theta_y is the pitch offset between the two codes'
    reference triads;
 2. reports reference-independent local quantities at the gear joints
    (ring ovalisation at x=4250, gear-leg relative dz, keel dimples,
    joint rotations R1);
 3. forms the effective wing incidence alpha + theta_y(root section)
    in each code's own frame, which does not depend on the reference.

Extra ASCENT runs (element-option variants) can be passed as name=path.

Usage:  cd solver && python scripts/sc_suport_reference.py <stem> [name=aload ...]
Output: sc_suport_reference_<stem>.json next to this script.
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
from r3_msc_displacement_metrics import parse_displacements  # noqa: E402
from r5_msc_holdout_metrics import parse_trim_msc  # noqa: E402
from sc_wash_from_displacement import rigid_fit, set_ids  # noqa: E402
from sc_msc_box_forces import ILC8  # noqa: E402

SUP = (101519, 101719, 101501)
TOP, BOT, SIDE = 101507, 101519, 101501
GEAR = (992011, 992012)
RIGHT_WING_SET = 101


def deg(v):
    return float(np.degrees(v)) if abs(v) < 1.0 else float(v)


def section_pitch(model, d, nids, y):
    """y 스테이션의 코드 방향 절점들로 dz/dx 최소제곱 → 피치각 −dz/dx [deg]."""
    pts = [n for n in nids if abs(model.nodes[n].xyz_global[1] - y) < 1.0]
    x = np.array([model.nodes[n].xyz_global[0] for n in pts])
    z = np.array([d[n][2] for n in pts])
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(np.degrees(-c[1]))


def main(stem, variants):
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.output.result_io import load_results

    model = parse_bdf(os.path.join(ILC8, f"{stem}.bdf"))
    xyz = {n: model.nodes[n].xyz_global for n in model.nodes}
    d_msc = parse_displacements(os.path.join(ILC8, f"{stem}_MSC.f06"))
    t_msc = parse_trim_msc(os.path.join(ILC8, f"{stem}_MSC.f06"))
    wing_nids = set_ids(model, RIGHT_WING_SET)
    ys = sorted(set(round(xyz[n][1], 1) for n in wing_nids))
    y_root, y_mid = ys[0], ys[len(ys) // 2]
    keel19 = [n for n in xyz if 100000 <= n <= 299999 and abs(xyz[n][1]) < 1 and abs(xyz[n][2] + 700) < 1
              and 0 < abs(xyz[n][0] - 4250) < 300]
    keel17 = [n for n in xyz if 100000 <= n <= 299999 and abs(xyz[n][1]) < 1 and abs(xyz[n][2] + 700) < 1
              and 0 < abs(xyz[n][0] - 4750) < 300]
    runs = [("AL", os.path.join(ILC8, f"{stem}.aload"))] + variants
    out = {"suport": SUP, "variants": {}}
    print(f"\n================ {stem} ================")
    print("SUPORT 절점:", {n: np.round(xyz[n], 0).tolist() for n in SUP})
    for name, path in runs:
        res, _ = load_results(path)
        d_al = {s.subcase_id: s.displacements for s in res.subcases}
        t_al = {s.subcase_id: s.trim_variables for s in res.subcases}
        rows = {}
        print(f"\n--- {name} ---")
        print(" SC |  nz  |  α_AL   α_MSC   Δα   | θy_far  Δα+θy | α_eff(루트) AL/MSC  Δ | α_eff(중앙) Δ | 타원화 MSC/AL | 기어Δz MSC/AL | 딤플19 MSC/AL | R1(101519) MSC/AL mrad")
        for sc in sorted(set(d_msc) & set(d_al)):
            dm, da = d_msc[sc], d_al[sc]
            common = sorted(set(dm) & set(da) & set(xyz))
            far = [n for n in common if 300000 <= n <= 699999]
            X = np.array([xyz[n] for n in far])
            dU = np.array([da[n][:3] - dm[n][:3] for n in far])
            p, resid = rigid_fit(X, dU)
            thy, thx = np.degrees(p[4]), np.degrees(p[3])
            a_al, a_m = deg(float(t_al[sc]["ANGLEA"])), deg(float(t_msc[sc]["ANGLEA"]))
            nz = float(t_al[sc].get("URDD3", np.nan))
            eff_al_r = a_al + section_pitch(model, da, wing_nids, y_root)
            eff_m_r = a_m + section_pitch(model, dm, wing_nids, y_root)
            eff_al_m = a_al + section_pitch(model, da, wing_nids, y_mid)
            eff_m_m = a_m + section_pitch(model, dm, wing_nids, y_mid)
            ov = (dm[TOP][2] - dm[BOT][2], da[TOP][2] - da[BOT][2])
            gear = (dm[GEAR[1]][2] - dm[BOT][2], da[GEAR[1]][2] - da[BOT][2])
            dimple19 = (dm[BOT][2] - np.mean([dm[n][2] for n in keel19]), da[BOT][2] - np.mean([da[n][2] for n in keel19]))
            dimple17 = (dm[101719][2] - np.mean([dm[n][2] for n in keel17]), da[101719][2] - np.mean([da[n][2] for n in keel17]))
            r1 = (dm[BOT][3] * 1e3, da[BOT][3] * 1e3)
            rows[sc] = dict(nz=nz, alpha_al=a_al, alpha_msc=a_m, d_alpha=a_al - a_m,
                            theta_y_far=float(thy), theta_x_far=float(thx),
                            residual_alpha=a_al - a_m + float(thy),
                            far_fit_resid_rms_mm=float(np.sqrt((resid ** 2).mean())),
                            alpha_eff_root=dict(al=eff_al_r, msc=eff_m_r, d=eff_al_r - eff_m_r),
                            alpha_eff_mid=dict(al=eff_al_m, msc=eff_m_m, d=eff_al_m - eff_m_m),
                            ovalisation_mm=dict(msc=ov[0], al=ov[1]),
                            gear_dz_mm=dict(msc=gear[0], al=gear[1]),
                            dimple_101519_mm=dict(msc=dimple19[0], al=dimple19[1]),
                            dimple_101719_mm=dict(msc=dimple17[0], al=dimple17[1]),
                            r1_101519_mrad=dict(msc=r1[0], al=r1[1]))
            print(f" {sc:2d} | {nz:4.1f} | {a_al:7.3f} {a_m:7.3f} {a_al - a_m:+6.3f} | {thy:+7.4f} {a_al - a_m + thy:+6.3f} | "
                  f"{eff_al_r:7.3f}/{eff_m_r:7.3f} {eff_al_r - eff_m_r:+6.3f} | {eff_al_m - eff_m_m:+6.3f} | "
                  f"{ov[0]:+6.2f}/{ov[1]:+6.2f} | {gear[0]:+5.2f}/{gear[1]:+5.2f} | {dimple19[0]:+5.2f}/{dimple19[1]:+5.2f} | {r1[0]:+5.2f}/{r1[1]:+5.2f}")
        out["variants"][name] = rows
    with open(os.path.join(HERE, f"sc_suport_reference_{stem}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    args = sys.argv[1:] or ["ilc8_msc_sol144_v9_holdout"]
    main(args[0], [tuple(a.split("=", 1)) for a in args[1:]])
