# 같은 변위장에서 두 코드의 스플라인 하강류가 어떻게 달라지는지 본다 — MSC·ASCENT 홀드아웃 변위를 ASCENT IPS 로 3/4·중심·1/4 코드에서 미분 (스플라인 결합 추적 C1)
"""Normalwash from a given displacement field, evaluated at three chordwise
points of each wing box.

MSC locates the aerodynamic grid points (k-set) at the BOX CENTERS
(Aeroelastic Analysis User's Guide, "Aerodynamic Grid Points"), so with
SPLINE1 the steady normalwash is the surface-spline slope at the box
center. ASCENT-Load evaluates the slope at the 3/4-chord collocation
point. Both are exact for a linear (twist-only) surface; they differ
when the interpolated surface has chordwise curvature.

For each subcase this script takes (a) the MSC displacement field and
(b) the ASCENT-Load displacement field on the right-wing spline nodes,
builds the IPS slope operator at the 3/4-chord, center and 1/4-chord
points of the 96 right-wing boxes, and reports the area-weighted mean
wash (as an equivalent angle of attack) for every combination. It also
fits the best rigid-body motion to u_AL - u_MSC over all common nodes to
check whether the two codes share the same SUPORT reference.

Usage:  cd solver && python scripts/sc_wash_from_displacement.py <stem> [<stem> ...]
Output: sc_wash_from_displacement_<stem>.json next to this script.
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
from sc_msc_box_forces import ILC8  # noqa: E402

RIGHT_WING_CAERO = 1000001
RIGHT_WING_SET = 101


def set_ids(model, sid):
    s = model.sets[sid]
    for a in ("ids", "node_ids", "grids", "nodes", "values"):
        v = getattr(s, a, None)
        if v:
            return [int(i) for i in v]
    raise AttributeError(f"SET1 {sid}: {dir(s)}")


def rigid_fit(xyz, u3):
    """u ≈ t + theta x r 최소제곱 (t[3], theta[3] rad) 와 잔차."""
    n = len(xyz)
    A = np.zeros((3 * n, 6))
    r = xyz - xyz.mean(axis=0)
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = np.eye(3)
        rx, ry, rz = r[i]
        A[3 * i:3 * i + 3, 3:] = [[0, rz, -ry], [-rz, 0, rx], [ry, -rx, 0]]
    p, *_ = np.linalg.lstsq(A, u3.reshape(-1), rcond=None)
    return p, (u3.reshape(-1) - A @ p).reshape(n, 3)


def main(stems):
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.aero.panel import generate_all_panels
    from ascent_load.aero.spline import build_ips_spline_slope, build_ips_spline
    from ascent_load.output.result_io import load_results

    for stem in stems:
        model = parse_bdf(os.path.join(ILC8, f"{stem}.bdf"))
        model.cross_reference()
        boxes = [b for b in generate_all_panels(model, use_nastran_eid=True)
                 if RIGHT_WING_CAERO <= b.box_id < RIGHT_WING_CAERO + 100000]
        nids = set_ids(model, RIGHT_WING_SET)
        xyz = np.array([model.nodes[n].xyz_global for n in nids])
        cp = np.array([b.control_point for b in boxes])          # 3/4 코드
        dp = np.array([b.doublet_point for b in boxes])          # 1/4 코드
        ctr = np.array([b.corners.mean(axis=0) for b in boxes])  # 박스 중심 (MSC k-set)
        area = np.array([b.area for b in boxes])
        Gs = {"cp34": build_ips_spline_slope(xyz, cp), "ctr": build_ips_spline_slope(xyz, ctr),
              "dp14": build_ips_spline_slope(xyz, dp)}
        Gz_ctr = build_ips_spline(xyz, ctr)
        d_msc = parse_displacements(os.path.join(ILC8, f"{stem}_MSC.f06"))
        res, _ = load_results(os.path.join(ILC8, f"{stem}.aload"))
        d_al = {s.subcase_id: s.displacements for s in res.subcases}
        ystrip = np.round(dp[:, 1], 1)
        strips = sorted(set(ystrip))
        out = {}
        print(f"\n================ {stem}: 우익 스플라인 절점 {len(nids)}, 박스 {len(boxes)} ================")
        print("등가 받음각 = −(면적가중 평균 dz/dx) [deg]; 양수 = 양력 증가 방향(wash-in)")
        for sc in sorted(set(d_msc) & set(d_al)):
            rec = {}
            # 공통 절점 강체 적합: u_AL − u_MSC 의 강체 성분 (기준계 일치 검사)
            common = sorted(set(d_msc[sc]) & set(d_al[sc]) & set(model.nodes))
            X = np.array([model.nodes[n].xyz_global for n in common])
            dU = np.array([d_al[sc][n][:3] - d_msc[sc][n][:3] for n in common])
            p, resid = rigid_fit(X, dU)
            rec["rigid_fit_dU"] = dict(t_mm=p[:3].tolist(), theta_deg=np.degrees(p[3:]).tolist(),
                                       resid_rms_mm=float(np.sqrt((resid ** 2).mean())),
                                       dU_rms_mm=float(np.sqrt((dU ** 2).mean())))
            # SUPORT 절점 변위
            sup = {n: (d_msc[sc].get(n, np.full(6, np.nan))[:3].tolist(),
                       d_al[sc].get(n, np.full(6, np.nan))[:3].tolist()) for n in (101519, 101719, 101501)}
            rec["suport_disp_msc_al"] = {str(k): v for k, v in sup.items()}
            print(f"\nSC{sc}: 공통 절점 {len(common)}  (u_AL−u_MSC) 강체 적합 θy {np.degrees(p[4]):+.4f}° "
                  f"θx {np.degrees(p[3]):+.4f}°  잔차 RMS {rec['rigid_fit_dU']['resid_rms_mm']:.3f} mm "
                  f"(ΔU RMS {rec['rigid_fit_dU']['dU_rms_mm']:.3f} mm)")
            print(f"   SUPORT 절점 T3: 101519 MSC {sup[101519][0][2]:+.4f} / AL {sup[101519][1][2]:+.4f}; "
                  f"101719 {sup[101719][0][2]:+.4f} / {sup[101719][1][2]:+.4f}; 101501 {sup[101501][0][2]:+.4f} / {sup[101501][1][2]:+.4f}")
            for tag, field in (("MSC", d_msc[sc]), ("AL", d_al[sc])):
                uz = np.array([field[n][2] for n in nids])
                uz_al = np.array([d_al[sc][n][2] for n in nids])
                row = {}
                for k, G in Gs.items():
                    w = G @ uz
                    row[k] = dict(alpha_eq_deg=float(-np.degrees((w * area).sum() / area.sum())),
                                  wash_rms_deg=float(np.degrees(np.sqrt((w ** 2).mean()))))
                # 스트립별 (코드 방향 평균) 세 평가점 기울기 + 기하학적 스트립 비틀림
                strip_rows = []
                for y in strips:
                    sel = ystrip == y
                    strip_rows.append(dict(y=float(y), **{k: float(-np.degrees(((Gs[k] @ uz)[sel] * area[sel]).sum() / area[sel].sum())) for k in Gs}))
                row["strips"] = strip_rows
                row["uz_tip_mm"] = float(uz[np.argmax(xyz[:, 1])])
                row["uz_L2_vs_AL_pct"] = float(np.linalg.norm(uz - uz_al) / max(np.linalg.norm(uz_al), 1e-12) * 100)
                rec[tag] = row
                print(f"   u_{tag:3s}: 팁 uz {row['uz_tip_mm']:8.2f} mm  |  α_eq 3/4코드 {row['cp34']['alpha_eq_deg']:+.4f}°  "
                      f"중심 {row['ctr']['alpha_eq_deg']:+.4f}°  1/4코드 {row['dp14']['alpha_eq_deg']:+.4f}°  "
                      f"(uz L2 vs AL {row['uz_L2_vs_AL_pct']:.2f}%)")
            # 스트립 표 (루트·중간·팁 3개만 출력)
            for i in (0, len(strips) // 2, len(strips) - 1):
                m, a = rec["MSC"]["strips"][i], rec["AL"]["strips"][i]
                print(f"     y={m['y']:7.1f}: MSC 3/4 {m['cp34']:+.4f} 중심 {m['ctr']:+.4f} 1/4 {m['dp14']:+.4f}  |  "
                      f"AL 3/4 {a['cp34']:+.4f} 중심 {a['ctr']:+.4f} 1/4 {a['dp14']:+.4f}")
            out[sc] = rec
        with open(os.path.join(HERE, f"sc_wash_from_displacement_{stem}.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["ilc8_msc_sol144_v9_holdout"])
