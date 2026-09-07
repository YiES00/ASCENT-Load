# MSC hold-out F06 를 ASCENT-Load 와 전면 대조한다 (r5 잔여 1번 / MC5)
"""Hold-out comparison against MSC Nastran, ILC-8 v9 deck.

Four independent measurements, all on the five hold-out subcases the
development never used:

  1. Trim variables (alpha, linked elevator) — and whether the
     discrepancy is one constant per g (it is: the bias per unit load
     factor is the same in every subcase).
  2. Elastic displacement field against the pre-declared criteria of
     r3_msc_displacement_metrics.py, plus the single-scale test: fit one
     scalar s to the whole elastic field and report how much of the L2
     error it explains.
  3. Applied nodal loads: MSC GPFORCE 'APP-LOAD' rows summed per
     structural component against ASCENT-Load's nodal combined forces.
  4. Section loads (shear / bending / torsion along span) computed with
     the SAME compute_vmt_all on both codes' nodal loads, so any
     difference is in the loads, not in the integration.

Usage:  cd solver && python scripts/r5_msc_holdout_metrics.py
Data:   tests/validation/ILC8/ilc8_msc_sol144_v9_holdout{_MSC,}.f06,
        ilc8_msc_sol144_v9_holdout.{bdf,aload}
Output: r5_msc_holdout_metrics.json next to this script.
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
sys.path.insert(0, HERE)

from r3_msc_displacement_metrics import (  # noqa: E402
    parse_displacements, remove_rigid, CRITERIA,
)

ILC8 = os.path.join(SOLVER, "tests", "validation", "ILC8")
STEM = "ilc8_msc_sol144_v9_holdout"
F_MSC = os.path.join(ILC8, f"{STEM}_MSC.f06")
F_AL = os.path.join(ILC8, f"{STEM}.f06")
DECK = os.path.join(ILC8, f"{STEM}.bdf")
ALOAD = os.path.join(ILC8, f"{STEM}.aload")

REGIONS = [("Fuselage", 100000, 299999), ("Left Wing", 300000, 399999),
           ("Right Wing", 400000, 499999), ("Left V-Tail", 500000, 599999),
           ("Right V-Tail", 600000, 699999), ("VTP", 700000, 799999),
           ("Booms/Rotors/Gear", 900000, 999999)]

_APP = re.compile(r"^\s*(\d+)\s+APP-LOAD\s+([-+.\dE]+)\s+([-+.\dE]+)\s+"
                  r"([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)")
_SC = re.compile(r"SUBCASE\s+(\d+)\s*$")
_TRIM_ROW = re.compile(
    r"^\s*(\d+)?\s+([A-Z][A-Z0-9]*)\s+"
    r"(RIGID BODY|CONTROL SURFACE|GENERAL CONTROL)\s+"
    r"(FIXED|FREE|LINKED)\s+([-+]?\d\.\d+E[-+]\d+)")


def parse_gpforce_applied(path):
    """{subcase: {nid: ndarray(6)}} from the GPFORCE 'APP-LOAD' rows.

    MSC prints the applied load at each grid in the grid's displacement
    frame; the ILC-8 deck has CD blank on every GRID, so this is basic.
    """
    out = {}
    sc = None
    in_blk = False
    with open(path, errors="ignore") as f:
        for line in f:
            m = _SC.search(line)
            if m:
                sc = int(m.group(1))
            if "G R I D   P O I N T   F O R C E" in line:
                in_blk = True
                continue
            if not in_blk:
                continue
            # 새 페이지의 첫 행은 캐리지 제어 '0' 으로 시작한다
            s = line[1:] if line[:1] == "0" else line
            r = _APP.match(s)
            if r:
                out.setdefault(sc, {})[int(r.group(1))] = np.array(
                    [float(r.group(i)) for i in range(2, 8)])
    return out


def parse_trim_msc(path):
    """{subcase: {label: value}} from AEROELASTIC TRIM VARIABLES."""
    out = {}
    sc = None
    in_blk = False
    with open(path, errors="ignore") as f:
        for line in f:
            m = _SC.search(line)
            if m:
                sc = int(m.group(1))
            if "AEROELASTIC TRIM VARIABLES" in line:
                in_blk = True
                continue
            if in_blk:
                r = _TRIM_ROW.match(line)
                if r:
                    out.setdefault(sc, {})[r.group(2)] = float(r.group(5))
                elif line.startswith("1") and sc in out:
                    in_blk = False
    return out


def scale_fit(a, b):
    """s = argmin |s a - b|, and the L2 error before/after."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    s = float(a @ b / (a @ a))
    l2_0 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    l2_s = float(np.linalg.norm(s * a - b) / np.linalg.norm(b))
    return s, l2_0, l2_s


def main():
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.output.result_io import load_results
    from ascent_load.loads_analysis.component_id import (
        identify_components_manual,
    )
    from ascent_load.loads_analysis.vmt import compute_vmt_all

    model = parse_bdf(DECK)
    res, _ = load_results(ALOAD)
    sub = {sr.subcase_id: sr for sr in res.subcases}
    d_msc = parse_displacements(F_MSC)
    d_al = parse_displacements(F_AL)
    app = parse_gpforce_applied(F_MSC)
    trim_msc = parse_trim_msc(F_MSC)
    out = {"criteria_declared": CRITERIA}

    # ---------------- 1. 트림 변수와 g 당 편향 ----------------
    print("=== 1. 트림 변수 (hold-out 5 케이스) ===")
    rows = []
    for sc in sorted(sub):
        tv = sub[sc].trim_variables
        nz = float(tv["URDD3"])
        a_al = np.rad2deg(float(tv["ANGLEA"]))
        a_ms = np.rad2deg(trim_msc[sc]["ANGLEA"])
        # AELINK 부호 규약이 달라 마스터 ELEV 대신 물리 조종면 ELEVR 로 비교
        e_al = float(tv["ELEV"]) + float(tv.get("RUD", 0.0))
        e_ms = trim_msc[sc]["ELEVR"]
        rows.append(dict(sc=sc, nz=nz, alpha_msc=a_ms, alpha_al=a_al,
                         elevr_msc=e_ms, elevr_al=e_al,
                         dalpha_per_g=(a_al - a_ms) / nz,
                         delev_per_g=(e_al - e_ms) / nz))
        print(f"  SC{sc} nz={nz:+.1f}  alpha {a_ms:9.4f}/{a_al:9.4f} "
              f"({abs(a_al - a_ms) / abs(a_ms) * 100:4.1f}%)  ELEVR "
              f"{e_ms:9.6f}/{e_al:9.6f} "
              f"({abs(e_al - e_ms) / abs(e_ms) * 100:4.1f}%)")
    da = np.array([r["dalpha_per_g"] for r in rows])
    de = np.array([r["delev_per_g"] for r in rows])
    bias = dict(alpha_deg_per_g=float(da.mean()),
                alpha_scatter_pct=float(da.std(ddof=1) / abs(da.mean()) * 100),
                elev_rad_per_g=float(de.mean()),
                elev_scatter_pct=float(de.std(ddof=1) / abs(de.mean()) * 100))
    for r in rows:
        r["alpha_resid_pct_after_bias"] = float(
            abs(r["alpha_al"] - bias["alpha_deg_per_g"] * r["nz"]
                - r["alpha_msc"]) / abs(r["alpha_msc"]) * 100)
    print(f"  alpha 편향 {bias['alpha_deg_per_g']:.5f} deg/g "
          f"(산포 {bias['alpha_scatter_pct']:.2f}%), 편향 제거 후 잔차 "
          f"{max(r['alpha_resid_pct_after_bias'] for r in rows):.3f}% 이하")
    out["trim"] = dict(rows=rows, bias=bias)

    # ---------------- 2. 탄성 변위장: 사전 기준 + 단일 배율 ----------------
    print("\n=== 2. 탄성 변위장 (사전 기준 A1/A2/A3 + 단일 배율 검정) ===")
    disp = {}
    wing = set(n for n in model.nodes if 300000 <= n <= 499999)
    for sc in sorted(set(d_msc) & set(d_al)):
        common = sorted(set(d_msc[sc]) & set(d_al[sc]) & set(model.nodes))
        xyz = np.array([model.nodes[n].xyz_global for n in common])
        em = remove_rigid(xyz, np.array([d_msc[sc][n] for n in common]))
        ea = remove_rigid(xyz, np.array([d_al[sc][n] for n in common]))
        t3m, t3a = em[:, 2], ea[:, 2]
        l2 = np.linalg.norm(t3a - t3m) / np.linalg.norm(t3m) * 100
        mx = np.max(np.abs(t3a - t3m)) / np.max(np.abs(t3m)) * 100
        wi = np.array([i for i, n in enumerate(common) if n in wing])
        ti = wi[np.argmax(np.abs(t3m[wi]))]
        tip = abs(t3a[ti] - t3m[ti]) / abs(t3m[ti]) * 100
        s3, _, l2_s3 = scale_fit(ea, em)          # 3성분 전체
        per_reg = {}
        for nm, lo, hi in REGIONS:
            ii = np.array([i for i, n in enumerate(common) if lo <= n <= hi])
            if len(ii) < 10:
                continue
            s_r, l0_r, ls_r = scale_fit(ea[ii, 2], em[ii, 2])
            per_reg[nm] = dict(n=int(len(ii)), s=s_r, l2_after_pct=ls_r * 100)
        disp[sc] = dict(
            n_common=len(common), L2_T3_pct=float(l2), max_T3_pct=float(mx),
            tip_msc_mm=float(t3m[ti]), tip_al_mm=float(t3a[ti]),
            tip_pct=float(tip),
            pass_A1=bool(l2 <= CRITERIA["A1_L2_T3_pct"]),
            pass_A2=bool(mx <= CRITERIA["A2_max_T3_pct"]),
            pass_A3=bool(tip <= CRITERIA["A3_tip_pct"]),
            scale_s=s3, L2_after_scale_pct=l2_s3 * 100,
            explained_pct=(1 - (l2_s3 / (l2 / 100)) ** 2) * 100,
            per_region=per_reg)
        d = disp[sc]
        print(f"  SC{sc}: L2 {l2:5.2f}% [{'PASS' if d['pass_A1'] else 'FAIL'}]"
              f"  max {mx:5.2f}% [{'PASS' if d['pass_A2'] else 'FAIL'}]"
              f"  tip {tip:5.2f}% [{'PASS' if d['pass_A3'] else 'FAIL'}]"
              f"   s={s3:.5f} -> L2 {l2_s3 * 100:.2f}% "
              f"(설명력 {d['explained_pct']:.2f}%)")
    S = np.array([disp[sc]["scale_s"] for sc in disp])
    out["displacement"] = dict(
        per_subcase=disp,
        scale_mean=float(S.mean()),
        scale_scatter_pct=float(S.std(ddof=1) / S.mean() * 100))
    print(f"  배율 s = {S.mean():.5f} (산포 {out['displacement']['scale_scatter_pct']:.3f}%), "
          f"1/s = {1 / S.mean():.5f}")
    print("  구성품별 s (SC3):", {k: round(v["s"], 4)
                                for k, v in disp[3]["per_region"].items()})

    # ---------------- 3. 적용 절점하중 (GPFORCE APP-LOAD) ----------------
    print("\n=== 3. 적용 절점하중: MSC APP-LOAD 대 ASCENT-Load 결합하중 ===")
    loads = {}
    for sc in sorted(sub):
        al = sub[sc].nodal_combined_forces
        ms = app[sc]
        common = sorted(set(al) & set(ms))
        per = {}
        worst = 0.0
        for nm, lo, hi in REGIONS:
            ids = [n for n in common if lo <= n <= hi]
            if not ids:
                continue
            fm = np.array([ms[n][:3] for n in ids])
            fa = np.array([al[n][:3] for n in ids])
            fz_m, fz_a = float(fm[:, 2].sum()), float(fa[:, 2].sum())
            ab_m, ab_a = float(np.abs(fm[:, 2]).sum()), float(np.abs(fa[:, 2]).sum())
            per[nm] = dict(n=len(ids), Fz_msc=fz_m, Fz_al=fz_a,
                           absFz_msc=ab_m, absFz_al=ab_a,
                           absFz_ratio=ab_a / ab_m)
            worst = max(worst, abs(ab_a / ab_m - 1) * 100)
        loads[sc] = dict(n_common=len(common), per_region=per,
                         worst_abs_ratio_dev_pct=worst)
        print(f"  SC{sc}: 공통 {len(common)} 절점, 구성품별 |Fz| 합 비 최대 편차 "
              f"{worst:.2f}%  (날개 Fz {per['Right Wing']['Fz_msc']:.1f}/"
              f"{per['Right Wing']['Fz_al']:.1f} N)")
    out["applied_loads"] = loads

    # ---------------- 4. 단면하중: 같은 적분기로 두 하중을 적분 ----------------
    print("\n=== 4. 단면하중 (compute_vmt_all, 동일 구성품·동일 적분) ===")

    def _nids(lo, hi, extra=()):
        ids = [n for n in model.nodes if lo <= n <= hi]
        return ids + [n for n in extra if n in model.nodes]

    WING = dict(span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1)
    hubs_r = (990103, 990104, 990107, 990108)
    hubs_l = (990101, 990102, 990105, 990106)
    comps = identify_components_manual(model, [
        dict(name="Right Wing", integration_sign=1.0, color="blue",
             node_ids=_nids(400000, 499999) + _nids(730000, 749999, hubs_r)
             + _nids(991400, 991799), **WING),
        dict(name="Left Wing", integration_sign=-1.0, color="dodgerblue",
             node_ids=_nids(300000, 399999) + _nids(710000, 729999, hubs_l)
             + _nids(991000, 991399), **WING),
        dict(name="Right V-Tail", integration_sign=1.0, color="red",
             node_ids=_nids(600000, 699999), **WING),
        dict(name="Left V-Tail", integration_sign=-1.0, color="salmon",
             node_ids=_nids(500000, 599999), **WING),
        dict(name="Fuselage", integration_sign=-1.0, color="gray",
             node_ids=_nids(100000, 299999, (990201,))
             + _nids(991900, 991999) + _nids(992000, 992999),
             span_axis=0, shear_axis=2, bending_axis=1, torsion_axis=0),
    ])
    vmt = {}
    for sc in sorted(sub):
        r_al = compute_vmt_all(model, sub[sc].nodal_combined_forces, comps,
                               n_stations=50, subcase_id=sc,
                               fuselage_cg_x=4450.0)
        r_ms = compute_vmt_all(model, app[sc], comps, n_stations=50,
                               subcase_id=sc, fuselage_cg_x=4450.0)
        ca = {c.component_name: c for c in r_al.curves}
        cm = {c.component_name: c for c in r_ms.curves}
        vmt[sc] = {}
        for name in ca:
            a, m = ca[name], cm[name]
            ent = {}
            # 편차의 기준: 전단은 자기 피크, 모멘트(M·T)는 그 구성품의
            # 더 큰 모멘트 피크. 대칭 케이스의 동체 비틀림처럼 물리적으로
            # 0 인 채널을 자기 피크(~1e-9)로 나누면 무의미한 폭발이 난다.
            mom_ref = max(float(np.max(np.abs(np.asarray(m.bending_moment)))),
                          float(np.max(np.abs(np.asarray(m.torsion)))), 1.0)
            for q, ka, km in (("V", a.shear, m.shear),
                              ("M", a.bending_moment, m.bending_moment),
                              ("T", a.torsion, m.torsion)):
                ka, km = np.asarray(ka), np.asarray(km)
                ref = max(float(np.max(np.abs(km))), 1.0) if q == "V" else mom_ref
                ent[q] = dict(root_msc=float(km[0]), root_al=float(ka[0]),
                              ref=ref,
                              max_abs_dev_pct_of_peak=float(
                                  np.max(np.abs(ka - km)) / ref * 100),
                              rms_dev_pct_of_peak=float(
                                  np.sqrt(np.mean((ka - km) ** 2)) / ref * 100))
            vmt[sc][name] = ent
        rw = vmt[sc]["Right Wing"]
        print(f"  SC{sc} Right Wing 뿌리: V {rw['V']['root_msc']:.1f}/"
              f"{rw['V']['root_al']:.1f} N, M {rw['M']['root_msc']:.3e}/"
              f"{rw['M']['root_al']:.3e}, T {rw['T']['root_msc']:.3e}/"
              f"{rw['T']['root_al']:.3e} N·mm; 최대 편차(피크 대비) "
              f"V {rw['V']['max_abs_dev_pct_of_peak']:.2f}% "
              f"M {rw['M']['max_abs_dev_pct_of_peak']:.2f}% "
              f"T {rw['T']['max_abs_dev_pct_of_peak']:.2f}%")
    worst = max(e[q]["max_abs_dev_pct_of_peak"]
                for sc in vmt for e in vmt[sc].values() for q in ("V", "M", "T"))
    print(f"  전 구성품·전 서브케이스 최대 편차(피크 대비) {worst:.2f}%")
    out["section_loads"] = dict(per_subcase=vmt, worst_dev_pct_of_peak=worst)

    path = os.path.join(HERE, "r5_msc_holdout_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
