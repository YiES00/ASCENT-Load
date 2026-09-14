# 설계하중 선정 2D potato vs 3D convex hull 비교 — ILC-8 55케이스 정량 분석
"""Compare 2-D potato-plane vs 3-D (V,M,T) convex-hull design-load
selection on the ILC-8 55-case set (39 trim + 16 landing).

For each method the compact design-case set is extracted; the
difference sets are then scored with a *directional exceedance*
metric: for a case selected only by the 3-D hull, how far outside
the convex hull of the 2-D-selected cases' (V,M,T) points does it
lie (scaled per-station so each quantity's range is 1.0, with the
range floored at SPAN_FLOOR times the component's largest station
range so that near-zero-load stations cannot inflate the metric)?
That distance is the worst-direction load underprediction a stress
engineer would accept by sizing with the 2-D set alone.

Usage:  python scripts/compare_hull_selection.py
Output: prints the comparison; writes hull_comparison.json here.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
ILC8 = os.path.join(SOLVER, "tests/validation/ILC8")
sys.path.insert(0, SOLVER)

from ascent_load.bdf.parser import parse_bdf                      # noqa: E402
from ascent_load.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig,
)
from ascent_load.loads_analysis.certification.batch_runner import (  # noqa: E402
    BatchResult, BatchRunner, CaseResult,
)
from ascent_load.loads_analysis.certification.envelope import (   # noqa: E402
    EnvelopeProcessor,
)
from ascent_load.loads_analysis.certification.load_case_matrix import (  # noqa: E402
    LoadCaseMatrix,
)
from ascent_load.loads_analysis.certification.vmt_bridge import (  # noqa: E402
    compute_vmt_for_batch,
)
from ascent_load.output.result_io import load_results             # noqa: E402


def build_batch():
    model = parse_bdf(os.path.join(ILC8, "ilc8.bdf"))
    with open(os.path.join(ILC8, "ilc8_cert_config.yaml")) as f:
        config = AircraftConfig.from_dict(yaml.safe_load(f))
    matrix = LoadCaseMatrix(config)
    matrix.generate_all(bdf_model=model, include_dynamic=False)
    meta = {c.case_id: c for c in matrix.flight_cases}

    results, _ = load_results(os.path.join(ILC8, "ilc8_cert_trim.aload"))
    batch = BatchResult()
    for sc in results.subcases:
        if not sc.nodal_combined_forces:
            continue
        fc = meta.get(sc.subcase_id)
        tc = getattr(fc, "trim_condition", None) if fc else None
        batch.case_results.append(CaseResult(
            case_id=sc.subcase_id,
            category=fc.category if fc else "trim",
            far_section=fc.far_section if fc else "",
            nz=(tc.nz if tc else 0.0), converged=True,
            nodal_forces=sc.nodal_combined_forces,
            label=(fc.label if fc else f"SC{sc.subcase_id}")))
        batch.completed_ids.add(sc.subcase_id)

    runner = BatchRunner(
        SimpleNamespace(config=config, flight_cases=[],
                        landing_cases=list(matrix.landing_cases),
                        dynamic_cases=[]),
        bdf_model=model)
    for cond in matrix.landing_cases:
        r = runner._solve_landing_case(cond)
        batch.case_results.append(r)
        batch.completed_ids.add(r.case_id)
    return model, batch


def run_selection(batch, vmt_data, mode: str):
    proc = EnvelopeProcessor(batch, vmt_data)
    proc.compute_envelopes()
    proc.identify_critical_cases()          # 축 극값은 공통
    if mode in ("2d", "both"):
        proc.add_interaction_critical_cases()
    if mode in ("3d", "both"):
        proc.add_interaction_critical_cases_3d()
    return proc, proc.select_design_cases()


SPAN_FLOOR = 0.05
# 스테이션별 정규화 스팬의 하한 = 그 축의 컴포넌트 최대 스테이션 범위 × 0.05.
# 하중이 0에 가까운 날개 끝 스테이션의 미소 범위(예: ILC-8 좌익 y=−6.0 m 에서
# V 39 N·M 20 N·mm)가 방향 초과율을 부풀리던 인공물을 막는다(2026-09-14).
# 선정(볼록 껍질 꼭짓점)은 축 스케일에 불변이므로 설계 세트는 영향이 없다.


def component_curves(vmt_data, cids, comp):
    """케이스 목록의 한 컴포넌트 (V,M,T) 곡선 → (n_cases, n_sta, 3)."""
    n_sta = len(vmt_data[cids[0]][comp]["stations"])
    P = np.empty((len(cids), n_sta, 3))
    for k, cid in enumerate(cids):
        d = vmt_data[cid][comp]
        P[k, :, 0] = d["shear"]
        P[k, :, 1] = d["bending"]
        P[k, :, 2] = d["torsion"]
    return P


def normalization_spans(P):
    """한 컴포넌트의 (n_cases, n_sta, 3) 곡선 → (n_sta, 3) 정규화 스팬.
    스테이션별 축 범위에 SPAN_FLOOR × (그 축의 컴포넌트 최대 스테이션
    범위)의 하한을 둔다. 세 초과율 구현(이 함수, hull3d_severity_search.
    exceedance_all, hull3d_combo_search 의 그리드 평가)이 공유한다."""
    span = np.ptp(P, axis=0)
    floor = SPAN_FLOOR * span.max(axis=0)
    span = np.maximum(span, floor[None, :])
    span[span == 0] = 1.0
    return span


def exceedance(vmt_data, base_ids, probe_id) -> float:
    """probe 케이스가 base 선정 세트의 (V,M,T) 헐 밖으로 나가는 최대
    거리 (스테이션별 각 축 범위=1로 정규화한 공간, 최악 방향;
    스팬 하한은 normalization_spans 참조)."""
    from scipy.spatial import ConvexHull

    worst = 0.0
    comps = set()
    for cd in vmt_data.values():
        comps.update(cd.keys())
    for comp in comps:
        cids = [c for c in vmt_data if comp in vmt_data[c]]
        if probe_id not in cids:
            continue
        base_rows = [k for k, c in enumerate(cids) if c in base_ids]
        if len(base_rows) < 4:
            continue
        P = component_curves(vmt_data, cids, comp)
        spans = normalization_spans(P)
        kp = cids.index(probe_id)
        for i in range(P.shape[1]):
            pts = P[:, i, :]
            lo = pts.min(axis=0)
            b = (pts[base_rows] - lo) / spans[i]
            p = (pts[kp] - lo) / spans[i]
            try:
                hull = ConvexHull(b)
            except Exception:
                continue
            viol = float(np.max(hull.equations[:, :3] @ p
                                + hull.equations[:, 3]))
            worst = max(worst, viol)
    return worst


def main() -> None:
    model, batch = build_batch()
    labels = {c.case_id: (c.label, c.category) for c in batch.case_results}
    print(f"batch: {len(batch.case_results)} cases", flush=True)
    vmt = compute_vmt_for_batch(model, batch, n_stations=50)

    proc2, dc2 = run_selection(batch, vmt, "2d")
    proc3, dc3 = run_selection(batch, vmt, "3d")
    procB, dcB = run_selection(batch, vmt, "both")
    n_crit = {"2d": len(proc2.get_critical_cases()),
              "3d": len(proc3.get_critical_cases()),
              "both": len(procB.get_critical_cases())}

    s2 = {d.case_id for d in dc2}
    s3 = {d.case_id for d in dc3}
    only3 = sorted(s3 - s2)
    only2 = sorted(s2 - s3)

    print(f"\n2D(축+3평면): critical {n_crit['2d']}건 → 설계하중 {len(s2)}개")
    print(f"3D(축+VMT헐): critical {n_crit['3d']}건 → 설계하중 {len(s3)}개")
    print(f"합집합      : critical {n_crit['both']}건 → "
          f"설계하중 {len({d.case_id for d in dcB})}개")
    print(f"\n2D에만 선정: {only2}  (이론상 공집합이어야 — 동률 제외)")
    print(f"3D에만 선정: {only3}")

    rows = []
    for cid in only3:
        exc = exceedance(vmt, s2, cid) * 100.0
        lab, cat = labels.get(cid, ("?", "?"))
        rows.append({"case_id": cid, "label": lab, "category": cat,
                     "exceedance_pct": round(exc, 2)})
    rows.sort(key=lambda r: -r["exceedance_pct"])
    print("\n3D 추가 케이스의 방향 초과율 (2D 선정 세트 헐 대비, 축범위%):")
    for r in rows:
        flag = ("의미있음" if r["exceedance_pct"] >= 5.0 else
                "미미" if r["exceedance_pct"] >= 1.0 else "과선정 성격")
        print(f"  C{r['case_id']:3d} {r['category']:12s} "
              f"{r['exceedance_pct']:6.2f}%  {flag}  {r['label'][:44]}")

    out = {"n_critical": n_crit,
           "design_2d": sorted(s2), "design_3d": sorted(s3),
           "only_3d": rows, "only_2d": only2}
    with open(os.path.join(HERE, "hull_comparison.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved: hull_comparison.json")


if __name__ == "__main__":
    main()
