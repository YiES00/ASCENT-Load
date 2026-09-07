# v8 정합 덱의 MSC 대 ASCENT-Load 트림 변수 비교를 JSON 으로 아카이브 (논문 1 r6 근거)
"""Archive the v8 matched-deck trim-variable comparison as JSON.

``compare_msc_sol144.py`` prints its comparison to stdout only. The r6
manuscript revision cites the per-subcase angle-of-attack and
control-surface deviations of the regenerated (QM6+DKQ) solver against
the archived MSC F06, so those numbers need a stored artifact.

Usage:  python scripts/r6_msc_v8_trim_archive.py
Output: scripts/r6_msc_v8_trim_metrics.json
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from compare_msc_sol144 import parse_msc_f06, solve_ascent_load

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ILC8 = REPO / "tests" / "validation" / "ILC8"
DECK = ILC8 / "ilc8_msc_sol144_v8_shellbend.bdf"
F06 = ILC8 / "ilc8_msc_sol144_v8_shellbend_MSC.f06"
ALOAD = ILC8 / "ilc8_msc_sol144_v8_shellbend.aload"
# 서브케이스별 하중배수 (덱 TRIM 카드 URDD3 / g)
NZ = {1: 1.0, 2: 1.0, 3: 1.0, 4: 3.8, 5: -1.52, 6: 3.8, 7: 1.0}


def _v8_aload() -> Path:
    """v8 덱의 .aload 가 추적되어 있지 않으므로 없으면 현행 솔버로 푼다."""
    if ALOAD.exists():
        return ALOAD
    tmp = Path(tempfile.mkdtemp(prefix="r6_v8_")) / "v8.aload"
    subprocess.run([sys.executable, "-m", "ascent_load", str(DECK),
                    "--save-results", str(tmp)],
                   cwd=REPO, check=True, capture_output=True)
    return tmp


def main() -> None:
    msc = parse_msc_f06(F06)
    al = solve_ascent_load(_v8_aload())
    r2d = math.degrees(1.0)
    rows = []
    for sc in sorted(msc):
        m, n = msc[sc], al.get(sc, {})
        a_m = m["ANGLEA"][0] * r2d
        a_n = n.get("ANGLEA", 0.0) * r2d
        elev_n, rud_n = n.get("ELEV", 0.0), n.get("RUD", 0.0)
        e_m = m["ELEVR"][0]
        e_n = elev_n + rud_n
        row = dict(sc=sc, nz=NZ[sc],
                   alpha_msc_deg=a_m, alpha_al_deg=a_n,
                   dalpha_deg=a_n - a_m,
                   dalpha_pct=100.0 * abs(a_n - a_m) / max(abs(a_m), 1e-12),
                   dalpha_per_g=(a_n - a_m) / NZ[sc],
                   elevr_msc_deg=e_m * r2d, elevr_al_deg=e_n * r2d,
                   delevr_deg=(e_n - e_m) * r2d,
                   delevr_pct=100.0 * abs(e_n - e_m) / max(abs(e_m), 1e-12))
        if abs(rud_n) > 1e-9 or abs(m.get("RUD", (0.0, ""))[0]) > 1e-9:
            rud_m = 0.5 * abs(m["ELEVR"][0] - m["ELEVL"][0])
            row.update(rudder_msc_deg=rud_m * r2d,
                       rudder_al_deg=abs(rud_n) * r2d,
                       drudder_deg=(abs(rud_n) - rud_m) * r2d,
                       drudder_pct=100.0 * abs(abs(rud_n) - rud_m)
                       / max(rud_m, 1e-12))
        rows.append(row)
    per_g = [r["dalpha_per_g"] for r in rows]
    bias = sum(per_g) / len(per_g)
    resid = [100.0 * abs(r["dalpha_deg"] - bias * r["nz"])
             / max(abs(r["alpha_msc_deg"]), 1e-12) for r in rows]
    # 방향타 자유 서브케이스(SC7)는 AELINK 부호 규약 차이로 ELEVR 마스터
    # 비교가 무의미하다 — 논문과 같이 |방향타| 만 비교하고 승강타 범위에서 뺀다
    rows_e = [r for r in rows if "rudder_msc_deg" not in r]
    out = dict(deck="ilc8_msc_sol144_v8_shellbend", f06=F06.name,
               aload=ALOAD.name, rows=rows,
               summary=dict(
                   dalpha_deg_range=[min(abs(r["dalpha_deg"]) for r in rows),
                                     max(abs(r["dalpha_deg"]) for r in rows)],
                   dalpha_pct_range=[min(r["dalpha_pct"] for r in rows),
                                     max(r["dalpha_pct"] for r in rows)],
                   delevr_deg_range=[min(abs(r["delevr_deg"]) for r in rows_e),
                                     max(abs(r["delevr_deg"]) for r in rows_e)],
                   delevr_pct_range=[min(r["delevr_pct"] for r in rows_e),
                                     max(r["delevr_pct"] for r in rows_e)],
                   elevator_rows_exclude_rudder_free=[
                       r["sc"] for r in rows if "rudder_msc_deg" in r],
                   alpha_bias_deg_per_g=bias,
                   alpha_bias_scatter_pct=100.0 * (max(per_g) - min(per_g))
                   / abs(bias),
                   alpha_resid_pct_after_bias_range=[min(resid), max(resid)]))
    path = HERE / "r6_msc_v8_trim_metrics.json"
    path.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"SC{r['sc']} nz={r['nz']:+.2f} alpha {r['alpha_msc_deg']:.3f}/"
              f"{r['alpha_al_deg']:.3f} ({r['dalpha_deg']:+.3f} deg, "
              f"{r['dalpha_pct']:.1f}%, {r['dalpha_per_g']:+.3f} deg/g)  "
              f"ELEVR {r['elevr_msc_deg']:.3f}/{r['elevr_al_deg']:.3f} "
              f"({r['delevr_deg']:+.3f} deg, {r['delevr_pct']:.1f}%)"
              + (f"  |rud| {r['rudder_msc_deg']:.4f}/{r['rudder_al_deg']:.4f}"
                 if "rudder_msc_deg" in r else ""))
    print(json.dumps(out["summary"], indent=1))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
