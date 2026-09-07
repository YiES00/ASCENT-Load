# MSC 변위에 ASCENT-Load 요소강성을 곱해 MSC GPFORCE 요소 행과 요소별로 대조한다
"""Element-level stiffness cross-check against MSC GPFORCE.

For every CBAR and CQUAD4 in the ILC-8 hold-out deck, build the element
stiffness exactly as ASCENT-Load's assembler does, multiply it by MSC's
own displacement vector for that element's nodes, and compare the
result with the element rows MSC prints in GPFORCE (which are
k_e^MSC u_e, sign-flipped: MSC prints the force the element exerts on
the grid so that APP-LOAD + sum(element rows) = 0).

If k_e^AL == k_e^MSC the two agree to round-off for every element and
every component. If ASCENT-Load's global stiffness is 25 % higher (as
the SOL 101 isolation showed), this pins the excess to an element type
and a DOF component.

Usage:  cd solver && python scripts/r5_msc_element_check.py [subcase=1]
Output: r5_msc_element_check.json next to this script.
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
from r3_msc_displacement_metrics import parse_displacements  # noqa: E402

ILC8 = os.path.join(SOLVER, "tests", "validation", "ILC8")
STEM = "ilc8_msc_sol144_v9_holdout"
SC = int(sys.argv[1]) if len(sys.argv) > 1 else 1

_EL = re.compile(r"^\s*(\d*)\s+(\d+)\s+(BAR|QUAD4|TRIA3)\s+([-+.\dE]+)\s+"
                 r"([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)\s+"
                 r"([-+.\dE]+)")
_SC = re.compile(r"SUBCASE\s+(\d+)\s*$")
COMP = ("T1", "T2", "T3", "R1", "R2", "R3")


def parse_gpforce_elements(path, want_sc):
    """{(nid, eid, type): ndarray(6)} for one subcase.

    In the GPFORCE block the POINT-ID is printed only on the first row of
    each grid; subsequent rows for the same grid leave it blank.
    """
    out = {}
    sc = None
    in_blk = False
    cur_nid = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = _SC.search(line)
            if m:
                sc = int(m.group(1))
            if "G R I D   P O I N T   F O R C E" in line:
                in_blk = sc == want_sc
                continue
            if not in_blk:
                continue
            s = line[1:] if line[:1] == "0" else line
            head = s[:12].strip()
            if head.isdigit():
                cur_nid = int(head)
            r = _EL.match(s)
            if r and cur_nid is not None:
                out[(cur_nid, int(r.group(2)), r.group(3))] = np.array(
                    [float(r.group(i)) for i in range(4, 10)])
    return out


def main():
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.elements.bar import CBarElement
    from ascent_load.elements.quad4 import CQuad4Element
    from ascent_load.fem.assembly import _resolve_bar_orientation

    model = parse_bdf(os.path.join(ILC8, f"{STEM}.bdf"))
    model.cross_reference()
    u = parse_displacements(os.path.join(ILC8, f"{STEM}_MSC.f06"))[SC]
    gp = parse_gpforce_elements(os.path.join(ILC8, f"{STEM}_MSC.f06"), SC)
    print(f"MSC GPFORCE 요소 행 {len(gp)} (SC{SC})")

    # 요소 종류별·성분별로 (ASCENT k·u) 와 (-MSC 행) 의 짝을 모은다
    pairs = {"BAR": {c: [] for c in COMP}, "QUAD4": {c: [] for c in COMP}}
    per_elem = {"BAR": [], "QUAD4": []}
    n_bar = n_quad = 0
    for eid, el in model.elements.items():
        typ = type(el).__name__
        if typ.startswith("CBAR"):
            prop = el.property_ref
            mat = prop.material_ref
            n1, n2 = (model.nodes[i] for i in el.node_ids)
            v = _resolve_bar_orientation(el, model, n1)
            k = CBarElement(n1.xyz_global, n2.xyz_global, v, mat.E, mat.G,
                            prop.A, prop.I1, prop.I2, prop.J, mat.rho,
                            prop.nsm, pa=getattr(el, "pa", 0),
                            pb=getattr(el, "pb", 0),
                            wa=getattr(el, "wa", None),
                            wb=getattr(el, "wb", None)).stiffness_matrix()
            key, nn = "BAR", 2
        elif typ.startswith("CQUAD4"):
            prop = el.property_ref
            mat = prop.material_ref
            xyz = np.array([model.nodes[i].xyz_global for i in el.node_ids])
            k = CQuad4Element(xyz, mat.E, mat.nu, prop.t, mat.rho,
                              r12=float(getattr(prop, "ratio_12it3", 1.0) or 1.0),
                              nsm=float(getattr(prop, "nsm", 0.0) or 0.0)
                              ).stiffness_matrix()
            key, nn = "QUAD4", 4
        else:
            continue
        if any(i not in u for i in el.node_ids):
            continue
        ue = np.concatenate([u[i] for i in el.node_ids])
        f = k @ ue
        rows = []
        for j, nid in enumerate(el.node_ids):
            g = gp.get((nid, eid, key))
            if g is None:
                continue
            fa = f[6 * j:6 * j + 6]
            fm = -g
            rows.append((fa, fm))
            for c in range(6):
                pairs[key][COMP[c]].append((fa[c], fm[c]))
        if rows:
            fa_all = np.concatenate([r[0] for r in rows])
            fm_all = np.concatenate([r[1] for r in rows])
            s = float(fa_all @ fm_all / (fm_all @ fm_all)) if fm_all @ fm_all > 0 else np.nan
            per_elem[key].append((eid, s, float(np.linalg.norm(fm_all))))
        if key == "BAR":
            n_bar += 1
        else:
            n_quad += 1
    print(f"대조한 요소: CBAR {n_bar}, CQUAD4 {n_quad}")

    out = {}
    print(f"\n{'종류':6s} {'성분':4s} {'n':>6s} {'s = AL/MSC':>11s} {'배율후 L2':>10s} "
          f"{'|MSC|':>12s}")
    for key in ("BAR", "QUAD4"):
        out[key] = {}
        for c in COMP:
            arr = np.array(pairs[key][c])
            if len(arr) == 0:
                continue
            a, b = arr[:, 0], arr[:, 1]
            if b @ b < 1e-20:
                continue
            s = float(a @ b / (b @ b))
            l2 = float(np.linalg.norm(a - s * b) / np.linalg.norm(b))
            l2_0 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            out[key][c] = dict(n=int(len(a)), s=s, l2_after=l2, l2_raw=l2_0,
                               norm_msc=float(np.linalg.norm(b)))
            print(f"{key:6s} {c:4s} {len(a):6d} {s:11.5f} {l2 * 100:9.2f}% "
                  f"{np.linalg.norm(b):12.4e}   (원시 L2 {l2_0 * 100:.2f}%)")
    # 요소별 배율 분포
    for key in ("BAR", "QUAD4"):
        ss = np.array([p[1] for p in per_elem[key] if np.isfinite(p[1])])
        w = np.array([p[2] for p in per_elem[key] if np.isfinite(p[1])])
        if len(ss):
            wm = float((ss * w).sum() / w.sum())
            print(f"{key}: 요소별 배율 중앙값 {np.median(ss):.4f}, 하중가중 평균 {wm:.4f}, "
                  f"5~95% {np.percentile(ss, 5):.4f}~{np.percentile(ss, 95):.4f}")
            out[key]["per_element"] = dict(median=float(np.median(ss)),
                                           load_weighted=wm,
                                           p5=float(np.percentile(ss, 5)),
                                           p95=float(np.percentile(ss, 95)))
    with open(os.path.join(HERE, "r5_msc_element_check.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
