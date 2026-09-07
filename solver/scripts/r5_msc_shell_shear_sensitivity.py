# CQUAD4 의 막·굽힘·횡전단·드릴링 기여를 각각 스케일해 MSC 대비 K 배율의 민감도를 잰다 (r5 MC5 추적)
"""Which part of the CQUAD4 carries ASCENT-Load's excess global stiffness?

The hold-out comparison (r5_msc_holdout_metrics.py) showed the elastic
field is one scalar off MSC's; r5_msc_k_isolation.py showed the scalar
persists in plain SOL 101 under MSC's own load (so it is K, not the load
or the aeroelastic coupling); r5_msc_element_check.py showed CBAR
stiffness is identical to MSC's while CQUAD4 differs. This script closes
the loop: it re-solves the K-isolation deck with each CQUAD4 stiffness
contribution scaled in turn and reports the global scale factor s
(ASCENT-Load elastic field = MSC field / s) per region.

The result on ILC-8 v9: only the TRANSVERSE SHEAR term moves s toward 1
(x1: 1.256, x0.1: 1.107, x0.01: 1.036, x0.001: 1.021, with the shape
residual falling to 0.6 %); bending, membrane and drilling do not. The
Mindlin shear penalty (kappa G t, one-point SRI) therefore over-constrains
the assembled bar-shell structure. Flat, distorted, curved and stiffened
unit tests all pass at 0.99-1.01, so the effect is assembly-specific; the
scaling here is a DIAGNOSTIC, not a fix -- the fix direction is a
formulation that enforces the thin-shell rotation/slope relation without a
penalty (discrete-Kirchhoff or MSC-equivalent shear-rigid treatment when
PSHELL MID3 is blank).

Requires: tests/validation/ILC8/ilc8_k_isolation_sc1.bdf (made by
          r5_msc_k_isolation.py) and the MSC F06.
Usage:    cd solver && python scripts/r5_msc_shell_shear_sensitivity.py
Output:   r5_msc_shell_shear_sensitivity.json next to this script.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)

from ascent_load.fem import assembly as asm                       # noqa: E402
from ascent_load.elements.quad4 import CQuad4Element, _GP2, _GW2  # noqa: E402
from r3_msc_displacement_metrics import parse_displacements, remove_rigid  # noqa: E402
from r5_msc_holdout_metrics import scale_fit, REGIONS             # noqa: E402

ILC8 = os.path.join(SOLVER, "tests", "validation", "ILC8")
STEM = "ilc8_msc_sol144_v9_holdout"
DECK = os.path.join(ILC8, "ilc8_k_isolation_sc1.bdf")

SCALE = dict(mem=1.0, bend=1.0, shear=1.0, drill=1.0)


def local_k(xy, E, nu, t, r12):
    """CQuad4Element._local_stiffness 를 기여별 스케일 훅과 함께 재현."""
    obj = CQuad4Element.__new__(CQuad4Element)
    obj.xy_local = xy
    Dm = SCALE["mem"] * (E * t / (1 - nu ** 2)) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Db = SCALE["bend"] * (r12 * E * t ** 3 / (12 * (1 - nu ** 2))) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ds = SCALE["shear"] * (5.0 / 6.0) * (E * t / (2 * (1 + nu))) * np.eye(2)
    k = np.zeros((24, 24))
    mem, bend, sh = [], [], []
    for n in range(4):
        mem += [6 * n, 6 * n + 1]
        bend += [6 * n + 3, 6 * n + 4]
        sh += [6 * n + 2, 6 * n + 3, 6 * n + 4]
    for i in range(2):
        for j in range(2):
            xi, eta = _GP2[i], _GP2[j]
            w = _GW2[i] * _GW2[j]
            N, dxi, deta = obj._shape_functions(xi, eta)
            J = obj._jacobian(dxi, deta)
            detJ = np.linalg.det(J)
            Ji = np.linalg.inv(J)
            dNdx = Ji[0, 0] * dxi + Ji[0, 1] * deta
            dNdy = Ji[1, 0] * dxi + Ji[1, 1] * deta
            Bm = np.zeros((3, 8))
            Bb = np.zeros((3, 8))
            for n in range(4):
                Bm[0, 2 * n] = dNdx[n]
                Bm[1, 2 * n + 1] = dNdy[n]
                Bm[2, 2 * n] = dNdy[n]
                Bm[2, 2 * n + 1] = dNdx[n]
                Bb[0, 2 * n + 1] = -dNdx[n]
                Bb[1, 2 * n] = dNdy[n]
                Bb[2, 2 * n] = dNdx[n]
                Bb[2, 2 * n + 1] = -dNdy[n]
            k[np.ix_(mem, mem)] += Bm.T @ Dm @ Bm * detJ * w
            k[np.ix_(bend, bend)] += Bb.T @ Db @ Bb * detJ * w
    N, dxi, deta = obj._shape_functions(0.0, 0.0)
    J = obj._jacobian(dxi, deta)
    detJ = np.linalg.det(J)
    Ji = np.linalg.inv(J)
    dNdx = Ji[0, 0] * dxi + Ji[0, 1] * deta
    dNdy = Ji[1, 0] * dxi + Ji[1, 1] * deta
    Bs = np.zeros((2, 12))
    for n in range(4):
        Bs[0, 3 * n] = dNdx[n]
        Bs[0, 3 * n + 2] = N[n]
        Bs[1, 3 * n] = dNdy[n]
        Bs[1, 3 * n + 1] = -N[n]
    k[np.ix_(sh, sh)] += Bs.T @ Ds @ Bs * detJ * 4.0
    d13 = xy[2] - xy[0]
    d24 = xy[3] - xy[1]
    area = 0.5 * abs(d13[0] * d24[1] - d13[1] * d24[0])
    for n in range(4):
        k[6 * n + 5, 6 * n + 5] += SCALE["drill"] * E * t * area * 1e-6
    return k


def _batch(xy_local, E_, nu_, t_, n_elem, r12_=1.0):
    r12 = np.broadcast_to(np.asarray(r12_, float), (n_elem,))
    return np.stack([local_k(xy_local[e], float(E_[e]), float(nu_[e]),
                             float(t_[e]), float(r12[e]))
                     for e in range(n_elem)])


def run_cli(bdf):
    import ascent_load.__main__ as cli
    old = sys.argv
    sys.argv = ["ascent_load", bdf]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.main()
    finally:
        sys.argv = old


def main():
    from ascent_load.bdf.parser import parse_bdf
    if not os.path.isfile(DECK):
        sys.exit("먼저 scripts/r5_msc_k_isolation.py 를 돌려 K 분리 덱을 만든다")
    asm._batch_cquad4_stiffness = _batch
    model = parse_bdf(os.path.join(ILC8, f"{STEM}.bdf"))
    d_msc = parse_displacements(os.path.join(ILC8, f"{STEM}_MSC.f06"))[1]
    f06 = DECK.replace(".bdf", ".f06")

    def measure():
        run_cli(DECK)
        d = parse_displacements(f06)
        d = d[sorted(d)[0]]
        common = sorted(set(d_msc) & set(d) & set(model.nodes))
        xyz = np.array([model.nodes[n].xyz_global for n in common])
        em = remove_rigid(xyz, np.array([d_msc[n] for n in common]))
        e1 = remove_rigid(xyz, np.array([d[n] for n in common]))
        s, l0, ls = scale_fit(e1, em)
        reg = {}
        for nm, lo, hi in REGIONS:
            ii = np.array([i for i, n in enumerate(common) if lo <= n <= hi])
            if len(ii) >= 10:
                reg[nm] = scale_fit(e1[ii, 2], em[ii, 2])[0]
        return dict(s=s, l2_pct=l0 * 100, l2_after_pct=ls * 100, per_region=reg)

    cases = [("baseline", {}),
             ("shear x0.1", dict(shear=0.1)), ("shear x0.01", dict(shear=0.01)),
             ("shear x0.001", dict(shear=0.001)),
             ("bend x0.1", dict(bend=0.1)), ("mem x0.8", dict(mem=0.8)),
             ("mem x0.5", dict(mem=0.5)), ("drill x0", dict(drill=0.0))]
    out = {}
    print("=== ILC-8 K 분리 (MSC SC1 하중, SOL 101): CQUAD4 기여별 스케일 ===")
    for label, sc in cases:
        SCALE.update(mem=1.0, bend=1.0, shear=1.0, drill=1.0)
        SCALE.update(sc)
        r = measure()
        out[label] = r
        reg = " ".join(f"{k[:5]}={v:.3f}" for k, v in r["per_region"].items())
        print(f"  {label:14s} s={r['s']:.4f}  L2 {r['l2_pct']:6.2f}% -> "
              f"{r['l2_after_pct']:5.2f}%   [{reg}]", flush=True)
    with open(os.path.join(HERE, "r5_msc_shell_shear_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
