# RBE2 종속 절점이 스플라인 SET 에 있을 때 공탄성 해석 G 행렬이 그 가중치를 주 자유도로 옮기는지(힘 전달 합 1, SUPORT 위치 불변) 검사한다
"""Spline coupling with RBE2-dependent structural nodes (2026-09-08).

ILC-8 ties every wing-root node to a fuselage skin node with an RBE2.
Those root nodes are in the SPLINE1 SET1, but RBE2 elimination removes
their DOFs from the f-set. Before the fix, `_fill_geff` silently dropped
their IPS weights from G_w / G_d used in the trim solve: the in-fuselage
root boxes (whose weights concentrate on the root station) reached the
structure with a force-transfer sum of -1.6 instead of 1, and the lost
lift (~30% of the wing) was reacted at the SUPORT nodes. On the ILC-8
keel-skin SUPORT that reaction dimpled the reference triad by 3.7 mm
and showed up as the 0.38 deg/g trim-alpha bias against MSC.

The model here is a flat symmetric plate wing (7 stations x 2 chord
nodes) whose two root nodes are RBE2 slaves of a fuselage mass node.
Two checks:
  1. every box's force-transfer sum (T3 columns of G_d, slaves folded
     into masters with coefficient 1) equals 1;
  2. the trimmed aero forces do not depend on where the SUPORT is
     (fuselage node vs wing 3-2-1), and the displacement fields differ
     by a rigid-body motion only.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from ascent_load.bdf.parser import parse_bdf
from ascent_load.solvers.sol144 import (_build_shared_data,
                                        _build_geff_per_spline, solve_trim)

YS = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]


def _deck(suport: str) -> str:
    L = ["SOL 144", "CEND", "TITLE = RBE2 SLAVE SPLINE WEIGHT TEST", "DISP = ALL",
         "SUBCASE 1", "  TRIM = 1", "BEGIN BULK", "PARAM   WTMASS  1.0"]
    f8 = lambda v: f"{v:<8.6g}"[:8]
    # 동체 질량 절점(1/4-코드 x=250 에 두어 피칭 모멘트 0), 날개 절점 10k+ (LE), 10k+1 (TE)
    L.append(f"GRID    1               {f8(250.0)}{f8(0.0)}{f8(-300.0)}")
    for k, y in enumerate(YS):
        L.append(f"GRID    {10 + 10 * k:<8d}        {f8(0.0)}{f8(y)}{f8(0.0)}")
        L.append(f"GRID    {11 + 10 * k:<8d}        {f8(1000.0)}{f8(y)}{f8(0.0)}")
    for k in range(6):
        a, b = 10 + 10 * k, 10 + 10 * (k + 1)
        L.append(f"CQUAD4  {k + 1:<8d}1       {a:<8d}{a + 1:<8d}{b + 1:<8d}{b:<8d}")
    L.append("PSHELL  1       1       40.0    1")
    L.append("MAT1    1       70000.0         0.3")  # RHO 0: 질량은 CONM2 뿐 → CG 가 1/4-코드, 피칭 모멘트 0
    L.append("CONM2   100     1       0       0.2")
    # 루트 절점(y=0: 40, 41)을 동체 절점 1 에 RBE2 로 묶는다
    L.append("RBE2    900     1       123456  40      41")
    L.append("AEROS   0       0       1000.0  6000.0  6.0+6")
    L.append("PAERO1  1")
    L.append("CAERO1  1001    1       0       6       1       0       0       1")
    L.append(f"+       {f8(0.0)}{f8(-3000.0)}{f8(0.0)}{f8(1000.0)}{f8(0.0)}{f8(3000.0)}{f8(0.0)}{f8(1000.0)}")
    nids = [10 + 10 * k for k in range(7)] + [11 + 10 * k for k in range(7)]
    L.append("SET1    1       " + "".join(f"{n:<8d}" for n in nids[:7]))
    L.append("+       " + "".join(f"{n:<8d}" for n in nids[7:]))
    L.append("SPLINE1 1       1001    1001    1006    1")
    L.append(suport)
    L.append("AESTAT  1       ANGLEA")
    L.append("AESTAT  2       URDD3")
    L.append("TRIM    1       0.0     0.001   URDD3   1.0")
    L.append("ENDDATA")
    return "\n".join(L) + "\n"


SUPORT_FUSELAGE = "SUPORT  1       123456"
# 날개 3-2-1: 70(123), 71(23: x 방향 1000 mm → 요·피치), 50(3: y 방향 −2000 mm → 롤)
SUPORT_WING = "SUPORT  70      123     71      23      50      3"


def _write(tmpdir, name, suport):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        f.write(_deck(suport))
    return path


def _t3_row_sums(model, slave_deps_on: bool):
    shared = _build_shared_data(model)
    dof_mgr = shared.dof_mgr
    if slave_deps_on:
        Gd = shared.G_disp.toarray()
    else:
        b2i = shared.box_id_to_index
        _, Gd = _build_geff_per_spline(model, shared.boxes, b2i, dof_mgr,
                                       list(shared.f_dofs), slave_deps=None)
    idx = {d: i for i, d in enumerate(shared.f_dofs)}
    t3 = [idx[dof_mgr.get_dof(n, 3)] for n in model.nodes
          if dof_mgr.get_dof(n, 3) in idx]
    return Gd[:, t3].sum(axis=1), shared


class TestRBE2SlaveSplineWeights:

    def test_force_transfer_sum_is_one_with_slaves_folded(self):
        with tempfile.TemporaryDirectory() as td:
            model = parse_bdf(_write(td, "a.bdf", SUPORT_FUSELAGE))
            model.cross_reference()
            s_fixed, shared = _t3_row_sums(model, slave_deps_on=True)
            assert len(shared.slave_deps) == 12          # 절점 40, 41 × 6 자유도
            np.testing.assert_allclose(s_fixed, 1.0, atol=1e-10)
            # 종속 자유도를 접지 않으면 루트 인접 박스의 힘 전달 합이 1 에서 벗어난다
            s_dropped, _ = _t3_row_sums(model, slave_deps_on=False)
            assert np.abs(s_dropped - 1.0).max() > 0.05

    def test_aero_forces_independent_of_suport_location(self):
        with tempfile.TemporaryDirectory() as td:
            mA = parse_bdf(_write(td, "a.bdf", SUPORT_FUSELAGE))
            mB = parse_bdf(_write(td, "b.bdf", SUPORT_WING))
            rA = solve_trim(mA, n_workers=0).subcases[0]
            rB = solve_trim(mB, n_workers=0).subcases[0]
            fA, fB = np.asarray(rA.aero_forces), np.asarray(rB.aero_forces)
            assert np.abs(fA[:, 2]).max() > 1.0
            np.testing.assert_allclose(fB, fA, rtol=1e-6, atol=1e-6 * np.abs(fA).max())
            # 변위장 차이는 강체 운동뿐: 절점 좌표로 t + θ×r 최소제곱 후 잔차 ≈ 0
            nids = sorted(set(rA.displacements) & set(rB.displacements))
            xyz = np.array([mA.nodes[n].xyz_global for n in nids])
            dU = np.array([rB.displacements[n][:3] - rA.displacements[n][:3] for n in nids])
            r = xyz - xyz.mean(axis=0)
            A = np.zeros((3 * len(nids), 6))
            for i, (rx, ry, rz) in enumerate(r):
                A[3 * i:3 * i + 3, :3] = np.eye(3)
                A[3 * i:3 * i + 3, 3:] = [[0, rz, -ry], [-rz, 0, rx], [ry, -rx, 0]]
            p, *_ = np.linalg.lstsq(A, dU.reshape(-1), rcond=None)
            resid = dU.reshape(-1) - A @ p
            scale = max(np.abs(rA.displacements[70][2]), 1e-9)
            assert np.abs(resid).max() / scale < 1e-6
            assert np.abs(dU).max() / scale > 1e-3   # 두 기준계는 실제로 다르다
