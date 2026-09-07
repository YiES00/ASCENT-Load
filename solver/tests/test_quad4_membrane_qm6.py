# CQUAD4 막의 QM6 비적합 모드가 면내 굽힘 잠금을 풀고 배치·클래스 경로가 일치하는지
"""QM6 membrane for CQUAD4 (2026-09-04).

The bilinear Q4 membrane locks in in-plane bending through parasitic
shear: a 2-element-deep cantilever strip loaded in its own plane gives
0.887 of the analytic tip deflection at element aspect ratio 1, 0.389 at
aspect 4 and 0.094 at aspect 10 (found while tracing the MSC hold-out
discrepancy, docs/peer-review-r5/msc-holdout-results.md). The QM6
incompatible modes with Taylor's centre-Jacobian correction restore
0.995 / 0.977 / 0.910. These tests pin that, keep the transverse
(plate) response unchanged, prove the vectorised assembler and the
element class produce the same matrices, and keep the archived-result
switch working.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascent_load.bdf.parser import parse_bdf
from ascent_load.elements.quad4 import CQuad4Element
from ascent_load.fem.assembly import assemble_global_matrices
from ascent_load.fem.model import FEModel
from ascent_load.solvers.sol101 import solve_static

E, NU, T = 52000.0, 0.31, 1.1
L, B, P = 1000.0, 100.0, 1.0


def _strip_deck(path, nx, ny=2, inplane=True, zig=0.0):
    """외팔 판 스트립. inplane=True 면 자기 평면 안에서 굽힌다."""
    nid = lambda i, j: i * (ny + 1) + j + 1
    dx = L / nx
    lines = ["SOL     101", "CEND", "SPC = 1", "LOAD = 1", "DISP = ALL",
             "BEGIN BULK",
             f"MAT1    1       {E:<8.1f}        {NU:<8.2f}2.7-9",
             f"PSHELL  1       1       {T:<8.3f}1"]
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = i * dx, j * B / ny
            if 0 < i < nx and 0 < j < ny:
                x += zig * dx * (1 if i % 2 else -1)
            lines.append("GRID*   %16d%16d%16.7E%16.7E" % (nid(i, j), 0, x, y))
            lines.append("*       %16.7E" % 0.0)
    e = 1
    for i in range(nx):
        for j in range(ny):
            lines.append("CQUAD4  %-8d1       %-8d%-8d%-8d%-8d" % (
                e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
            e += 1
    for j in range(ny + 1):
        lines.append("SPC1    1       123456  %d" % nid(0, j))
    w = [0.25, 0.5, 0.25]
    d = (0.0, 1.0, 0.0) if inplane else (0.0, 0.0, 1.0)
    for j in range(ny + 1):
        lines.append("FORCE*  %16d%16d%16d%16.8E" % (1, nid(nx, j), 0, P * w[j]))
        lines.append("*       %16.8E%16.8E%16.8E" % d)
    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n")
    return [nid(nx, j) for j in range(ny + 1)]


def _tip_ratio(path, nx, inplane, zig=0.0):
    tips = _strip_deck(path, nx, inplane=inplane, zig=zig)
    model = parse_bdf(str(path))
    sc = solve_static(model).subcases[0]
    comp = 1 if inplane else 2
    got = np.mean([sc.displacements[n][comp] for n in tips])
    G = E / (2 * (1 + NU))
    if inplane:
        analytic = P * L ** 3 / (3 * E * (T * B ** 3 / 12)) + P * L / ((5 / 6) * G * B * T)
    else:
        analytic = P * L ** 3 / (3 * E * (B * T ** 3 / 12))
    return got / analytic


class TestInPlaneBendingNoLongerLocks:
    @pytest.mark.parametrize("nx,aspect,floor", [(20, 1, 0.99), (5, 4, 0.97),
                                                  (2, 10, 0.90)])
    def test_aspect_ratio(self, tmp_path, nx, aspect, floor):
        r = _tip_ratio(tmp_path / "s.bdf", nx, inplane=True)
        assert r > floor, f"aspect {aspect}: {r:.4f} (Q4 gave 0.887/0.389/0.094)"
        assert r < 1.02

    def test_distorted_mesh_still_fine(self, tmp_path):
        # 내부 절점 지그재그 (기하 고정). Q4 는 0.805, QM6 프로토타입은 0.916.
        r = _tip_ratio(tmp_path / "z.bdf", 20, inplane=True, zig=0.4)
        assert 0.90 < r < 1.02, r


class TestPlateResponseUnchanged:
    def test_transverse_cantilever(self, tmp_path, monkeypatch):
        """막 정식화는 판(횡) 응답을 건드리지 않는다 — 굽힘 정식화가 무엇이든.

        절대값은 굽힘 정식화에 달렸다(Mindlin+SRI 0.9936, DKQ 0.9897)
        so qm6 와 q4 의 횡하중 결과가 같은지를 본다.
        """
        r_qm6 = _tip_ratio(tmp_path / "t6.bdf", 20, inplane=False)
        monkeypatch.setenv("ASCENT_QUAD4_MEMBRANE", "q4")
        r_q4 = _tip_ratio(tmp_path / "t4.bdf", 20, inplane=False)
        assert r_qm6 == pytest.approx(r_q4, rel=1e-9)
        assert 0.985 < r_qm6 < 1.0


class TestPatchTest:
    def test_constant_strain_on_distorted_patch(self, tmp_path):
        """단축 인장을 받는 왜곡 5요소 패치 — 변위장은 선형 정확해야 한다."""
        a, b = 100.0, 60.0
        pts = {1: (0, 0), 2: (a, 0), 3: (a, b), 4: (0, b),
               5: (0.28 * a, 0.22 * b), 6: (0.71 * a, 0.31 * b),
               7: (0.66 * a, 0.74 * b), 8: (0.24 * a, 0.68 * b)}
        quads = [(1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8), (5, 6, 7, 8)]
        sx = 10.0                       # 단축 응력 -> 변위 u = sx x / E, v = -nu sx y / E
        lines = ["SOL     101", "CEND", "SPC = 1", "LOAD = 1", "DISP = ALL",
                 "BEGIN BULK",
                 f"MAT1    1       {E:<8.1f}        {NU:<8.2f}2.7-9",
                 f"PSHELL  1       1       {T:<8.3f}1"]
        for n, (x, y) in pts.items():
            lines.append("GRID*   %16d%16d%16.7E%16.7E" % (n, 0, x, y))
            lines.append("*       %16.7E" % 0.0)
        for k, q in enumerate(quads, 1):
            lines.append("CQUAD4  %-8d1       %-8d%-8d%-8d%-8d" % (k, *q))
        # 경계: 정확해를 SPCD 없이 강제하기 위해 x=0 변은 u=0, 절점 1 은 v=0,
        # 면외 자유도(w, θx, θy)는 전부 잠근다. x=a 변에 등가 절점하중.
        lines += ["SPC1    1       1       1       4", "SPC1    1       2       1",
                  "SPC1    1       345     1       THRU    8"]
        f_edge = sx * T * b
        for n, frac in ((2, 0.5), (3, 0.5)):
            lines.append("FORCE*  %16d%16d%16d%16.8E" % (1, n, 0, f_edge * frac))
            lines.append("*       %16.8E%16.8E%16.8E" % (1.0, 0.0, 0.0))
        lines.append("ENDDATA")
        p = tmp_path / "patch.bdf"
        p.write_text("\n".join(lines) + "\n")
        model = parse_bdf(str(p))
        sc = solve_static(model).subcases[0]
        for n, (x, y) in pts.items():
            u_exact = sx * x / E
            assert sc.displacements[n][0] == pytest.approx(u_exact, abs=1e-10 + 1e-8 * abs(u_exact))
        # v 는 강체 회전 성분이 자유라 기울기만 본다 (dv/dy = -nu sx / E)
        v3, v2 = sc.displacements[3][1], sc.displacements[2][1]
        assert (v3 - v2) / b == pytest.approx(-NU * sx / E, rel=1e-8)


class TestBatchMatchesElementClass:
    def test_assembled_stiffness_identical(self, tmp_path):
        tips = _strip_deck(tmp_path / "d.bdf", 6, inplane=True, zig=0.35)
        model = parse_bdf(str(tmp_path / "d.bdf"))
        model.cross_reference()
        fe = FEModel(model)
        K_batch, _, _ = assemble_global_matrices(model, fe.dof_mgr)
        K_batch = K_batch.toarray()
        K_cls = np.zeros_like(K_batch)
        for eid, el in model.elements.items():
            if not type(el).__name__.startswith("CQUAD4"):
                continue
            prop = el.property_ref
            mat = prop.material_ref
            xyz = np.array([model.nodes[i].xyz_global for i in el.node_ids])
            ke = CQuad4Element(xyz, mat.E, mat.nu, prop.t, mat.rho).stiffness_matrix()
            d = np.asarray(fe.dof_mgr.get_element_dofs(el.node_ids))
            K_cls[np.ix_(d, d)] += ke
        scale = np.abs(K_batch).max()
        assert np.abs(K_batch - K_cls).max() < 1e-9 * scale


class TestArchivedSwitch:
    def test_q4_reproduces_old_locking(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASCENT_QUAD4_MEMBRANE", "q4")
        r = _tip_ratio(tmp_path / "q4.bdf", 5, inplane=True)
        assert r == pytest.approx(0.389, abs=0.01)
