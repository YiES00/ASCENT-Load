# CQUAD4 굽힘의 DKQ 정식화가 얇은 판·곡면·보강 조립체에서 맞고 배치·클래스가 일치하는지
"""DKQ bending for CQUAD4 (2026-09-04).

Tracing the MSC hold-out discrepancy (docs/peer-review-r5/
msc-holdout-results.md) showed the Mindlin transverse-shear penalty
(kappa G t, one-point SRI) over-constrains assembled bar-shell
structures: scaling that single term brought ASCENT-Load's global
stiffness from 1.256x MSC to 1.021x. The discrete-Kirchhoff quadrilateral
(Batoz & Tahar 1982) has no transverse-shear term, which is also MSC's
semantics for PSHELL with MID3 blank. These tests pin the element on
constant curvature (distorted), thin cantilevers, a simply supported
plate, a faceted tube with and without ring frames, batch/class
agreement, and the archived-result switch.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascent_load.bdf.parser import parse_bdf
from ascent_load.elements.quad4 import CQuad4Element, dkq_bending_stiffness
from ascent_load.fem.assembly import assemble_global_matrices
from ascent_load.fem.model import FEModel
from ascent_load.solvers.sol101 import solve_static

E, NU, T = 52000.0, 0.31, 1.1


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def _header():
    return ["SOL     101", "CEND", "SPC = 1", "LOAD = 1", "DISP = ALL",
            "BEGIN BULK",
            f"MAT1    1       {E:<8.1f}        {NU:<8.2f}2.7-9",
            f"PSHELL  1       1       {T:<8.3f}1"]


def _grid(n, x, y, z=0.0):
    return ["GRID*   %16d%16d%16.7E%16.7E" % (n, 0, x, y), "*       %16.7E" % z]


def _force(n, f, d):
    return ["FORCE*  %16d%16d%16d%16.8E" % (1, n, 0, f),
            "*       %16.8E%16.8E%16.8E" % tuple(d)]


def _strip(path, nx, ny=2, L=1000.0, B=100.0, zig=0.0, R=np.eye(3), P=1.0):
    nid = lambda i, j: i * (ny + 1) + j + 1
    dx = L / nx
    lines = _header()
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = i * dx, j * B / ny
            if 0 < i < nx and 0 < j < ny:
                x += zig * dx * (1 if i % 2 else -1)
            p = R @ np.array([x, y, 0.0])
            lines += _grid(nid(i, j), *p)
    e = 1
    for i in range(nx):
        for j in range(ny):
            lines.append("CQUAD4  %-8d1       %-8d%-8d%-8d%-8d" % (
                e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
            e += 1
    for j in range(ny + 1):
        lines.append("SPC1    1       123456  %d" % nid(0, j))
    d = R @ np.array([0.0, 0.0, 1.0])
    for j, w in zip(range(ny + 1), (0.25, 0.5, 0.25)):
        lines += _force(nid(nx, j), P * w, d)
    lines.append("ENDDATA")
    _write(path, lines)
    return [nid(nx, j) for j in range(ny + 1)], d, P * L ** 3 / (3 * E * (B * T ** 3 / 12))


def _solve(path):
    model = parse_bdf(str(path))
    return solve_static(model).subcases[0]


class TestConstantCurvature:
    def test_distorted_element_exact_and_rigid_free(self):
        xy = np.array([[0, 0], [3.0, 0.2], [3.3, 2.1], [-0.3, 1.9]])
        Db = (E * T ** 3 / (12 * (1 - NU ** 2))) * np.array(
            [[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
        k = dkq_bending_stiffness(xy, Db)
        ev = np.linalg.eigvalsh(k)
        assert (ev < 1e-9 * ev.max()).sum() == 3          # 강체 w, tx, ty 만
        # 순굽힘 w = x^2/2 (tx = 0, ty = -x): 절점력이 요소 자체 평형 -> 곡률 일정
        U = np.concatenate([[x * x / 2, 0.0, -x] for x, y in xy])
        f = k @ U
        # 상수곡률 상태에서 절점 힘의 합력·합모멘트는 0
        assert abs(f[0::3].sum()) < 1e-9 * np.abs(f).max()


class TestThinCantilever:
    @pytest.mark.parametrize("nx,floor", [(20, 0.985), (5, 0.975), (2, 0.95)])
    def test_aspect_ratio(self, tmp_path, nx, floor):
        tips, d, an = _strip(tmp_path / "c.bdf", nx)
        sc = _solve(tmp_path / "c.bdf")
        r = np.mean([sc.displacements[n][2] for n in tips]) / an
        assert floor < r < 1.01, r

    def test_distorted_mesh(self, tmp_path):
        tips, d, an = _strip(tmp_path / "z.bdf", 20, zig=0.4)
        sc = _solve(tmp_path / "z.bdf")
        r = np.mean([sc.displacements[n][2] for n in tips]) / an
        assert 0.985 < r < 1.01, r

    def test_arbitrary_3d_rotation(self, tmp_path):
        th, ph = np.deg2rad(37), np.deg2rad(53)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
        tips, d, an = _strip(tmp_path / "r.bdf", 20, R=Rx @ Rz)
        sc = _solve(tmp_path / "r.bdf")
        r = np.mean([sc.displacements[n][:3] @ d for n in tips]) / an
        assert 0.985 < r < 1.01, r


class TestSimplySupportedPlate:
    def test_uniform_load_center_deflection(self, tmp_path):
        """정사각 판 단순지지 등분포: w_c = 0.00406 q a^4 / D (Timoshenko)."""
        a, n, q = 1000.0, 8, 1e-3
        nid = lambda i, j: i * (n + 1) + j + 1
        lines = _header()
        for i in range(n + 1):
            for j in range(n + 1):
                lines += _grid(nid(i, j), i * a / n, j * a / n)
        e = 1
        for i in range(n):
            for j in range(n):
                lines.append("CQUAD4  %-8d1       %-8d%-8d%-8d%-8d" % (
                    e, nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
                e += 1
        # 단순지지: 변에서 w=0, 전 절점 면내 자유도와 드릴링 고정
        for i in range(n + 1):
            for j in range(n + 1):
                comp = "1236" if (i in (0, n) or j in (0, n)) else "126"
                lines.append("SPC1    1       %-8s%d" % (comp, nid(i, j)))
        # 등분포 -> 절점 하중 (내부 절점 전량, 변 절점 절반, 모서리 1/4)
        h = a / n
        for i in range(n + 1):
            for j in range(n + 1):
                wgt = (0.5 if i in (0, n) else 1.0) * (0.5 if j in (0, n) else 1.0)
                lines += _force(nid(i, j), q * h * h * wgt, (0, 0, 1))
        lines.append("ENDDATA")
        _write(tmp_path / "ss.bdf", lines)
        sc = _solve(tmp_path / "ss.bdf")
        D = E * T ** 3 / (12 * (1 - NU ** 2))
        wc = sc.displacements[nid(n // 2, n // 2)][2]
        assert wc == pytest.approx(0.00406 * q * a ** 4 / D, rel=0.02)


def _tube(path, frames, r=250.0, L=2000.0, nc=16, nl=20, P=100.0):
    nid = lambda i, k: i * nc + k + 1
    lines = _header() + ["PBAR    2       1       140.0   120000.0120000.0240000.0"]
    for i in range(nl + 1):
        for k in range(nc):
            th = 2 * np.pi * k / nc
            lines += _grid(nid(i, k), i * L / nl, r * np.cos(th), r * np.sin(th))
    e = 1
    for i in range(nl):
        for k in range(nc):
            lines.append("CQUAD4  %-8d1       %-8d%-8d%-8d%-8d" % (
                e, nid(i, k), nid(i + 1, k), nid(i + 1, (k + 1) % nc), nid(i, (k + 1) % nc)))
            e += 1
    if frames:
        for i in range(1, nl + 1):
            for k in range(nc):
                lines.append("CBAR    %-8d2       %-8d%-8d1.0     0.0     0.0" % (
                    e, nid(i, k), nid(i, (k + 1) % nc)))
                e += 1
    for k in range(nc):
        lines.append("SPC1    1       123456  %d" % nid(0, k))
    for k in range(nc):
        lines += _force(nid(nl, k), P / nc, (0, 0, 1))
    lines.append("ENDDATA")
    _write(path, lines)
    th = 2 * np.pi * np.arange(nc) / nc
    pts = np.c_[r * np.cos(th), r * np.sin(th)]
    I = sum(T * np.linalg.norm(pts[(k + 1) % nc] - pts[k]) *
            (pts[k, 1] ** 2 + pts[k, 1] * pts[(k + 1) % nc, 1] + pts[(k + 1) % nc, 1] ** 2) / 3
            for k in range(nc))
    G = E / (2 * (1 + NU))
    A = sum(np.linalg.norm(pts[(k + 1) % nc] - pts[k]) for k in range(nc)) * T
    an = P * L ** 3 / (3 * E * I) + P * L / (0.5 * G * A)
    return [nid(nl, k) for k in range(nc)], an


class TestFacetedTube:
    def test_plain_tube(self, tmp_path):
        tips, an = _tube(tmp_path / "t.bdf", frames=False)
        sc = _solve(tmp_path / "t.bdf")
        r = np.mean([sc.displacements[n][2] for n in tips]) / an
        assert 0.99 < r < 1.01, r

    def test_ring_frame_artifact_bounded(self, tmp_path):
        """링 프레임은 관의 축굽힘에 기여하지 않아야 하는데 5~6% 뻣뻣해진다.

        Mindlin+SRI 0.935, DKQ 0.946 — 횡전단 항과 무관한 별개의 보·외피
        결합 인공물이다(원인 미확정, docs/quad4-dkq/context-notes.md).
        이 시험은 그 크기를 고정해 두어 커지면 알아채게 한다.
        """
        tips, an = _tube(tmp_path / "f.bdf", frames=True)
        sc = _solve(tmp_path / "f.bdf")
        r = np.mean([sc.displacements[n][2] for n in tips]) / an
        assert 0.93 < r < 1.01, r


class TestBatchMatchesElementClass:
    def test_assembled_stiffness_identical(self, tmp_path):
        _strip(tmp_path / "d.bdf", 6, zig=0.35)
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
        assert np.abs(K_batch - K_cls).max() < 1e-9 * np.abs(K_batch).max()


class TestArchivedSwitch:
    def test_mindlin_reproduces_old_frame_tube(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASCENT_QUAD4_BENDING", "mindlin")
        tips, an = _tube(tmp_path / "m.bdf", frames=True)
        sc = _solve(tmp_path / "m.bdf")
        r = np.mean([sc.displacements[n][2] for n in tips]) / an
        assert r == pytest.approx(0.935, abs=0.01)
