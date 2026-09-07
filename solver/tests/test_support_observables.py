# T5 지지 적합성 항등식(아핀 재현·합력 보존·분할 지지 가상일)과 드릴링 훅 연결의 단위 시험
"""Unit tests for the observable-dependent support study (T5).

Pins the algebraic identities the study rests on:
  * Lemma S7.1 — affine-reproduction identities of the augmented IPS
    weights (values: sum 1, sum x = x_j, sum y = y_j; slopes: sum 0,
    sum x = 1, sum y = 0) for an arbitrary non-collinear support.
  * Proposition S7.2 — force transfer through the transpose preserves
    the resultant and the in-plane moments of any box load, hence root
    section loads are support-independent.
  * Proposition S7.5 — the virtual-work identity of the force transfer
    involves only the force matrix, so it holds for any wash support.
  * The ASCENT_DRILL_SCALE research hook is read by the vectorized
    CQUAD4 assembly path and by CTRIA3, scales only the drilling (theta_z)
    diagonals, and leaves every other entry bit-identical; the batch path
    agrees with the element classes at a non-unit scale (regression for
    the r6 fix; the hook had been read by the element class only, so the
    archived drilling sweep was an artifact).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from ascent_load.aero.spline import build_ips_spline, build_ips_spline_slope


def _support(seed: int, n: int):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.0, 400.0, n),
                            rng.uniform(0.0, 2000.0, n)])


def _points(seed: int, n: int):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(20.0, 380.0, n),
                            rng.uniform(50.0, 1950.0, n)])


@pytest.mark.parametrize("n_sup,dz", [(6, 0.0), (12, 0.0), (12, 50.0), (40, 0.0)])
def test_affine_reproduction_identities(n_sup, dz):
    """L5.1: 값 가중은 (1, x, y)를, 기울기 가중은 (0, 1, 0)을 재현한다."""
    xs = _support(1, n_sup)
    pts = _points(2, 30)
    W = build_ips_spline(xs, pts, dz)
    Wx = build_ips_spline_slope(xs, pts, dz)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(W @ xs[:, 0], pts[:, 0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(W @ xs[:, 1], pts[:, 1], rtol=0, atol=1e-8)
    np.testing.assert_allclose(Wx.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(Wx @ xs[:, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(Wx @ xs[:, 1], 0.0, atol=1e-10)


def test_transfer_conserves_resultant_and_moments_for_any_support():
    """5-A(i): F = W^T f 는 지지 집합과 무관하게 총력·면내 모멘트를 보존한다."""
    pts = _points(3, 64)
    f = np.random.default_rng(4).normal(size=64)
    ref = np.array([f.sum(), (pts[:, 0] * f).sum(), (pts[:, 1] * f).sum()])
    for seed, n in ((5, 6), (6, 15), (7, 60)):
        xs = _support(seed, n)
        F = build_ips_spline(xs, pts).T @ f
        got = np.array([F.sum(), (xs[:, 0] * F).sum(), (xs[:, 1] * F).sum()])
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9 * abs(ref).max())


def test_root_section_loads_are_support_invariant():
    """5-A(ii): 루트 외측 합력(V, M, T)은 어떤 지지에서도 같다."""
    pts = _points(8, 48)
    f = np.random.default_rng(9).uniform(0.5, 1.5, 48)
    x_ea = 200.0
    exact = (f.sum(), (pts[:, 1] * f).sum(), -((pts[:, 0] - x_ea) * f).sum())
    for seed, n in ((10, 6), (11, 20)):
        xs = _support(seed, n)
        F = build_ips_spline(xs, pts).T @ f
        root = (F.sum(), (xs[:, 1] * F).sum(), -((xs[:, 0] - x_ea) * F).sum())
        np.testing.assert_allclose(root, exact, rtol=1e-9)


def test_interior_station_loads_differ_between_supports():
    """5-A(iii)/5-F: 루트는 같아도 내부 스테이션 전단은 지지에 따라 다르다
    (합력 수준 검사가 지지 부적합에 무정보임을 같은 자료로 보인다)."""
    pts = _points(8, 48)
    f = np.ones(48)
    y_s = 1000.0
    exact = f[pts[:, 1] >= y_s].sum()
    xs_sparse, xs_dense = _support(12, 6), _support(13, 60)
    V = []
    for xs in (xs_sparse, xs_dense):
        F = build_ips_spline(xs, pts).T @ f
        V.append(F[xs[:, 1] >= y_s].sum())
    assert abs(V[0] - exact) > 50 * abs(V[1] - exact)


def test_split_support_virtual_work_identity():
    """5-D(i): 가상일 항등식은 힘 행렬만 포함하므로 워시 지지와 무관하다."""
    pts = _points(14, 40)
    xs_d = _support(15, 30)
    f = np.random.default_rng(16).normal(size=40)
    du = np.random.default_rng(17).normal(size=30)
    W_d = build_ips_spline(xs_d, pts)
    lhs = du @ (W_d.T @ f)
    rhs = (W_d @ du) @ f
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_drill_scale_hook_reaches_batch_assembly(tmp_path, monkeypatch):
    """ASCENT_DRILL_SCALE 은 배치 CQUAD4·CTRIA3 조립의 θz 대각만 스케일한다.

    스케일 100 대 1 에서 (i) 모든 절점의 θz 대각이 정확히 100배이고
    Σ E·t·A·1e-6·scale 과 일치하며(절점 1 은 CQUAD4 전용, 절점 9 는
    CTRIA3 전용), (ii) 병진·굽힘 회전·비대각 성분은 비트 단위로 불변,
    (iii) 배치 경로가 요소 클래스와 일치, (iv) 환경변수 미설정 = 1.0.
    """
    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.elements.quad4 import CQuad4Element
    from ascent_load.elements.tria3 import CTria3Element
    from ascent_load.fem.model import FEModel

    E, T = 70000.0, 2.0
    deck = "\n".join([
        "SOL 101", "CEND", "SPC = 1", "BEGIN BULK",
        f"MAT1,1,{E},,0.33,2.7-9", f"PSHELL,1,1,{T},1",
        "GRID,1,,0.0,0.0,0.0", "GRID,2,,100.0,0.0,0.0",
        "GRID,3,,200.0,0.0,0.0", "GRID,4,,0.0,100.0,0.0",
        "GRID,5,,100.0,100.0,0.0", "GRID,6,,200.0,100.0,0.0",
        "GRID,7,,0.0,200.0,0.0", "GRID,8,,100.0,200.0,0.0",
        "GRID,9,,200.0,200.0,0.0",
        "CQUAD4,1,1,1,2,5,4", "CQUAD4,2,1,2,3,6,5",
        "CQUAD4,3,1,4,5,8,7", "CTRIA3,4,1,5,6,9", "CTRIA3,5,1,5,9,8",
        "SPC1,1,123456,1,2,3", "ENDDATA", ""])
    path = tmp_path / "plate.bdf"
    path.write_text(deck)

    def assemble(scale):
        if scale is None:
            monkeypatch.delenv("ASCENT_DRILL_SCALE", raising=False)
        else:
            monkeypatch.setenv("ASCENT_DRILL_SCALE", scale)
        model = parse_bdf(str(path))
        model.cross_reference()
        fe = FEModel(model)
        return model, fe, fe.K.toarray()

    _, _, K_unset = assemble(None)
    model, fe, K1 = assemble("1.0")
    assert np.array_equal(K_unset, K1)                          # (iv)
    _, _, K100 = assemble("100.0")                              # 이후 훅 = 100

    def area(el):
        xy = np.array([model.nodes[n].xyz_global for n in el.node_ids])[:, :2]
        x, y = xy[:, 0], xy[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    pen = {n: 0.0 for n in model.nodes}
    for el in model.elements.values():
        for n in el.node_ids:
            pen[n] += E * T * area(el) * 1e-6
    rz = np.array([fe.dof_mgr.get_dof(n, 6) for n in model.nodes])
    expect = np.array([pen[n] for n in model.nodes])
    # 평판이라 θz 대각은 드릴링 벌칙 합 그 자체(막·굽힘은 θz 를 쓰지 않음)
    assert np.allclose(K1[rz, rz], expect, rtol=1e-12, atol=0.0)
    assert np.allclose(K100[rz, rz], 100.0 * expect, rtol=1e-12, atol=0.0)   # (i)
    D = K100 - K1
    D[rz, rz] = 0.0
    assert not D.any()                                          # (ii)

    K_cls = np.zeros_like(K100)
    for el in model.elements.values():
        xyz = np.array([model.nodes[n].xyz_global for n in el.node_ids])
        prop = el.property_ref
        mat = prop.material_ref
        cls = CQuad4Element if el.type == "CQUAD4" else CTria3Element
        ke = cls(xyz, mat.E, mat.nu, prop.t, mat.rho).stiffness_matrix()
        d = np.asarray(fe.dof_mgr.get_element_dofs(el.node_ids))
        K_cls[np.ix_(d, d)] += ke
    assert np.abs(K100 - K_cls).max() < 1e-9 * np.abs(K100).max()   # (iii)
