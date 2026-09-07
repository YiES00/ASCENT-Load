"""Base element interface."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
import numpy as np


def drill_scale() -> float:
    """드릴링 정칙화 스케일 훅 (연구용, 기본 1.0).

    ASCENT_DRILL_SCALE 환경변수로 alpha_drill = E*t*A*1e-6 항을
    일괄 스케일한다. 발산 스펙트럼의 정칙화 의존성 연구(T1-E1)
    전용이며 생산 해석에서는 건드리지 않는다.
    """
    try:
        return float(os.environ.get("ASCENT_DRILL_SCALE", "1.0"))
    except ValueError:
        return 1.0


def membrane_mode() -> str:
    """CQUAD4 막 정식화 선택 훅. 'qm6'(기본) 또는 'q4'.

    qm6: Taylor-Beresford-Wilson 비적합 모드(중심 야코비안 보정) —
    쌍선형 Q4 의 면내 굽힘 기생전단 잠금을 제거한다(종횡비 4 에서
    0.39 -> 0.98). q4: 2026-09-04 이전 archived 결과 재현용.
    """
    v = os.environ.get("ASCENT_QUAD4_MEMBRANE", "qm6").strip().lower()
    return "q4" if v == "q4" else "qm6"


def bending_mode() -> str:
    """CQUAD4 굽힘 정식화 선택 훅. 'dkq'(기본) 또는 'mindlin'.

    dkq: Batoz-Tahar 이산 Kirchhoff 사각형 — 횡전단 항이 없어 얇은
    외피가 보와 절점을 공유하는 조립체를 과잉 구속하지 않는다(MSC
    hold-out 대조에서 Mindlin 전단 벌칙이 전기체 강성을 25% 올렸다).
    PSHELL MID3 공란 = 전단강체라는 MSC 의미론과 같다.
    mindlin: 2026-09-04 이전 archived 결과 재현용 (SRI 1점 전단).
    """
    v = os.environ.get("ASCENT_QUAD4_BENDING", "dkq").strip().lower()
    return "mindlin" if v == "mindlin" else "dkq"


class BaseElement(ABC):
    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray: ...
    @abstractmethod
    def mass_matrix(self) -> np.ndarray: ...
    @abstractmethod
    def dof_count(self) -> int: ...
