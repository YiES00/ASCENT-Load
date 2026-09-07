# GACOMP 정적 발산 MSC 대조 덱 세 장을 생성한다 (r5 MC2 잔여 / r4 MC1 필수수정 9)
"""Write the MSC Nastran divergence cross-check decks for GACOMP.

The comparison model itself lives in a git-ignored directory
(``.gitignore`` excludes ``solver/tests/validation/GACOMP/``), so the
decks cannot be versioned. This generator is versioned instead: run it
and the three decks reappear beside the master trim deck, byte for byte.

Why these decks exist
---------------------
ASCENT-Load computes divergence from the eigenvalues of
``C = G_w K^-1 G_d^T A(q=1)``, so it needs ``K^-1`` and therefore a
regularization ``eps`` on the zero-stiffness DOFs. Peer review r4
(Major Comment 1) showed the surface-construction result is sensitive to
that coefficient; r5 kept the four-axis sensitivity numbers flagged as
"legacy 1e-8 operator" and deferred re-quantification to this MSC
cross-check. MSC's ``DIVERG`` solves the generalized problem
``[K_ll - lambda Q_ll]{u_l} = 0`` directly and never inverts ``K``, so it
sidesteps the regularization question entirely.

Three decks
-----------
v0_smoke    one Mach (0.50), 3 roots -- confirms the DIVERG/CMETHOD setup
            runs on this model before committing to the full sweep.
v1_matched  restraint identical to the paper's divergence operator: the
            deck's own 3-point reference mount (7 DOF) plus the virtual
            3-2-1 mount the study adds (6 DOF) = 13 DOF.
v2_native   the deck's own 3-point reference mount only (7 DOF). Control
            group -- isolates whether v1's double restraint distorts q_div.

The virtual-mount DOFs are not guessed. They were dumped from
``ascent_load.solvers.sol144._select_virtual_mount`` on this model, which
is what the paper's ``docs/dissertation/scripts/r4_divergence_final.py``
calls (``_suport_mount_idx`` returns None -- the deck has no SUPORT).

Card layouts are taken from the MSC Nastran Aeroelastic Analysis User's
Guide, "Divergence Analysis" (Listing 3-1): a ``DIVERG`` Case Control
command in a subcase, paired with ``CMETHOD``/``EIGC`` because the
divergence eigenanalysis uses a complex eigensolver.

Usage:  cd solver && python scripts/make_gacomp_diverg_decks.py
Output: tests/validation/GACOMP/gacomp_msc_diverg_v{0,1,2}_*.bdf
Run instructions: docs/peer-review-r5/msc-run-package.md
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.normpath(
    os.path.join(HERE, "..", "tests", "validation", "GACOMP"))

MACH = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70"]
Q = ["7.0927-4", "2.8371-3", "6.3835-3", "1.1348-2",
     "1.7732-2", "2.5534-2", "3.4754-2"]

INCLUDES = [
    "INCLUDE './BULK/p400r3-aero-asym_flex.dat'",
    "INCLUDE './BULK/p400r3-struct-asym_flex.dat'",
    "INCLUDE './BULK/p400r3-spline-asym_flex.dat'",
    "INCLUDE './DATA/101006_MASS_DB/mft400r5-L01_wo_LRU.mass'",
    "INCLUDE './BULK/balance_101012.mass'",
    "INCLUDE './BULK/system_101012.mass'",
    "INCLUDE './BULK/cabin_empty_100714.mass'",
]

# 원본 덱 SPC set 2 의 실제 내용. SPCADD 2 는 미정의 집합 101106 도
# 참조하므로 새 집합을 명시적으로 정의해 그 문제를 우회한다.
NATIVE = [("123", "112020"), ("23", "211019"), ("23", "211067")]
# 논문 발산 스크립트가 추가로 넣는 가상 3-2-1 마운트
MOUNT = [("123", "112030"), ("23", "290596"), ("3", "910033")]


# DISP = ALL 은 고유벡터를 전 절점에 인쇄한다. 절점 22,640 x 복소 2행 x
# NROOT x 마하 7이므로 v1/v2 에서 F06 이 100 MB 급으로 커진다. 아래
# 표본 SET 은 구성품별 표본 + 임계 패치(369008~369018) + 구속 절점을
# 담은 59절점이며, 덱 안에 주석으로 함께 넣어 한 줄 교체로 쓸 수 있게
# 한다. 지역 표본은 struct INCLUDE 의 GRID 를 구간별로 균등 추출한 값이다.
SAMPLE_NODES = [
    111003, 112020, 112030, 160205, 185073, 186189, 211019, 211067,
    251779, 270200, 290596, 311101, 323143, 331373, 341377, 356007,
    367148, 369008, 369009, 369010, 369011, 369012, 369013, 369014,
    369015, 369016, 369017, 369018, 411101, 423159, 431420, 441444,
    455227, 467109, 510001, 530161, 540157, 553106, 554089, 554345,
    610001, 630194, 640218, 653193, 654328, 655109, 710001, 730171,
    741011, 741372, 753121, 754072, 800001, 800464, 800926, 801388,
    801850, 802312, 910033,
]


def sample_set_comment(sid: int = 100) -> list[str]:
    """표본 SET 카드를 '$' 주석 줄들로 만든다 (68열에서 접는다)."""
    out, cur = [], f"SET {sid} = "
    for i, v in enumerate(SAMPLE_NODES):
        t = str(v) + (", " if i < len(SAMPLE_NODES) - 1 else "")
        if len(cur) + len(t) > 68:
            out.append(cur.rstrip())
            cur = "        " + t
        else:
            cur += t
    out.append(cur.rstrip())
    return (["$ F06 을 줄이려면 DISP = ALL 을 아래 두 줄로 바꾼다. 발산 동압"
             " 표는",
             "$ DISP 와 무관하게 NROOT 만큼 인쇄되므로 요약표는 그대로 "
             "남는다.",
             "$ 표본 59절점 = 구성품별 균등 표본 + 임계 패치 369008~369018 "
             "+ 구속 절점"]
            + ["$   " + o for o in out]
            + ["$   DISP     = 100"])


def card(*fields, cont: str | None = None) -> str:
    """8열 고정형식 한 줄. fields[0]=카드명, 이후 필드 2..9, cont=필드 10."""
    line = ""
    for i, v in enumerate(fields):
        s = "" if v is None else str(v)
        if len(s) > 8:
            raise ValueError(f"field {i + 1} too wide: {s!r}")
        line += s.ljust(8)
    if cont is not None:
        line = line.ljust(72) + cont.ljust(8)
    return line.rstrip()


HEAD = """\
$-----------------------------------------------------------------------$
$ GACOMP 정적 발산 MSC 대조 덱 {ver} -- {what}
$-----------------------------------------------------------------------$
$ 목적: 논문 1 검토 잔여 항목 -- ASCENT-Load 가 계산한 발산 동압을
$       상용 솔버의 독립 정식화로 대조한다. ASCENT-Load 는
$       C = G_w K^-1 G_d^T A(q=1) 의 고유치로 1/q_div 를 구하므로 K 의
$       역행렬이 필요하고, 그래서 영강성 자유도에 정칙화 eps 를 넣는다.
$       MSC 의 DIVERG 는 [K_ll - lambda Q_ll]{{u_l}} = 0 의 일반화 고유치를
$       직접 풀어 K 를 뒤집지 않는다. 따라서 이 실행은 정칙화 의존성
$       문제 자체를 우회하는 독립 검증이다.
$
$ 계보: p400r3-free-trim.bdf (SOL 144, TRIM 1~7) 에서 트림을 떼고
$       발산 고유치 해석으로 바꾼 것. 구조·공력·스플라인·질량 INCLUDE 는
$       원본을 그대로 쓴다. 실행은 이 파일이 놓인 GACOMP 디렉터리에서
$       하라 -- INCLUDE 가 './BULK/...' 상대경로이므로 원본 마스터 덱과
$       동일하게 해석된다.
$ 생성: solver/scripts/make_gacomp_diverg_decks.py (직접 수정하지 말 것)
$ 실행 안내: docs/peer-review-r5/msc-run-package.md
$
$ W2GJ(DMI) 는 넣지 않았다. 캠버·비틀림 초기 하강류는 정적 하중이고
$ 발산 고유치 문제 [K - lambda Q]u = 0 에는 들어가지 않는다.
$ AESTAT/TRIM 도 넣지 않았다. MSC 지침의 발산 예제(Aeroelastic Analysis
$ User's Guide, Listing 3-1)는 TRIM 없이 DIVERG/CMETHOD 만으로 구성된다.
$ 만약 MSC 가 AESTAT 부재를 문제 삼으면 원본 마스터 덱의 AESTAT 여섯
$ 장을 그대로 붙이면 된다.
$
$ 구속: {restraint}
$
$ 비행 동압 (N/mm^2, 여유 = q_div/q_flight 를 만들 때 쓸 값)
{qtable}$
$ ASCENT-Load 기준값 (생산 연산자, eps = mean(양의 대각)*1e-12
$ = 9.662383e-05, 자유 자유도 132,316)
$   M     q_div(회전)   q_div(표면)   여유(표면)   표면/회전
$   0.1   3.9213e-06    5.5652e-04    0.7846       141.9
$   0.4   3.8727e-06    5.1387e-04    0.0453       132.7
$   0.7   3.7524e-06    4.0475e-04    0.0116       107.9
$   회전 구성은 정칙화 계수를 열 자릿수 바꿔도 세 자리까지 불변이고,
$   표면 구성은 1e-12 에서 수렴한다.
$
$ 출력에서 볼 것: F06 의 'D I V E R G E N C E   S U M M A R Y' 블록.
$   MACH NUMBER 별로 ROOT NO. / DIVERGENCE DYNAMIC PRESSURE / EIGENVALUE
$   (REAL, IMAGINARY) 가 인쇄된다. 양수 q_div 만 물리적 의미가 있고
$   그중 최소값이 임계 발산 동압이다. DISP 출력의 해당 모드 형상으로
$   임계 모드의 정체를 판정한다. ASCENT-Load 가 수렴 연산자에서 실측한
$   임계 모드는 날개 연질 패치(절점 369008~369018, 참여도 날개 99.99%
$   / T3 94.8%, 분리도 4.3~4.6, 고유쌍 잔차 1e-12)이고, 회전 구성의
$   모드는 이와 별개로 동체 theta_z 세 자유도다. 구판 정칙화(1e-8)에서
$   보였던 수평미익 전역 모드는 폐기된 서술이므로 대조 기준으로 쓰지
$   말 것.
$-----------------------------------------------------------------------$"""


def build(fname, ver, what, title, spc_sid, rows, machs, nroot, ndroot,
          restraint):
    qtable = "".join(f"$   M {m}   q = {q}\n" for m, q in zip(MACH, Q))
    L = [HEAD.format(ver=ver, what=what, restraint=restraint, qtable=qtable)]
    L += ["SOL     144", "CEND",
          f"TITLE    = GACOMP DIVERGENCE {title}",
          "SUBTITLE = STATIC AEROELASTIC DIVERGENCE - MSC CROSS-CHECK",
          "ECHO     = NONE",
          "MPC      = 999",
          f"SPC      = {spc_sid}",
          "DISP     = ALL"]
    if len(machs) > 1:
        L += sample_set_comment()
    L += ["SPCF     = ALL",
          "AEROF    = ALL",
          "APRES    = ALL",
          "SUBCASE 1",
          f"  SUBTITLE = DIVERGENCE M{machs[0]}"
          + (f" THRU M{machs[-1]}" if len(machs) > 1 else ""),
          "  DIVERG   = 900",
          "  CMETHOD  = 900",
          "BEGIN BULK"]
    L += INCLUDES
    L.append("$")
    L += [card("PARAM", "POST", "0"),
          card("PARAM", "OPPHIPA", "1"),
          card("PARAM", "WTMASS", "1.00"),
          card("PARAM", "GRDPNT", "0"),
          card("PARAM", "BAILOUT", "-1")]
    L.append("$")
    L.append("$ DIVERG  SID     NROOT   M1      M2      M3      M4      M5"
             "      M6")
    if len(machs) <= 6:
        L.append(card("DIVERG", "900", str(nroot), *machs))
    else:
        L.append(card("DIVERG", "900", str(nroot), *machs[:6], cont="+DIV"))
        L.append(card("+DIV", *machs[6:]))
    L.append("$ EIGC 는 CMETHOD 가 고르는 복소 고유해석 제어이다. 발산 "
             "고유해석은")
    L.append("$ 복소 Lanczos(CLAN)를 쓴다. ND0(필드 8)는 뽑을 복소근 수이다.")
    L.append("$ EIGC    SID     METHOD  NORM    G       C       E       ND0")
    L.append(card("EIGC", "900", "CLAN", "MAX", None, None, None, str(ndroot)))
    L.append("$")
    L.append(f"$ 구속 자유도 {sum(len(d) for d, _ in rows)} 개")
    for dof, nid in rows:
        L.append(card("SPC1", str(spc_sid), dof, nid))
    L.append("ENDDATA")
    path = os.path.join(OUTDIR, fname)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def main():
    if not os.path.isdir(OUTDIR):
        sys.exit(f"비교 모델 디렉터리가 없다: {OUTDIR}")
    made = [
        build("gacomp_msc_diverg_v0_smoke.bdf", "v0",
              "설정 확인용 단일 마하 시험", "V0 SMOKE M0.50",
              902, NATIVE, ["0.50"], 3, 6,
              "원본 덱 3점 기준 마운트만 (7 자유도). 설정이 도는지 먼저 "
              "확인하는 용도"),
        build("gacomp_msc_diverg_v1_matched.bdf", "v1",
              "논문 발산 연산자와 동일한 구속", "V1 MATCHED RESTRAINT",
              901, NATIVE + MOUNT, MACH, 5, 10,
              "원본 덱 3점 기준 마운트(7 자유도) + 논문 스크립트의 가상 "
              "3-2-1\n$       마운트(6 자유도) = 13 자유도. ASCENT-Load "
              "발산 연산자와 동일하다"),
        build("gacomp_msc_diverg_v2_native.bdf", "v2", "덱 고유 구속만",
              "V2 NATIVE RESTRAINT", 902, NATIVE, MACH, 5, 10,
              "원본 덱 3점 기준 마운트만 (7 자유도). v1 의 13 자유도 이중 "
              "구속이\n$       q_div 를 왜곡하는지 가리는 대조군"),
    ]
    for p in made:
        print(f"작성: {p}")
    missing = [i.split("'")[1] for i in INCLUDES
               if not os.path.isfile(os.path.join(OUTDIR, i.split("'")[1]))]
    if missing:
        print("경고 -- 해석되지 않는 INCLUDE:")
        for m in missing:
            print(f"  {m}")
    else:
        print(f"INCLUDE {len(INCLUDES)}개 모두 해석됨")


if __name__ == "__main__":
    main()
