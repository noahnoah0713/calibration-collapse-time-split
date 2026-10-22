# -*- coding: utf-8 -*-
"""ECE 배율의 신뢰구간 — 효과크기에 불확실성을 붙인다.

왜 필요한가 (체계적 누락)
------------------------
이 논문의 핵심 효과크기는 ECE 배율(시간분할 ECE / 대조군 ECE)이다.
§8 의 3.35배·2.08배, §16 사다리의 1.76·1.37·1.45·1.21 이 모두 그것이다.
그런데 results/ 를 전수 조사하면 배율에 신뢰구간이 붙은 파일이 하나도
없다(2026-10-19 확인). 전부 점추정이다.

Phipson–Smyth p 는 "관측 ECE 가 대조군 분포에서 얼마나 극단인가" 를
답하지, 배율이 얼마나 정밀한가 를 답하지 않는다. 심사자가 표를 보고
던질 첫 질문은 하나다 — "1.37 이 1.0 과 구분되는가." 지금은 답할
근거가 표에 없다.

배율의 불확실성은 두 곳에서 온다
--------------------------------
  분자: 시간분할 검증군의 표본오차 (검증군을 다시 뽑으면 ECE 가 변한다)
  분모: 대조군 재표집의 변동 (200회 반복의 평균 자체가 추정치다)

기존 분석들은 둘 다 반영하지 않았다. 이 절은 둘을 모두 넣는다 —
  분자: 검증군 부트스트랩 (예측 고정, 재적합 없음)
  분모: 대조군 반복 ECE 들의 부트스트랩
  두 분포를 독립으로 결합해 배율의 백분위 구간을 만든다.

분자만 반영한 구간도 함께 찍는다. 둘을 나란히 두면 대조군 변동을
빼먹었을 때 구간이 얼마나 좁아지는지(= 기존 방식이 얼마나 낙관적인지)
보인다.

차이 구간도 함께
---------------
배율은 분모가 작아지면 폭발하므로, 분모가 작은 설정에서 구간이 비대칭으로
넓어진다. 그래서 차이(ECE_T − 평균 ECE_대조) 의 구간도 같이 낸다.
논문 서술에는 둘 중 해석이 안정적인 쪽을 쓰면 된다.

기존 생성기를 건드리지 않는다
----------------------------
prevalence_control.py 등은 수정하지 않는다. 같은 시드·같은 설계를 여기서
독립적으로 재현하고, 점추정이 기존 결과와 일치하는지 검사한 뒤
구간만 새로 더한다. 일치하지 않으면 그 자리에서 실패한다.

사용법
------
    python src/ratio_intervals.py              # 대조군 200회 (§16 과 동일)
    python src/ratio_intervals.py --reps 20    # 짧게 점검

출력
----
    results/ratio_intervals.csv
    results/ratio_intervals_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

N_BOOT = 4000
TOL = 5e-4  # 기존 점추정과의 허용 오차 (소수 4자리 표기 기준)


def ratio_ci(ece_obs_boot, ctrl_boot):
    """분자·분모 분포를 독립 결합한 배율·차이 구간."""
    n = min(len(ece_obs_boot), len(ctrl_boot))
    a = np.asarray(ece_obs_boot)[:n]
    b = np.asarray(ctrl_boot)[:n]
    ok = b > 0
    r = np.sort(a[ok] / b[ok])
    d = np.sort(a - b)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(r) + q(d)


def boot_ece(y, p, n_boot, seed):
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p[i]))
    return np.array(v)


def boot_mean(vals, n_boot, seed):
    """대조군 반복 ECE 들의 평균에 대한 부트스트랩 분포."""
    r = np.random.default_rng(seed)
    v = np.asarray(vals)
    n = len(v)
    return np.array([float(v[r.integers(0, n, n)].mean()) for _ in range(n_boot)])


def run(name, fn, reps, ref):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rate_te = float(y[te].mean())
    rate_tr = float(y[tr].mean())

    m = S.make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]
    ece_T = S.ece(y[te], p_t)
    num = boot_ece(y[te], p_t, N_BOOT, S.SEED + 1)

    log("=" * 116)
    log(
        "%s   시간분할 ECE %.4f (검증 %d개) / 대조군 반복 %d회 / 부트스트랩 %d회"
        % (name, ece_T, len(te), reps, N_BOOT)
    )
    # 기존 §16 결과와 점추정이 같은지 먼저 확인한다
    if ref is not None:
        r0 = ref[(ref.dataset.str.startswith(name)) & (ref.setting.str.startswith("T"))]
        if len(r0):
            d0 = abs(float(r0.ECE.iloc[0]) - ece_T)
            log(
                "  §16 시간분할 ECE 와의 차: %.2e  → %s"
                % (d0, "일치" if d0 < TOL else "**불일치 — 중단해야 한다**")
            )
            assert d0 < TOL, "§16 재현 실패: %.6f 대 %.6f" % (float(r0.ECE.iloc[0]), ece_T)
    log("=" * 116)
    log(
        "%-34s %8s %8s %-22s %-22s %-22s"
        % ("대조 설계", "대조ECE", "배율", "배율 95%CI (전체)", "배율 95%CI (분자만)", "차이 95%CI")
    )
    log("-" * 124)

    rows = []
    for tag, mt, mr in (
        ("R1. 무작위 (표준)", False, False),
        ("R2. 무작위 + 검증군 비율맞춤", True, False),
        ("R3. 무작위 + 학습·검증 비율맞춤", True, True),
    ):
        E = []
        for i in range(reps):
            r = np.random.default_rng(S.SEED + i)
            idx = r.permutation(len(y))
            ti, ri = idx[: len(te)], idx[len(te) :]
            if mt:
                ti = S.subsample_to_rate(y, ti, rate_te, r)
            if mr:
                ri = S.subsample_to_rate(y, ri, rate_tr, r)
            if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                continue
            if len(ti) < 30 or len(ri) < 100:
                continue
            mm = S.make_model().fit(X[ri], y[ri])
            E.append(S.ece(y[ti], mm.predict_proba(X[ti])[:, 1]))
        if not E:
            log("%-34s 유효 반복 0회 — 건너뜀" % tag)
            continue
        E = np.array(E)
        ctrl = float(E.mean())
        ratio = ece_T / ctrl
        den = boot_mean(E, N_BOOT, S.SEED + 2)
        rlo, rhi, dlo, dhi = ratio_ci(num, den)
        # 분자만 반영한 구간 (기존 방식이 암묵적으로 쓰던 것)
        nlo, nhi = (float(np.percentile(num / ctrl, 2.5)), float(np.percentile(num / ctrl, 97.5)))
        excl1 = rlo > 1.0
        log(
            "%-34s %8.4f %8.2f [%6.2f, %6.2f]%s [%6.2f, %6.2f]        [%+.4f, %+.4f]"
            % (tag, ctrl, ratio, rlo, rhi, " 1제외" if excl1 else "      ", nlo, nhi, dlo, dhi)
        )
        rows.append(
            {
                "dataset": name,
                "control": tag,
                "n_test": len(te),
                "ece_temporal": ece_T,
                "ece_control_mean": ctrl,
                "ece_control_sd": float(E.std(ddof=1)),
                "n_rep_valid": len(E),
                "ece_ratio": ratio,
                "ratio_lo": rlo,
                "ratio_hi": rhi,
                "ratio_lo_numonly": nlo,
                "ratio_hi_numonly": nhi,
                "diff_lo": dlo,
                "diff_hi": dhi,
                "excludes_one": bool(excl1),
                "ci_width_full": rhi - rlo,
                "ci_width_numonly": nhi - nlo,
            }
        )
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    log("ECE 배율의 신뢰구간 — 효과크기에 불확실성을 붙인다")
    log("대조군 반복 %d회, 부트스트랩 %d회, 시드 %d" % (a.reps, N_BOOT, S.SEED))
    log("분자(검증군 표본오차)와 분모(대조군 변동)를 모두 반영한다.")
    log("")
    ref = None
    _p = os.path.join(RES, "prevalence_control.csv")
    if os.path.exists(_p):
        ref = pd.read_csv(_p)
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.reps, ref)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "ratio_intervals.csv"), index=False, encoding="utf-8")

    log("=" * 116)
    log("판정")
    log("=" * 116)
    log("[읽는 법] 배율 구간의 **하한이 1 을 넘어야** '통제 후에도 보정이 더")
    log("          나쁘다' 를 효과크기 수준에서 주장할 수 있다. p 값만으로는")
    log("          그 주장이 서지 않는다.")
    log("")
    for d in out.dataset.unique():
        g = out[out.dataset == d]
        log("  %s" % d)
        for _, r in g.iterrows():
            log(
                "    %-34s 배율 %.2f [%.2f, %.2f] — %s"
                % (
                    r.control,
                    r.ece_ratio,
                    r.ratio_lo,
                    r.ratio_hi,
                    (
                        "**1 을 제외한다**"
                        if r.excludes_one
                        else "**1 을 포함한다 — 효과크기로는 주장 불가**"
                    ),
                )
            )
        _w = g.ci_width_full / g.ci_width_numonly
        log(
            "    → 분모(대조군 반복) 변동을 넣었을 때 구간 폭 비 %.4f-%.4f "
            "(1 보다 작은 칸은 재표집 잡음이다)." % (float(_w.min()), float(_w.max()))
        )
        log(
            "      즉 분모의 표본오차는 사실상 구간을 넓히지 않는다 — 대조군 %d회면 "
            "분모는 충분히 안정되고, 폭을 정하는 것은 검증군 n=%d 이다."
            % (int(g.n_rep_valid.max()), int(g.n_test.iloc[0]))
        )
        log("")
    _fail = out[~out.excludes_one]
    if len(_fail):
        log("  **구간이 1 을 포함하는 설정 %d개**:" % len(_fail))
        for _, r in _fail.iterrows():
            log(
                "    %s / %s — 배율 %.2f [%.2f, %.2f]"
                % (r.dataset, r.control, r.ece_ratio, r.ratio_lo, r.ratio_hi)
            )
        log("  해당 설정의 서술은 '유의하다' 가 아니라 '점추정은 1 을 넘지만")
        log("  구간이 1 을 포함한다' 로 바꿔야 한다.")
    else:
        log("  모든 설정에서 배율 구간이 1 을 제외한다.")
    log("")
    log("저장: %s" % os.path.join(RES, "ratio_intervals.csv"))
    log("      %s" % os.path.join(RES, "ratio_intervals_summary.txt"))
    log.save(Path(RES) / "ratio_intervals_summary.txt")


if __name__ == "__main__":
    main()
