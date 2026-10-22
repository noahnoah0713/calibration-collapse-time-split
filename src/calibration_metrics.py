# -*- coding: utf-8 -*-
"""구간화에 의존하지 않는 보정 지표 — ECE 한 지표에 걸린 위험을 없앤다.

왜 필요한가 (방법론적 누락)
--------------------------
이 논문의 헤드라인은 전부 **ECE**(균등폭 10구간)로 쓰여 있다. 보정 문헌을
아는 심사자가 던지는 표준 질문이 둘 있고, 지금 저장소에는 답이 없다.

  (1) **ECE 는 구간 수에 의존한다.** §9.4 에서 균등폭·균등질량 × 5·10·15·20
      구간을 다 보여 주지만, 그것은 "구간을 바꿔도 결론이 같다" 는 말이고
      "구간을 아예 쓰지 않아도 같다" 는 말은 아니다.

  (2) **ECE 는 위로 편향된 추정량이다.** 완벽히 보정된 모델이라도 유한
      표본에서는 각 구간의 관측 비율이 예측 평균과 우연히 어긋나므로 ECE 가
      0 보다 크게 나온다. 검증군이 156·267개로 작을 때 이 편향은 작지 않다.
      **그래서 "시간분할 ECE 0.4384" 중 얼마가 편향인지 밝혀야 한다.**

이 스크립트가 더하는 것
----------------------
  ICI / E50 / E90 / Emax  구간을 쓰지 않는다. 검증군에서 loess 보정곡선
                          f(p) 를 적합해 |p - f(p)| 의 평균·중앙값·90백분위·
                          최대값을 쓴다 (Austin & Steyerberg 2019).
  Spiegelhalter z         보정에 대한 정식 가설검정. z = Σ(y-p)(1-2p) /
                          sqrt(Σ(1-2p)^2 p(1-p)). 구간도 곡선도 쓰지 않는다.
  편향 제거 ECE           예측을 고정하고 y* ~ Bernoulli(p) 를 반복 생성한다.
                          이렇게 만든 표본은 **정의상 완벽히 보정**되어 있으므로
                          그 ECE 분포가 곧 '보정이 완벽할 때의 ECE' 다.
                          관측 ECE 에서 그 평균을 빼면 편향 제거값이 되고,
                          관측값이 그 분포의 97.5 백분위를 넘는지도 본다.
  대형 보정오차           mean(p) - mean(y). 절편 이동만 보는 1차 지표.

그리고 **같은 네 지표를 §8 의 대조군(R1·R2·R3)에서도 계산해 배율을 다시
낸다.** 핵심 주장(통제 후에도 시간분할이 더 나쁘다)이 ECE 라는 특정
지표의 산물인지 아닌지가 이 표로 결정된다.

기존 결과를 건드리지 않는다
--------------------------
prevalence_control.py 와 **같은 시드·같은 대조 설계**를 독립적으로 재현하고,
ECE 점추정이 기존 결과와 일치하는지 먼저 검사한 뒤(TOL) 새 지표만 더한다.

사용법
------
    python src/calibration_metrics.py                 # 대조군 200회 (§8 과 동일)
    python src/calibration_metrics.py --reps 20       # 짧게 점검
    python src/calibration_metrics.py --nsim 2000     # 편향 귀무표본 수

출력
----
    results/calibration_metrics.csv
    results/calibration_metrics_ratio.csv
    results/calibration_metrics_summary.txt
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

TOL = 5e-4  # 기존 ECE 점추정과의 허용 오차 (소수 4자리 표기 기준)
LOESS_FRAC = 0.75  # Austin & Steyerberg 가 쓴 기본값


# ---------------------------------------------------------------- 지표 정의
def loess_curve(y, p):
    """검증군에서 적합한 loess 보정곡선 f(p). 구간을 쓰지 않는다."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    f = lowess(endog=y, exog=p, frac=LOESS_FRAC, it=0, return_sorted=False)
    return np.clip(np.asarray(f, dtype=float), 0.0, 1.0)


def ici_family(y, p):
    """ICI·E50·E90·Emax. 전부 |p - f(p)| 의 요약통계다."""
    d = np.abs(loess_curve(y, p) - np.asarray(p, dtype=float))
    return (float(d.mean()), float(np.median(d)), float(np.percentile(d, 90)), float(d.max()))


def spiegelhalter(y, p):
    """보정에 대한 정식 검정. 귀무가설 = '예측확률이 참값이다'.

    Brier 점수를 그 귀무가설 아래의 기대값·분산으로 표준화한 통계량이다.
    구간화도 곡선 적합도 없으므로 임의 선택이 들어가지 않는다.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
    num = float(np.sum((y - p) * (1 - 2 * p)))
    den = float(np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p))))
    if den == 0:
        return float("nan"), float("nan")
    z = num / den
    # 양측 p — 정규근사. 표본이 작으면 근사이므로 z 와 함께 읽어야 한다.
    from math import erfc, sqrt

    return float(z), float(erfc(abs(z) / sqrt(2.0)))


def ece_null(p, nsim, seed):
    """예측을 고정하고 y* ~ Bernoulli(p) 를 뽑아 만든 ECE 의 귀무분포.

    이렇게 만든 표본은 정의상 완벽히 보정돼 있다. 그래서 이 분포는
    '보정이 완벽할 때 유한표본이 만들어내는 ECE' 이고, 곧 ECE 의 편향이다.
    """
    r = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    v = np.empty(nsim, dtype=float)
    for i in range(nsim):
        v[i] = S.ece((r.random(len(p)) < p).astype(int), p)
    return v


def all_metrics(y, p, nsim, seed, with_auroc=True):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    e = S.ece(y, p)
    ici, e50, e90, emax = ici_family(y, p)
    z, zp = spiegelhalter(y, p)
    nul = ece_null(p, nsim, seed)
    out = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "ece": e,
        "ece_null_mean": float(nul.mean()),
        "ece_null_p975": float(np.percentile(nul, 97.5)),
        "ece_debiased": float(e - nul.mean()),
        "ece_exceeds_null": bool(e > np.percentile(nul, 97.5)),
        "ici": ici,
        "e50": e50,
        "e90": e90,
        "emax": emax,
        "sh_z": z,
        "sh_p": zp,
        "cal_large": float(p.mean() - y.mean()),
        "slope": S.calib_slope(y, p),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if with_auroc else float("nan")
    return out


# ---------------------------------------------------------------- 실행
def run(name, fn, reps, nsim, ref):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rate_te, rate_tr = float(y[te].mean()), float(y[tr].mean())

    m = S.make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]

    log("=" * 118)
    log(
        "%s   학습 %d / 검증 %d   대조군 %d회 · 편향 귀무표본 %d회"
        % (name, len(tr), len(te), reps, nsim)
    )
    log("=" * 118)

    T = all_metrics(y[te], p_t, nsim, S.SEED + 1)
    if ref is not None:
        r0 = ref[(ref.dataset.str.startswith(name)) & (ref.setting.str.startswith("T"))]
        if len(r0):
            d0 = abs(float(r0.ECE.iloc[0]) - T["ece"])
            log(
                "  §8 시간분할 ECE 와의 차 %.2e → %s"
                % (d0, "일치" if d0 < TOL else "**불일치 — 중단**")
            )
            assert d0 < TOL, "§8 재현 실패: %.6f 대 %.6f" % (float(r0.ECE.iloc[0]), T["ece"])

    rows = [
        dict(T, dataset=name, arm="T. 시간분할", n_rep=1, sh_p_frac_sig=float(T["sh_p"] < 0.05))
    ]

    for tag, mt, mr in (
        ("R1. 무작위 (표준)", False, False),
        ("R2. 무작위 + 검증군 비율맞춤", True, False),
        ("R3. 무작위 + 학습·검증 비율맞춤", True, True),
    ):
        acc = []
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
            pp = mm.predict_proba(X[ti])[:, 1]
            # 대조군 반복마다 귀무표본을 전부 돌리면 비싸다. 편향은 n 과
            # 예측분포가 정하므로 반복당 귀무표본을 줄이고 반복으로 평균낸다.
            acc.append(all_metrics(y[ti], pp, max(20, nsim // 20), S.SEED + 10000 + i))
        if not acc:
            log("%-34s 유효 반복 0회 — 건너뜀" % tag)
            continue
        A = pd.DataFrame(acc)
        mean = {c: float(A[c].mean()) for c in A.columns if A[c].dtype.kind in "fi"}
        mean["ece_exceeds_null"] = float(A.ece_exceeds_null.mean())
        # p 값의 평균은 뜻이 없다. 대조군 팔에서는 유의한 반복의 비율로 쓴다.
        mean["sh_p"] = float("nan")
        mean["sh_p_frac_sig"] = float((A.sh_p < 0.05).mean())
        rows.append(dict(mean, dataset=name, arm=tag, n_rep=len(A)))

    out = pd.DataFrame(rows)
    log("")
    log(
        "%-34s %7s %9s %9s %8s %8s %8s %8s %8s"
        % ("분할", "n", "ECE", "편향제거", "ICI", "E50", "E90", "SH z", "AUROC")
    )
    log("-" * 118)
    for _, r in out.iterrows():
        log(
            "%-34s %7.0f %9.4f %9.4f %8.4f %8.4f %8.4f %8.2f %8.4f"
            % (r.arm, r.n, r.ece, r.ece_debiased, r.ici, r.e50, r.e90, r.sh_z, r.auroc)
        )
    log("")
    log(
        "  시간분할의 ECE 편향(완벽 보정 가정 하 기대 ECE) = %.4f "
        "→ 관측 %.4f 의 %.1f%%"
        % (T["ece_null_mean"], T["ece"], 100.0 * T["ece_null_mean"] / max(T["ece"], S.EPS))
    )
    log(
        "  관측 ECE 가 귀무 97.5 백분위(%.4f)를 넘는가: %s"
        % (T["ece_null_p975"], "예" if T["ece_exceeds_null"] else "아니오")
    )
    log(
        "  Spiegelhalter z = %.2f (양측 p = %.3g) — %s"
        % (T["sh_z"], T["sh_p"], "보정 가설 기각" if T["sh_p"] < 0.05 else "기각하지 못함")
    )
    log("")

    # ---- 배율을 네 지표로 다시 ----
    rat = []
    base = out[out.arm.str.startswith("T")].iloc[0]
    for _, r in out[~out.arm.str.startswith("T")].iterrows():
        row = {"dataset": name, "control": r.arm, "n_rep": int(r.n_rep)}
        for k in ("ece", "ece_debiased", "ici", "e90"):
            row["%s_temporal" % k] = float(base[k])
            row["%s_control" % k] = float(r[k])
            row["%s_ratio" % k] = float(base[k]) / float(r[k]) if float(r[k]) > 0 else float("nan")
        rat.append(row)
    R = pd.DataFrame(rat)
    log(
        "%-34s %9s %9s %9s %9s" % ("대조 설계", "ECE 배율", "편향제거 배율", "ICI 배율", "E90 배율")
    )
    log("-" * 118)
    for _, r in R.iterrows():
        log(
            "%-34s %9.2f %9.2f %9.2f %9.2f"
            % (r.control, r.ece_ratio, r.ece_debiased_ratio, r.ici_ratio, r.e90_ratio)
        )
    log("")
    return rows, rat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--nsim", type=int, default=1000)
    a = ap.parse_args()

    log("구간화에 의존하지 않는 보정 지표 — ECE 한 지표에 걸린 위험을 없앤다")
    log("ICI·E50·E90·Emax(loess, frac=%.2f) / Spiegelhalter z / 편향 제거 ECE" % LOESS_FRAC)
    log("대조군 %d회, 편향 귀무표본 %d회, 시드 %d" % (a.reps, a.nsim, S.SEED))
    log("")

    ref = None
    _p = os.path.join(RES, "prevalence_control.csv")
    if os.path.exists(_p):
        ref = pd.read_csv(_p)

    rows, rat = [], []
    for name, fn in S.DATASETS:
        x, z = run(name, fn, a.reps, a.nsim, ref)
        rows += x
        rat += z

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "calibration_metrics.csv"), index=False, encoding="utf-8")
    R = pd.DataFrame(rat)
    R.to_csv(os.path.join(RES, "calibration_metrics_ratio.csv"), index=False, encoding="utf-8")

    log("=" * 118)
    log("판정")
    log("=" * 118)
    log("[읽는 법] 네 지표의 배율이 같은 방향이면 결론은 ECE 라는 지표의")
    log("          산물이 아니다. 하나라도 1 아래로 떨어지면 그 지표에서는")
    log("          주장이 서지 않으므로 본문에 그대로 적어야 한다.")
    log("")
    bad = []
    for _, r in R.iterrows():
        ks = [
            ("ECE", r.ece_ratio),
            ("편향제거", r.ece_debiased_ratio),
            ("ICI", r.ici_ratio),
            ("E90", r.e90_ratio),
        ]
        lo = [k for k, v in ks if not (v > 1.0)]
        log(
            "  %-10s %-34s %s"
            % (
                r.dataset,
                r.control,
                "네 지표 모두 1 초과" if not lo else "**1 이하인 지표: %s**" % ", ".join(lo),
            )
        )
        if lo:
            bad.append((r.dataset, r.control, lo))
    log("")
    if bad:
        log("  **%d개 설정에서 일부 지표가 1 을 넘지 못한다. 본문에 명시할 것.**" % len(bad))
    else:
        log("  모든 설정에서 네 지표 전부 1 을 넘는다 — 결론은 지표 선택에")
        log("  의존하지 않는다.")
    log("")
    for name, _ in S.DATASETS:
        t = out[(out.dataset == name) & (out.arm.str.startswith("T"))].iloc[0]
        log(
            "  %s 시간분할: ECE 중 편향 %.1f%%, 편향 제거 후 %.4f, ICI %.4f"
            % (name, 100.0 * t.ece_null_mean / max(t.ece, S.EPS), t.ece_debiased, t.ici)
        )
    log("")
    log("저장: %s" % os.path.join(RES, "calibration_metrics.csv"))
    log("      %s" % os.path.join(RES, "calibration_metrics_ratio.csv"))
    from pathlib import Path

    log.save(Path(RES) / "calibration_metrics_summary.txt")


if __name__ == "__main__":
    main()
