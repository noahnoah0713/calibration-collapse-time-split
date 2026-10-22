# -*- coding: utf-8 -*-
"""결정곡선 분석 — 보정이 무너지면 의사결정에서 무엇을 잃는가.

왜 필요한가
----------
지금 논문은 "ECE 가 0.1207 에서 0.4384 로 나빠졌다" 까지 말한다. 독자가
반드시 묻는 다음 질문은 "그래서 쓸 때 뭐가 달라지는가" 다. 판별력
지표(AUROC)는 임계값과 무관하고 보정 지표(ECE)는 확률의 눈금을 재지만,
둘 다 결정의 손익을 말하지 않는다.

순편익(net benefit)은 그 간극을 메운다 (Vickers & Elkin 2006):

    NB(p_t) = TP/N − (FP/N) × p_t/(1 − p_t)

p_t 는 임계확률 이고, 의사결정자가 "양성 하나를 놓치는 손해"와
"음성 하나를 잘못 걸러내는 손해"를 어떤 비로 보는지를 뜻한다. 비교 기준
두 개가 함께 그려진다 — 전부 양성 취급(모든 화합물을 추가 시험으로
보낸다)과 전부 음성 취급(아무것도 안 한다, NB = 0).

모델의 NB 가 두 기준선보다 높은 구간이 있어야 그 모델이 쓸모가 있다.
보정이 무너지면 예측확률의 눈금이 틀어지므로, 임계확률을 기준으로 한
결정이 체계적으로 어긋난다. 그것이 AUROC 로는 보이지 않는 손해다.

이 절의 해석 주의
----------------
독성 예측에서 "양성 판정" 은 치료 투여가 아니라 추가 시험·보류 를
뜻한다. 따라서 p_t 는 "이 정도 위험이면 더 보겠다" 는 역치로 읽어야 하고,
임상 치료 결정의 p_t 와 같은 의미가 아니다. 또한 NB 는 검증군 유병률에
직접 의존하므로, 시간분할 검증군과 무작위분할 검증군의 NB 를 같은 눈금
위에서 비교할 수 없다. 그래서 두 분할을 각각 자기 기준선과 비교한다.

비교하는 네 가지
---------------
  A. 시간분할, 무보정            ← §8 설정
  B. 시간분할, Platt(학습군 내부)  ← 검증군 라벨 없이 할 수 있는 것
  C. 시간분할, 사전확률 보정       ← 검증군 유병률을 안다고 가정 (§10 D)
  R. 무작위분할, 무보정           ← 참고 기준

사용법
------
    python src/decision_curve.py

출력
----
    results/decision_curve.csv
    results/decision_curve_summary.txt
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

PT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
N_BOOT = 2000


def net_benefit(y, p, pt):
    """Vickers & Elkin 2006. 임계확률 pt 에서의 순편익."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    flag = p >= pt
    tp = int(((flag) & (y == 1)).sum())
    fp = int(((flag) & (y == 0)).sum())
    return tp / n - (fp / n) * (pt / (1.0 - pt))


def nb_all(y, pt):
    """전부 양성 취급."""
    y = np.asarray(y)
    n = len(y)
    return (y == 1).sum() / n - ((y == 0).sum() / n) * (pt / (1.0 - pt))


def boot_nb(y, p, pt, n_boot=N_BOOT, seed=S.SEED):
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    v = [net_benefit(y[i], p[i], pt) for i in (r.integers(0, n, n) for _ in range(n_boot))]
    a = np.sort(np.array(v))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def platt(y_tr, p_tr, p_te):
    lp = lambda q: np.log(np.clip(q, S.EPS, 1 - S.EPS) / np.clip(1 - q, S.EPS, 1 - S.EPS))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(lp(p_tr).reshape(-1, 1), y_tr)
    return lr.predict_proba(lp(p_te).reshape(-1, 1))[:, 1]


def prior_shift(p, pi_src, pi_tgt):
    """Saerens 류 사전확률 보정 — 오즈에 유병률 비를 곱한다."""
    o = np.clip(p, S.EPS, 1 - S.EPS) / np.clip(1 - p, S.EPS, 1 - S.EPS)
    k = (pi_tgt / (1 - pi_tgt)) / (pi_src / (1 - pi_src))
    o2 = o * k
    return o2 / (1.0 + o2)


def run(name, fn):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]

    m = S.make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]
    p_tr = m.predict_proba(X[tr])[:, 1]
    p_platt = platt(y[tr], p_tr, p_t)
    p_prior = prior_shift(p_t, float(y[tr].mean()), float(y[te].mean()))

    r = np.random.default_rng(S.SEED)
    idx = r.permutation(len(y))
    rte, rtr = idx[: len(te)], idx[len(te) :]
    mr = S.make_model().fit(X[rtr], y[rtr])
    p_r = mr.predict_proba(X[rte])[:, 1]

    arms = [
        ("A. 시간분할 무보정", y[te], p_t),
        ("B. 시간분할 + Platt(학습군)", y[te], p_platt),
        ("C. 시간분할 + 사전확률 보정", y[te], p_prior),
        ("R. 무작위분할 무보정 (참고)", y[rte], p_r),
    ]

    log("=" * 104)
    log(
        "%s   시간분할 검증군 %d개(양성률 %.4f) / 무작위분할 검증군 %d개(%.4f)"
        % (name, len(te), float(y[te].mean()), len(rte), float(y[rte].mean()))
    )
    log("  NB 는 검증군 유병률에 직접 의존한다 — 두 분할은 각자의 기준선과만 비교한다.")
    log("=" * 104)
    log(
        "%-28s %7s %9s %9s %9s %-18s %s"
        % ("설정", "p_t", "NB", "전부양성", "차(모델−전부)", "NB 95%CI", "판정")
    )
    log("-" * 112)

    rows = []
    for tag, yy, pp in arms:
        for pt in PT:
            nb = net_benefit(yy, pp, pt)
            na = nb_all(yy, pt)
            lo, hi = boot_nb(yy, pp, pt)
            # 쓸모 판정: 두 기준선(전부양성, 전부음성=0) 모두보다 높아야 한다
            useful = (nb > na) and (nb > 0)
            log(
                "%-28s %7.2f %9.4f %9.4f %9.4f [%+.4f, %+.4f] %s"
                % (tag, pt, nb, na, nb - na, lo, hi, "쓸모 있음" if useful else "")
            )
            rows.append(
                {
                    "dataset": name,
                    "arm": tag,
                    "p_t": pt,
                    "net_benefit": nb,
                    "nb_treat_all": na,
                    "nb_minus_all": nb - na,
                    "nb_lo": lo,
                    "nb_hi": hi,
                    "beats_both": bool(useful),
                    "n_test": len(yy),
                    "test_rate": float(yy.mean()),
                    "auroc": float(roc_auc_score(yy, pp)),
                    "ece": S.ece(yy, pp),
                }
            )
        log("-" * 112)
    log("")
    return rows


def main():
    log("결정곡선 분석 — 보정 붕괴의 의사결정 비용 (Vickers & Elkin 2006)")
    log("임계확률 %s, 부트스트랩 %d회" % (PT, N_BOOT))
    log("[주의] 독성 예측의 '양성 판정' 은 추가 시험·보류를 뜻한다. p_t 는")
    log("       '이 정도 위험이면 더 본다' 는 역치로 읽어야 한다.")
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "decision_curve.csv"), index=False, encoding="utf-8")

    log("=" * 104)
    log("판정")
    log("=" * 104)
    log("[읽는 법] '쓸모 있는 p_t 지점 수' 는 쓰지 않는다 — 그 값은 검증군")
    log("          유병률에 지배당한다. 유병률이 높으면 '전부 양성' 기준선이")
    log("          세어서 어떤 모델도 낮은 임계확률에서 이기지 못한다.")
    log("          대신 두 가지만 본다 — (1) NB 가 **음수** 로 내려가는 지점")
    log("          (아무것도 안 하느니만 못한 구간), (2) 무보정 대비 보정이")
    log("          되살리는 NB 의 크기.")
    log("")
    for d in out.dataset.unique():
        g = out[out.dataset == d]
        base = g[g.arm.str.startswith("A.")].set_index("p_t")
        log("  %s   검증군 양성률 %.4f" % (d, float(base.test_rate.iloc[0])))
        neg = base[base.net_benefit < 0]
        if len(neg):
            log(
                "    무보정 NB 가 음수인 구간: p_t >= %.2f "
                "(최저 %.4f at p_t %.2f) — **아무것도 안 하느니만 못하다**"
                % (
                    float(neg.index.min()),
                    float(neg.net_benefit.min()),
                    float(neg.net_benefit.idxmin()),
                )
            )
        else:
            log("    무보정 NB 는 전 구간에서 0 이상이다")
        for arm in g.arm.unique():
            if arm.startswith("A."):
                continue
            a2 = g[g.arm == arm].set_index("p_t")
            if arm.startswith("R."):
                log("    %-28s (다른 검증군 — NB 직접 비교 불가, 참고용)" % arm)
                continue
            dd = (a2.net_benefit - base.net_benefit).dropna()
            if not len(dd):
                continue
            log(
                "    %-28s 무보정 대비 NB 차: 최대 %+.4f (p_t %.2f) / "
                "최소 %+.4f (p_t %.2f)"
                % (arm, float(dd.max()), float(dd.idxmax()), float(dd.min()), float(dd.idxmin()))
            )
            hi = dd[dd.index >= 0.40]
            if len(hi):
                log("      p_t >= 0.40 구간 평균 %+.4f" % float(hi.mean()))
        log("")
    log("  [해석 주의] 보정오차와 의사결정 손해는 단조 관계가 아니다. 과대예측")
    log("              모델은 낮은 임계확률에서 '전부 양성' 에 가깝게 작동하므로")
    log("              그 구간에서는 손해가 작다. 손해는 **높은 임계확률** 에서")
    log("              나타난다 — 거짓양성을 꺼리는 의사결정자일수록 크게 잃는다.")
    log("              이 비대칭은 과대예측이라는 **방향** 때문이며, 방향이")
    log("              반대인 모델에서는 반대로 나타날 수 있다.")
    log("")
    log("저장: %s" % os.path.join(RES, "decision_curve.csv"))
    log("      %s" % os.path.join(RES, "decision_curve_summary.txt"))
    log.save(Path(RES) / "decision_curve_summary.txt")


if __name__ == "__main__":
    main()
