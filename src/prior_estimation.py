# -*- coding: utf-8 -*-
"""§10.2 — 검증군 유병률을 '모르는 상태에서' 추정해 사전확률 보정을 걸 수 있는가.

왜 필요한가
----------
§10 의 재보정 다섯 방법 중 D(사전확률 보정)는 DILIrank 에서 가장 잘 듣는다
(ECE 0.4466 -> 0.2959, 회복률 37.8%). 그런데 그 계산은 검증군의 실제 양성률
`y_test.mean()` 을 가져다 쓴다(recalibration.py). 미래 집합의 유병률은 배포
시점에 알 수 없으므로 **D 는 그대로 배포할 수 없다.** 그러면 §10 의 결론을
"학습군 내부 재보정으로는 일부만 회복된다" 로 쓸 때 D 를 근거로 삼을 수 없다.

이 스크립트는 그 공백을 메운다. **라벨을 쓰지 않고** 목표 유병률을 추정하는
방법 네 가지를 같은 자료에 적용해, 추정값이 실제 유병률에 얼마나 가까운지와
그 추정값으로 같은 절편 보정을 걸었을 때 ECE 가 얼마나 회복되는지를 본다.

추정량
------
  P0 train      학습군 양성률을 그대로 쓴다 (이동이 없다고 가정하는 기본값)
  P1 mean_pred  검증군 예측확률의 평균 (가장 단순한 대입. 보정이 무너진
                상황에서는 크게 편향된다 — 그 사실 자체가 보고할 값이다)
  P2 EM         Saerens, Latinne, Decaestecker (2002) 의 EM.
                라벨 없이 예측확률만으로 목표 사전확률을 반복 추정한다.
  P3 BBSE       Lipton, Wang, Smola (2018) 의 black box shift estimation.
                학습군 내부 보정셋(§10 과 같은 구간)에서 혼동행렬을 구하고,
                검증군의 예측 라벨 분포를 역변환한다. 라벨은 **학습군 쪽만**
                쓴다.
  P4 oracle     검증군 실제 양성률 (= §10 의 D). 상한 비교용이며 배포 불가.

모든 추정값에 recalibration.py 의 prior_shift 를 그대로 적용하므로, 방법 간
차이는 '사전확률 추정값' 하나뿐이다. 무보정(A)과 Oracle Platt(E)도 같은
스크립트에서 계산해 §10 과 같은 정의로 회복률을 낸다.

사용법
------
    python src/prior_estimation.py

출력
----
    results/prior_estimation.csv
    results/prior_estimation_summary.txt
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S
import recalibration as RC  # featurize·make_model·assess·prior_shift 재사용

warnings.filterwarnings("ignore")

RES = RC.RES
PROC = RC.PROC
EM_MAX_ITER = 1000
EM_TOL = 1e-10
log = S.Log()


# ---------------------------------------------------------------- 추정량
def prior_em(p, prior_src, max_iter=EM_MAX_ITER, tol=EM_TOL):
    """Saerens-Latinne-Decaestecker EM. 라벨을 쓰지 않는다.

    q_i = (r*p_i) / (r*p_i + s*(1-p_i)),  r = pi_t/pi_s, s = (1-pi_t)/(1-pi_s)
    pi_t <- mean(q)
    """
    p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
    pi_s = float(np.clip(prior_src, S.EPS, 1 - S.EPS))
    pi_t = pi_s
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        r = pi_t / pi_s
        s = (1 - pi_t) / (1 - pi_s)
        q = (r * p) / (r * p + s * (1 - p))
        new = float(q.mean())
        if abs(new - pi_t) < tol:
            pi_t = new
            break
        pi_t = new
    return float(np.clip(pi_t, S.EPS, 1 - S.EPS)), n_iter


def prior_bbse(p_val, y_val, p_test, thr=0.5):
    """BBSE-hard. 학습군 내부 보정셋의 혼동행렬로 검증군 사전확률을 역추정.

    C[j, k] = P(yhat = j, y = k)  (보정셋에서 추정)
    q[j]    = P(yhat = j)         (검증군, 라벨 불필요)
    C @ pi = q  를 풀고 [0,1] 로 자른 뒤 정규화한다.
    """
    yh_v = (np.asarray(p_val) >= thr).astype(int)
    yv = np.asarray(y_val).astype(int)
    n = len(yv)
    C = np.zeros((2, 2))
    for j in (0, 1):
        for k in (0, 1):
            C[j, k] = np.sum((yh_v == j) & (yv == k)) / n
    yh_t = (np.asarray(p_test) >= thr).astype(int)
    q = np.array([np.mean(yh_t == 0), np.mean(yh_t == 1)])
    det = np.linalg.det(C)
    if abs(det) < 1e-12:
        return None, det  # 역변환 불가 — 보고만 하고 쓰지 않는다
    pi = np.linalg.solve(C, q)
    pi = np.clip(pi, 0.0, 1.0)
    tot = pi.sum()
    if tot <= 0:
        return None, det
    return float(np.clip(pi[1] / tot, S.EPS, 1 - S.EPS)), det


# ---------------------------------------------------------------- 실행
def run(name, path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.sort_values("approval_year").reset_index(drop=True)

    X = RC.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_all = yr < RC.CUTOFF
    te = ~tr_all

    # §10 과 동일한 학습군 내부 분할 (앞 75% 적합 / 최근 25% 보정셋)
    tr_idx = np.where(tr_all)[0]
    n_cal = int(round(len(tr_idx) * RC.CALIB_FRAC))
    fit_idx, cal_idx = tr_idx[:-n_cal], tr_idx[-n_cal:]

    # A. 무보정 (전체 학습) — §10 의 A 와 같은 모델
    m_all = RC.make_model().fit(X[tr_all], y[tr_all])
    p_base = m_all.predict_proba(X[te])[:, 1]
    # A'. 보정셋을 뺀 모델 — BBSE 의 혼동행렬은 이 모델로 구해야 한다
    m_fit = RC.make_model().fit(X[fit_idx], y[fit_idx])
    p_cal = m_fit.predict_proba(X[cal_idx])[:, 1]
    p_te_fit = m_fit.predict_proba(X[te])[:, 1]

    yte = y[te]
    pi_src = float(y[tr_all].mean())
    pi_true = float(yte.mean())

    A = RC.assess(yte, p_base, "A. 무보정 (전체 학습)")
    E = RC.assess(yte, RC.platt(p_base, yte, p_base), "E. Oracle Platt (상한선)")
    denom = A["ECE"] - E["ECE"]

    pi_em, em_iter = prior_em(p_base, pi_src)
    pi_bb, bb_det = prior_bbse(p_cal, y[cal_idx], p_te_fit)

    ests = [
        ("P0 train (이동 없다고 가정)", pi_src, "학습군 라벨만"),
        ("P1 mean_pred (예측확률 평균)", float(p_base.mean()), "라벨 없음"),
        ("P2 EM (Saerens 2002)", pi_em, "라벨 없음"),
        ("P3 BBSE (Lipton 2018)", pi_bb, "학습군 라벨만"),
        ("P4 oracle (§10 의 D)", pi_true, "**검증군 라벨**"),
    ]

    log("=" * 96)
    log(f"{name}   학습 {int(tr_all.sum())} / 검증 {int(te.sum())}")
    log("=" * 96)
    log(
        f"학습군 양성률 {pi_src:.4f} / 검증군 실제 양성률 {pi_true:.4f} "
        f"(이동 {pi_true - pi_src:+.4f})"
    )
    log(f"EM 반복 {em_iter}회 수렴 / BBSE 혼동행렬 행렬식 {bb_det:.6f}")
    log("")
    log(
        f"{'추정량':30s} {'쓰는 정보':14s} {'추정 유병률':>10s} {'오차':>8s} "
        f"{'AUROC':>7s} {'ECE':>8s} {'기울기':>8s} {'회복률':>8s}"
    )
    log("-" * 96)

    rows = []
    for tag, pi_hat, info in ests:
        if pi_hat is None:
            log(f"{tag:30s} {info:14s}      추정 실패 (혼동행렬 역변환 불가)")
            rows.append(
                {
                    "dataset": name,
                    "estimator": tag,
                    "info_used": info,
                    "pi_hat": np.nan,
                    "pi_true": pi_true,
                    "pi_train": pi_src,
                    "pi_err": np.nan,
                    "AUROC": np.nan,
                    "ECE": np.nan,
                    "slope": np.nan,
                    "recovery": np.nan,
                    "em_iter": em_iter,
                    "bbse_det": bb_det,
                }
            )
            continue
        p_adj = RC.prior_shift(p_base, pi_src, pi_hat)
        a = RC.assess(yte, p_adj, tag)
        rec = (A["ECE"] - a["ECE"]) / denom * 100
        log(
            f"{tag:30s} {info:14s} {pi_hat:10.4f} {pi_hat - pi_true:+8.4f} "
            f"{a['AUROC']:7.4f} {a['ECE']:8.4f} {a['slope']:8.4f} {rec:7.1f}%"
        )
        rows.append(
            {
                "dataset": name,
                "estimator": tag,
                "info_used": info,
                "pi_hat": pi_hat,
                "pi_true": pi_true,
                "pi_train": pi_src,
                "pi_err": pi_hat - pi_true,
                "AUROC": a["AUROC"],
                "ECE": a["ECE"],
                "slope": a["slope"],
                "recovery": rec,
                "em_iter": em_iter,
                "bbse_det": bb_det,
            }
        )

    log("-" * 96)
    log(
        f"{'(참고) A. 무보정':30s} {'—':14s} {'—':>10s} {'—':>8s} "
        f"{A['AUROC']:7.4f} {A['ECE']:8.4f} {A['slope']:8.4f} {0.0:7.1f}%"
    )
    log(
        f"{'(참고) E. Oracle Platt':30s} {'검증군 라벨 전부':14s} {'—':>10s} {'—':>8s} "
        f"{E['AUROC']:7.4f} {E['ECE']:8.4f} {E['slope']:8.4f} {100.0:7.1f}%"
    )
    log("")
    for tag, meta in (("A. 무보정 (전체 학습)", A), ("E. Oracle Platt (상한선)", E)):
        rows.append(
            {
                "dataset": name,
                "estimator": "(기준) " + tag,
                "info_used": "—" if tag.startswith("A") else "검증군 라벨 전부",
                "pi_hat": np.nan,
                "pi_true": pi_true,
                "pi_train": pi_src,
                "pi_err": np.nan,
                "AUROC": meta["AUROC"],
                "ECE": meta["ECE"],
                "slope": meta["slope"],
                "recovery": 0.0 if tag.startswith("A") else 100.0,
                "em_iter": em_iter,
                "bbse_det": bb_det,
            }
        )
    return rows


def main():
    log("사전확률 추정 — 검증군 유병률을 모른다고 할 때 (§10.2)")
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, PROC / fn)
        log("")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "prior_estimation.csv", index=False, encoding="utf-8")

    log("=" * 96)
    log("판정")
    log("=" * 96)
    log("[읽는 법] 라벨을 쓰지 않는 추정량(P1·P2)이 oracle(P4)의 회복률에 가까우면,")
    log("          사전확률 보정은 실제로 배포할 수 있는 처방이다. 멀면 §10 의 D 는")
    log("          상한선이고 '배포 가능한 재보정' 은 B·C 뿐이라고 써야 한다.")
    log("          P0 는 '이동이 없다고 가정' 이므로 보정이 걸리지 않는다(회복률 0 근처).")
    log("")
    for d in out.dataset.unique():
        g = out[(out.dataset == d) & (out.estimator.str.startswith("P"))]
        o = g[g.estimator.str.startswith("P4")]
        best = g[~g.estimator.str.startswith("P4")].dropna(subset=["recovery"])
        if not len(o) or not len(best):
            continue
        b = best.loc[best.recovery.idxmax()]
        log(
            f"  {d}: oracle 회복률 {float(o.recovery.iloc[0]):.1f}% / "
            f"라벨 없는 최선 {b.estimator} {float(b.recovery):.1f}% "
            f"(유병률 추정 오차 {float(b.pi_err):+.4f})"
        )
    log("")
    log(f"저장: {RES / 'prior_estimation.csv'}")
    log(f"      {RES / 'prior_estimation_summary.txt'}")
    log.save(RES / "prior_estimation_summary.txt")


if __name__ == "__main__":
    main()
