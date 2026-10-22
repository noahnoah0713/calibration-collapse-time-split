# -*- coding: utf-8 -*-
"""검증군 라벨 없이 유병률을 추정한다 — 정량화 문헌의 추정량 아홉 가지.

왜 필요한가 (한계 16번의 미검증 추측)
-----------------------------------
§10.2 는 추정량 넷(P0 가정·P1 예측평균·P2 EM·P3 BBSE)을 시도했고 **모두
유병률을 위쪽으로 틀렸다.** 한계 16번은 그 결과를 이렇게 적어 두었다 —

    "무너진 보정으로 유병률을 추정하려는 순환이므로, 더 정교한 추정량을
     붙여도 같은 벽에 부딪힐 것으로 본다 — **다만 그것은 추측이고
     검증하지 않았다.**"

이 논문의 한계 24개 중 **미검증 추측이 본문에 적혀 있는 유일한 항목**이다.
심사자가 가장 먼저 집을 자리다. 그리고 그 추측에는 **이미 발표된 반례
후보**가 있다 — Alexandari, Kundaje & Shrikumar (2020) 은 EM/MLLS 가
**기반 분류기를 먼저 보정하면** 잘 작동한다고 보였다. §10.2 의 EM 은
보정하지 않은 확률에 돌렸다. 즉 "순환" 이라는 진단이 맞다면 **보정을
먼저 끊어 주면 EM 이 살아날 수 있다.**

그래서 이 스크립트는 추측을 검증한다. 살아나면 논문에 **배포 가능한
처방이 처음으로 생긴다.** 살아나지 않으면 추측이 사실이 되고 한계 16번의
"검증하지 않았다" 가 사라진다. 어느 쪽이든 닫힌다.

추정량 아홉
----------
정량화(quantification) 문헌의 표준 추정량을 모았다. TPR·FPR 과 점수 분포는
**학습군 내부 분할**에서만 추정한다 — 검증군 라벨을 쓰지 않는다.

  Q1 CC    classify & count. 0.5 를 넘는 비율
  Q2 ACC   adjusted CC. (CC − FPR) / (TPR − FPR)
  Q3 PACC  probabilistic ACC. 확률 평균을 같은 식으로 보정
  Q4 SLD   Saerens-Latinne-Decaestecker EM (= §10.2 의 P2)
  Q5 SLD-c **Platt 보정 후** EM — Alexandari et al. 2020 의 처방
  Q6 BBSE-hard  경질 혼동행렬 역변환 (= §10.2 의 P3)
  Q7 BBSE-soft  연질(확률) 혼동행렬 역변환
  Q8 HDy   양·음 점수 분포의 혼합이 검증군 분포와 가장 가까워지는 비율
           (Hellinger 거리 최소화)
  Q9 MS    median sweep. 여러 임계값의 ACC 추정값 중앙값

각 추정량마다 **검증군 부트스트랩으로 구간**도 낸다(예측 고정, 재적합 없음).
점추정값만 보고 판정하지 않기 위해서다.

판정 기준
--------
추정값이 좋다는 것은 두 가지를 뜻한다.
  (1) 참 유병률에 가깝다 — pi 오차
  (2) 그 값으로 사전확률 보정을 걸면 **ECE 가 실제로 내려간다** — 회복률
(1)이 좋아도 (2)가 음수면 처방이 아니다. 그래서 둘을 같이 찍는다.

§10.2 와 같은 설계·같은 순서(승인연도 정렬)를 쓰고, 겹치는 추정량
(Q1↔P1 계열·Q4↔P2·Q6↔P3)이 기존 결과와 일치하는지 먼저 검사한다.

사용법
------
    python src/prior_estimation_v2.py            # 부트스트랩 2000회
    python src/prior_estimation_v2.py --boot 200

출력
----
    results/prior_estimation_v2.csv
    results/prior_estimation_v2_summary.txt
"""

import argparse
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

CALIB_FRAC = 0.25  # §10 과 같은 보정셋 비율
EM_MAX = 1000
EM_TOL = 1e-8
N_BINS_HD = 10
TOL = 5e-3  # 기존 §10.2 재현 허용 오차 (정렬·적합 잡음 포함)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
    return np.log(p / (1 - p))


def platt_fit(p_cal, y_cal):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def prior_shift(p, src, tgt):
    src = float(np.clip(src, S.EPS, 1 - S.EPS))
    tgt = float(np.clip(tgt, S.EPS, 1 - S.EPS))
    odds = (tgt / (1 - tgt)) / (src / (1 - src))
    return 1.0 / (1.0 + np.exp(-(logit(p) + np.log(odds))))


# ---------------------------------------------------------------- 추정량
def q_cc(p_te, thr=0.5):
    return float((np.asarray(p_te) >= thr).mean())


def q_acc(p_te, tpr, fpr, thr=0.5):
    cc = q_cc(p_te, thr)
    den = tpr - fpr
    if abs(den) < 1e-6:
        return float("nan")
    return float(np.clip((cc - fpr) / den, 0.0, 1.0))


def q_pacc(p_te, e_pos, e_neg):
    """확률 평균판 ACC. e_pos/e_neg 는 학습군에서 잰 클래스별 확률 평균."""
    den = e_pos - e_neg
    if abs(den) < 1e-6:
        return float("nan")
    return float(np.clip((float(np.mean(p_te)) - e_neg) / den, 0.0, 1.0))


def q_em(p_te, pi_src, max_iter=EM_MAX, tol=EM_TOL):
    """Saerens-Latinne-Decaestecker EM."""
    p = np.clip(np.asarray(p_te, dtype=float), S.EPS, 1 - S.EPS)
    pi_s = float(np.clip(pi_src, S.EPS, 1 - S.EPS))
    pi = pi_s
    for i in range(max_iter):
        w1 = (pi / pi_s) * p
        w0 = ((1 - pi) / (1 - pi_s)) * (1 - p)
        post = w1 / np.clip(w1 + w0, S.EPS, None)
        new = float(post.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return float(np.clip(pi, 0.0, 1.0)), i + 1


def q_bbse(p_te, p_cal, y_cal, soft=False, thr=0.5):
    """혼동행렬 역변환. soft=True 면 확률을 그대로 쓴다."""
    y_cal = np.asarray(y_cal)
    if soft:
        e_pos = float(np.mean(p_cal[y_cal == 1]))
        e_neg = float(np.mean(p_cal[y_cal == 0]))
        return q_pacc(p_te, e_pos, e_neg)
    tpr = float((np.asarray(p_cal)[y_cal == 1] >= thr).mean())
    fpr = float((np.asarray(p_cal)[y_cal == 0] >= thr).mean())
    return q_acc(p_te, tpr, fpr, thr)


def q_hdy(p_te, p_cal, y_cal, n_bins=N_BINS_HD):
    """양·음 점수 히스토그램의 혼합이 검증군 히스토그램과 가장 가까운 비율."""
    y_cal = np.asarray(y_cal)
    edges = np.linspace(0, 1, n_bins + 1)

    def hist(v):
        h, _ = np.histogram(np.asarray(v), bins=edges)
        return h / max(h.sum(), 1)

    h_pos, h_neg = hist(p_cal[y_cal == 1]), hist(p_cal[y_cal == 0])
    h_te = hist(p_te)
    grid = np.linspace(0.0, 1.0, 101)
    best, best_pi = None, float("nan")
    for pi in grid:
        mix = pi * h_pos + (1 - pi) * h_neg
        # Hellinger 거리
        d = float(np.sqrt(np.sum((np.sqrt(mix) - np.sqrt(h_te)) ** 2)) / np.sqrt(2))
        if best is None or d < best:
            best, best_pi = d, float(pi)
    return best_pi


def q_ms(p_te, p_cal, y_cal, thrs=None):
    """median sweep — 여러 임계값의 ACC 추정값 중앙값."""
    y_cal = np.asarray(y_cal)
    if thrs is None:
        thrs = np.linspace(0.05, 0.95, 19)
    vals = []
    for t in thrs:
        tpr = float((np.asarray(p_cal)[y_cal == 1] >= t).mean())
        fpr = float((np.asarray(p_cal)[y_cal == 0] >= t).mean())
        if abs(tpr - fpr) < 0.05:  # 분모가 작으면 버린다
            continue
        v = q_acc(p_te, tpr, fpr, t)
        if v == v:
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


# ---------------------------------------------------------------- 실행
def run(name, fn, n_boot, ref):
    # §10.2 와 같은 순서 (승인연도 정렬)
    df = pd.read_csv(os.path.join(PROC, fn))
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.sort_values("approval_year").reset_index(drop=True)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m = yr < S.CUTOFF
    tr, te = np.where(tr_m)[0], np.where(~tr_m)[0]
    n_cal = int(round(len(tr) * CALIB_FRAC))
    fit_i, cal_i = tr[:-n_cal], tr[-n_cal:]
    pi_tr, pi_te = float(y[tr].mean()), float(y[te].mean())

    m_all = S.make_model().fit(X[tr], y[tr])
    p_te = m_all.predict_proba(X[te])[:, 1]
    ece_A = S.ece(y[te], p_te)

    # 보정셋 — 학습군 내부에서만
    m_fit = S.make_model().fit(X[fit_i], y[fit_i])
    p_cal = m_fit.predict_proba(X[cal_i])[:, 1]
    p_te_fit = m_fit.predict_proba(X[te])[:, 1]
    cal_fn = platt_fit(p_cal, y[cal_i])
    p_cal_c = cal_fn(p_cal)
    p_te_c = cal_fn(p_te_fit)

    log("=" * 122)
    log(
        "%s   학습 %d (적합 %d / 보정셋 %d) / 검증 %d"
        % (name, len(tr), len(fit_i), len(cal_i), len(te))
    )
    log("  학습군 유병률 %.4f, 검증군 참값 %.4f, 무보정 ECE %.4f" % (pi_tr, pi_te, ece_A))
    if ref is not None:
        q = ref[(ref.dataset == name) & (ref.estimator.str.startswith("(기준) A"))]
        if len(q):
            d0 = abs(float(q.ECE.iloc[0]) - ece_A)
            log(
                "  §10.2 의 A 행 ECE 와의 차 %.2e → %s"
                % (d0, "일치" if d0 < TOL else "**불일치 — 확인 필요**")
            )
    log("=" * 122)

    # 학습군에서 잰 보조량
    tpr = float((p_cal[y[cal_i] == 1] >= 0.5).mean())
    fpr = float((p_cal[y[cal_i] == 0] >= 0.5).mean())
    e_pos = float(np.mean(p_cal[y[cal_i] == 1]))
    e_neg = float(np.mean(p_cal[y[cal_i] == 0]))
    log(
        "  학습군 보조량: TPR %.4f / FPR %.4f / E[p|y=1] %.4f / E[p|y=0] %.4f"
        % (tpr, fpr, e_pos, e_neg)
    )
    log("")

    em_u, it_u = q_em(p_te, pi_tr)  # 전체 학습 모델, 무보정 (= §10.2 P2)
    em_f, it_f = q_em(p_te_fit, pi_tr)  # 보정셋 제외 모델, 무보정
    em_c, it_c = q_em(p_te_c, pi_tr)  # 보정셋 제외 모델, Platt 보정
    # Q4b 와 Q5 는 **같은 기반 모델**을 쓴다. 그래야 Platt 보정의 효과만
    # 떼어낼 수 있다. Q4 는 §10.2 의 P2 와 같은 설정(전체 학습)으로 남긴다.
    EST = [
        ("Q1 CC (classify & count)", q_cc(p_te), "라벨 없음"),
        ("Q2 ACC (adjusted CC)", q_acc(p_te, tpr, fpr), "학습군 라벨만"),
        ("Q3 PACC (probabilistic ACC)", q_pacc(p_te, e_pos, e_neg), "학습군 라벨만"),
        ("Q4 SLD/EM, 전체 학습 (= §10.2 P2)", em_u, "라벨 없음"),
        ("Q4b SLD/EM, 보정셋 제외 학습 (Q5 의 짝)", em_f, "라벨 없음"),
        ("Q5 SLD/EM + Platt 보정 (Alexandari 2020)", em_c, "학습군 라벨만"),
        (
            "Q6 ACC/BBSE-hard (조건부율 역변환)",
            q_bbse(p_te, p_cal, y[cal_i], False),
            "학습군 라벨만",
        ),
        (
            "Q7 BBSE-soft (= Q3 과 대수적으로 동일)",
            q_bbse(p_te, p_cal, y[cal_i], True),
            "학습군 라벨만",
        ),
        ("Q8 HDy (Hellinger 최소화)", q_hdy(p_te, p_cal, y[cal_i]), "학습군 라벨만"),
        ("Q9 MS (median sweep)", q_ms(p_te, p_cal, y[cal_i]), "학습군 라벨만"),
        ("Q0 참값 (oracle, 달성 불가)", pi_te, "**검증군 라벨**"),
    ]

    # oracle Platt 상한 — 회복률 분모
    oracle = prior_shift(p_te, pi_tr, pi_te)
    ece_oracle_ps = S.ece(y[te], oracle)
    cal_or = platt_fit(p_te, y[te])  # 검증군 라벨을 쓴 상한선
    ece_E = S.ece(y[te], cal_or(p_te))

    log(
        "%-40s %-14s %9s %9s %9s %10s %-20s"
        % ("추정량", "쓰는 정보", "pi 추정", "pi 오차", "보정 ECE", "회복률", "pi 95% 구간")
    )
    log("-" * 122)
    rows = []
    rb = np.random.default_rng(S.SEED + 21)
    for lab, pi_hat, info in EST:
        if pi_hat != pi_hat:
            log("%-40s %-14s %9s  추정 불가 (분모 0)" % (lab, info, "—"))
            rows.append(
                {"dataset": name, "estimator": lab, "info_used": info, "pi_hat": float("nan")}
            )
            continue
        p_corr = prior_shift(p_te, pi_tr, pi_hat)
        ece_c = S.ece(y[te], p_corr)
        recov = (ece_A - ece_c) / (ece_A - ece_E) * 100.0 if (ece_A - ece_E) != 0 else float("nan")
        # 검증군 부트스트랩으로 pi 추정의 구간 (예측 고정)
        bs = []
        for _ in range(n_boot):
            i = rb.integers(0, len(te), len(te))
            try:
                if lab.startswith("Q1"):
                    v = q_cc(p_te[i])
                elif lab.startswith("Q2"):
                    v = q_acc(p_te[i], tpr, fpr)
                elif lab.startswith("Q3"):
                    v = q_pacc(p_te[i], e_pos, e_neg)
                elif lab.startswith("Q4b"):
                    v = q_em(p_te_fit[i], pi_tr)[0]
                elif lab.startswith("Q4"):
                    v = q_em(p_te[i], pi_tr)[0]
                elif lab.startswith("Q5"):
                    v = q_em(p_te_c[i], pi_tr)[0]
                elif lab.startswith("Q6"):
                    v = q_bbse(p_te[i], p_cal, y[cal_i], False)
                elif lab.startswith("Q7"):
                    v = q_bbse(p_te[i], p_cal, y[cal_i], True)
                elif lab.startswith("Q8"):
                    v = q_hdy(p_te[i], p_cal, y[cal_i])
                elif lab.startswith("Q9"):
                    v = q_ms(p_te[i], p_cal, y[cal_i])
                else:
                    v = float(y[te][i].mean())
            except Exception:
                v = float("nan")
            if v == v:
                bs.append(v)
        lo = float(np.percentile(bs, 2.5)) if bs else float("nan")
        hi = float(np.percentile(bs, 97.5)) if bs else float("nan")
        rows.append(
            {
                "dataset": name,
                "estimator": lab,
                "info_used": info,
                "pi_hat": pi_hat,
                "pi_true": pi_te,
                "pi_train": pi_tr,
                "pi_err": pi_hat - pi_te,
                "pi_lo": lo,
                "pi_hi": hi,
                "covers_true": bool(lo <= pi_te <= hi),
                "ece_base": ece_A,
                "ece_corrected": ece_c,
                "ece_oracle_platt": ece_E,
                "recovery": recov,
                "n_boot_ok": len(bs),
            }
        )
        log(
            "%-40s %-14s %9.4f %+9.4f %9.4f %9.1f%% [%.4f, %.4f]"
            % (lab, info, pi_hat, pi_hat - pi_te, ece_c, recov, lo, hi)
        )
    log("")
    _nc = [n for n, it in (("Q4", it_u), ("Q4b", it_f), ("Q5", it_c)) if it >= EM_MAX]
    log("  EM 반복 횟수: Q4 %d회 / Q4b %d회 / Q5 %d회 (상한 %d)" % (it_u, it_f, it_c, EM_MAX))
    if _nc:
        log(
            "  **%s 는 반복 상한에서 멈췄다 — 수렴하지 않았으므로 그 추정값은 "
            "믿을 수 없다.**" % ", ".join(_nc)
        )
    log("  참조: 사전확률 보정의 oracle ECE %.4f, Oracle Platt 상한 %.4f" % (ece_oracle_ps, ece_E))

    # §10.2 와 겹치는 추정량 대조
    if ref is not None:
        log("")
        log("  §10.2 와 겹치는 추정량 대조 (정렬·보정셋 설계가 같다)")
        pairs = [
            ("Q4 SLD/EM, 전체 학습 (= §10.2 P2)", "P2 EM"),
            ("Q6 ACC/BBSE-hard (조건부율 역변환)", "P3 BBSE"),
        ]
        for new_lab, old_pre in pairs:
            q = ref[(ref.dataset == name) & (ref.estimator.str.startswith(old_pre))]
            r2 = [r for r in rows if r["estimator"] == new_lab]
            if len(q) and r2 and r2[0]["pi_hat"] == r2[0]["pi_hat"]:
                d = r2[0]["pi_hat"] - float(q.pi_hat.iloc[0])
                _ok = abs(d) < TOL
                log(
                    "    %-28s §10.2 %.4f / 여기 %.4f (차 %+.4f) → %s"
                    % (
                        old_pre,
                        float(q.pi_hat.iloc[0]),
                        r2[0]["pi_hat"],
                        d,
                        "일치" if _ok else "**다른 구현**",
                    )
                )
                if old_pre.startswith("P3") and not _ok:
                    log("      §10.2 의 P3 은 **결합 혼동행렬** P(yhat, y) 를 풀고")
                    log("      이쪽 Q6 은 **조건부율** (CC-FPR)/(TPR-FPR) 을 쓴다.")
                    log("      둘 다 문헌에서 BBSE 로 불리지만 같은 추정량이 아니다.")
                    log(
                        "      §10.2 의 행렬식이 %.4f 로 거의 특이해서 그 해가 불안정하다 —"
                        % float(q.bbse_det.iloc[0])
                    )
                    log("      **두 합당한 구현이 %.2f 나 벌어진다는 사실 자체가" % abs(d))
                    log("      이 과제에서 사전확률 추정이 불안정하다는 증거다.**")
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    log("검증군 라벨 없이 유병률을 추정한다 — 정량화 문헌의 추정량 아홉 가지")
    log('한계 16번의 미검증 추측("더 정교한 추정량도 같은 벽에 부딪힌다")을 검증한다.')
    log("핵심 비교는 Q4b(무보정 EM) 대 Q5(Platt 보정 후 EM) 다 — 둘은 같은 기반")
    log("모델을 쓰므로 보정의 효과만 떼어낼 수 있다. Q4 는 §10.2 P2 와 같은 설정이다.")
    log("Alexandari et al. 2020 은 보정이 EM 을 살린다고 보였다.")
    log("검증군 부트스트랩 %d회, 시드 %d" % (a.boot, S.SEED))
    log("")
    _p = os.path.join(RES, "prior_estimation.csv")
    ref = pd.read_csv(_p) if os.path.exists(_p) else None

    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.boot, ref)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "prior_estimation_v2.csv"), index=False, encoding="utf-8")

    log("=" * 122)
    log("판정")
    log("=" * 122)
    log("[읽는 법] pi 오차가 작아도 **회복률이 양수가 아니면 처방이 아니다.**")
    log("          둘을 같이 본다. 구간이 참값을 덮는지도 본다.")
    log("          Q3 PACC 와 Q7 BBSE-soft 는 이진 문제에서 대수적으로 같은")
    log("          추정량이다 — 값이 같은 것은 오류가 아니라 확인이다.")
    log("          **회복률이 양수라도 pi 를 잘 맞혀서인지, 아래로 틀려서인지")
    log("          구분해야 한다** — §23 은 유병률을 아래로 틀리면 oracle 보다")
    log("          ECE 가 낮아진다는 것을 이미 보였다.")
    log("")
    d = out.dropna(subset=["pi_hat"])
    d = d[~d.estimator.str.startswith("Q0")]
    for ds, g in d.groupby("dataset", sort=False):
        best_pi = g.loc[g.pi_err.abs().idxmin()]
        best_rc = g.loc[g.recovery.idxmax()]
        _pos = g[g.recovery > 0]
        _cov = g[g.covers_true]
        log("  %s" % ds)
        log(
            "    pi 오차 최소: %s (오차 %+.4f, 회복률 %.1f%%)"
            % (best_pi.estimator, best_pi.pi_err, best_pi.recovery)
        )
        log(
            "    회복률 최대: %s (%.1f%%, pi 오차 %+.4f)"
            % (best_rc.estimator, best_rc.recovery, best_rc.pi_err)
        )
        log(
            "    회복률이 양수인 추정량 %d/%d, 구간이 참값을 덮는 추정량 %d/%d"
            % (len(_pos), len(g), len(_cov), len(g))
        )
        _up = int((g.pi_err > 0).sum())
        log("    유병률을 **위쪽으로** 틀린 추정량 %d/%d" % (_up, len(g)))
        q4 = g[g.estimator.str.startswith("Q4b")]
        q5 = g[g.estimator.str.startswith("Q5")]
        if len(q4) and len(q5):
            log(
                "    핵심 비교 (같은 기반 모델) — Q4b 무보정 EM 오차 %+.4f / "
                "회복률 %.1f%%  대" % (q4.pi_err.iloc[0], q4.recovery.iloc[0])
            )
            log(
                "                Q5 보정 후 EM 오차 %+.4f / 회복률 %.1f%%"
                % (q5.pi_err.iloc[0], q5.recovery.iloc[0])
            )
            if q5.recovery.iloc[0] > 0 and q5.recovery.iloc[0] > q4.recovery.iloc[0]:
                log("                → **보정이 EM 을 살린다. 추측이 반박된다.**")
            elif abs(q5.pi_err.iloc[0]) < abs(q4.pi_err.iloc[0]):
                log("                → 보정이 pi 추정은 개선하지만 회복률은 " "양수가 아니다.")
            else:
                log("                → **보정해도 살아나지 않는다. 추측이 지지된다.**")
        log("")
    _under = d[(d.recovery > 0) & (d.pi_err < 0)]
    _over = d[(d.recovery > 0) & (d.pi_err >= 0)]
    if len(_under):
        log("  **회복률이 양수인 추정량 중 %d개는 유병률을 아래로 틀렸다.**" % len(_under))
        for _, r in _under.iterrows():
            log(
                "      %-10s %-40s pi 오차 %+.4f, 회복률 %.1f%%"
                % (r.dataset, r.estimator, r.pi_err, r.recovery)
            )
        log("  §23 이 보인 대로 아래로 틀리면 ECE 가 내려간다(oracle 보다도). 그래서")
        log("  이 회복은 **추정이 성공해서가 아니라 과소추정의 부수효과일 수 있다.**")
        log("  위로 틀리면서 회복한 추정량은 %d개다." % len(_over))
        log("")
    _any = d[d.recovery > 0]
    if len(_any):
        log(
            "  **회복률이 양수인 추정량이 %d개 있다** — %s. 한계 16번의 추측은 "
            "그만큼 수정해야 한다."
            % (
                len(_any),
                ", ".join(
                    "%s %s" % (r.dataset, str(r.estimator).split(" ")[0])
                    for _, r in _any.iterrows()
                ),
            )
        )
    else:
        log(
            "  **아홉 추정량 전부 회복률이 0 이하다. 한계 16번의 추측이 "
            "검증됐다** — 더 정교한 추정량도 같은 벽에 부딪힌다."
        )
    log("")
    log("저장: %s" % os.path.join(RES, "prior_estimation_v2.csv"))
    log.save(Path(RES) / "prior_estimation_v2_summary.txt")


if __name__ == "__main__":
    main()
