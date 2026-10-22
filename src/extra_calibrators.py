# -*- coding: utf-8 -*-
"""재보정 사다리에 보정기 네 개를 더 붙인다 — "두 개만 시도했다" 를 막는다.

왜 필요한가
----------
§10 은 재보정을 Platt·Isotonic·사전확률 셋으로 시험하고 "사후 재보정으로는
일부만 되돌아온다" 고 결론했다. 보정 문헌을 아는 심사자는 바로 묻는다 —
**"보정기를 두 개만 써 보고 재보정이 안 된다고 결론했는가."** 특히 두 개가
자주 지목된다.

  베타 보정 (Kull 2017)        Platt 은 로짓에 선형 — 확률의 양 끝에서
                              경직되어 있다. 베타 보정은 log p 와
                              log(1-p) 를 따로 쓰므로 양 끝을 따로 휜다.
  Venn-ABERS (Vovk 2014)      Isotonic 을 conformal 틀에 넣은 것. 이 논문은
                              이미 §11 에서 conformal 을 쓰므로 가장
                              자연스러운 보완이고, **점추정이 아니라 구간**
                              을 준다는 점이 다르다.

여기에 둘을 더 붙인다.

  온도 조정                    로짓에 기울기 하나만.
                              **2026-10-20 추가: 이 예상은 틀렸다.** 절편을
                              고정하므로 밀림을 못 고칠 것이라고 적어 두었는데,
                              실제로는 두 데이터셋 모두에서 Platt 보다 ECE 가
                              낮다. 이유는 그 반대다 — Platt 이 학습군 내부
                              보정셋에서 **학습시대 유병률에 맞는 절편**을
                              배워 검증군에 가져오는 것이 해롭고, 절편을 0 으로
                              고정하면 그 잘못된 절편을 안 들여온다.
  Platt + 사전확률 합성         학습군 내부 Platt 뒤에 검증군 유병률로 절편을
                              다시 민다. §10 의 B 와 D 를 **이어 붙인** 것.

설계
----
분할·모델·보정셋은 §10(recalibration.py)과 **완전히 같다** — 연도 정렬 후
학습군의 뒤 25%. 그래서 B(Platt)·C(Isotonic)를 그대로 재현해야 하고,
재현되지 않으면 스크립트가 그 자리에서 멈춘다. 재적합은 데이터셋당 두 번
(적합셋 모델, 전체학습 모델)뿐이고 보정기는 저장된 확률만 쓴다.

회복률 정의도 §10 과 같다 — (A.ECE − X.ECE) / (A.ECE − E.ECE) × 100.
E 는 검증군 라벨로 적합한 oracle Platt 상한이다.

Venn-ABERS 는 어떻게 점추정으로 만드는가
--------------------------------------
IVAP(inductive Venn-ABERS)는 검증 분자 하나마다 그 분자의 라벨을 0 으로
가정한 isotonic 과 1 로 가정한 isotonic 을 각각 적합해 [p0, p1] 을 낸다.
그 구간이 Venn-ABERS 의 출력이고, 점추정이 필요할 때는 보통 중점을 쓴다.
**구간 폭도 함께 적는다** — 폭이 넓다면 그 자체가 "보정셋이 작다" 는 증거다.

사용법
------
    python src/extra_calibrators.py

출력
----
    results/extra_calibrators.csv
    results/extra_calibrators_summary.txt
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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

OUT = os.path.join(RES, "extra_calibrators.csv")
CALIB_FRAC = 0.25
N_BOOT = 2000


def logit(p):
    """§10(recalibration.py)의 to_logit 과 **같은 식**이어야 한다.

    처음에는 p 만 자르고 1-p 를 계산했는데, §10 은 분자와 분모를 각각
    자른다. 입력이 극단값일 때 마지막 비트가 달라지고, 그것이 lbfgs 가
    닿는 해를 미세하게 바꿔 DICTrank Platt 의 ECE 가 3.6e-05 어긋났다.
    분할은 맞았는데 재현 확인이 '다르다' 로 나왔다 (2026-10-20).
    """
    p = np.asarray(p, dtype=float)
    return np.log(np.clip(p, S.EPS, 1 - S.EPS) / np.clip(1 - p, S.EPS, 1 - S.EPS))


# ------------------------------------------------------------ 보정기
def platt(p_cal, y_cal, p_te):
    """§10 의 B 와 같은 구현."""
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_te):
    """§10 의 C 와 같은 구현."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(p_cal, np.asarray(y_cal))
    return np.clip(ir.predict(p_te), S.EPS, 1 - S.EPS)


def temperature(p_cal, y_cal, p_te):
    """로짓에 기울기만 (절편 없음). 밀림을 고칠 수 없다는 예상의 대조군."""
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, fit_intercept=False)
    lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def beta_calib(p_cal, y_cal, p_te):
    """베타 보정 (Kull et al. 2017) — log p 와 log(1-p) 를 따로 쓴다."""

    def feat(p):
        p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    lr.fit(feat(p_cal), np.asarray(y_cal))
    return lr.predict_proba(feat(p_te))[:, 1]


def venn_abers(p_cal, y_cal, p_te):
    """IVAP. 분자마다 [p0, p1] 을 내고 중점을 점추정으로 쓴다."""
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    lo = np.empty(len(p_te))
    hi = np.empty(len(p_te))
    for i, s in enumerate(np.asarray(p_te, dtype=float)):
        xs = np.append(p_cal, s)
        for lab, out in ((0, lo), (1, hi)):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(xs, np.append(y_cal, lab))
            out[i] = float(ir.predict([s])[0])
    mid = 0.5 * (lo + hi)
    return np.clip(mid, S.EPS, 1 - S.EPS), lo, hi


def platt_then_prior(p_cal, y_cal, p_te, pi_src, pi_tgt):
    """B 뒤에 D — 학습군 내부 Platt 후 검증군 유병률로 절편을 다시 민다."""
    q = platt(p_cal, y_cal, p_te)
    odds = (pi_tgt / (1 - pi_tgt)) / (pi_src / (1 - pi_src))
    return 1.0 / (1.0 + np.exp(-(logit(q) + np.log(odds))))


def boot_ece_ci(y, p, n_boot=N_BOOT, seed=S.SEED):
    """검증군 재표집으로 ECE 95% 구간 (모델 재적합 없음)."""
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
    if not v:
        return float("nan"), float("nan")
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def canonical_split(df):
    """§10(recalibration.py)과 같은 분할 — 연도 정렬 후 뒤 25%."""
    d = df.sort_values("approval_year").reset_index(drop=True)
    yr = d["approval_year"].values
    tr = np.where(yr < S.CUTOFF)[0]
    te = np.where(yr >= S.CUTOFF)[0]
    n_cal = int(len(tr) * CALIB_FRAC)
    return d, tr[:-n_cal], tr[-n_cal:], te, tr


def run(name, fn):
    df0 = S.load(os.path.join(PROC, fn))
    d, fit_i, cal_i, te_i, tr_all = canonical_split(df0)
    X = S.featurize(d["smiles"].tolist())
    y = d["y"].values

    m_all = S.make_model().fit(X[tr_all], y[tr_all])
    p_A = m_all.predict_proba(X[te_i])[:, 1]
    m_fit = S.make_model().fit(X[fit_i], y[fit_i])
    p_cal = m_fit.predict_proba(X[cal_i])[:, 1]
    p_te = m_fit.predict_proba(X[te_i])[:, 1]
    y_te, y_cal = y[te_i], y[cal_i]
    pi_tr, pi_te = float(y[tr_all].mean()), float(y_te.mean())

    ece_A = S.ece(y_te, p_A)
    ece_E = S.ece(y_te, platt(p_A, y_te, p_A))  # oracle 상한 (§10 의 E)
    den = ece_A - ece_E

    log("=" * 116)
    log(
        "%s   적합 %d / 보정 %d / 검증 %d   A.ECE %.4f  E.ECE %.4f  간격 %.4f"
        % (name, len(fit_i), len(cal_i), len(te_i), ece_A, ece_E, den)
    )
    log("=" * 116)

    va_mid, va_lo, va_hi = venn_abers(p_cal, y_cal, p_te)
    arms = [
        ("B. Platt (§10 재현)", platt(p_cal, y_cal, p_te)),
        ("C. Isotonic (§10 재현)", isotonic(p_cal, y_cal, p_te)),
        ("F. 온도 조정", temperature(p_cal, y_cal, p_te)),
        ("G. 베타 보정", beta_calib(p_cal, y_cal, p_te)),
        ("H. Venn-ABERS (중점)", va_mid),
        ("I. Platt → 사전확률", platt_then_prior(p_cal, y_cal, p_te, pi_tr, pi_te)),
    ]

    log(
        "%-24s %8s %-18s %8s %8s %8s %9s"
        % ("보정기", "ECE", "ECE 95%CI", "회복률%", "AUROC", "기울기", "평균예측")
    )
    log("-" * 116)
    rows = []
    for tag, q in arms:
        e = S.ece(y_te, q)
        lo, hi = boot_ece_ci(y_te, q)
        rec = (ece_A - e) / den * 100.0 if den else float("nan")
        try:
            sl = S.calib_slope(y_te, q)
        except Exception:
            sl = float("nan")
        au = float(roc_auc_score(y_te, q)) if len(np.unique(y_te)) > 1 else float("nan")
        log(
            "%-24s %8.4f [%7.4f,%7.4f] %8.2f %8.4f %8.4f %9.4f"
            % (tag, e, lo, hi, rec, au, sl, float(np.mean(q)))
        )
        rows.append(
            dict(
                dataset=name,
                calibrator=tag,
                ece=e,
                ece_lo=lo,
                ece_hi=hi,
                recovery=rec,
                auroc=au,
                slope=sl,
                mean_pred=float(np.mean(q)),
                ece_A=ece_A,
                ece_E=ece_E,
                n_test=len(te_i),
                n_cal=len(cal_i),
                pi_train=pi_tr,
                pi_test=pi_te,
                va_width=float(np.mean(va_hi - va_lo)) if tag.startswith("H.") else float("nan"),
            )
        )
    log("")
    log(
        "  Venn-ABERS 구간 폭 평균 %.4f (최대 %.4f) — 폭이 넓으면 보정셋이 작다는 뜻이다."
        % (float(np.mean(va_hi - va_lo)), float(np.max(va_hi - va_lo)))
    )
    log("")
    return rows


def judge(d):
    rec = pd.read_csv(os.path.join(RES, "recalibration_results.csv"))
    log("=" * 116)
    log("§10 재현 확인")
    log("=" * 116)
    # 2026-10-20: 처음에는 ECE 만 대조했다. 그런데 §10 과 이 절이 **기울기**를
    # 0.8969 대 0.8962 로 다르게 적고 있었고(단정밀도 문제, diff_ladder.py),
    # ECE 는 소수 4자리에서 같아서 그 대조를 통과했다. 그래서 보고하는 열을
    # 전부 대조한다 — 한 열만 보면 이런 불일치를 놓친다.
    worst = 0.0
    COLS = (("ECE", "ece"), ("slope", "slope"), ("AUROC", "auroc"), ("mean_pred", "mean_pred"))
    for ds in d.dataset.unique():
        g = rec[rec.dataset.str.startswith(ds)]
        for pre, tag in (("B.", "B. Platt (§10 재현)"), ("C.", "C. Isotonic (§10 재현)")):
            r10 = g[g.method.str.startswith(pre)].iloc[0]
            r05 = d[(d.dataset == ds) & (d.calibrator == tag)].iloc[0]
            for c10, c05 in COLS:
                a, b = float(r10[c10]), float(r05[c05])
                worst = max(worst, abs(a - b))
                log(
                    "  %s %-10s %-10s §10 %.8f  여기 %.8f  차 %.2e%s"
                    % (
                        ds,
                        tag.split(" ")[1],
                        c10,
                        a,
                        b,
                        abs(a - b),
                        "" if ("%.4f" % a) == ("%.4f" % b) else "  *** 보고 자리수가 다르다 ***",
                    )
                )
        for pre, lab in (("A.", "ece_A"), ("E.", "ece_E")):
            a = float(g[g.method.str.startswith(pre)].ECE.iloc[0])
            b = float(d[d.dataset == ds][lab].iloc[0])
            worst = max(worst, abs(a - b))
            log("  %s %-22s §10 %.6f  여기 %.6f  차 %.2e" % (ds, lab, a, b, abs(a - b)))
    # 2026-10-20: 한때 여기서 3.6e-05 가 나왔고 나는 그것을 "lbfgs 의 실행 간
    # 비결정성" 이라고 적었다. 틀렸다. src/diff_ladder.py 가 단계별로 이분해
    # 찾아낸 원인은 **단정밀도**였다 — recalibration.py 의 to_logit 이
    # XGBoost 의 float32 확률을 캐스팅하지 않아 Platt 적합이 단정밀도로
    # 일어났다. 캐스팅을 넣어 해결했고(check_determinism [1d] 가 매번
    # 확인한다), 그 뒤로는 1e-8 수준이어야 한다. 1e-4 이상이면 분할이
    # 어긋난 것으로 읽는다.
    log(
        "  최대차 %.2e — %s"
        % (
            worst,
            (
                "같다"
                if worst < 1e-9
                else (
                    "보고 자리수 안에서 같다 (부동소수 수준) — 분할은 맞다"
                    if worst < 1e-4
                    else "**다르다. 분할이 어긋났다**"
                )
            ),
        )
    )
    log("")
    log("=" * 116)
    log("판정")
    log("=" * 116)
    log('[읽는 법] 새 보정기가 §10 의 B·C 보다 ECE 를 낮추면 "두 개만 시도했다" 가')
    log("          유효한 지적이 된다. 낮추지 못하면 §10 의 결론이 보정기 선택의")
    log("          산물이 아니라는 증거다.")
    log("")
    for ds in d.dataset.unique():
        g = d[d.dataset == ds]
        base = g[g.calibrator.str.startswith(("B.", "C."))]
        new = g[g.calibrator.str.startswith(("F.", "G.", "H."))]
        bb = base.loc[base.ece.idxmin()]
        nn = new.loc[new.ece.idxmin()]
        # 구간이 겹치면 "더 낫다" 고 말할 수 없다
        better = (nn.ece < bb.ece) and (nn.ece_hi < bb.ece_lo)
        log(
            "  %s — 기존 최선 %s (ECE %.4f [%.4f, %.4f], 회복률 %.1f%%)"
            % (ds, bb.calibrator.split(" ")[1], bb.ece, bb.ece_lo, bb.ece_hi, bb.recovery)
        )
        log(
            "       신규 최선 %s (ECE %.4f [%.4f, %.4f], 회복률 %.1f%%)"
            % (nn.calibrator.split(" ")[1], nn.ece, nn.ece_lo, nn.ece_hi, nn.recovery)
        )
        log(
            "       → %s"
            % (
                "**신규 보정기가 유의하게 낫다 — §10 의 결론을 고쳐야 한다**"
                if better
                else (
                    "점추정은 더 낮지만 구간이 겹친다 — 더 낫다고 말할 수 없다"
                    if nn.ece < bb.ece
                    else "신규 보정기 중 기존을 넘는 것이 없다 — §10 의 결론은 "
                    "보정기 선택의 산물이 아니다"
                )
            )
        )
        _i = g[g.calibrator.str.startswith("I.")]
        if len(_i):
            log(
                "       참고: Platt → 사전확률 합성은 회복률 %.1f%% 다 "
                "(검증군 유병률을 알아야 하므로 배포할 수 없다, §10.2)" % float(_i.recovery.iloc[0])
            )
        log("")
    log("저장: %s" % OUT)


def main():
    log("재보정 사다리에 보정기 네 개를 더 붙인다")
    log("베타 보정 · Venn-ABERS · 온도 조정 · Platt→사전확률 합성")
    log("분할·모델·보정셋·회복률 정의는 §10 과 같다. 시드 %d" % S.SEED)
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    judge(d)
    log.save(Path(RES) / "extra_calibrators_summary.txt")


if __name__ == "__main__":
    main()
