"""
[보완 4] 내부 보정셋 비율 민감도

문제:
  recalibration.py 는 CALIB_FRAC = 0.25 (학습군 중 최근 25%) 단일값이다.
  §10 의 결론 — "전향적 제약을 지키는 B·C 는 회복 가능분의 8~10%(DILIrank),
  40~42%(DICTrank)만 회복하고 데이터셋 간 4~5배 차이가 난다" — 가
  이 임의 선택 하나에 걸려 있다. conformal_prediction.py 도 같은 값을 쓴다.

이 스크립트:
  CALIB_FRAC 을 0.15 / 0.20 / 0.25 / 0.33 / 0.40 으로 바꿔
  A / A' / B(Platt) / C(Isotonic) / D(사전확률) / E(Oracle) 의 ECE 와
  회복률을 재계산한다. 회복률 정의는 보고서 §10 과 같다:
      회복률 = (A − 해당방법) / (A − E)

사용법:
    python src/recalibration_sensitivity.py

출력:
    results/recalibration_sensitivity.csv
    results/recalibration_sensitivity_summary.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
FRACS = [0.15, 0.20, 0.25, 0.33, 0.40]
log = S.Log()


def logit(p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 다. 캐스팅하지
    # 않으면 로짓과 로지스틱 적합이 단정밀도로 일어난다
    # (src/diff_ladder.py 단계 10). float64 로 통일한다.
    p = np.clip(np.asarray(p, dtype=np.float64), S.EPS, 1 - S.EPS)
    return np.log(p / (1 - p))


def platt(p_cal, y_cal, p_te):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_te):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_cal, y_cal)
    return ir.predict(p_te)


def prior_shift(p, src_rate, tgt_rate):
    """Saerens 절편 보정: log-odds 를 목표/원천 odds 비로 이동."""
    off = np.log(tgt_rate / (1 - tgt_rate)) - np.log(src_rate / (1 - src_rate))
    z = logit(p) + off
    return 1 / (1 + np.exp(-z))


def run(name, path):
    df = S.load(path)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr, te = S.temporal_masks(df)
    yte = y[te]
    rate_te = float(yte.mean())

    order = np.argsort(df["approval_year"].values[tr], kind="stable")
    tr_idx = np.where(tr)[0][order]

    log("=" * 88)
    log(f"{name}   학습 {len(tr_idx)} / 검증 {int(te.sum())}  " f"검증군 양성 {rate_te:.1%}")
    log("=" * 88)

    # A: 학습군 전체로 학습 (보정셋 비율과 무관)
    mA = S.make_model().fit(X[tr_idx], y[tr_idx])
    pA = mA.predict_proba(X[te])[:, 1]
    eceA, aucA = S.ece(yte, pA), roc_auc_score(yte, pA)
    rate_all = float(y[tr_idx].mean())
    pD = prior_shift(pA, rate_all, rate_te)
    # E: Oracle Platt (검증군 라벨 사용, 상한선)
    pE = platt(pA, yte, pA)
    eceD, eceE = S.ece(yte, pD), S.ece(yte, pE)
    log(f"\n  A  무보정(전체 학습)  ECE {eceA:.4f}  AUROC {aucA:.4f}")
    log(f"  D  사전확률 보정       ECE {eceD:.4f}   (보정셋 비율과 무관)")
    log(f"  E  Oracle Platt 상한   ECE {eceE:.4f}   (보정셋 비율과 무관)")
    denom = eceA - eceE
    log(f"  회복 가능분 (A − E) = {denom:.4f}\n")

    log(
        f"  {'비율':>6s} {'적합n':>6s} {'보정n':>6s} "
        f"{'A′ECE':>8s} {'B ECE':>8s} {'C ECE':>8s} "
        f"{'B회복':>7s} {'C회복':>7s} {'A′AUROC':>8s} {'B AUROC':>8s}"
    )
    log("  " + "-" * 84)

    rows = [
        dict(
            dataset=name,
            calib_frac=np.nan,
            n_fit=len(tr_idx),
            n_calib=0,
            method="A. 무보정 (전체 학습)",
            ece=eceA,
            auroc=aucA,
            recovery=np.nan,
        ),
        dict(
            dataset=name,
            calib_frac=np.nan,
            n_fit=len(tr_idx),
            n_calib=0,
            method="D. 사전확률 보정",
            ece=eceD,
            auroc=aucA,
            recovery=100 * (eceA - eceD) / denom,
        ),
        dict(
            dataset=name,
            calib_frac=np.nan,
            n_fit=len(tr_idx),
            n_calib=0,
            method="E. Oracle Platt (상한)",
            ece=eceE,
            auroc=aucA,
            recovery=100.0,
        ),
    ]

    for f in FRACS:
        n_cal = int(len(tr_idx) * f)
        if n_cal < 30 or len(tr_idx) - n_cal < 100:
            log(f"  {f:6.2f} 표본 부족 (적합 {len(tr_idx)-n_cal}, 보정 {n_cal})")
            continue
        fit_i, cal_i = tr_idx[:-n_cal], tr_idx[-n_cal:]
        if len(np.unique(y[cal_i])) < 2:
            log(f"  {f:6.2f} 보정셋 단일 클래스 - 건너뜀")
            continue
        m = S.make_model().fit(X[fit_i], y[fit_i])
        pAp = m.predict_proba(X[te])[:, 1]
        p_cal = m.predict_proba(X[cal_i])[:, 1]
        pB = platt(p_cal, y[cal_i], pAp)
        pC = isotonic(p_cal, y[cal_i], pAp)
        e = {k: S.ece(yte, v) for k, v in (("Ap", pAp), ("B", pB), ("C", pC))}
        a = {k: roc_auc_score(yte, v) for k, v in (("Ap", pAp), ("B", pB), ("C", pC))}
        recB = 100 * (eceA - e["B"]) / denom
        recC = 100 * (eceA - e["C"]) / denom
        log(
            f"  {f:6.2f} {len(fit_i):6d} {n_cal:6d} "
            f"{e['Ap']:8.4f} {e['B']:8.4f} {e['C']:8.4f} "
            f"{recB:6.1f}% {recC:6.1f}% {a['Ap']:8.4f} {a['B']:8.4f}"
        )
        for k, lab in [
            ("Ap", "A'. 무보정 (보정셋 제외 학습)"),
            ("B", "B. Platt (학습군 내부)"),
            ("C", "C. Isotonic (학습군 내부)"),
        ]:
            rows.append(
                dict(
                    dataset=name,
                    calib_frac=f,
                    n_fit=len(fit_i),
                    n_calib=n_cal,
                    method=lab,
                    ece=e[k],
                    auroc=a[k],
                    recovery=(100 * (eceA - e[k]) / denom) if k != "Ap" else np.nan,
                )
            )
    log("")
    return pd.DataFrame(rows)


def main():
    log("내부 보정셋 비율 민감도 — recalibration.py 의 CALIB_FRAC 스윕")
    log(f"비율 {FRACS}, seed {S.SEED}, 절단 {S.CUTOFF}")
    log("회복률 = (A − 해당방법) / (A − E),  보고서 §10 과 같은 정의\n")
    out = []
    for nm, fn in S.DATASETS:
        p = PROC / fn
        if p.exists():
            out.append(run(nm, p))
    if not out:
        log("데이터 없음")
        return
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "recalibration_sensitivity.csv", index=False)

    log("=" * 88)
    log("판정 — 전향적 방법(B·C)의 회복률이 보정셋 비율에 얼마나 흔들리나")
    log("=" * 88)
    for ds in df.dataset.unique():
        s = df[(df.dataset == ds) & df.calib_frac.notna()]
        log(f"\n{ds}")
        for meth in ["B. Platt (학습군 내부)", "C. Isotonic (학습군 내부)"]:
            v = s[s.method == meth]["recovery"].dropna()
            if not len(v):
                continue
            log(
                f"  {meth:26s} 회복률 {v.min():5.1f}% ~ {v.max():5.1f}% "
                f"(중앙값 {v.median():5.1f}%, 변동폭 {v.max()-v.min():4.1f}%p)"
            )
        d = df[(df.dataset == ds) & (df.method == "D. 사전확률 보정")]["recovery"]
        if len(d):
            log(f"  {'D. 사전확률 보정':26s} 회복률 {d.iloc[0]:5.1f}% " "(보정셋 비율과 무관)")
    log("")
    log("[읽는 법] B·C 의 변동폭이 작으면 §10 결론이 25% 선택에 의존하지 않는다.")
    log("          변동폭이 크면 본문에 범위로 보고하고 25% 를 택한 이유를 밝혀야 한다.")
    log.save(RES / "recalibration_sensitivity_summary.txt")
    print(f"\n저장: {RES/'recalibration_sensitivity.csv'}")
    print(f"      {RES/'recalibration_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
