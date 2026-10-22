"""
시간분할 보정 붕괴에 대한 재보정(recalibration) 실험

문제: 시간분할에서 ECE가 2~3.4배 악화되고 보정 기울기가 0.24~0.29로 붕괴
질문: 재보정으로 회복 가능한가? 회복해도 AUROC 격차는 남는가?

전략 (검증군 라벨을 쓰지 않는 것이 핵심):
  A. 무보정                  기준선
  B. Platt (학습군 내부)      학습군을 시간순으로 나눠 뒷부분을 보정셋으로
  C. Isotonic (학습군 내부)   위와 동일, 등위회귀
  D. Platt (사전확률만)       검증군 유병률만 안다고 가정하고 절편 보정
  E. Oracle (검증군 사용)     상한선 - 실제로는 불가능하나 회복 가능성의 천장

핵심 논점: AUROC는 단조변환에 불변이므로 재보정으로 바뀌지 않는다.
          즉 판별력 격차는 남고 보정만 회복된다면,
          "시간분할의 피해는 보정에 집중되며 부분적으로 교정 가능"이 결론.

사용법:
    python src/recalibration.py

출력:
    results/recalibration_results.csv
    results/recalibration_summary.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
CALIB_FRAC = 0.25  # 학습군 중 최근 25%를 보정셋으로
N_BINS = 10
EPS = 1e-6

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def featurize(smiles):
    fps, desc = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=np.int8))
        desc.append(
            [
                Descriptors.MolWt(m),
                Descriptors.MolLogP(m),
                Descriptors.TPSA(m),
                Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m),
            ]
        )
    return np.hstack([np.array(fps), np.array(desc)])


def make_model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        hi = p <= 1 if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return tot


def calib_line(y, p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 를 돌려준다.
    # 캐스팅하지 않으면 로짓과 그 뒤의 로지스틱 적합이 **단정밀도**로
    # 일어나고, 같은 지표가 어디서 왔는지에 따라 정밀도가 달라진다
    # (§10 Platt 계수가 §10.5 와 2.1e-04 어긋난 원인이다,
    # src/diff_ladder.py 단계 10). float64 로 올려 통일한다.
    p = np.asarray(p, dtype=np.float64)
    lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def assess(y, p, tag):
    pred = (p >= 0.5).astype(int)
    slope, inter = calib_line(y, p)
    return {
        "method": tag,
        "AUROC": roc_auc_score(y, p),
        "Brier": brier_score_loss(y, p),
        "ECE": ece(y, p),
        "slope": slope,
        "intercept": inter,
        "mean_pred": float(p.mean()),
        "BalAcc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def to_logit(p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 를 돌려준다.
    # 캐스팅하지 않으면 로짓과 그 뒤의 로지스틱 적합이 **단정밀도**로
    # 일어나고, 같은 지표가 어디서 왔는지에 따라 정밀도가 달라진다
    # (§10 Platt 계수가 §10.5 와 2.1e-04 어긋난 원인이다,
    # src/diff_ladder.py 단계 10). float64 로 올려 통일한다.
    p = np.asarray(p, dtype=np.float64)
    return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))


def platt(p_cal, y_cal, p_test):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(to_logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(to_logit(p_test).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_test):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(p_cal, y_cal)
    return np.clip(ir.predict(p_test), EPS, 1 - EPS)


def prior_shift(p_test, prior_src, prior_tgt):
    """사전확률만 알 때의 절편 보정 (Saerens et al. 방식의 단순형)."""
    odds = (prior_tgt / (1 - prior_tgt)) / (prior_src / (1 - prior_src))
    lo = to_logit(p_test) + np.log(odds)
    return 1 / (1 + np.exp(-lo))


def run(name, path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.sort_values("approval_year").reset_index(drop=True)

    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values

    tr_all = yr < CUTOFF
    te = ~tr_all

    # 학습군을 시간순으로 나눠 뒷부분(최근)을 보정셋으로 사용
    tr_idx = np.where(tr_all)[0]  # 이미 연도순 정렬됨
    n_cal = int(len(tr_idx) * CALIB_FRAC)
    fit_idx = tr_idx[:-n_cal]
    cal_idx = tr_idx[-n_cal:]

    log("=" * 68)
    log(f"{name}")
    log("=" * 68)
    log(f"  학습(적합) n={len(fit_idx)} (승인 ~{yr[fit_idx].max()})")
    log(f"  보정셋      n={len(cal_idx)} (승인 {yr[cal_idx].min()}~{yr[cal_idx].max()})")
    log(f"  검증군      n={te.sum()} (승인 {CUTOFF}~), 양성 {y[te].mean():.1%}")

    yte = y[te]

    # 전체 학습군으로 학습한 기준 모델
    m_full = make_model().fit(X[tr_all], y[tr_all])
    p_base = m_full.predict_proba(X[te])[:, 1]

    # 보정셋을 뺀 부분으로 학습한 모델 (보정에 사용)
    m_fit = make_model().fit(X[fit_idx], y[fit_idx])
    p_cal = m_fit.predict_proba(X[cal_idx])[:, 1]
    p_te_fit = m_fit.predict_proba(X[te])[:, 1]

    rows = [assess(yte, p_base, "A. 무보정 (전체 학습)")]
    rows.append(assess(yte, p_te_fit, "A'. 무보정 (보정셋 제외 학습)"))

    # B. Platt (학습군 내부 보정셋)
    rows.append(assess(yte, platt(p_cal, y[cal_idx], p_te_fit), "B. Platt (학습군 내부)"))

    # C. Isotonic
    rows.append(assess(yte, isotonic(p_cal, y[cal_idx], p_te_fit), "C. Isotonic (학습군 내부)"))

    # D. 사전확률 보정 (검증군 유병률만 안다고 가정)
    rows.append(assess(yte, prior_shift(p_base, y[tr_all].mean(), yte.mean()), "D. 사전확률 보정"))

    # E. Oracle (검증군 라벨 사용 - 상한선)
    rows.append(assess(yte, platt(p_base, yte, p_base), "E. Oracle Platt (상한선)"))

    out = pd.DataFrame(rows)

    log(f"\n{'방법':32s} {'AUROC':>7s} {'ECE':>7s} {'기울기':>8s} " f"{'절편':>8s} {'MCC':>7s}")
    log("-" * 76)
    for _, r in out.iterrows():
        log(
            f"{r['method']:32s} {r['AUROC']:7.4f} {r['ECE']:7.4f} "
            f"{r['slope']:8.4f} {r['intercept']:8.4f} {r['MCC']:7.4f}"
        )

    base = out.iloc[0]
    log(f"\n[회복률] 무보정 ECE {base['ECE']:.4f} 대비")
    for _, r in out.iloc[2:].iterrows():
        red = (base["ECE"] - r["ECE"]) / base["ECE"] * 100
        log(f"  {r['method']:32s} ECE {r['ECE']:.4f}  ({red:+.0f}%)")

    log("\n[AUROC 불변 확인]")
    log("  Platt/사전확률 보정은 단조변환이므로 AUROC 유지가 정상")
    log("  Isotonic은 동점 처리로 미세하게 달라질 수 있음")

    log("\n[임계값 성능]")
    for _, r in out.iterrows():
        log(f"  {r['method']:32s} BalAcc {r['BalAcc']:.4f}  MCC {r['MCC']:.4f}")

    out.insert(0, "dataset", name)
    log()
    return out


def main():
    res = []
    for nm, p in [
        ("DILIrank (간독성)", PROC / "dilirank2_with_year.csv"),
        ("DICTrank (심장독성)", PROC / "dictrank_processed.csv"),
    ]:
        if p.exists():
            res.append(run(nm, p))

    if not res:
        log("데이터 파일을 찾을 수 없습니다.")
        return

    allr = pd.concat(res, ignore_index=True)
    allr.to_csv(RES / "recalibration_results.csv", index=False)

    log("=" * 68)
    log("종합 판정")
    log("=" * 68)
    for ds in allr["dataset"].unique():
        s = allr[allr.dataset == ds]
        b = s.iloc[0]
        best = s.iloc[2:5]  # B, C, D 중 최선
        bi = best.loc[best.ECE.idxmin()]
        orc = s.iloc[5]
        log(f"\n{ds}")
        log(f"  무보정        ECE {b['ECE']:.4f}, 기울기 {b['slope']:.3f}")
        log(f"  최선 재보정   {bi['method']} -> ECE {bi['ECE']:.4f}, " f"기울기 {bi['slope']:.3f}")
        log(f"  Oracle 상한   ECE {orc['ECE']:.4f}")
        rec = (b["ECE"] - bi["ECE"]) / (b["ECE"] - orc["ECE"] + 1e-9) * 100
        log(f"  -> 회복 가능분의 {np.clip(rec, 0, 100):.0f}% 달성")
        log(
            f"  -> AUROC는 {b['AUROC']:.4f}에서 {bi['AUROC']:.4f}로 "
            f"{'변화 없음' if abs(bi['AUROC']-b['AUROC']) < 0.01 else '변화'} "
            f"(판별력 격차는 재보정으로 해결되지 않음)"
        )

    (RES / "recalibration_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'recalibration_results.csv'}")
    log(f"      {RES/'recalibration_summary.txt'}")


if __name__ == "__main__":
    main()
