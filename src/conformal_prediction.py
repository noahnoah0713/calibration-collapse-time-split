"""
Split Conformal Prediction 커버리지 검증

목적:
    Platt/Isotonic 재보정으로 확인한 "보정 붕괴"가 특정 재보정 기법에
    국한된 현상이 아니라, 분포 가정이 없는(distribution-free) 최신
    불확실성 정량화 프레임워크에서도 동일하게 나타나는지 확인한다.

원리:
    Split conformal prediction의 커버리지 보장(marginal coverage >=
    1-alpha)은 "보정셋과 검증셋이 교환가능(exchangeable)하다"는 가정에
    의존한다. 무작위 분할에서는 이 가정이 성립하지만, 시간분할에서는
    검증군이 미래 시점의 약물이므로 교환가능성이 위반된다. 따라서
    이론적으로 시간분할에서는 명목 커버리지(1-alpha)가 깨질 것으로
    예상된다.

    또한 marginal coverage(전체 커버리지)뿐 아니라 클래스별(양성/음성)
    커버리지를 따로 보고한다. 어느 클래스에서 커버리지가 더 크게
    무너지는지를 보면, 모델이 어느 방향으로 편향되는지(과대예측 vs
    과소예측)가 별도의 가설검정 없이 자연스럽게 드러난다.

방법:
    1. 시간분할: 학습군(승인연도 < CUTOFF) 중 최근 CALIB_FRAC 비율을
       보정셋으로, 나머지를 적합셋으로 사용. 검증군(승인연도 >= CUTOFF)
       에서 커버리지 평가.
    2. 무작위 분할 대조군: 동일한 적합/보정/검증 크기 비율을 무작위로
       N_REPEATS회 반복하여 평균 커버리지 산출 (교환가능성이 성립하는
       경우의 기준선).

사용법:
    python src/conformal_prediction.py

출력:
    results/conformal_prediction.csv
    results/conformal_prediction_summary.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
CALIB_FRAC = 0.25  # recalibration.py와 동일한 비율 (최근 25%를 보정셋으로)
ALPHAS = [0.10, 0.20]  # 목표 커버리지 90%, 80%
N_REPEATS = 20

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def featurize(smiles):
    fps, desc = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=np.float32))
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


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def conformal_eval(p_calib, y_calib, p_test, y_test, alpha):
    """split conformal: nonconformity = 1 - p_hat(true class)."""
    s_calib = 1 - np.where(y_calib == 1, p_calib, 1 - p_calib)
    n = len(s_calib)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(s_calib, q_level, method="higher")

    # 예측집합: 클래스 c가 집합에 포함 <=> 1 - p_hat(c) <= qhat
    p1 = p_test
    p0 = 1 - p_test
    in_set_1 = (1 - p1) <= qhat
    in_set_0 = (1 - p0) <= qhat
    set_size = in_set_1.astype(int) + in_set_0.astype(int)

    covered = np.where(y_test == 1, in_set_1, in_set_0)
    marginal_cov = covered.mean()
    pos_mask, neg_mask = (y_test == 1), (y_test == 0)
    cov_pos = covered[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
    cov_neg = covered[neg_mask].mean() if neg_mask.sum() > 0 else np.nan

    return dict(
        qhat=qhat,
        marginal_coverage=marginal_cov,
        coverage_positive=cov_pos,
        coverage_negative=cov_neg,
        mean_set_size=set_size.mean(),
        empty_set_rate=(set_size == 0).mean(),
    )


def temporal_run(df):
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values

    train_mask = yr < CUTOFF
    test_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    train_idx_sorted = train_idx[np.argsort(yr[train_idx])]
    n_calib = int(len(train_idx_sorted) * CALIB_FRAC)
    calib_idx = train_idx_sorted[-n_calib:]  # 학습군 내 가장 최근 25%
    fit_idx = train_idx_sorted[:-n_calib]

    m = make_model().fit(X[fit_idx], y[fit_idx])
    p_calib = m.predict_proba(X[calib_idx])[:, 1]
    p_test = m.predict_proba(X[test_mask])[:, 1]

    return y[calib_idx], p_calib, y[test_mask], p_test


def random_run(df, n_fit, n_calib, n_test, seed):
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    fit_idx = idx[:n_fit]
    calib_idx = idx[n_fit : n_fit + n_calib]
    test_idx = idx[n_fit + n_calib : n_fit + n_calib + n_test]

    m = make_model().fit(X[fit_idx], y[fit_idx])
    p_calib = m.predict_proba(X[calib_idx])[:, 1]
    p_test = m.predict_proba(X[test_idx])[:, 1]
    return y[calib_idx], p_calib, y[test_idx], p_test


def run_dataset(name, path):
    if not path.exists():
        log(f"  {name} 건너뜀: {path} 없음")
        return []

    df = load(path)
    y_calib_t, p_calib_t, y_test_t, p_test_t = temporal_run(df)
    n_fit = int(len(df) * (1 - CALIB_FRAC)) - len(y_test_t)  # 근사치, 아래서 실제 크기 사용
    n_calib = len(y_calib_t)
    n_test = len(y_test_t)
    # 실제 fit 크기: 전체 - calib - test
    n_fit = len(df) - n_calib - n_test

    rows = []
    log(f"\n{'='*72}\n{name}  (calib n={n_calib}, test n={n_test})\n{'='*72}")

    for alpha in ALPHAS:
        r_t = conformal_eval(p_calib_t, y_calib_t, p_test_t, y_test_t, alpha)

        # 무작위 대조군 N_REPEATS회 반복 평균
        accum = []
        for i in range(N_REPEATS):
            yc, pc, yt, pt = random_run(df, n_fit, n_calib, n_test, seed=SEED + i)
            accum.append(conformal_eval(pc, yc, pt, yt, alpha))
        r_r = {k: np.nanmean([a[k] for a in accum]) for k in r_t}

        target = 1 - alpha
        log(f"\n  목표 커버리지 {target:.0%} (alpha={alpha})")
        log(f"    {'':22s}{'무작위(평균)':>14s}{'시간분할':>12s}")
        for k, label in [
            ("marginal_coverage", "전체 커버리지"),
            ("coverage_positive", "양성 클래스 커버리지"),
            ("coverage_negative", "음성 클래스 커버리지"),
            ("mean_set_size", "평균 예측집합 크기"),
            ("empty_set_rate", "공집합 비율"),
        ]:
            log(f"    {label:22s}{r_r[k]:>14.3f}{r_t[k]:>12.3f}")

        gap_pos = target - r_t["coverage_positive"]
        gap_neg = target - r_t["coverage_negative"]
        direction = (
            "양성(간독성/심장독성 있음) 과소보장" if gap_pos > gap_neg else "음성(안전) 과소보장"
        )
        log(f"    -> 시간분할에서 커버리지 부족이 더 큰 쪽: {direction}")

        rows.append(
            {
                "dataset": name,
                "alpha": alpha,
                "target_coverage": target,
                "marginal_coverage_random": r_r["marginal_coverage"],
                "marginal_coverage_temporal": r_t["marginal_coverage"],
                "coverage_positive_random": r_r["coverage_positive"],
                "coverage_positive_temporal": r_t["coverage_positive"],
                "coverage_negative_random": r_r["coverage_negative"],
                "coverage_negative_temporal": r_t["coverage_negative"],
                "mean_set_size_random": r_r["mean_set_size"],
                "mean_set_size_temporal": r_t["mean_set_size"],
            }
        )

    return rows


def main():
    log("Split Conformal Prediction: 시간분할에서 분포무관 커버리지 보장 붕괴 검증")

    all_rows = []
    all_rows += run_dataset("DILIrank", PROC / "dilirank2_with_year.csv")
    all_rows += run_dataset("DICTrank", PROC / "dictrank_processed.csv")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(RES / "conformal_prediction.csv", index=False)
    (RES / "conformal_prediction_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'conformal_prediction.csv'}")
    log(f"      {RES/'conformal_prediction_summary.txt'}")


if __name__ == "__main__":
    main()
