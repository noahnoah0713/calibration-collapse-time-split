"""
견고성 검증: 모델·표현·절단연도에 걸친 일관성 확인

목적: 시간분할 성능 저하와 보정 붕괴가 특정 모델(XGBoost)이나
      특정 표현(ECFP4)의 산물이 아님을 확인한다.

검증 축:
  1) 모델      XGBoost, LightGBM, RandomForest, LogisticRegression
  2) 표현      ECFP4+desc, ECFP4 단독, MACCS, RDKit 기술자 단독
  3) 절단연도  2005, 2010, 2015

사용법:
    python src/robustness.py

출력:
    results/robustness_models.csv       모델x표현 조합별 결과
    results/robustness_cutoff.csv       절단연도별 결과
    results/robustness_summary.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 42
N_RAND = 15
N_BINS = 10
EPS = 1e-6

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


# ------------------------------------------------------------- 표현
# 주 분석 스크립트(temporal_split.py, calibration.py 등)와 동일한 기술자 6종.
# 이전 판은 4종(NumAromaticRings, FractionCSP3, HeavyAtomCount, RingCount)이
# 더 들어 있어 같은 조건인데도 주 분석과 AUROC가 어긋났음 -> 6종으로 통일.
DESC_FN = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
]


# 표현 정의는 _shared.grid_reps() 한 곳에만 있다 (2026-10-01 통합).
# 예전에는 이 파일과 grid_calibration.py 가 각자 정의해, 값은 같지만
# dtype 이 달랐다 (이쪽 float32 / 저쪽 int8+float64). 트리 모델은 같은
# 결과를 냈지만 LogisticRegression 과 LightGBM 이 어긋나, §9.1 과 §9.5 의
# 같은 칸이 32칸 중 12칸에서 달랐다. dtype 은 주 분석과 같은 쪽으로 맞췄다.
REPS = _shared.grid_reps()
rep_ecfp_desc = REPS["ECFP4+desc"]


# ------------------------------------------------------------- 모델
def models():
    """모델 정의는 _shared.grid_models() 한 곳에만 있다 (2026-09-30 통합).
    예전에는 이 파일과 grid_calibration.py 가 각자 정의해, 같은 이름의 모델이
    RandomForest min_samples_leaf / LightGBM num_leaves·max_depth /
    LogisticRegression max_iter 에서 달랐다. 그 결과 §9.1 과 §9.5 의
    '32조합' 이 실제로는 다른 격자였다."""
    return _shared.grid_models()


# ------------------------------------------------------------- 평가
def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        hi = p <= 1 if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return tot


def calib_slope(y, p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 다. 캐스팅하지
    # 않으면 로짓과 로지스틱 적합이 단정밀도로 일어난다
    # (src/diff_ladder.py 단계 10). float64 로 통일한다.
    p = np.asarray(p, dtype=np.float64)
    lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y, p),
        "ECE": ece(y, p),
        "slope": calib_slope(y, p),
        "Brier": brier_score_loss(y, p),
        "BalAcc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def rand_baseline(X, y, n_test, mk):
    rows = []
    for i in range(N_RAND):
        r = np.random.default_rng(SEED + i)
        idx = r.permutation(len(y))
        ti, ri = idx[:n_test], idx[n_test:]
        if len(np.unique(y[ti])) < 2:
            continue
        m = mk().fit(X[ri], y[ri])
        rows.append(metrics(y[ti], m.predict_proba(X[ti])[:, 1]))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ------------------------------------------------------------- main
def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def run_models(name, df, cutoff=2010):
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    y = df["y"].values
    tr = df["approval_year"].values < cutoff
    te = ~tr

    log("=" * 74)
    log(f"{name} - 모델x표현 조합별 검증 (절단 {cutoff})")
    log("=" * 74)
    log(f"학습 {tr.sum()} / 검증 {te.sum()}")
    log()
    log(
        f"{'표현':13s} {'모델':13s} {'무작위':>8s} {'시간분할':>8s} "
        f"{'ΔAUROC':>8s} {'ECE무작위':>9s} {'ECE시간':>8s} {'ECE배율':>7s}"
    )
    log("-" * 82)

    rows = []
    for rname, rfn in REPS.items():
        X = rfn(mols)
        for mname, mk in models().items():
            try:
                m = mk().fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                t = metrics(y[te], p)
                r = rand_baseline(X, y, int(te.sum()), mk)
                ratio = t["ECE"] / r["ECE"] if r["ECE"] > 0 else np.nan
                rows.append(
                    {
                        "dataset": name,
                        "rep": rname,
                        "model": mname,
                        "auroc_random": r["AUROC"],
                        "auroc_temporal": t["AUROC"],
                        "d_auroc": r["AUROC"] - t["AUROC"],
                        "ece_random": r["ECE"],
                        "ece_temporal": t["ECE"],
                        "ece_ratio": ratio,
                        "slope_temporal": t["slope"],
                        "mcc_temporal": t["MCC"],
                    }
                )
                log(
                    f"{rname:13s} {mname:13s} {r['AUROC']:8.4f} {t['AUROC']:8.4f} "
                    f"{r['AUROC']-t['AUROC']:+8.4f} {r['ECE']:9.4f} "
                    f"{t['ECE']:8.4f} {ratio:7.2f}"
                )
            except Exception as e:
                log(f"{rname:13s} {mname:13s} 실패: {str(e)[:40]}")
    log()
    return pd.DataFrame(rows)


def run_cutoffs(name, df):
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    X = rep_ecfp_desc(mols)
    y = df["y"].values
    yr = df["approval_year"].values

    log("=" * 74)
    log(f"{name} - 절단연도 민감도 (ECFP4+desc, XGBoost)")
    log("=" * 74)
    log(
        f"{'절단':>6s} {'학습n':>7s} {'검증n':>7s} {'검증양성':>8s} "
        f"{'무작위':>8s} {'시간분할':>8s} {'ΔAUROC':>8s} {'ECE배율':>7s}"
    )
    log("-" * 70)

    rows = []
    mk = models()["XGBoost"]
    for cut in [2005, 2010, 2015]:
        tr = yr < cut
        te = ~tr
        if te.sum() < 40 or len(np.unique(y[te])) < 2 or tr.sum() < 100:
            log(f"{cut:6d} 표본 부족 (학습 {tr.sum()}, 검증 {te.sum()})")
            continue
        m = mk().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        t = metrics(y[te], p)
        r = rand_baseline(X, y, int(te.sum()), mk)
        ratio = t["ECE"] / r["ECE"] if r["ECE"] > 0 else np.nan
        rows.append(
            {
                "dataset": name,
                "cutoff": cut,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "test_pos_rate": float(y[te].mean()),
                "auroc_random": r["AUROC"],
                "auroc_temporal": t["AUROC"],
                "d_auroc": r["AUROC"] - t["AUROC"],
                "ece_random": r["ECE"],
                "ece_temporal": t["ECE"],
                "ece_ratio": ratio,
                "slope": t["slope"],
            }
        )
        log(
            f"{cut:6d} {tr.sum():7d} {te.sum():7d} {y[te].mean():7.1%} "
            f"{r['AUROC']:8.4f} {t['AUROC']:8.4f} "
            f"{r['AUROC']-t['AUROC']:+8.4f} {ratio:7.2f}"
        )
    log()
    return pd.DataFrame(rows)


def main():
    mr, cr = [], []
    for nm, p in [
        ("DILIrank", PROC / "dilirank2_with_year.csv"),
        ("DICTrank", PROC / "dictrank_processed.csv"),
    ]:
        if not p.exists():
            continue
        df = load(p)
        mr.append(run_models(nm, df))
        cr.append(run_cutoffs(nm, df))

    if not mr:
        log("데이터 파일 없음")
        return

    M = pd.concat(mr, ignore_index=True)
    C = pd.concat(cr, ignore_index=True)
    M.to_csv(RES / "robustness_models.csv", index=False)
    C.to_csv(RES / "robustness_cutoff.csv", index=False)

    log("=" * 74)
    log("일관성 판정")
    log("=" * 74)
    for ds in M["dataset"].unique():
        s = M[M.dataset == ds]
        n_pos = (s.d_auroc > 0).sum()
        n_ece = (s.ece_ratio > 1).sum()
        log(f"\n{ds}  (조합 {len(s)}개)")
        log(
            f"  ΔAUROC > 0 인 조합: {n_pos}/{len(s)} "
            f"(중앙값 {s.d_auroc.median():+.4f}, "
            f"범위 {s.d_auroc.min():+.4f}~{s.d_auroc.max():+.4f})"
        )
        log(
            f"  ECE 악화 조합:      {n_ece}/{len(s)} "
            f"(중앙값 {s.ece_ratio.median():.2f}배, "
            f"범위 {s.ece_ratio.min():.2f}~{s.ece_ratio.max():.2f}배)"
        )
        log(f"  보정기울기 중앙값:  {s.slope_temporal.median():.3f} (1이 이상적)")
        # 판정
        if n_pos >= len(s) * 0.8 and n_ece >= len(s) * 0.8:
            log("  => 모델·표현에 무관하게 일관된 패턴")
        elif n_pos >= len(s) * 0.6:
            log("  => 대체로 일관하나 일부 조합에서 예외")
        else:
            log("  => 일관성 부족 - 특정 조합에 의존할 가능성")

    if len(C):
        log("\n절단연도 민감도")
        for ds in C["dataset"].unique():
            s = C[C.dataset == ds]
            log(
                f"  {ds}: ΔAUROC "
                f"{', '.join(f'{r.cutoff}년 {r.d_auroc:+.3f}' for _, r in s.iterrows())}"
            )

    (RES / "robustness_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'robustness_models.csv'}")
    log(f"      {RES/'robustness_cutoff.csv'}")
    log(f"      {RES/'robustness_summary.txt'}")


if __name__ == "__main__":
    main()
