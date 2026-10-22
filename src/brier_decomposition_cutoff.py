"""
Brier score Murphy 분해 - 절단연도 민감도 확장

목적:
    §5(robustness.py)에서 확인한 "AUROC 격차는 절단연도에 따라 불안정
    (0.010~0.098)하지만 ECE 배율은 안정적(1.8~3.3배)"이라는 발견을,
    Brier score 분해(보정 기여도 %)에도 동일하게 적용하여 절단연도에
    걸쳐 안정적인지 확인한다.

    brier_decomposition.py와 동일한 방법론을 CUTOFF=2005, 2010, 2015
    세 지점에 반복 적용한다. 새로운 방법론이 아니라 기존 분석의
    반복이므로 별도 파일로 분리했다.

사용법:
    python src/brier_decomposition_cutoff.py

출력:
    results/brier_decomposition_cutoff.csv
    results/brier_decomposition_cutoff_summary.txt
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

SEED = 42
CUTOFFS = [2005, 2010, 2015]
N_BINS = 10
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


def bin_table(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        hi = (p <= 1) if i == n_bins - 1 else (p < edges[i + 1])
        m = (p >= edges[i]) & hi
        if m.sum() > 0:
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return rows


def murphy_decomposition(bins):
    N = sum(n for _, _, n in bins)
    obar = sum(o * n for _, o, n in bins) / N
    uncertainty = obar * (1 - obar)
    resolution = sum(n * (o - obar) ** 2 for _, o, n in bins) / N
    reliability = sum(n * (f - o) ** 2 for f, o, n in bins) / N
    bs = uncertainty - resolution + reliability
    return dict(
        N=N,
        obar=obar,
        uncertainty=uncertainty,
        resolution=resolution,
        reliability=reliability,
        brier_approx=bs,
    )


def get_temporal_probs(df, cutoff, X, y):
    tr = df["approval_year"].values < cutoff
    te = ~tr
    if tr.sum() < 30 or te.sum() < 30:
        return None
    m = make_model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    return y[te], p


def get_random_splits(X, y, n_test):
    """무작위 분할 N_REPEATS회. 각 회차의 (y, p)를 그대로 반환.

    brier_decomposition.py와 동일하게, 풀링하지 않고 회차별로 분해한 뒤
    평균낸다. 시간분할과 같은 표본 크기에서 재야 reliability의
    소표본 상향 편향이 양쪽에서 같은 방향으로 작용한다.
    """
    out = []
    rng = np.random.default_rng(SEED)
    for i in range(N_REPEATS):
        idx = rng.permutation(len(y))
        te_idx, tr_idx = idx[:n_test], idx[n_test:]
        m = make_model().fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te_idx])[:, 1]
        out.append((y[te_idx], p))
    return out


def run_dataset(name, path):
    if not path.exists():
        log(f"  {name} 건너뜀: {path} 없음")
        return []

    df = load(path)
    X = featurize(df["smiles"].tolist())
    y = df["y"].values

    rows = []
    log(f"\n{'='*72}\n{name}\n{'='*72}")
    log(
        f"{'절단':>6s}{'검증n':>8s}{'무작위%(resolution)':>20s}"
        f"{'무작위%(reliability)':>20s}{'시간%(resolution)':>18s}"
        f"{'시간%(reliability)':>18s}"
    )

    for cutoff in CUTOFFS:
        res_t = get_temporal_probs(df, cutoff, X, y)
        if res_t is None:
            log(f"  절단 {cutoff}: 표본 부족, 건너뜀")
            continue
        y_t, p_t = res_t
        splits = get_random_splits(X, y, n_test=len(y_t))

        dt = murphy_decomposition(bin_table(y_t, p_t))
        _per = [murphy_decomposition(bin_table(yy, pp)) for yy, pp in splits]
        dr = {
            k: float(np.mean([d[k] for d in _per]))
            for k in ("N", "obar", "uncertainty", "resolution", "reliability", "brier_approx")
        }

        d_resolution = dr["resolution"] - dt["resolution"]
        d_reliability = dt["reliability"] - dr["reliability"]
        total = d_resolution + d_reliability
        pct_res = 100 * d_resolution / total if total > 0 else np.nan
        pct_rel = 100 * d_reliability / total if total > 0 else np.nan

        log(f"{cutoff:>6d}{len(y_t):>8d}{pct_res:>20.1f}{pct_rel:>20.1f}" f"{'':>18s}{'':>18s}")
        log(
            f"  -> Brier 악화 중 보정(reliability) 기여 {pct_rel:.1f}%, "
            f"판별력(resolution) 기여 {pct_res:.1f}% "
            f"(비율 {d_reliability/d_resolution if d_resolution>0 else float('inf'):.2f}배)"
        )

        rows.append(
            {
                "dataset": name,
                "cutoff": cutoff,
                "n_test": len(y_t),
                "resolution_random_mean": dr["resolution"],
                "resolution_temporal": dt["resolution"],
                "reliability_random_mean": dr["reliability"],
                "reliability_temporal": dt["reliability"],
                "brier_random_mean": dr["brier_approx"],
                "brier_temporal": dt["brier_approx"],
                "pct_from_resolution": pct_res,
                "pct_from_reliability": pct_rel,
            }
        )

    if rows:
        pcts = [r["pct_from_reliability"] for r in rows]
        log(
            f"\n  절단연도별 보정 기여도 범위: {min(pcts):.1f}% ~ {max(pcts):.1f}% "
            f"(변동폭 {max(pcts)-min(pcts):.1f}%p)"
        )

    return rows


def main():
    log("Brier 분해 절단연도 민감도: 보정 기여도가 절단연도에 걸쳐 안정적인지 확인")
    log("무작위 기준 = 분할별 분해 후 평균 (brier_decomposition.py의 주 분석과 동일)\n")

    all_rows = []
    all_rows += run_dataset("DILIrank", PROC / "dilirank2_with_year.csv")
    all_rows += run_dataset("DICTrank", PROC / "dictrank_processed.csv")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(RES / "brier_decomposition_cutoff.csv", index=False)
    (RES / "brier_decomposition_cutoff_summary.txt").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )
    log(f"\n저장: {RES/'brier_decomposition_cutoff.csv'}")
    log(f"      {RES/'brier_decomposition_cutoff_summary.txt'}")


if __name__ == "__main__":
    main()
