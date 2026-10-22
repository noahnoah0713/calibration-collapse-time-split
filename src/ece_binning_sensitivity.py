"""
ECE / Brier 분해의 구간(bin) 개수 민감도 검증

목적:
    지금까지 보고한 핵심 수치(ECE, Brier Murphy 분해의 reliability/
    resolution 기여도, §14의 88.7%/90.7%)는 전부 N_BINS=10 균등폭
    구간으로 계산되었다. 그러나 ECE류 지표는 구간 개수·방식(균등폭 vs
    균등빈도)에 민감하다는 것이 보정(calibration) 문헌에서 잘 알려진
    비판이다(Nixon et al. 2019 등). 검증군 표본이 작을 때(DILIrank
    n=157) 특히 문제가 될 수 있다.

    이 스크립트는 새로운 가설을 탐색하는 것이 아니라, 이미 보고한
    핵심 지표가 구간화 방식 선택에 취약하지 않은지 확인하는
    견고성(robustness) 검증이다.

방법:
    시간분할 모델(seed=42, 기존 보고값과 동일 설정)을 1회 학습하고,
    동일한 예측확률에 대해 구간 수(5/10/15/20) 및 방식(균등폭 vs
    균등빈도/quantile)을 바꿔가며 ECE와 Brier 분해를 재계산한다.

사용법:
    python src/ece_binning_sensitivity.py

출력:
    results/ece_binning_sensitivity.csv
    results/ece_binning_sensitivity_summary.txt
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
BIN_COUNTS = [5, 10, 15, 20]

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


def bin_edges_equal_width(p, n_bins):
    return np.linspace(0, 1, n_bins + 1)


def bin_edges_equal_freq(p, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


def ece_and_decomp(y, p, edges):
    n_bins = len(edges) - 1
    N = len(y)
    obar = y.mean()
    ece_val, resolution, reliability = 0.0, 0.0, 0.0
    for i in range(n_bins):
        hi = (p <= 1) if i == n_bins - 1 else (p < edges[i + 1])
        m = (p >= edges[i]) & hi
        if m.sum() > 0:
            n_k = m.sum()
            f_k, o_k = p[m].mean(), y[m].mean()
            ece_val += n_k * abs(f_k - o_k) / N
            resolution += n_k * (o_k - obar) ** 2 / N
            reliability += n_k * (f_k - o_k) ** 2 / N
    return ece_val, reliability, resolution


def run_dataset(name, path):
    if not path.exists():
        log(f"  {name} 건너뜀: {path} 없음")
        return []

    df = load(path)
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr

    m = make_model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    y_te = y[te]

    rows = []
    log(f"\n{'='*72}\n{name} (검증군 n={te.sum()})\n{'='*72}")
    log(f"{'방식':>10s}{'구간수':>8s}{'ECE':>10s}{'reliability':>13s}" f"{'resolution':>12s}")

    for scheme, edge_fn in [("균등폭", bin_edges_equal_width), ("균등빈도", bin_edges_equal_freq)]:
        for nb in BIN_COUNTS:
            edges = edge_fn(p, nb)
            e, rel, res = ece_and_decomp(y_te, p, edges)
            total = rel + res if (rel + res) > 0 else np.nan
            pct_rel = 100 * rel / (rel + res) if (rel + res) > 0 else np.nan
            # 판별력/보정 기여도는 무작위 대조군 없이는 정의 불가하므로
            # 여기서는 ECE와 reliability/resolution 절대값만 비교
            log(f"{scheme:>10s}{nb:>8d}{e:>10.4f}{rel:>13.4f}{res:>12.4f}")
            rows.append(
                {
                    "dataset": name,
                    "scheme": scheme,
                    "n_bins": nb,
                    "ece": e,
                    "reliability": rel,
                    "resolution": res,
                }
            )

    d = pd.DataFrame(rows)
    log(
        f"\n  ECE 범위 (구간수 5~20, 균등폭+균등빈도 전체): "
        f"{d['ece'].min():.4f} ~ {d['ece'].max():.4f}"
    )
    log(f"  reliability 범위: {d['reliability'].min():.4f} ~ {d['reliability'].max():.4f}")
    log(f"  resolution 범위: {d['resolution'].min():.4f} ~ {d['resolution'].max():.4f}")
    base = d[(d.scheme == "균등폭") & (d.n_bins == 10)].iloc[0]
    log(
        f"  기존 보고값(균등폭 10구간): ECE {base['ece']:.4f}, "
        f"reliability {base['reliability']:.4f}, resolution {base['resolution']:.4f}"
    )
    rel_ratio = d["reliability"].max() / max(d["reliability"].min(), 1e-6)
    log(
        "  reliability가 resolution을 압도한다는 결론 유지 여부: "
        + (
            "유지됨 (reliability >> resolution, 모든 구간화 방식에서)"
            if d["reliability"].min() > d["resolution"].max()
            else "구간화 방식에 따라 역전 가능성 있음, 확인 필요"
        )
    )

    return rows


def main():
    log("ECE/Brier 분해 구간수·방식 민감도 검증")

    all_rows = []
    all_rows += run_dataset("DILIrank", PROC / "dilirank2_with_year.csv")
    all_rows += run_dataset("DICTrank", PROC / "dictrank_processed.csv")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(RES / "ece_binning_sensitivity.csv", index=False)
    (RES / "ece_binning_sensitivity_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'ece_binning_sensitivity.csv'}")
    log(f"      {RES/'ece_binning_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
