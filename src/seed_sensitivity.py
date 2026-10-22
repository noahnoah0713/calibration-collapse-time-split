"""
시드 민감도 검증

목적:
    지금까지 보고한 모든 핵심 수치(시간분할 AUROC, ECE, Brier 분해,
    conformal coverage)는 XGBoost random_state=42 단일 학습 결과에
    기반한다. 데이터 분할(시간분할 자체)은 무작위성이 없어 고정이지만,
    모델 학습 과정의 시드에 따라 결과가 크게 달라진다면 지금까지의
    핵심 발견의 신뢰성에 의문이 생긴다. 이를 확인하기 위해 동일한
    시간분할(train<2010, test>=2010)에서 모델 시드만 10개로 바꿔
    재학습하고, AUROC/ECE/Brier 분해가 얼마나 안정적인지 확인한다.

    주의: 이는 새로운 가설을 탐색하는 실험이 아니라, 이미 보고한
    핵심 결과의 재현성(reproducibility)을 검증하는 것이다.

사용법:
    python src/seed_sensitivity.py

출력:
    results/seed_sensitivity.csv
    results/seed_sensitivity_summary.txt
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

CUTOFF = 2010
SEEDS = [42, 0, 1, 7, 13, 21, 100, 123, 777, 2024]  # 원 시드(42) 포함 10개
N_BINS = 10

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


def make_model(seed):
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def ece(y, p, n_bins=N_BINS):
    """_shared.ece 와 식까지 같아야 한다.

    2026-10-20: 이 파일만 구간별 가중을 마지막에 n 으로 나누고 있었다
    (다른 여섯 정의는 구간마다 m.sum()/len(p) 로 나눈다). 수학적으로는
    같지만 부동소수 합산 순서가 달라 마지막 비트가 어긋났다
    (0.2586619759775421 대 0.25866197597754215). 보고 자리수에는 영향이
    없지만, 지표 정의가 갈라져 있다는 사실 자체가 위험이다 — 이 저장소는
    같은 이유로 모델 하이퍼파라미터(2026-09-30)와 표현 dtype(2026-10-01)
    에서 이미 두 번 어긋난 적이 있다. check_determinism.py [1c] 가 이제
    모든 ece 정의를 고정 입력으로 비트 단위 대조한다.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        out += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(out)


def brier_reliability_resolution(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    N = len(y)
    obar = y.mean()
    resolution, reliability = 0.0, 0.0
    for i in range(n_bins):
        hi = (p <= 1) if i == n_bins - 1 else (p < edges[i + 1])
        m = (p >= edges[i]) & hi
        if m.sum() > 0:
            n_k = m.sum()
            f_k, o_k = p[m].mean(), y[m].mean()
            resolution += n_k * (o_k - obar) ** 2 / N
            reliability += n_k * (f_k - o_k) ** 2 / N
    return reliability, resolution


def run_dataset(name, path):
    from sklearn.metrics import roc_auc_score

    if not path.exists():
        log(f"  {name} 건너뜀: {path} 없음")
        return []

    df = load(path)
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr

    rows = []
    log(f"\n{'='*68}\n{name} (검증군 n={te.sum()})\n{'='*68}")
    log(f"{'시드':>6s}{'AUROC':>10s}{'ECE':>10s}{'reliability':>13s}{'resolution':>12s}")

    for seed in SEEDS:
        m = make_model(seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        auroc = roc_auc_score(y[te], p)
        e = ece(y[te], p)
        rel, res = brier_reliability_resolution(y[te], p)
        log(f"{seed:>6d}{auroc:>10.4f}{e:>10.4f}{rel:>13.4f}{res:>12.4f}")
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "auroc": auroc,
                "ece": e,
                "reliability": rel,
                "resolution": res,
            }
        )

    d = pd.DataFrame(rows)
    log(
        f"\n  AUROC   평균 {d['auroc'].mean():.4f} ± {d['auroc'].std():.4f} "
        f"(범위 {d['auroc'].min():.4f}~{d['auroc'].max():.4f})"
    )
    log(
        f"  ECE     평균 {d['ece'].mean():.4f} ± {d['ece'].std():.4f} "
        f"(범위 {d['ece'].min():.4f}~{d['ece'].max():.4f})"
    )
    seed42 = d[d.seed == 42].iloc[0]
    log(f"  기존 보고값(seed=42): AUROC {seed42['auroc']:.4f}, ECE {seed42['ece']:.4f}")
    log(
        "  -> seed=42가 10개 시드 범위 안에서 대표성을 갖는지 확인됨"
        if (d["auroc"].min() <= seed42["auroc"] <= d["auroc"].max())
        else "  -> 경고: seed=42가 이례적일 수 있음, 재확인 필요"
    )

    return rows


def main():
    log("시드 민감도 검증: 핵심 결과가 특정 시드에 의존하지 않는지 확인")

    all_rows = []
    all_rows += run_dataset("DILIrank", PROC / "dilirank2_with_year.csv")
    all_rows += run_dataset("DICTrank", PROC / "dictrank_processed.csv")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(RES / "seed_sensitivity.csv", index=False)
    (RES / "seed_sensitivity_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'seed_sensitivity.csv'}")
    log(f"      {RES/'seed_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
