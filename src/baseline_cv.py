"""
분할 전략 비교 (Fig 1a 전용): 무작위 5-fold CV / 스캐폴드 5-fold CV / 시간분할

기존 baseline.py와의 차이 — 왜 별도 스크립트가 필요한가:
  baseline.py는 승인연도 매칭 '이전'의 DILIrank 870개로 CV를 돌렸고 DICTrank는
  아예 다루지 않는다. 그 결과 Fig 1a의 '무작위' 막대(870개 CV)와 '시간분할'
  막대(806개)가 서로 다른 데이터에서 나온 값이었고, DICTrank의 무작위 막대는
  또 다른 출처(순열검정 귀무평균)에서 왔다.

  이 스크립트는 두 데이터셋 모두 '시간분할에 실제로 쓰인 분석 대상'
  (DILIrank 806, DICTrank 953) 위에서 세 가지 분할을 동일한 절차로 평가한다.
  따라서 Fig 1a의 세 막대는 같은 화합물 집합, 같은 모델, 같은 특징에서 나온다.

사용법:
    python src/baseline_cv.py

출력:
    results/baseline_cv_per_fold.csv   fold별 상세 + 시간분할 행
    results/baseline_cv_summary.txt
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
N_FOLDS = 5
N_BINS = 10
N_BOOT = 1000

rng = np.random.default_rng(SEED)
log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


# --------------------------------------------------- 특징 (주 분석과 동일)
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


def boot_ci_auc(y, p, n=N_BOOT):
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        vals.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------- 스캐폴드 (baseline.py와 동일)
def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_folds(smiles_list, n_folds=N_FOLDS):
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[scaffold_of(smi)].append(i)
    ordered = sorted(groups.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_folds)]
    for grp in ordered:
        folds.sort(key=len)
        folds[0].extend(grp)
    return folds


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def run_cv(X, y, folds, dataset, split):
    rows = []
    for k, test_idx in enumerate(folds, 1):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(np.unique(y[test_idx])) < 2:
            log(f"    [{split}] fold {k}: 한 클래스만 존재 - 건너뜀")
            continue
        m = make_model().fit(X[train_idx], y[train_idx])
        p = m.predict_proba(X[test_idx])[:, 1]
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "fold": k,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "pos_rate": float(y[test_idx].mean()),
                "AUROC": roc_auc_score(y[test_idx], p),
                "ECE": ece(y[test_idx], p),
                "auroc_lo": np.nan,
                "auroc_hi": np.nan,
            }
        )
    return rows


def run_dataset(name, path):
    if not path.exists():
        log(f"  {name} 건너뜀: {path.name} 없음")
        return []
    df = load(path)
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values

    log("=" * 68)
    log(f"{name}   (분석 대상 {len(df)}개 — 시간분할에 쓰인 집합과 동일)")
    log("=" * 68)

    scafs = pd.Series([scaffold_of(s) for s in df["smiles"]])
    log(f"  고유 스캐폴드 {scafs.nunique()}개 (단독 출현 {(scafs.value_counts()==1).sum()}개)")

    rows = []

    # 1) 무작위 5-fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows += run_cv(X, y, [te for _, te in skf.split(X, y)], name, "random_cv")

    # 2) 스캐폴드 5-fold
    rows += run_cv(X, y, scaffold_folds(df["smiles"].tolist()), name, "scaffold_cv")

    # 3) 시간분할 (단일 분할 + 부트스트랩 CI)
    tr, te = yr < CUTOFF, yr >= CUTOFF
    m = make_model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    lo, hi = boot_ci_auc(y[te], p)
    rows.append(
        {
            "dataset": name,
            "split": "temporal",
            "fold": np.nan,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "pos_rate": float(y[te].mean()),
            "AUROC": roc_auc_score(y[te], p),
            "ECE": ece(y[te], p),
            "auroc_lo": lo,
            "auroc_hi": hi,
        }
    )

    d = pd.DataFrame(rows)
    log(f"\n  {'분할':14s} {'AUROC':>16s} {'ECE':>16s}")
    log("  " + "-" * 48)
    for sp in ["random_cv", "scaffold_cv", "temporal"]:
        s = d[d.split == sp]
        if not len(s):
            continue
        if sp == "temporal":
            r = s.iloc[0]
            log(f"  {sp:14s} {r.AUROC:8.4f} [{r.auroc_lo:.3f}-{r.auroc_hi:.3f}] " f"{r.ECE:10.4f}")
        else:
            log(
                f"  {sp:14s} {s.AUROC.mean():8.4f} ± {s.AUROC.std(ddof=1):.4f}   "
                f"{s.ECE.mean():8.4f} ± {s.ECE.std(ddof=1):.4f}"
            )
    log()
    return rows


def main():
    log("분할 전략 비교 — 두 데이터셋을 동일한 절차로 평가")
    log("(무작위 5-fold CV / Bemis-Murcko 스캐폴드 5-fold CV / 시간분할)\n")

    rows = []
    rows += run_dataset("DILIrank", PROC / "dilirank2_with_year.csv")
    rows += run_dataset("DICTrank", PROC / "dictrank_processed.csv")

    if not rows:
        log("데이터 파일 없음")
        return

    out = pd.DataFrame(rows)
    out.to_csv(RES / "baseline_cv_per_fold.csv", index=False)

    log("=" * 68)
    log("요약 (Fig 1a에 들어갈 값)")
    log("=" * 68)
    log(f"{'데이터셋':12s} {'분할':14s} {'AUROC':>10s} {'±SD/CI':>18s}")
    for ds in out.dataset.unique():
        for sp in ["random_cv", "scaffold_cv", "temporal"]:
            s = out[(out.dataset == ds) & (out.split == sp)]
            if not len(s):
                continue
            if sp == "temporal":
                r = s.iloc[0]
                log(
                    f"{ds:12s} {sp:14s} {r.AUROC:10.4f} "
                    f"{'[' + format(r.auroc_lo, '.3f') + '-' + format(r.auroc_hi, '.3f') + ']':>18s}"
                )
            else:
                log(
                    f"{ds:12s} {sp:14s} {s.AUROC.mean():10.4f} "
                    f"{'±' + format(s.AUROC.std(ddof=1), '.4f'):>18s}"
                )

    (RES / "baseline_cv_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'baseline_cv_per_fold.csv'}")
    log(f"      {RES/'baseline_cv_summary.txt'}")


if __name__ == "__main__":
    main()
