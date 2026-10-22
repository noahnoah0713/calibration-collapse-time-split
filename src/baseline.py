"""
DILIrank 2.0 예비 실험
- SMILES 표준화 (염 제거, 중화, 중복 제거)
- ECFP4 지문 + RDKit 기술자
- 무작위 5-fold CV vs Bemis-Murcko 스캐폴드 5-fold CV 비교

사용법:
    python src/baseline.py

출력:
    data/processed/dilirank2_clean.csv     표준화 완료 데이터
    results/baseline_results.csv           성능 요약
    results/baseline_per_fold.csv          fold별 상세
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5


# ------------------------------------------------------------------ 표준화
def standardize(smi: str):
    """염 제거 후 최대 유기 조각만 반환."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    # 탄소를 포함한 조각 중 원자 수 최대.
    # 탄소 조각이 하나도 없으면 무기물·금속착물·방사성 진단제이므로
    # 제외한다. 예전 판은 이때 가장 큰 조각을 그대로 가져가서
    # ferumoxytol -> [O-2], cisplatin -> [Cl][Pt][Cl] 같은 무의미한
    # 구조가 만들어졌고, Morgan 지문이 의미를 잃었다.
    organic = [f for f in frags if any(a.GetSymbol() == "C" for a in f.GetAtoms())]
    if not organic:
        return None
    best = max(organic, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(best)
    except Exception:
        return None


def featurize(smiles_list):
    """ECFP4(2048bit) + 물성 기술자 6종."""
    fps, desc = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(np.array(fp, dtype=np.int8))
        desc.append(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
            ]
        )
    return np.hstack([np.array(fps), np.array(desc)])


def scaffold_of(smi: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_folds(smiles_list, n_folds=N_FOLDS):
    """스캐폴드 그룹을 크기순으로 가장 작은 fold에 배정 (DeepChem 방식)."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[scaffold_of(smi)].append(i)
    ordered = sorted(groups.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_folds)]
    for grp in ordered:
        folds.sort(key=len)
        folds[0].extend(grp)
    return folds


# ------------------------------------------------------------------ 평가
def evaluate(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Acc": accuracy_score(y_true, y_pred),
    }


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


def run_cv(X, y, folds, name):
    rows = []
    for k, test_idx in enumerate(folds, 1):
        test_idx = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        if len(np.unique(y[test_idx])) < 2:
            print(f"  [{name}] fold {k}: 한 클래스만 존재 - 건너뜀")
            continue
        m = make_model().fit(X[train_idx], y[train_idx])
        prob = m.predict_proba(X[test_idx])[:, 1]
        r = evaluate(y[test_idx], prob)
        r.update(split=name, fold=k, n_test=len(test_idx), pos_rate=float(y[test_idx].mean()))
        rows.append(r)
        print(
            f"  [{name}] fold {k}: AUROC {r['AUROC']:.3f} | "
            f"MCC {r['MCC']:.3f} | n={len(test_idx)}"
        )
    return rows


# ------------------------------------------------------------------ main
def main():
    df = pd.read_csv(PROC / "dilirank2_with_smiles.csv")
    print(f"입력: {len(df)}행")

    # 이진 라벨 + SMILES 있는 것만
    df = df.dropna(subset=["y", "smiles_raw"]).copy()
    print(f"이진 라벨 + SMILES 보유: {len(df)}행")

    # 표준화
    df["smiles"] = df["smiles_raw"].apply(standardize)
    df = df.dropna(subset=["smiles"])
    print(f"구조 표준화 통과: {len(df)}행")

    # InChIKey 기준 중복 제거 (스켈레톤 14자리)
    df["ikey14"] = df["smiles"].apply(
        lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] if Chem.MolFromSmiles(s) else None
    )
    before = len(df)
    df = df.drop_duplicates(subset=["ikey14"], keep="first")
    print(f"중복 제거: {before} -> {len(df)}행 ({before - len(df)}개 제거)")

    df = df.reset_index(drop=True)
    y = df["y"].astype(int).values
    print(
        f"\n최종 데이터셋: N={len(df)}, " f"양성={y.sum()} ({y.mean():.1%}), 음성={len(y)-y.sum()}"
    )

    df.to_csv(PROC / "dilirank2_clean.csv", index=False)

    print("\n특징 추출 중...")
    X = featurize(df["smiles"].tolist())
    print(f"특징 행렬: {X.shape}")

    # 스캐폴드 통계
    scafs = df["smiles"].apply(scaffold_of)
    print(
        f"고유 스캐폴드: {scafs.nunique()}개 " f"(단독 출현 {(scafs.value_counts() == 1).sum()}개)"
    )

    all_rows = []

    print("\n--- 무작위 5-fold CV ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rand_folds = [te for _, te in skf.split(X, y)]
    all_rows += run_cv(X, y, rand_folds, "random")

    print("\n--- 스캐폴드 5-fold CV ---")
    scaf_folds = scaffold_folds(df["smiles"].tolist(), N_FOLDS)
    all_rows += run_cv(X, y, scaf_folds, "scaffold")

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(RES / "baseline_per_fold.csv", index=False)

    metrics = ["AUROC", "AUPRC", "BalAcc", "MCC", "F1", "Acc"]
    summary = per_fold.groupby("split")[metrics].agg(["mean", "std"]).round(3)
    summary.to_csv(RES / "baseline_results.csv")

    print("\n" + "=" * 60)
    print("결과 요약 (평균 ± 표준편차)")
    print("=" * 60)
    for split in ["random", "scaffold"]:
        sub = per_fold[per_fold.split == split]
        if len(sub) == 0:
            continue
        print(f"\n[{split}]")
        for m in metrics:
            print(f"  {m:8s} {sub[m].mean():.3f} ± {sub[m].std():.3f}")

    if {"random", "scaffold"} <= set(per_fold.split):
        print("\n" + "-" * 60)
        print("분할 간 격차 (무작위 - 스캐폴드)")
        for m in metrics:
            d = (
                per_fold[per_fold.split == "random"][m].mean()
                - per_fold[per_fold.split == "scaffold"][m].mean()
            )
            print(f"  {m:8s} {d:+.3f}")

    print(f"\n저장: {RES / 'baseline_results.csv'}")
    print(f"      {RES / 'baseline_per_fold.csv'}")


if __name__ == "__main__":
    main()
