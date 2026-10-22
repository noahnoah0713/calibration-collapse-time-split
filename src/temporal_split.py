"""
DILIrank 2.0 시간분할 실험

핵심 질문: 2009년 이전 승인약으로 학습한 모델이
           2010년 이후 신규 승인약을 얼마나 예측하는가?

교란 통제: 학습군 양성 65% vs 검증군 양성 31%로 클래스 비율이 역전되므로
           (a) AUROC를 주지표로 사용 (유병률 불변)
           (b) 검증군을 학습군 비율에 맞춰 재표집한 결과 병기
           (c) 무작위 분할을 동일 크기·동일 비율로 맞춘 대조군 추가

사용법:
    python src/temporal_split.py

출력:
    results/temporal_results.csv
    results/temporal_detail.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, DataStructs
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
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
CUTOFF = 2010  # 2010년 이후 = 검증군
N_BOOT = 1000
N_REPEAT = 20  # 재표집·무작위 대조 반복 횟수

rng = np.random.default_rng(SEED)


def featurize(smiles_list):
    fps, desc = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(
            np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.int8)
        )
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


def evaluate(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "BalAcc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
    }


def boot_ci(y, p, n=N_BOOT):
    """AUROC 부트스트랩 95% 신뢰구간."""
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        vals.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tanimoto_stats(train_smi, test_smi):
    """검증군 각 분자의 학습군 최근접 유사도."""
    tr = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in train_smi]
    out = []
    for s in test_smi:
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, tr)))
    return np.array(out)


def main():
    df = pd.read_csv(PROC / "dilirank2_with_year.csv")
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    print(f"분석 대상: {len(df)}개 (승인연도·SMILES·라벨 모두 보유)\n")

    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values

    tr_m = yr < CUTOFF
    te_m = ~tr_m
    Xtr, ytr = X[tr_m], y[tr_m]
    Xte, yte = X[te_m], y[te_m]

    print(f"학습군 (~{CUTOFF-1}): N={len(ytr)}, 양성 {ytr.mean():.1%}")
    print(f"검증군 ({CUTOFF}~): N={len(yte)}, 양성 {yte.mean():.1%}\n")

    rows = []

    # ---------------------------------------------------- 1. 시간분할 (주실험)
    print("--- 1. 시간분할 ---")
    model = make_model().fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    r = evaluate(yte, prob)
    lo, hi = boot_ci(yte, prob)
    print(f"  AUROC {r['AUROC']:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
    print(f"  MCC   {r['MCC']:.3f} | BalAcc {r['BalAcc']:.3f} | AUPRC {r['AUPRC']:.3f}")
    rows.append(
        {
            **r,
            "setting": "temporal",
            "n_test": len(yte),
            "pos_rate": float(yte.mean()),
            "auroc_lo": lo,
            "auroc_hi": hi,
        }
    )

    # -------------------------------- 2. 검증군 비율 맞춤 재표집 (교란 통제)
    print(f"\n--- 2. 검증군을 학습군 비율({ytr.mean():.1%})로 재표집 ---")
    pos_i = np.where(yte == 1)[0]
    neg_i = np.where(yte == 0)[0]
    target = ytr.mean()
    n_pos = len(pos_i)
    n_neg_needed = int(round(n_pos * (1 - target) / target))
    accs = []
    if n_neg_needed <= len(neg_i):
        for _ in range(N_REPEAT):
            sel = np.concatenate([pos_i, rng.choice(neg_i, n_neg_needed, replace=False)])
            accs.append(evaluate(yte[sel], prob[sel]))
        m = {k: np.mean([a[k] for a in accs]) for k in accs[0]}
        s = {k: np.std([a[k] for a in accs]) for k in accs[0]}
        print(
            f"  AUROC {m['AUROC']:.3f} ± {s['AUROC']:.3f}  "
            f"(n={n_pos + n_neg_needed}, 양성 {target:.1%})"
        )
        print(f"  MCC   {m['MCC']:.3f} ± {s['MCC']:.3f} | BalAcc {m['BalAcc']:.3f}")
        rows.append(
            {
                **m,
                "setting": "temporal_rebalanced",
                "n_test": n_pos + n_neg_needed,
                "pos_rate": target,
                "auroc_lo": None,
                "auroc_hi": None,
            }
        )
    else:
        print("  음성 표본 부족으로 생략")

    # ------------------------- 3. 무작위 대조군 (시간분할과 동일 크기)
    print(f"\n--- 3. 무작위 분할 대조군 (동일 크기 n={len(yte)}, {N_REPEAT}회) ---")
    accs = []
    for i in range(N_REPEAT):
        r_ = np.random.default_rng(SEED + i)
        idx = r_.permutation(len(y))
        te_i, tr_i = idx[: len(yte)], idx[len(yte) :]
        if len(np.unique(y[te_i])) < 2:
            continue
        m_ = make_model().fit(X[tr_i], y[tr_i])
        p_ = m_.predict_proba(X[te_i])[:, 1]
        accs.append(evaluate(y[te_i], p_))
    m = {k: np.mean([a[k] for a in accs]) for k in accs[0]}
    s = {k: np.std([a[k] for a in accs]) for k in accs[0]}
    print(f"  AUROC {m['AUROC']:.3f} ± {s['AUROC']:.3f}")
    print(f"  MCC   {m['MCC']:.3f} ± {s['MCC']:.3f} | BalAcc {m['BalAcc']:.3f}")
    rows.append(
        {
            **m,
            "setting": "random_matched",
            "n_test": len(yte),
            "pos_rate": float(y.mean()),
            "auroc_lo": None,
            "auroc_hi": None,
        }
    )

    # ---------------------------------------------------- 4. 화학공간 이동 진단
    print("\n--- 4. 화학공간 이동 진단 ---")
    sim = tanimoto_stats(df.loc[tr_m, "smiles"].tolist(), df.loc[te_m, "smiles"].tolist())
    print(
        f"  검증군의 학습군 최근접 Tanimoto: 중앙값 {np.median(sim):.3f}, " f"평균 {sim.mean():.3f}"
    )
    print(
        f"  유사도 < 0.4 (외삽 영역): {(sim < 0.4).sum()}/{len(sim)} " f"({(sim < 0.4).mean():.1%})"
    )

    # 유사도 구간별 성능
    print("\n  유사도 구간별 AUROC:")
    for lo_, hi_ in [(0, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]:
        m_ = (sim >= lo_) & (sim < hi_)
        if m_.sum() >= 20 and len(np.unique(yte[m_])) == 2:
            print(
                f"    [{lo_:.1f}-{hi_:.1f}) n={m_.sum():3d}  "
                f"AUROC {roc_auc_score(yte[m_], prob[m_]):.3f}"
            )
        elif m_.sum() > 0:
            print(f"    [{lo_:.1f}-{hi_:.1f}) n={m_.sum():3d}  (표본 부족)")

    # ---------------------------------------------------- 저장 및 요약
    out = pd.DataFrame(rows)
    out.to_csv(RES / "temporal_results.csv", index=False)

    det = df.loc[te_m, ["LTKBID", "CompoundName", "approval_year", "y"]].copy()
    det["prob"] = prob
    det["nn_tanimoto"] = sim
    det.to_csv(RES / "temporal_detail.csv", index=False)

    print("\n" + "=" * 58)
    print("핵심 비교")
    print("=" * 58)
    t = out[out.setting == "temporal"].iloc[0]
    rnd = out[out.setting == "random_matched"].iloc[0]
    print(f"  무작위 분할  AUROC {rnd['AUROC']:.3f}")
    print(
        f"  시간분할     AUROC {t['AUROC']:.3f}  " f"[CI {t['auroc_lo']:.3f}-{t['auroc_hi']:.3f}]"
    )
    print(f"  격차 (ΔAUROC)      {rnd['AUROC'] - t['AUROC']:+.3f}")
    if "temporal_rebalanced" in set(out.setting):
        rb = out[out.setting == "temporal_rebalanced"].iloc[0]
        print(f"\n  비율 보정 후 시간분할 AUROC {rb['AUROC']:.3f}")
        print(f"  → 보정 전후 차이 {rb['AUROC'] - t['AUROC']:+.3f} " f"(클래스 비율 기여분)")

    print(f"\n저장: {RES / 'temporal_results.csv'}")
    print(f"      {RES / 'temporal_detail.csv'}")


if __name__ == "__main__":
    main()
