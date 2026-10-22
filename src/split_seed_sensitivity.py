"""
[보완 7] 분할 배정 시드 민감도 — 무작위 CV 와 스캐폴드 CV

문제:
  seed_sensitivity.py 는 '모델 학습 시드' 10종만 바꾼다. 분할을 만드는
  쪽의 임의성은 검증되지 않았다.
    - baseline_cv.py 의 random_cv 는 StratifiedKFold(random_state=SEED) 로
      시드 하나만 쓴다.
    - scaffold_folds() 는 시드가 아예 없다. 스캐폴드 그룹을 크기순으로
      정렬해 가장 작은 fold 에 차례로 넣는 결정론적 그리디 배정이다.
      즉 '스캐폴드 CV 의 ECE 가 무작위 CV 와 비슷하다' 는 §4 결론이
      이 한 가지 배정 방식에만 근거한다.

이 스크립트:
  A) random_cv 를 시드 10종으로 반복
  B) scaffold_cv 를 두 가지 방식으로 반복
       B1 결정론적 그리디 (현재 방식, 참고용 1회)
       B2 무작위 배정: 스캐폴드 그룹 순서를 섞은 뒤 같은 그리디 —
          시드 10종. 그룹 단위 분리는 유지되므로 스캐폴드 CV 의 성질은
          그대로고, 배정의 임의성만 검증한다.
  각 조건에서 AUROC 와 ECE 를 낸다. 모델 시드는 42 로 고정한다.

사용법:
    python src/split_seed_sensitivity.py [--seeds N]

출력:
    results/split_seed_sensitivity.csv
    results/split_seed_sensitivity_summary.txt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
N_FOLDS = 5
SEEDS = [42, 0, 1, 7, 13, 21, 100, 123, 777, 2024]
log = S.Log()


def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_folds(smiles_list, n_folds=N_FOLDS, seed=None):
    """seed=None 이면 baseline_cv.py 와 동일한 결정론적 그리디."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[scaffold_of(smi)].append(i)
    vals = list(groups.values())
    if seed is None:
        ordered = sorted(vals, key=len, reverse=True)
    else:
        r = np.random.default_rng(seed)
        r.shuffle(vals)
        ordered = vals  # 크기 정렬 없이 무작위 순서
    folds = [[] for _ in range(n_folds)]
    for grp in ordered:
        folds.sort(key=len)
        folds[0].extend(grp)
    return folds


def eval_folds(X, y, folds):
    oof = np.zeros(len(y), dtype=float)
    ok = np.zeros(len(y), dtype=bool)
    per = []
    for f in folds:
        te = np.array(sorted(f), dtype=int)
        if len(te) == 0:
            continue
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = S.make_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te], ok[te] = p, True
        per.append((roc_auc_score(y[te], p), S.ece(y[te], p), len(te)))
    if not per:
        return None
    return dict(
        auroc_oof=roc_auc_score(y[ok], oof[ok]),
        ece_oof=S.ece(y[ok], oof[ok]),
        auroc_fold_mean=float(np.mean([a for a, _, _ in per])),
        auroc_fold_sd=float(np.std([a for a, _, _ in per])),
        ece_fold_mean=float(np.mean([e for _, e, _ in per])),
        ece_fold_sd=float(np.std([e for _, e, _ in per])),
        n_folds_used=len(per),
    )


def run(name, path, n_seeds):
    df = S.load(path)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    smi = df["smiles"].tolist()
    seeds = SEEDS[:n_seeds]

    scafs = [scaffold_of(s) for s in smi]
    n_uniq = len(set(scafs))
    from collections import Counter

    single = sum(1 for _, c in Counter(scafs).items() if c == 1)

    log("=" * 86)
    log(
        f"{name}   n={len(df)}   고유 스캐폴드 {n_uniq} (단독 출현 {single}, "
        f"{single/n_uniq:.1%})"
    )
    log("=" * 86)

    rows = []

    log(f"\n[A] 무작위 CV — StratifiedKFold 시드 {len(seeds)}종")
    log(
        f"    {'시드':>6s} {'AUROC(oof)':>11s} {'ECE(oof)':>10s} "
        f"{'AUROC fold평균±SD':>20s} {'ECE fold평균±SD':>20s}"
    )
    for sd in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=sd)
        folds = [list(te) for _, te in skf.split(X, y)]
        r = eval_folds(X, y, folds)
        if r is None:
            continue
        rows.append(dict(dataset=name, split="random_cv", seed=sd, **r))
        log(
            f"    {sd:6d} {r['auroc_oof']:11.4f} {r['ece_oof']:10.4f} "
            f"{r['auroc_fold_mean']:10.4f}±{r['auroc_fold_sd']:.4f} "
            f"{r['ece_fold_mean']:11.4f}±{r['ece_fold_sd']:.4f}"
        )

    log(f"\n[B1] 스캐폴드 CV — 결정론적 그리디 (baseline_cv.py 현재 방식)")
    r = eval_folds(X, y, scaffold_folds(smi))
    if r:
        rows.append(dict(dataset=name, split="scaffold_cv_greedy", seed=np.nan, **r))
        log(
            f"    AUROC(oof) {r['auroc_oof']:.4f}  ECE(oof) {r['ece_oof']:.4f}  "
            f"AUROC fold {r['auroc_fold_mean']:.4f}±{r['auroc_fold_sd']:.4f}  "
            f"ECE fold {r['ece_fold_mean']:.4f}±{r['ece_fold_sd']:.4f}"
        )

    log(f"\n[B2] 스캐폴드 CV — 그룹 순서 무작위화, 시드 {len(seeds)}종")
    log(
        f"    {'시드':>6s} {'AUROC(oof)':>11s} {'ECE(oof)':>10s} "
        f"{'AUROC fold평균±SD':>20s} {'ECE fold평균±SD':>20s}"
    )
    for sd in seeds:
        r = eval_folds(X, y, scaffold_folds(smi, seed=sd))
        if r is None:
            continue
        rows.append(dict(dataset=name, split="scaffold_cv_random", seed=sd, **r))
        log(
            f"    {sd:6d} {r['auroc_oof']:11.4f} {r['ece_oof']:10.4f} "
            f"{r['auroc_fold_mean']:10.4f}±{r['auroc_fold_sd']:.4f} "
            f"{r['ece_fold_mean']:11.4f}±{r['ece_fold_sd']:.4f}"
        )

    out = pd.DataFrame(rows)
    log(f"\n[요약] {name}")
    for sp in ["random_cv", "scaffold_cv_greedy", "scaffold_cv_random"]:
        s = out[out.split == sp]
        if not len(s):
            continue
        log(
            f"    {sp:20s} AUROC {s.auroc_oof.mean():.4f} "
            f"(범위 {s.auroc_oof.min():.4f}-{s.auroc_oof.max():.4f})  "
            f"ECE {s.ece_oof.mean():.4f} "
            f"(범위 {s.ece_oof.min():.4f}-{s.ece_oof.max():.4f})"
        )
    log("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    log("분할 배정 시드 민감도 — 무작위 CV / 스캐폴드 CV")
    log(f"모델 시드는 {S.SEED} 로 고정, {N_FOLDS}-fold\n")
    out = []
    for nm, fn in S.DATASETS:
        p = PROC / fn
        if p.exists():
            out.append(run(nm, p, a.seeds))
    if not out:
        log("데이터 없음")
        return
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "split_seed_sensitivity.csv", index=False)
    log("=" * 86)
    log("판정")
    log("=" * 86)
    log("§4 결론은 '스캐폴드 CV 의 ECE 가 무작위 CV 와 비슷하고, 둘 다")
    log("시간분할보다 훨씬 낮다' 는 것이다. 아래 두 조건이 필요하다:")
    log("  (1) 무작위 CV ECE 가 시드에 따라 크게 흔들리지 않아야 한다")
    log("  (2) 스캐폴드 CV ECE 가 배정 방식(그리디 vs 무작위)에 따라")
    log("      크게 흔들리지 않아야 한다")
    log("")
    for ds in df.dataset.unique():
        s = df[df.dataset == ds]
        rc = s[s.split == "random_cv"].ece_oof
        sg = s[s.split == "scaffold_cv_greedy"].ece_oof
        sr = s[s.split == "scaffold_cv_random"].ece_oof
        log(f"{ds}")
        log(
            f"  무작위 CV ECE          {rc.mean():.4f} "
            f"(범위 {rc.min():.4f}-{rc.max():.4f}, 폭 {rc.max()-rc.min():.4f})"
        )
        if len(sg):
            log(f"  스캐폴드 CV ECE 그리디   {sg.iloc[0]:.4f}")
        log(
            f"  스캐폴드 CV ECE 무작위배정 {sr.mean():.4f} "
            f"(범위 {sr.min():.4f}-{sr.max():.4f}, 폭 {sr.max()-sr.min():.4f})"
        )
        log(f"  => 두 분할의 ECE 차이 {abs(sr.mean()-rc.mean()):.4f}")
        log("")
    log("이 차이가 시간분할 ECE(DILIrank 0.4384 / DICTrank 0.2702) 와의")
    log("간격보다 훨씬 작으면 §4 결론이 배정 방식에 의존하지 않는다.")
    log.save(RES / "split_seed_sensitivity_summary.txt")
    print(f"\n저장: {RES/'split_seed_sensitivity.csv'}")
    print(f"      {RES/'split_seed_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
