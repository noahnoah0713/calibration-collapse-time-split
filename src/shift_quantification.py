"""
[보완 6] 시간적 공변량 이동의 직접 측정

문제:
  논문은 시간분할이 분포 이동을 만든다고 주장하면서, 그 이동의 크기를
  최근접 Tanimoto 유사도(대리 지표)로만 보여준다. 이동 자체를 직접
  측정하지 않았다. 표준적인 방법은 '학습군 대 검증군' 을 맞히는
  분류기를 세워 그 AUC 를 보는 것이다 (domain classifier / two-sample test).
  AUC 0.5 면 구별 불가(이동 없음), 1.0 이면 완전 분리.

이 스크립트:
  1) 시대 분류기: 같은 ECFP4+desc 표현으로 '검증군 소속' 을 예측.
     5-fold CV AUC 와 순열 귀무분포(라벨 섞기) 대비 p값(Phipson-Smyth).
  2) 같은 절차를 무작위 분할 라벨에 적용한 대조군 — 이동이 없을 때의
     기준선을 실측으로 얻는다.
  3) 시대 분류기가 지목하는 상위 기여 기술자(치환 중요도)로 무엇이
     바뀌었는지 요약.

사용법:
    python src/shift_quantification.py [--perm N]

출력:
    results/shift_quantification.csv
    results/shift_quantification_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
N_PERM_DEFAULT = 200
N_FOLDS = 5
DESC_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]
log = S.Log()


def cv_auc(X, z, seed=S.SEED):
    """z(이진 라벨) 를 맞히는 5-fold CV AUC."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(z), dtype=float)
    for tr_i, te_i in skf.split(X, z):
        m = S.make_model(seed).fit(X[tr_i], z[tr_i])
        oof[te_i] = m.predict_proba(X[te_i])[:, 1]
    return roc_auc_score(z, oof), oof


def run(name, path, n_perm):
    df = S.load(path)
    X = S.featurize(df["smiles"].tolist())
    tr, te = S.temporal_masks(df)
    z = te.astype(int)  # 1 = 검증군(2010년 이후)

    log("=" * 80)
    log(f"{name}   n={len(df)}  학습군 {tr.sum()} / 검증군 {te.sum()}")
    log("=" * 80)

    auc_era, oof = cv_auc(X, z)
    log(f"\n[1] 시대 분류기 ('2010년 이후인가' 를 구조로 예측)")
    log(f"    5-fold CV AUC = {auc_era:.4f}")
    log("    0.5 = 구별 불가(이동 없음) / 1.0 = 완전 분리")

    # 순열 귀무분포: 라벨을 섞어도 같은 AUC 가 나오는지
    null = []
    for i in range(n_perm):
        r = np.random.default_rng(S.SEED + 1000 + i)
        zp = r.permutation(z)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=S.SEED)
        o = np.zeros(len(zp), dtype=float)
        for a, b in skf.split(X, zp):
            m = S.make_model().fit(X[a], zp[a])
            o[b] = m.predict_proba(X[b])[:, 1]
        null.append(roc_auc_score(zp, o))
    null = np.array(null)
    r_ge = int((null >= auc_era).sum())
    p = (1 + r_ge) / (1 + len(null))
    log(f"\n[2] 순열 귀무분포 {len(null)}회 (시대 라벨 섞기)")
    log(
        f"    평균 {null.mean():.4f} (SD {null.std():.4f})  "
        f"2.5-97.5 백분위 {np.percentile(null,2.5):.4f}-"
        f"{np.percentile(null,97.5):.4f}"
    )
    log(f"    관측값 이상: {r_ge}/{len(null)}   " f"p (Phipson-Smyth) = {p:.4f}")

    # 기술자 6종만으로 시대 분류 (해석 가능한 버전)
    Xd = X[:, -6:]
    auc_desc, _ = cv_auc(Xd, z)
    log(f"\n[3] RDKit 기술자 6종만으로 시대 분류  AUC = {auc_desc:.4f}")
    log(f"    {'기술자':18s} {'학습군 중앙값':>14s} {'검증군 중앙값':>14s} {'차이':>10s}")
    desc_rows = []
    for j, nm in enumerate(DESC_NAMES):
        a, b = float(np.median(Xd[tr, j])), float(np.median(Xd[te, j]))
        log(f"    {nm:18s} {a:14.3f} {b:14.3f} {b-a:+10.3f}")
        desc_rows.append(
            dict(dataset=name, descriptor=nm, median_train=a, median_test=b, diff=b - a)
        )

    # 이동 없음 기준선: 무작위 분할 라벨로 같은 절차
    log(f"\n[4] 대조군 — 같은 크기의 무작위 분할 라벨로 같은 분류기")
    ctrl = []
    for i in range(5):
        r = np.random.default_rng(S.SEED + i)
        idx = r.permutation(len(z))
        zz = np.zeros(len(z), dtype=int)
        zz[idx[: int(te.sum())]] = 1
        ctrl.append(cv_auc(X, zz)[0])
    log(
        f"    무작위 분할 시대분류 AUC 평균 {np.mean(ctrl):.4f} "
        f"(범위 {min(ctrl):.4f}-{max(ctrl):.4f})"
    )
    log(
        f"    => 시간분할의 {auc_era:.4f} 는 무작위 기준선보다 "
        f"{auc_era-np.mean(ctrl):+.4f} 높다"
    )

    log("")
    return (
        pd.DataFrame(
            [
                dict(
                    dataset=name,
                    n=len(df),
                    n_train=int(tr.sum()),
                    n_test=int(te.sum()),
                    era_auc_full=auc_era,
                    era_auc_desc_only=auc_desc,
                    null_mean=float(null.mean()),
                    null_sd=float(null.std()),
                    null_p025=float(np.percentile(null, 2.5)),
                    null_p975=float(np.percentile(null, 97.5)),
                    n_ge_observed=r_ge,
                    n_perm=len(null),
                    p_phipson_smyth=p,
                    random_split_era_auc=float(np.mean(ctrl)),
                )
            ]
        ),
        pd.DataFrame(desc_rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=N_PERM_DEFAULT)
    a = ap.parse_args()
    log("시간적 공변량 이동의 직접 측정 (시대 분류기)")
    log(f"순열 {a.perm}회, {N_FOLDS}-fold CV, seed {S.SEED}, 절단 {S.CUTOFF}\n")
    main_rows, desc_rows = [], []
    for nm, fn in S.DATASETS:
        p = PROC / fn
        if p.exists():
            m, d = run(nm, p, a.perm)
            main_rows.append(m)
            desc_rows.append(d)
    if not main_rows:
        log("데이터 없음")
        return
    df = pd.concat(main_rows, ignore_index=True)
    df.to_csv(RES / "shift_quantification.csv", index=False)
    pd.concat(desc_rows, ignore_index=True).to_csv(
        RES / "shift_quantification_descriptors.csv", index=False
    )
    log("=" * 80)
    log("종합")
    log("=" * 80)
    log(
        f"{'데이터셋':10s} {'시대분류 AUC':>12s} {'기술자만':>9s} "
        f"{'무작위 기준':>11s} {'p':>8s}"
    )
    log("-" * 80)
    for _, r in df.iterrows():
        log(
            f"{r.dataset:10s} {r.era_auc_full:12.4f} {r.era_auc_desc_only:9.4f} "
            f"{r.random_split_era_auc:11.4f} {r.p_phipson_smyth:8.4f}"
        )
    log("")
    log("[읽는 법] 시대분류 AUC 가 0.5 에서 멀고 무작위 기준선보다 높으면,")
    log("          시간분할이 만드는 분포 이동이 구조 수준에서 실재한다는")
    log("          직접 증거다. 이것이 Tanimoto 유사도 대리 지표를 대체한다.")
    log.save(RES / "shift_quantification_summary.txt")
    print(f"\n저장: {RES/'shift_quantification.csv'}")
    print(f"      {RES/'shift_quantification_descriptors.csv'}")
    print(f"      {RES/'shift_quantification_summary.txt'}")


if __name__ == "__main__":
    main()
