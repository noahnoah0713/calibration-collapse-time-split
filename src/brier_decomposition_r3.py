"""
[보완 1] Brier Murphy 분해 — 유병률 맞춤 대조군(R3) 기준

문제:
  brier_decomposition.py 의 "보정 기여율"은 R1(맨 무작위) 대조군 위에서
      share = d_reliability / (d_reliability + d_resolution)
  로 계산된다. 이 식은 uncertainty 항을 분모에서 제외한다. 그런데
  uncertainty = obar(1-obar) 는 검증군 양성률만으로 정해지고,
  R1 의 양성률은 시간분할과 크게 다르다. 그 결과 Brier 악화 중
  uncertainty 가 차지하는 몫이 작지 않다 (DICTrank 약 35%).

해결:
  R3(학습·검증 양측 비율맞춤) 대조군은 검증군 양성률이 시간분할과 같으므로
  d_uncertainty = 0 이 된다. R3 위에서 분해하면 2분할 기준과 3분할 기준의
  기여율이 일치하고, 분모 선택 논쟁이 사라진다.

이 스크립트는 R1/R2/R3 세 대조군 모두에서 분해하고,
  (A) 2분할 기준  rel/(rel+res)      <- 기존 보고 방식
  (B) 3분할 기준  rel/ΔBrier          <- uncertainty 포함
둘을 나란히 출력한다.

사용법:
    python src/brier_decomposition_r3.py [--reps N]

출력:
    results/brier_decomposition_r3.csv
    results/brier_decomposition_r3_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
N_REP_DEFAULT = 200  # prevalence_control.py 와 동일
log = S.Log()

CONFIGS = [
    ("R1. 무작위 (표준)", False, False),
    ("R2. 무작위 + 검증군 비율맞춤", True, False),
    ("R3. 무작위 + 학습·검증 비율맞춤", True, True),
]


def run(name, path, n_rep):
    df = S.load(path)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr, te = S.temporal_masks(df)
    n_test = int(te.sum())
    rate_t, rate_tr = float(y[te].mean()), float(y[tr].mean())

    log("=" * 74)
    log(f"{name}   학습 {tr.sum()} / 검증 {n_test}")
    log(f"  시간분할 양성률 {rate_t:.4f}  학습군 양성률 {rate_tr:.4f}")
    log("=" * 74)

    # ---------------------------------------------------- 시간분할
    m = S.make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]
    yte = y[te]
    dt = S.decompose(yte, p_t)
    auc_t, ece_t = roc_auc_score(yte, p_t), S.ece(yte, p_t)
    log(f"\n[T] 시간분할  AUROC {auc_t:.4f}  ECE {ece_t:.4f}")
    log(
        f"    reliability {dt['reliability']:.4f}  "
        f"resolution {dt['resolution']:.4f}  "
        f"uncertainty {dt['uncertainty']:.4f}  "
        f"Brier(근사) {dt['brier_approx']:.4f}"
    )

    rows = []
    for tag, match_test, match_train in CONFIGS:
        acc = {
            k: [] for k in ("reliability", "resolution", "uncertainty", "brier_approx", "obar", "N")
        }
        aucs, eces = [], []
        used = 0
        for i in range(n_rep):
            r = np.random.default_rng(S.SEED + i)
            idx = r.permutation(len(y))
            ti, ri = idx[:n_test], idx[n_test:]
            if match_test:
                ti = S.subsample_to_rate(y, ti, rate_t, r)
            if match_train:
                ri = S.subsample_to_rate(y, ri, rate_tr, r)
            if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                continue
            mm = S.make_model().fit(X[ri], y[ri])
            pp = mm.predict_proba(X[ti])[:, 1]
            d = S.decompose(y[ti], pp)
            for k in acc:
                acc[k].append(d[k])
            aucs.append(roc_auc_score(y[ti], pp))
            eces.append(S.ece(y[ti], pp))
            used += 1

        mean = {k: float(np.mean(v)) for k, v in acc.items()}
        d_rel = dt["reliability"] - mean["reliability"]
        d_res = mean["resolution"] - dt["resolution"]  # 판별력 손실(양수면 악화)
        d_unc = dt["uncertainty"] - mean["uncertainty"]
        d_bs = dt["brier_approx"] - mean["brier_approx"]

        share2 = 100 * d_rel / (d_rel + d_res) if (d_rel + d_res) else np.nan
        share3_rel = 100 * d_rel / d_bs if d_bs else np.nan
        share3_res = 100 * d_res / d_bs if d_bs else np.nan
        share3_unc = 100 * d_unc / d_bs if d_bs else np.nan

        log(
            f"\n[{tag}]  유효 {used}/{n_rep} 회, 평균 검증군 n={mean['N']:.0f}, "
            f"양성률 {mean['obar']:.4f}"
        )
        log(
            f"    무작위 평균:  rel {mean['reliability']:.4f}  "
            f"res {mean['resolution']:.4f}  unc {mean['uncertainty']:.4f}  "
            f"Brier {mean['brier_approx']:.4f}"
        )
        log("    차이(시간분할 − 무작위):")
        log(f"      Δreliability  {d_rel:+.4f}")
        log(f"      Δresolution   {d_res:+.4f}   (손실 방향을 양수로)")
        log(f"      Δuncertainty  {d_unc:+.4f}   <-- R3 에서 0 이어야 한다")
        log(f"      ΔBrier        {d_bs:+.4f}")
        log(
            f"    (A) 2분할 기준  보정 {share2:5.1f}%  판별력 {100-share2:5.1f}%"
            "      <- 기존 보고 방식"
        )
        log(
            f"    (B) 3분할 기준  보정 {share3_rel:5.1f}%  "
            f"판별력 {share3_res:5.1f}%  불확실성 {share3_unc:5.1f}%"
        )
        if abs(d_unc) < 5e-4:
            log(
                "    => Δuncertainty ≈ 0 이므로 (A) 와 (B) 가 일치한다. "
                "분모 선택이 결론을 바꾸지 않는다."
            )
        else:
            log(
                f"    => Δuncertainty 가 0 이 아니므로 (A) 는 ΔBrier 의 "
                f"{100-abs(share3_unc):.0f}% 안에서만 계산된 비율이다."
            )

        rows.append(
            dict(
                dataset=name,
                baseline=tag,
                n_rep_valid=used,
                n_test_temporal=n_test,
                n_test_random=mean["N"],
                obar_temporal=dt["obar"],
                obar_random=mean["obar"],
                rel_temporal=dt["reliability"],
                res_temporal=dt["resolution"],
                unc_temporal=dt["uncertainty"],
                brier_temporal=dt["brier_approx"],
                rel_random=mean["reliability"],
                res_random=mean["resolution"],
                unc_random=mean["uncertainty"],
                brier_random=mean["brier_approx"],
                d_reliability=d_rel,
                d_resolution=d_res,
                d_uncertainty=d_unc,
                d_brier=d_bs,
                share2_reliability=share2,
                share3_reliability=share3_rel,
                share3_resolution=share3_res,
                share3_uncertainty=share3_unc,
                auroc_temporal=auc_t,
                ece_temporal=ece_t,
                auroc_random=float(np.mean(aucs)),
                ece_random=float(np.mean(eces)),
            )
        )
    log("")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP_DEFAULT)
    a = ap.parse_args()

    log("Brier Murphy 분해 — 대조군별 비교 (R1 / R2 / R3)")
    log(f"반복 {a.reps}회, seed {S.SEED}, 절단 {S.CUTOFF}, 구간 {S.N_BINS}\n")

    out = []
    for nm, fn in S.DATASETS:
        p = PROC / fn
        if p.exists():
            out.append(run(nm, p, a.reps))
    if not out:
        log("데이터 없음")
        return
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "brier_decomposition_r3.csv", index=False)

    log("=" * 74)
    log("종합 — 대조군 선택이 보정 기여율에 미치는 영향")
    log("=" * 74)
    log(f"{'데이터셋':10s} {'대조군':30s} {'Δunc':>8s} " f"{'(A)2분할':>9s} {'(B)3분할':>9s}")
    log("-" * 74)
    for _, r in df.iterrows():
        log(
            f"{r.dataset:10s} {r.baseline:30s} {r.d_uncertainty:+8.4f} "
            f"{r.share2_reliability:8.1f}% {r.share3_reliability:8.1f}%"
        )
    log("")
    log("R3 행에서 Δunc ≈ 0 이고 (A) ≈ (B) 이면, 보고할 기여율은 R3 값이다.")
    log("R1 의 (A) 와 (B) 차이가 크다는 사실 자체가 대조군 선택의 중요성을 보여준다.")

    log.save(RES / "brier_decomposition_r3_summary.txt")
    print(f"\n저장: {RES/'brier_decomposition_r3.csv'}")
    print(f"      {RES/'brier_decomposition_r3_summary.txt'}")


if __name__ == "__main__":
    main()
