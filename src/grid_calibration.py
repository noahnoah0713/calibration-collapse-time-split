"""
[보완 3] Brier 분해와 conformal 커버리지를 모델×표현 격자 전체로 확장

문제:
  ECE 배율은 32조합 전부에서 확인됐지만(robustness.py), 헤드라인인
  "Brier 악화 중 보정 기여 86.1% / 82.5%" 와 conformal 커버리지 붕괴는
  ECFP4+desc / XGBoost 한 조합에서만 계산됐다.
  "보정 기여율이 모델 특이적인가" 라는 질문에 답이 없다.

이 스크립트:
  16개 조합(표현 4 × 모델 4) × 2 데이터셋 각각에 대해
    - Murphy 분해 보정 기여율  (크기맞춤 R1 / 크기+유병률맞춤 R2 두 기준)
    - split conformal 주변·클래스별 커버리지 (alpha 0.1, 0.2)
  를 계산한다.

사용법:
    python src/grid_calibration.py [--reps N] [--only-decomp | --only-conformal]

출력:
    results/grid_calibration.csv
    results/grid_calibration_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
N_REP_DEFAULT = 15  # robustness.py 와 동일
CALIB_FRAC = 0.25  # conformal_prediction.py 와 동일
ALPHAS = [0.10, 0.20]
log = S.Log()


# ------------------------------------------------------------ 표현 (robustness.py 와 동일)
def _mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


# 표현 정의는 _shared.grid_reps() 한 곳에만 있다 (2026-10-01 통합).
# 위 주석은 'robustness.py 와 동일' 이라고 적혀 있었지만 실제로는 dtype 이
# 달랐다 (이쪽 int8+float64 / 저쪽 float32). 자세한 경위는 grid_reps() 참고.
REPS = S.grid_reps()
rep_ecfp_desc = REPS["ECFP4+desc"]


def models():
    """모델 정의는 _shared.grid_models() 한 곳에만 있다 (2026-09-30 통합).
    예전에는 이 파일과 robustness.py 가 각자 정의해 하이퍼파라미터가 어긋났다."""
    return S.grid_models()


# ------------------------------------------------------------ conformal
def conformal_eval(p_calib, y_calib, p_test, y_test, alpha):
    """conformal_prediction.py 와 동일한 구현."""
    s_calib = 1 - np.where(y_calib == 1, p_calib, 1 - p_calib)
    n = len(s_calib)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(s_calib, q_level, method="higher")
    in1 = (1 - p_test) <= qhat
    in0 = (1 - (1 - p_test)) <= qhat
    size = in1.astype(int) + in0.astype(int)
    cov = np.where(y_test == 1, in1, in0)
    pos, neg = y_test == 1, y_test == 0
    return dict(
        marginal=cov.mean(),
        cov_pos=cov[pos].mean() if pos.sum() else np.nan,
        cov_neg=cov[neg].mean() if neg.sum() else np.nan,
        set_size=size.mean(),
        empty=(size == 0).mean(),
    )


def run(name, path, n_rep, do_decomp, do_conf):
    df = S.load(path)
    mols = _mols(df["smiles"].tolist())
    y = df["y"].values
    tr, te = S.temporal_masks(df)
    n_test = int(te.sum())
    rate_t = float(y[te].mean())
    order = np.argsort(df["approval_year"].values[tr], kind="stable")
    tr_idx = np.where(tr)[0][order]  # 학습군을 승인연도순
    n_cal = int(len(tr_idx) * CALIB_FRAC)
    fit_idx, cal_idx = tr_idx[:-n_cal], tr_idx[-n_cal:]

    log("=" * 96)
    log(
        f"{name}   학습 {tr.sum()} / 검증 {n_test}   "
        f"(conformal: 적합 {len(fit_idx)} / 보정 {len(cal_idx)})"
    )
    log("=" * 96)

    rows = []
    for rname, rfn in REPS.items():
        X = rfn(mols)
        for mname, mk in models().items():
            row = dict(dataset=name, rep=rname, model=mname)
            try:
                m = mk().fit(X[tr], y[tr])
                p_t = m.predict_proba(X[te])[:, 1]
                yte = y[te]
                row["auroc_temporal"] = roc_auc_score(yte, p_t)
                row["ece_temporal"] = S.ece(yte, p_t)

                if do_decomp:
                    dt = S.decompose(yte, p_t)
                    for base, target in [("r1", None), ("r2", rate_t)]:
                        acc = {
                            k: []
                            for k in ("reliability", "resolution", "uncertainty", "brier_approx")
                        }
                        ec = []
                        for i in range(n_rep):
                            r = np.random.default_rng(S.SEED + i)
                            idx = r.permutation(len(y))
                            ti, ri = idx[:n_test], idx[n_test:]
                            if target is not None:
                                ti = S.subsample_to_rate(y, ti, target, r)
                            if len(np.unique(y[ti])) < 2:
                                continue
                            mm = mk().fit(X[ri], y[ri])
                            pp = mm.predict_proba(X[ti])[:, 1]
                            d = S.decompose(y[ti], pp)
                            for k in acc:
                                acc[k].append(d[k])
                            ec.append(S.ece(y[ti], pp))
                        mean = {k: float(np.mean(v)) for k, v in acc.items()}
                        dR = dt["reliability"] - mean["reliability"]
                        dS = mean["resolution"] - dt["resolution"]
                        dU = dt["uncertainty"] - mean["uncertainty"]
                        dB = dt["brier_approx"] - mean["brier_approx"]
                        row[f"{base}_ece_random"] = float(np.mean(ec))
                        row[f"{base}_ece_ratio"] = row["ece_temporal"] / float(np.mean(ec))
                        row[f"{base}_d_reliability"] = dR
                        row[f"{base}_d_resolution"] = dS
                        row[f"{base}_d_uncertainty"] = dU
                        row[f"{base}_share2"] = 100 * dR / (dR + dS) if (dR + dS) else np.nan
                        row[f"{base}_share3"] = 100 * dR / dB if dB else np.nan

                if do_conf:
                    mf = mk().fit(X[fit_idx], y[fit_idx])
                    p_cal = mf.predict_proba(X[cal_idx])[:, 1]
                    p_te = mf.predict_proba(X[te])[:, 1]
                    for a in ALPHAS:
                        c = conformal_eval(p_cal, y[cal_idx], p_te, yte, a)
                        tag = f"conf{int((1-a)*100)}"
                        row[f"{tag}_marginal"] = c["marginal"]
                        row[f"{tag}_cov_pos"] = c["cov_pos"]
                        row[f"{tag}_cov_neg"] = c["cov_neg"]
                        row[f"{tag}_sep"] = c["cov_pos"] - c["cov_neg"]
                        row[f"{tag}_empty"] = c["empty"]
                rows.append(row)
                msg = f"  {rname:12s} {mname:13s} AUROC {row['auroc_temporal']:.4f}"
                if do_decomp:
                    msg += (
                        f"  ECE배율 R1 {row['r1_ece_ratio']:.2f} / "
                        f"R2 {row['r2_ece_ratio']:.2f}"
                        f"  보정기여 R1 {row['r1_share2']:5.1f}% / "
                        f"R2 {row['r2_share2']:5.1f}%"
                    )
                if do_conf:
                    msg += f"  conf90 {row['conf90_marginal']:.3f}"
                log(msg)
            except Exception as e:
                log(f"  {rname:12s} {mname:13s} 실패: {str(e)[:60]}")
    log("")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP_DEFAULT)
    ap.add_argument("--only-decomp", action="store_true")
    ap.add_argument("--only-conformal", action="store_true")
    a = ap.parse_args()
    do_d = not a.only_conformal
    do_c = not a.only_decomp

    log("격자 확장 — Brier 분해 + conformal 커버리지를 모델×표현 전체로")
    log(f"무작위 대조군 {a.reps}회, seed {S.SEED}, 절단 {S.CUTOFF}")
    log("R1 = 크기만 맞춘 무작위 / R2 = 크기+유병률 맞춘 무작위\n")

    out = []
    for nm, fn in S.DATASETS:
        p = PROC / fn
        if p.exists():
            out.append(run(nm, p, a.reps, do_d, do_c))
    if not out:
        log("데이터 없음")
        return
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "grid_calibration.csv", index=False)

    log("=" * 96)
    log("일관성 판정")
    log("=" * 96)
    for ds in df.dataset.unique():
        s = df[df.dataset == ds]
        log(f"\n{ds}  (조합 {len(s)}개)")
        if do_d:
            for base, lab in [("r1", "R1 크기맞춤"), ("r2", "R2 크기+유병률맞춤")]:
                v = s[f"{base}_share2"].dropna()
                if not len(v):
                    continue
                log(
                    f"  [{lab}] 보정 기여율 중앙값 {v.median():.1f}%  "
                    f"범위 {v.min():.1f}~{v.max():.1f}%  "
                    f"50% 초과 조합 {(v > 50).sum()}/{len(v)}"
                )
                r = s[f"{base}_ece_ratio"].dropna()
                log(
                    f"           ECE 배율 중앙값 {r.median():.2f}  "
                    f"범위 {r.min():.2f}~{r.max():.2f}  "
                    f"1 초과 조합 {(r > 1).sum()}/{len(r)}"
                )
        if do_c:
            for tag, t in [("conf90", 0.90), ("conf80", 0.80)]:
                mg = s[f"{tag}_marginal"].dropna()
                sp = s[f"{tag}_sep"].dropna()
                if not len(mg):
                    continue
                log(
                    f"  [conformal 목표 {t:.0%}] 주변 커버리지 중앙값 "
                    f"{mg.median():.3f}  범위 {mg.min():.3f}~{mg.max():.3f}  "
                    f"목표 미달 조합 {(mg < t).sum()}/{len(mg)}"
                )
                log(
                    f"           양−음 분리 중앙값 {sp.median():.3f}  "
                    f"범위 {sp.min():.3f}~{sp.max():.3f}"
                )
    log("")
    log("[읽는 법] 보고할 값은 R2 기준이다. R1 은 유병률 이동이 섞여 있다.")
    log("          '50% 초과 조합' 이 16/16 이면 보정 우세가 모델 특이적이지 않다.")
    log.save(RES / "grid_calibration_summary.txt")
    print(f"\n저장: {RES/'grid_calibration.csv'}")
    print(f"      {RES/'grid_calibration_summary.txt'}")


if __name__ == "__main__":
    main()
