"""
보정 붕괴의 유병률 교란 배제 + 신뢰구간

문제 제기 (예상 심사 지적):
  "시간분할에서 ECE가 나빠지는 것은 학습군과 검증군의 양성 비율이
   다르기 때문 아닌가? 무작위 분할은 비율이 같으니 유리한 것 아닌가?"

검증 설계:
  R1. 무작위 분할 (표준)              - 검증군 비율 = 전체 비율
  R2. 무작위 분할 + 비율 맞춤          - 검증군 비율을 시간분할과 동일하게 재표집
  R3. 무작위 분할 + 학습군까지 비율 맞춤 - 학습/검증 비율을 시간분할과 동일하게
  T.  시간분할

  R3에서도 ECE가 낮게 유지되면, 보정 붕괴는 비율 차이가 아니라
  시간에 따른 분포 이동 자체에서 기인함이 확인된다.

추가:
  - ECE 배율의 부트스트랩 95% 신뢰구간
  - 시간분할 ECE vs 무작위 ECE 분포의 위치 (순열 검정)
    p 는 Phipson-Smyth 편향보정 추정량 (1+r)/(1+n) 이다. 단순 비율은
    0 이 될 수 있지만 순열 p-value 는 0 이 될 수 없다.
    Phipson B, Smyth GK. Stat Appl Genet Mol Biol. 2010;9(1):Article39.

사용법:
    python src/prevalence_control.py

출력:
    results/prevalence_control.csv
    results/prevalence_control_summary.txt
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
N_REP = int(os.environ.get("N_REP", "200"))  # 무작위 분할 반복
#   30회였을 때 순열 p 의 하한이 1/31 = 0.0323 이어서 여섯 개
#   대조군의 p 가 전부 같은 값으로 찍혔다. 200회면 하한이
#   1/201 = 0.0050 으로 내려가 FDR 보정에 여유가 생긴다.
#   빠른 확인이 필요하면 N_REP=30 python src/prevalence_control.py
N_BOOT = 2000  # 부트스트랩
N_BINS = 10
EPS = 1e-6

rng = np.random.default_rng(SEED)
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


def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        hi = p <= 1 if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            tot += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return tot


def calib_slope(y, p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 다. 캐스팅하지
    # 않으면 로짓과 로지스틱 적합이 단정밀도로 일어난다
    # (src/diff_ladder.py 단계 10). float64 로 통일한다.
    p = np.asarray(p, dtype=np.float64)
    lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def subsample_to_rate(y, idx, target_rate, r):
    """idx 중 일부를 골라 양성 비율을 target_rate에 맞춘다 (최대 크기 유지)."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    # 양성 전부 사용 시 필요한 음성 수
    n_neg_a = int(round(len(pos) * (1 - target_rate) / max(target_rate, EPS)))
    # 음성 전부 사용 시 필요한 양성 수
    n_pos_b = int(round(len(neg) * target_rate / max(1 - target_rate, EPS)))
    if n_neg_a <= len(neg):
        sel = np.concatenate([pos, r.choice(neg, n_neg_a, replace=False)])
    elif n_pos_b <= len(pos):
        sel = np.concatenate([r.choice(pos, n_pos_b, replace=False), neg])
    else:
        return idx
    return sel


def boot_ci(func, *args, n=N_BOOT):
    """부트스트랩 백분위 신뢰구간."""
    vals = []
    n_obs = len(args[0])
    for _ in range(n):
        s = rng.choice(n_obs, n_obs, replace=True)
        try:
            v = func(*[a[s] for a in args])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run(name, path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.reset_index(drop=True)

    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr
    n_test = int(te.sum())
    rate_t = float(y[te].mean())
    rate_tr = float(y[tr].mean())

    log("=" * 70)
    log(f"{name}")
    log("=" * 70)
    log(f"  학습군 n={tr.sum()}, 양성 {rate_tr:.1%}")
    log(f"  검증군 n={n_test}, 양성 {rate_t:.1%}")

    # ---------------------------------------------------- T. 시간분할
    m = make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]
    yte = y[te]
    e_t, a_t, s_t = ece(yte, p_t), roc_auc_score(yte, p_t), calib_slope(yte, p_t)
    lo_e, hi_e = boot_ci(ece, yte, p_t)

    log("\n[T] 시간분할")
    log(f"    ECE {e_t:.4f} [95% CI {lo_e:.4f}-{hi_e:.4f}]")
    log(f"    AUROC {a_t:.4f} | 보정기울기 {s_t:.4f}")

    rows = [
        {
            "setting": "T. 시간분할",
            "n": n_test,
            "pos_rate": rate_t,
            "ECE": e_t,
            "ECE_lo": lo_e,
            "ECE_hi": hi_e,
            "AUROC": a_t,
            "slope": s_t,
            "Brier": brier_score_loss(yte, p_t),
        }
    ]

    # ---------------------------------------------------- R1/R2/R3
    configs = [
        ("R1. 무작위 (표준)", False, False),
        ("R2. 무작위 + 검증군 비율맞춤", True, False),
        ("R3. 무작위 + 학습·검증 비율맞춤", True, True),
    ]

    for tag, match_test, match_train in configs:
        E, A, S, B, N, P = [], [], [], [], [], []
        for i in range(N_REP):
            r = np.random.default_rng(SEED + i)
            idx = r.permutation(len(y))
            ti, ri = idx[:n_test], idx[n_test:]

            if match_test:
                ti = subsample_to_rate(y, ti, rate_t, r)
            if match_train:
                ri = subsample_to_rate(y, ri, rate_tr, r)

            if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                continue
            if len(ti) < 30 or len(ri) < 100:
                continue

            mm = make_model().fit(X[ri], y[ri])
            pp = mm.predict_proba(X[ti])[:, 1]
            E.append(ece(y[ti], pp))
            A.append(roc_auc_score(y[ti], pp))
            S.append(calib_slope(y[ti], pp))
            B.append(brier_score_loss(y[ti], pp))
            N.append(len(ti))
            P.append(float(y[ti].mean()))

        if not E:
            log(f"\n[{tag}] 유효 반복 없음")
            continue

        log(f"\n[{tag}]  ({len(E)}회 반복)")
        log(f"    검증군 n={np.mean(N):.0f}, 양성 {np.mean(P):.1%}")
        log(f"    ECE {np.mean(E):.4f} ± {np.std(E):.4f}  " f"(범위 {min(E):.4f}-{max(E):.4f})")
        log(f"    AUROC {np.mean(A):.4f} | 보정기울기 {np.mean(S):.4f}")
        log(f"    시간분할 대비 ECE 배율: {e_t/np.mean(E):.2f}배")

        # 순열 검정: 시간분할 ECE가 무작위 분포에서 얼마나 극단적인가
        #
        # 유한 번 추출한 순열에서 p 를 단순 비율로 내면 0 이 나올 수 있는데,
        # 순열 p-value 는 0 이 될 수 없다. 초과 사례가 하나도 없다는 것은
        # "p = 0" 이 아니라 "p < 1/(n+1)" 이라는 뜻이다. Phipson & Smyth
        # (2010) 의 편향보정 추정량 (1+r)/(1+n) 을 논문 값으로 쓴다.
        n_ge = int((np.array(E) >= e_t).sum())
        n_rep_valid = len(E)
        p_raw = n_ge / n_rep_valid
        p_perm = (1 + n_ge) / (1 + n_rep_valid)
        log(f"    무작위 ECE >= 시간분할 ECE : {n_ge}/{n_rep_valid}")
        log(
            f"    p (Phipson-Smyth) = {p_perm:.4f}   (단순 비율 {p_raw:.4f})"
            f"  {'*시간분할이 유의하게 나쁨*' if p_perm < 0.05 else ''}"
        )

        rows.append(
            {
                "setting": tag,
                "n": float(np.mean(N)),
                "pos_rate": float(np.mean(P)),
                "ECE": float(np.mean(E)),
                "ECE_sd": float(np.std(E)),
                "AUROC": float(np.mean(A)),
                "slope": float(np.mean(S)),
                "Brier": float(np.mean(B)),
                "ece_ratio": e_t / float(np.mean(E)),
                "n_rep_valid": n_rep_valid,
                "n_ge_observed": n_ge,
                "p_raw_proportion": float(p_raw),
                "p_perm": float(p_perm),
            }
        )

    out = pd.DataFrame(rows)
    out.insert(0, "dataset", name)

    # ---------------------------------------------------- 판정
    log("\n[판정]")
    r3 = out[out.setting.str.startswith("R3")]
    r1 = out[out.setting.str.startswith("R1")]
    if len(r3) and len(r1):
        log(
            f"  비율을 완전히 맞춘 무작위 분할(R3)에서도 "
            f"ECE {r3.iloc[0]['ECE']:.4f}로 낮게 유지됨"
        )
        log(f"  시간분할 ECE {e_t:.4f}는 그 {r3.iloc[0]['ece_ratio']:.2f}배")
        if r3.iloc[0]["p_perm"] < 0.05:
            log("  => 보정 붕괴는 유병률 차이가 아니라 " "시간에 따른 분포 이동에서 기인함")
        else:
            log("  => 유병률 교란을 배제할 수 없음 (재검토 필요)")
    log()
    return out


def main():
    res = []
    for nm, p in [
        ("DILIrank (간독성)", PROC / "dilirank2_with_year.csv"),
        ("DICTrank (심장독성)", PROC / "dictrank_processed.csv"),
    ]:
        if p.exists():
            res.append(run(nm, p))

    if not res:
        log("데이터 파일 없음")
        return

    allr = pd.concat(res, ignore_index=True)
    allr.to_csv(RES / "prevalence_control.csv", index=False)

    log("=" * 70)
    log("종합")
    log("=" * 70)
    for ds in allr["dataset"].unique():
        s = allr[allr.dataset == ds]
        t = s[s.setting.str.startswith("T")].iloc[0]
        log(f"\n{ds}   시간분할 ECE = {t['ECE']:.4f}")
        for _, r in s[s.setting.str.startswith("R")].iterrows():
            log(
                f"  {r['setting']:32s} ECE {r['ECE']:.4f} "
                f"(배율 {r['ece_ratio']:.2f}, p={r['p_perm']:.4f})"
            )

    (RES / "prevalence_control_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'prevalence_control.csv'}")
    log(f"      {RES/'prevalence_control_summary.txt'}")


if __name__ == "__main__":
    main()
