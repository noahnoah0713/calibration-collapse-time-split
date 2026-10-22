"""
Brier score의 Murphy 분해 (reliability - resolution + uncertainty)

목적:
    "보정 붕괴가 판별력 저하보다 크다"는 주장(§4)을 수사적 표현
    ("9% 대 39%, 4배 이상")이 아니라 Brier score의 표준 분해식으로
    정량화한다.

        BS = reliability - resolution + uncertainty

    - uncertainty : 검증군 자체의 클래스 불확실성 (모델과 무관, obar*(1-obar))
    - resolution  : 모델이 서로 다른 사례를 얼마나 구별해내는가 (높을수록 좋음,
                    판별력에 대응하는 항)
    - reliability : 예측확률이 실제 관측빈도와 얼마나 일치하는가 (낮을수록
                    좋음, 보정에 대응하는 항)

    시간분할과 무작위 분할 사이 Brier 변화(ΔBS)를 reliability 변화분과
    resolution 변화분으로 쪼개서, "판별력 저하 때문에 늘어난 오차"와
    "보정 붕괴 때문에 늘어난 오차"의 기여도를 각각 %로 제시한다.

사용법:
    python src/brier_decomposition.py

출력:
    results/brier_decomposition.csv
    results/brier_decomposition_summary.txt

주의:
    다른 스크립트(robustness.py 등)와 동일하게 ECFP4+desc / XGBoost
    조합을 사용한다.

    무작위 기준을 두 가지 방식으로 계산한다.

      (A) 분할별 평균  [주 분석]
          N_REPEATS회 각각에 대해 시간분할과 동일한 검증군 크기로
          분해한 뒤 항별로 평균낸다. reliability는 표본이 작을수록
          잡음 때문에 위로 치우치므로, 양쪽을 같은 표본 크기에서
          재야 비교가 공정하다.

      (B) 풀링        [민감도]
          N_REPEATS회를 전부 합쳐 한 번에 구간화한다. 무작위 쪽만
          N_REPEATS배 많은 표본을 쓰게 되어 그쪽 reliability가
          낮아지고, 결과적으로 보정 기여율이 과대평가된다.

    보정 기여율에는 검증군 부트스트랩(N_BOOT회)으로 95% 구간을 붙인다.
    이 구간은 검증군 표집 변동만 반영하고 무작위 기준의 변동은 고정으로
    둔다. 무작위 쪽 변동은 분할 간 SD로 따로 보고한다.
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
N_BINS = 10
N_REPEATS = 20  # 무작위 분할 반복 횟수 (temporal_split.py와 동일)
N_BOOT = 2000  # 기여율 신뢰구간용 부트스트랩 반복

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


# ------------------------------------------------------------- 공통
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


def bin_table(y, p, n_bins=N_BINS):
    """(f_k, o_k, n_k) 리스트 반환."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        hi = (p <= 1) if i == n_bins - 1 else (p < edges[i + 1])
        m = (p >= edges[i]) & hi
        if m.sum() > 0:
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return rows


def murphy_decomposition(bins):
    N = sum(n for _, _, n in bins)
    obar = sum(o * n for _, o, n in bins) / N
    uncertainty = obar * (1 - obar)
    resolution = sum(n * (o - obar) ** 2 for _, o, n in bins) / N
    reliability = sum(n * (f - o) ** 2 for f, o, n in bins) / N
    bs = uncertainty - resolution + reliability
    return dict(
        N=N,
        obar=obar,
        uncertainty=uncertainty,
        resolution=resolution,
        reliability=reliability,
        brier_approx=bs,
    )


def get_temporal_probs(df):
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr
    m = make_model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    return y[te], p


def get_random_splits(df, n_test):
    """무작위 분할 N_REPEATS회. 각 회차의 (y, p)를 리스트로 반환."""
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    out = []
    rng = np.random.default_rng(SEED)
    for i in range(N_REPEATS):
        idx = rng.permutation(len(y))
        te_idx, tr_idx = idx[:n_test], idx[n_test:]
        m = make_model().fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te_idx])[:, 1]
        out.append((y[te_idx], p))
    return out


def shares(rel_r, res_r, rel_t, res_t):
    """보정/판별력 기여율(%). 분모가 0 이하이면 (nan, nan)."""
    d_rel = rel_t - rel_r  # 보정 악화분
    d_res = res_r - res_t  # 판별력 저하분
    tot = d_rel + d_res
    if tot <= 0:
        return np.nan, np.nan, d_res, d_rel
    return 100 * d_rel / tot, 100 * d_res / tot, d_res, d_rel


def boot_share_ci(y_t, p_t, rel_r, res_r, n_boot=N_BOOT, seed=SEED):
    """검증군을 부트스트랩해 보정 기여율의 95% 구간을 구한다.

    주의: 무작위 기준(rel_r, res_r)은 고정한다. 따라서 이 구간은 검증군
    표집에서 오는 불확실성만 담고, 무작위 분할 선택에서 오는 불확실성은
    담지 않는다. 후자는 분할 간 SD로 따로 보고한다. 논문에 쓸 때 이
    범위를 그대로 밝힐 것.
    """
    rng = np.random.default_rng(seed)
    n = len(y_t)
    vals = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        d = murphy_decomposition(bin_table(y_t[s], p_t[s]))
        v = shares(rel_r, res_r, d["reliability"], d["resolution"])[0]
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 100:
        return np.nan, np.nan, len(vals), np.nan
    over = float(np.mean(np.array(vals) > 100))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals), over)


def run_dataset(name, path):
    if not path.exists():
        log(f"  {name} 건너뜀: {path} 없음")
        return None

    df = load(path)
    y_t, p_t = get_temporal_probs(df)
    n_test = y_t.shape[0]
    splits = get_random_splits(df, n_test=n_test)

    dt = murphy_decomposition(bin_table(y_t, p_t))

    # (A) 분할별 평균 - 주 분석
    per = [murphy_decomposition(bin_table(yy, pp)) for yy, pp in splits]
    dr_mean = {
        k: float(np.mean([d[k] for d in per]))
        for k in ("uncertainty", "resolution", "reliability", "brier_approx")
    }
    rel_sd = float(np.std([d["reliability"] for d in per], ddof=1))
    res_sd = float(np.std([d["resolution"] for d in per], ddof=1))

    # (B) 풀링 - 민감도
    y_pool = np.concatenate([yy for yy, _ in splits])
    p_pool = np.concatenate([pp for _, pp in splits])
    dr_pool = murphy_decomposition(bin_table(y_pool, p_pool))

    pct_rel_A, pct_res_A, d_res_A, d_rel_A = shares(
        dr_mean["reliability"], dr_mean["resolution"], dt["reliability"], dt["resolution"]
    )
    pct_rel_B, pct_res_B, d_res_B, d_rel_B = shares(
        dr_pool["reliability"], dr_pool["resolution"], dt["reliability"], dt["resolution"]
    )

    lo, hi, n_ok, over100 = boot_share_ci(y_t, p_t, dr_mean["reliability"], dr_mean["resolution"])

    log(f"\n{'='*72}\n{name}   (검증군 n={n_test})\n{'='*72}")
    log(f"{'':14s}{'무작위 분할별 평균':>20s}{'시간분할':>14s}{'차이':>12s}")
    for k, label in [
        ("uncertainty", "uncertainty"),
        ("resolution", "resolution (판별력, 높을수록 좋음)"),
        ("reliability", "reliability (보정, 낮을수록 좋음)"),
        ("brier_approx", "Brier (근사)"),
    ]:
        log(f"  {label:34s} {dr_mean[k]:>10.4f} {dt[k]:>14.4f} {dt[k]-dr_mean[k]:>+12.4f}")
    log(f"  (무작위 분할 간 SD: reliability {rel_sd:.4f}, resolution {res_sd:.4f})")

    log("\n  [A] 주 분석 - 분할별 평균 기준")
    ci = f"[95% CI {lo:.1f}-{hi:.1f}]" if np.isfinite(lo) else "[CI 계산 불가]"
    log(f"      보정(reliability) 기여   {pct_rel_A:.1f}%  {ci}")
    log(f"      판별력(resolution) 기여  {pct_res_A:.1f}%")
    if np.isfinite(over100) and over100 > 0:
        log(
            f"      (부트스트랩 표본의 {over100:.1%}에서 기여율이 100%를 넘는다. "
            f"판별력이 오히려 개선된 재표본이며, 구간 상한이 100%를 넘는 이유다.)"
        )
    if d_res_A > 0:
        log(f"      보정/판별력 비율          {d_rel_A/d_res_A:.2f}배")

    log(f"\n  [B] 민감도 - 풀링 {N_REPEATS}회 기준 (무작위 쪽만 표본 {N_REPEATS}배)")
    log(f"      reliability {dr_pool['reliability']:.4f}  resolution {dr_pool['resolution']:.4f}")
    log(f"      보정 기여   {pct_rel_B:.1f}%   판별력 기여 {pct_res_B:.1f}%")
    log(f"      -> 풀링은 보정 기여율을 {pct_rel_B-pct_rel_A:+.1f}%p 만큼 높게 보고한다")

    return {
        "dataset": name,
        "N_temporal": dt["N"],
        "N_random_pooled": dr_pool["N"],
        "n_repeats": N_REPEATS,
        "resolution_temporal": dt["resolution"],
        "reliability_temporal": dt["reliability"],
        "brier_temporal": dt["brier_approx"],
        # (A) 주 분석
        "resolution_random_mean": dr_mean["resolution"],
        "reliability_random_mean": dr_mean["reliability"],
        "brier_random_mean": dr_mean["brier_approx"],
        "reliability_random_sd": rel_sd,
        "resolution_random_sd": res_sd,
        "d_resolution_A": d_res_A,
        "d_reliability_A": d_rel_A,
        "pct_from_reliability_A": pct_rel_A,
        "pct_from_resolution_A": pct_res_A,
        "pct_reliability_A_lo": lo,
        "pct_reliability_A_hi": hi,
        "n_boot_ok": n_ok,
        "frac_boot_over_100": over100,
        # (B) 민감도
        "resolution_random_pooled": dr_pool["resolution"],
        "reliability_random_pooled": dr_pool["reliability"],
        "brier_random_pooled": dr_pool["brier_approx"],
        "pct_from_reliability_B": pct_rel_B,
        "pct_from_resolution_B": pct_res_B,
    }


def main():
    log("Brier score Murphy 분해: 판별력 저하 vs 보정 악화 기여도")
    log(f"주 분석 = 무작위 분할별 평균 (양쪽 동일 표본 크기), 민감도 = 풀링 {N_REPEATS}회\n")

    rows = []
    r1 = run_dataset("DILIrank", PROC / "dilirank2_with_year.csv")
    if r1:
        rows.append(r1)
    r2 = run_dataset("DICTrank", PROC / "dictrank_processed.csv")
    if r2:
        rows.append(r2)

    if rows:
        pd.DataFrame(rows).to_csv(RES / "brier_decomposition.csv", index=False)
    (RES / "brier_decomposition_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'brier_decomposition.csv'}")
    log(f"      {RES/'brier_decomposition_summary.txt'}")


if __name__ == "__main__":
    main()
