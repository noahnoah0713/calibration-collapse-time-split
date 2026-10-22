"""
정렬 동점 처리 민감도 — §10(재보정)·§11(conformal) 이 임의 선택에 얼마나 의존하나
(2026-09-22 추가)

왜 필요한가
----------
§10·§11 은 학습군을 승인연도순으로 정렬한 뒤 '최근 CALIB_FRAC' 를 내부
보정셋으로 쓴다. 그런데 경계 연도에 화합물이 몰려 있다:

    DILIrank  경계 1997년  그 해 학습군 16개 중 2개만 보정셋에 들어간다
    DICTrank  경계 1998년  그 해 학습군 20개 중 3개만 보정셋에 들어간다

나머지 14개 / 17개를 누가 대신할지는 정렬 함수의 동점 처리가 정한다.

    conformal_prediction.py : np.argsort(yr[train_idx])   -> quicksort (불안정)
    recalibration.py        : df.sort_values("approval_year")  -> quicksort (불안정)
    grid_calibration.py     : np.argsort(..., kind="stable")   -> 안정

셋 다 난수가 없어 재현은 되지만 어느 것도 원리적 근거가 없다. 실제로
§11 의 DILIrank 90% 주변 커버리지는 0.737, 같은 조합을 안정 정렬로
계산한 §9.6 은 0.705 다. 이 0.032 차이가 결론을 위협하는지 재야 한다.

게다가 행 순서가 바뀌면 XGBoost 의 subsample/colsample 이 다른 행을
뽑으므로 모델 자체가 달라진다. 두 효과를 분리한다.

실험 3종
--------
  E1. 경계 동점 배정만 변화 — 경계연도 화합물 중 누가 보정셋에 들어갈지를
      무작위로 고르고, 적합셋 행 순서는 항상 정규화(연도→원본 index)
  E2. 적합셋 행 순서만 변화 — 보정셋은 안정 정렬로 고정, 적합셋 행 순서만 섞음
  E3. 현행 구현 3종 재현 — np.argsort 기본 / pandas sort_values 기본 / 안정 정렬

읽는 법
------
E1 의 변동폭이 크면 §10·§11 수치를 단일 값으로 쓸 수 없고 범위로 써야 한다.
E2 의 변동폭이 크면 그 수치는 모델 적합의 잡음이지 시간분할의 성질이 아니다.
둘 다 작으면 §10·§11 은 그대로 쓸 수 있다.

사용법
------
    python src/tiebreak_sensitivity.py [--reps 20]

출력
----
    results/tiebreak_sensitivity.csv
    results/tiebreak_sensitivity_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CALIB_FRAC = 0.25  # recalibration.py / conformal_prediction.py 와 동일
ALPHAS = [0.10, 0.20]
log = S.Log()


# ------------------------------------------------------------ 재보정 (recalibration.py 와 동일)
def to_logit(p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 다. 캐스팅하지
    # 않으면 로짓과 로지스틱 적합이 단정밀도로 일어난다
    # (src/diff_ladder.py 단계 10). float64 로 통일한다.
    p = np.asarray(p, dtype=np.float64)
    return np.log(np.clip(p, S.EPS, 1 - S.EPS) / np.clip(1 - p, S.EPS, 1 - S.EPS))


def platt(p_cal, y_cal, p_test):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(to_logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(to_logit(p_test).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_test):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(p_cal, y_cal)
    return np.clip(ir.predict(p_test), S.EPS, 1 - S.EPS)


# ------------------------------------------------------------ conformal (conformal_prediction.py 와 동일)
def conformal_eval(p_calib, y_calib, p_test, y_test, alpha):
    s_calib = 1 - np.where(y_calib == 1, p_calib, 1 - p_calib)
    n = len(s_calib)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(s_calib, q_level, method="higher")
    in1 = (1 - p_test) <= qhat
    in0 = p_test <= qhat
    cov = np.where(y_test == 1, in1, in0)
    pos, neg = y_test == 1, y_test == 0
    return dict(
        marginal=float(cov.mean()),
        cov_pos=float(cov[pos].mean()) if pos.any() else np.nan,
        cov_neg=float(cov[neg].mean()) if neg.any() else np.nan,
        empty=float(((~in0) & (~in1)).mean()),
    )


# ------------------------------------------------------------ 분할 변형
def canonical_order(yr, idx):
    """연도 오름차순, 동률은 원본 행 번호 오름차순. 유일하게 정해지는 순서."""
    return idx[np.lexsort((idx, yr[idx]))]


def variant_indices(kind, yr, tr_idx, n_cal, rng=None, df=None):
    """(fit_idx, cal_idx) 를 kind 에 따라 만든다. 둘 다 모델에 먹일 순서 그대로."""
    if kind == "stable":  # grid_calibration.py / 보충 분석
        srt = tr_idx[np.argsort(yr[tr_idx], kind="stable")]
        return srt[:-n_cal], srt[-n_cal:]
    if kind == "npargsort":  # conformal_prediction.py 현행
        srt = tr_idx[np.argsort(yr[tr_idx])]
        return srt[:-n_cal], srt[-n_cal:]
    if kind == "pandas":  # recalibration.py 현행
        srt = np.asarray(pd.Series(yr[tr_idx], index=tr_idx).sort_values().index.values)
        return srt[:-n_cal], srt[-n_cal:]
    if kind == "E1":  # 경계 동점 배정만 무작위
        srt = canonical_order(yr, tr_idx)
        boundary = yr[srt[-n_cal]]
        above = srt[yr[srt] > boundary]  # 무조건 보정셋
        tie = srt[yr[srt] == boundary]  # 경쟁 구간
        below = srt[yr[srt] < boundary]  # 무조건 적합셋
        k = n_cal - len(above)
        pick = rng.choice(tie, size=k, replace=False) if k > 0 else np.array([], int)
        cal = np.concatenate([np.sort(pick), above])
        fit = canonical_order(yr, np.setdiff1d(tr_idx, cal, assume_unique=False))
        return fit, cal
    if kind == "E2":  # 적합셋 행 순서만 섞음
        srt = canonical_order(yr, tr_idx)
        fit, cal = srt[:-n_cal], srt[-n_cal:]
        fit = fit.copy()
        rng.shuffle(fit)
        return fit, cal
    raise ValueError(kind)


# ------------------------------------------------------------ 한 변형 평가
def evaluate(X, y, te, fit_idx, cal_idx, A_ece, E_ece):
    m = S.make_model().fit(X[fit_idx], y[fit_idx])
    p_cal = m.predict_proba(X[cal_idx])[:, 1]
    p_te = m.predict_proba(X[te])[:, 1]
    yte, ycal = y[te], y[cal_idx]

    p_B = platt(p_cal, ycal, p_te)
    p_C = isotonic(p_cal, ycal, p_te)
    den = A_ece - E_ece
    row = dict(
        n_fit=len(fit_idx),
        n_cal=len(cal_idx),
        auroc=roc_auc_score(yte, p_te),
        ece_Ap=S.ece(yte, p_te),
        ece_B=S.ece(yte, p_B),
        ece_C=S.ece(yte, p_C),
    )
    row["rec_B"] = 100 * (A_ece - row["ece_B"]) / den
    row["rec_C"] = 100 * (A_ece - row["ece_C"]) / den
    for a in ALPHAS:
        c = conformal_eval(p_cal, ycal, p_te, yte, a)
        t = "conf%d" % int((1 - a) * 100)
        row[t + "_marginal"] = c["marginal"]
        row[t + "_pos"] = c["cov_pos"]
        row[t + "_neg"] = c["cov_neg"]
        row[t + "_empty"] = c["empty"]
    return row


# ------------------------------------------------------------ 데이터셋 1개
def run(name, path, reps):
    df = S.load(DATA / path)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr, te = S.temporal_masks(df)
    tr_idx = np.where(tr)[0]
    n_cal = int(len(tr_idx) * CALIB_FRAC)

    srt = canonical_order(yr, tr_idx)
    boundary = int(yr[srt[-n_cal]])
    n_tie = int((yr[tr_idx] == boundary).sum())
    n_tie_in = int((yr[srt[-n_cal:]] == boundary).sum())

    log("=" * 92)
    log(f"{name}   학습 {len(tr_idx)} / 검증 {int(te.sum())}   보정셋 {n_cal}")
    log(f"  경계 연도 {boundary} — 그 해 학습군 {n_tie}개 중 {n_tie_in}개만 보정셋")
    log(f"  => 나머지 {n_tie - n_tie_in}개의 배정은 정렬 동점 처리가 정하는 임의 선택")
    log("=" * 92)

    # 기준값 A(전체 학습 무보정) 와 E(Oracle) — 변형과 무관하게 한 번만, 정규 순서로
    full = canonical_order(yr, tr_idx)
    m_full = S.make_model().fit(X[full], y[full])
    p_base = m_full.predict_proba(X[te])[:, 1]
    A_ece = S.ece(y[te], p_base)
    E_ece = S.ece(y[te], platt(p_base, y[te], p_base))
    log(
        f"\n  기준 A 무보정 ECE {A_ece:.4f} / E Oracle ECE {E_ece:.4f} "
        f"(회복 가능분 {A_ece - E_ece:.4f}) — 정규 순서 고정"
    )

    rows = []

    # --- E3. 현행 구현 3종
    log("\n--- E3. 현행 구현 3종 재현 ---")
    for kind, tag in (
        ("npargsort", "np.argsort 기본 (§11 현행)"),
        ("pandas", "pandas sort_values 기본 (§10 현행)"),
        ("stable", "안정 정렬 (§9.6 보충분석)"),
    ):
        fit_idx, cal_idx = variant_indices(kind, yr, tr_idx, n_cal)
        r = evaluate(X, y, te, fit_idx, cal_idx, A_ece, E_ece)
        r.update(dataset=name, experiment="E3. 현행 구현", variant=tag, seed="")
        rows.append(r)
        log(
            f"  {tag:32s} conf90 {r['conf90_marginal']:.4f}  conf80 {r['conf80_marginal']:.4f}"
            f"  B회복 {r['rec_B']:5.1f}%  C회복 {r['rec_C']:5.1f}%"
        )
    # 세 변형의 보정셋이 실제로 다른지
    sets = {}
    for kind in ("npargsort", "pandas", "stable"):
        _, c = variant_indices(kind, yr, tr_idx, n_cal)
        sets[kind] = set(c.tolist())
    log(
        f"  보정셋 구성 비교: np.argsort∩pandas {len(sets['npargsort'] & sets['pandas'])}/{n_cal}, "
        f"np.argsort∩stable {len(sets['npargsort'] & sets['stable'])}/{n_cal}"
    )

    # --- E1. 경계 동점 배정만
    log(f"\n--- E1. 경계 동점 배정만 무작위 ({reps}회) ---")
    for i in range(reps):
        rng = np.random.default_rng(S.SEED + i)
        fit_idx, cal_idx = variant_indices("E1", yr, tr_idx, n_cal, rng)
        r = evaluate(X, y, te, fit_idx, cal_idx, A_ece, E_ece)
        r.update(dataset=name, experiment="E1. 경계 동점 배정", variant="무작위", seed=S.SEED + i)
        rows.append(r)
    sub = pd.DataFrame([r for r in rows if r["experiment"].startswith("E1")])
    for col, lab in (
        ("conf90_marginal", "conf90 주변커버리지"),
        ("conf80_marginal", "conf80 주변커버리지"),
        ("rec_B", "B Platt 회복률 %"),
        ("rec_C", "C Isotonic 회복률 %"),
        ("auroc", "AUROC"),
    ):
        log(
            f"  {lab:22s} 중앙값 {sub[col].median():8.4f}  "
            f"범위 {sub[col].min():8.4f}–{sub[col].max():8.4f}  폭 {sub[col].max()-sub[col].min():.4f}"
        )

    # --- E2. 적합셋 행 순서만
    log(f"\n--- E2. 적합셋 행 순서만 무작위 ({reps}회, 보정셋 고정) ---")
    for i in range(reps):
        rng = np.random.default_rng(1000 + S.SEED + i)
        fit_idx, cal_idx = variant_indices("E2", yr, tr_idx, n_cal, rng)
        r = evaluate(X, y, te, fit_idx, cal_idx, A_ece, E_ece)
        r.update(
            dataset=name, experiment="E2. 적합셋 행 순서", variant="무작위", seed=1000 + S.SEED + i
        )
        rows.append(r)
    sub = pd.DataFrame([r for r in rows if r["experiment"].startswith("E2")])
    for col, lab in (
        ("conf90_marginal", "conf90 주변커버리지"),
        ("conf80_marginal", "conf80 주변커버리지"),
        ("rec_B", "B Platt 회복률 %"),
        ("rec_C", "C Isotonic 회복률 %"),
        ("auroc", "AUROC"),
    ):
        log(
            f"  {lab:22s} 중앙값 {sub[col].median():8.4f}  "
            f"범위 {sub[col].min():8.4f}–{sub[col].max():8.4f}  폭 {sub[col].max()-sub[col].min():.4f}"
        )

    for r in rows:
        r["A_ece"] = A_ece
        r["E_ece"] = E_ece
        r["boundary_year"] = boundary
        r["n_tie"] = n_tie
        r["n_tie_in_calib"] = n_tie_in
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    log("정렬 동점 처리 민감도 — §10·§11 의 임의 선택 의존도")
    log(f"반복 {a.reps}회, seed {S.SEED}, 절단 {S.CUTOFF}, 보정셋 비율 {CALIB_FRAC}")
    log("")

    allrows = []
    for name, f in S.DATASETS:
        allrows += run(name, f, a.reps)
        log("")

    out = pd.DataFrame(allrows)
    cols = [
        "dataset",
        "experiment",
        "variant",
        "seed",
        "boundary_year",
        "n_tie",
        "n_tie_in_calib",
        "n_fit",
        "n_cal",
        "auroc",
        "A_ece",
        "E_ece",
        "ece_Ap",
        "ece_B",
        "ece_C",
        "rec_B",
        "rec_C",
        "conf90_marginal",
        "conf90_pos",
        "conf90_neg",
        "conf90_empty",
        "conf80_marginal",
        "conf80_pos",
        "conf80_neg",
        "conf80_empty",
    ]
    out = out[cols]
    out.to_csv(RES / "tiebreak_sensitivity.csv", index=False)

    # ------------------------------------------------ 판정
    log("=" * 92)
    log("판정")
    log("=" * 92)
    log("")
    log(
        f"{'데이터셋':10s} {'지표':22s} {'E3 현행 3종':>22s} {'E1 동점 폭':>12s} {'E2 행순서 폭':>13s}"
    )
    log("-" * 92)
    for name, _ in S.DATASETS:
        d = out[out.dataset == name]
        e3 = d[d.experiment.str.startswith("E3")]
        for col, lab in (
            ("conf90_marginal", "conf90 주변커버리지"),
            ("conf80_marginal", "conf80 주변커버리지"),
            ("rec_B", "B Platt 회복률 %"),
            ("rec_C", "C Isotonic 회복률 %"),
            ("auroc", "AUROC"),
        ):
            r3 = "%.4f / %.4f / %.4f" % tuple(e3[col].tolist())
            w1 = d[d.experiment.str.startswith("E1")][col]
            w2 = d[d.experiment.str.startswith("E2")][col]
            log(
                f"{name:10s} {lab:22s} {r3:>22s} {w1.max()-w1.min():12.4f} {w2.max()-w2.min():13.4f}"
            )
        log("-" * 92)
    log("")
    log("[E3 열 순서] np.argsort 기본(§11 현행) / pandas 기본(§10 현행) / 안정 정렬(§9.6)")
    log("[읽는 법] E1 폭이 크면 §10·§11 수치를 범위로 써야 한다.")
    log("          E2 폭이 크면 그 수치는 모델 적합 잡음이지 시간분할의 성질이 아니다.")
    log(
        "          E1·E2 폭이 모두 결론의 간격(예: 0.90 대 0.74)보다 훨씬 작으면 현행 값을 그대로 쓴다."
    )
    log("")
    log(f"저장: {RES / 'tiebreak_sensitivity.csv'}")
    log(f"      {RES / 'tiebreak_sensitivity_summary.txt'}")
    log.save(RES / "tiebreak_sensitivity_summary.txt")


if __name__ == "__main__":
    main()
