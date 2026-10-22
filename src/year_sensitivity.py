"""
승인연도 배정 오류에 대한 민감도 분석

배경
----
시간분할의 절단선(2010년)은 "승인연도"에 의존한다. 승인연도는
Drugs@FDA 의 Products/Submissions 기록에서 성분명으로 매칭해 붙였는데,
1938년 이전부터 쓰이던 고전 약물은 FDA 신약승인 기록 자체가 없거나
훨씬 나중의 제형 변경 승인만 남아 있다. check_years.py 가 이 문제를
검수해 두 종류를 찾아냈다.

  (A) 더 이른 Drugs@FDA 기록이 실제로 존재하는데 늦은 연도가 붙은 것
      -> results/year_corrections.csv 의 8건. 대체 연도가 문서로
         확인되므로 그 값으로 바꿔 다시 돌린다.

  (B) Drugs@FDA 기록상으로는 늦은 승인이 맞지만 분자 자체는 훨씬
      오래된 것. 예: Phenobarbital sodium 은 1912년부터 임상에서
      쓰였으나 Drugs@FDA 에 남은 최초 NDA 기록은 2022년이다.
      근거 있는 대체 연도가 없으므로 "값을 지어내지 않고 제외"하는
      쪽을 택한다. 제외는 보수적인 검정이다.

무엇을 확인하는가
-----------------
  세 가지 시나리오에서 시간분할을 다시 돌려, 논문의 두 핵심 수치가
  유지되는지 본다.
      - 시간분할 ECE 및 무작위 대비 배율
      - Brier 분해에서 보정(reliability) 기여율

  무작위 기준은 brier_decomposition.py 의 주 분석과 동일하게
  "시간분할과 같은 검증군 크기의 무작위 분할 N_REPEATS 회 평균"이다.
  연도만 바뀌는 S1 은 화합물 집합이 S0 과 같으므로 무작위 기준을
  다시 계산하지 않고 공유한다 (같은 값이 나오는 것이 당연하고,
  괜히 다시 돌리면 결과만 흔들린다).

사용법
------
    python src/year_sensitivity.py

출력
----
    results/year_sensitivity.csv
    results/year_sensitivity_summary.txt

의존
----
    results/year_corrections.csv   (check_years.py 가 만든다)
    src/brier_decomposition.py     (분해 함수를 그대로 가져다 쓴다)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 분해 수식은 주 분석 스크립트에서 그대로 가져온다 (구현 중복 금지)
from brier_decomposition import (  # noqa: E402
    CUTOFF,
    N_REPEATS,
    SEED,
    bin_table,
    featurize,
    make_model,
    murphy_decomposition,
    shares,
)

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

N_BINS = 10

DATASETS = [
    ("DILIrank", PROC / "dilirank2_with_year.csv"),
    ("DICTrank", PROC / "dictrank_processed.csv"),
]

# (B) 유형: 근거 있는 대체 연도가 없어 제외하는 화합물
DROP_UNCERTAIN = {
    "DILIrank": [
        ("Phenobarbital sodium", "1912년부터 임상 사용, Drugs@FDA 최초 NDA 기록은 2022년"),
    ],
    "DICTrank": [],
}

log_lines = []


def log(m=""):
    print(m, flush=True)
    log_lines.append(str(m))


def ece(y, p, n_bins=N_BINS):
    """Expected calibration error (동일 폭 10구간). prevalence_control.py 와 동일."""
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(y)
    for i in range(n_bins):
        hi = (p <= 1) if i == n_bins - 1 else (p < edges[i + 1])
        m = (p >= edges[i]) & hi
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def load(path):
    df = pd.read_csv(path).dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def temporal_fit(df):
    """시간분할 학습 후 검증군의 (y, p) 반환."""
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    m = make_model().fit(X[tr], y[tr])
    return y[~tr], m.predict_proba(X[~tr])[:, 1], int(tr.sum()), int((~tr).sum())


def random_baseline(df, n_test):
    """무작위 분할 N_REPEATS 회의 reliability/resolution/ECE 평균."""
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    rng = np.random.default_rng(SEED)
    rel, res, ec = [], [], []
    for _ in range(N_REPEATS):
        idx = rng.permutation(len(y))
        te, tr = idx[:n_test], idx[n_test:]
        m = make_model().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        d = murphy_decomposition(bin_table(y[te], p))
        rel.append(d["reliability"])
        res.append(d["resolution"])
        ec.append(ece(y[te], p))
    return (float(np.mean(rel)), float(np.mean(res)), float(np.mean(ec)), float(np.std(ec, ddof=1)))


def apply_corrections(df, corr, name):
    """year_corrections.csv 의 suggested_year 를 적용. 못 찾으면 멈춘다."""
    sub = corr[corr.dataset == name]
    df = df.copy()
    applied = []
    for r in sub.itertuples():
        m = df["CompoundName"].astype(str) == str(r.compound)
        if not m.any():
            raise SystemExit(
                f"[중단] {name}: '{r.compound}' 를 CompoundName 에서 찾지 못했습니다.\n"
                f"        year_corrections.csv 와 처리된 CSV 가 어긋났습니다. "
                f"check_years.py 를 다시 돌리세요."
            )
        old = int(df.loc[m, "approval_year"].iloc[0])
        df.loc[m, "approval_year"] = int(r.suggested_year)
        applied.append((r.compound, old, int(r.suggested_year), int(m.sum())))
    return df, applied


def evaluate(df, base_rel, base_res, base_ece):
    y_t, p_t, n_tr, n_te = temporal_fit(df)
    d = murphy_decomposition(bin_table(y_t, p_t))
    pct_rel, pct_res, d_res, d_rel = shares(base_rel, base_res, d["reliability"], d["resolution"])
    e = ece(y_t, p_t)
    return {
        "n": len(df),
        "n_train": n_tr,
        "n_test": n_te,
        "pos_rate_test": float(y_t.mean()),
        "AUROC": float(roc_auc_score(y_t, p_t)),
        "Brier": float(brier_score_loss(y_t, p_t)),
        "ECE": float(e),
        "ECE_random_mean": base_ece,
        "ece_ratio": float(e / base_ece) if base_ece else np.nan,
        "reliability": d["reliability"],
        "resolution": d["resolution"],
        "uncertainty": d["uncertainty"],
        "reliability_random_mean": base_rel,
        "resolution_random_mean": base_res,
        "pct_from_reliability": pct_rel,
        "pct_from_resolution": pct_res,
    }


def main():
    cpath = RES / "year_corrections.csv"
    if not cpath.exists():
        raise SystemExit(
            f"[중단] {cpath} 가 없습니다. 먼저 " f"python src/check_years.py 를 실행하세요."
        )
    corr = pd.read_csv(cpath)

    log("=" * 74)
    log("승인연도 배정 민감도 분석")
    log("=" * 74)
    log(f"절단 {CUTOFF}년 | 무작위 기준 {N_REPEATS}회 평균 | seed {SEED}")
    log(f"수정 후보 {len(corr)}건 (results/year_corrections.csv)")
    log()

    rows = []
    for name, path in DATASETS:
        if not path.exists():
            log(f"[건너뜀] {name}: {path} 없음")
            continue

        df0 = load(path)
        log("-" * 74)
        log(f"{name}  n={len(df0)}")
        log("-" * 74)

        # --- S1: 연도 수정 (화합물 집합은 그대로)
        df1, applied = apply_corrections(df0, corr, name)
        if applied:
            log("  적용한 연도 수정:")
            for cmp_, old, new, k in applied:
                moved = "검증군 -> 학습군" if old >= CUTOFF > new else "분할 변화 없음"
                log(f"    {cmp_:32s} {old} -> {new}   ({moved})")
        else:
            log("  이 데이터셋에 적용할 연도 수정 없음")

        # --- S2: S1 + 근거 없는 항목 제외
        drops = DROP_UNCERTAIN.get(name, [])
        df2 = df1.copy()
        for cmp_, why in drops:
            m = df2["CompoundName"].astype(str) == cmp_
            if not m.any():
                log(f"    [주의] 제외 대상 '{cmp_}' 가 데이터에 없습니다 (무시)")
                continue
            log(f"    제외: {cmp_}  ({why})")
            df2 = df2[~m].reset_index(drop=True)

        # --- 무작위 기준: 화합물 집합이 같으면 재사용
        n_test0 = int((df0["approval_year"].values >= CUTOFF).sum())
        log(f"  무작위 기준 계산 중 ({N_REPEATS}회, 검증군 크기 {n_test0}) ...")
        base_full = random_baseline(df0, n_test0)

        scenarios = [("S0 기준", df0, base_full), ("S1 연도수정", df1, base_full)]
        if len(df2) != len(df0):
            n_test2 = int((df2["approval_year"].values >= CUTOFF).sum())
            log(f"  무작위 기준 재계산 (집합 {len(df2)}개, 검증군 {n_test2}) ...")
            scenarios.append(("S2 연도수정+제외", df2, random_baseline(df2, n_test2)))
        else:
            scenarios.append(("S2 연도수정+제외", df2, base_full))

        log()
        for label, d, base in scenarios:
            r = evaluate(d, base[0], base[1], base[2])
            r.update(dataset=name, scenario=label)
            rows.append(r)
            log(
                f"  {label:16s} n={r['n']:4d} 검증군 {r['n_test']:3d} "
                f"양성 {r['pos_rate_test']:.1%}"
            )
            log(
                f"  {'':16s} AUROC {r['AUROC']:.4f} | "
                f"ECE {r['ECE']:.4f} (무작위 {r['ECE_random_mean']:.4f}, "
                f"배율 {r['ece_ratio']:.2f})"
            )
            log(
                f"  {'':16s} 보정 기여율 {r['pct_from_reliability']:.1f}% "
                f"(판별력 {r['pct_from_resolution']:.1f}%)"
            )
            log()

    if not rows:
        raise SystemExit("[중단] 평가된 시나리오가 없습니다.")

    out = pd.DataFrame(rows)[
        [
            "dataset",
            "scenario",
            "n",
            "n_train",
            "n_test",
            "pos_rate_test",
            "AUROC",
            "Brier",
            "ECE",
            "ECE_random_mean",
            "ece_ratio",
            "reliability",
            "resolution",
            "uncertainty",
            "reliability_random_mean",
            "resolution_random_mean",
            "pct_from_reliability",
            "pct_from_resolution",
        ]
    ]
    out.to_csv(RES / "year_sensitivity.csv", index=False)

    log("=" * 74)
    log("판정")
    log("=" * 74)
    for name in out.dataset.unique():
        g = out[out.dataset == name].set_index("scenario")
        b = g.loc["S0 기준"]
        log(f"\n  {name}")
        for sc in g.index:
            r = g.loc[sc]
            log(
                f"    {sc:16s} ECE 배율 {r['ece_ratio']:5.2f} "
                f"(기준 대비 {r['ece_ratio']-b['ece_ratio']:+.2f})   "
                f"보정 기여율 {r['pct_from_reliability']:5.1f}% "
                f"(기준 대비 {r['pct_from_reliability']-b['pct_from_reliability']:+.1f}pp)"
            )
        worst_e = (g["ece_ratio"] - b["ece_ratio"]).abs().max()
        worst_s = (g["pct_from_reliability"] - b["pct_from_reliability"]).abs().max()
        log(f"    -> 최대 변동: ECE 배율 {worst_e:.2f}, " f"보정 기여율 {worst_s:.1f}pp")
        if g["ece_ratio"].min() > 1.0 and g["pct_from_reliability"].min() > 50:
            log("    -> 세 시나리오 모두에서 (ECE 배율 > 1, 보정 기여율 > 50%) 유지")
        else:
            log("    -> 일부 시나리오에서 결론이 유지되지 않음. 본문에 명시할 것.")

    log()
    log("주의: 이 표는 연도 배정 오류에 대한 민감도만 본다. 논문 본문 수치는")
    log("      S0(기준) 시나리오의 값이며, S1/S2 는 그 값이 연도 오류에")
    log("      좌우되지 않음을 보이는 용도다.")

    (RES / "year_sensitivity_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'year_sensitivity.csv'}")
    log(f"      {RES/'year_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
