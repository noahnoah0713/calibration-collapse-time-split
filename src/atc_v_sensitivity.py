"""
진단·조영·방사성의약품(ATC V) 제외 민감도 분석

왜 주 분석이 아니라 민감도 분석인가
-----------------------------------
ATC V 계열(조영제, 방사성진단제, 해독제 등)은 독성이 유기 골격이 아니라
금속·방사성핵종·삼투압에서 온다. ECFP4 지문으로 심독성·간독성을
예측한다는 전제가 이 계열에는 성립하지 않는다. 그래서 처음부터
제외했다면 완전히 타당한 배제 기준이었을 것이다.

문제는 우리가 결과를 본 뒤에 이 문제를 발견했다는 점이다. 지금 와서
주 분석의 데이터셋 정의를 바꾸면, 숫자를 보고 대상을 고른 것처럼
보인다. 그래서 주 분석은 건드리지 않고, "이 계열을 빼도 결론이
유지되는가" 만 따로 보고한다. 연도 민감도(year_sensitivity.py)와
같은 취급이다.

본문에 쓸 때
------------
  - 주 분석 수치는 전체 집합(V 포함) 값이다.
  - 이 표는 Results 의 강건성 소절 또는 보충자료에 넣고,
    Limitations 에 "진단·조영제가 포함되어 있다" 를 명시한다.

사용법
------
    python src/atc_v_sensitivity.py

의존
----
    results/atc_audit.csv   (src/check_atc.py 가 만든다)
    src/brier_decomposition.py

출력
----
    results/atc_v_sensitivity.csv
    results/atc_v_sensitivity_summary.txt
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
from year_sensitivity import ece, load, random_baseline, temporal_fit  # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

DATASETS = [
    ("DILIrank", PROC / "dilirank2_with_year.csv"),
    ("DICTrank", PROC / "dictrank_processed.csv"),
]

log_lines = []


def log(m=""):
    print(m, flush=True)
    log_lines.append(str(m))


def evaluate(df, base):
    y_t, p_t, n_tr, n_te = temporal_fit(df)
    d = murphy_decomposition(bin_table(y_t, p_t))
    pct_rel, pct_res, _, _ = shares(base[0], base[1], d["reliability"], d["resolution"])
    e = ece(y_t, p_t)
    return {
        "n": len(df),
        "n_train": n_tr,
        "n_test": n_te,
        "pos_rate_test": float(y_t.mean()),
        "AUROC": float(roc_auc_score(y_t, p_t)),
        "Brier": float(brier_score_loss(y_t, p_t)),
        "ECE": float(e),
        "ECE_random_mean": base[2],
        "ece_ratio": float(e / base[2]) if base[2] else np.nan,
        "reliability": d["reliability"],
        "resolution": d["resolution"],
        "pct_from_reliability": pct_rel,
        "pct_from_resolution": pct_res,
    }


def main():
    apath = RES / "atc_audit.csv"
    if not apath.exists():
        raise SystemExit(
            f"[중단] {apath} 가 없습니다. " f"먼저 python src/check_atc.py 를 실행하세요."
        )
    audit = pd.read_csv(apath)

    log("=" * 74)
    log("ATC V(진단·조영·방사성의약품) 제외 민감도 분석")
    log("=" * 74)
    log(f"절단 {CUTOFF}년 | 무작위 기준 {N_REPEATS}회 평균 | seed {SEED}")
    log()

    rows = []
    for name, path in DATASETS:
        if not path.exists():
            log(f"[건너뜀] {name}: {path} 없음")
            continue

        df0 = load(path)
        v = set(audit[(audit.dataset == name) & (audit.atc1 == "V")]["compound"].astype(str))
        m = df0["CompoundName"].astype(str).isin(v)
        df1 = df0[~m].reset_index(drop=True)

        log("-" * 74)
        log(f"{name}  n={len(df0)}  ->  V 제외 후 {len(df1)}  " f"(제외 {int(m.sum())}건)")
        log("-" * 74)
        if m.any():
            ex = df0[m]
            held = int((ex["approval_year"] >= CUTOFF).sum())
            log(f"  제외: 학습군 {len(ex)-held}건, 검증군 {held}건")
            log("  " + ", ".join(ex["CompoundName"].astype(str).head(25)))
        log()

        n0 = int((df0["approval_year"].values >= CUTOFF).sum())
        log(f"  무작위 기준 계산 중 (전체 집합, {N_REPEATS}회) ...")
        b0 = random_baseline(df0, n0)
        n1 = int((df1["approval_year"].values >= CUTOFF).sum())
        log(f"  무작위 기준 계산 중 (V 제외 집합, {N_REPEATS}회) ...")
        b1 = random_baseline(df1, n1)

        for label, d, b in [("전체 (주 분석)", df0, b0), ("ATC V 제외", df1, b1)]:
            r = evaluate(d, b)
            r.update(dataset=name, scenario=label)
            rows.append(r)
            log(
                f"\n  {label:14s} n={r['n']:4d} 검증군 {r['n_test']:3d} "
                f"양성 {r['pos_rate_test']:.1%}"
            )
            log(
                f"  {'':14s} AUROC {r['AUROC']:.4f} | ECE {r['ECE']:.4f} "
                f"(무작위 {r['ECE_random_mean']:.4f}, 배율 {r['ece_ratio']:.2f})"
            )
            log(
                f"  {'':14s} 보정 기여율 {r['pct_from_reliability']:.1f}% "
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
            "pct_from_reliability",
            "pct_from_resolution",
        ]
    ]
    out.to_csv(RES / "atc_v_sensitivity.csv", index=False)

    log("=" * 74)
    log("판정")
    log("=" * 74)
    ok = True
    for name in out.dataset.unique():
        g = out[out.dataset == name].set_index("scenario")
        b, s = g.loc["전체 (주 분석)"], g.loc["ATC V 제외"]
        log(f"\n  {name}")
        log(
            f"    ECE 배율      {b['ece_ratio']:.2f} -> {s['ece_ratio']:.2f} "
            f"({s['ece_ratio']-b['ece_ratio']:+.2f})"
        )
        log(
            f"    보정 기여율   {b['pct_from_reliability']:.1f}% -> "
            f"{s['pct_from_reliability']:.1f}% "
            f"({s['pct_from_reliability']-b['pct_from_reliability']:+.1f}pp)"
        )
        log(f"    AUROC         {b['AUROC']:.4f} -> {s['AUROC']:.4f}")
        if not (s["ece_ratio"] > 1 and s["pct_from_reliability"] > 50):
            ok = False
    log()
    if ok:
        log("  => 진단·조영제를 빼도 (ECE 배율 > 1, 보정 기여율 > 50%) 유지.")
        log("     보정 붕괴가 이 계열 때문에 생긴 착시가 아니다.")
    else:
        log("  => 일부 데이터셋에서 결론이 유지되지 않는다. 본문에 명시할 것.")
    log()
    log("주의: 본문 주 분석 수치는 '전체' 행이다. 이 표는 강건성 근거다.")

    (RES / "atc_v_sensitivity_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'atc_v_sensitivity.csv'}")
    log(f"      {RES/'atc_v_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
