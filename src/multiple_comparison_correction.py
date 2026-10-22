"""
다중비교 보정 (Benjamini-Hochberg FDR)

본 연구의 모든 통계 검정을 세 그룹(family)으로 나누어 각 그룹 안에서만
FDR 보정을 적용한다. 성격이 다른 검정을 한 그룹으로 묶으면 사전에 계획한
핵심 가설이 사후 탐색의 개수에 휘둘리기 때문이다.

  1군 (주 확증 검정)   : 사전 계획된 핵심 가설 — 시간분할 성능 저하 자체.
  2군 (유병률 통제)     : 하나의 현상(보정 붕괴가 유병률 차이 때문이 아님)을
                         여러 각도에서 확인하는 대조군.
  3군 (탐색적 하위분석) : 사후에 뒤진 하위집단 — ATC 계열, 구조 층화,
                         화학공간 유사도. 다중비교 문제가 실제로 생기는
                         지점이며 FDR 보정이 반드시 필요하다.

이 스크립트가 바뀐 이유
-----------------------
예전 판본은 p-value 를 소스코드에 손으로 적어 두었다. 분석을 다시 돌릴
때마다 값이 바뀌는데 표를 고치는 것을 잊으면 논문에 옛 숫자가 그대로
들어간다. 실제로 구조 필터를 적용한 뒤 여러 p 값이 크게 움직였다
(예: N>=6 0.0530 -> 0.0210). 그래서 이제 모든 p 를 결과 파일에서
읽어 오고, 읽을 파일이 없거나 데이터보다 오래되었으면 멈춘다.

읽는 파일
---------
  1군 : results/permutation_test.csv      (src/permutation_test.py)
  2군 : results/prevalence_control.csv    (src/prevalence_control.py)
  3군 : results/atc_tests.csv             (src/atc_kegg.py)
        results/similarity_tests.csv      (src/stats_analysis.py)

사용법
------
    python src/multiple_comparison_correction.py
    python src/multiple_comparison_correction.py --allow-stale   # 경고만

출력
----
    results/multiple_comparison_correction.csv
    results/multiple_comparison_correction_summary.txt
"""

import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

ALPHA = 0.05
ALLOW_STALE = "--allow-stale" in sys.argv

SOURCES = {
    "permutation_test.csv": "src/permutation_test.py",
    "prevalence_control.csv": "src/prevalence_control.py",
    "atc_tests.csv": "src/atc_kegg.py",
    "similarity_tests.csv": "src/stats_analysis.py",
}

DATA_FILES = ["dilirank2_with_year.csv", "dictrank_processed.csv"]

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def require(fname):
    """결과 파일을 읽는다. 없거나 데이터보다 오래되었으면 멈춘다."""
    path = RES / fname
    if not path.exists():
        raise SystemExit(
            f"[중단] {path} 가 없습니다.\n" f"        먼저 python {SOURCES[fname]} 를 실행하세요."
        )

    data_mtimes = [(PROC / d).stat().st_mtime for d in DATA_FILES if (PROC / d).exists()]
    if data_mtimes and path.stat().st_mtime < max(data_mtimes):
        msg = (
            f"[오래됨] {fname} 이 처리된 데이터보다 먼저 만들어졌습니다.\n"
            f"          python {SOURCES[fname]} 를 다시 실행하세요."
        )
        if ALLOW_STALE:
            log(f"  !! {msg.splitlines()[0]}")
        else:
            raise SystemExit(msg + "\n        (의도한 것이라면 --allow-stale 을 붙이세요.)")
    return pd.read_csv(path)


# ------------------------------------------------------------ 1군
def build_family1():
    pt = require("permutation_test.csv")
    need = {"dataset", "p_permutation", "n_perm_valid"}
    if not need <= set(pt.columns):
        raise SystemExit(
            f"[중단] permutation_test.csv 에 {sorted(need - set(pt.columns))} "
            f"열이 없습니다. src/permutation_test.py 를 다시 실행하세요."
        )
    rows = []
    for r in pt.itertuples():
        rows.append(
            {
                "dataset": r.dataset,
                "test": f"시간분할 성능 저하 (순열 {int(r.n_perm_valid)}회)",
                "p": float(r.p_permutation),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 2군
def build_family2():
    pv = require("prevalence_control.csv")
    pv = pv[pv["p_perm"].notna()].copy()

    if "n_ge_observed" in pv.columns and "n_rep_valid" in pv.columns:
        # 편향보정 p 를 여기서 다시 계산해 둔다 (원본이 어느 판본이든 동일)
        pv["p_use"] = (1 + pv["n_ge_observed"]) / (1 + pv["n_rep_valid"])
        note = "Phipson-Smyth (1+r)/(1+n)"
    else:
        pv["p_use"] = pv["p_perm"]
        note = (
            "단순 비율 — prevalence_control.py 가 옛 판본입니다. "
            "다시 실행하면 편향보정 p 를 씁니다."
        )
        log(f"  !! 2군 p 추정량: {note}")

    rows = []
    for r in pv.itertuples():
        ds = "DILIrank" if str(r.dataset).startswith("DILIrank") else "DICTrank"
        rows.append(
            {
                "dataset": ds,
                "test": f"{str(r.setting).strip()} (ECE 배율 {r.ece_ratio:.2f})",
                "p": float(r.p_use),
            }
        )
    return pd.DataFrame(rows), note


# ------------------------------------------------------------ 3군
def build_family3():
    atc = require("atc_tests.csv")
    sim = require("similarity_tests.csv")
    rows = []
    for d in (atc, sim):
        for r in d.itertuples():
            rows.append({"dataset": r.dataset, "test": r.test, "p": float(r.p)})
    return pd.DataFrame(rows)


def apply_fdr(df, alpha=ALPHA):
    if not len(df):
        return df.assign(q_BH=[], significant_after_FDR=[])
    rej, q, _, _ = multipletests(df["p"].values, alpha=alpha, method="fdr_bh")
    df = df.copy()
    df["q_BH"] = q
    df["significant_after_FDR"] = rej
    return df


def show(title, df):
    log("=" * 78)
    log(title)
    log("=" * 78)
    d = df.copy()
    d["p"] = d["p"].map(lambda v: f"{v:.4f}")
    d["q_BH"] = d["q_BH"].map(lambda v: f"{v:.4f}")
    log(d.to_string(index=False))
    log()


def main():
    log("다중비교 보정 (Benjamini-Hochberg, alpha = %.2f)" % ALPHA)
    log("모든 p 는 결과 파일에서 읽는다. 소스코드에 적힌 값은 없다.")
    log()

    f1 = apply_fdr(build_family1())
    f2_raw, note2 = build_family2()
    f2 = apply_fdr(f2_raw)
    f3 = apply_fdr(build_family3())

    show(f"1군: 주 확증 검정 (n={len(f1)})", f1)
    show(f"2군: 유병률 통제 대조군 (n={len(f2)})  p 추정량: {note2}", f2)
    show(f"3군: 탐색적 하위집단 분석 (n={len(f3)})", f3)

    log("=" * 78)
    log("판정")
    log("=" * 78)
    log(f"  1군: {f1['significant_after_FDR'].sum()}/{len(f1)} 유의 유지")
    log(f"  2군: {f2['significant_after_FDR'].sum()}/{len(f2)} 유의 유지")
    log(f"  3군: {f3['significant_after_FDR'].sum()}/{len(f3)} 유의 유지")
    log()

    if f1["significant_after_FDR"].all():
        log("  -> 핵심 결과(시간분할 성능 저하)는 보정 후에도 두 데이터셋 모두 유의함.")
    else:
        bad = f1[~f1.significant_after_FDR]["dataset"].tolist()
        log(f"  -> 주의: 1군에서 {bad} 가 보정 후 유의하지 않음. 본문에 명시할 것.")

    if f2["significant_after_FDR"].all():
        log("  -> 보정 붕괴가 유병률 차이 때문이 아니라는 결론은 보정 후에도 유지됨.")
    else:
        bad = f2[~f2.significant_after_FDR]["test"].tolist()
        log(f"  -> 주의: 2군에서 {bad} 가 보정 후 유의하지 않음.")

    n3 = int(f3["significant_after_FDR"].sum())
    if n3 == 0:
        log("  -> 탐색적 하위집단 분석은 보정 후 유의한 것이 하나도 없다.")
        log("     항종양제(L) 성능 붕괴를 포함해 어떤 하위집단 효과도 핵심")
        log("     주장으로 쓸 수 없다. 본문에서 '가설 생성용'으로만 다룰 것.")
    else:
        keep = f3[f3.significant_after_FDR][["dataset", "test", "q_BH"]]
        log(f"  -> 보정 후에도 남은 하위집단 효과 {n3}건:")
        log(keep.to_string(index=False))
        log("     그래도 사후 탐색이므로 확증이 아니라 가설 생성으로 서술할 것.")

    out = pd.concat(
        [
            f1.assign(family="1_primary"),
            f2.assign(family="2_prevalence_control"),
            f3.assign(family="3_exploratory_subgroup"),
        ],
        ignore_index=True,
    )
    out.to_csv(RES / "multiple_comparison_correction.csv", index=False)
    (RES / "multiple_comparison_correction_summary.txt").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )
    log(f"\n저장: {RES/'multiple_comparison_correction.csv'}")
    log(f"      {RES/'multiple_comparison_correction_summary.txt'}")


if __name__ == "__main__":
    main()
