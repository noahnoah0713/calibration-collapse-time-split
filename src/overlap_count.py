"""
두 데이터셋의 화합물 중첩 집계 (2026-09-28)

왜 별도 스크립트인가
--------------------
§18 '방법론적 선택' 의 중첩 수치(공통 분자 개수와 두 비율, 전용 개수)는
문서 작성 중 손으로 계산한 값이었고, 2026-09-28 재검산에서 재현되지
않았다(부록 B-8). 이 숫자는 "두 데이터셋에서 재현" 이라는 주장의 강도를
직접 정하므로, 기계가 매번 다시 세도록 스크립트로 고정한다.

rdkit·xgboost 를 쓰지 않는다 — ikey14 열은 이미 전처리 산출물에 있고,
이 집계는 pandas 만으로 끝난다. 그래서 어느 환경에서도 재확인된다.

키 정의
-------
ikey14 = InChIKey 앞 14자리(골격 블록). 상류 파이프라인이 이 키로
중복 제거를 이미 수행했으므로 각 데이터셋 안에서는 유일하다(아래에서
확인해 인쇄한다). 27자리 전체 키로 세면 입체이성질체가 분리되어
공통 분자가 줄어드는데, 그 값도 함께 인쇄한다.

실행
----
    python src/overlap_count.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CUTOFF = 2010
DATASETS = [("DILIrank", "dilirank2_with_year.csv"), ("DICTrank", "dictrank_processed.csv")]

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def load(path):
    df = pd.read_csv(path).dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def main():
    frames, keys = {}, {}
    log("두 데이터셋의 화합물 중첩 집계")
    log(f"키: ikey14 (InChIKey 앞 14자리), 절단 {CUTOFF}년")
    log("=" * 88)
    for name, f in DATASETS:
        df = load(PROC / f)
        frames[name] = df
        keys[name] = set(df["ikey14"].dropna())
        dup = len(df) - df["ikey14"].nunique()
        log(
            f"  {name:10s} 분석집합 {len(df)}행   고유 ikey14 "
            f"{df['ikey14'].nunique()}개   결측 "
            f"{int(df['ikey14'].isna().sum())}   집합 내 중복 {dup}"
        )

    a, b = [n for n, _ in DATASETS]
    shared = keys[a] & keys[b]
    log("")
    log(f"  공통 분자 {len(shared)}개")
    log(
        f"    {a} 의 {len(shared)/len(frames[a])*100:.1f}%"
        f"  /  {b} 의 {len(shared)/len(frames[b])*100:.1f}%"
    )

    # 27자리 전체 키로도 세어 본다 (두 파일에 모두 inchikey 열이 있을 때만)
    if all("inchikey" in frames[n].columns for n in (a, b)):
        k27 = {n: set(frames[n]["inchikey"].dropna()) for n in (a, b)}
        s27 = k27[a] & k27[b]
        log(
            f"  [참고] 27자리 전체 InChIKey 기준 공통 {len(s27)}개"
            f"  ({len(s27)/len(frames[a])*100:.1f}% / "
            f"{len(s27)/len(frames[b])*100:.1f}%)"
        )

    rows = []
    log("")
    log("전용(상대 데이터셋에 없는) 분자의 시간분할 분포")
    log("-" * 88)
    log(
        f"{'데이터셋':10s} {'전용':>6s} {'학습군':>7s} {'양성률':>7s} "
        f"{'검증군':>7s} {'양성':>5s} {'음성':>5s} {'양성률':>7s}"
    )
    for name in (a, b):
        df = frames[name]
        other = keys[b] if name == a else keys[a]
        ex = df[~df["ikey14"].isin(other)]
        tr = ex[ex.approval_year < CUTOFF]
        te = ex[ex.approval_year >= CUTOFF]
        log(
            f"{name:10s} {len(ex):6d} {len(tr):7d} "
            f"{tr.y.mean()*100:6.1f}% {len(te):7d} {int(te.y.sum()):5d} "
            f"{len(te)-int(te.y.sum()):5d} {te.y.mean()*100:6.1f}%"
        )
        rows.append(
            dict(
                dataset=name,
                n_analysis=len(df),
                n_unique_key=int(df["ikey14"].nunique()),
                n_shared=len(shared),
                shared_pct=len(shared) / len(df) * 100,
                n_exclusive=len(ex),
                n_exclusive_train=len(tr),
                n_exclusive_test=len(te),
                n_exclusive_test_pos=int(te.y.sum()),
                n_exclusive_test_neg=len(te) - int(te.y.sum()),
                pos_rate_exclusive_train=float(tr.y.mean()),
                pos_rate_exclusive_test=float(te.y.mean()),
                n_shared_train=int((df["ikey14"].isin(shared) & (df.approval_year < CUTOFF)).sum()),
                n_shared_test=int((df["ikey14"].isin(shared) & (df.approval_year >= CUTOFF)).sum()),
            )
        )

    pd.DataFrame(rows).to_csv(RES / "compound_overlap.csv", index=False)
    log("")
    log("[읽는 법] 공통 분자 비율이 높으면 두 데이터셋의 추정치는 상관돼")
    log("          있다. 라벨(간독성/심장독성)만 다르고 구조·승인연도·")
    log("          시간분할 소속이 겹치기 때문이다. 전용 검증군의 n 이")
    log("          곧 '독립 재현' 을 확인할 수 있는 표본 크기다.")
    log("")
    log(f"저장: {RES/'compound_overlap.csv'}")
    (RES / "compound_overlap_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
