# -*- coding: utf-8 -*-
"""승인연도 후보 7건의 Drugs@FDA 근거를 뽑는다 (results/year_evidence.csv).

check_years.py 가 results/year_corrections.csv 에 "더 이른 승인 기록이 있다"
고 표시한 화합물 7건이 있다. 그 7건을 육안으로 판정하려면 근거(어느 신청번호,
어느 제품, 언제)가 필요하므로, 이 스크립트가 프로젝트 안의 Drugs@FDA 원자료
(data/raw/drugsfda/{Products,Submissions,Applications}.txt) 만 써서 근거표를
만든다. 외부 조회는 하지 않는다.

성분 매칭은 check_years.py 의 함수(norm, strip_salt, SALT_WORDS)를 그대로
불러 쓴다. 제안 연도를 만든 규칙과 근거표를 만든 규칙이 달라지면 근거가
근거가 아니게 되므로, 정의를 두 번 쓰지 않고 한 곳에서 가져온다. 규칙은
"정규화한 성분명이 화합물명과 같거나, 염 표기를 떼면 같다" 이다. 그래서
AMPHETAMINE 은 AMPHETAMINE SULFATE 에는 걸리지만 AMPHETAMINE ASPARTATE,
AMPHETAMINE RESIN COMPLEX, METHAMPHETAMINE HYDROCHLORIDE 에는 걸리지 않는다.

승인일은 신청번호별로 Submissions.txt 의 승인(AP) 기록 중 가장 이른 날짜다
(check_years.py 와 동일). 표에는 이른 순으로 8건까지 싣고, 매칭된 신청번호
전체 수는 따로 적는다. 한 신청번호에 제품이 여럿이면 제품번호가 가장 작은
것의 제품명을 쓴다.

    python src/ev_years.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_years as CY

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "drugsfda"
RES = ROOT / "results"
N_SHOW = 8

COLS = [
    "데이터셋",
    "화합물",
    "현재배정",
    "스크립트제안",
    "근거_ApplNo",
    "종류",
    "제품명",
    "성분표기",
    "최초승인일",
    "연도",
    "신청자",
    "전체매칭수",
    "NDA최소연도",
    "전체최소연도",
]


def load_products():
    prod = pd.read_csv(
        RAW / "Products.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = pd.read_csv(
        RAW / "Submissions.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    appl = pd.read_csv(
        RAW / "Applications.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = subm[subm["SubmissionStatus"] == "AP"].copy()
    subm["date"] = pd.to_datetime(subm["SubmissionStatusDate"], errors="coerce")
    subm = subm.dropna(subset=["date"])
    first = subm.groupby("ApplNo")["date"].min().reset_index().rename(columns={"date": "first_ap"})
    prod = prod.merge(first, on="ApplNo").dropna(subset=["ActiveIngredient"])
    prod = prod.merge(appl[["ApplNo", "ApplType", "SponsorName"]], on="ApplNo", how="left")
    prod["ProductNo_i"] = pd.to_numeric(prod["ProductNo"], errors="coerce")
    prod["ing_list"] = prod["ActiveIngredient"].str.split(";")
    ex = prod.explode("ing_list").copy()
    ex["ing"] = ex["ing_list"].str.strip()
    ex["k_full"] = ex["ing"].apply(CY.norm)
    ex["k_salt"] = ex["ing"].apply(CY.strip_salt)
    return ex


def main():
    corr = RES / "year_corrections.csv"
    if not corr.exists():
        sys.exit("먼저 check_years.py 를 돌려 %s 를 만들어야 한다." % corr)
    cands = pd.read_csv(corr)
    ex = load_products()

    out = []
    for _, c in cands.iterrows():
        n1, n2 = CY.norm(c["compound"]), CY.strip_salt(c["compound"])
        m = ex[(ex.k_full == n1) | (ex.k_salt == n2) | (ex.k_full == n2)]
        if not len(m):
            print("%-10s %-28s 매칭 0건" % (c["dataset"], c["compound"]))
            continue
        # 신청번호 단위로 접는다: 제품번호가 가장 작은 행을 대표로.
        m = m.sort_values(["first_ap", "ApplNo", "ProductNo_i"])
        one = m.drop_duplicates("ApplNo", keep="first")
        n_all = len(one)
        y_all = int(one.first_ap.dt.year.min())
        nda = one[one.ApplType == "NDA"]
        y_nda = int(nda.first_ap.dt.year.min()) if len(nda) else ""
        for _, r in one.head(N_SHOW).iterrows():
            out.append(
                {
                    "데이터셋": c["dataset"],
                    "화합물": c["compound"],
                    "현재배정": c["assigned_year"],
                    "스크립트제안": c["suggested_year"],
                    "근거_ApplNo": r["ApplNo"],
                    "종류": r["ApplType"] if isinstance(r["ApplType"], str) else "",
                    "제품명": r["DrugName"],
                    "성분표기": r["ing"],
                    "최초승인일": r["first_ap"].strftime("%Y-%m-%d"),
                    "연도": int(r["first_ap"].year),
                    "신청자": r["SponsorName"] if isinstance(r["SponsorName"], str) else "",
                    "전체매칭수": n_all,
                    "NDA최소연도": y_nda,
                    "전체최소연도": y_all,
                }
            )
        print(
            "%-10s %-28s 배정 %s / 제안 %s | 매칭 %d건, 최소 NDA %s, 최소 전체 %s"
            % (
                c["dataset"],
                c["compound"],
                c["assigned_year"],
                c["suggested_year"],
                n_all,
                y_nda,
                y_all,
            )
        )

    df = pd.DataFrame(out, columns=COLS)
    p = RES / "year_evidence.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    print()
    print("%s  (%d행)" % (p, len(df)))


if __name__ == "__main__":
    main()
