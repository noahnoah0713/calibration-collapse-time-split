"""
Drugs@FDA -> DILIrank 승인연도 매핑

Products.txt (성분명) + Submissions.txt (승인일)을 ApplNo로 연결하여
각 성분의 '최초 승인연도'를 구하고, DILIrank 화합물명과 매칭한다.

매칭은 4단계로 시도:
  1) 정규화 완전일치
  2) 염/에스터 접미어 제거 후 일치
  3) 첫 단어(기본 성분명) 일치
  4) 부분 문자열 포함

사용법:
    python src/map_approval_year.py

출력:
    data/processed/dilirank2_with_year.csv
    data/processed/year_unmatched.csv
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "drugsfda"
PROC = ROOT / "data" / "processed"

IN_CSV = PROC / "dilirank2_clean.csv"  # baseline.py 산출물
FALLBACK = PROC / "dilirank2_with_smiles.csv"  # clean이 없으면 이것
OUT_CSV = PROC / "dilirank2_with_year.csv"
UNMATCHED = PROC / "year_unmatched.csv"

# 염·에스터·수화물 등 접미어
SALT_WORDS = {
    "hydrochloride",
    "hcl",
    "sulfate",
    "sulphate",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "maleate",
    "tartrate",
    "citrate",
    "acetate",
    "mesylate",
    "besylate",
    "tosylate",
    "fumarate",
    "succinate",
    "phosphate",
    "nitrate",
    "bromide",
    "chloride",
    "iodide",
    "oxalate",
    "malate",
    "lactate",
    "gluconate",
    "stearate",
    "palmitate",
    "carbonate",
    "bitartrate",
    "dihydrochloride",
    "hydrobromide",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "anhydrous",
    "hemihydrate",
    "pamoate",
    "napsylate",
    "embonate",
    "aspartate",
    "glutamate",
    "lysine",
    "meglumine",
    "tromethamine",
    "diolamine",
    "olamine",
    "benzoate",
    "salicylate",
    "propionate",
    "valerate",
    "dipropionate",
    "furoate",
    "xinafoate",
    "disodium",
    "dipotassium",
    "trisodium",
    "hyclate",
    "besilate",
    "camsylate",
    "edisylate",
    "isethionate",
    "lactobionate",
    "mucate",
    "pantothenate",
    "polistirex",
    "subsalicylate",
    "sulfoxide",
    "titrate",
    "arginine",
    "choline",
    "ethanolamine",
    "piperazine",
    "zinc",
    "aluminum",
    "micronized",
    "anhydrate",
    "hydrate",
    "salt",
    "base",
    "free",
    "monosodium",
    "monopotassium",
    "hemifumarate",
    "dimaleate",
    "hydrochlorides",
    "sulfates",
    "acid",
    "alfa",
    "beta",
    "gamma",
}


def norm(s) -> str:
    """소문자화 + 특수문자 제거 + 공백 정리."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_salt(s: str) -> str:
    """뒤쪽 염/수식어 토큰 제거."""
    toks = norm(s).split()
    while len(toks) > 1 and toks[-1] in SALT_WORDS:
        toks.pop()
    return " ".join(toks)


def first_word(s: str) -> str:
    t = norm(s).split()
    return t[0] if t else ""


# ------------------------------------------------------------------ FDA 적재
def load_fda() -> pd.DataFrame:
    prod = pd.read_csv(
        RAW / "Products.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = pd.read_csv(
        RAW / "Submissions.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )

    print(f"Products.txt   : {len(prod):,}행")
    print(f"Submissions.txt: {len(subm):,}행")

    # 승인(AP) 상태 + 날짜 있는 것만
    subm = subm[subm["SubmissionStatus"] == "AP"].copy()
    subm["date"] = pd.to_datetime(subm["SubmissionStatusDate"], errors="coerce")
    subm = subm.dropna(subset=["date"])

    # 신청번호별 최초 승인일
    first_ap = (
        subm.groupby("ApplNo")["date"].min().reset_index().rename(columns={"date": "first_ap"})
    )

    prod = prod.merge(first_ap, on="ApplNo", how="inner")
    prod = prod.dropna(subset=["ActiveIngredient"])

    # 복합제(;로 구분)는 각 성분으로 분리
    prod["ActiveIngredient"] = prod["ActiveIngredient"].str.split(";")
    prod = prod.explode("ActiveIngredient")
    prod["ing"] = prod["ActiveIngredient"].str.strip()
    prod = prod[prod["ing"].str.len() > 0]

    # 성분별 최초 승인일
    ing = prod.groupby("ing")["first_ap"].min().reset_index()
    ing["year"] = ing["first_ap"].dt.year
    ing["k_full"] = ing["ing"].apply(norm)
    ing["k_salt"] = ing["ing"].apply(strip_salt)
    ing["k_first"] = ing["ing"].apply(first_word)

    print(f"고유 성분: {len(ing):,}개")
    print(f"승인연도 범위: {int(ing.year.min())} ~ {int(ing.year.max())}\n")
    return ing


def build_maps(ing: pd.DataFrame):
    """각 매칭 단계용 딕셔너리 (동일 키는 가장 이른 연도)."""

    def mk(col):
        return ing.sort_values("year").drop_duplicates(col).set_index(col)["year"].to_dict()

    return mk("k_full"), mk("k_salt"), mk("k_first")


def main() -> None:
    src = IN_CSV if IN_CSV.exists() else FALLBACK
    df = pd.read_csv(src)
    print(f"입력: {src.name} ({len(df)}행)\n")

    ing = load_fda()
    m_full, m_salt, m_first = build_maps(ing)
    all_keys = list(m_full.keys())

    years, methods = [], []
    for name in df["CompoundName"].astype(str):
        n1, n2, n3 = norm(name), strip_salt(name), first_word(name)

        if n1 in m_full:
            years.append(m_full[n1])
            methods.append("exact")
        elif n2 in m_salt:
            years.append(m_salt[n2])
            methods.append("salt_stripped")
        elif n2 in m_full:
            years.append(m_full[n2])
            methods.append("salt_to_full")
        elif n3 and n3 in m_first:
            years.append(m_first[n3])
            methods.append("first_word")
        else:
            # 부분 포함 (짧은 이름은 오매칭 위험 -> 5자 이상만)
            hit = None
            if len(n2) >= 5:
                for k in all_keys:
                    if n2 in k or k in n2:
                        hit = m_full[k]
                        break
            if hit:
                years.append(hit)
                methods.append("substring")
            else:
                years.append(None)
                methods.append("unmatched")

    df["approval_year"] = years
    df["year_match"] = methods

    df.to_csv(OUT_CSV, index=False)

    # ------------------------------------------------------------ 결과
    ok = df["approval_year"].notna().sum()
    print("=" * 58)
    print(f"승인연도 매칭: {ok}/{len(df)}  ({ok/len(df):.1%})")
    print("\n매칭 방법별 건수:")
    print(df["year_match"].value_counts().to_string())

    yr = df["approval_year"].dropna().astype(int)
    if len(yr):
        print(f"\n승인연도 분포: {yr.min()} ~ {yr.max()}")
        print(f"  ~2009년 승인: {(yr <= 2009).sum()}개")
        print(f"  2010년 이후 : {(yr >= 2010).sum()}개")

        if "y" in df.columns:
            d = df.dropna(subset=["approval_year", "y"]).copy()
            d["era"] = (d.approval_year >= 2010).map({True: "2010+", False: "~2009"})
            print("\n시대별 클래스 분포:")
            print(
                pd.crosstab(d["era"], d["y"].astype(int), rownames=["승인시기"], colnames=["DILI"])
            )

        # Comment(New/Unchanged)와 실제 승인연도 비교
        if "Comment" in df.columns:
            d = df.dropna(subset=["approval_year"])
            print("\n[검증] Comment vs 실제 승인연도 (2010년 기준)")
            print(
                pd.crosstab(
                    d["Comment"], (d.approval_year >= 2010).map({True: "2010+", False: "~2009"})
                )
            )

    miss = df[df["approval_year"].isna()]
    if len(miss):
        cols = [c for c in ["LTKBID", "CompoundName", "label", "Comment"] if c in df.columns]
        miss[cols].to_csv(UNMATCHED, index=False)
        print(f"\n미매칭 {len(miss)}건 -> {UNMATCHED.name}")
        print("예시:")
        for n in miss["CompoundName"].head(15):
            print("   -", n)

    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
