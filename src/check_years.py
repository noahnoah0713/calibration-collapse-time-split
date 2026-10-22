"""
승인연도 배정 검수

왜 필요한가:
    Drugs@FDA의 Submissions.txt에는 1982년 이전 승인 기록이 거의 없다.
    그래서 오래된 약은 '가장 이른 승인'이 훗날의 보충신청이나 신규 제형으로
    잡혀, 실제보다 훨씬 최근 연도가 배정될 수 있다. 그런 화합물이 검증군
    (2010년 이후)에 잘못 들어가면 시간분할 성능이 실제보다 좋게 나온다.

무엇을 확인하는가:
    1) 라벨 공개연도 이후로 배정된 화합물  (논리적으로 불가능)
    2) 배정된 연도와, 더 느슨한 이름 키로 찾은 최소 연도의 차이
       (차이가 크면 같은 성분의 더 이른 승인을 놓쳤다는 뜻)
    3) 두 데이터셋에 공통으로 있는 화합물의 연도 불일치
    4) 검증군(2010년 이후) 전체 목록 - 육안 검수용

사용법:
    python src/check_years.py

출력:
    results/year_audit.csv          화합물별 상세
    results/year_audit_summary.txt  요약
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "drugsfda"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CUTOFF = 2010
LABEL_FREEZE = {"DILIrank": 2025, "DICTrank": 2023}
BIG_GAP = 5  # 배정연도와 최소연도가 이만큼 벌어지면 보고

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
    "disodium",
    "dipotassium",
    "hyclate",
    "dimaleate",
    "hemifumarate",
    "camsylate",
    "besilate",
    "salt",
    "hydrate",
    "acid",
    "alfa",
    "beta",
}

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def norm(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_salt(s):
    t = norm(s).split()
    while len(t) > 1 and t[-1] in SALT_WORDS:
        t.pop()
    return " ".join(t)


def first_word(s):
    t = norm(s).split()
    return t[0] if t else ""


def load_fda():
    prod = pd.read_csv(
        RAW / "Products.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = pd.read_csv(
        RAW / "Submissions.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = subm[subm["SubmissionStatus"] == "AP"].copy()
    subm["date"] = pd.to_datetime(subm["SubmissionStatusDate"], errors="coerce")
    subm = subm.dropna(subset=["date"])
    first = subm.groupby("ApplNo")["date"].min().reset_index().rename(columns={"date": "first_ap"})
    prod = prod.merge(first, on="ApplNo").dropna(subset=["ActiveIngredient"])
    prod["ActiveIngredient"] = prod["ActiveIngredient"].str.split(";")
    prod = prod.explode("ActiveIngredient")
    prod["ing"] = prod["ActiveIngredient"].str.strip()
    ing = prod.groupby("ing")["first_ap"].min().reset_index()
    ing["year"] = ing["first_ap"].dt.year
    ing["k_full"] = ing["ing"].apply(norm)
    ing["k_salt"] = ing["ing"].apply(strip_salt)
    ing["k_first"] = ing["ing"].apply(first_word)
    return ing


def min_year_tight(ing, name):
    """엄격한 키(정규화 전체 / 염 제거)로만 찾은 최소 연도.

    첫 단어 키는 쓰지 않는다. 'Sodium zirconium cyclosilicate'가 'sodium'에
    걸려 1946년으로 잡히는 식의 거짓 양성이 대량 발생하기 때문이다.
    """
    n1, n2 = norm(name), strip_salt(name)
    m = (ing.k_full == n1) | (ing.k_salt == n2) | (ing.k_full == n2)
    s = ing.loc[m, "year"]
    return int(s.min()) if len(s) else None


def main():
    log("승인연도 배정 검수")
    log("=" * 70)
    ing = load_fda()
    log(f"Drugs@FDA 고유 성분 {len(ing):,}개, 연도 {int(ing.year.min())}-{int(ing.year.max())}")
    log(
        f"1982년 이전 승인으로 기록된 성분: {(ing.year < 1982).sum():,}개 "
        f"({(ing.year < 1982).mean():.1%})"
    )
    log("  -> 이 비율이 낮다는 것 자체가 옛 승인 기록의 결손을 보여준다.\n")

    rows = []
    for ds, path, col in [
        ("DILIrank", PROC / "dilirank2_with_year.csv", "CompoundName"),
        ("DICTrank", PROC / "dictrank_processed.csv", "CompoundName"),
    ]:
        if not path.exists():
            log(f"{ds}: {path.name} 없음 - 건너뜀")
            continue
        df = pd.read_csv(path).dropna(subset=["approval_year"]).copy()
        df["approval_year"] = df["approval_year"].astype(int)
        freeze = LABEL_FREEZE[ds]
        log("=" * 70)
        log(f"{ds}   (n={len(df)}, 라벨 공개 {freeze}년)")
        log("=" * 70)

        for _, r in df.iterrows():
            nm = str(r[col])
            mn = min_year_tight(ing, nm)
            rows.append(
                {
                    "dataset": ds,
                    "compound": nm,
                    "assigned_year": r["approval_year"],
                    "min_year_tight": mn,
                    "gap": (r["approval_year"] - mn) if mn is not None else None,
                    "after_label_freeze": r["approval_year"] > freeze,
                    "in_heldout": r["approval_year"] >= CUTOFF,
                }
            )

        sub = pd.DataFrame([x for x in rows if x["dataset"] == ds])
        bad = sub[sub.after_label_freeze]
        log(f"\n[1] 라벨 공개연도 이후로 배정: {len(bad)}건")
        if len(bad):
            log(bad[["compound", "assigned_year"]].to_string(index=False))

        gapped = sub[(sub.gap.notna()) & (sub.gap >= BIG_GAP)]
        log(f"\n[2] 느슨한 키로 찾은 최소 연도보다 {BIG_GAP}년 이상 늦게 배정: {len(gapped)}건")
        if len(gapped):
            g = gapped.sort_values("gap", ascending=False)
            log(
                g[["compound", "assigned_year", "min_year_tight", "gap", "in_heldout"]]
                .head(30)
                .to_string(index=False)
            )
            cross = g[(g.assigned_year >= CUTOFF) & (g.min_year_tight < CUTOFF)]
            log(f"\n    이 중 절단선을 넘는 것(배정>=2010, 최소<2010): {len(cross)}건")
            if len(cross):
                log(cross[["compound", "assigned_year", "min_year_tight"]].to_string(index=False))

        ho = sub[sub.in_heldout].sort_values("assigned_year", ascending=False)
        log(f"\n[3] 검증군({CUTOFF}년 이후) {len(ho)}건 전체 - 육안 검수용")
        log("    (오래된 약이 여기 있으면 매칭 오류다. 예: phenobarbital, vancomycin,")
        log("     amantadine 같은 고전 약물)")
        log(ho[["compound", "assigned_year"]].to_string(index=False))
        log("")

    audit = pd.DataFrame(rows)

    # 두 데이터셋 공통 화합물 연도 일치 여부
    a = audit[audit.dataset == "DILIrank"].copy()
    b = audit[audit.dataset == "DICTrank"].copy()
    a["k"] = a.compound.map(norm)
    b["k"] = b.compound.map(norm)
    m = (
        a[["k", "assigned_year"]]
        .rename(columns={"assigned_year": "yr_dili"})
        .merge(b[["k", "assigned_year"]].rename(columns={"assigned_year": "yr_dict"}), on="k")
        .drop_duplicates("k")
    )
    log("=" * 70)
    log(f"[4] 두 데이터셋 공통 화합물 {len(m)}개의 연도 일치 여부")
    dis = m[m.yr_dili != m.yr_dict]
    log(f"    불일치: {len(dis)}건")
    if len(dis):
        log(dis.to_string(index=False))

    # 절단선을 넘는 수정 후보만 따로 저장 (민감도 분석용)
    fix = audit[
        (audit.min_year_tight.notna())
        & (audit.assigned_year >= CUTOFF)
        & (audit.min_year_tight < CUTOFF)
    ].copy()
    fix = fix[["dataset", "compound", "assigned_year", "min_year_tight"]]
    fix = fix.rename(columns={"min_year_tight": "suggested_year"})
    fix["suggested_year"] = fix["suggested_year"].astype(int)
    fix.to_csv(RES / "year_corrections.csv", index=False)
    log("=" * 70)
    log(f"[5] 절단선을 넘는 수정 후보 {len(fix)}건 -> results/year_corrections.csv")
    log("    이 화합물들은 배정연도상 검증군이지만 실제로는 학습군에 속한다.")
    log("    민감도 분석: 이 연도를 적용해 시간분할을 다시 돌리고 결론이")
    log("    유지되는지 확인할 것. 전부 육안 확인 후 적용하라.")
    if len(fix):
        log(fix.to_string(index=False))

    audit.to_csv(RES / "year_audit.csv", index=False)
    (RES / "year_audit_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'year_audit.csv'}")
    log(f"      {RES/'year_audit_summary.txt'}")


if __name__ == "__main__":
    main()
