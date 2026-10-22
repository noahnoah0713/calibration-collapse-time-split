"""
ATC 배정 검수 — KEGG 이름 매칭이 믿을 만한가

배경
----
ATC 계열별 층화(특히 "항종양제 L 계열에서 성능이 무너진다")는 전적으로
화합물 이름 -> KEGG -> ATC 매칭에 의존한다. 지금까지 이 매칭은 알려진
정답 5건만 눈으로 확인했을 뿐이다. 779/797 에 코드가 붙었지만 그게
맞는지는 본 적이 없다.

이 스크립트가 확인하는 것
-------------------------
  [1] 매칭 방식별 분포
      atc_kegg.py 의 lookup() 은 두 가지만 시도한다.
        (a) 정규화한 전체 이름이 KEGG 키에 그대로 있는 경우  -> 강한 매칭
        (b) 염 표기를 떼어낸 이름이 KEGG 키에 있는 경우       -> 약한 매칭
      퍼지 매칭은 없으므로 엉뚱한 약에 붙을 위험은 낮지만,
      (b) 로만 붙은 건은 따로 세어 둘 가치가 있다.

  [2] 두 데이터셋 교차 검증
      두 데이터셋에 같은 분자(InChIKey 14자리 기준)가 들어 있으면
      이름 표기가 달라도 같은 ATC 가 나와야 한다. 다르면 매칭 오류다.
      이름이 달라도 같은 답이 나오는지 보는 것이므로 실질적인 검정이다.

  [3] 미배정 목록
      코드가 안 붙은 화합물. 층화에서 통째로 빠지므로 목록을 남긴다.

  [4] 사람이 볼 목록 (results/atc_review_L.csv)
      논문의 하위집단 주장이 걸려 있는 L(항종양·면역조절) 배정 전체.
      수십 건이므로 눈으로 5분이면 확인된다. 이게 이 스크립트의
      진짜 산출물이다 — 자동으로 참/거짓을 가릴 방법이 없기 때문이다.

이 스크립트가 확인하지 못하는 것 (반드시 알고 있을 것)
------------------------------------------------------
  - KEGG 한 약물에 ATC 코드가 여러 개 달린 경우, atc_kegg.py 는
    sorted(...)[0] 로 알파벳 순 첫 코드만 남긴다. 예를 들어 D 와 L
    을 모두 가진 약은 D 로 기록된다. 캐시에는 채택된 코드 하나만
    남아 있어 여기서 되짚을 수 없다.
  - KEGG 이름 키를 만들 때 drop_duplicates("key") 로 충돌을 접었다.
    서로 다른 약이 같은 키로 정규화됐다면 먼저 온 쪽이 이긴다.
    이 역시 캐시만으로는 복원되지 않는다.
  두 항목은 rest.kegg.jp 를 다시 받아야 확인할 수 있는데, 현재
  네트워크에서 그 호스트가 막혀 있다(403). 확인하려면 허용목록에
  rest.kegg.jp 를 추가한 뒤 data/processed/kegg_atc_map.csv 를 지우고
  atc_kegg.py 를 다시 돌려야 한다. 그 전까지는 논문에서 L 계열
  결과를 "가설 생성" 이상으로 쓰지 말 것.

사용법
------
    python src/check_atc.py

출력
----
    results/atc_audit.csv        화합물별 매칭 방식과 배정 코드
    results/atc_review_L.csv     L 배정 전체 (사람 확인용)
    results/atc_audit_summary.txt
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 매칭 규칙은 본 분석 스크립트에서 그대로 가져온다 (구현 중복 금지)
from atc_kegg import ATC_NAMES, norm, strip_salt  # noqa: E402

DATASETS = [
    ("DILIrank", PROC / "dilirank2_with_year.csv"),
    ("DICTrank", PROC / "dictrank_processed.csv"),
]

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def classify(name, kmap):
    """(atc, 매칭방식, 매칭에 쓰인 키) 반환."""
    n, s = norm(name), strip_salt(name)
    if n in kmap:
        return kmap[n], "exact", n
    if s in kmap:
        return kmap[s], "salt_stripped", s
    return None, "unmatched", ""


def main():
    cache = PROC / "kegg_atc_map.csv"
    if not cache.exists():
        raise SystemExit(
            f"[중단] {cache} 가 없습니다. " f"먼저 python src/atc_kegg.py 를 실행하세요."
        )
    m = pd.read_csv(cache)
    kmap = dict(zip(m["key"], m["atc"]))

    log("ATC 배정 검수")
    log("=" * 74)
    log(f"KEGG 이름 키 {len(kmap):,}개 (캐시: {cache.name})")
    log()

    rows = []
    for name, path in DATASETS:
        if not path.exists():
            log(f"[건너뜀] {name}: {path} 없음")
            continue
        df = pd.read_csv(path).dropna(subset=["y", "smiles", "approval_year"])
        for r in df.itertuples():
            cmp_ = str(getattr(r, "CompoundName"))
            atc, how, key = classify(cmp_, kmap)
            rows.append(
                {
                    "dataset": name,
                    "compound": cmp_,
                    "ikey14": getattr(r, "ikey14", None),
                    "atc": atc,
                    "atc1": atc[0] if atc else None,
                    "atc1_name": ATC_NAMES.get(atc[0], "") if atc else "",
                    "match_type": how,
                    "matched_key": key,
                    "norm_name": norm(cmp_),
                    "salt_stripped": strip_salt(cmp_),
                    "approval_year": int(getattr(r, "approval_year")),
                    "y": int(getattr(r, "y")),
                }
            )

    if not rows:
        raise SystemExit("[중단] 검수할 데이터셋이 없습니다.")
    a = pd.DataFrame(rows)

    # ------------------------------------------------ [1] 매칭 방식
    log("[1] 매칭 방식별 분포")
    for ds, g in a.groupby("dataset"):
        n = len(g)
        vc = g["match_type"].value_counts()
        log(f"    {ds}  (n={n})")
        for k in ("exact", "salt_stripped", "unmatched"):
            v = int(vc.get(k, 0))
            log(f"      {k:14s} {v:4d}  ({v/n:5.1%})")
    log()
    log("    exact         = 정규화한 전체 이름이 KEGG 키와 일치. 가장 안전.")
    log("    salt_stripped = 염 표기를 뗀 뒤에야 일치. 대체로 정상이지만,")
    log("                    뗀 부분이 실제 활성성분의 일부였다면 오류가 된다.")
    log()

    # ------------------------------------------------ [2] 교차 검증
    log("[2] 두 데이터셋에 공통으로 있는 분자의 ATC 일치 여부")
    log("    (같은 분자인데 이름 표기가 달라도 같은 코드가 나와야 한다)")
    have = a[a["ikey14"].notna() & a["atc"].notna()]
    piv = have.pivot_table(index="ikey14", columns="dataset", values="atc", aggfunc="first")
    n_conflict = 0
    if {"DILIrank", "DICTrank"} <= set(piv.columns):
        both = piv.dropna()
        same = both["DILIrank"] == both["DICTrank"]
        same1 = both["DILIrank"].str[0] == both["DICTrank"].str[0]
        n_conflict = int((~same1).sum())
        log(f"    양쪽 모두 코드가 붙은 공통 분자: {len(both)}개")
        log(f"      전체 코드까지 일치 : {int(same.sum())}개")
        log(f"      1단계(대분류) 일치 : {int(same1.sum())}개")
        log(f"      1단계 불일치       : {n_conflict}개")
        if n_conflict:
            bad = both[~same1].copy()
            nm = have.drop_duplicates(["ikey14", "dataset"]).pivot_table(
                index="ikey14", columns="dataset", values="compound", aggfunc="first"
            )
            bad = bad.join(nm, rsuffix="_name")
            log("      !! 아래 분자는 데이터셋마다 다른 대분류를 받았다:")
            log(bad.head(30).to_string())
    else:
        log("    한쪽 데이터셋만 있어 교차 검증 불가")
    log()

    # ------------------------------------------------ [3] 미배정
    log("[3] ATC 미배정 화합물 (층화 분석에서 제외됨)")
    for ds, g in a.groupby("dataset"):
        un = g[g["match_type"] == "unmatched"]
        log(f"    {ds}: {len(un)}개")
        if len(un):
            log("      " + ", ".join(un["compound"].head(30).tolist()))
    log()

    # ------------------------------------------------ [4] L 검토 목록
    log("[4] L(항종양·면역조절) 배정 — 사람이 확인할 목록")
    L = a[a["atc1"] == "L"].copy()
    log(f"    전체 {len(L)}개")
    for ds, g in L.groupby("dataset"):
        held = g[g["approval_year"] >= 2010]
        log(f"    {ds}: {len(g)}개 (그중 검증군 {len(held)}개)")
        log(
            "      매칭 방식: "
            + ", ".join(f"{k} {v}" for k, v in g["match_type"].value_counts().items())
        )
    L.sort_values(["dataset", "compound"]).to_csv(RES / "atc_review_L.csv", index=False)
    log()
    log("    -> results/atc_review_L.csv 를 열어 '항종양제가 맞는가' 를")
    log("       한 줄씩 확인할 것. 자동으로 가릴 방법이 없다.")
    log("       특히 검증군(승인 2010년 이후) 항목이 중요하다. 논문의")
    log("       L 계열 주장이 그 표본에 걸려 있다.")
    log()

    a.to_csv(RES / "atc_audit.csv", index=False)

    # ------------------------------------------------ 판정
    log("=" * 74)
    log("판정")
    log("=" * 74)
    n_un = int((a["match_type"] == "unmatched").sum())
    n_salt = int((a["match_type"] == "salt_stripped").sum())
    log(
        f"  미배정 {n_un}개 | 염 제거 후 매칭 {n_salt}개 | "
        f"데이터셋 간 대분류 불일치 {n_conflict}개"
    )
    log()
    if n_conflict == 0:
        log("  이름 표기가 다른 같은 분자들이 모두 같은 대분류를 받았다.")
        log("  매칭이 이름 표기에 흔들리지 않는다는 뜻이다.")
    else:
        log("  !! 대분류 불일치가 있다. 위 표를 확인하고 고칠 것.")
    log()
    log("  확인하지 못한 것 (이 스크립트의 한계):")
    log("    - KEGG 한 약물에 ATC 가 여러 개일 때 알파벳 첫 코드만 채택됨")
    log("    - KEGG 이름 키 충돌 시 먼저 온 약물이 이김")
    log("    두 항목은 rest.kegg.jp 재수집이 필요한데 현재 막혀 있다(403).")
    log("    그때까지 L 계열 결과는 '가설 생성' 으로만 서술할 것.")
    log("    (다중비교 보정 후에도 L 은 유의하지 않으므로 결론은 어차피 같다.)")

    (RES / "atc_audit_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'atc_audit.csv'}")
    log(f"      {RES/'atc_review_L.csv'}")
    log(f"      {RES/'atc_audit_summary.txt'}")


if __name__ == "__main__":
    main()
