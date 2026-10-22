"""
라벨 충돌 화합물 제외 필터

무엇을 거르나
-------------
구조 표준화 후 InChIKey 앞 14자리로 중복을 제거하는데, 14자리는
입체화학과 염을 버린다. 그래서 서로 다른 제품이 한 줄로 접힌다.
접힌 행들의 라벨(y)이 서로 다르면 keep="first" 가 사실상 라벨을
동전던지기로 고른 것이 된다.

  예) VANCOMYCIN y=1  /  vancomycin hydrochloride y=0
      ACYCLOVIR  y=0  /  ACYCLOVIR SODIUM        y=1
      Ferric pyrophosphate citrate / Gallium citrate / Lithium citrate
      / Sodium citrate  -> 넷 다 시트르산 이온 하나로 접힘 (라벨 1,0,1,0)

어느 쪽이 맞는지 알 방법이 없으므로 한쪽을 고르지 않고 그룹 전체를
분석에서 뺀다. 이 규칙은 결과를 보지 않고 적용된다 — 어느 라벨이
논문에 유리한지와 무관하게 충돌한 그룹은 전부 빠진다.

주의: 이것은 "무엇이 맞는가" 를 판정하는 필터가 아니라 "모르는 것을
      모른 채로 두는" 필터다. 제외된 목록은 보충자료에 싣고, 본문
      Methods 에 이 규칙을 그대로 적을 것.

사용법
------
    python src/apply_label_conflict_filter.py

  멱등이다. 여러 번 돌려도 같은 결과가 되고, 최초 1회만 백업한다.
  apply_structure_filter.py 다음에 돌린다.

출력
----
    data/processed/*_preconflict.csv          (최초 1회 백업)
    data/processed/dilirank2_with_year.csv    (제자리 수정)
    data/processed/dictrank_processed.csv     (제자리 수정)
    results/label_conflict_filter_log.txt
    results/label_conflict_groups.csv         제외된 그룹 전체 목록
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

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
# 표준화와 충돌 탐지는 본 분석·검수 스크립트에서 그대로 가져온다
from baseline import standardize  # noqa: E402
from check_labels import build_dictrank_prededup  # noqa: E402

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def ikey14(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToInchiKey(m)[:14] if m else None


def conflicted_keys(pre, name):
    """중복 제거 이전 표에서 라벨이 충돌하는 ikey14 집합과 그 내역."""
    d = pre.dropna(subset=["y", "smiles_raw"]).copy()
    d["_std"] = d["smiles_raw"].apply(standardize)
    d = d.dropna(subset=["_std"])
    d["_k"] = d["_std"].apply(ikey14)
    d = d.dropna(subset=["_k"])
    g = d.groupby("_k")["y"].nunique()
    keys = set(g[g > 1].index)
    detail = d[d["_k"].isin(keys)][["CompoundName", "y", "_k"]].copy()
    detail.insert(0, "dataset", name)
    return keys, detail.sort_values(["_k", "CompoundName"])


def apply_to(path, keys, name):
    """처리된 CSV 에서 충돌 키에 해당하는 행을 뺀다. 멱등."""
    if not path.exists():
        log(f"[건너뜀] {name}: {path.name} 없음")
        return 0
    backup = path.with_name(path.stem + "_preconflict.csv")
    df = pd.read_csv(path)
    if "ikey14" not in df.columns:
        log(f"[건너뜀] {name}: ikey14 열이 없다")
        return 0

    drop = df[df["ikey14"].isin(keys)]
    if not len(drop):
        log(f"{name}: 제외할 행 없음 (이미 적용됨 또는 해당 없음) — {len(df)}행 유지")
        return 0

    if not backup.exists():
        df.to_csv(backup, index=False)
        log(f"{name}: 원본 백업 -> {backup.name}")

    log(f"\n{name}   {len(df)} -> {len(df) - len(drop)}   (제외 {len(drop)}건)")
    held = drop[drop["approval_year"] >= 2010] if "approval_year" in drop else drop
    log(f"  학습군 {len(drop)-len(held)}건, 검증군 {len(held)}건")
    cols = [c for c in ["CompoundName", "approval_year", "y", "ikey14"] if c in drop.columns]
    log(drop[cols].to_string(index=False))

    df[~df["ikey14"].isin(keys)].reset_index(drop=True).to_csv(path, index=False)
    return len(drop)


def main():
    log("라벨 충돌 필터: 같은 구조로 접히면서 라벨이 엇갈린 그룹을 제외")
    log("=" * 74)
    log("규칙: 충돌한 그룹은 어느 쪽도 채택하지 않고 전부 뺀다.")
    log("      결과를 보지 않고 적용되는 규칙이다.")
    log()

    details, total = [], 0

    # ---------------------------------------------------------- DILIrank
    pre = PROC / "dilirank2_with_smiles.csv"
    if pre.exists():
        keys, det = conflicted_keys(pd.read_csv(pre), "DILIrank")
        log(f"DILIrank: 라벨 충돌 그룹 {len(keys)}개")
        details.append(det)
        total += apply_to(PROC / "dilirank2_with_year.csv", keys, "DILIrank")
    else:
        log(f"[건너뜀] DILIrank: {pre.name} 없음")
    log()

    # ---------------------------------------------------------- DICTrank
    fr = build_dictrank_prededup()
    if fr is not None:
        keys, det = conflicted_keys(fr, "DICTrank")
        log(f"DICTrank: 라벨 충돌 그룹 {len(keys)}개")
        details.append(det)
        total += apply_to(PROC / "dictrank_processed.csv", keys, "DICTrank")
    else:
        log("[건너뜀] DICTrank: 원자료 또는 SMILES 캐시 없음")

    if details:
        pd.concat(details, ignore_index=True).to_csv(RES / "label_conflict_groups.csv", index=False)

    log()
    log("=" * 74)
    log(f"총 {total}건 제외")
    log()
    log("되돌리려면 *_preconflict.csv 를 원래 이름으로 복사하면 된다.")
    if total:
        log("이 필터를 적용한 뒤에는 모든 하류 분석을 다시 돌려야 한다.")
    log(f"제외된 그룹 전체 내역: {RES/'label_conflict_groups.csv'}")
    log("  (보충자료용. 본문 Methods 에 규칙을 그대로 적을 것)")

    (RES / "label_conflict_filter_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'label_conflict_filter_log.txt'}")


if __name__ == "__main__":
    main()
