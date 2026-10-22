"""
구조 필터 적용: 표준화 후 탄소가 없는 화합물 제외

왜 필요한가:
    표준화 함수는 '탄소를 포함한 가장 큰 조각'을 남긴다. 그런데 탄소를
    포함한 조각이 아예 없으면 그냥 가장 큰 조각을 가져간다. 무기물·
    금속착물·방사성 진단제에서 이 분기가 타고, 그 결과 다음이 나온다.

        Ferumoxytol          -> [O-2]
        Arsenic trioxide     -> [O-2]
        Sodium nitroprusside -> [C-]#N
        Cisplatin            -> [Cl][Pt][Cl]
        Sterile water        -> O

    Morgan 지문은 원자 환경의 해시다. 위 구조들에 대해 정의는 되지만
    의미가 없다. 이런 화합물을 QSAR 적용 범위에서 제외하는 것은 표준
    관행이다.

무엇을 하는가:
    처리된 CSV 두 개에서 '표준화 후 탄소 원자가 하나도 없는' 행을
    제거한다. 원본은 *_prefilter.csv 로 한 번만 백업한다.

    탄소가 있으면 작아도 남긴다. 하이드록시우레아(76), DMSO(78),
    에탄올(46) 등은 지문이 빈약할 뿐 구조는 옳다. 크기로 자르면
    실제 독성 약물을 근거 없이 버리게 된다.

    이미 적용된 파일에 다시 돌려도 안전하다 (제거할 행이 0건).

주의:
    map_approval_year.py 나 dictrank_pipeline.py 를 다시 돌리면 처리
    CSV가 새로 만들어지므로, 그 뒤에는 이 스크립트를 다시 돌려야 한다.

사용법:
    python src/apply_structure_filter.py

출력:
    data/processed/*_prefilter.csv   (최초 1회 백업)
    results/structure_filter_log.txt
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CUTOFF = 2010
TARGETS = [("DILIrank", "dilirank2_with_year.csv"), ("DICTrank", "dictrank_processed.csv")]

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def has_carbon(smi):
    """표준화된 SMILES에 탄소 원자가 있는가."""
    if not isinstance(smi, str) or not smi:
        return False
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return any(a.GetSymbol() == "C" for a in m.GetAtoms())


def main():
    log("구조 필터: 표준화 후 탄소가 없는 화합물 제외")
    log("=" * 70)

    total_removed = 0
    for name, fn in TARGETS:
        path = PROC / fn
        if not path.exists():
            log(f"{name}: {fn} 없음 - 건너뜀")
            continue

        backup = PROC / fn.replace(".csv", "_prefilter.csv")
        df = pd.read_csv(path)

        if not backup.exists():
            df.to_csv(backup, index=False)
            log(f"{name}: 원본 백업 -> {backup.name}")
        else:
            log(f"{name}: 백업이 이미 있음 ({backup.name}) - 덮어쓰지 않음")

        before = len(df)
        keep = df["smiles"].apply(has_carbon)
        removed = df[~keep].copy()
        df2 = df[keep].reset_index(drop=True)

        log("")
        log(f"{name}   {before} -> {len(df2)}   (제거 {len(removed)}건)")

        if len(removed):
            cols = [
                c for c in ["CompoundName", "approval_year", "y", "smiles"] if c in removed.columns
            ]
            r = removed[cols].copy()
            if "approval_year" in r.columns:
                r["group"] = [
                    "검증군" if pd.notna(v) and v >= CUTOFF else "학습군"
                    for v in r["approval_year"]
                ]
                n_ho = int((r["group"] == "검증군").sum())
                log(f"  학습군 {len(r)-n_ho}건, 검증군 {n_ho}건")
            log(r.to_string(index=False))

        df2.to_csv(path, index=False)
        total_removed += len(removed)
        log("")

    log("=" * 70)
    log(f"총 {total_removed}건 제거")
    log("")
    log("되돌리려면 *_prefilter.csv 를 원래 이름으로 복사하면 된다.")
    log("이 필터를 적용한 뒤에는 모든 하류 분석을 다시 돌려야 한다.")

    (RES / "structure_filter_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'structure_filter_log.txt'}")


if __name__ == "__main__":
    main()
