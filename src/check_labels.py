"""
라벨 이진화 검수 — 원자료에서 y 까지 역추적

왜 필요한가
-----------
두 데이터셋 모두 다범주 독성 등급을 0/1 로 접어서 쓴다. 이 변환이
틀리면 논문의 모든 수치가 동시에 틀린다. 그런데 지금까지 이 변환을
원자료와 대조한 적이 없었다. 중간 산출 CSV 를 믿고 이어 달렸을 뿐이다.

이 스크립트는 중간 산출물을 일절 믿지 않는다. 엑셀 원본을 직접 열어
규칙을 다시 적용하고, 분석에 실제로 쓰인 CSV 의 y 와 한 행씩 맞춰 본다.

검수하는 규칙
-------------
  DILIrank 2.0  vDILI-Concern 열
      vMost-DILI-concern  -> y = 1
      vLess-DILI-concern  -> y = 1
      vNo-DILI-concern    -> y = 0
      Ambiguous-DILI-concern -> 제외
      (대소문자 표기가 섞여 있다: vMOST, vNo-DILI-Concern 등)

  DICTrank      DICT _ Concern 열
      most -> 1 / less -> 1 / no -> 0 / ambiguous -> 제외

  두 데이터셋 모두 "most 와 less 를 합쳐 양성" 이다. 이 선택 자체는
  설계 결정이지 오류가 아니다. 다만 본문 Methods 에 그대로 적혀
  있는지 확인할 것. 여기서는 "적힌 규칙대로 실행됐는가" 만 본다.

같이 보는 것
------------
  - 원자료의 중증도 열(SeverityClass / DIC Severity Level)과
    concern 열이 서로 모순되지 않는지 교차표로 확인
  - 원자료에는 있는데 분석 집합에서 사라진 행이 어느 단계에서
    빠졌는지 (Ambiguous / SMILES 실패 / 중복 / 구조 필터)
  - 분석 집합에는 있는데 원자료에서 못 찾는 행 (있으면 심각)
  - 구조 중복 제거(InChIKey 앞 14자리)로 서로 다른 화합물이 합쳐지면서
    라벨이 조용히 덮어써진 경우. 14자리 키는 입체화학을 버리므로
    거울상 이성질체가 한 줄로 접힌다.

사용법
------
    python src/check_labels.py

출력
----
    results/label_audit.csv          행 단위 대조 결과
    results/label_audit_summary.txt
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

DILI_XLSX = RAW / "Drug Induced Liver Injury Rank (DILIrank 2.0) Dataset  FDA.xlsx"
DICT_XLSX = RAW / "dictrank_dataset_508.xlsx"

POSITIVE = {"most", "less"}
NEGATIVE = {"no"}
EXCLUDE = {"ambiguous"}

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def concern_token(v):
    """'vMOST-DILI-concern', 'less', 'Ambiguous-DILI-concern' -> 'most'/'less'/'no'/'ambiguous'."""
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    s = s.replace("-dili-concern", "").replace("_", " ").strip()
    if s.startswith("v"):
        s = s[1:]
    s = s.strip()
    return s if s in (POSITIVE | NEGATIVE | EXCLUDE) else f"?{s}"


def expected_y(tok):
    if tok in POSITIVE:
        return 1
    if tok in NEGATIVE:
        return 0
    return None  # ambiguous 또는 해석 불가 -> 제외 대상


def compare(name, raw, proc, key, concern_col, severity_col):
    """raw(원자료)와 proc(분석집합)을 key 로 맞춰 y 를 대조한다."""
    log("=" * 74)
    log(f"{name}")
    log("=" * 74)

    raw = raw.copy()
    raw["_tok"] = raw[concern_col].map(concern_token)
    raw["_y_expected"] = raw["_tok"].map(expected_y)

    log(f"\n[1] 원자료 {len(raw)}행의 concern 열 분포")
    vc = raw["_tok"].value_counts(dropna=False)
    for k, v in vc.items():
        mark = ""
        if isinstance(k, str) and k.startswith("?"):
            mark = "   <-- 규칙에 없는 값. 확인 필요"
        log(f"    {str(k):12s} {v:5d}{mark}")

    unknown = raw[raw["_tok"].astype(str).str.startswith("?")]
    if len(unknown):
        log(f"\n    !! 규칙에 없는 값 {len(unknown)}행. 아래 표에서 확인할 것:")
        log(unknown[[key, concern_col]].head(20).to_string(index=False))

    n_pos = int((raw["_y_expected"] == 1).sum())
    n_neg = int((raw["_y_expected"] == 0).sum())
    n_drop = int(raw["_y_expected"].isna().sum())
    log(
        f"\n[2] 규칙 적용 결과: 양성 {n_pos}, 음성 {n_neg}, 제외 {n_drop} "
        f"(합 {n_pos+n_neg+n_drop} / 원자료 {len(raw)})"
    )

    # 중증도 열과의 교차 확인
    if severity_col and severity_col in raw.columns:
        log(
            f"\n[3] {severity_col} x concern 교차표 "
            f"(모순이 있으면 한 칸에 몰리지 않고 흩어진다)"
        )
        ct = pd.crosstab(raw[severity_col].astype(str), raw["_tok"])
        log(ct.to_string())
    else:
        log("\n[3] 중증도 열 없음 - 교차 확인 생략")

    # ---- 분석 집합과 대조
    log(f"\n[4] 분석 집합 {len(proc)}행과 행 단위 대조 (키: {key})")
    r = raw[[key, concern_col, "_tok", "_y_expected"]].copy()
    r[key] = r[key].astype(str).str.strip()

    p = proc[[key, "y"]].copy()
    p[key] = p[key].astype(str).str.strip()
    p = p.rename(columns={"y": "y_actual"})

    n_raw_rows = len(r)
    dup_raw = int(r[key].duplicated().sum())
    dup_proc = int(p[key].duplicated().sum())
    n_conflict = 0
    if dup_raw or dup_proc:
        log(f"    [주의] 키 중복 — 원자료 {dup_raw}건, 분석집합 {dup_proc}건.")
        # 중복 키가 서로 다른 라벨을 갖고 있으면 "첫 행 채택"이 임의 선택이
        # 되어 버린다. 그런 경우가 있는지 먼저 본다.
        g = r.groupby(key)["_y_expected"].nunique(dropna=False)
        conflict_keys = g[g > 1].index.tolist()
        n_conflict = len(conflict_keys)
        if n_conflict:
            log(
                f"    !! 그중 {n_conflict}개 키는 중복 행끼리 라벨이 다르다."
                f"  <-- 첫 행 채택이 임의 선택이 된다. 확인 필요:"
            )
            log(
                r[r[key].isin(conflict_keys)][[key, concern_col, "_tok", "_y_expected"]]
                .sort_values(key)
                .head(30)
                .to_string(index=False)
            )
        else:
            log("    중복 키끼리 라벨은 모두 같다 — 첫 행 채택이 안전하다.")
        r = r.drop_duplicates(key)
        p = p.drop_duplicates(key)

    m = p.merge(r, on=key, how="left", indicator=True)

    orphan = m[m["_merge"] == "left_only"]
    if len(orphan):
        log(
            f"    !! 분석집합에 있는데 원자료에서 못 찾은 행: {len(orphan)}건"
            f"  <-- 심각. 아래 확인:"
        )
        log(orphan[[key]].head(20).to_string(index=False))
    else:
        log("    분석집합의 모든 행을 원자료에서 찾음 (고아 0건)")

    ok = m[m["_merge"] == "both"].copy()
    ok["y_actual"] = ok["y_actual"].astype(float)
    mism = ok[ok["y_actual"] != ok["_y_expected"]]
    log(f"    y 불일치: {len(mism)}건 / 대조 가능 {len(ok)}건")
    if len(mism):
        log("    !! 아래 행에서 원자료 규칙과 분석집합의 y 가 다르다:")
        log(
            mism[[key, concern_col, "_tok", "_y_expected", "y_actual"]]
            .head(30)
            .to_string(index=False)
        )

    # ---- 탈락 경로
    log("\n[5] 원자료에서 분석집합까지 줄어든 경로")
    eligible = r[r["_y_expected"].notna()]
    in_proc = set(p[key])
    lost = eligible[~eligible[key].isin(in_proc)]
    if n_raw_rows != len(r):
        log(f"    원자료 {n_raw_rows}행 (키 중복 {n_raw_rows - len(r)}행 접음)")
    log(f"    고유 키 {len(r)}행")
    log(f"      - concern 제외(ambiguous 등) {len(r) - len(eligible)}행")
    log(f"      = 이진화 대상 {len(eligible)}행")
    log(f"      - 이후 단계에서 탈락 {len(lost)}행")
    log("        (SMILES 확보 실패 / 승인연도 미매칭 / 구조 중복 제거 /")
    log("         탄소 없는 화합물 필터 — 각각은 별도 로그에 기록됨)")
    log(f"      = 분석집합 {len(p)}행")

    out = m.drop(columns=["_merge"]).copy()
    out.insert(0, "dataset", name)
    out["mismatch"] = out["y_actual"] != out["_y_expected"]
    return out, len(mism), len(orphan), len(unknown), n_conflict


def build_dictrank_prededup():
    """DICTrank 의 '중복 제거 이전' 표를 원자료 + SMILES 캐시로 되살린다.

    dictrank_pipeline.py 를 다시 돌리면 처리된 CSV 가 새로 만들어지면서
    구조 필터가 되돌아가고 전체 재실행이 필요해진다. 검수 하나 때문에
    그럴 필요는 없으므로, 그 스크립트가 쓰는 것과 같은 입력(엑셀 원자료
    + data/processed/smiles_cache.json)만 읽어 같은 표를 다시 만든다.
    네트워크를 쓰지 않고, 어떤 파일도 건드리지 않는다.
    """
    cache_path = PROC / "smiles_cache.json"
    if not (DICT_XLSX.exists() and cache_path.exists()):
        return None
    import json

    cache = json.loads(cache_path.read_text())

    raw = pd.read_excel(DICT_XLSX, sheet_name="Table S1")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.rename(columns={"Active Ingredient(s)": "CompoundName", "DICT _ Concern": "concern"})
    raw["_tok"] = raw["concern"].map(concern_token)
    raw["y"] = raw["_tok"].map(expected_y)
    raw = raw.dropna(subset=["y", "CompoundName"]).copy()

    def smi(n):
        v = cache.get(str(n).strip())
        return v.get("smiles_raw") if isinstance(v, dict) else None

    raw["smiles_raw"] = raw["CompoundName"].map(smi)
    return raw.dropna(subset=["smiles_raw"]).reset_index(drop=True)


def check_structure_merge(name, path, smiles_col="smiles_raw", frame=None):
    """구조 중복 제거로 서로 다른 화합물이 합쳐진 곳을 찾는다.

    baseline.py 는 InChIKey 앞 14자리로 중복을 제거한다. 14자리는
    분자 골격만 담고 입체화학과 염을 버리므로, 거울상 이성질체나
    복합제가 한 줄로 접힐 수 있다. 접힌 두 행의 라벨이 다르면
    keep='first' 가 사실상 라벨을 임의로 고른 셈이 된다.
    """
    log("-" * 74)
    log(f"[6] {name} — 구조 중복 제거로 합쳐진 화합물")
    log("-" * 74)
    if frame is None and not path.exists():
        log(f"    건너뜀: {path.name} 없음 (중복 제거 이전 표가 필요하다)")
        return 0
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from baseline import standardize
    except Exception as e:  # noqa: BLE001
        log(f"    건너뜀: {type(e).__name__} {e}")
        return 0

    d = (
        (frame.copy() if frame is not None else pd.read_csv(path))
        .dropna(subset=["y", smiles_col])
        .copy()
    )
    d["_std"] = d[smiles_col].apply(standardize)
    d = d.dropna(subset=["_std"])
    d["_k"] = d["_std"].apply(
        lambda x: (
            Chem.MolToInchiKey(Chem.MolFromSmiles(x))[:14] if Chem.MolFromSmiles(x) else None
        )
    )
    d = d.dropna(subset=["_k"])

    g = d.groupby("_k")
    coll = g.filter(lambda x: len(x) > 1)
    n_group = coll["_k"].nunique()
    log(f"    표준화 통과 {len(d)}행")
    log(
        f"    같은 14자리 키를 갖는 서로 다른 이름: {n_group}그룹 "
        f"/ {len(coll)}행  -> 중복 제거로 {len(coll)-n_group}행 사라짐"
    )

    conf = g.filter(lambda x: len(x) > 1 and x["y"].nunique() > 1)
    n_conf_group = conf["_k"].nunique() if len(conf) else 0
    if n_conf_group:
        log(f"    !! 그중 라벨(y)이 서로 다른 그룹: {n_conf_group}그룹 " f"/ {len(conf)}행")
        log("       합쳐질 때 한쪽 라벨이 버려졌다. 아래를 확인할 것:")
        log(
            conf[["CompoundName", "y", "_k"]]
            .sort_values(["_k", "CompoundName"])
            .to_string(index=False)
        )
        log()
        log("       선택지: (a) Methods 에 그대로 밝히고 둔다")
        log("               (b) 해당 쌍을 분석에서 제외한다")
        log("               (c) 27자리 전체 InChIKey 로 중복을 제거한다")
        log("                   (염은 이미 표준화에서 떨어졌으므로 이 경우")
        log("                    입체이성질체만 분리된다. 단 전체 재실행 필요)")
    else:
        log("    합쳐진 그룹끼리 라벨은 모두 같다 — 라벨이 버려진 곳은 없다.")
    log()
    return n_conf_group


def main():
    log("라벨 이진화 검수 — 엑셀 원자료에서 분석용 y 까지 역추적")
    log("중간 산출 CSV 는 쓰지 않는다. 원자료를 직접 다시 읽는다.")
    log()

    problems = []
    frames = []

    # ------------------------------------------------------- DILIrank
    if DILI_XLSX.exists() and (PROC / "dilirank2_with_year.csv").exists():
        raw = pd.read_excel(DILI_XLSX, skiprows=1)
        raw.columns = [str(c).strip() for c in raw.columns]
        proc = pd.read_csv(PROC / "dilirank2_with_year.csv")
        proc = proc.dropna(subset=["y", "smiles", "approval_year"])
        f, nm, no_, nu, nc = compare(
            "DILIrank", raw, proc, "LTKBID", "vDILI-Concern", "SeverityClass"
        )
        frames.append(f)
        problems.append(("DILIrank", nm, no_, nu, nc))
        log()
    else:
        log("[건너뜀] DILIrank: 원자료 또는 처리 CSV 없음")

    # ------------------------------------------------------- DICTrank
    if DICT_XLSX.exists() and (PROC / "dictrank_processed.csv").exists():
        raw = pd.read_excel(DICT_XLSX, sheet_name="Table S1")
        raw.columns = [str(c).strip() for c in raw.columns]
        raw = raw.rename(
            columns={"Active Ingredient(s)": "CompoundName", "DICT _ Concern": "concern"}
        )
        proc = pd.read_csv(PROC / "dictrank_processed.csv")
        proc = proc.dropna(subset=["y", "smiles", "approval_year"])
        f, nm, no_, nu, nc = compare(
            "DICTrank", raw, proc, "CompoundName", "concern", "DIC Severity Level"
        )
        frames.append(f)
        problems.append(("DICTrank", nm, no_, nu, nc))
        log()
    else:
        log("[건너뜀] DICTrank: 원자료 또는 처리 CSV 없음")

    n_merge = check_structure_merge("DILIrank", PROC / "dilirank2_with_smiles.csv")
    dict_pre = PROC / "dictrank_prededup.csv"
    if dict_pre.exists():
        n_merge += check_structure_merge("DICTrank", dict_pre)
    else:
        # 저장된 스냅샷이 없으면 원자료 + SMILES 캐시로 되살려서 검수한다.
        fr = build_dictrank_prededup()
        if fr is None:
            log("-" * 74)
            log("[6] DICTrank — 건너뜀: 원자료 또는 SMILES 캐시 없음")
            log("-" * 74)
            log()
        else:
            n_merge += check_structure_merge("DICTrank", dict_pre, frame=fr)

    if not frames:
        raise SystemExit("[중단] 검수할 데이터셋이 없습니다.")

    pd.concat(frames, ignore_index=True).to_csv(RES / "label_audit.csv", index=False)

    log("=" * 74)
    log("판정")
    log("=" * 74)
    total = 0
    log(f"  구조 중복 제거로 라벨이 버려진 그룹: {n_merge}개")
    for name, nm, no_, nu, nc in problems:
        total += nm + no_ + nu + nc
        log(
            f"  {name:9s} y 불일치 {nm}건 | 원자료 미발견 {no_}건 | "
            f"규칙 밖 concern 값 {nu}건 | 중복키 라벨충돌 {nc}건"
        )
    log()
    if total == 0:
        log("  => 라벨 이진화는 원자료와 완전히 일치한다.")
        log("     (이 검수가 보장하는 것: 적힌 규칙이 그대로 실행됐다는 것.")
        log("      most+less 를 양성으로 묶는 선택이 타당한가는 별개 문제이며,")
        log("      Methods 와 Discussion 에서 근거를 밝혀야 한다.)")
    else:
        log("  => 문제가 발견되었다. 위 표를 확인하고 고치기 전에는")
        log("     어떤 수치도 논문에 넣지 말 것.")

    (RES / "label_audit_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'label_audit.csv'}")
    log(f"      {RES/'label_audit_summary.txt'}")
    if total:
        sys.exit(1)


if __name__ == "__main__":
    main()
