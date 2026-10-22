"""
DILIrank 2.0 화합물명 -> PubChem SMILES 조회

사용법:
    python src/fetch_smiles.py

입력:  data/processed/dilirank2_normalized.csv
출력:  data/processed/dilirank2_with_smiles.csv
       data/processed/smiles_cache.json   (중간 저장 - 중단돼도 이어서 실행 가능)
       data/processed/smiles_failed.csv   (조회 실패 목록 - 수작업 확인용)

중단됐다가 다시 실행하면 캐시에 있는 것은 건너뛰고 이어서 진행합니다.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------- 경로 설정
ROOT = Path(__file__).resolve().parents[1]  # ~/research/dilirank
PROC = ROOT / "data" / "processed"

IN_CSV = PROC / "dilirank2_normalized.csv"
OUT_CSV = PROC / "dilirank2_with_smiles.csv"
CACHE = PROC / "smiles_cache.json"
FAILED = PROC / "smiles_failed.csv"

# ---------------------------------------------------------------- 설정
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SLEEP = 0.25  # PubChem 권장: 초당 5회 이하
TIMEOUT = 20
RETRY = 2


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))


def query_pubchem(name: str) -> dict | None:
    """화합물명으로 PubChem 조회. 성공 시 dict, 실패 시 None."""
    url = (
        f"{BASE}/compound/name/{requests.utils.quote(name)}/property/"
        f"CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,"
        f"InChIKey/JSON"
    )
    for attempt in range(RETRY + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                props = r.json()["PropertyTable"]["Properties"][0]
                return {
                    "pubchem_cid": props.get("CID"),
                    "smiles_raw": props.get("IsomericSMILES") or props.get("CanonicalSMILES"),
                    "formula": props.get("MolecularFormula"),
                    "mw": props.get("MolecularWeight"),
                    "inchikey": props.get("InChIKey"),
                }
            if r.status_code == 404:
                return None  # 이름을 못 찾음 - 재시도 무의미
            time.sleep(1.0)  # 503 등 일시 오류
        except requests.RequestException:
            time.sleep(1.0)
    return None


def main() -> None:
    df = pd.read_csv(IN_CSV)
    names = df["CompoundName"].astype(str).str.strip().tolist()

    cache = load_cache()
    todo = [n for n in dict.fromkeys(names) if n not in cache]

    print(f"전체 고유 화합물: {len(set(names))}개")
    print(f"캐시에 있음:     {len(cache)}개")
    print(f"조회할 대상:     {len(todo)}개")
    print(f"예상 소요:       약 {len(todo) * SLEEP / 60:.0f}분\n")

    for i, name in enumerate(todo, 1):
        cache[name] = query_pubchem(name)
        time.sleep(SLEEP)

        if i % 25 == 0 or i == len(todo):
            hit = sum(1 for v in cache.values() if v)
            print(f"  [{i}/{len(todo)}] 진행 중 · 누적 성공 {hit}건")
            save_cache(cache)

    save_cache(cache)

    # ------------------------------------------------------ 결과 병합
    fields = ["pubchem_cid", "smiles_raw", "formula", "mw", "inchikey"]
    for f in fields:
        df[f] = [(cache.get(n) or {}).get(f) for n in names]

    df.to_csv(OUT_CSV, index=False)

    ok = df["smiles_raw"].notna().sum()
    ng = len(df) - ok
    print(f"\n{'='*50}")
    print(f"성공: {ok}/{len(df)}  ({ok/len(df):.1%})")
    print(f"실패: {ng}건")
    print(f"저장: {OUT_CSV}")

    if ng:
        cols = ["LTKBID", "CompoundName", "label", "Comment"]
        cols = [c for c in cols if c in df.columns]
        df.loc[df["smiles_raw"].isna(), cols].to_csv(FAILED, index=False)
        print(f"실패 목록: {FAILED}")
        print("\n실패한 이름 예시:")
        for n in df.loc[df["smiles_raw"].isna(), "CompoundName"].head(15):
            print("   -", n)


if __name__ == "__main__":
    main()
