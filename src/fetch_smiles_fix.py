"""
SMILES 조회 수정판

이전 실행에서 PubChem 응답 키 이름이 바뀐 것을 반영하지 못해 SMILES가 비었음.
(요청은 IsomericSMILES로 보내도 응답 키는 SMILES로 옴)

캐시에 CID가 이미 저장되어 있으므로, CID로 일괄 조회하여 빠르게 채운다.
CID가 없는 항목만 이름으로 재조회한다.

사용법:
    python src/fetch_smiles_fix.py

출력:
    data/processed/dilirank2_with_smiles.csv
    data/processed/smiles_failed.csv
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

IN_CSV = PROC / "dilirank2_normalized.csv"
CACHE = PROC / "smiles_cache.json"
OUT_CSV = PROC / "dilirank2_with_smiles.csv"
FAILED = PROC / "smiles_failed.csv"

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,InChIKey"
BATCH = 100
SLEEP = 0.3


def pick_smiles(p: dict):
    """새 키 이름과 옛 키 이름 모두 대응."""
    for k in ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"):
        if p.get(k):
            return p[k]
    return None


def fetch_by_cids(cids: list[int]) -> dict[int, dict]:
    """CID 목록으로 일괄 조회 (POST)."""
    out = {}
    for i in range(0, len(cids), BATCH):
        chunk = cids[i : i + BATCH]
        try:
            r = requests.post(
                f"{BASE}/compound/cid/property/{PROPS}/JSON",
                data={"cid": ",".join(map(str, chunk))},
                timeout=60,
            )
            if r.status_code == 200:
                for p in r.json()["PropertyTable"]["Properties"]:
                    out[p["CID"]] = {
                        "smiles_raw": pick_smiles(p),
                        "formula": p.get("MolecularFormula"),
                        "mw": p.get("MolecularWeight"),
                        "inchikey": p.get("InChIKey"),
                    }
            else:
                print(f"  배치 {i//BATCH + 1} 실패: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"  배치 {i//BATCH + 1} 오류: {e}")
        print(f"  배치 {i//BATCH + 1}/{(len(cids)-1)//BATCH + 1} 완료 " f"(누적 {len(out)}건)")
        time.sleep(SLEEP)
    return out


def fetch_by_name(name: str) -> dict | None:
    """이름으로 개별 조회."""
    url = f"{BASE}/compound/name/{requests.utils.quote(name)}" f"/property/{PROPS}/JSON"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            p = r.json()["PropertyTable"]["Properties"][0]
            return {
                "pubchem_cid": p.get("CID"),
                "smiles_raw": pick_smiles(p),
                "formula": p.get("MolecularFormula"),
                "mw": p.get("MolecularWeight"),
                "inchikey": p.get("InChIKey"),
            }
    except requests.RequestException:
        pass
    return None


def main() -> None:
    df = pd.read_csv(IN_CSV)
    names = df["CompoundName"].astype(str).str.strip().tolist()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"캐시 항목: {len(cache)}개")

    # 1) 캐시에서 CID 수집
    name2cid = {n: v["pubchem_cid"] for n, v in cache.items() if v and v.get("pubchem_cid")}
    no_cid = [n for n in dict.fromkeys(names) if n not in name2cid]

    print(f"CID 보유:  {len(name2cid)}개")
    print(f"CID 없음:  {len(no_cid)}개\n")

    # 2) CID로 일괄 조회
    if name2cid:
        print("CID 일괄 조회 중...")
        cid_data = fetch_by_cids(sorted(set(name2cid.values())))
        for n, cid in name2cid.items():
            if cid in cid_data:
                cache[n] = {"pubchem_cid": cid, **cid_data[cid]}
        print()

    # 3) CID 없는 것은 이름으로 재조회
    if no_cid:
        print(f"이름으로 재조회 중... ({len(no_cid)}개)")
        for i, n in enumerate(no_cid, 1):
            res = fetch_by_name(n)
            if res:
                cache[n] = res
            time.sleep(SLEEP)
            if i % 25 == 0 or i == len(no_cid):
                print(f"  [{i}/{len(no_cid)}]")
        print()

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    # 4) 병합
    for f in ["pubchem_cid", "smiles_raw", "formula", "mw", "inchikey"]:
        df[f] = [(cache.get(n) or {}).get(f) for n in names]

    df.to_csv(OUT_CSV, index=False)

    ok = df["smiles_raw"].notna().sum()
    print("=" * 55)
    print(f"SMILES 확보: {ok}/{len(df)}  ({ok/len(df):.1%})")
    print(f"저장: {OUT_CSV}")

    miss = df[df["smiles_raw"].isna()]
    if len(miss):
        cols = [c for c in ["LTKBID", "CompoundName", "label", "Comment"] if c in df.columns]
        miss[cols].to_csv(FAILED, index=False)
        print(f"\n미확보 {len(miss)}건 -> {FAILED}")
        print("예시:")
        for n in miss["CompoundName"].head(20):
            print("   -", n)

    # 참고: 이진화 대상만 따로
    if "y" in df.columns:
        b = df.dropna(subset=["y"])
        bok = b["smiles_raw"].notna().sum()
        print(f"\n[참고] Ambiguous 제외 {len(b)}개 중 SMILES 확보 {bok}개 " f"({bok/len(b):.1%})")


if __name__ == "__main__":
    main()
