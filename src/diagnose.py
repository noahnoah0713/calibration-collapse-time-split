"""PubChem API 진단 - 어떤 속성명이 통하는지 확인"""

import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TEST = "acetaminophen"

candidates = [
    "CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,InChIKey",
    "SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,InChIKey",
    "SMILES,MolecularFormula,MolecularWeight,InChIKey",
    "IsomericSMILES",
    "CanonicalSMILES",
    "SMILES",
    "MolecularFormula",
]

print(f"테스트 화합물: {TEST}\n" + "=" * 60)

for props in candidates:
    url = f"{BASE}/compound/name/{TEST}/property/{props}/JSON"
    try:
        r = requests.get(url, timeout=20)
        status = r.status_code
        body = r.text[:220].replace("\n", " ")
        mark = "OK  " if status == 200 else "FAIL"
        print(f"[{mark}] {status} | {props}")
        if status != 200:
            print(f"        -> {body}")
        else:
            print(f"        -> {body}")
    except Exception as e:
        print(f"[ERR ] {props}\n        -> {e}")
    print()

# 네트워크 자체 확인
print("=" * 60)
try:
    r = requests.get(f"{BASE}/compound/name/{TEST}/cids/JSON", timeout=20)
    print(f"CID 조회 (가장 단순한 요청): {r.status_code} | {r.text[:120]}")
except Exception as e:
    print(f"네트워크 오류: {e}")
