"""PubChem ATC 응답 구조 진단 - 왜 전부 'A'로 나왔는지 확인"""

import json
import re
import requests

# 명확한 정답이 있는 화합물들
TESTS = [
    ("Abemaciclib", "L01EF"),  # 항종양제
    ("Abiraterone acetate", "L02BX"),
    ("Atorvastatin calcium", "C10AA"),  # 심혈관
    ("Acetaminophen", "N02BE"),  # 신경계
]

for name, expect in TESTS:
    print("=" * 70)
    print(f"{name}  (기대: {expect}...)")
    print("=" * 70)

    u = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/cids/JSON"
    )
    r = requests.get(u, timeout=20)
    if r.status_code != 200:
        print(f"  CID 조회 실패: {r.status_code}\n")
        continue
    cid = r.json()["IdentifierList"]["CID"][0]
    print(f"  CID = {cid}")

    u2 = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
        f"{cid}/JSON?heading=ATC+Code"
    )
    r2 = requests.get(u2, timeout=25)
    print(f"  PUG-View 상태: {r2.status_code}, 응답 길이 {len(r2.text):,}자")

    if r2.status_code != 200:
        print(f"  본문 앞부분: {r2.text[:200]}\n")
        continue

    # (1) 기존 방식: 원문 전체 정규식
    codes = sorted(set(re.findall(r"\b([A-V]\d{2}[A-Z]{2}\d{2})\b", r2.text)))
    print(f"\n  [기존 방식] 정규식 결과 {len(codes)}개")
    print(f"    앞 10개: {codes[:10]}")
    print(f"    -> 첫 글자로 잡히는 값: '{codes[0][0] if codes else 'None'}'")

    # (2) JSON 구조를 따라 실제 값만 추출
    def walk(node, out):
        if isinstance(node, dict):
            if "String" in node and isinstance(node["String"], str):
                out.append(node["String"])
            for v in node.values():
                walk(v, out)
        elif isinstance(node, list):
            for v in node:
                walk(v, out)

    strings = []
    walk(r2.json(), strings)
    real = []
    for s in strings:
        real += re.findall(r"\b([A-V]\d{2}[A-Z]{2}\d{2})\b", s)
    real = sorted(set(real))
    print(f"\n  [구조 파싱] 값 문자열에서 추출 {len(real)}개")
    print(f"    {real[:10]}")

    # 실제 String 값 몇 개 보기
    hits = [s for s in strings if re.search(r"\b[A-V]\d{2}[A-Z]{2}\d{2}\b", s)]
    print(f"\n  ATC 포함 문자열 {len(hits)}개, 앞 5개:")
    for h in hits[:5]:
        print(f"    - {h[:110]}")
    print()
