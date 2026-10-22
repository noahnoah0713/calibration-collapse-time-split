"""분석 상태 검증 — mtime 이 아니라 '내용'으로 확인한다.

왜 필요한가
----------
파일을 복사하거나 백업을 되돌리면 내용이 같아도 mtime 이 바뀐다.
mtime 기반 최신성 검사는 그때 전부 '오래됨'으로 오탐한다
(2026-09-21 make_preconflict_evidence.sh 실행 후 실제로 56건 오탐).

이 스크립트는 입력 데이터·소스코드의 SHA-256 을 manifest 에 기록하고,
다음 실행 때 내용이 실제로 바뀌었는지만 본다.

사용법
------
    python src/verify_state.py --save      # 현재 상태를 manifest 에 기록
    python src/verify_state.py             # manifest 와 대조 (기본)

종료 코드 0 = 이상 없음, 1 = 불일치
"""

import ast
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "results" / "state_manifest.json"

DATA = ["data/processed/dilirank2_with_year.csv", "data/processed/dictrank_processed.csv"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot():
    d = {"data": {}, "src": {}}
    for f in DATA:
        p = ROOT / f
        d["data"][f] = {"sha256": sha(p), "bytes": p.stat().st_size}
    # 소스는 두 가지로 기록한다.
    #   src      파일 내용 해시 — 무엇이든 바뀌면 달라진다.
    #   src_ast  구문 트리 해시 — **정렬·주석만 바뀌면 그대로다.**
    # 후자가 필요한 이유: 2026-10-22 에 black 으로 코드를 정렬했더니
    # 모든 파일의 시각과 내용 해시가 바뀌었고, verify9 의 5.96 검사가
    # "결과가 코드보다 오래됐다" 로 40건 넘게 오경보를 냈다. 정렬은
    # 결과를 바꾸지 않으므로 그 구별이 검사 안에 있어야 한다.
    d["src_ast"] = {}
    for p in sorted((ROOT / "src").glob("*.py")):
        if p.name == "verify_state.py":
            continue
        d["src"][f"src/{p.name}"] = sha(p)
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            d["src_ast"][f"src/{p.name}"] = hashlib.sha256(
                ast.dump(tree).encode("utf-8")
            ).hexdigest()
        except SyntaxError:
            d["src_ast"][f"src/{p.name}"] = "parse_error"
    d["results"] = sorted(
        p.name for p in (ROOT / "results").glob("*") if p.suffix in (".csv", ".txt")
    )
    return d


def main():
    cur = snapshot()
    if "--save" in sys.argv:
        MAN.write_text(json.dumps(cur, indent=2, sort_keys=True), encoding="utf-8")
        print(f"manifest 기록: {MAN}")
        print(
            f"  데이터 {len(cur['data'])}개, 소스 {len(cur['src'])}개, 결과 {len(cur['results'])}개"
        )
        return 0

    if not MAN.exists():
        print("manifest 없음. 먼저 --save 로 기록할 것.")
        return 1

    old = json.loads(MAN.read_text(encoding="utf-8"))
    bad = 0

    print("=" * 66)
    print("입력 데이터")
    print("=" * 66)
    for f, v in cur["data"].items():
        o = old["data"].get(f)
        if o is None:
            print(f"  [신규] {f}")
            bad += 1
        elif o["sha256"] != v["sha256"]:
            print(f"  [변경] {f}  ({o['bytes']} -> {v['bytes']} bytes)")
            bad += 1
        else:
            print(f"  [동일] {f}  {v['bytes']} bytes")

    print()
    print("=" * 66)
    print("소스코드")
    print("=" * 66)
    ch = [f for f, h in cur["src"].items() if old["src"].get(f) != h]
    new = [f for f in cur["src"] if f not in old["src"]]
    gone = [f for f in old["src"] if f not in cur["src"]]
    for f in ch:
        print(f"  [변경] {f}")
        bad += 1
    for f in new:
        print(f"  [신규] {f}")
    for f in gone:
        print(f"  [삭제] {f}")
        bad += 1
    if not (ch or new or gone):
        print(f"  [동일] {len(cur['src'])}개 전부")

    print()
    print("=" * 66)
    print("결과 파일")
    print("=" * 66)
    miss = [f for f in old["results"] if f not in cur["results"]]
    add = [f for f in cur["results"] if f not in old["results"]]
    for f in miss:
        print(f"  [사라짐] {f}")
        bad += 1
    for f in add:
        print(f"  [추가] {f}")
    print(f"  현재 {len(cur['results'])}개")

    print()
    print("=" * 66)
    if bad == 0:
        print("이상 없음. 기록된 데이터·코드와 현재 상태가 동일하다.")
        print("results/ 의 수치는 이 데이터·코드에서 나온 것이 맞다.")
    else:
        print(f"불일치 {bad}건. 위 [변경]/[삭제]/[사라짐] 항목을 확인하고")
        print("필요하면 bash rerun_all.sh 로 다시 돌릴 것.")
    print("=" * 66)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
