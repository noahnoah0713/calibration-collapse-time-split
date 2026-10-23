"""원고의 인용과 .bib 를 대조한다 — 번호가 아니라 키로.

왜 필요한가
----------
2026-10-23 에 Methods 의 인용 40곳이 **전부 다른 문헌을 가리키고 있었다.**
`[n]` 번호가 72개짜리 옛 목록 기준이었고 최종 목록은 88개라서, 중간에
끼어든 16개 때문에 어긋남이 +3 부터 +16 까지 제각각이었다. 번호를 손으로
다시 매기면 다음에 목록이 바뀔 때 또 어긋난다. 그래서 `\\cite{키}` 로
바꿨고, 이 스크립트가 그 키를 지킨다.

무엇을 보는가
------------
1. 원고가 쓰는데 .bib 에 없는 키 — LaTeX 가 [?] 로 찍는다. 실패다.
2. .bib 에 있는데 원고가 한 번도 안 쓰는 키 — 출력되지 않으므로 오류는
   아니지만, "넣으려다 만 문헌" 이거나 "쓸 자리를 아직 안 쓴 문헌" 이다.
   투고 전에 하나하나 판단해야 하므로 눈에 띄게 찍는다. 종료 코드는
   건드리지 않는다 (원고를 쓰는 내내 빨간불이면 실패를 무시하게 된다 —
   verify12 의 그림 인용과 같은 이유다).
3. 본문에 `[12]` 같은 **맨 번호 인용이 남아 있는가.** 하나라도 남으면
   그것은 옛 번호이고 틀린 문헌을 가리킨다. 실패다.
4. .bib 항목에 doi 나 note 가 하나도 없는 항목 — 학술지가 요구하는
   웹 주소·DOI 가 빠진 자리다. 경고로만 찍는다.

사용법
------
    python src/verify_refs.py
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "manuscript")
BIB = os.path.join(MAN, "sn-bibliography.bib")

# 원고 본문 파일. submission.tex 는 조립본이므로 세지 않는다 (같은 인용을
# 두 번 세게 된다). main.tex 는 초록·표제가 있으므로 본다.
DOCS = [
    "main.tex",
    "introduction.tex",
    "results_body_v2.tex",
    "discussion_body_v1.tex",
    "methods_body.tex",
    "declarations.tex",
    "figures.tex",
    "tables_main.tex",
    "tables_supp.tex",
    # Additional file 1 — 본문과 별개 문서지만 같은 .bib 를 쓴다.
    "supplementary.tex",
]

# 본문에 남아 있어도 인용이 아닌 대괄호. 늘려야 하면 여기에 적는다.
NOT_A_CITATION = re.compile(
    r"\[(?:"
    r"TODO|REPOSITORY[^\]]*|Aim[^\]]*|Sentence \d[^\]]*|Internet|"
    r"\d+(?:\.\d+)?(?:--|-)\d+(?:\.\d+)?%?|"  # [0.1-0.3) 같은 구간
    r"[^\]]*데이터 수집일[^\]]*"
    r")\]"
)


def strip_comments(t):
    """LaTeX 주석을 지운다. 주석에 적어 둔 옛 번호가 잡히면 안 된다."""
    out = []
    for line in t.split("\n"):
        i = 0
        while True:
            i = line.find("%", i)
            if i < 0:
                break
            if i == 0 or line[i - 1] != "\\":
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def main():
    if not os.path.exists(BIB):
        # 원고는 .gitignore 로 공개 저장소에서 빠져 있다 ("논문 원고는
        # 재현과 무관하다 — 게재 후 추가"). 새로 클론한 곳에는 manuscript/
        # 가 아예 없으므로, 없음을 실패로 세면 CI 가 늘 빨간불이 된다.
        # 건너뛰되 왜 건너뛰는지 찍는다.
        print("원고가 없어 건너뛴다 (manuscript/ 는 .gitignore 대상이다).")
        return 0

    bib = io.open(BIB, encoding="utf-8").read()
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    dup = sorted({k for k in keys if keys.count(k) > 1})
    have = set(keys)

    used = {}
    numeric = []
    for d in DOCS:
        p = os.path.join(MAN, d)
        if not os.path.exists(p):
            continue
        t = strip_comments(io.open(p, encoding="utf-8").read())
        for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", t):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    used.setdefault(k, []).append(d)
        # 맨 번호 인용 — [12] 또는 [12, 13] 또는 [12--14].
        # 앞 글자가 공백이나 ~ 인 것만 본다. `\author*[1]`,
        # `\includegraphics[width=...]` 처럼 명령의 선택인수로 붙은
        # 대괄호는 인용이 아니다 (main.tex 의 소속 표시 [1] 이 그렇게
        # 세 번 잡혔다).
        for m in re.finditer(r"\[\s*\d+(?:\s*(?:,|--|-)\s*\d+)*\s*\]", t):
            i = m.start()
            if i and t[i - 1] not in " \t\n~":
                continue
            if not NOT_A_CITATION.fullmatch(m.group(0).replace(" ", "")):
                numeric.append((d, m.group(0)))

    print("원고 인용 대조 (.bib <-> 원고)")
    print("")
    print("  .bib 항목 %d개 / 원고가 쓰는 키 %d개" % (len(keys), len(used)))

    fails = 0

    if dup:
        fails += len(dup)
        print("")
        print("  X .bib 에 키가 두 번 있다 — BibTeX 가 뒤쪽을 버린다")
        for k in dup:
            print("       %s" % k)

    missing = sorted(k for k in used if k not in have)
    if missing:
        fails += len(missing)
        print("")
        print("  X 원고가 쓰는데 .bib 에 없는 키 — LaTeX 가 [?] 로 찍는다")
        for k in missing:
            print("       %-32s (%s)" % (k, ", ".join(sorted(set(used[k])))))

    if numeric:
        fails += len(numeric)
        print("")
        print("  X 맨 번호 인용이 남아 있다 — 옛 목록 번호이고 틀린 문헌을 가리킨다")
        for d, s in numeric[:40]:
            print("       %-24s %s" % (d, s))
        if len(numeric) > 40:
            print("       ... 그리고 %d곳 더" % (len(numeric) - 40))

    unused = sorted(k for k in have if k not in used)
    if unused:
        print("")
        print("  ! .bib 에 있는데 원고가 안 쓰는 키 %d개 — 출력되지 않는다" % len(unused))
        for k in unused:
            print("       %s" % k)

    thin = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
        if (
            "doi" not in m.group(2)
            and "note" not in m.group(2)
            and "howpublished" not in m.group(2)
        ):
            thin.append(m.group(1))
    if thin:
        print("")
        print("  ! DOI·주소가 없는 항목 %d개 (옛 학술지·학회·단행본이면 정상)" % len(thin))
        for k in thin:
            print("       %s" % k)

    print("")
    print("실패 %d건" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
