"""sn-bibliography.bib 에서 사람이 읽는 참고문헌 목록을 만든다.

왜 필요한가
----------
참고문헌이 두 곳에 있으면 반드시 어긋난다. 실제로 2026-10-23 에
`refs_final_1to88.txt`(평문 목록)와 원고의 `[n]` 번호가 어긋나 있었고,
Methods 의 인용 40곳이 전부 다른 문헌을 가리켰다. 그래서 이제 **진짜**
목록은 `manuscript/sn-bibliography.bib` 하나다. 사람이 훑어볼 목록이
필요할 때는 손으로 쓰지 않고 이 스크립트로 만든다.

`.bib` 의 `%%--- N. 제목 ---%%` 주석을 장 구분으로 읽어 그 순서대로 쓴다.
아래 번호는 목록 순번일 뿐이고, 논문에 찍히는 번호는 LaTeX 이 본문에
처음 나오는 순서로 매긴다.

사용법
------
    python src/make_refs_list.py
"""

import io
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "manuscript")
BIB = os.path.join(MAN, "sn-bibliography.bib")
OUT = os.path.join(MAN, "참고문헌_목록.md")

# {\'e} 처럼 적힌 악센트를 유니코드로 되돌린다. BibTeX 은 이 형태를
# 요구하지만 사람이 읽는 목록에서는 é 로 보이는 쪽이 검수하기 쉽다.
COMBINING = {"'": "́", "`": "̀", '"': "̈", "~": "̃", "^": "̂"}


def raw(body, name):
    """필드 값을 중괄호 깊이를 세어 꺼낸다 (값 안에 {} 가 들어 있다)."""
    m = re.search(r"\n\s*%s\s*=\s*\{" % name, body)
    if not m:
        return ""
    i = m.end() - 1
    depth, j = 0, i
    while j < len(body):
        if body[j] == "{":
            depth += 1
        elif body[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return re.sub(r"\s+", " ", body[i + 1 : j]).strip()


def unaccent(v):
    v = re.sub(r"\{\\v ?(\w)\}", lambda m: m.group(1) + "̌", v)
    v = re.sub(r"\{\\\. ?(\w)\}", lambda m: m.group(1) + "̇", v)
    for mark, comb in COMBINING.items():
        v = re.sub(
            r"\{\\%s ?(\w)\}" % re.escape(mark),
            lambda m, c=comb: m.group(1) + c,
            v,
        )
    return unicodedata.normalize("NFC", v)


def plain(v):
    v = unaccent(v).replace("\\&", "&").replace("\\url", "")
    return v.replace("{", "").replace("}", "").replace("--", "–").strip()


def authors(v):
    """단체 저자는 {..} 로 통째로 묶여 있으므로 ' and ' 로 쪼개지 않는다."""
    v = unaccent(v)
    if v.startswith("{") and v.endswith("}") and v.count("{") == 1:
        return v[1:-1]
    out = []
    for n in v.split(" and "):
        n = n.strip()
        out.append("et al." if n == "others" else plain(n))
    return ", ".join(out)


def entry(n, key, body):
    seg = []
    for value in (authors(raw(body, "author")), plain(raw(body, "title"))):
        if value:
            seg.append(value.rstrip("."))
    venue = plain(raw(body, "journal") or raw(body, "booktitle") or raw(body, "howpublished"))
    if venue:
        seg.append("*%s*" % venue.rstrip("."))
    publisher = plain(raw(body, "publisher"))
    if publisher:
        seg.append(publisher)

    year = raw(body, "year")
    vol, num = plain(raw(body, "volume")), plain(raw(body, "number"))
    pages = plain(raw(body, "pages"))
    loc = year
    if vol:
        loc += ";%s" % vol + ("(%s)" % num if num else "")
        if pages:
            loc += ":%s" % pages
    elif pages:
        loc += ":%s" % pages
    seg.append(loc)

    line = "%d. **`%s`** — %s." % (n, key, ". ".join(seg))
    doi = raw(body, "doi")
    if doi:
        line += " doi:%s" % doi
    note = plain(raw(body, "note"))
    if note:
        line += "  \n   *%s*" % note
    return line


def main():
    if not os.path.exists(BIB):
        print("원고가 없어 건너뛴다 (manuscript/ 는 .gitignore 대상이다).")
        return 0

    bib = io.open(BIB, encoding="utf-8").read()
    parts = re.split(r"%%-{10,}%%\n%% (.+?)\s*%%\n%%-{10,}%%", bib)

    out = [
        "# 참고문헌 목록 — sn-bibliography.bib",
        "",
        "`manuscript/sn-bibliography.bib` 에서 **자동으로 만든 파일**이다.",
        "손으로 고치지 말 것 — `.bib` 를 고치고 `python src/make_refs_list.py` 를 다시 돌린다.",
        "",
        "아래 번호는 이 목록의 순번일 뿐이다. 논문에 찍히는 번호는 LaTeX 이",
        "**본문에 처음 나오는 순서로** 매긴다. 원고가 가리키는 것은 백틱 안의 키다.",
        "",
    ]
    total = 0
    for i in range(1, len(parts), 2):
        name, body = parts[i].strip(), parts[i + 1]
        ents = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", body, re.S)
        if not ents:
            continue
        out += ["## %s" % name, ""]
        for key, b in ents:
            total += 1
            out.append(entry(total, key, b))
        out.append("")
    out += ["---", "", "**총 %d개.**" % total]

    io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    declared = len(re.findall(r"^@", bib, re.M))
    print("항목 %d개 -> %s" % (total, OUT))
    if total != declared:
        print(
            "  X .bib 의 @항목은 %d개인데 %d개만 썼다 — 장 주석 밖의 항목이 있다"
            % (declared, total)
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
