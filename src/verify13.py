# -*- coding: utf-8 -*-
"""표가 저널 판면에 들어가는지 pdflatex 로 실측한다.

왜 필요한가
-----------
sn-jnl 단일단의 판면은 A4(210x297mm) 에 본문 31pc x 194.25mm 다
(sn-jnl.cls v3.1 의 geometry: paperwidth=210mm, paperheight=297mm,
text={31pc,194.25mm}). 31pc = 372pt 다. 2026-10-24 에 처음에는 155x235mm /
119mm 로 적힌 **구버전** cls 를 보고 339pt 로 잡았는데, 그 버전에는
Numbered·NameDate 옵션도 없다. 원고가 쓰는 v3.1 계열(sn-mathphys-num,
sn-vancouver-num, Numbered 가 있는 판)은 A4·31pc 다. 넘치면
Overleaf 는 Overfull hbox 경고만 내고 조용히 글자를 여백 밖으로 밀어낸다.

그래서 표마다 tabular 의 **자연 폭**을 재고, make_tables.py 의 LAYOUT 이
정한 글자 크기·방향에서 그 폭이 들어가는지 확인한다. 눈으로 PDF 를 보지
않아도 되고, 표를 하나 고쳐 넓어지면 여기서 멈춘다.

재는 방법
--------
표의 tabular 를 그대로 아주 넓은 종이의 \\sbox 에 담고 \\the\\wd 를 로그로
뽑는다. 조판기가 실제로 계산한 폭이므로 글자폭 추정이 들어가지 않는다.
p{} 열은 폭이 고정이라 자연 폭 계산에 그대로 들어간다.

실행
----
    python src/verify13.py
"""

import io
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "manuscript")
SRC = os.path.join(ROOT, "src")

MM = 2.84526  # 1mm 의 pt 값
TEXT_W = 31.0 * 12.0  # 31pc = 372pt — 세로쓰기 표가 들어갈 폭
TEXT_H = 194.25 * MM  # 553pt — 가로로 돌린 표가 들어갈 폭
SIZES = ["normalsize", "small", "footnotesize", "scriptsize"]

PRE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=400cm,paperheight=400cm,margin=1cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}
\providecommand{\botrule}{}
\newsavebox{\snbox}
\begin{document}
"""


def tables(path):
    """(라벨, 글자크기, 가로쓰기, tabular 원문) 목록."""
    if not os.path.exists(path):
        return []
    t = io.open(path, encoding="utf-8").read()
    out = []
    pat = (
        r"\\begin\{(table|sidewaystable)\}.*?"
        r"\\label\{(tab:[^}]+)\}\s*"
        r"(?:\\(normalsize|small|footnotesize|scriptsize|tiny)\s*)?"
        r"(\\begin\{tabular\}.*?\\end\{tabular\})"
    )
    for m in re.finditer(pat, t, re.S):
        out.append(
            (m.group(2), m.group(3) or "normalsize", m.group(1) == "sidewaystable", m.group(4))
        )
    return out


def measure(items):
    """라벨마다 {크기: 폭pt} 를 돌려준다."""
    doc = [PRE]
    for lab, _, _, tab in items:
        for s in SIZES:
            doc.append("\\begin{lrbox}{\\snbox}\\%s" % s)
            doc.append(tab)
            doc.append("\\end{lrbox}")
            doc.append("\\typeout{SNWIDTH %s %s \\the\\wd\\snbox}" % (lab, s))
    doc.append("\\end{document}")
    d = tempfile.mkdtemp(prefix="snwidth")
    f = os.path.join(d, "w.tex")
    io.open(f, "w", encoding="utf-8").write("\n".join(doc))
    try:
        subprocess.call(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "w.tex"],
            cwd=d,
            stdout=open(os.devnull, "w"),
            stderr=subprocess.STDOUT,
        )
    except OSError:
        print("  건너뜀 — pdflatex 가 없다.")
        return None
    log = os.path.join(d, "w.log")
    if not os.path.exists(log):
        print("  건너뜀 — pdflatex 가 로그를 남기지 않았다.")
        return None
    w = {}
    for m in re.finditer(
        r"SNWIDTH (tab:[A-Za-z0-9:_-]+) (\w+) ([0-9.]+)pt",
        io.open(log, encoding="utf-8", errors="replace").read(),
    ):
        w.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    return w


def main():
    print("표 폭 검사 — sn-jnl 단일단 판면 %.0fpt x %.0fpt" % (TEXT_W, TEXT_H))
    print("  (sn-jnl.cls v3.1: paperwidth=210mm paperheight=297mm text={31pc,194.25mm})")
    print("")
    items = []
    for f in ("tables_main.tex", "tables_supp.tex"):
        items += tables(os.path.join(MAN, f))
    if not items:
        print("건너뜀 — manuscript/ 에 표가 없다 (원고는 이 저장소에 없다).")
        return 0
    w = measure(items)
    if w is None:
        return 0

    nbad = 0
    print("%-16s %5s %5s %-13s %-9s %s" % ("표", "폭", "한계", "크기", "방향", "판정"))
    for lab, size, sideways, _ in items:
        got = w.get(lab, {}).get(size)
        if got is None:
            print(
                "%-16s     ?   —  %-13s %-9s X 폭을 재지 못했다"
                % (lab.replace("tab:", ""), size, "")
            )
            nbad += 1
            continue
        lim = TEXT_H if sideways else TEXT_W
        ok = got <= lim + 0.5
        if not ok:
            nbad += 1
        # 들어가는 가장 큰 설정을 같이 알려 준다.
        hint = ""
        if not ok:
            for s in SIZES:
                if w[lab][s] <= TEXT_W:
                    hint = "  -> %s 로 세로쓰기" % s
                    break
            else:
                for s in SIZES:
                    if w[lab][s] <= TEXT_H:
                        hint = "  -> %s 로 가로쓰기" % s
                        break
                else:
                    hint = "  -> 가로로 돌려도 안 들어간다. 열을 줄이거나 추가파일로 뺄 것"
        print(
            "%-16s %5.0f %5.0f %-13s %-9s %s%s"
            % (
                lab.replace("tab:", ""),
                got,
                lim,
                size,
                "sideways" if sideways else "upright",
                "OK" if ok else "X",
                hint,
            )
        )

    print("")
    if nbad:
        print("판면을 넘는 표 %d개. make_tables.py 의 LAYOUT 과 열 서식을 고칠 것." % nbad)
    else:
        print("표 %d개 전부 판면에 들어간다." % len(items))
    print("실패 %d건" % nbad)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
