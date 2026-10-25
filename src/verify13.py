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
# 판면은 추정하지 않는다. 실제 클래스 파일로 빈 문서를 컴파일해
# \the\textwidth·\the\textheight 를 찍어 얻은 값이다 (2026-10-25).
#   본문  \documentclass[pdflatex,sn-mathphys-num,referee,lineno]{sn-jnl}
#   보충  같은 클래스 + \geometry{landscape,a4paper,margin=20mm}
# 전에는 보충 판면을 "297mm-40mm" 로 어림해 484pt 로 뒀는데, geometry 가
# sn-jnl 의 top·headheight·footskip 을 그대로 두기 때문에 실제와 달랐고
# 0.2pt 차이로 넘치는 표를 통과시켰다.
TEXT_W = 372.0
TEXT_H = 552.69
SUPP_W = 714.16
SUPP_H = 483.69
# float 하나에는 캡션 위아래 간격(\abovecaptionskip 등)과 본문과의
# 간격(\textfloatsep)이 더 붙고, 범례는 manyfoot 이 따로 조판한다.
# 아래 minipage 모형은 그만큼을 세지 못하므로 높이 한계에서 미리 뺀다.
# 20pt 로 두면 LaTeX 의 "Float too large" 경고와 판정이 일치한다
# (2026-10-25 에 세 표로 맞춰 확인했다).
SAFETY = 20.0
SIZES = ["normalsize", "small", "footnotesize", "scriptsize", "tiny"]

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


def floats(path):
    """(라벨, 글자크기, 가로쓰기, 캡션, tabular, 범례) 목록.

    높이까지 재려면 캡션과 범례도 있어야 한다. 표만 재면 2026-10-24 처럼
    회전한 표의 캡션과 머리행이 판면 밖으로 잘려 나가는 것을 놓친다.
    """
    if not os.path.exists(path):
        return []
    t = io.open(path, encoding="utf-8").read()
    out = []
    pat = (
        r"\\begin\{(table|sidewaystable)\}.*?"
        r"\\caption\{(.*?)\}\\label\{(tab:[^}]+)\}\s*"
        r"(?:\\(normalsize|small|footnotesize|scriptsize|tiny)\s*)?"
        r"(\\begin\{tabular\}.*?\\end\{tabular\})\s*"
        r"(?:\\footnotetext\{(.*?)\}\s*)?"
        r"\\end\{\1\}"
    )
    for m in re.finditer(pat, t, re.S):
        out.append(
            (
                m.group(3),
                m.group(4) or "normalsize",
                m.group(1) == "sidewaystable",
                m.group(2),
                m.group(5),
                m.group(6) or "",
            )
        )
    return out


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


def measure_height(items):
    """라벨마다 (캡션+표+범례) 의 전체 높이를 pt 로 돌려준다.

    세로쓰기 표는 폭 TEXT_W 의 단에 들어가므로 높이가 TEXT_H 를 넘으면
    안 되고, 가로로 돌린 표는 폭 TEXT_H 의 단에 들어가므로 높이가 TEXT_W
    를 넘으면 안 된다. 넘치면 LaTeX 가 "Float too large" 를 내거나(세로)
    아무 말 없이 잘라 버린다(가로).
    """
    doc = [PRE]
    for lab, size, sideways, cap, tab, leg in items:
        w = TEXT_H if sideways else TEXT_W
        doc.append("\\begin{lrbox}{\\snbox}%")
        doc.append("\\begin{minipage}{%.2fpt}%%" % w)
        doc.append("\\textbf{Table~1}~%s\\par\\vskip 4pt" % cap)
        doc.append("\\%s" % size)
        doc.append(tab)
        doc.append("\\par\\vskip 4pt\\footnotesize %s" % leg)
        doc.append("\\end{minipage}%")
        doc.append("\\end{lrbox}")
        doc.append("\\dimen0=\\ht\\snbox \\advance\\dimen0 by \\dp\\snbox")
        doc.append("\\typeout{SNHEIGHT %s \\the\\dimen0}" % lab)
    doc.append("\\end{document}")
    d = tempfile.mkdtemp(prefix="snheight")
    f = os.path.join(d, "h.tex")
    io.open(f, "w", encoding="utf-8").write("\n".join(doc))
    try:
        subprocess.call(
            ["pdflatex", "-interaction=nonstopmode", "h.tex"],
            cwd=d,
            stdout=open(os.devnull, "w"),
            stderr=subprocess.STDOUT,
        )
    except OSError:
        return None
    log = os.path.join(d, "h.log")
    if not os.path.exists(log):
        return None
    h = {}
    for m in re.finditer(
        r"SNHEIGHT (tab:[A-Za-z0-9:_-]+) ([0-9.]+)pt",
        io.open(log, encoding="utf-8", errors="replace").read(),
    ):
        h[m.group(1)] = float(m.group(2))
    return h


def main():
    print("표 판면 검사 — 폭과 높이를 pdflatex 로 실측한다")
    print("  본문  세로 A4, 판면 %.0f x %.0f pt (sn-jnl text={31pc,194.25mm})" % (TEXT_W, TEXT_H))
    print("  보충  가로 A4, 판면 %.0f x %.0f pt (여백 20mm)" % (SUPP_W, SUPP_H))
    print("")
    jobs = []
    for f, lim in (("tables_main.tex", (TEXT_W, TEXT_H)), ("tables_supp.tex", (SUPP_W, SUPP_H))):
        for it in floats(os.path.join(MAN, f)):
            jobs.append((f, it, lim))
    if not jobs:
        print("건너뜀 — manuscript/ 에 표가 없다 (원고는 이 저장소에 없다).")
        return 0

    items = [it for _, it, _ in jobs]
    w = measure([(a, b, c, e) for a, b, c, _d, e, _g in items])
    h = measure_height(items)
    if w is None or h is None:
        print("건너뜀 — pdflatex 가 없다.")
        return 0

    nbad = 0
    print("%-18s %-6s %-12s %9s %9s %s" % ("표", "방향", "크기", "폭/한계", "높이/한계", "판정"))
    for f, it, (W, H) in jobs:
        lab, size, sw, _c, _t, _l = it
        lw, lh = (H, W) if sw else (W, H)
        lh -= SAFETY
        gw = w.get(lab, {}).get(size)
        gh = h.get(lab)
        if gw is None or gh is None:
            print("%-18s %s" % (lab.replace("tab:", ""), "X 재지 못했다"))
            nbad += 1
            continue
        over = []
        if gw > lw + 0.5:
            over.append("폭 %.0fpt 넘침" % (gw - lw))
        if gh > lh + 0.5:
            over.append("높이 %.0fpt 넘침" % (gh - lh))
        if over:
            nbad += 1
        print(
            "%-18s %-6s %-12s %4.0f/%4.0f %4.0f/%4.0f %s"
            % (
                lab.replace("tab:", ""),
                "가로" if sw else "세로",
                size,
                gw,
                lw,
                gh,
                lh,
                "OK" if not over else "X " + ", ".join(over),
            )
        )

    print("")
    if nbad:
        print("판면을 넘는 표 %d개. make_tables.py 의 LAYOUT·열 서식을 고치거나" % nbad)
        print("행이 너무 많으면 표를 쪼개거나 추가파일(.csv)로 뺄 것.")
    else:
        print("표 %d개 전부 판면에 들어간다 (폭·높이 모두)." % len(jobs))
    print("실패 %d건" % nbad)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
