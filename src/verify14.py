# -*- coding: utf-8 -*-
"""그림이 저널 판면 폭에 맞게 그려졌는지 확인한다.

왜 필요한가
-----------
figures.tex 는 그림을 width=\\textwidth 로 넣는다. sn-jnl 단일단의
\\textwidth 는 119mm = 339pt 다 (sn-jnl.cls: text={119mm,194.25mm}).
그림 PDF 가 그보다 넓으면 LaTeX 가 줄여서 앉히고, 줄인 만큼 글자도
작아진다. 2026-10-24 에 그림이 130.7mm 로 그려져 있어서 7pt 로 지정한
축 글자가 6.4pt 로 실리고 있었다. 경고는 아무데도 찍히지 않는다.

그래서 만들어진 PDF 의 MediaBox 를 직접 읽어 축척을 계산하고, 축척과
그 축척에서의 글자 크기가 허용 범위에 있는지 본다.

허용 범위
--------
  축척 0.95 ~ 1.10   (조금 키우는 쪽은 안전하다)
  축 글자 >= 6.0pt   (지정값 7pt x 축척)
  선 굵기 >= 0.25pt  (지정값 0.5pt x 축척) — 저널 규정

실행
----
    python src/verify14.py
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 기본은 figures/ 지만 FIG_OUT 으로 바꿀 수 있다 — make_figures.py 와 같은
# 환경변수를 쓰므로 미리보기로 그린 폴더를 그대로 검사할 수 있다.
FIGDIR = os.environ.get("FIG_OUT") or os.path.join(ROOT, "figures")
MAN = os.path.join(ROOT, "manuscript")

MM = 2.84526
TEXT_W = 119.0 * MM  # 339pt
SCALE_LO, SCALE_HI = 0.95, 1.10
TICK_PT = 6.5  # make_figures.py 의 xtick/ytick labelsize
AXIS_PT = 7.0  # axes.labelsize
RULE_PT = 0.5  # axes.linewidth
MIN_FONT = 6.0
MIN_RULE = 0.25


def mediabox(path):
    with open(path, "rb") as f:
        b = f.read()
    m = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", b)
    if not m:
        return None
    v = [float(x) for x in m[0].split()]
    if len(v) != 4:
        return None
    return v[2] - v[0], v[3] - v[1]


def main():
    if not os.path.isdir(FIGDIR):
        print("건너뜀 — figures/ 가 없다.")
        return 0
    # 원고는 이 저장소에 없다 (게재 전이다). 원고가 없으면 "원고가 쓰지
    # 않는 그림" 을 물어볼 수 없으므로 그 검사만 건너뛴다 — 폭 검사는
    # figures/ 만 있으면 할 수 있고, 그쪽이 이 스크립트의 본론이다.
    used = set()
    have_man = False
    for n in ("figures.tex", "supplementary.tex"):
        f = os.path.join(MAN, n)
        if os.path.exists(f):
            have_man = True
            txt = io.open(f, encoding="utf-8").read()
            used |= set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", txt))

    print("그림 폭 검사 — sn-jnl 단일단 판면 폭 %.0fpt (119mm)" % TEXT_W)
    print("")
    print("%-34s %7s %6s %7s %7s %s" % ("그림", "폭pt", "축척", "글자pt", "선pt", "판정"))
    nbad = 0
    names = sorted(x for x in os.listdir(FIGDIR) if x.lower().endswith(".pdf"))
    if not names:
        print("건너뜀 — figures/ 에 PDF 가 없다.")
        return 0
    for n in names:
        box = mediabox(os.path.join(FIGDIR, n))
        if box is None:
            print("%-34s      ?                        X MediaBox 를 읽지 못했다" % n)
            nbad += 1
            continue
        w = box[0]
        sc = TEXT_W / w
        font = TICK_PT * sc
        rule = RULE_PT * sc
        bad = []
        if not (SCALE_LO <= sc <= SCALE_HI):
            bad.append("축척 %.3f" % sc)
        if font < MIN_FONT:
            bad.append("글자 %.1fpt" % font)
        if rule < MIN_RULE:
            bad.append("선 %.2fpt" % rule)
        if have_man and n not in used and n[:-4] not in used:
            bad.append("원고가 쓰지 않는다")
        if bad:
            nbad += 1
        print(
            "%-34s %7.1f %6.3f %7.1f %7.2f %s"
            % (n, w, sc, font, rule, "OK" if not bad else "X " + ", ".join(bad))
        )

    print("")
    if nbad:
        print("문제 %d건. src/make_figures.py 의 COL1·COL2 를 보고 다시 그릴 것." % nbad)
    else:
        print(
            "그림 %d개 전부 판면 폭에 맞다.%s"
            % (len(names), "" if have_man else " (원고가 없어 사용 여부는 보지 않았다)")
        )
    print("실패 %d건" % nbad)
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
