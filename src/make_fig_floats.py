# -*- coding: utf-8 -*-
"""manuscript/figures.tex 를 그림 하나짜리 파일로 쪼개 floats/ 에 쓴다.

왜 — 본문은 그림을 인용한 자리에서 \\input{floats/fig_<key>} 한다. 전에는
그 파일들을 손으로 만들어 두어 figures.tex 와 두 벌이 되었고, 그림 번호를
바꿀 때 한쪽만 고쳐져 본문이 옛 파일 이름을 가리켰다 (2026-10-25).
이제 figures.tex 가 유일한 원본이고 이 스크립트가 파생본을 만든다.
표는 src/make_tables.py 가 같은 일을 한다.
"""

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "manuscript")
SRC = os.path.join(MAN, "figures.tex")
OUT = os.path.join(MAN, "floats")


def main():
    raw = io.open(SRC, encoding="utf-8").read()
    blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", raw, re.S)
    assert blocks, "figures.tex 에서 figure 환경을 찾지 못했다"
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    keys = []
    for b in blocks:
        m = re.search(r"\\label\{fig:([^}]+)\}", b)
        assert m, "label 이 없는 그림이 있다"
        key = m.group(1)
        assert key not in keys, "라벨이 겹친다: %s" % key
        keys.append(key)
        g = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", b)
        assert g, "includegraphics 가 없는 그림이 있다: %s" % key
        pdf = os.path.join(ROOT, "figures", g.group(1))
        if not os.path.exists(pdf):
            print("  주의: 그림 파일이 아직 없다 — %s" % g.group(1))
        p = os.path.join(OUT, "fig_%s.tex" % key)
        io.open(p, "w", encoding="utf-8").write(b.strip() + "\n")
    # 더는 쓰이지 않는 파생본이 남아 본문이 옛 그림을 가리키는 일을 막는다.
    for f in sorted(os.listdir(OUT)):
        if f.startswith("fig_") and f[4:-4] not in keys:
            os.remove(os.path.join(OUT, f))
            print("  삭제: floats/%s (figures.tex 에 없다)" % f)
    print("저장: %s 에 그림 %d개" % (os.path.relpath(OUT, ROOT), len(keys)))
    for i, k in enumerate(keys, 1):
        print("  Figure %d  fig:%s" % (i, k))


if __name__ == "__main__":
    main()
