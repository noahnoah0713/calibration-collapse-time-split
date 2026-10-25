"""원고 투고본 한 파일을 조립한다 — \\input 을 펼친다.

왜 필요한가
----------
작업용 원고는 부분 파일로 쪼개 두는 쪽이 안전하다. 표는 생성기가 만들고
(`make_tables.py`), Results·Discussion·Methods 는 각자 파일이며, 한 군데를
고칠 때 다른 데를 건드리지 않는다. 그런데 Springer Nature 템플릿은 투고
파일에 `\\input` 을 쓰지 말라고 적어 두었다(sn-article.tex 머리 주석).

그래서 작업용은 `manuscript/main.tex`(\\input 유지), 투고용은 이 스크립트가
펼친 `manuscript/submission.tex` 로 나눈다. submission.tex 를 손으로
고치지 말 것 — 고치면 다음 조립에서 덮인다.

조립 순서는 main.tex 가 정한다. 이 스크립트는 순서를 모른다.

사용법
------
    python src/make_manuscript.py
    python src/make_manuscript.py --check    # 조립만 해 보고 쓰지 않는다

출력
----
    manuscript/submission.tex
"""

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "manuscript")
SRC = os.path.join(MAN, "main.tex")
OUT = os.path.join(MAN, "submission.tex")
# 보충자료도 같이 펼친다. 저널은 "LaTeX 원고를 한 .tex 문서로 내라" 고
# 하는데 supplementary.tex 는 tables_supp.tex 를 \input 하고 있었다.
# 그대로 올리면 투고 시스템이 tables_supp.tex 를 못 찾아 멈춘다
# (2026-10-24 에 실제로 "File `tables_supp.tex\' not found" 로 멈췄다).
SRC2 = os.path.join(MAN, "supplementary.tex")
OUT2 = os.path.join(MAN, "additional_file_1.tex")

if not os.path.exists(SRC):
    # 2026-10-22: 공개 저장소에는 원고를 넣지 않는다 (재현과 무관하고
    # 게재 전이다). 원고가 없으면 조립할 것이 없으므로 실패가 아니라
    # 건너뛴다. rerun_all.sh 의 run() 은 0 이 아닌 종료코드를 받으면
    # 전체를 중단시키므로, 그대로 두면 받은 사람의 재현이 여기서 멈춘다.
    print("건너뜀 — %s 가 없다 (원고는 이 저장소에 포함하지 않는다)." % SRC)
    raise SystemExit(0)


def expand(path, seen, depth=0):
    if depth > 8:
        raise SystemExit("\\input 중첩이 너무 깊다: %s" % path)
    if not os.path.exists(path):
        raise SystemExit("없는 파일을 \\input 한다: %s" % path)
    real = os.path.realpath(path)
    if real in seen:
        raise SystemExit("\\input 순환: %s" % path)
    seen = seen | {real}
    out = []
    for ln in io.open(path, encoding="utf-8").read().split("\n"):
        m = re.match(r"\s*\\input\{([^}]+)\}\s*$", ln)
        if m and not ln.strip().startswith("%"):
            sub = m.group(1)
            if not sub.endswith(".tex"):
                sub += ".tex"
            out.append("%% ---- %s 시작 (make_manuscript.py 가 펼쳤다) ----" % sub)
            out.append(expand(os.path.join(MAN, sub), seen, depth + 1))
            out.append("%% ---- %s 끝 ----" % sub)
            continue
        out.append(ln)
    return "\n".join(out)


def main():
    txt = expand(SRC, set())
    # 펼친 결과에 \input 이 남아 있으면 조립이 덜 된 것이다.
    left = [
        ln
        for ln in txt.split("\n")
        if re.match(r"\s*\\input\{", ln) and not ln.strip().startswith("%")
    ]
    if left:
        raise SystemExit("펼치지 못한 \\input 이 남았다: %s" % left[:3])
    head = (
        "%% 투고용 단일 파일 — src/make_manuscript.py 가 "
        "manuscript/main.tex 에서 조립했다.\n"
        "%% **손으로 고치지 말 것.** 부분 파일을 고치고 다시 조립할 것.\n"
    )
    txt = head + txt
    n_sec = len(re.findall(r"^\\section\{", txt, re.M))
    n_sub = len(re.findall(r"^\\subsection\{", txt, re.M))
    n_tab = len(re.findall(r"\\begin\{table\}", txt))
    n_fig = len(re.findall(r"\\begin\{figure\}", txt))
    n_todo = len(re.findall(r"\[TODO", txt)) + txt.count("[이름]")
    print(
        "조립: 절 %d / 하위절 %d / 표 %d / 그림 %d / %d행"
        % (n_sec, n_sub, n_tab, n_fig, txt.count("\n") + 1)
    )
    print("남은 [TODO]·플레이스홀더 %d개" % n_todo)
    for pat, lab in (
        (r"\\begin\{document\}", "begin{document}"),
        (r"\\end\{document\}", "end{document}"),
        (r"\\maketitle", "maketitle"),
        (r"\\backmatter", "backmatter"),
        (r"\\bibliography\{", "bibliography"),
    ):
        c = len(re.findall(pat, txt))
        if c != 1:
            print("  [주의] %s 가 %d개다 (1개여야 한다)" % (lab, c))
    if "--check" in sys.argv:
        print("(--check: 쓰지 않았다)")
        return 0
    io.open(OUT, "w", encoding="utf-8").write(txt)
    print("저장: %s" % OUT)

    # 보충자료도 한 파일로 펼친다.
    if os.path.exists(SRC2):
        t2 = expand(SRC2, set())
        left2 = [
            ln
            for ln in t2.split("\n")
            if re.match(r"\s*\\input\{", ln) and not ln.strip().startswith("%")
        ]
        if left2:
            raise SystemExit("보충자료에 펼치지 못한 \\input 이 남았다: %s" % left2[:3])
        t2 = head + t2 if not t2.startswith("%%") else t2
        io.open(OUT2, "w", encoding="utf-8").write(t2)
        print("저장: %s" % OUT2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
