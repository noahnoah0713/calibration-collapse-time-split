# -*- coding: utf-8 -*-
"""원고 .tex 산문의 수치를 현재 결과와 대조한다 — 표가 아닌 쪽.

왜 필요한가
----------
표는 `src/make_tables.py` 가 `results/` 에서 만든다. 그래서 분석을 다시
돌리면 표는 따라온다. 그런데 **원고 본문 산문은 손으로 쓴 것**이다. 값이
바뀌어도 산문은 따라오지 않는다. 표와 본문이 다른 숫자를 말하는 것은
심사자가 가장 쉽게 찾아내는 결함이다.

왜 토큰 대조로는 못 잡는가
------------------------
처음에는 verify9 의 5번 검사처럼 "문서의 숫자가 results/ 어딘가에 있는가"
로 짰다. 그러면 거의 전부 '설명됨' 으로 나온다 — 결과 파일 전체에서 만든
토큰이 55만개라 세 자리 소수는 우연히 다른 셀과 일치한다. 실제로 그
방식으로는 원고의 옛 수치 중 하나(0.825)만 걸렸고, 나머지 열다섯 개는
전부 통과했다. **그래서 verify10 과 같은 방향으로 짠다** — 결과 CSV 의
특정 행·열에서 기대값을 만들고, 원고가 그 자리에 다른 숫자를 쓰고 있으면
잡는다.

사용법
------
    python src/verify12.py
    python src/verify12.py --quiet     # 불일치만

이 스크립트는 "원고가 최신인가" 를 묻는다. 불일치가 있으면 **원고를 고쳐야
한다** — 결과가 아니라 원고가 틀린 것이다.
"""

import io
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MAN = os.path.join(ROOT, "manuscript")
# 검사 대상. **문서를 합쳐서 보면 안 된다** — 한 파일이 최신이면 다른
# 파일의 옛 수치가 가려진다. 실제로 조립본의 낡은 값 18개가 그렇게 가려졌다.
DOCS = [
    "submission.tex",
    "main.tex",
    "results_body_v2.tex",
    "discussion_body_v1.tex",
    "methods_body.tex",
    "introduction.tex",
]
# \input 으로 다른 파일을 끌어오는 조립본은 그 파일들까지 펼쳐서 본다.
EXPAND = True

bad, ok, miss = [], 0, []


def cv(n):
    return pd.read_csv(os.path.join(RES, n))


def _strip(path, seen=None):
    r"""표 환경과 주석을 뺀 산문. \input 은 펼친다."""
    seen = seen or set()
    if path in seen or not os.path.exists(path):
        return ""
    seen.add(path)
    keep, depth = [], 0
    for ln in io.open(path, encoding="utf-8", errors="replace").read().split("\n"):
        if ln.strip().startswith("%"):
            continue
        m = re.match(r"\s*\\input\{([^}]+)\}", ln)
        if m and EXPAND:
            sub = m.group(1)
            if not sub.endswith(".tex"):
                sub += ".tex"
            keep.append(_strip(os.path.join(MAN, sub), seen))
            continue
        if re.search(r"\\begin\{(table|tabular|longtable|sidewaystable)", ln):
            depth += 1
        if depth > 0:
            if re.search(r"\\end\{(table|tabular|longtable|sidewaystable)", ln):
                depth -= 1
            continue
        keep.append(re.sub(r"(?<!\\)%.*$", "", ln))
    return "\n".join(keep)


def load_prose():
    out = {}
    for name in DOCS:
        p = os.path.join(MAN, name)
        if os.path.exists(p):
            out[name] = _strip(p)
    return out


def _unused():
    out = {}
    for name in DOCS:
        p = os.path.join(MAN, name)
        if not os.path.exists(p):
            continue
        keep, depth = [], 0
        for ln in io.open(p, encoding="utf-8", errors="replace").read().split("\n"):
            if ln.strip().startswith("%"):
                continue
            if re.search(r"\\begin\{(table|tabular|longtable|sidewaystable)", ln):
                depth += 1
            if depth > 0:
                if re.search(r"\\end\{(table|tabular|longtable|sidewaystable)", ln):
                    depth -= 1
                continue
            keep.append(re.sub(r"(?<!\\)%.*$", "", ln))
        out[name] = "\n".join(keep)
    return out


PROSE = load_prose()


CUR = [None]  # 지금 검사 중인 문서


def claim(label, fmt, value, alt=()):
    """원고 산문에 이 값이 적혀 있는가.

    label  무엇을 재는가 (사람이 읽는 이름)
    fmt    원고가 쓰는 서식 (예 "%.3f", "%.1f%%")
    value  현재 결과에서 읽은 참값
    alt    원고에 남아 있는 **옛 값** 후보. 이것이 발견되면 불일치로 적는다.
    """
    global ok
    want = fmt % value
    t = PROSE.get(CUR[0], "")
    if want in t:
        ok += 1
        return
    found = [a for a in alt if a in t]
    if found:
        bad.append((CUR[0], label, want, ", ".join(found)))
    else:
        miss.append((CUR[0], label, want))


def checks():
    """현재 결과에서 기대값을 만든다 — 원고가 그 자리에 다른 숫자를 쓰면 잡힌다."""
    pt = cv("permutation_test.csv")
    pc = cv("prevalence_control.csv")
    mc = cv("multiple_comparison_correction.csv")
    st = cv("similarity_tests.csv")
    rec = cv("recalibration_results.csv")

    OLD_N = {"DILIrank": (["157"], ["806"]), "DICTrank": (["268"], ["953"])}
    OLD_A = {"DILIrank": "0.654", "DICTrank": "0.627"}
    OLD_M = {"DILIrank": "0.744", "DICTrank": "0.696"}
    OLD_SD = {"DILIrank": "0.040", "DICTrank": "0.028"}
    OLD_P = {"DILIrank": "0.020", "DICTrank": "0.010"}
    for _, r in pt.iterrows():
        d = r.dataset
        claim("%s 검증군 n" % d, "%d", int(r.n_test), OLD_N[d][0])
        claim("%s 분석집합 n" % d, "%d", int(r.n), OLD_N[d][1])
        claim("%s 시간분할 AUROC" % d, "%.3f", r.auroc_temporal, [OLD_A[d]])
        claim("%s 무작위 기준 AUROC" % d, "%.3f", r.null_mean, [OLD_M[d]])
        claim("%s 무작위 기준 SD" % d, "%.3f", r.null_sd, [OLD_SD[d]])
        claim("%s 순열 p" % d, "%.3f", r.p_permutation, [OLD_P[d]])

    OLD_E = {
        ("DILIrank", "T"): "0.431",
        ("DICTrank", "T"): "0.272",
        ("DILIrank", "R1"): "0.127",
        ("DICTrank", "R1"): "0.135",
        ("DILIrank", "R2"): "0.246",
        ("DICTrank", "R2"): "0.185",
        ("DILIrank", "R3"): "0.310",
        ("DICTrank", "R3"): "0.216",
    }
    OLD_R = {
        ("DILIrank", "R1"): "3.40",
        ("DICTrank", "R1"): "2.02",
        ("DILIrank", "R2"): "1.75",
        ("DICTrank", "R2"): "1.47",
        ("DILIrank", "R3"): "1.39",
        ("DICTrank", "R3"): "1.26",
    }
    for _, r in pc.iterrows():
        d = str(r.dataset).split(" ")[0]
        nm = str(r.setting).split(".")[0]
        claim("%s %s ECE" % (d, nm), "%.3f", r.ECE, [OLD_E.get((d, nm), "")])
        if r.ece_ratio == r.ece_ratio:
            claim("%s %s ECE 배율" % (d, nm), "%.2f", r.ece_ratio, [OLD_R.get((d, nm), "")])

    claim("2군 최대 q", "%.3f", mc[mc.family == "2_prevalence_control"].q_BH.max(), ["0.033"])

    _sp = st[st.test.str.contains("Spearman")]
    if len(_sp):
        claim("유사도-정답확률 Spearman", "%.3f", float(_sp.stat.iloc[0]), ["0.102"])
        claim("유사도-정답확률 p", "%.3f", float(_sp.p.iloc[0]), ["0.204"])
    _md = st[st.test.str.contains("AUROC")]
    if len(_md):
        claim("유사도 중위분할 차이", "%.3f", float(_md.stat.iloc[0]), ["0.055"])
        claim("유사도 중위분할 p", "%.3f", float(_md.p.iloc[0]), ["0.591"])

    for d in ("DILIrank", "DICTrank"):
        g = rec[rec.dataset.str.startswith(d)]
        claim(
            "%s 무보정 ECE (§10)" % d,
            "%.3f",
            g[g.method.str.startswith("A.")].ECE.iloc[0],
            ["0.425"] if d == "DILIrank" else [],
        )
        claim(
            "%s Oracle ECE (§10)" % d,
            "%.3f",
            g[g.method.str.startswith("E.")].ECE.iloc[0],
            ["0.049"] if d == "DILIrank" else [],
        )

    # 2026-10-20 추가 — 헤드라인 수치를 빠뜨리고 있었다. 보정 기여율은 이
    # 논문의 중심 숫자인데 대조 목록에 없어서, 10-04 조립본이 88.7% / 90.7%
    # (옛 값)을 싣고 있던 것을 verify12 가 잡지 못했다. 현행 본문은 86.1% /
    # 82.5% 로 맞다. 그 사실을 우연에 맡기지 않는다.
    bd = cv("brier_decomposition.csv")
    for _, r in bd.iterrows():
        d = str(r.dataset)
        claim(
            "%s 보정 기여율" % d,
            "%.1f",
            float(r.pct_from_reliability_A),
            {"DILIrank": ["88.7"], "DICTrank": ["90.7"]}.get(d, []),
        )
        claim(
            "%s 판별 기여율" % d,
            "%.1f",
            float(r.pct_from_resolution_A),
            {"DILIrank": ["11.3"], "DICTrank": ["9.3"]}.get(d, []),
        )
    r3 = cv("brier_decomposition_r3.csv")
    for _, r in r3[r3.baseline.str.startswith("R3")].iterrows():
        claim("%s R3 보정 기여율" % r.dataset, "%.1f", float(r.share2_reliability))
    # §10 표의 보정 기울기 — 2026-10-20 단정밀도 수정으로 움직인 칸이다.
    for d in ("DILIrank", "DICTrank"):
        g = rec[rec.dataset.str.startswith(d)]
        for pre, nm in (("A.", "무보정"), ("B.", "Platt"), ("C.", "Isotonic")):
            claim(
                "%s %s 보정 기울기 (§10)" % (d, nm),
                "%.4f",
                float(g[g.method.str.startswith(pre)].slope.iloc[0]),
                ["0.8969"] if (d == "DICTrank" and pre == "B.") else [],
            )


def main():
    global ok
    quiet = "--quiet" in sys.argv
    print("원고 산문 대조 (verify10 과 같은 방향 — 결과 CSV → 원고)")
    print("")
    nbad = 0
    for doc in DOCS:
        if doc not in PROSE:
            continue
        CUR[0] = doc
        ok = 0
        del bad[:], miss[:]
        checks()
        _b = len(bad)
        nbad += _b
        print("%-32s 일치 %3d / **옛 수치 %d** / 원고에 없음 %d" % (doc, ok, _b, len(miss)))
        for _d, lab, want, got in bad:
            print("     X %-26s 현재 %-10s 원고 %s" % (lab, want, got))
        if miss and not quiet:
            print(
                "       (안 적힌 값 %d개: %s%s)"
                % (len(miss), ", ".join(m[1] for m in miss[:6]), " ..." if len(miss) > 6 else "")
            )
    # ---- 그림 인용 점검 (2026-10-22) -----------------------------------
    # 왜 — 본문이 한 번도 가리키지 않는 그림은 학술지에서 결함으로 잡힌다.
    # 실제로 Fig4·Fig5·Fig6 이 라벨만 있고 \ref 가 0회였다. 그림을 넣는
    # 것과 본문이 그 그림을 쓰는 것은 다른 일이고, 지금까지 어느 검증기도
    # 보지 않았다.
    figtex = os.path.join(MAN, "figures.tex")
    nfig = 0
    if os.path.exists(figtex):
        ft = io.open(figtex, encoding="utf-8").read()
        labels = re.findall(r"\\label\{(fig:[a-zA-Z0-9_]+)\}", ft)
        body = ""
        for d in DOCS:
            q = os.path.join(MAN, d)
            if os.path.exists(q) and d != "figures.tex":
                body += io.open(q, encoding="utf-8").read()
        print("")
        print("그림 인용 — 정의된 라벨 %d개" % len(labels))
        for lb in labels:
            n = len(re.findall(r"\\(?:ref|autoref|cref)\{%s\}" % re.escape(lb), body))
            if n == 0:
                nfig += 1
                print("     X %-16s 본문에서 한 번도 인용되지 않는다" % lb)
        if nfig == 0:
            print("     전부 본문에서 인용된다")

    print("")
    print("옛 수치 합계 %d건   인용 안 되는 그림 %d개" % (nbad, nfig))
    # 그림 인용은 **원고 완성도** 문제이지 코드 결함이 아니다. 실패
    # 건수에 더했더니 원고를 쓰는 내내 redo_docs.sh 가 빨간불로 끝났고,
    # 그러면 실패를 무시하는 습관이 생긴다 (2026-10-22 수정). 눈에 띄게
    # 찍되 종료 코드는 건드리지 않는다. 투고 전 점검표에 올려 둔다.
    if nfig:
        print("  ^ 투고 전에 반드시 닫을 것 — 본문이 가리키지 않는 그림은")
        print("    편집부가 지적한다. 코드 문제가 아니라 원고 문제다.")
    if nbad:
        print("원고가 틀린 것이다 — 결과가 아니라 원고를 고쳐야 한다.")
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
