"""원고 .tex 산문의 수치를 현재 결과와 대조한다 — 표가 아닌 쪽.

왜 필요한가
----------
표는 `src/make_tables.py` 가 `results/` 에서 만든다. 그래서 분석을 다시
돌리면 표는 따라온다. 그런데 원고 본문 산문은 손으로 쓴 것이다. 값이
바뀌어도 산문은 따라오지 않는다. 표와 본문이 다른 숫자를 말하는 것은
심사자가 가장 쉽게 찾아내는 결함이다.

왜 토큰 대조로는 못 잡는가
------------------------
처음에는 verify9 의 5번 검사처럼 "문서의 숫자가 results/ 어딘가에 있는가"
로 짰다. 그러면 거의 전부 '설명됨' 으로 나온다 — 결과 파일 전체에서 만든
토큰이 55만개라 세 자리 소수는 우연히 다른 셀과 일치한다. 실제로 그
방식으로는 원고의 옛 수치 중 하나(0.825)만 걸렸고, 나머지 열다섯 개는
전부 통과했다. 그래서 verify10 과 같은 방향으로 짠다 — 결과 CSV 의
특정 행·열에서 기대값을 만들고, 원고가 그 자리에 다른 숫자를 쓰고 있으면
잡는다.

사용법
------
    python src/verify12.py
    python src/verify12.py --quiet     # 불일치만

이 스크립트는 "원고가 최신인가" 를 묻는다. 불일치가 있으면 원고를 고쳐야
한다 — 결과가 아니라 원고가 틀린 것이다.
"""

import io
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MAN = os.path.join(ROOT, "manuscript")
# 검사 대상. 문서를 합쳐서 보면 안 된다 — 한 파일이 최신이면 다른
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


PROSE = load_prose()


CUR = [None]  # 지금 검사 중인 문서


def written(num, t):
    """`num` 이 그 수로서 산문에 적혀 있는가 — 부분문자열로 묻지 않는다.

    왜 부분문자열이 안 되는가. 2026-10-23 에 양방향으로 틀렸다.

    1) 거짓 경보. 옛 q 값 후보 "0.033" 이 새로 쓴 재학습 문단의 신뢰구간
       하한 "$-0.033$" 안에 들어 있어서, 부호도 가리키는 양도 다른 수가
       '원고에 남은 옛 값' 으로 잡혔다.
    2) 조용한 통과. 반대로 참값 "0.005" 는 원고가 "$p = 0.0050$" 으로
       적어도 부분문자열로 걸리므로 '적혀 있다' 로 세어졌다. 자리수만
       다른 같은 수라서 이번에는 맞았지만, 같은 느슨함이면 "0.0051" 도
       통과한다.

    그래서 숫자를 글자가 아니라 수로 비교한다. 산문의 숫자를 모두 뽑아
    기대값의 소수 자리수로 반올림해 같은지 본다. 앞에 부호나 숫자가
    붙은 토큰은 다른 수이므로 건너뛴다.
    """
    core, suf = num, ""
    if core.endswith("\\%"):
        core, suf = core[:-2], "\\%"
    try:
        target = float(core)
    except ValueError:
        return core in t
    dec = len(core.split(".")[1]) if "." in core else 0
    # 음수를 찾을 때는 앞의 빼기표를 **요구**한다. 양수를 찾을 때는
    # 빼기표가 붙은 토큰을 건너뛴다 (2026-10-26). 전에는 음수 질의가
    # 어떤 경우에도 맞지 않아, 본문이 "$-0.458$" 로 적은 값을 대조할
    # 방법이 없었다.
    neg = target < 0
    target = abs(target)
    for m in re.finditer(r"(\d+(?:\.\d+)?)" + re.escape(suf), t):
        k = m.start(1)
        prev = t[k - 1] if k else ""
        if k and (prev == "." or prev.isdigit()):
            continue
        if neg != (prev in "-\u2212"):
            continue
        tok = m.group(1)
        frac = tok.split(".")[1] if "." in tok else ""
        if len(frac) < dec:
            # 자리수가 모자라면 그 수의 표기가 아니다. "0.31" 은 "0.310"
            # 이 가리키는 양이 아니고 다른 양을 두 자리로 적은 것이다
            # (실제로 고찰의 "ECE 배율이 최대 0.31" 이 R3 ECE 0.310 의
            # 옛 값으로 잡혔다).
            continue
        if dec == 0 and frac:
            continue
        if round(float(tok), dec) == round(target, dec):
            return True
    return False


def claim(label, fmt, value, alt=()):
    """원고 산문에 이 값이 적혀 있는가.

    label  무엇을 재는가 (사람이 읽는 이름)
    fmt    원고가 쓰는 서식 (예 "%.3f", "%.1f%%")
    value  현재 결과에서 읽은 참값
    alt    원고에 남아 있는 옛 값 후보. 이것이 발견되면 불일치로 적는다.
    """
    global ok
    want = fmt % value
    t = PROSE.get(CUR[0], "")
    if written(want, t):
        ok += 1
        return
    found = [a for a in alt if written(a, t)]
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
    # 2026-10-26 추가 — 초록의 머리 수치가 대조 목록에 없었다. 실제로
    # "재학습+재보정" 의 짝지은 개선이 0.1654552 인데 원고 네 곳이 0.166 을
    # 싣고 있었다 (보충표 각주의 네 자리 값 -0.1655 를 한 번 더 반올림한
    # 이중 반올림이다). 아래 세 묶음이 초록이 내세우는 수치 전부다.
    for _, r in pt.iterrows():
        claim("%s AUROC 하락폭" % r.dataset, "%.3f", float(r.delta_auroc))

    pre = cv("prequential_retraining_delta.csv")
    for _, r in pre[pre.arm.str.startswith("P2")].iterrows():
        claim("%s 재학습+재보정 ECE 개선" % r.dataset, "%.3f", abs(float(r.delta)), ["0.166"])

    cf = cv("conformal_prediction.csv")
    for _, r in cf[cf.alpha == 0.1].iterrows():
        claim(
            "%s 시간분할 적합예측 포함률" % r.dataset,
            "%.1f\\%%",
            100.0 * float(r.marginal_coverage_temporal),
        )

    # §22 결정곡선 — 2026-10-26 에 본문으로 들어온 값이다.
    dc = cv("decision_curve.csv")
    _dili = dc[(dc.dataset == "DILIrank")]
    for armpre, nm in (("A.", "무보정"), ("C.", "사전확률 보정")):
        g = _dili[_dili.arm.str.startswith(armpre)].set_index("p_t")
        for thr in (0.4, 0.5, 0.6):
            claim(
                "DILIrank 순편익 %s p_t=%.1f" % (nm, thr),
                "%.3f",
                float(g.loc[thr, "net_benefit"]),
            )

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
    if not os.path.isdir(MAN) or not any(os.path.exists(os.path.join(MAN, d)) for d in DOCS):
        # 원고가 없으면 대조할 산문이 없다. 건너뛴다 (2026-10-22).
        print("건너뜀 — %s 에 원고가 없다 (이 저장소에 포함하지 않는다)." % MAN)
        return
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
    # ---- 그림 범례가 조판되는가 (2026-10-26) ----------------------------
    # sn-jnl 에서 figure 환경 안의 \footnotetext 는 어디에도 조판되지 않는다.
    # 표 각주는 나오므로 눈치채기 어렵다. 실제로 그림 10개의 범례가 전부
    # PDF 에서 빠져 있었다. 그림 범례는 캡션 안에 있어야 한다.
    nleg = 0
    print("")
    print("그림 범례 검사")
    for fn in ("figures.tex", "supplementary.tex", "additional_file_1.tex"):
        q = os.path.join(MAN, fn)
        if not os.path.exists(q):
            continue
        raw = io.open(q, encoding="utf-8", errors="replace").read()
        for blk in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", raw, re.S):
            if "\\footnotetext" in blk:
                lab = re.search(r"\\label\{([^}]+)\}", blk)
                nleg += 1
                print(
                    "     X %s 의 %s 범례가 figure 안의 footnotetext 다 — 조판되지 않는다"
                    % (fn, lab.group(1) if lab else "?")
                )
    if not nleg:
        print("     figure 환경 안에 조판되지 않는 footnotetext 가 없다")
    nbad += nleg

    # ---- 되살아나면 안 되는 옛 수치 (2026-10-26) ------------------------
    # claim() 의 alt 는 "0.166" 처럼 양수로 적힌 경우만 잡는다. 원고가
    # "$-0.166$" 로 적으면 written() 이 앞의 빼기표를 보고 건너뛰어 놓친다.
    # 부호와 무관하게 막아야 하는 값은 여기에 적는다.
    STALE = [
        (r"(?<![\d.])0\.166(?![\d])", "재학습+재보정 ECE 개선 — 현재 값은 0.165 다"),
    ]
    nstale = 0
    print("")
    print("되살아나면 안 되는 옛 수치")
    for doc in DOCS:
        t = PROSE.get(doc, "")
        for pat, why in STALE:
            hits = len(re.findall(pat, t))
            if hits:
                nstale += hits
                print("     X %-24s %d곳: %s" % (doc, hits, why))
    if not nstale:
        print("     금지 목록 %d개 — 어느 문서에도 없다" % len(STALE))
    nbad += nstale

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

    # ---------------------------------------------------------------- 7
    # 표·그림이 번호 순서대로 인용되는가, 전부 인용되는가.
    # 왜 — 학술지 규정이다. 표는 "numbered and cited in the text in
    # sequence", 그림은 "numbered in the order they are first mentioned".
    # 2026-10-24 확인: 본문 표 20개 중 11개가 미인용이었고, 인용되는 9개도
    # 번호 순서와 등장 순서가 달랐다 (Table 10 이 Table 7 보다 먼저).
    norder = 0
    prose = ""
    for d in DOCS:
        q = os.path.join(MAN, d)
        if os.path.exists(q) and d not in ("submission.tex", "main.tex"):
            prose += _strip(q)
    print("")
    print("표·그림 인용 순서")
    for kind, src, pre in (("표", "tables_main.tex", "tab"), ("그림", "figures.tex", "fig")):
        q = os.path.join(MAN, src)
        if not os.path.exists(q):
            continue
        raw = io.open(q, encoding="utf-8", errors="replace").read()
        nocom = "\n".join(re.sub(r"(?<!\\\\)%.*$", "", ln) for ln in raw.split("\n"))
        labs = re.findall(r"\\label\{(%s:[^}]+)\}" % pre, nocom)
        seq = []
        for m in re.finditer(r"\\(?:ref|autoref|cref)\{(%s:[^}]+)\}" % pre, prose):
            if m.group(1) in labs and m.group(1) not in seq:
                seq.append(m.group(1))
        missing = [x for x in labs if x not in seq]
        nums = [labs.index(x) + 1 for x in seq]
        if missing:
            norder += len(missing)
            print(
                "     X %s %d개가 본문에 인용되지 않는다: %s"
                % (kind, len(missing), ", ".join(missing))
            )
        if nums != sorted(nums):
            norder += 1
            print("     X %s 번호 순서와 등장 순서가 다르다: %s" % (kind, nums))
        if not missing and nums == sorted(nums):
            print("     %s %d개 — 전부 인용되고 번호 순서대로다" % (kind, len(labs)))
    nbad += norder

    # ---------------------------------------------------------------- 7b
    # 보충 표는 다른 문서에 있어 \\ref 가 안 되고, 본문이 "Additional file 1:
    # Table S<n>" 이라는 글자로 가리킨다. 그래서 tables_supp.tex 의 표 순서가
    # 곧 S 번호다. 아무도 이 대응을 검사하지 않으면 표를 하나 끼워 넣는
    # 순간 본문의 S 번호가 전부 한 칸씩 어긋난다. 2026-10-24 에 실제로
    # 인용 순서가 S11, S2, S3, ... 로 흩어져 있었다.
    nsupp = 0
    print("")
    print("보충 표 번호 검사")
    q = os.path.join(MAN, "tables_supp.tex")
    if os.path.exists(q):
        raw = io.open(q, encoding="utf-8", errors="replace").read()
        nocom = "\n".join(re.sub(r"(?<!\\\\)%.*$", "", ln) for ln in raw.split("\n"))
        slabs = re.findall(r"\\label\{(tab:[^}]+)\}", nocom)
        seq = []
        # "Table S7" 과 "Tables S13 and S15" 둘 다 잡는다.
        for m in re.finditer(r"Tables?[~ ]S\d+(?:\s*(?:,|and|to|--)\s*(?:Table[~ ])?S\d+)*", prose):
            for g in re.findall(r"S(\d+)", m.group(0)):
                k = int(g)
                if k not in seq:
                    seq.append(k)
        oob = [k for k in seq if k < 1 or k > len(slabs)]
        uncited = [i + 1 for i in range(len(slabs)) if i + 1 not in seq]
        if oob:
            nsupp += len(oob)
            print("     X 없는 보충 표를 가리킨다: %s (보충 표는 %d개)" % (oob, len(slabs)))
        if uncited:
            nsupp += len(uncited)
            print(
                "     X 보충 표 %d개가 본문에 언급되지 않는다: %s"
                % (len(uncited), ", ".join("S%d (%s)" % (i, slabs[i - 1]) for i in uncited))
            )
        if seq != sorted(seq):
            nsupp += 1
            print("     X 보충 표 번호 순서와 등장 순서가 다르다: %s" % seq)
        if not oob and not uncited and seq == sorted(seq):
            print("     보충 표 %d개 — 전부 언급되고 번호 순서대로다" % len(slabs))
    nbad += nsupp

    # ---------------------------------------------------------------- 7b-2
    # 보충 **그림**도 표와 같은 규정을 따른다. 본문이 "Additional file 1:
    # Figure S<n>" 이라는 글자로 가리키므로 supplementary.tex 의 그림 순서가
    # 곧 S 번호다. 2026-10-26 에 보충 그림이 1개에서 3개로 늘었다.
    nsfig = 0
    print("")
    print("보충 그림 번호 검사")
    q = os.path.join(MAN, "supplementary.tex")
    if os.path.exists(q):
        raw = io.open(q, encoding="utf-8", errors="replace").read()
        nocom = "\n".join(re.sub(r"(?<!\\\\)%.*$", "", ln) for ln in raw.split("\n"))
        blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", nocom, re.S)
        flabs = []
        for b in blocks:
            mm = re.search(r"\\label\{(fig:[^}]+)\}", b)
            if mm:
                flabs.append(mm.group(1))
        seqf = []
        for mm in re.finditer(
            r"Figures?[~ ]S\d+(?:\s*(?:,|and|to|--)\s*(?:Figure[~ ])?S\d+)*", prose
        ):
            for g in re.findall(r"S(\d+)", mm.group(0)):
                k = int(g)
                if k not in seqf:
                    seqf.append(k)
        oobf = [k for k in seqf if k < 1 or k > len(flabs)]
        unc = [i + 1 for i in range(len(flabs)) if i + 1 not in seqf]
        if oobf:
            nsfig += len(oobf)
            print("     X 없는 보충 그림을 가리킨다: %s (보충 그림은 %d개)" % (oobf, len(flabs)))
        if unc:
            nsfig += len(unc)
            print(
                "     X 보충 그림 %d개가 본문에 언급되지 않는다: %s"
                % (len(unc), ", ".join("S%d (%s)" % (i, flabs[i - 1]) for i in unc))
            )
        if seqf != sorted(seqf):
            nsfig += 1
            print("     X 보충 그림 번호 순서와 등장 순서가 다르다: %s" % seqf)
        if not oobf and not unc and seqf == sorted(seqf):
            print("     보충 그림 %d개 — 전부 언급되고 번호 순서대로다" % len(flabs))
    nbad += nsfig

    # ---------------------------------------------------------------- 7c
    # 문서를 넘는 \\ref 가 없는가.
    # 왜 — 본문과 보충자료는 별도 문서다. 보충 표의 각주가 본문 표를
    # \\ref 로 가리키면 컴파일 결과에 ?? 가 찍힌다. 2026-10-24 에 표
    # 12개를 보충자료로 옮기면서 11곳이 그렇게 됐다.
    nx = 0
    print("")
    print("문서를 넘는 참조 검사")
    labs = {}
    for d in ("tables_main.tex", "tables_supp.tex", "figures.tex"):
        q = os.path.join(MAN, d)
        if os.path.exists(q):
            labs[d] = set(
                re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", io.open(q, encoding="utf-8").read())
            )
    own = {
        "tables_main.tex": labs.get("tables_main.tex", set()) | labs.get("figures.tex", set()),
        "tables_supp.tex": labs.get("tables_supp.tex", set()),
    }
    for d, mine in own.items():
        q = os.path.join(MAN, d)
        if not os.path.exists(q):
            continue
        txt = io.open(q, encoding="utf-8").read()
        for m in re.finditer(r"\\ref\{((?:tab|fig):[^}]+)\}", txt):
            if m.group(1) not in mine:
                nx += 1
                print("     X %s 가 다른 문서의 %s 를 \\ref 한다" % (d, m.group(1)))
    if not nx:
        print("     표 각주의 모든 \\ref 가 같은 문서 안을 가리킨다")
    nbad += nx

    # ---------------------------------------------------------------- 8
    # 표의 열 수가 서식과 맞는가.
    # 왜 — tab:mondrian 의 서식이 "llllccccc"[:10] 로 적혀 있었다. 그
    # 문자열은 9자라 슬라이스가 9열을 돌려주는데 행은 10열이었고, LaTeX
    # 가 "Extra alignment tab" 으로 멈췄다 (2026-10-23 확인). 서식과 행을
    # 같이 세면 눈으로 볼 필요가 없다.
    ncol = 0
    print("")
    print("표 열 수 검사")
    for d in ("tables_main.tex", "tables_supp.tex"):
        q = os.path.join(MAN, d)
        if not os.path.exists(q):
            continue
        t = io.open(q, encoding="utf-8").read()
        for m in re.finditer(
            r"\\label\{(tab:[^}]+)\}.*?\\begin\{tabular\}(.*?)\\end\{tabular\}",
            t,
            re.S,
        ):
            lab, rest = m.group(1), m.group(2)
            # tabular 의 열 서식은 중괄호가 중첩된다 (@{}llll@{}). 깊이를
            # 세어 꺼내지 않으면 @{ 에서 끊긴다.
            k, depth, end = rest.index("{"), 0, -1
            for j in range(k, len(rest)):
                if rest[j] == "{":
                    depth += 1
                elif rest[j] == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            spec, body = rest[k + 1 : end], rest[end + 1 :]
            # 서식에서 열을 센다. @{..} >{..} <{..} 는 열이 아니고,
            # p{..} m{..} b{..} 와 P{..} 같은 newcolumntype 은 한 열이다.
            s = re.sub(r"[@><]\{[^}]*\}", "", spec)
            s = re.sub(r"[A-Za-z]\{[^}]*\}", "X", s)
            want = len(re.sub(r"[|\s]", "", s))
            for line in body.split("\n"):
                line = line.strip()
                if not line.endswith("\\\\"):
                    continue
                # \multicolumn{n} 는 열 n 개를 차지한다.
                spans = [int(x) for x in re.findall(r"\\multicolumn\{(\d+)\}", line)]
                got = line.count("&") + 1 + sum(n - 1 for n in spans)
                if got != want:
                    ncol += 1
                    print("     X %-14s 서식 %d열 / 행 %d열  %s" % (lab, want, got, line[:46]))
                    break
    if ncol == 0:
        print("     모든 표가 서식과 같은 열 수다")
    nbad += ncol

    # ---------------------------------------------------------------- 9
    # 본문에 한국어가 남아 있는가.
    # 왜 — 표는 src/make_tables.py 가 결과 CSV 에서 만들고, 그 CSV 의
    # 라벨은 한국어다. 번역 사전에 빠진 라벨이 하나 생기면 영문 투고본에
    # 한국어가 그대로 찍힌다. 실제로 보충표에 10줄이 그렇게 들어가 있었다
    # (2026-10-23). 주석의 한국어는 세지 않는다 — 작업 메모는 한국어로
    # 쓰는 것이 이 저장소의 방식이다.
    nko = 0
    print("")
    print("본문 한국어 검사")
    for d in DOCS + ["tables_main.tex", "tables_supp.tex", "figures.tex", "supplementary.tex"]:
        q = os.path.join(MAN, d)
        if not os.path.exists(q):
            continue
        lines = io.open(q, encoding="utf-8").read().split("\n")
        for i, line in enumerate(lines, 1):
            bare = re.sub(r"(?<!\\)%.*$", "", line)
            if re.search(r"[\uac00-\ud7a3]", bare):
                nko += 1
                print("     X %s:%d  %s" % (d, i, bare.strip()[:70]))
    if nko == 0:
        print("     주석 밖에 한국어 없음")
    nbad += nko

    print("")
    print("옛 수치 합계 %d건   인용 안 되는 그림 %d개" % (nbad, nfig))
    # 그림 인용은 원고 완성도 문제이지 코드 결함이 아니다. 실패
    # 건수에 더했더니 원고를 쓰는 내내 문서 재생성이 빨간불로 끝났고,
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
