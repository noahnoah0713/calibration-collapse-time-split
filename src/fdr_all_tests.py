"""§15 의 가족에서 빠져 있던 검정을 전부 편입한다 — 한계 19번.

왜 필요한가
----------
한계 19번은 이렇게 적혀 있다.

    "§12.2·§13.5 의 p 값은 §15 의 어느 가족에도 들어가 있지 않다
     (§15.1-4). 그래서 두 절의 근거는 효과크기(ECE 배율)로 쓰고 p 는
     보조로만 썼다."

"가족에 넣지 않았으니 보조로만 쓴다" 는 회피다. 심사자는 "왜 그 p 들만
보정 밖에 두었나" 를 묻는다. 그래서 빼지 않고 넣고, 넣었을 때 무엇이
살아남는지 센다. 유의성을 지키는 것이 목적이 아니라 결론이 바뀌는지 보는
것이 목적이다.

무엇을 편입하는가
---------------
    §12.2  역방향 시간분할 순열검정   (reverse_split.csv)
    §13.5  비시간 축 분할 순열검정    (shift_matched_split.csv)
    §11.1  Mondrian conformal 클래스별 커버리지 단측 z
           (mondrian_conformal.csv — z 를 정규근사 p 로 바꾼다)

음성 대조군 행(§13.5 의 R. 무작위, §12.2 의 정방향 참고)도 빼지 않고
넣는다. 가족이 커지면 보정이 더 보수적이 되므로 넣는 쪽이 자기에게 불리한
선택이다.

가족 구성 셋
----------
    기존   §15 의 1·2·3군 (비교 기준, 그대로 둔다)
    F4     빠져 있던 검정만 모은 새 가족
    F0     기존 세 가족 + F4 를 한 가족으로 합친 것 (가장 보수적)

모델을 다시 적합하지 않는다 — 저장된 p 와 z 만 쓴다.

사용법
------
    python src/fdr_all_tests.py

출력
----
    results/fdr_all_tests.csv
    results/fdr_all_tests_summary.txt
"""

import math
import os
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ALPHA = 0.05
# 편입하는 절 — 출력 순서를 고정한다.
SECS = ("§11.1", "§12.2", "§13.5", "§18", "§21", "§24")


class _Log(object):
    def __init__(self):
        self.lines = []

    def __call__(self, m=""):
        print(m, flush=True)
        self.lines.append(str(m))

    def save(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


log = _Log()


def bh(ps):
    """Benjamini-Hochberg q — verify9.py 의 bh() 와 같은 구현."""
    n = len(ps)
    idx = sorted(range(n), key=lambda i: ps[i])
    q = [0.0] * n
    prev = 1.0
    for r in range(n, 0, -1):
        i = idx[r - 1]
        v = min(prev, ps[i] * n / r)
        q[i] = v
        prev = v
    return q


def norm_sf(z):
    """P(Z > z) — scipy 없이 erfc 로 계산한다."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def cv(n):
    return pd.read_csv(os.path.join(RES, n))


def collect():
    """편입할 검정 목록. (절, 데이터셋, 검정이름, p, 비고)"""
    rows = []
    rs = cv("reverse_split.csv")
    for _, r in rs.iterrows():
        if r.p_phipson_smyth != r.p_phipson_smyth:
            continue
        rows.append(
            (
                "§12.2",
                str(r.dataset),
                "%s %s" % (str(r.direction), str(r.setting)),
                float(r.p_phipson_smyth),
                "순열 %d회" % int(r.n_rep_valid),
            )
        )
    sm = cv("shift_matched_split.csv")
    for _, r in sm.iterrows():
        if r.p_phipson_smyth != r.p_phipson_smyth:
            continue
        rows.append(
            (
                "§13.5",
                str(r.dataset),
                str(r.split),
                float(r.p_phipson_smyth),
                "순열 %d회" % int(r.n_rep_valid),
            )
        )
    for fn, sec, labs in (
        ("concept_shift.csv", "§18", ("arm",)),
        ("applicability_domain.csv", "§21", ("stratum",)),
        ("label_threshold.csv", "§24", ("label_def", "setting")),
    ):
        t = cv(fn)
        for _, r in t.iterrows():
            if r.p_phipson_smyth != r.p_phipson_smyth:
                continue
            rows.append(
                (
                    sec,
                    str(r.dataset),
                    " ".join(str(r[c]) for c in labs),
                    float(r.p_phipson_smyth),
                    "순열 %d회" % int(r.n_rep_valid),
                )
            )
    md = cv("mondrian_conformal.csv")
    md = md[md.arm.str.startswith("A.")]
    for _, r in md.iterrows():
        tgt = 1.0 - float(r.alpha)
        for cls, ck, nk in (("양성", "cov_pos", "n_pos"), ("음성", "cov_neg", "n_neg")):
            n = float(r[nk])
            if n <= 0:
                continue
            se = (tgt * (1 - tgt) / n) ** 0.5
            z = (float(r[ck]) - tgt) / se
            rows.append(
                (
                    "§11.1",
                    str(r.dataset),
                    "%s 목표 %.0f%% %s 클래스 커버리지" % (str(r.method), tgt * 100, cls),
                    norm_sf(-z),
                    "단측 z=%+.2f, n=%d" % (z, int(n)),
                )
            )
    return rows


def main():
    log("§15 의 가족에서 빠져 있던 검정을 전부 편입한다 — 한계 19번")
    log("모델을 다시 적합하지 않는다. 저장된 p 와 z 만 쓴다. α = %.2f" % ALPHA)
    log("")

    mc = cv("multiple_comparison_correction.csv")
    new = collect()
    log(
        "편입 대상 %d건 (%s), 기존 %d건 → 합계 %d건"
        % (
            len(new),
            " · ".join("%s %d" % (sc, sum(1 for x in new if x[0] == sc)) for sc in SECS),
            len(mc),
            len(new) + len(mc),
        )
    )
    log("")

    # ---- F4 — 빠져 있던 검정만
    q4 = bh([x[3] for x in new])

    # ---- F0 — 전부 한 가족
    all_p = list(mc.p) + [x[3] for x in new]
    all_lab = [
        ("%s" % r.family, str(r.dataset), str(r.test), float(r.p), "기존") for _, r in mc.iterrows()
    ] + [(x[0], x[1], x[2], x[3], x[4]) for x in new]
    q0 = bh(all_p)

    out = []
    for i, (fam, ds, name, p, note) in enumerate(all_lab):
        is_new = i >= len(mc)
        out.append(
            dict(
                source=fam,
                dataset=ds,
                test=name,
                p=p,
                note=note,
                newly_included=is_new,
                q_existing=(float(mc.q_BH.iloc[i]) if not is_new else float("nan")),
                q_F4=(q4[i - len(mc)] if is_new else float("nan")),
                q_F0=q0[i],
                sig_existing=(bool(mc.significant_after_FDR.iloc[i]) if not is_new else None),
                sig_F4=((q4[i - len(mc)] <= ALPHA) if is_new else None),
                sig_F0=bool(q0[i] <= ALPHA),
            )
        )
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(RES, "fdr_all_tests.csv"), index=False)

    # ---- 표 1: 새 가족 F4
    log("=" * 112)
    log("F4 — 빠져 있던 검정만 모은 가족 (%d건)" % len(new))
    log("=" * 112)
    log(
        "%-7s %-9s %-44s %9s %9s %9s %6s"
        % ("절", "데이터셋", "검정", "p", "q (F4)", "q (F0)", "F4 유의")
    )
    log("-" * 112)
    g = d[d.newly_included].sort_values("p")
    for _, r in g.iterrows():
        log(
            "%-7s %-9s %-44s %9.4f %9.4f %9.4f %6s"
            % (
                r.source,
                r.dataset,
                r.test[:44],
                r.p,
                r.q_F4,
                r.q_F0,
                "예" if r.sig_F4 else "아니오",
            )
        )
    log("")

    # ---- 표 2: 기존 결론이 F0 에서 살아남는가
    log("=" * 112)
    log("F0 — 전부 한 가족으로 합쳤을 때 기존 %d건의 결론 변화" % len(mc))
    log("=" * 112)
    old = d[~d.newly_included].copy()
    # sig_existing 열은 새 행에 None 이 들어가 object dtype 이 된다.
    # ~ 연산이 정수 보수로 동작해 개수가 음수로 나왔다 (2026-10-20).
    old["sig_existing"] = old.sig_existing.astype(bool)
    old["sig_F0"] = old.sig_F0.astype(bool)
    flip = old[old.sig_existing & ~old.sig_F0]
    gain = old[~old.sig_existing & old.sig_F0]
    log("%-22s %-44s %9s %9s %9s" % ("가족", "검정", "p", "q (기존)", "q (F0)"))
    log("-" * 112)
    for _, r in old[old.sig_existing].sort_values("p").iterrows():
        log(
            "%-22s %-44s %9.4f %9.4f %9.4f%s"
            % (
                r.source,
                ("%s %s" % (r.dataset, r.test))[:44],
                r.p,
                r.q_existing,
                r.q_F0,
                "" if r.sig_F0 else "   ← 유의성 상실",
            )
        )
    log("")
    log(
        "  기존 유의 %d건 중 F0 에서 유의성을 잃는 것: %d건"
        % (int(old.sig_existing.sum()), len(flip))
    )
    log(
        "  기존 비유의 %d건 중 F0 에서 유의해지는 것: %d건"
        % (int((~old.sig_existing).sum()), len(gain))
    )
    log("")

    # ---- 판정
    log("=" * 112)
    log("판정")
    log("=" * 112)
    _n4 = int(g.sig_F4.sum())
    log(
        "1. 빠져 있던 %d건을 한 가족으로 보정하면 %d건이 유의하다(α=%.2f)." % (len(new), _n4, ALPHA)
    )
    for src in SECS:
        _a = g[g.source == src]
        log(
            "   %s — %d건 중 %d건 유의%s"
            % (
                src,
                len(_a),
                int(_a.sig_F4.sum()),
                (
                    " (%s)"
                    % ", ".join(
                        "%s %s" % (r.dataset, r.test[:30]) for _, r in _a[_a.sig_F4].iterrows()
                    )
                    if int(_a.sig_F4.sum())
                    else ""
                ),
            )
        )
    log("")
    log(
        "2. 전부 한 가족(F0, %d건)으로 합쳐도 기존 결론은 %s."
        % (
            len(all_p),
            (
                "그대로다"
                if len(flip) == 0
                else "%d건이 유의성을 잃는다 — %s"
                % (
                    len(flip),
                    ", ".join("%s %s" % (r.dataset, r.test[:30]) for _, r in flip.iterrows()),
                )
            ),
        )
    )
    log("")
    # §11.1 은 두 절차를 함께 싣는다. 주변형의 음성 클래스 z 는 §11 이 이미
    # 보고한 붕괴 자체이므로 매우 유의하고, 새 주장은 Mondrian 쪽이다.
    # 둘을 뭉쳐서 "유의한 것이 있다" 고 쓰면 새 주장이 유의한 것처럼 읽힌다.
    _mon = g[(g.source == "§11.1") & g.test.str.startswith("Mondrian")]
    _mar = g[(g.source == "§11.1") & g.test.str.startswith("주변형")]
    log("3. §11.1 은 절차 둘을 함께 싣는다 — 나누어 읽어야 한다.")
    log(
        "   주변형 %d건 중 %d건 유의 (§11 이 이미 보고한 커버리지 붕괴 자체다)."
        % (len(_mar), int(_mar.sig_F4.sum()))
    )
    log(
        "   Mondrian %d건 중 %d건 유의%s"
        % (
            len(_mon),
            int(_mon.sig_F4.sum()),
            (
                "."
                if int(_mon.sig_F4.sum())
                else " — **클래스별 보장을 붙인 뒤의 미달분은 FDR 보정 후 유의하지 "
                "않다**. 그래서 §11.1 의 근거는 효과크기(커버리지 부족분)로 쓰고 "
                "z 는 보조로만 쓴다."
            ),
        )
    )
    log("")
    # 민감도 — 주변형 행을 빼면 (가족이 작아지고 작은 p 의 도움이 사라진다)
    _sub = [x for x in new if not (x[0] == "§11.1" and x[2].startswith("주변형"))]
    _qs = bh([x[3] for x in _sub])
    _mon2 = [(x, q) for x, q in zip(_sub, _qs) if x[0] == "§11.1" and x[2].startswith("Mondrian")]
    log(
        "   민감도 — §11.1 주변형 행(§11 이 이미 보고한 것)을 가족에서 빼면 "
        "%d건이 되고, Mondrian 쪽 q 는 %s 다. 어느 구성에서도 %s."
        % (
            len(_sub),
            " / ".join("%.4f" % q for _x, q in sorted(_mon2, key=lambda t: t[1])[:4]),
            "유의해지지 않는다" if all(q > ALPHA for _x, q in _mon2) else "일부가 유의해진다",
        )
    )
    log("")
    log(
        "4. 한계 19번은 이 절로 닫힌다 — 빠져 있던 검정을 빼지 않고 넣었고, "
        "넣었을 때 무엇이 살아남는지 적었다."
    )
    log("")
    log("저장: %s" % os.path.join(RES, "fdr_all_tests.csv"))
    log.save(os.path.join(RES, "fdr_all_tests_summary.txt"))


if __name__ == "__main__":
    main()
