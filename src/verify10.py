# -*- coding: utf-8 -*-
"""논문용 보고서 셀 단위 대조 — verify9 의 약점을 메운다.

verify9 의 수치 추적 검사는 "문서의 숫자가 results/ 어딘가에 있는가" 만 본다.
그래서 **엉뚱한 열의 값을 써도** 우연히 다른 셀과 일치하면 통과한다
(돌연변이 시험에서 실제로 통과했다).

이 스크립트는 반대 방향으로 검사한다 — CSV 의 특정 행·열에서 기대 문자열을
직접 만들어 문서에 그 문자열이 있는지 확인한다. 열을 잘못 고르면 잡힌다.

    python src/verify10.py
"""

import glob
import io
import os
import re
import sys
import pandas as pd


def _find_root():
    """프로젝트 루트 = 이 파일이 있는 src/ 의 상위 디렉터리.
    저장소를 어디에 두어도 그대로 돌아간다. FDA_ROOT 환경변수로 덮어쓸 수 있다."""
    import os as _o

    r = _o.environ.get("FDA_ROOT") or _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
    r = _o.path.abspath(_o.path.expanduser(r))
    if not (_o.path.isdir(_o.path.join(r, "results")) and _o.path.isdir(_o.path.join(r, "src"))):
        raise SystemExit("프로젝트 루트를 찾지 못했다 (results/ 와 src/ 가 있어야 한다): %s" % r)
    return r


ROOT = _find_root()
DOC = os.path.join(ROOT, "RESULTS_REPORT.md")
RES = os.path.join(ROOT, "results")

S = io.open(DOC, encoding="utf-8").read()
FLAT = S.replace("**", "")  # 강조 표시를 지운 사본으로 대조한다
fails, n_cells = [], 0


def cv(n):
    return pd.read_csv(os.path.join(RES, n))


def f4(x):
    return "%.4f" % x


def f2(x):
    return "%.2f" % x


def p1(x):
    return "%.1f%%" % x


def sg(x):
    return "%+.4f" % x


def cell(val, where):
    """이 값이 문서에 (강조 여부와 무관하게) 있는지."""
    global n_cells
    n_cells += 1
    if val not in FLAT:
        fails.append("%s: %r 가 문서에 없다" % (where, val))


def rowstr(vals, where):
    """표 한 행 전체가 그 순서대로 붙어 있는지."""
    global n_cells
    n_cells += 1
    s = "| " + " | ".join(vals) + " |"
    if s not in FLAT:
        fails.append("%s: 행이 문서에 없다\n      기대: %s" % (where, s))


DS = ["DILIrank", "DICTrank"]

# ---- §4 분할 전략 -------------------------------------------------------
bc = cv("baseline_cv_per_fold.csv")
for d in DS:
    g = bc[bc.dataset == d]
    for sp, lab in [("random_cv", "무작위 5-fold CV"), ("scaffold_cv", "스캐폴드 5-fold CV")]:
        gg = g[g.split == sp]
        rowstr(
            [
                d,
                lab,
                "%s ± %s" % (f4(gg.AUROC.mean()), f4(gg.AUROC.std(ddof=1))),
                "%s ± %s" % (f4(gg.ECE.mean()), f4(gg.ECE.std(ddof=1))),
            ],
            "§4 %s %s" % (d, sp),
        )
    t = g[g.split == "temporal"].iloc[0]
    rowstr(
        [d, "시간분할", "%s [%s–%s]" % (f4(t.AUROC), f4(t.auroc_lo), f4(t.auroc_hi)), f4(t.ECE)],
        "§4 %s temporal" % d,
    )

# ---- §5 순열검정 --------------------------------------------------------
pt = cv("permutation_test.csv")
for _, r in pt.iterrows():
    rowstr(
        [
            r.dataset,
            f4(r.auroc_temporal),
            "%s (%s)" % (f4(r.null_mean), f4(r.null_sd)),
            "%s–%s" % (f4(r.null_p2_5), f4(r.null_p97_5)),
            "+%s" % f4(r.delta_auroc),
            "%d/%d" % (r.n_le_observed, r.n_perm_valid),
            f4(r.p_permutation),
        ],
        "§5 %s" % r.dataset,
    )

# ---- §6 보정 ------------------------------------------------------------
cal = cv("calibration_results.csv")
C = {
    (d, st): cal[(cal.dataset.str.startswith(d)) & (cal.setting == st)].iloc[0]
    for d in DS
    for st in ("random", "temporal")
}
for lab, k, good in [
    ("AUROC", "AUROC", "높을수록"),
    ("Brier", "Brier", "낮을수록"),
    ("ECE", "ECE", "낮을수록"),
    ("MCE", "MCE", "낮을수록"),
    ("보정기울기", "calib_slope", "1에 가까울수록"),
    ("보정절편", "calib_intercept", "0에 가까울수록"),
    ("BalAcc @0.5", "BalAcc@0.5", "높을수록"),
    ("MCC @0.5", "MCC@0.5", "높을수록"),
]:
    rowstr(
        [
            lab,
            f4(C[("DILIrank", "random")][k]),
            f4(C[("DILIrank", "temporal")][k]),
            f4(C[("DICTrank", "random")][k]),
            f4(C[("DICTrank", "temporal")][k]),
            good,
        ],
        "§6 %s" % lab,
    )
for d in DS:
    for st, lab in [("random", "무작위"), ("temporal", "시간분할")]:
        r = C[(d, st)]
        rowstr(
            [d, lab, f4(r.prevalence), f4(r.mean_pred), sg(r.mean_pred - r.prevalence)],
            "§6 치우침 %s %s" % (d, lab),
        )
    r = C[(d, "temporal")]
    rowstr(
        [
            d,
            f4(r.best_thr),
            "%s → %s (%s)"
            % (f4(r["BalAcc@0.5"]), f4(r["BalAcc@best"]), sg(r["BalAcc@best"] - r["BalAcc@0.5"])),
            "%s → %s (%s)"
            % (f4(r["MCC@0.5"]), f4(r["MCC@best"]), sg(r["MCC@best"] - r["MCC@0.5"])),
        ],
        "§6 임계값 %s" % d,
    )

# ---- §7 Brier -----------------------------------------------------------
bd = cv("brier_decomposition.csv")
r3 = cv("brier_decomposition_r3.csv")
B = {d: bd[bd.dataset == d].iloc[0] for d in DS}
for lab, kt, kr in [
    ("resolution (판별력)", "resolution_temporal", "resolution_random_mean"),
    ("reliability (보정)", "reliability_temporal", "reliability_random_mean"),
    ("Brier", "brier_temporal", "brier_random_mean"),
]:
    vals = [lab]
    for d in DS:
        r = B[d]
        vals += [f4(r[kr]), f4(r[kt]), sg(r[kt] - r[kr])]
    rowstr(vals, "§7 %s" % lab)
rowstr(
    ["보정(reliability) 기여"]
    + [
        "%s [95%% CI %s–%s]"
        % (
            p1(B[d].pct_from_reliability_A),
            p1(B[d].pct_reliability_A_lo),
            p1(B[d].pct_reliability_A_hi),
        )
        for d in DS
    ],
    "§7 기여율",
)
rowstr(["판별력(resolution) 기여"] + [p1(B[d].pct_from_resolution_A) for d in DS], "§7 판별력 기여")
rowstr(
    ["보정 / 판별력 비"] + ["%s배" % f2(B[d].d_reliability_A / B[d].d_resolution_A) for d in DS],
    "§7 비",
)
for d in DS:
    for b in ("R1", "R2", "R3"):
        r = r3[(r3.dataset == d) & (r3.baseline.str.startswith(b))].iloc[0]
        rowstr(
            [
                d,
                b,
                sg(r.d_reliability),
                sg(r.d_resolution),
                sg(r.d_uncertainty),
                p1(r.share2_reliability),
                p1(r.share3_reliability),
            ],
            "§7.1 %s %s" % (d, b),
        )

# ---- §7.2 / §8.1 matched-n ---------------------------------------------
mn = cv("matched_n_control.csv")
pc = cv("prevalence_control.csv")
for d in DS:
    dfull = mn[(mn.dataset == d) & (mn.baseline == "R1")].delta_T_matched.iloc[0]
    for b in ("R1", "R2", "R3"):
        r = mn[(mn.dataset == d) & (mn.baseline == b)].iloc[0]
        same = abs(r.ratio_old - r.ratio_new) < 1e-12
        ar = lambda a, bb, fmt: (fmt(a) if same else "%s → %s" % (fmt(a), fmt(bb)))
        rowstr(
            [
                d,
                b,
                "%d / %d" % (r.n_ctrl, r.n_test_temporal),
                ar(r.ratio_old, r.ratio_new, f2),
                ar(r.p_old, r.p_new, f4),
                ar(r.share_old, r.share_new, p1),
                ar(dfull - r.delta_ctrl, r.delta_T_matched - r.delta_ctrl, sg),
            ],
            "§7.2 %s %s" % (d, b),
        )
        ng = int(
            pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith(b))].n_ge_observed.iloc[
                0
            ]
        )
        rowstr(
            [
                d,
                b,
                "%s (%s)" % (f4(r.ece_ctrl), f4(r.ece_ctrl_sd)),
                "%s / %s" % (f4(r.ece_T_full), f4(r.ece_T_matched)),
                f2((r.ece_T_full - r.ece_ctrl) / r.ece_ctrl_sd),
                f2((r.ece_T_matched - r.ece_ctrl) / r.ece_ctrl_sd),
                "%d/200" % ng,
                "%s → %s" % (f4(r.p_old), f4(r.p_new)),
            ],
            "§8.1 %s %s" % (d, b),
        )

# ---- §8 유병률 통제 -----------------------------------------------------
for d in DS:
    t = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("T"))].iloc[0]
    rowstr(
        [
            d,
            "T. 시간분할",
            "%d" % t.n,
            f4(t.pos_rate),
            "%s [%s–%s]" % (f4(t.ECE), f4(t.ECE_lo), f4(t.ECE_hi)),
            f4(t.AUROC),
            f4(t.slope),
            "—",
            "—",
            "—",
        ],
        "§8 %s T" % d,
    )
    for b, lab in [
        ("R1", "R1. 무작위 (크기맞춤)"),
        ("R2", "R2. + 검증군 비율맞춤"),
        ("R3", "R3. + 학습·검증 비율맞춤"),
    ]:
        r = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith(b))].iloc[0]
        rowstr(
            [
                d,
                lab,
                "%.0f" % r.n,
                f4(r.pos_rate),
                "%s ± %s" % (f4(r.ECE), f4(r.ECE_sd)),
                f4(r.AUROC),
                f4(r.slope),
                "%s배" % f2(r.ece_ratio),
                "%d/%d" % (r.n_ge_observed, r.n_rep_valid),
                f4(r.p_perm),
            ],
            "§8 %s %s" % (d, b),
        )
    ex = t.ECE - pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("R1"))].ECE.iloc[0]
    rm = t.ECE - pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("R3"))].ECE.iloc[0]
    rowstr([d, f4(ex), f4(rm), p1((ex - rm) / ex * 100)], "§8 해소율 %s" % d)

# ---- §9 강건성 ----------------------------------------------------------
rob = cv("robustness_models.csv")
for d in DS:
    g = rob[rob.dataset == d]
    rowstr(
        [
            d,
            "%d" % len(g),
            "%d/%d" % ((g.d_auroc > 0).sum(), len(g)),
            "%s (%s ~ %s)" % (sg(g.d_auroc.median()), sg(g.d_auroc.min()), sg(g.d_auroc.max())),
            "%d/%d" % ((g.ece_ratio > 1).sum(), len(g)),
            "%s (%s ~ %s)"
            % (f2(g.ece_ratio.median()), f2(g.ece_ratio.min()), f2(g.ece_ratio.max())),
            f4(g.slope_temporal.median()),
        ],
        "§9.1 %s" % d,
    )
robc = cv("robustness_cutoff.csv")
for d in DS:
    for _, r in robc[robc.dataset == d].sort_values("cutoff").iterrows():
        rowstr(
            [
                d,
                "%d" % r.cutoff,
                "%d" % r.n_train,
                "%d" % r.n_test,
                f4(r.test_pos_rate),
                f4(r.auroc_random),
                f4(r.auroc_temporal),
                sg(r.d_auroc),
                "%s배" % f2(r.ece_ratio),
                f4(r.slope),
            ],
            "§9.2 %s %d" % (d, r.cutoff),
        )
seed = cv("seed_sensitivity.csv")
for d in DS:
    g = seed[seed.dataset == d]
    b = g[g.seed == 42].iloc[0]
    rowstr(
        [
            d,
            "%s ± %s (%s–%s)"
            % (f4(g.auroc.mean()), f4(g.auroc.std(ddof=1)), f4(g.auroc.min()), f4(g.auroc.max())),
            "%s ± %s (%s–%s)"
            % (f4(g.ece.mean()), f4(g.ece.std(ddof=1)), f4(g.ece.min()), f4(g.ece.max())),
            "AUROC %s, ECE %s" % (f4(b.auroc), f4(b.ece)),
        ],
        "§9.3 시드 %s" % d,
    )
binn = cv("ece_binning_sensitivity.csv")
for d in DS:
    g = binn[binn.dataset == d]
    b = g[(g.scheme == "균등폭") & (g.n_bins == 10)].iloc[0]
    rowstr(
        [
            d,
            "%s–%s" % (f4(g.ece.min()), f4(g.ece.max())),
            "%s–%s" % (f4(g.reliability.min()), f4(g.reliability.max())),
            "%s–%s" % (f4(g.resolution.min()), f4(g.resolution.max())),
            "ECE %s / rel %s / res %s" % (f4(b.ece), f4(b.reliability), f4(b.resolution)),
        ],
        "§9.4 %s" % d,
    )
grid = cv("grid_calibration.csv")
for d in DS:
    g = grid[grid.dataset == d]
    rowstr(
        [
            d,
            "%s (%s–%s)"
            % (f2(g.r1_ece_ratio.median()), f2(g.r1_ece_ratio.min()), f2(g.r1_ece_ratio.max())),
            "%s (%s–%s)"
            % (f2(g.r2_ece_ratio.median()), f2(g.r2_ece_ratio.min()), f2(g.r2_ece_ratio.max())),
            "%d/%d" % ((g.r1_share2 > 50).sum(), len(g)),
            "%d/%d" % ((g.r2_share2 > 50).sum(), len(g)),
            "%d/%d" % ((g.conf90_marginal < 0.9).sum(), len(g)),
            "%d/%d" % ((g.conf80_marginal < 0.8).sum(), len(g)),
        ],
        "§9.5 %s" % d,
    )
tie = cv("tiebreak_sensitivity.csv")
for d in DS:
    g = tie[tie.dataset == d]
    e3 = g[g.experiment.str.startswith("E3")]
    e1 = g[g.experiment.str.startswith("E1")]
    e2 = g[g.experiment.str.startswith("E2")]
    for lab, k, fmt in [
        ("conf90 주변커버리지", "conf90_marginal", f4),
        ("conf80 주변커버리지", "conf80_marginal", f4),
        ("Platt 회복률 %", "rec_B", f2),
        ("Isotonic 회복률 %", "rec_C", f2),
        ("AUROC", "auroc", f4),
    ]:
        rowstr(
            [
                d,
                lab,
                " / ".join(fmt(v) for v in e3[k]),
                "%s–%s (폭 %s)"
                % (fmt(e1[k].min()), fmt(e1[k].max()), fmt(e1[k].max() - e1[k].min())),
                "%s–%s (폭 %s)"
                % (fmt(e2[k].min()), fmt(e2[k].max()), fmt(e2[k].max() - e2[k].min())),
            ],
            "§9.6 %s %s" % (d, lab),
        )

# ---- §10 재보정 ---------------------------------------------------------
rec = cv("recalibration_results.csv")
for d in DS:
    g = rec[rec.dataset.str.startswith(d)]
    a = g[g.method.str.startswith("A.")].iloc[0]
    e = g[g.method.str.startswith("E.")].iloc[0]
    for _, r in g.iterrows():
        recv = "—" if r.method.startswith("A") else p1((a.ECE - r.ECE) / (a.ECE - e.ECE) * 100)
        rowstr(
            [
                r.method,
                f4(r.AUROC),
                f4(r.ECE),
                f4(r.slope),
                sg(r.intercept),
                f4(r.mean_pred),
                f4(r.MCC),
                recv,
            ],
            "§10 %s %s" % (d, r.method),
        )
recs = cv("recalibration_sensitivity.csv")
for d in DS:
    b = recs[(recs.dataset == d) & (recs.method.str.startswith("B."))].recovery
    c = recs[(recs.dataset == d) & (recs.method.str.startswith("C."))].recovery
    both = pd.concat([b, c])
    rowstr(
        [
            d,
            "%s–%s%%" % (f2(b.min()), f2(b.max())),
            "%s–%s%%" % (f2(c.min()), f2(c.max())),
            "%s–%s%%" % (f2(both.min()), f2(both.max())),
        ],
        "§10.1 %s" % d,
    )

# ---- §11 conformal -----------------------------------------------------
conf = cv("conformal_prediction.csv")
for _, r in conf.iterrows():
    rowstr(
        [
            r.dataset,
            "%.0f%%" % (r.target_coverage * 100),
            f4(r.marginal_coverage_random),
            f4(r.marginal_coverage_temporal),
            "%s → %s" % (f4(r.coverage_positive_random), f4(r.coverage_positive_temporal)),
            "%s → %s" % (f4(r.coverage_negative_random), f4(r.coverage_negative_temporal)),
            "%s → %s" % (f4(r.mean_set_size_random), f4(r.mean_set_size_temporal)),
        ],
        "§11 %s a=%s" % (r.dataset, r.alpha),
    )

# ---- §12 검열 -----------------------------------------------------------
cens = cv("censoring_results.csv")
for d in DS:
    g = cens[cens.dataset.str.startswith(d)]
    base = g[g.exp.str.startswith("기준")].iloc[0]
    for _, r in g.iterrows():
        rel = "기준" if r.exp.startswith("기준") else sg(base.gap - r.gap)
        rowstr(
            [d, r.exp, "%d" % r.n_test, f4(r.pos_rate), f4(r.auroc), "+%s" % f4(r.gap), rel],
            "§12 %s %s" % (d, r.exp),
        )
cc = cv("calibration_confounders.csv")
for d in DS:
    for _, r in cc[(cc.dataset == d) & (cc.group.isin(["기준", "라벨성숙도"]))].iterrows():
        rowstr(
            [
                d,
                r.condition,
                "%d" % r.n,
                f4(r.pos_rate),
                f4(r.ece),
                "%s배" % f2(r.ece_ratio_pm),
                p1(r.share_reliability_pm),
            ],
            "§12.1 %s %s" % (d, r.condition),
        )

# ---- §13 이동 -----------------------------------------------------------
ep = cv("era_permutation.csv")
for _, r in ep.iterrows():
    rowstr(
        [
            r.dataset,
            "%d (%d)" % (r.n, r.n_post),
            f4(r.era_auc),
            "%s (%s)" % (f4(r.null_mean), f4(r.null_sd)),
            "%s–%s" % (f4(r.null_p025), f4(r.null_p975)),
            "%d/%d" % (r.n_ge, r.n_perm),
            f4(r.p_phipson_smyth),
            f2(r.z),
        ],
        "§13.1 %s" % r.dataset,
    )
sqd = cv("shift_quantification_descriptors.csv")
for desc in ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumRotatableBonds", "NumHDonors"]:
    vals = [desc]
    for d in DS:
        r = sqd[(sqd.dataset == d) & (sqd.descriptor == desc)].iloc[0]
        fm = (
            (lambda x: "%.0f" % x)
            if desc.startswith("Num")
            else ((lambda x: "%.1f" % x) if desc in ("MolWt", "TPSA") else (lambda x: "%.2f" % x))
        )
        vals.append("%s → %s (+%s)" % (fm(r.median_train), fm(r.median_test), fm(r["diff"])))
    rowstr(vals, "§13.2 %s" % desc)
for d in DS:
    for _, r in cc[(cc.dataset == d) & (cc.condition.str.startswith("B 유사도 ["))].iterrows():
        rowstr(
            [d, r.condition.replace("B 유사도 ", ""), "%d" % r.n, f4(r.auroc), f4(r.ece)],
            "§13.3 %s %s" % (d, r.condition),
        )
    for _, r in cc[(cc.dataset == d) & (cc.group == "연도구간")].iterrows():
        rowstr(
            [d, r.condition.replace("C ", ""), "%d" % r.n, f4(r.pos_rate), f4(r.auroc), f4(r.ece)],
            "§13.4 %s %s" % (d, r.condition),
        )

# ---- §14 하위집단 -------------------------------------------------------
atcs = cv("atc_stratified.csv")
atct = cv("atc_tests.csv")
aa = cv("atc_audit.csv")
for code in ["A", "B", "C", "J", "L", "N", "V"]:
    parts = [code]
    a = atcs[(atcs.dataset == "DILIrank") & (atcs.atc == code)]
    b = atcs[(atcs.dataset == "DICTrank") & (atcs.atc == code)]
    parts.append((a.name.iloc[0] if len(a) else b.name.iloc[0]))
    for g in (a, b):
        r = g.iloc[0]
        parts += [
            "%d" % r.n,
            p1(r.pos_rate * 100),
            ("정의 불가" if pd.isna(r.auroc) else f4(r.auroc)),
        ]
    rowstr(parts, "§14.1 ATC %s" % code)
aa_te = aa[aa.approval_year >= 2010]
for d in DS:
    r = atct[(atct.dataset == d) & (atct.test.str.startswith("ATC L"))].iloc[0]
    g = aa_te[aa_te.dataset == d]
    rowstr(
        [
            d,
            "n=%d, 양성 %s, AUROC %s" % (r.n_a, p1(g[g.atc1 == "L"].y.mean() * 100), f4(r.auroc_a)),
            "n=%d, 양성 %s, AUROC %s" % (r.n_b, p1(g[g.atc1 != "L"].y.mean() * 100), f4(r.auroc_b)),
            sg(r["diff"]),
            f4(r.p),
        ],
        "§14.1 L vs 그외 %s" % d,
    )
    for _, r in atct[(atct.dataset == d) & (atct.test.str.startswith("구조"))].iterrows():
        rowstr(
            [
                d,
                r.test.replace("구조: ", ""),
                "%d, %s" % (r.n_a, f4(r.auroc_a)),
                "%d, %s" % (r.n_b, f4(r.auroc_b)),
                sg(r["diff"]),
                f4(r.p),
            ],
            "§14.2 %s %s" % (d, r.test),
        )

# ---- §16 민감도 ---------------------------------------------------------
for fn, sec in [("year_sensitivity.csv", "§16.1"), ("atc_v_sensitivity.csv", "§16.2")]:
    df = cv(fn)
    for d in DS:
        for _, r in df[df.dataset == d].iterrows():
            rowstr(
                [
                    d,
                    r.scenario,
                    "%d" % r.n,
                    "%d" % r.n_test,
                    f4(r.AUROC),
                    f4(r.ECE),
                    f4(r.ECE_random_mean),
                    "%s배" % f2(r.ece_ratio),
                    p1(r.pct_from_reliability),
                ],
                "%s %s %s" % (sec, d, r.scenario),
            )
ster = cv("stereo_sensitivity.csv")
for d in DS:
    for _, r in ster[ster.dataset == d].iterrows():
        rowstr(
            [
                d,
                r.variant,
                "%d" % r.n,
                "%d / %d" % (r.n_train, r.n_test),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ece_random_pm),
                "%s배" % f2(r.ece_ratio_pm),
                p1(r.share_reliability),
            ],
            "§16.3 %s %s" % (d, r.variant),
        )

# ---- §17 중첩 -----------------------------------------------------------
ov = cv("compound_overlap.csv")
for _, r in ov.iterrows():
    rowstr(
        [
            r.dataset,
            "%d" % r.n_analysis,
            "%d" % r.n_shared,
            p1(r.shared_pct),
            "%d" % r.n_exclusive,
            "%d / %d" % (r.n_exclusive_train, r.n_exclusive_test),
            "%d / %d" % (r.n_exclusive_test_pos, r.n_exclusive_test_neg),
        ],
        "§17.1 %s" % r.dataset,
    )
dj = cv("disjoint_subset.csv")
for d in DS:
    for v in ["V0", "V1", "V1c", "V2", "V3"]:
        r = dj[(dj.dataset == d) & (dj.variant == v)].iloc[0]
        rowstr(
            [
                d,
                v,
                "%d" % r.n_train,
                "%d" % r.n_test,
                "%d/%d" % (r.n_test_pos, r.n_test_neg),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ctrl_ece),
                "%s배" % f2(r.ece_ratio),
                f4(r.p_phipson_smyth),
                p1(r.share_calib),
                "표본부족" if r.underpowered else "",
            ],
            "§17.2 %s %s" % (d, v),
        )

# ---- §1 핵심 결과 요약표 (2026-10-02 추가) -----------------------------
# §1 은 문서에서 가장 많이 인용되는 표인데 지금까지 verify10 의 대상이 아니었다.
# verify9 의 토큰 추적만으로는 '엉뚱한 열을 써도 통과' 하는 약점이 남으므로,
# 15행 전부를 각 결과 파일에서 독립적으로 다시 만들어 대조한다.
bc1 = cv("baseline_cv_per_fold.csv")
cal1 = cv("calibration_results.csv")
pt1 = cv("permutation_test.csv")
pc1 = cv("prevalence_control.csv")
bd1 = cv("brier_decomposition.csv")
r31 = cv("brier_decomposition_r3.csv")
recs1 = cv("recalibration_sensitivity.csv")
conf1 = cv("conformal_prediction.csv")
ep1 = cv("era_permutation.csv")
rv1 = cv("reverse_split.csv")
sm1 = cv("shift_matched_split.csv")
pe1 = cv("prior_estimation.csv")
KO = {"DILIrank": "DILIrank", "DICTrank": "DICTrank"}
for d in DS:
    T1 = bc1[(bc1.dataset == d) & (bc1.split == "temporal")].iloc[0]
    rcv = bc1[(bc1.dataset == d) & (bc1.split == "random_cv")]
    CT1 = cal1[(cal1.dataset.str.startswith(d)) & (cal1.setting == "temporal")].iloc[0]
    CR1 = cal1[(cal1.dataset.str.startswith(d)) & (cal1.setting == "random")].iloc[0]
    P1 = pt1[pt1.dataset == d].iloc[0]
    R1r = pc1[(pc1.dataset.str.startswith(d)) & (pc1.setting.str.startswith("R1"))].iloc[0]
    R3r = pc1[(pc1.dataset.str.startswith(d)) & (pc1.setting.str.startswith("R3"))].iloc[0]
    B1 = bd1[bd1.dataset == d].iloc[0]
    X3 = r31[(r31.dataset == d) & (r31.baseline.str.startswith("R3"))].iloc[0]
    E1 = ep1[ep1.dataset == d].iloc[0]
    V1 = rv1[
        (rv1.dataset == d) & (rv1.direction == "역방향") & (rv1.setting.str.startswith("R1"))
    ].iloc[0]
    # 1행: 분석집합 n
    cell("%d (%d / %d)" % (T1.n_train + T1.n_test, T1.n_train, T1.n_test), "§1-1 %s n" % d)
    # 2행: AUROC 무작위 CV -> 시간분할
    cell("%s → " % f4(rcv.AUROC.mean()), "§1-2 %s 무작위CV AUROC" % d)
    cell(f4(T1.AUROC), "§1-2 %s 시간분할 AUROC" % d)
    # 3행: ΔAUROC + p
    cell("+%s (p = %s)" % (f4(P1.delta_auroc), f4(P1.p_permutation)), "§1-3 %s ΔAUROC" % d)
    # 4행: ECE
    cell("%s → " % f4(CR1.ECE), "§1-4 %s 무작위 ECE" % d)
    cell(f4(CT1.ECE), "§1-4 %s 시간분할 ECE" % d)
    # 5·6행: ECE 배율 R1 / R3
    cell("%s배" % f2(R1r.ece_ratio), "§1-5 %s R1 배율" % d)
    cell("%s배 (p = %s)" % (f2(R3r.ece_ratio), f4(R3r.p_perm)), "§1-6 %s R3 배율·p" % d)
    # 7행: 보정기울기
    cell("%s → " % f4(CR1.calib_slope), "§1-7 %s 무작위 기울기" % d)
    cell(f4(CT1.calib_slope), "§1-7 %s 시간분할 기울기" % d)
    # 8·9행: 보정 기여율 주 분석 / R3
    cell(p1(B1.pct_from_reliability_A), "§1-8 %s 기여율(주)" % d)
    cell(p1(X3.share2_reliability), "§1-9 %s 기여율(R3)" % d)
    # 10행: 재보정 회복률 범위 (B·C 합집합)
    g = recs1[(recs1.dataset == d) & (recs1.method.str.startswith(("B.", "C.")))]
    cell("%s–%s%%" % (f2(g.recovery.min()), f2(g.recovery.max())), "§1-10 %s 회복률 범위" % d)
    # 11행: conformal 90% 실측
    cell(
        f4(conf1[(conf1.dataset == d) & (conf1.alpha == 0.1)].marginal_coverage_temporal.iloc[0]),
        "§1-11 %s conf90" % d,
    )
    # 12행: 시대 AUC + p
    cell("%s (p = %s)" % (f4(E1.era_auc), f4(E1.p_phipson_smyth)), "§1-12 %s 시대 AUC·p" % d)
    # 13행: 역방향 배율 + p
    cell("%s배 (p = %s)" % (f2(V1.ece_ratio), f4(V1.p_phipson_smyth)), "§1-13 %s 역방향 배율·p" % d)
    # 14행: 비시간 분할 최대 배율
    cell(
        "%s배" % f2(sm1[(sm1.dataset == d) & (sm1.split.str.startswith("X"))].ece_ratio.max()),
        "§1-14 %s 비시간 최대배율" % d,
    )
    # 15행: 라벨 없는 사전확률 최선 회복률
    nl = pe1[
        (pe1.dataset == d)
        & (pe1.estimator.str.startswith("P"))
        & (~pe1.estimator.str.startswith("P4"))
    ]
    cell(p1(nl.recovery.max()), "§1-15 %s 라벨없는 최선" % d)

# ---- §2.3 시간분할 표 (2026-10-02 추가) ---------------------------------
# 학습군 양성률·검증군 양성/음성은 처리 데이터에서 센 값이라 결과 CSV 에 없다.
# make_report 와 같은 방식으로 다시 세서 대조한다.
_PF = {"DILIrank": "dilirank2_with_year.csv", "DICTrank": "dictrank_processed.csv"}
for d in DS:
    x = pd.read_csv(os.path.join(ROOT, "data", "processed", _PF[d])).dropna(
        subset=["approval_year", "y", "smiles"]
    )
    tr = x[x.approval_year.astype(int) < 2010]
    te = x[x.approval_year.astype(int) >= 2010]
    cell(f4(tr.y.mean()), "§2.3 %s 학습군 양성률" % d)
    cell(f4(te.y.mean()), "§2.3 %s 검증군 양성률" % d)
    cell(
        "%d / %d" % (int((te.y == 1).sum()), int((te.y == 0).sum())), "§2.3 %s 검증군 양성/음성" % d
    )
    cell(
        (
            "| %d | %d |"
            % (len(tr), int(bc1[(bc1.dataset == d) & (bc1.split == "temporal")].iloc[0].n_test))
            if False
            else "%d" % len(tr)
        ),
        "§2.3 %s 학습군 n" % d,
    )

# ---- §10.2 사전확률 추정 (2026-10-02) ----------------------------------
pe = cv("prior_estimation.csv")
for d in DS:
    g = pe[(pe.dataset == d) & (pe.estimator.str.startswith("P"))]
    for _, r in g.iterrows():
        rowstr(
            [
                r.estimator,
                r.info_used.replace("**", ""),
                f4(r.pi_hat),
                sg(r.pi_err),
                f4(r.ECE),
                p1(r.recovery),
            ],
            "§10.2 %s %s" % (d, r.estimator),
        )
    b = pe[(pe.dataset == d) & (pe.estimator.str.startswith("(기준)"))]
    for _, r in b.iterrows():
        cell(f4(r.ECE), "§10.2 %s %s ECE" % (d, r.estimator))

# ---- §12.2 역방향 시간분할 (2026-10-02) --------------------------------
rv = cv("reverse_split.csv")
NTR = {}
bc10 = cv("baseline_cv_per_fold.csv")
for d in DS:
    t = bc10[(bc10.dataset == d) & (bc10.split == "temporal")].iloc[0]
    NTR[(d, "정방향(참고)")] = int(t.n_train)
    NTR[(d, "역방향")] = int(t.n_test)
for d in DS:
    for _, r in rv[rv.dataset == d].iterrows():
        rat = "—" if pd.isna(r.ece_ratio) else "%s배" % f2(r.ece_ratio)
        sh = "—" if pd.isna(r.share_rel) else p1(r.share_rel)
        nv = "—" if pd.isna(r.n_rep_valid) else "%d/200" % int(r.n_rep_valid)
        pv = "—" if pd.isna(r.p_phipson_smyth) else f4(r.p_phipson_smyth)
        rowstr(
            [
                d,
                r.direction,
                r.setting,
                "%d" % NTR[(d, r.direction)],
                "%d" % r.n_test,
                f4(r.auroc),
                f4(r.ece),
                f4(r.slope),
                rat,
                sh,
                nv,
                pv,
            ],
            "§12.2 %s %s %s" % (d, r.direction, r.setting),
        )

# ---- §13.5 이동량 대조 분할 (2026-10-02) -------------------------------
sm = cv("shift_matched_split.csv")
for d in DS:
    for _, r in sm[sm.dataset == d].iterrows():
        rowstr(
            [
                d,
                r.split,
                "%d" % r.n_test,
                f4(r.pos_rate_train),
                f4(r.pos_rate_test),
                sg(r.pos_rate_test - r.pos_rate_train),
                f4(r.shift_auc),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ctrl_ece),
                "%s배" % f2(r.ece_ratio),
                p1(r.share_rel),
                f4(r.p_phipson_smyth),
            ],
            "§13.5 %s %s" % (d, r.split),
        )


# ---- §17.3 공유 분자 상관 (2026-10-09) ---------------------------------
def f3(x):
    return "%.3f" % x


cso = cv("concept_shift_overlap.csv")
_QN = {
    "\uc794\ucc28 y-p (\ubd80\ud638 \uc788\uc74c)": "\uc794\ucc28 y \u2212 p (\ubd80\ud638 \uc788\uc74c)",
    "\uc808\ub300\uc794\ucc28 |y-p|": "\uc808\ub300\uc794\ucc28 abs(y \u2212 p)",
    "\ub77c\ubca8 y (phi)": "\ub77c\ubca8 y (phi \uacc4\uc218)",
    "\uc608\uce21 p": "\uc608\uce21\ud655\ub960 p",
}
for _, r in cso.iterrows():
    rowstr(
        [
            _QN[str(r.quantity)],
            "%d" % int(r.n),
            sg(r.pearson),
            "[%s, %s]" % (f3(r.ci_lo), f3(r.ci_hi)),
            sg(r.spearman),
        ],
        "\u00a717.3 %s" % r.quantity,
    )

# ---- §18 개념 이동 (2026-10-09) ----------------------------------------
cs = cv("concept_shift.csv")
_AL = {
    "A.": "A. \uc2dc\ub300\uc77c\uce58 (\ubb34\uc791\uc704 5-fold)",
    "A2": "A2. \uc2dc\ub300\uc77c\uce58 (\uc2a4\uce90\ud3f4\ub4dc 5-fold)",
    "B.": "B. \uc815\ubc29\ud5a5 \uc804\uccb4 (\u00a78 \uc124\uc815)",
    "C.": "C. \uc815\ubc29\ud5a5 \ud06c\uae30\ub9de\ucda4",
    "D.": "D. \uc815\ubc29\ud5a5 \ud06c\uae30\u00b7\uc720\ubcd1\ub960\ub9de\ucda4",
}
CS = {}
for _, r in cs.iterrows():
    k = "A2" if str(r.arm).startswith("A2") else str(r.arm)[:2]
    CS[(r.dataset, k)] = r
    rowstr(
        [
            r.dataset,
            _AL[k],
            "post-2010" if k in ("A.", "A2") else "pre-2010",
            "%d" % int(round(r.train_n)),
            f4(r.train_rate),
            f4(r.auroc),
            f4(r.ece),
            f4(r.slope),
            f4(r.rel),
            "\u2014" if r.ece_ratio != r.ece_ratio else "%s\ubc30" % f2(r.ece_ratio),
            "\u2014" if r.p_phipson_smyth != r.p_phipson_smyth else f4(r.p_phipson_smyth),
        ],
        "\u00a718 %s %s" % (r.dataset, k),
    )

# §18 내부 정합성 표: 팔 B 가 §8 시간분할을 재현하는지 — 두 CSV 를 직접 맞댄다
_cal = cv("calibration_results.csv")
for d in DS:
    c8 = _cal[(_cal.dataset.str.startswith(d)) & (_cal.setting == "temporal")].iloc[0]
    for lab, a, b in (
        ("AUROC", c8.AUROC, CS[(d, "B.")].auroc),
        ("ECE", c8.ECE, CS[(d, "B.")].ece),
        ("\ubcf4\uc815 \uae30\uc6b8\uae30", c8.calib_slope, CS[(d, "B.")].slope),
    ):
        assert f4(a) == f4(b), "\u00a718 \uc815\ud569\uc131 %s %s: %s != %s" % (
            d,
            lab,
            f4(a),
            f4(b),
        )
        rowstr([d, lab, f4(a), f4(b), "\uc608"], "\u00a718 \uc815\ud569\uc131 %s %s" % (d, lab))


# §18 3분해: 합이 B-A 와 같은지 재계산으로 확인하고 표 행도 대조한다
def _pct1(x):
    return "%.1f%%" % x


for d in DS:
    era = CS[(d, "D.")].ece - CS[(d, "A.")].ece
    rate = CS[(d, "C.")].ece - CS[(d, "D.")].ece
    siz = CS[(d, "B.")].ece - CS[(d, "C.")].ece
    tot = CS[(d, "B.")].ece - CS[(d, "A.")].ece
    assert abs(era + rate + siz - tot) < 1e-9, "\u00a718 3\ubd84\ud574 \ud569 %s" % d
    rowstr(
        [
            d,
            "%s (%s)" % (sg(era), _pct1(era / tot * 100)),
            "%s (%s)" % (sg(rate), _pct1(rate / tot * 100)),
            "%s (%s)" % (sg(siz), _pct1(siz / tot * 100)),
            sg(tot),
        ],
        "\u00a718 3\ubd84\ud574 %s" % d,
    )

# ---- §19 유병률 스윕 (2026-10-09) --------------------------------------
sw = cv("prevalence_sweep.csv")
for d in DS:
    for _, r in sw[sw.dataset == d].sort_values("level").iterrows():
        mk = (
            " (\uac80\uc99d\uad70 \uc591\uc131\ub960)"
            if bool(r.is_test_rate)
            else (" (pre-2010 \uc591\uc131\ub960)" if bool(r.is_pre2010_rate) else "")
        )
        gp = 0.0 if abs(float(r.gap)) < 5e-5 else float(r.gap)
        rowstr(
            [
                d,
                f4(r.level) + mk,
                sg(gp),
                "%d" % int(r.train_n),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ece_sd),
                f4(r.slope),
                f4(r.rel),
            ],
            "\u00a719 %s %s" % (d, f4(r.level)),
        )

swf = cv("prevalence_sweep_fit.csv")
for _, r in swf.iterrows():
    mg = 0.0 if abs(float(r.min_gap)) < 5e-5 else float(r.min_gap)
    rowstr(
        [
            r.dataset,
            sg(mg),
            f4(r.min_ece),
            f4(r["slope_per_0.10"]),
            "[%s, %s]" % (f3(r.slope_lo), f3(r.slope_hi)),
            f4(r["slope_neg_side_per_0.10"]),
            f4(r["slope_pos_side_per_0.10"]),
        ],
        "\u00a719 \uc801\ud569 %s" % r.dataset,
    )
    rowstr(
        [
            r.dataset,
            f4(r.ece_at_gap0),
            sg(r.gap_obs),
            f4(r.ece_at_obs_gap),
            sg(float(r.ece_at_obs_gap) - float(r.ece_at_gap0)),
            "%s\ubc30" % f2(float(r.ece_at_obs_gap) / float(r.ece_at_gap0)),
        ],
        "\u00a719 \ubcf4\uac04 %s" % r.dataset,
    )


# ---- §20 최신성 가중 (2026-10-10) --------------------------------------
def f1(x):
    return "%.1f" % x


rw = cv("recency_weighting.csv")
for _, r in rw[rw.kind.isin(["weight", "window"])].iterrows():
    # 보고서 §20 은 데이터셋을 열이 아니라 소제목으로 쓴다 → 첫 칸 없음
    rowstr(
        [
            str(r.tag),
            "%d" % int(r.n_train),
            f1(r.n_eff),
            f4(r.train_rate_weighted),
            sg(r.gap),
            f4(r.auroc),
            f4(r.ece),
            "[%s, %s]" % (f4(r.ece_lo), f4(r.ece_hi)),
            f4(r.slope),
            f4(r.rel),
            f4(r.res),
        ],
        "\u00a720 %s %s" % (r.dataset, r.tag),
    )

# §20.1 균등 대 정직선택 요약행
for d in DS:
    b = rw[(rw.dataset == d) & (rw.kind == "weight") & (rw.tau.isna())].iloc[0]
    h = rw[(rw.dataset == d) & (rw.kind == "honest")].iloc[0]
    g = rw[(rw.dataset == d) & (rw.kind == "weight")]
    bs = g.loc[g.ece.idxmin()]
    dd = float(h.ece) - float(b.ece)
    rowstr(
        [
            d,
            f4(b.ece),
            "\ubc18\uac10\uae30 %g\ub144" % float(h.tau) if h.tau == h.tau else "\uade0\ub4f1",
            f4(h.ece),
            "%s (%s %.1f%%)"
            % (sg(dd), "\uc545\ud654" if dd > 0 else "\uac1c\uc120", abs(dd) / float(b.ece) * 100),
            "%s (%s)" % (f4(bs.ece), bs.tag),
        ],
        "\u00a720.1 %s" % d,
    )

# §20.2 예측 검정: §19 기울기로 ΔECE 를 재계산해 표와 맞댄다
_swf2 = cv("prevalence_sweep_fit.csv")
_SW2 = {r.dataset: r for _, r in _swf2.iterrows()}
for d in DS:
    sl = float(_SW2[d]["slope_per_0.10"]) / 0.10
    g = rw[(rw.dataset == d) & (rw.kind == "weight")]
    b = g[g.tau.isna()].iloc[0]
    for _, r in g[g.tau.notna()].iterrows():
        dg = abs(float(r.gap)) - abs(float(b.gap))
        pr = sl * dg
        ob = float(r.ece) - float(b.ece)
        rowstr(
            [d, str(r.tag), sg(r.gap), sg(dg), sg(pr), sg(ob), sg(ob - pr)],
            "\u00a720.2 %s %s" % (d, r.tag),
        )


# ---- §21 적용범위 (2026-10-17) -----------------------------------------
ad = cv("applicability_domain.csv")
adf = cv("applicability_domain_filter.csv")
_strat = ad[~ad.stratum.str.contains("\uae30\uc6b8\uae30")]
for _, r in _strat.iterrows():
    if bool(r.underpowered):
        rowstr(
            [
                str(r.stratum),
                "%d" % int(r.n),
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\u2014",
                "\ud45c\ubcf8\ubd80\uc871",
            ],
            "\u00a721 %s %s" % (r.dataset, r.stratum),
        )
        continue
    rowstr(
        [
            str(r.stratum),
            "%d" % int(r.n),
            f4(r.pos_rate),
            f4(r.auroc),
            f4(r.ece),
            sg(r.mean_resid),
            f4(r.ctrl_ece),
            "%s\ubc30" % f2(r.ece_ratio),
            "[%s, %s]" % (f2(r.ratio_lo), f2(r.ratio_hi)),
            f4(r.p_phipson_smyth),
        ],
        "\u00a721 %s %s" % (r.dataset, r.stratum),
    )

# §21.1 기울기 두 줄 — 보정과 판별
for d in DS:
    for lab, key, nd in (
        (
            "\ubcf4\uc815 (ECE \ubc30\uc728)",
            "\uc720\uc0ac\ub3c4 \uae30\uc6b8\uae30(\ud558\uc704\u2212\uc0c1\uc704)",
            2,
        ),
        (
            "\ud310\ubcc4 (AUROC)",
            "\ud310\ubcc4 \uae30\uc6b8\uae30(\uc0c1\uc704\u2212\ud558\uc704 AUROC)",
            4,
        ),
    ):
        g = ad[(ad.dataset == d) & (ad.stratum == key)]
        if not len(g):
            continue
        r = g.iloc[0]
        fm = (lambda x: "%.2f" % x) if nd == 2 else f4
        inc = bool(r.ratio_lo <= 0.0 <= r.ratio_hi)
        rowstr(
            [
                d,
                lab,
                fm(r.ece_ratio),
                "[%s, %s]" % (fm(r.ratio_lo), fm(r.ratio_hi)),
                # FLAT 은 문서에서 ** 를 지운 사본이므로 기대값에도 쓰지 않는다
                "\uc608 \u2014 \uc8fc\uc7a5 \ubd88\uac00" if inc else "\uc544\ub2c8\uc624",
            ],
            "\u00a721.1 %s %s" % (d, lab),
        )


# §21.2 AD 필터
def _p0(x):
    return "%.1f%%" % x  # 보고서는 pct1 로 소수 1자리를 쓴다


def _f3(x):
    return "%.3f" % x


for _, r in adf.iterrows():
    rowstr(
        [
            _p0(r.drop_frac * 100),
            _f3(r.coverage),
            "%d" % int(r.n_keep),
            f4(r.ece_ad),
            f4(r.auroc_ad),
            f4(r.ece_random_drop),
            sg(r.delta),
        ],
        "\u00a721.2 %s %.0f%%" % (r.dataset, r.drop_frac * 100),
    )


# ---- §15 다중비교 보정 (2026-10-19 추가) --------------------------------
# Major 2 의 핵심이 이 표의 q 값이다(DICTrank R3 가 유의성을 잃는 지점).
# 그런데 셀 단위 대조에서 빠져 있었다. 전수 대조를 추가한다.
mc = cv("multiple_comparison_correction.csv")
for fam in ("1_primary", "2_prevalence_control", "3_exploratory_subgroup"):
    g = mc[mc.family == fam]
    for _, r in g.sort_values("p").iterrows():
        # make_report.py 와 같은 라벨 치환을 적용한다
        _tn = str(r.test).replace(
            "ATC L(\ud56d\uc885\uc591)", "ATC L(\ud56d\uc885\uc591\u00b7\uba74\uc5ed\uc870\uc808)"
        )
        rowstr(
            [str(r.dataset), _tn, f4(r.p), f4(r.q_BH)],
            "\u00a715 %s %s %s" % (fam, r.dataset, str(r.test)[:24]),
        )
# 가족별 유의 건수도 대조한다 (제목 줄에 박혀 있다)
for fam, lab in (
    ("1_primary", "1\uad70"),
    ("2_prevalence_control", "2\uad70"),
    ("3_exploratory_subgroup", "3\uad70"),
):
    g = mc[mc.family == fam]
    n_sig, n_all = int(g.significant_after_FDR.sum()), len(g)
    cell(
        "(n = %d) \u2192 %d/%d \uc720\uc758" % (n_all, n_sig, n_all),
        "\u00a715 %s \uc720\uc758\uac74\uc218" % fam,
    )

# ---- §13 공변량 이동 정량 (2026-10-19 추가) -----------------------------
sq = cv("shift_quantification.csv")
for _, r in sq.iterrows():
    cell(
        f4(r.random_split_era_auc),
        "\u00a713 %s \ubb34\uc791\uc704\ubd84\ud560 \uc2dc\ub300AUC" % r.dataset,
    )
    cell(
        f4(r.era_auc_desc_only), "\u00a713 %s \uae30\uc220\uc790\ub9cc \uc2dc\ub300AUC" % r.dataset
    )


# ---- §22 결정곡선 / §23 클래스 가중 / §24 라벨 역치 (2026-10-19) --------
dc = cv("decision_curve.csv")
for d in DS:
    g = dc[dc.dataset == d]
    arms = [a for a in g.arm.unique() if not a.startswith("R.")]
    for pt in sorted(g.p_t.unique()):
        cells = []
        for a in arms:
            v = g[(g.arm == a) & (g.p_t == pt)]
            cells.append(sg(v.net_benefit.iloc[0]) if len(v) else "\u2014")
        na = g[(g.arm == arms[0]) & (g.p_t == pt)].nb_treat_all.iloc[0]
        rowstr([f2(pt), sg(na)] + cells, "\u00a722 %s p_t %.2f" % (d, pt))

cw = cv("class_weighting.csv")
for _, r in cw.iterrows():
    rowstr(
        [
            str(r.arm),
            "\u2014" if r.pi_target != r.pi_target else f4(r.pi_target),
            "\u2014" if r.scale_pos_weight != r.scale_pos_weight else "%.3f" % r.scale_pos_weight,
            f4(r.auroc),
            f4(r.ece),
            "[%s, %s]" % (f4(r.ece_lo), f4(r.ece_hi)),
            f4(r.slope),
            f4(r.res),
        ],
        "\u00a723 %s %s" % (r.dataset, r.arm),
    )

lt = cv("label_threshold.csv")
for _, r in lt[lt.setting.str.startswith("T")].iterrows():
    rowstr(
        [
            str(r.dataset),
            str(r.label_def),
            "%d" % int(r.n_test_pos),
            f4(r.rate_train),
            f4(r.rate_test),
            sg(r.rate_test - r.rate_train),
            f4(r.ece),
        ],
        "\u00a724 \uae30\ubcf8 %s %s" % (r.dataset, r.label_def),
    )
for d in DS:
    for ld in lt[lt.dataset == d].label_def.unique():
        g = lt[(lt.dataset == d) & (lt.label_def == ld)]
        rr = {b: g[g.setting.str.startswith(b)] for b in ("R1", "R2", "R3")}
        if any(not len(v) for v in rr.values()):
            continue
        v = [float(rr[b].ece_ratio.iloc[0]) for b in ("R1", "R2", "R3")]
        mono = v[0] > v[1] > v[2]
        rowstr(
            [
                d,
                ld,
                f2(v[0]),
                f2(v[1]),
                f2(v[2]),
                f4(rr["R1"].p_phipson_smyth.iloc[0]),
                f4(rr["R3"].p_phipson_smyth.iloc[0]),
                "\uc608" if all(x > 1 for x in v) else "\uc544\ub2c8\uc624",
                "\uc608" if mono else "\uc544\ub2c8\uc624",
            ],
            "\u00a724 \ube44\uc728 %s %s" % (d, ld),
        )

# ---------------------------------------------------------------- §9.7 절단별 Brier
bdc = cv("brier_decomposition_cutoff.csv")
for d in DS:
    for _, r in bdc[bdc.dataset == d].sort_values("cutoff").iterrows():
        rowstr(
            [
                d,
                "%d" % int(r.cutoff),
                "%d" % int(r.n_test),
                "%s → %s" % (f4(r.reliability_random_mean), f4(r.reliability_temporal)),
                "%s → %s" % (f4(r.resolution_random_mean), f4(r.resolution_temporal)),
                "%s → %s" % (f4(r.brier_random_mean), f4(r.brier_temporal)),
                p1(r.pct_from_reliability),
            ],
            "§9.7 %s 절단 %d" % (d, int(r.cutoff)),
        )

# ---------------------------------------------------------------- §13.1 보조 순열
mep = cv("matched_n_era_permutation.csv")
for d in DS:
    r = mep[mep.dataset == d].iloc[0]
    cell(f4(r.era_auc), "§13.1 보조 %s 시대 AUC" % d)
    cell(f4(r.null_mean), "§13.1 보조 %s 귀무 평균" % d)
    cell(f2(r.z), "§13.1 보조 %s z" % d)

# ---------------------------------------------------------------- §25 상수 예측기
tb = cv("trivial_baseline.csv")
_LAB = {
    "M.": "모델 (XGBoost)",
    "C0.": "C0. 상수 = 학습군 양성률",
    "C1.": "C1. 상수 = 0.5",
    "C2.": "C2. 상수 = 검증군 양성률 (oracle)",
}
for d in DS:
    for pref in ("M.", "C0.", "C1.", "C2."):
        t = tb[
            (tb.dataset == d) & (tb.split == "시간분할") & (tb.predictor.str.startswith(pref))
        ].iloc[0]
        r = tb[
            (tb.dataset == d) & (tb.split == "무작위분할") & (tb.predictor.str.startswith(pref))
        ].iloc[0]
        rowstr(
            [_LAB[pref], f4(t.brier), sg(t.bss_vs_C0), f4(t.ece), f4(r.brier), sg(r.bss_vs_C0)],
            "§25 %s %s" % (d, pref),
        )
    m = tb[(tb.dataset == d) & (tb.split == "시간분할") & (tb.predictor.str.startswith("M."))].iloc[
        0
    ]
    cell(f4(m.auroc), "§25 %s 시간분할 AUROC" % d)
    cell(f4(m.rel), "§25 %s reliability" % d)
    cell(f4(m.res), "§25 %s resolution" % d)

# ---------------------------------------------------------------- §25.1 재보정 BSS
rec25 = cv("recalibration_results.csv")
_c0 = {
    d: float(
        tb[
            (tb.dataset == d) & (tb.split == "시간분할") & (tb.predictor.str.startswith("C0."))
        ].brier.iloc[0]
    )
    for d in DS
}
for d in DS:
    for _, r in rec25[rec25.dataset.str.startswith(d)].iterrows():
        _bss = 1.0 - float(r.Brier) / _c0[d]
        rowstr(
            [d, str(r.method), f4(r.Brier), sg(_bss), "예" if _bss > 0 else "아니오"],
            "§25.1 %s %s" % (d, str(r.method)[:2]),
        )

# ---------------------------------------------------------------- §26 배율 신뢰구간
ri = cv("ratio_intervals.csv")
for d in DS:
    for _, r in ri[ri.dataset == d].iterrows():
        rowstr(
            [
                d,
                str(r.control),
                f4(r.ece_control_mean),
                f2(r.ece_ratio),
                "[%s, %s]" % (f2(r.ratio_lo), f2(r.ratio_hi)),
                "[%s, %s]" % (f2(r.ratio_lo_numonly), f2(r.ratio_hi_numonly)),
                "[%s, %s]" % (sg(r.diff_lo), sg(r.diff_hi)),
                "예" if bool(r.excludes_one) else "아니오",
            ],
            "§26 %s %s" % (d, str(r.control)[:2]),
        )
    _r3 = ri[(ri.dataset == d) & (ri.control.str.startswith("R3"))].iloc[0]
    cell("%s [%s, %s]" % (f2(_r3.ece_ratio), f2(_r3.ratio_lo), f2(_r3.ratio_hi)), "§1 17번 %s" % d)

# ---------------------------------------------------------------- §1 16번
rowstr(
    [
        "16",
        "시간분할 Brier, 모델 대 상수(학습군 양성률) — BSS",
        sg(
            tb[
                (tb.dataset == "DILIrank")
                & (tb.split == "시간분할")
                & (tb.predictor.str.startswith("M."))
            ].bss_vs_C0.iloc[0]
        ),
        sg(
            tb[
                (tb.dataset == "DICTrank")
                & (tb.split == "시간분할")
                & (tb.predictor.str.startswith("M."))
            ].bss_vs_C0.iloc[0]
        ),
        "§25",
    ],
    "§1 16번 행",
)

# ---------------------------------------------------------------- §27 지표 비의존성
cm27 = cv("calibration_metrics.csv")
for d in DS:
    for _, r in cm27[cm27.dataset == d].iterrows():
        rowstr(
            [
                str(r.arm),
                "%d" % int(round(r.n)),
                f4(r.ece),
                f4(r.ece_debiased),
                f4(r.ici),
                f4(r.e50),
                f4(r.e90),
                f4(r.emax),
                f2(r.sh_z),
                sg(r.cal_large),
                f4(r.auroc),
            ],
            "§27 %s %s" % (d, str(r.arm)[:2]),
        )
cmr27 = cv("calibration_metrics_ratio.csv")
for d in DS:
    for _, r in cmr27[cmr27.dataset == d].iterrows():
        rowstr(
            [
                d,
                str(r.control),
                f2(r.ece_ratio),
                f2(r.ece_debiased_ratio),
                f2(r.ici_ratio),
                f2(r.e90_ratio),
            ],
            "§27.1 %s %s" % (d, str(r.control)[:2]),
        )
    _r3 = cmr27[(cmr27.dataset == d) & (cmr27.control.str.startswith("R3"))].iloc[0]
    cell("%s배" % f2(_r3.ece_debiased_ratio), "§1 18번 %s" % d)
    _t = cm27[(cm27.dataset == d) & (cm27.arm.str.startswith("T"))].iloc[0]
    cell(f4(_t.ece_null_p975), "§27 %s 귀무 97.5" % d)
    cell(f4(_t.ece_null_mean), "§27 %s 귀무 평균" % d)

# ---------------------------------------------------------------- §28 하이퍼파라미터
hp28 = cv("hparam_sensitivity.csv")
hpc28 = cv("hparam_sensitivity_ctrl.csv")
for d in DS:
    g = hp28[hp28.dataset == d]
    f = g[g.is_fixed].iloc[0]
    for crit, r in (
        ("내부 log loss", g.loc[g.inner_logloss.idxmin()]),
        ("내부 ECE", g.loc[g.inner_ece.idxmin()]),
        ("내부 AUROC", g.loc[g.inner_auroc.idxmax()]),
    ):
        rowstr(
            [
                d,
                crit,
                "`%s`" % r.config,
                f4(r.temporal_ece),
                f4(r.temporal_auroc),
                sg(r.temporal_ece - f.temporal_ece),
            ],
            "§28.1 %s %s" % (d, crit),
        )
    rowstr(
        [d, "(고정, 논문)", "`%s`" % f.config, f4(f.temporal_ece), f4(f.temporal_auroc), "—"],
        "§28.1 %s 고정" % d,
    )
    for _, r in hpc28[hpc28.dataset == d].iterrows():
        rowstr(
            [
                d,
                str(r.arm),
                "`%s`" % r.config,
                f4(r.temporal_ece),
                f4(r.ctrl_R1_ece),
                f2(r.ratio_R1),
                f4(r.ctrl_R3_ece),
                f2(r.ratio_R3),
            ],
            "§28.2 %s %s" % (d, r.arm),
        )
for col, lab in (
    ("max_depth", "트리 깊이"),
    ("n_estimators", "나무 수"),
    ("learning_rate", "학습률"),
    ("min_child_weight", "min_child_weight"),
):
    for v in sorted(hp28[col].unique()):
        rowstr(
            [
                lab,
                "%g" % v,
                f4(hp28[(hp28.dataset == DS[0]) & (hp28[col] == v)].temporal_ece.mean()),
                f4(hp28[(hp28.dataset == DS[1]) & (hp28[col] == v)].temporal_ece.mean()),
            ],
            "§28.3 %s %g" % (col, v),
        )

# ---------------------------------------------------------------- §29 처방 검정력
rp29 = cv("remedy_power.csv")
for d in DS:
    for _, r in rp29[rp29.dataset == d].iterrows():
        rowstr(
            [
                str(r.remedy),
                str(r.deployable),
                f4(r.ece_remedy),
                sg(r.delta),
                "[%s, %s]" % (sg(r.delta_lo), sg(r.delta_hi)),
                f4(r.mde80),
                p1(100 * r.mde_over_gap),
                str(r.verdict).split(" ")[0],
            ],
            "§29 %s %s" % (d, str(r.remedy)[:2]),
        )
    cell(f4(rp29[rp29.dataset == d].gap_to_control.iloc[0]), "§29 %s 간격" % d)
rpa29 = cv("remedy_power_ad.csv")
for d in DS:
    for _, r in rpa29[rpa29.dataset == d].iterrows():
        rowstr(
            [
                d,
                f2(r.drop_frac),
                "%d" % int(r.n_keep),
                f4(r.ece_ad),
                f4(r.ece_random_drop),
                sg(r.delta),
                "[%s, %s]" % (sg(r.delta_lo), sg(r.delta_hi)),
                f4(r.mde80),
            ],
            "§29.3 %s %.1f" % (d, r.drop_frac),
        )

# ---------------------------------------------------------------- §30 공변량이동 가중
csw30 = cv("covariate_shift_weighting.csv")
for d in DS:
    for _, r in csw30[(csw30.dataset == d) & (csw30.kind == "outer")].iterrows():
        rowstr(
            [
                str(r.arm),
                "—" if r.cap != r.cap else ("%g" % r.cap),
                "%.1f" % r.n_eff,
                f2(r.w_max),
                f4(r.auroc),
                f4(r.ece),
                f4(r.slope),
                f4(r.res),
                f4(r.mean_pred),
            ],
            "§30 %s %s" % (d, str(r.arm)[:6]),
        )
cswc30 = cv("covariate_shift_weighting_ctrl.csv")
for d in DS:
    for _, r in cswc30[cswc30.dataset == d].iterrows():
        rowstr(
            [d, str(r.control), f4(r.ctrl_ece), f2(r.ratio), f4(r.ctrl_era_auc)],
            "§30.2 %s %s" % (d, r.control),
        )

# ---------------------------------------------------------------- §31 격자 R3
g331 = cv("grid_r3.csv")
for d in DS:
    for _, r in g331[g331.dataset == d].iterrows():
        rowstr(
            [
                str(r.rep),
                str(r.model),
                f4(r.ece_temporal),
                f4(r.r3_ece_random),
                f2(r.r3_ece_ratio),
                sg(r.r3_d_reliability),
                sg(r.r3_d_resolution),
            ],
            "§31 %s %s/%s" % (d, r.rep, r.model),
        )
for m, g in g331.groupby("model"):
    rowstr(["모델 %s" % m, f2(g.r3_ece_ratio.median())], "§31 모델 %s" % m)
for m, g in g331.groupby("rep"):
    rowstr(["표현 %s" % m, f2(g.r3_ece_ratio.median())], "§31 표현 %s" % m)

# ---------------------------------------------------------------- §32 지평별 배율
hz32 = cv("horizon_ratio.csv")
for d in DS:
    g = hz32[hz32.dataset == d].sort_values("year_from")
    for _, r in g.iterrows():
        rowstr(
            [
                str(r.window),
                "%d" % int(r.n),
                f4(r.pos_rate),
                sg(r.gap_vs_train),
                f4(r.auroc),
                f4(r.ece),
                f2(r.r1_ratio),
                f2(r.r2_ratio),
                f2(r.r3_ratio),
                "[%s, %s]" % (f2(r.r3_ratio_lo), f2(r.r3_ratio_hi)),
            ],
            "§32 %s %s" % (d, r.window),
        )
    rowstr(
        [
            d,
            " → ".join(f4(v) for v in g.ece),
            " → ".join(f4(v) for v in g.pos_rate),
            " → ".join(f2(v) for v in g.r1_ratio),
            " → ".join(f2(v) for v in g.r3_ratio),
            sg(g.r3_ratio.iloc[-1] - g.r3_ratio.iloc[0]),
            "%.3f"
            % (
                0.5
                * (
                    (g.r3_ratio_hi.iloc[0] - g.r3_ratio_lo.iloc[0])
                    + (g.r3_ratio_hi.iloc[-1] - g.r3_ratio_lo.iloc[-1])
                )
                / 2
            ),
        ],
        "§32.1 %s 추세" % d,
    )

# ---------------------------------------------------------------- §33 이동량 맞춤
sm233 = cv("shift_matched_v2.csv")
sm2m33 = cv("shift_matched_v2_matched.csv")
for d in DS:
    for _, r in sm2m33[sm2m33.dataset == d].sort_values("axis").iterrows():
        rowstr(
            [
                d,
                str(r.axis),
                f2(r.alpha_star),
                "%s (%s)" % (f4(r.shift_auc_matched), f4(r.shift_auc_temporal)),
                f4(r.ece_matched),
                f4(r.ece_matched_sd),
                f4(r.ece_temporal),
                sg(r.ece_diff),
                "%.1f" % r.diff_se_units,
                f2(r.ece_ratio_T_over_matched),
                str(r.verdict).split(" (")[0],
            ],
            "§33.2 %s %s" % (d, r.axis),
        )
    for ax in sorted(sm233[sm233.dataset == d].axis.unique()):
        g = sm233[(sm233.dataset == d) & (sm233.axis == ax)].set_index("alpha_num")
        cells = [
            f4(g.shift_auc.loc[al]) if al in g.index else "—"
            for al in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        rowstr(
            [d, ax] + cells + [f4(sm233[sm233.dataset == d].shift_auc_temporal.iloc[0])],
            "§33.1 %s %s" % (d, ax),
        )

# ---------------------------------------------------------------- §29.5 기준 스윕
uts29 = cv("useful_threshold_sweep.csv")
for _, r in uts29.iterrows():
    rowstr(
        [
            str(r.dataset),
            str(r.remedy),
            str(r.deployable),
            "기준 무관" if bool(r.u_independent) else "%.3f" % r.u_star,
            "(%s)" % r.verdict_at_ref,
        ],
        "§29.5 %s %s" % (r.dataset, str(r.remedy)[:2]),
    )
utg29 = cv("useful_threshold_grid.csv")
for _, r in utg29.iterrows():
    # 본문 기준 행에는 " ← 본문" 표시가 붙는다
    _mk = " ← 본문" if abs(float(r.useful) - 0.25) < 1e-9 else ""
    rowstr(
        [
            "%.0f%%" % (r.useful * 100),
            "%d" % int(r.n_ga),
            "%d" % int(r.n_ga_bad),
            "%d" % int(r.n_na),
            "%d%s" % (int(r.n_da), _mk),
        ],
        "§29.5 기준 %.0f%%" % (r.useful * 100),
    )

# ---------------------------------------------------------------- §10.3 추정량 9종
pe2_10 = cv("prior_estimation_v2.csv")
for d in DS:
    for _, r in pe2_10[pe2_10.dataset == d].iterrows():
        if r.pi_hat != r.pi_hat:
            continue
        # FLAT 은 ** 를 지우므로 기대값에서도 지운다 (oracle 행의 info_used)
        rowstr(
            [
                str(r.estimator),
                str(r.info_used).replace("**", ""),
                f4(r.pi_hat),
                sg(r.pi_err),
                "[%s, %s]" % (f4(r.pi_lo), f4(r.pi_hi)),
                "예" if bool(r.covers_true) else "아니오",
                f4(r.ece_corrected),
                p1(r.recovery),
            ],
            "§10.3 %s %s" % (d, str(r.estimator)[:3]),
        )

# ---------------------------------------------------------------- §10.4 임의성 제거
tba10 = cv("tiebreak_averaged.csv")
rec10c = cv("recalibration_results.csv")
conf10c = cv("conformal_prediction.csv")


def _r10row(d, pref):
    g = rec10c[rec10c.dataset.str.startswith(d)]
    return g[g.method.str.startswith(pref)].iloc[0]


def _r10rec(d, pref):
    a, e, x = _r10row(d, "A."), _r10row(d, "E."), _r10row(d, pref)
    return (a.ECE - x.ECE) / (a.ECE - e.ECE) * 100


def _r10cf(d, al):
    q = conf10c[(conf10c.dataset == d) & ((conf10c.alpha - al).abs() < 1e-9)]
    return float(q.marginal_coverage_temporal.iloc[0])


_TBA_N = int(tba10.groupby("dataset").size().min())
for d in DS:
    g = tba10[tba10.dataset == d]
    QTY = [
        ("A 무보정 ECE", "ece_A", float(_r10row(d, "A.").ECE), f4),
        ("A' 보정셋 제외 ECE", "ece_Ap", float(_r10row(d, "A'.").ECE), f4),
        ("B Platt ECE", "ece_B", float(_r10row(d, "B.").ECE), f4),
        ("C Isotonic ECE", "ece_C", float(_r10row(d, "C.").ECE), f4),
        ("D 사전확률 ECE", "ece_D", float(_r10row(d, "D.").ECE), f4),
        ("E Oracle 상한 ECE", "ece_E", float(_r10row(d, "E.").ECE), f4),
        ("B 회복률 (%)", "rec_B", _r10rec(d, "B."), f2),
        ("C 회복률 (%)", "rec_C", _r10rec(d, "C."), f2),
        ("D 회복률 (%)", "rec_D", _r10rec(d, "D."), f2),
        ("conformal 90% 커버리지", "conf90_marginal", _r10cf(d, 0.1), f4),
        ("conformal 80% 커버리지", "conf80_marginal", _r10cf(d, 0.2), f4),
        ("AUROC (무보정)", "auroc_A", float(_r10row(d, "A.").AUROC), f4),
    ]
    for lab, col, ref, fmt in QTY:
        m, sd = float(g[col].mean()), float(g[col].std(ddof=1))
        lo = float(g[col].quantile(0.025))
        hi = float(g[col].quantile(0.975))
        mn, mx = float(g[col].min()), float(g[col].max())
        rowstr(
            [
                d,
                lab,
                fmt(m),
                fmt(sd),
                "%s–%s" % (fmt(lo), fmt(hi)),
                "%s–%s" % (fmt(mn), fmt(mx)),
                fmt(ref),
                "%+.1f SD" % ((ref - m) / sd),
            ],
            "§10.4 %s %s" % (d, col),
        )
    # 결론 점검 횟수
    cell("%d/%d" % (int((g.conf90_marginal < 0.90).sum()), _TBA_N), "§10.4 %s C1" % d)
    cell("%d/%d" % (int((g.rec_B < 100).sum()), _TBA_N), "§10.4 %s C2" % d)
_p10 = tba10.pivot_table(index="rep", columns="dataset", values="best_any", aggfunc="first")
cell("%d/%d" % (int((_p10["DILIrank"] == _p10["DICTrank"]).sum()), _TBA_N), "§10.4 C3")

# ---------------------------------------------------------------- §11.1 Mondrian
mdc10 = cv("mondrian_conformal.csv")
_MA = ["A. 시간분할", "B. 무작위", "C. 무작위+양성률 맞춤"]
_MM = ["주변형", "Mondrian"]


def _mz10(cov, n, tgt):
    se = (tgt * (1 - tgt) / n) ** 0.5
    return (cov - tgt) / se


for d in DS:
    for al in sorted(mdc10.alpha.unique()):
        tgt = 1 - al
        for arm in _MA:
            for met in _MM:
                g = mdc10[
                    (mdc10.dataset == d)
                    & ((mdc10.alpha - al).abs() < 1e-9)
                    & (mdc10.arm == arm)
                    & (mdc10.method == met)
                ]
                if not len(g):
                    continue
                r = g.iloc[0]
                _sd = r.get("cov_all_sd")
                _c = f4(r["cov_all"]) if _sd != _sd else "%s ± %s" % (f4(r["cov_all"]), f4(_sd))
                _one = int(r["reps"]) == 1
                rowstr(
                    [
                        d,
                        "%.0f%%" % (tgt * 100),
                        arm,
                        met,
                        _c,
                        f4(r["cov_pos"]),
                        f4(r["cov_neg"]),
                        f2(r["set_size"]),
                        ("%+.2f" % _mz10(r["cov_pos"], r["n_pos"], tgt)) if _one else "—",
                        ("%+.2f" % _mz10(r["cov_neg"], r["n_neg"], tgt)) if _one else "—",
                    ],
                    "§11.1 %s %.2f %s %s" % (d, al, arm[:1], met),
                )

# §1 요약표 25행
_m25 = mdc10[
    (mdc10.dataset == "DILIrank")
    & (mdc10.arm.str.startswith("A."))
    & (mdc10.method == "Mondrian")
    & ((mdc10.alpha - 0.10).abs() < 1e-9)
]
_m25c = mdc10[
    (mdc10.dataset == "DICTrank")
    & (mdc10.arm.str.startswith("A."))
    & (mdc10.method == "Mondrian")
    & ((mdc10.alpha - 0.10).abs() < 1e-9)
]
rowstr(
    [
        "25",
        "클래스별 보장(Mondrian conformal) 후 음성 클래스 커버리지, 목표 90%",
        f4(_m25.cov_neg.iloc[0]),
        f4(_m25c.cov_neg.iloc[0]),
        "§11.1",
    ],
    "§1 요약표 25행",
)

# ---------------------------------------------------------------- §15.2 전수 편입
fat10 = cv("fdr_all_tests.csv")
_fn10 = fat10[fat10.newly_included]


def _fq10(x):
    return "< 0.0001" if float(x) < 1e-4 else f4(x)


for src in ("§11.1", "§12.2", "§13.5", "§18", "§21", "§24"):
    g = _fn10[_fn10.source == src]
    if not len(g):
        continue
    rowstr(
        [
            src,
            "%d" % len(g),
            "%d" % int(g.sig_F4.astype(bool).sum()),
            _fq10(g.q_F4.min()),
            "확증 주장에 쓰지 않는다" if src in ("§18", "§21") else "효과크기로 읽는다",
        ],
        "§15.2 %s" % src,
    )
rowstr(
    [
        "합계",
        "%d" % len(_fn10),
        "%d" % int(_fn10.sig_F4.astype(bool).sum()),
        _fq10(_fn10.q_F4.min()),
        "—",
    ],
    "§15.2 합계",
)

# ---------------------------------------------------------------- §14.3 하위집단 보정
sgc10 = cv("subgroup_calibration.csv")
_sg10 = sgc10[~sgc10.underpowered.astype(bool)]
_dup10 = set(zip(_sg10.dataset, _sg10.stratum))
for d in DS:
    for _, r in _sg10[_sg10.dataset == d].iterrows():
        if r.stratum == "ATC L 대 그외: L" and (d, "ATC L") in _dup10:
            continue
        _ex = (r.ratio_lo > 1) or (r.ratio_hi < 1)
        rowstr(
            [
                d,
                str(r.stratum),
                "%d" % int(r.n),
                f4(r.pos_rate),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ctrl_ece),
                f2(r.ece_ratio),
                "[%s, %s]" % (f2(r.ratio_lo), f2(r.ratio_hi)),
                f4(r.p_phipson_smyth),
            ],
            "§14.3 %s %s" % (d, str(r.stratum)[:16]),
        )

# ---------------------------------------------------------------- §10.5 보정기 추가
xc10 = cv("extra_calibrators.csv")
for d in DS:
    g = xc10[xc10.dataset == d]
    for _, r in g.iterrows():
        rowstr(
            [
                d,
                str(r.calibrator),
                f4(r.ece),
                "[%s, %s]" % (f4(r.ece_lo), f4(r.ece_hi)),
                "%.1f%%" % r.recovery,
                f4(r.auroc),
                f4(r.slope),
                f4(r.mean_pred),
            ],
            "§10.5 %s %s" % (d, str(r.calibrator)[:3]),
        )

# ---------------------------------------------------------------- §34 배포 시뮬레이션
pq10 = cv("prequential_retraining.csv")
pqd10 = cv("prequential_retraining_delta.csv")
_PQA10 = [
    "P0. 고정 모델",
    "P1. 해마다 재학습",
    "P2. 재학습+재보정",
    "C1. 연도 무작위",
    "C2. 연도 무작위+양성률 맞춤",
    "C1r. 연도 무작위+재보정",
    "C2r. 연도 무작위+양성률 맞춤+재보정",
]
for d in DS:
    for arm in _PQA10:
        g = pq10[(pq10.dataset == d) & (pq10.arm == arm)]
        if not len(g):
            continue
        _sd = "%s (%d회)" % (f4(g.ece.std(ddof=1)), len(g)) if len(g) > 1 else "—"
        rowstr(
            [
                d,
                arm,
                "%d" % int(g.n.mean()),
                f4(g.ece.mean()),
                _sd,
                f4(g.auroc.mean()),
                f4(g.slope.mean()),
                f4(g.mean_pred.mean()),
                f4(g.obs_rate.mean()),
            ],
            "§34 %s %s" % (d, arm.split(".")[0]),
        )
for _, r in pqd10.iterrows():
    rowstr(
        [
            str(r.dataset),
            "%s 대 P0" % str(r.arm).split(". ")[1],
            sg(r.delta),
            "[%s, %s]" % (sg(r.delta_lo), sg(r.delta_hi)),
            ("**%s**" % r.verdict).replace("**", "") if r.delta_hi < 0 else str(r.verdict),
            "%s %s" % (str(r.ctrl_arm).split(". ")[0], f4(r.ctrl_ece)),
        ],
        "§34 ΔECE %s %s" % (r.dataset, str(r.arm).split(".")[0]),
    )

# ---------------------------------------------------------------- §34.1 재학습 주기
rin10 = cv("retrain_interval.csv")
rind10 = cv("retrain_interval_delta.csv")
for d in DS:
    for rc in (True, False):
        for k in [1, 2, 3, 5, 0]:
            g = rin10[(rin10.dataset == d) & (rin10.interval == k) & (rin10.recal == rc)]
            if not len(g):
                continue
            r = g.iloc[0]
            q = rind10[(rind10.dataset == d) & (rind10.interval == k) & (rind10.recal == rc)]
            rowstr(
                [
                    d,
                    "있음" if rc else "없음",
                    ("%d년마다" % k) if k else "하지 않음(고정)",
                    "%d" % int(r.n_refit),
                    f2(r.mean_age),
                    f4(r.ece),
                    f4(r.auroc),
                    "기준" if not len(q) else sg(q.delta.iloc[0]),
                    (
                        "—"
                        if not len(q)
                        else "[%s, %s]" % (sg(q.delta_lo.iloc[0]), sg(q.delta_hi.iloc[0]))
                    ),
                    "—" if not len(q) else str(q.verdict.iloc[0]),
                ],
                "§34.1 %s recal=%s k=%d" % (d, rc, k),
            )

# ---------------------------------------------------------------- §29.5 AD 팔
utsa10 = cv("useful_threshold_sweep_ad.csv")
for _, r in utsa10.iterrows():
    rowstr(
        [
            str(r.dataset),
            "%.0f%%" % (100 * r.drop_frac),
            "%d" % int(r.n_keep),
            f4(r.ece_ad),
            f4(r.ece_random_drop),
            sg(r.delta),
            "[%s, %s]" % (sg(r.delta_lo), sg(r.delta_hi)),
            f4(r.mde80),
            "%.3f" % r.u_star,
            str(r.verdict).split(" ")[0],
        ],
        "§29.5 AD %s %.0f%%" % (r.dataset, 100 * r.drop_frac),
    )


# ---- §3.1 결정성·지표 정의 단일성 (2026-10-21) --------------------------
def _tf10(col):
    if pd.api.types.is_bool_dtype(col):
        return col
    return col.astype(str).str.strip().str.lower().isin(("true", "1"))


_det = cv("determinism.csv")
_detx = cv("determinism_cross_process.csv")
for _lab, _d in (("한 프로세스 안", _det), ("프로세스 간(스레드 고정 해제)", _detx)):
    rowstr(
        [
            _lab,
            "%d개" % len(_d),
            "%.1e" % float(_d.max_pred_diff.abs().max()),
            "%.1e" % float(_d.ece_diff.abs().max()),
            "%d/%d" % (int(_tf10(_d.within_report_digits).sum()), len(_d)),
        ],
        "§3.1 결정성 %s" % _lab,
    )

_mdup = cv("metric_duplicates.csv")
for _, r in _mdup.iterrows():
    rowstr(
        ["`%s`" % r.metric, "%d" % int(r.n_defs), "%d" % int(r.n_identical)],
        "§3.1 복제 정의 %s" % r.metric,
    )

# ---- §16.4 큐레이션 감사 (2026-10-21) -----------------------------------
_ya = cv("year_audit.csv")
_yc = cv("year_corrections.csv")
_la = cv("label_audit.csv")
_lc = cv("label_conflict_groups.csv")
_sa = cv("smiles_audit.csv")

for d in DS:
    g = _ya[_ya.dataset == d]
    if not len(g):
        continue
    rowstr(
        [
            d,
            "%d" % len(g),
            "%d" % int((g.gap > 0).sum()),
            "%d" % int((g.gap >= 5).sum()),
            "%d년" % int(g.gap.max()),
            "%d" % int(g.gap.isna().sum()),
            "%d" % int(_tf10(g.after_label_freeze).sum()),
            "%d" % int((_tf10(g.after_label_freeze) & _tf10(g.in_heldout)).sum()),
        ],
        "§16.4 연도감사 %s" % d,
    )

for _, r in _yc.iterrows():
    rowstr(
        [
            r.dataset,
            r.compound,
            "%d" % int(r.assigned_year),
            "%d" % int(r.suggested_year),
            "%d년" % (int(r.assigned_year) - int(r.suggested_year)),
        ],
        "§16.4 연도정정 %s" % r.compound,
    )

for d in DS:
    g = _lc[_lc.dataset == d]
    rowstr([d, "%d" % g["_k"].nunique(), "%d" % len(g)], "§16.4 라벨충돌 %s" % d)

for d in DS:
    g = _sa[_sa.dataset == d]
    if not len(g):
        continue
    rowstr(
        [
            d,
            "%d" % len(g),
            "%d" % int(_tf10(g.salt_in_name).sum()),
            "%d" % int((g.skel_std != g.skel_raw).sum()),
            "%d" % int(g.pubchem_cid.isna().sum()),
        ],
        "§16.4 구조감사 %s" % d,
    )

# 라벨 불일치는 본문 문장 안에 있다 (표가 아니다).
cell(
    "DILIrank %d건 / DICTrank %d건"
    % (
        int(_tf10(_la[_la.dataset == "DILIrank"].mismatch).sum()),
        int(_tf10(_la[_la.dataset == "DICTrank"].mismatch).sum()),
    ),
    "§16.4 라벨 불일치 문장",
)

# ---- §10.5 캐스팅 수정 기록 (2026-10-21) --------------------------------
_pf21p = os.path.join(RES, "precision_fix_20261021.csv")
if os.path.exists(_pf21p):
    _pf21 = cv("precision_fix_20261021.csv")
    cell("바뀐 칸은 %d개" % len(_pf21), "§10.5 10-21 바뀐 칸 수")
    cell("바뀌는 것은 %d개" % int(_tf10(_pf21.changes_4dp).sum()), "§10.5 10-21 4자리 변동 수")
    cell("최대차가 %s 다" % ("%.0e" % float(_pf21.abs_diff.abs().max())), "§10.5 10-21 최대차")

# ---- §13.5 대조군 반복 수가 산문과 결과 파일에서 같은가 (2026-10-21) ----
# 왜 — 10-20 전체 재실행이 shift_matched_split 을 인자 없이 불러 기본
# 50회로 떨어뜨렸는데, 산문에는 "200회" 가 **손으로 적혀** 있었다. 두
# 검증기 모두 못 잡았다. verify9 는 소수와 정해진 정수 목록만 보고,
# verify10 은 이 문장을 칸으로 등록하지 않았기 때문이다. 등록한다.
_sm10 = cv("shift_matched_split.csv")
cell("무작위 대조군 %d회(§8 의 R2 설계)" % int(_sm10.n_rep_valid.max()), "§13.5 대조군 반복 수")

# ---- §34.2 라벨 지연 민감도 (2026-10-22) -------------------------------
_pqy = cv("prequential_delay.csv")
_pqdd = cv("prequential_delay_delta.csv")
_pqc = cv("prequential_delay_check.csv")
cell("차가 %s 다" % ("%.2e" % float(_pqc.abs_diff.abs().max())), "§34.2 지연 0 재현 최대차")
for d in DS:
    g = _pqy[_pqy.dataset == d]
    for dl in sorted(g.delay.unique()):
        r0 = g[(g.delay == dl) & g.arm.str.startswith("P0")].iloc[0]
        r1 = g[(g.delay == dl) & g.arm.str.startswith("P1")].iloc[0]
        r2 = g[(g.delay == dl) & g.arm.str.startswith("P2")].iloc[0]
        q = _pqdd[(_pqdd.dataset == d) & (_pqdd.delay == dl) & (_pqdd.vs == "P0. 고정 모델")]
        a1 = q[q.arm.str.startswith("P1")].iloc[0]
        a2 = q[q.arm.str.startswith("P2")].iloc[0]
        rowstr(
            [
                d,
                "%d년" % int(dl),
                f4(r0.ece),
                f4(r1.ece),
                f4(r2.ece),
                "%s [%s, %s]%s"
                % (
                    sg(a1.dece),
                    sg(a1.lo),
                    sg(a1.hi),
                    "" if bool(a1.excludes_zero) else " (0 포함)",
                ),
                "%s [%s, %s]%s"
                % (
                    sg(a2.dece),
                    sg(a2.lo),
                    sg(a2.hi),
                    "" if bool(a2.excludes_zero) else " (0 포함)",
                ),
            ],
            "§34.2 %s 지연 %d년" % (d, int(dl)),
        )

print("verify10 — 셀 단위 대조 (CSV → 문서 방향)")
print("  대조한 행·칸: %d개" % n_cells)
if not fails:
    io.open(os.path.join(RES, "verify10_cells.txt"), "w", encoding="utf-8").write("%d\n" % n_cells)
print("실패 %d건" % len(fails))
for x in fails[:40]:
    print("  X", x)
if len(fails) > 40:
    print("  ... 그 외 %d건" % (len(fails) - 40))
sys.exit(1 if fails else 0)
