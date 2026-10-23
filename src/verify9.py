# -*- coding: utf-8 -*-
"""논문용 보고서(RESULTS_REPORT.md) 전수 검증.

1. 문서의 소수 4자리 이상 숫자 전부가 results/ 원본이나 재계산으로 설명되는가
2. 표 구조 (열 수 일치, 구분선 존재)
3. 절 번호 연속성과 §n.m 참조 해소
4. §15 의 BH q 값 전부를 정밀 p 로 재계산해 대조
5. 포맷 미치환(%s 등)·nan·빈 셀 없음
6. 생성기 재실행 결과가 현재 파일과 바이트 단위로 같은가 (수동 편집 감지)

    python src/verify9.py
"""

import glob
import hashlib
import io
import json
import os
import re
import subprocess
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
ND = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

fails = []


def chk(c, m):
    if not c:
        fails.append(m)


s = io.open(DOC, encoding="utf-8").read()
s0_doc = s
body = re.sub(r"```.*?```", " ", s, flags=re.S)
lines = s.split("\n")

# ---- 1. 포맷·결측 -------------------------------------------------------
for pat, name in [
    (r"%[sdf]", "포맷 미치환"),
    (r"\bnan\b", "nan"),
    (r"\bNone\b", "None"),
    (r"\bNaN\b", "NaN"),
    (r"�", "U+FFFD"),
]:
    m = re.search(pat, body)
    chk(m is None, "%s 발견: %r" % (name, body[max(0, m.start() - 50) : m.end() + 30] if m else ""))
chk("\t" not in s, "탭 문자 존재")
chk("\n\n\n\n" not in s, "빈 줄 4연속")

# ---- 2. 표 구조 ---------------------------------------------------------
i = ntab = 0
while i < len(lines):
    if (
        lines[i].startswith("|")
        and i + 1 < len(lines)
        and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1])
    ):
        ntab += 1
        n = lines[i].count("|")
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            chk(lines[j].count("|") == n, "표 열 수 불일치 %d행: %s" % (j + 1, lines[j][:70]))
            j += 1
        chk(j - i >= 3, "표에 데이터 행이 없다 (%d행)" % (i + 1))
        i = j
    else:
        if lines[i].startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", lines[i]):
            prev_tab = i > 0 and lines[i - 1].startswith("|")
            chk(prev_tab, "구분선 없는 표 행 %d: %s" % (i + 1, lines[i][:60]))
        i += 1

# ---- 3. 절 번호와 참조 --------------------------------------------------
secs = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\. ", s, re.M)]
chk(secs == list(range(1, len(secs) + 1)), "절 번호 불연속: %s" % secs)
subs = {}
for m in re.finditer(r"^### (\d+)\.(\d+) ", s, re.M):
    subs.setdefault(int(m.group(1)), []).append(int(m.group(2)))
for sec, got in subs.items():
    chk(got == list(range(1, len(got) + 1)), "§%d 하위절 불연속: %s" % (sec, got))
have = set("%d" % x for x in secs) | set("%d.%d" % (a, b) for a, bs in subs.items() for b in bs)
for m in re.finditer(r"§(\d+(?:\.\d+)?)", s):
    chk(m.group(1) in have, "해소되지 않는 참조 §%s" % m.group(1))
chk(len(secs) >= 19, "절이 %d개뿐" % len(secs))


# ---- 4. §15 BH 재계산 ---------------------------------------------------
def bh(ps):
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


mc = pd.read_csv(os.path.join(RES, "multiple_comparison_correction.csv"))
for fam, g in mc.groupby("family"):
    q = bh(list(g.p))
    for (_, r), qq in zip(g.iterrows(), q):
        chk(
            abs(round(qq, 10) - round(r.q_BH, 10)) < 1e-9,
            "%s %s q 재계산 %.6f != 파일 %.6f" % (fam, r.test, qq, r.q_BH),
        )
        # make_report.py 가 §14.1 에서 정한 정확한 군 이름으로 바꿔 인쇄한다 (2026-10-01).
        _tn = r.test.replace("ATC L(항종양)", "ATC L(항종양·면역조절)")
        chk(
            "| %s | %s | %.4f | " % (r.dataset, _tn, r.p) in s
            or "| %s | %s | %.4f |" % (r.dataset, _tn, r.p) in s.replace("**", ""),
            "§15 에 %s / %s 행 없음" % (r.dataset, _tn),
        )
ep = pd.read_csv(os.path.join(RES, "era_permutation.csv"))
era_p = float(ep.p_phipson_smyth.iloc[0])
chk(abs(era_p - 1 / 2001) < 1e-15, "시대 분류기 p 가 1/2001 이 아니다")
chk(int(ep.n_perm.iloc[0]) == 2000 and int(ep.n_perm.iloc[1]) == 2000, "순열 횟수가 2000 이 아니다")
g1 = list(mc[mc.family == "1_primary"].p)
g2 = list(mc[mc.family == "2_prevalence_control"].p)
g3 = list(mc[mc.family == "3_exploratory_subgroup"].p)
for nm, base, want in (("1군", g1, None), ("2군", g2, None), ("3군", g3, None)):
    qq = bh(base + [era_p, era_p])[len(base)]
    chk(("q = %.4f" % qq) in s or ("%.4f" % qq) in s, "%s 편입 q %.4f 가 문서에 없다" % (nm, qq))

# ---- 5. 수치 추적 가능성 -----------------------------------------------
tok = set()
for f in sorted(glob.glob(os.path.join(RES, "*.csv"))):
    try:
        d = pd.read_csv(f)
    except Exception as e:
        fails.append("CSV 읽기 실패 %s: %s" % (os.path.basename(f), e))
        continue
    for c in d.columns:
        col = pd.to_numeric(d[c], errors="coerce").dropna()
        for v in col:
            tok.add(repr(float(v)))
            for nd in ND:
                tok.add("%.*f" % (nd, v))
                tok.add("%.*f" % (nd, v * 100))
for f in sorted(glob.glob(os.path.join(RES, "*.txt"))):
    t = io.open(f, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"\d+\.\d+", t):
        c = m.group(0)
        tok.add(c)
        try:
            v = float(c)
        except ValueError:
            continue
        for nd in ND:
            tok.add("%.*f" % (nd, v))


def derived():
    out = {}
    bc = pd.read_csv(os.path.join(RES, "baseline_cv_per_fold.csv"))
    pc = pd.read_csv(os.path.join(RES, "prevalence_control.csv"))
    mn = pd.read_csv(os.path.join(RES, "matched_n_control.csv"))
    r3 = pd.read_csv(os.path.join(RES, "brier_decomposition_r3.csv"))
    bd = pd.read_csv(os.path.join(RES, "brier_decomposition.csv"))
    cens = pd.read_csv(os.path.join(RES, "censoring_results.csv"))
    rec = pd.read_csv(os.path.join(RES, "recalibration_results.csv"))
    ys = pd.read_csv(os.path.join(RES, "year_sensitivity.csv"))
    gr = pd.read_csv(os.path.join(RES, "grid_calibration.csv"))
    ster = pd.read_csv(os.path.join(RES, "stereo_sensitivity.csv"))
    seed = pd.read_csv(os.path.join(RES, "seed_sensitivity.csv"))
    binn = pd.read_csv(os.path.join(RES, "ece_binning_sensitivity.csv"))
    rob = pd.read_csv(os.path.join(RES, "robustness_models.csv"))
    tie = pd.read_csv(os.path.join(RES, "tiebreak_sensitivity.csv"))
    aa = pd.read_csv(os.path.join(RES, "atc_audit.csv"))
    # 순열 하한
    for n in (201, 1001, 2001):
        out["하한 1/%d" % n] = 1.0 / n
    # §7: uncertainty 행이 다른 실행에서 왔으므로 무작위 열의 Murphy 항등식이
    # 정확히 닫히지 않는다. 그 '닫히지 않는 값' 과 차이를 본문에 적었다 (2026-10-01).
    conf = pd.read_csv(os.path.join(RES, "conformal_prediction.csv"))
    for d in ("DILIrank", "DICTrank"):
        b = bd[bd.dataset == d].iloc[0]
        u = r3[(r3.dataset == d) & (r3.baseline.str.startswith("R1"))].iloc[0]
        imp = b.reliability_random_mean - b.resolution_random_mean + u.unc_random
        out["%s 무작위 항등식 합" % d] = imp
        out["%s 무작위 항등식 불일치" % d] = abs(b.brier_random_mean - imp)
    # §13.5: 분할별 유병률 이동과 배율-예측변수 Spearman 상관 (2026-10-02).
    sm9 = pd.read_csv(os.path.join(RES, "shift_matched_split.csv"))
    for _, r9 in sm9.iterrows():
        out["%s %s 유병률 이동" % (r9.dataset, r9.split)] = r9.pos_rate_test - r9.pos_rate_train

    def _sp(a, b):
        a = pd.Series(a).rank().to_numpy(dtype=float)
        b = pd.Series(b).rank().to_numpy(dtype=float)
        a = a - a.mean()
        b = b - b.mean()
        den = ((a * a).sum() * (b * b).sum()) ** 0.5
        return float((a * b).sum() / den) if den else float("nan")

    # §17.2: V0→V2 검증군 교체 효과와 V2 기준 크기/구성 몫 (2026-10-04).
    dj9 = pd.read_csv(os.path.join(RES, "disjoint_subset.csv"))
    _dc = {
        v: dj9[(dj9.dataset == "DICTrank") & (dj9.variant == v)].iloc[0]
        for v in ("V0", "V1", "V1c", "V2")
    }
    _sw = _dc["V2"].auroc - _dc["V0"].auroc
    _sz = _dc["V2"].auroc - _dc["V1c"].auroc
    _cp = _dc["V1c"].auroc - _dc["V1"].auroc
    out["§17.2 검증군 교체"] = _sw
    out["§17.2 크기 몫"] = _sz
    out["§17.2 구성 몫"] = _cp
    out["§17.2 크기 몫 %"] = _sz / (_sz + _cp) * 100
    out["§17.2 구성 몫 %"] = _cp / (_sz + _cp) * 100
    # §15: 인쇄된 네 자리 p 로 BH 를 다시 돌렸을 때의 3군 최소 q (2026-10-04).
    mc9 = pd.read_csv(os.path.join(RES, "multiple_comparison_correction.csv"))
    _g9 = mc9[mc9.family == "3_exploratory_subgroup"].sort_values("p")
    _p9 = [float("%.4f" % x) for x in _g9.p]
    _m9 = len(_p9)
    _q9 = [0.0] * _m9
    _r9 = 1.0
    for _k in range(_m9, 0, -1):
        _r9 = min(_r9, _p9[_k - 1] * _m9 / _k)
        _q9[_k - 1] = _r9
    out["§15 인쇄p 기준 3군 최소 q"] = min(_q9)
    out["§13.5 rho 이동AUC"] = _sp(sm9.shift_auc, sm9.ece_ratio)
    out["§13.5 rho 유병률이동"] = _sp((sm9.pos_rate_test - sm9.pos_rate_train).abs(), sm9.ece_ratio)
    # §11: DICTrank 90% 무작위 커버리지가 명목값에서 떨어진 양 (2026-10-01).
    _c = conf[(conf.dataset == "DICTrank") & (conf.alpha == 0.1)].iloc[0]
    out["DICTrank conf90 무작위 명목 미달"] = 0.90 - _c.marginal_coverage_random
    for d in ("DILIrank", "DICTrank"):
        gb = bc[bc.dataset == d]
        T = gb[gb.split == "temporal"].iloc[0]
        for sp in ("random_cv", "scaffold_cv"):
            gg = gb[gb.split == sp]
            out["%s %s AUROC 평균" % (d, sp)] = gg.AUROC.mean()
            out["%s %s AUROC SD" % (d, sp)] = gg.AUROC.std(ddof=1)
            out["%s %s ECE 평균" % (d, sp)] = gg.ECE.mean()
            out["%s %s ECE SD" % (d, sp)] = gg.ECE.std(ddof=1)
            out["%s %s ECE 배율" % (d, sp)] = T.ECE / gg.ECE.mean()
        out["%s 스캐폴드-무작위 AUROC 차" % d] = (
            gb[gb.split == "scaffold_cv"].AUROC.mean() - gb[gb.split == "random_cv"].AUROC.mean()
        )
        # ECE 초과분·잔여·해소율
        Tr = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("T"))].iloc[0]
        R1 = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("R1"))].iloc[0]
        R3r = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("R3"))].iloc[0]
        ex, rm = Tr.ECE - R1.ECE, Tr.ECE - R3r.ECE
        out["%s 초과분" % d] = ex
        out["%s 잔여" % d] = rm
        out["%s 해소율" % d] = (ex - rm) / ex * 100
        # z
        for b in ("R1", "R2", "R3"):
            r = mn[(mn.dataset == d) & (mn.baseline == b)].iloc[0]
            out["%s %s z_old" % (d, b)] = (r.ece_T_full - r.ece_ctrl) / r.ece_ctrl_sd
            out["%s %s z_new" % (d, b)] = (r.ece_T_matched - r.ece_ctrl) / r.ece_ctrl_sd
            dfull = mn[(mn.dataset == d) & (mn.baseline == "R1")].delta_T_matched.iloc[0]
            out["%s %s δ_old" % (d, b)] = dfull - r.delta_ctrl
            out["%s %s δ_new" % (d, b)] = r.delta_T_matched - r.delta_ctrl
            out["%s %s 기여율차" % (d, b)] = r.share_new - r.share_old
        # Brier 관련
        b_ = bd[bd.dataset == d].iloc[0]
        out["%s 보정/판별 비" % d] = b_.d_reliability_A / b_.d_resolution_A
        out["%s 부트 100 초과" % d] = b_.frac_boot_over_100 * 100
        for k in ("resolution", "reliability", "brier"):
            out["%s %s 차" % (d, k)] = b_["%s_temporal" % k] - b_["%s_random_mean" % k]
        out["%s unc 차" % d] = (
            r3[(r3.dataset == d)].unc_temporal.iloc[0] - r3[(r3.dataset == d)].unc_random.iloc[0]
        )
        for b in ("R2", "R3"):
            rr = r3[(r3.dataset == d) & (r3.baseline.str.startswith(b))].iloc[0]
            out["%s %s 2항3항차" % (d, b)] = abs(rr.share2_reliability - rr.share3_reliability)
        # 치우침
        cal = pd.read_csv(os.path.join(RES, "calibration_results.csv"))
        for st in ("random", "temporal"):
            rr = cal[(cal.dataset.str.startswith(d)) & (cal.setting == st)].iloc[0]
            out["%s %s 치우침" % (d, st)] = rr.mean_pred - rr.prevalence
            out["%s %s BalAcc 개선" % (d, st)] = rr["BalAcc@best"] - rr["BalAcc@0.5"]
            out["%s %s MCC 개선" % (d, st)] = rr["MCC@best"] - rr["MCC@0.5"]
        # 재보정 회복률
        g = rec[rec.dataset.str.startswith(d)]
        a = g[g.method.str.startswith("A.")].iloc[0]
        e = g[g.method.str.startswith("E.")].iloc[0]
        for _, rr in g.iterrows():
            out["%s %s 회복률" % (d, rr.method)] = (a.ECE - rr.ECE) / (a.ECE - e.ECE) * 100
        # 검열
        gc = cens[cens.dataset.str.startswith(d)]
        base = gc[gc.exp.str.startswith("기준")].iloc[0]
        for _, rr in gc.iterrows():
            out["%s %s 기준대비" % (d, rr.exp)] = base.gap - rr.gap
        cut = gc[gc.exp.str.startswith("절단")].gap.min()
        out["%s 검열 축소율" % d] = (base.gap - cut) / base.gap * 100
        out["%s 가중 최대차" % d] = abs(gc[gc.exp.str.startswith("가중")].gap - base.gap).max()
        # 연도 민감도 변동폭
        gy = ys[ys.dataset == d]
        b0 = gy[gy.scenario.str.startswith("S0")].iloc[0]
        out["%s 연도 배율 변동" % d] = (gy.ece_ratio - b0.ece_ratio).abs().max()
        out["%s 연도 기여율 변동" % d] = (
            (gy.pct_from_reliability - b0.pct_from_reliability).abs().max()
        )
        out["%s S0 배율(20회 분모)" % d] = T.ECE / b0.ECE_random_mean
        out["%s 격자 R1 배율" % d] = T.ECE / gr[gr.dataset == d].r1_ece_random.iloc[0]
        for b in ("R1", "R2", "R3"):
            out["%s %s 배율(전정밀)" % (d, b)] = (
                T.ECE
                / pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith(b))].ECE.iloc[0]
            )
        # 시드·구간화·격자·동점 범위
        gs = seed[seed.dataset == d]
        for k in ("auroc", "ece"):
            out["%s 시드 %s 평균" % (d, k)] = gs[k].mean()
            out["%s 시드 %s SD" % (d, k)] = gs[k].std(ddof=1)
        gt = tie[tie.dataset == d]
        for exp in ("E1", "E2"):
            ge = gt[gt.experiment.str.startswith(exp)]
            for k in ("conf90_marginal", "conf80_marginal", "rec_B", "rec_C", "auroc"):
                out["%s %s %s 폭" % (d, exp, k)] = ge[k].max() - ge[k].min()
        # ATC L 양성률
        at = aa[(aa.dataset == d) & (aa.approval_year >= 2010)]
        out["%s L 양성률" % d] = at[at.atc1 == "L"].y.mean() * 100
        out["%s 비L 양성률" % d] = at[at.atc1 != "L"].y.mean() * 100
    # Tanimoto
    t = pd.read_csv(os.path.join(RES, "temporal_detail.csv")).nn_tanimoto
    out["Tanimoto 중앙값"] = t.median()
    out["Tanimoto 평균"] = t.mean()
    out["Tanimoto <0.4 비율"] = (t < 0.4).mean() * 100
    # BH 확장
    fam = g1 + g2 + g3 + [era_p, era_p]
    for i, q in enumerate(bh(fam)):
        out["22건 q %d" % i] = q
    r3p = float(mn[(mn.dataset == "DICTrank") & (mn.baseline == "R3")].p_old.iloc[0])
    r3n = float(mn[(mn.dataset == "DICTrank") & (mn.baseline == "R3")].p_new.iloc[0])
    famm = [r3n if abs(x - r3p) < 1e-15 else x for x in fam]
    for i, q in enumerate(bh(famm)):
        out["22건m q %d" % i] = q
    for nm, base in (("1군", g1), ("2군", g2), ("3군", g3)):
        for i, q in enumerate(bh(base + [era_p, era_p])):
            out["%s편입 q %d" % (nm, i)] = q
    out["R3 하한 배수"] = r3p / (1 / 201)
    # 격자·강건성의 중앙값·범위 (요약 통계는 원본 열에 그 값이 없다)
    for d in ("DILIrank", "DICTrank"):
        gb = rob[rob.dataset == d]
        for c in ("d_auroc", "ece_ratio", "slope_temporal", "auroc_temporal", "auroc_random"):
            out["%s rob %s 중앙값" % (d, c)] = gb[c].median()
            out["%s rob %s 최소" % (d, c)] = gb[c].min()
            out["%s rob %s 최대" % (d, c)] = gb[c].max()
        gg = gr[gr.dataset == d]
        for c in (
            "r1_ece_ratio",
            "r2_ece_ratio",
            "r1_share2",
            "r2_share2",
            "conf90_marginal",
            "conf80_marginal",
        ):
            out["%s grid %s 중앙값" % (d, c)] = gg[c].median()
            out["%s grid %s 최소" % (d, c)] = gg[c].min()
            out["%s grid %s 최대" % (d, c)] = gg[c].max()
        gs2 = seed[seed.dataset == d]
        for c in ("auroc", "ece"):
            out["%s seed %s 최소" % (d, c)] = gs2[c].min()
            out["%s seed %s 최대" % (d, c)] = gs2[c].max()
        gb2 = binn[binn.dataset == d]
        for c in ("ece", "reliability", "resolution"):
            out["%s bin %s 최소" % (d, c)] = gb2[c].min()
            out["%s bin %s 최대" % (d, c)] = gb2[c].max()
        gt2 = tie[tie.dataset == d]
        for exp in ("E1", "E2"):
            ge = gt2[gt2.experiment.str.startswith(exp)]
            for c in ("conf90_marginal", "conf80_marginal", "rec_B", "rec_C", "auroc"):
                out["%s %s %s 최소" % (d, exp, c)] = ge[c].min()
                out["%s %s %s 최대" % (d, exp, c)] = ge[c].max()
        gstr = ster[ster.dataset == d]
        for c in ("share_reliability", "ece_ratio_pm"):
            for i in range(len(gstr)):
                out["%s ster %s %d" % (d, c, i)] = gstr[c].iloc[i]
    # 분할 배정 시드 범위
    ss = pd.read_csv(os.path.join(RES, "split_seed_sensitivity.csv"))
    for (d, sp), g4 in ss.groupby(["dataset", "split"]):
        for c in ("auroc_oof", "ece_oof"):
            out["%s %s %s 최소" % (d, sp, c)] = g4[c].min()
            out["%s %s %s 최대" % (d, sp, c)] = g4[c].max()
    # 재보정 회복률 범위
    rs = pd.read_csv(os.path.join(RES, "recalibration_sensitivity.csv"))
    for (d, mt), g5 in rs.groupby(["dataset", "method"]):
        r5 = g5.recovery.dropna()
        if len(r5):
            out["%s %s rec 최소" % (d, mt)] = r5.min()
            out["%s %s rec 최대" % (d, mt)] = r5.max()
    for d in ("DILIrank", "DICTrank"):
        both = rs[(rs.dataset == d) & (rs.method.str.startswith(("B.", "C.")))].recovery.dropna()
        out["%s BC 합집합 최소" % d] = both.min()
        out["%s BC 합집합 최대" % d] = both.max()
    # ATC 계열 AUROC 범위
    at2 = pd.read_csv(os.path.join(RES, "atc_stratified.csv"))
    for d in ("DILIrank", "DICTrank"):
        g6 = at2[at2.dataset == d].auroc.dropna()
        out["%s ATC AUROC 최소" % d] = g6.min()
        out["%s ATC AUROC 최대" % d] = g6.max()
    # 최선 재보정 ECE (A·E 제외)
    for d in ("DILIrank", "DICTrank"):
        g7 = rec[(rec.dataset.str.startswith(d)) & (~rec.method.str.startswith(("A", "E")))]
        out["%s 최선 재보정 ECE" % d] = g7.ECE.min()
    # §25.1 의 C0 기준 BSS 와 §25 4번의 C1 기준 BSS — 보고서가 Brier 에서 재계산한다
    tb9 = pd.read_csv(os.path.join(RES, "trivial_baseline.csv"))
    for d in ("DILIrank", "DICTrank"):

        def _pick(sp, pref, _d=d):
            return float(
                tb9[
                    (tb9.dataset == _d) & (tb9.split == sp) & (tb9.predictor.str.startswith(pref))
                ].brier.iloc[0]
            )

        c0 = _pick("시간분할", "C0.")
        g8 = rec[rec.dataset.str.startswith(d)]
        for i in range(len(g8)):
            out["%s 재보정 C0기준 BSS %d" % (d, i)] = 1.0 - float(g8.Brier.iloc[i]) / c0
        out["%s C1 기준 BSS" % d] = 1.0 - _pick("시간분할", "M.") / _pick("시간분할", "C1.")
    # §28.3 의 손잡이별 평균 검증 ECE (보고서가 격자에서 집계한다)
    hp9 = pd.read_csv(os.path.join(RES, "hparam_sensitivity.csv"))
    for d in ("DILIrank", "DICTrank"):
        g9 = hp9[hp9.dataset == d]
        for c in ("max_depth", "n_estimators", "learning_rate", "min_child_weight"):
            for v in sorted(hp9[c].unique()):
                out["%s %s %s 평균 ECE" % (d, c, v)] = g9[g9[c] == v].temporal_ece.mean()
        out["%s 격자 ECE 최소" % d] = g9.temporal_ece.min()
        out["%s 격자 ECE 최대" % d] = g9.temporal_ece.max()
        out["%s 격자 AUROC 최소" % d] = g9.temporal_auroc.min()
        out["%s 격자 AUROC 최대" % d] = g9.temporal_auroc.max()
    # §30 의 유효 표본 손실률 (보고서가 n_eff 에서 계산한다)
    cw9 = pd.read_csv(os.path.join(RES, "covariate_shift_weighting.csv"))
    for d in ("DILIrank", "DICTrank"):
        g9 = cw9[(cw9.dataset == d) & (cw9.kind == "outer")]
        w09 = float(g9[g9.arm.str.startswith("W0")].n_eff.iloc[0])
        iw9 = g9[g9.arm.str.startswith("IW")]
        for i in range(len(iw9)):
            out["%s IW %d 유효n 손실" % (d, i)] = 100.0 * (1 - float(iw9.n_eff.iloc[i]) / w09)
        out["%s IW 최대 손실" % d] = 100.0 * (1 - iw9.n_eff.min() / w09)
        out["%s IW ECE 최소" % d] = iw9.ece.min()
        out["%s IW AUROC 최대" % d] = iw9.auroc.max()
    # §31.1 의 격자 기여율 이상치와 분모 분위 (보고서가 격자에서 집계한다)
    g39 = pd.read_csv(os.path.join(RES, "grid_r3.csv"))
    out["격자 기여율 절대 최대"] = g39.r3_share2.abs().max()
    _den9 = (g39.r3_d_reliability + g39.r3_d_resolution).abs()
    out["격자 분모 10백분위"] = _den9.quantile(0.1)
    out["격자 배율 최소"] = g39.r3_ece_ratio.min()
    out["격자 배율 최대"] = g39.r3_ece_ratio.max()
    out["격자 배율 중앙"] = g39.r3_ece_ratio.median()
    for _k in ("model", "rep"):
        _m9 = g39.groupby(_k).r3_ece_ratio.median()
        out["격자 %s 중앙 최소" % _k] = _m9.min()
        out["격자 %s 중앙 최대" % _k] = _m9.max()
    # §26 의 구간 폭 비 (분자+분모 / 분자만)
    ri9 = pd.read_csv(os.path.join(RES, "ratio_intervals.csv"))
    _w9 = ri9.ci_width_full / ri9.ci_width_numonly
    out["배율 구간 폭비 최소"] = _w9.min()
    out["배율 구간 폭비 최대"] = _w9.max()
    # §10.4 — 50회 평균·표준편차·백분위·최소최대는 전부 파생값이다.
    # 우연히 어떤 반복의 값과 일치해 '설명된' 것처럼 보이지 않도록 명시한다.
    tba9 = pd.read_csv(os.path.join(RES, "tiebreak_averaged.csv"))
    for d in ("DILIrank", "DICTrank"):
        g9 = tba9[tba9.dataset == d]
        for c in (
            "ece_A",
            "ece_Ap",
            "ece_B",
            "ece_C",
            "ece_D",
            "ece_E",
            "rec_B",
            "rec_C",
            "rec_D",
            "conf90_marginal",
            "conf80_marginal",
            "auroc_A",
        ):
            out["%s %s 평균" % (d, c)] = g9[c].mean()
            out["%s %s 표준편차" % (d, c)] = g9[c].std(ddof=1)
            out["%s %s 2.5백분위" % (d, c)] = g9[c].quantile(0.025)
            out["%s %s 97.5백분위" % (d, c)] = g9[c].quantile(0.975)
            out["%s %s 최소" % (d, c)] = g9[c].min()
            out["%s %s 최대" % (d, c)] = g9[c].max()
            out["%s %s 표준오차" % (d, c)] = g9[c].std(ddof=1) / (len(g9) ** 0.5)
    # §11.1 — Mondrian conformal 의 z 와 커버리지 부족분은 파생값이다.
    mdc9 = pd.read_csv(os.path.join(RES, "mondrian_conformal.csv"))
    for _, r9 in mdc9.iterrows():
        tgt9 = 1.0 - float(r9["alpha"])
        for cls9, ck9, nk9 in (("양성", "cov_pos", "n_pos"), ("음성", "cov_neg", "n_neg")):
            key = "%s %s %s %.2f %s" % (r9["dataset"], r9["arm"], r9["method"], r9["alpha"], cls9)
            n9 = float(r9[nk9])
            if n9 > 0:
                se9 = (tgt9 * (1 - tgt9) / n9) ** 0.5
                out[key + " z"] = (float(r9[ck9]) - tgt9) / se9
            out[key + " 부족분"] = tgt9 - float(r9[ck9])
        out["%s %s %s %.2f 집합크기차" % (r9["dataset"], r9["arm"], r9["method"], r9["alpha"])] = (
            float(r9["set_size"])
        )
    # §14.3 — 층별 배율·구간은 파생값이다
    sgc9 = pd.read_csv(os.path.join(RES, "subgroup_calibration.csv"))
    for _, r9 in sgc9.iterrows():
        if bool(r9["underpowered"]):
            continue
        k9 = "%s %s" % (r9["dataset"], r9["stratum"])
        for c9 in (
            "ece_ratio",
            "ratio_lo",
            "ratio_hi",
            "ctrl_ece",
            "ece",
            "auroc",
            "pos_rate",
            "p_phipson_smyth",
        ):
            v9 = r9[c9]
            if v9 == v9:
                out["%s %s" % (k9, c9)] = float(v9)
    # §10.5·§34 — 추가 보정기와 배포 시뮬레이션의 파생값
    xc9 = pd.read_csv(os.path.join(RES, "extra_calibrators.csv"))
    for _, r9 in xc9.iterrows():
        k9 = "%s %s" % (r9["dataset"], str(r9["calibrator"])[:3])
        for c9 in (
            "ece",
            "ece_lo",
            "ece_hi",
            "recovery",
            "auroc",
            "slope",
            "mean_pred",
            "va_width",
        ):
            v9 = r9[c9]
            if v9 == v9:
                out["%s %s" % (k9, c9)] = float(v9)
    pq9 = pd.read_csv(os.path.join(RES, "prequential_retraining.csv"))
    for (d9, a9), g9 in pq9.groupby(["dataset", "arm"]):
        k9 = "%s %s" % (d9, str(a9).split(".")[0])
        for c9 in ("ece", "auroc", "slope", "mean_pred", "obs_rate"):
            out["%s %s 평균" % (k9, c9)] = float(g9[c9].mean())
        if len(g9) > 1:
            out["%s ece 표준편차" % k9] = float(g9.ece.std(ddof=1))
    # P2 대 조건맞춤 기준선 배율, §10 간격 환산 회복률
    rec9b = pd.read_csv(os.path.join(RES, "recalibration_results.csv"))
    for d9 in ("DILIrank", "DICTrank"):
        _p2 = float(pq9[(pq9.dataset == d9) & pq9.arm.str.startswith("P2")].ece.iloc[0])
        _p0 = float(pq9[(pq9.dataset == d9) & pq9.arm.str.startswith("P0")].ece.iloc[0])
        for pre in ("C1.", "C2.", "C1r", "C2r"):
            _c = pq9[(pq9.dataset == d9) & pq9.arm.str.startswith(pre)]
            if len(_c):
                out["%s P2/%s 배율" % (d9, pre)] = _p2 / float(_c.ece.mean())
                out["%s %s 기준선차" % (d9, pre)] = _p2 - float(_c.ece.mean())
        _g = rec9b[rec9b.dataset.str.startswith(d9)]
        _den = float(_g[_g.method.str.startswith("A.")].ECE.iloc[0]) - float(
            _g[_g.method.str.startswith("E.")].ECE.iloc[0]
        )
        out["%s P2 간격환산 회복률" % d9] = 100.0 * (_p0 - _p2) / _den
        out["%s P2-P0" % d9] = _p2 - _p0
    pqd9 = pd.read_csv(os.path.join(RES, "prequential_retraining_delta.csv"))
    for _, r9 in pqd9.iterrows():
        k9 = "%s %s" % (r9["dataset"], str(r9["arm"]).split(".")[0])
        for c9 in ("delta", "delta_lo", "delta_hi", "own_ece", "ctrl_ece", "ctrl_sd"):
            v9 = r9[c9]
            if v9 == v9:
                out["%s %s" % (k9, c9)] = float(v9)
    # §29.5 AD 팔 — U* 와 MDE80
    utsa9 = pd.read_csv(os.path.join(RES, "useful_threshold_sweep_ad.csv"))
    for _, r9 in utsa9.iterrows():
        k9 = "%s AD %.0f%%" % (r9["dataset"], 100 * float(r9["drop_frac"]))
        for c9 in (
            "ece_ad",
            "ece_random_drop",
            "delta",
            "delta_lo",
            "delta_hi",
            "mde80",
            "u_star",
            "se",
        ):
            v9 = r9[c9]
            if v9 == v9:
                out["%s %s" % (k9, c9)] = float(v9)
    # §34.1 — 주기별 ECE·묵은햇수·짝지은 ΔECE
    rin9 = pd.read_csv(os.path.join(RES, "retrain_interval.csv"))
    for _, r9 in rin9.iterrows():
        k9 = "%s k=%s recal=%s" % (r9["dataset"], r9["interval"], r9["recal"])
        for c9 in ("ece", "auroc", "slope", "mean_pred", "mean_age", "max_age"):
            out["%s %s" % (k9, c9)] = float(r9[c9])
    rind9 = pd.read_csv(os.path.join(RES, "retrain_interval_delta.csv"))
    for _, r9 in rind9.iterrows():
        k9 = "%s k=%s recal=%s" % (r9["dataset"], r9["interval"], r9["recal"])
        for c9 in ("delta", "delta_lo", "delta_hi"):
            out["%s %s" % (k9, c9)] = float(r9[c9])
    for d9 in ("DILIrank", "DICTrank"):
        _f = rind9[(rind9.dataset == d9) & (rind9.interval == 0) & rind9.recal]
        _5 = rind9[(rind9.dataset == d9) & (rind9.interval == 5) & rind9.recal]
        if len(_f) and len(_5):
            out["%s 고정/5년 배수" % d9] = float(_f.delta.iloc[0]) / float(_5.delta.iloc[0])
    # §15.2 — BH 를 두 가족 구성에서 다시 돌린 q 전부
    fat9 = pd.read_csv(os.path.join(RES, "fdr_all_tests.csv"))
    for _, r9 in fat9.iterrows():
        k9 = "%s %s %s" % (r9["source"], r9["dataset"], str(r9["test"])[:40])
        for c9 in ("q_F4", "q_F0", "q_existing", "p"):
            v9 = r9[c9]
            if v9 == v9:
                out["%s %s" % (k9, c9)] = float(v9)
    for d in ("DILIrank", "DICTrank"):
        for al in sorted(mdc9.alpha.unique()):
            for me in ("주변형", "Mondrian"):
                q = mdc9[
                    (mdc9.dataset == d) & (mdc9.method == me) & ((mdc9.alpha - al).abs() < 1e-9)
                ]
                a9 = q[q.arm == "A. 시간분할"]
                b9 = q[q.arm == "B. 무작위"]
                c9 = q[q.arm == "C. 무작위+양성률 맞춤"]
                if len(b9) and len(c9):
                    out["%s %s %.2f B→C 변화" % (d, me, al)] = abs(
                        float(c9.cov_all.iloc[0]) - float(b9.cov_all.iloc[0])
                    )
                if len(a9):
                    o9 = q[(q.arm == "A. 시간분할") & (q.method == "주변형")]
                    if me == "Mondrian" and len(o9):
                        out["%s %.2f 집합크기 증가" % (d, al)] = float(a9.set_size.iloc[0]) - float(
                            o9.set_size.iloc[0]
                        )

    # §16.4 큐레이션 감사에서 파생되는 비율 (2026-10-21).
    # 감사 CSV 에는 분자별 행만 있고 비율은 없으므로 여기서 다시 센다.
    def _tf9(col):
        if pd.api.types.is_bool_dtype(col):
            return col
        return col.astype(str).str.strip().str.lower().isin(("true", "1"))

    ya9 = pd.read_csv(os.path.join(RES, "year_audit.csv"))
    sa9 = pd.read_csv(os.path.join(RES, "smiles_audit.csv"))
    out["연도 gap>=5 비율"] = 100.0 * float((ya9.gap >= 5).sum()) / len(ya9)
    for d9 in ("DILIrank", "DICTrank"):
        g9 = sa9[sa9.dataset == d9]
        if len(g9):
            out["%s 이름에 염 비율" % d9] = 100.0 * float(_tf9(g9.salt_in_name).mean())
            out["%s 표준화로 골격 변경 비율" % d9] = (
                100.0 * float((g9.skel_std != g9.skel_raw).sum()) / len(g9)
            )
    return out


dv = derived()
dtok = set()
for v in dv.values():
    try:
        fv = float(v)
    except (TypeError, ValueError):
        continue
    for nd in ND:
        dtok.add("%.*f" % (nd, fv))
        dtok.add("%.*f" % (nd, abs(fv)))  # 본문이 부호를 따로 쓰는 경우
        dtok.add("%.*f" % (nd, fv * 100))

# 절·하위절 번호, 표 구분선, 참조 표기는 수치가 아니므로 검사 대상에서 뺀다.
scan = re.sub(r"§\d+(?:\.\d+)?", " ", body)  # §7.1 등
scan = re.sub(r"^#+ .*$", " ", scan, flags=re.M)  # 제목의 절 번호
scan = re.sub(r"\bL0\d[A-Z]{0,2}\d*\b", " ", scan)  # ATC 코드
scan = re.sub(r"\b(?:NDA|ANDA)\s*\d+\b", " ", scan)  # 신청번호
scan = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", scan)  # 날짜

# 논문 문맥의 기준 상수·정의값. 데이터에서 온 값이 아니라 정의이므로 별도 허용한다.
CONST = {
    "0.5": "무작위 기준 AUROC / 기본 임계값",
    "0.50": "무작위 기준 AUROC",
    "1.0": "이상적 보정기울기 / 양성 가중 기준",
    "0.0": "이상적 보정절편",
    "0.90": "conformal 명목 커버리지",
    "0.80": "conformal 명목 커버리지",
    "0.05": "유의수준 alpha",
    "0.4": "외삽 영역 문턱 (사전 정의)",
    "0.15": "보정셋 비율 격자",
    "0.20": "보정셋 비율 격자",
    "0.25": "보정셋 비율 격자",
    "0.33": "보정셋 비율 격자",
    "0.40": "보정셋 비율 격자",
    "0.1": "conformal alpha",
    "0.2": "conformal alpha",
    "2.0": "ECFP4 반지름",
    "0.3": "유사도 구간 경계",
    "0.6": "유사도 구간 경계",
    "1.4": "hydroxycarbamide DDD (본문 미사용)",
    "0.12": "ATC DDD (본문 미사용)",
}
found = {}
for m in re.finditer(r"(?<![\d.\-])(\d+\.\d{2,})", scan):
    found[m.group(1)] = found.get(m.group(1), 0) + 1
unknown = sorted(
    [k for k in found if k not in tok and k not in dtok and k not in CONST], key=lambda x: -found[x]
)
for k in unknown:
    m = re.search(re.escape(k), scan)
    fails.append(
        "근거 불명 %s (x%d)  …%s"
        % (k, found[k], scan[max(0, m.start() - 70) : m.end() + 25].replace("\n", " "))
    )
n_const = sum(1 for k in found if k in CONST)

# ---- 5b. 백분율(소수 1자리)과 문서에 직접 적은 정수 ----------------------
pcts = {}
for m in re.finditer(r"(?<![\d.])(\d+\.\d)%", scan):
    pcts[m.group(1)] = pcts.get(m.group(1), 0) + 1
pct_bad = [k for k in pcts if k not in tok and k not in dtok]
for k in pct_bad:
    m = re.search(re.escape(k) + "%", scan)
    fails.append(
        "근거 불명 백분율 %s%% (x%d)  …%s"
        % (k, pcts[k], scan[max(0, m.start() - 70) : m.end() + 20].replace("\n", " "))
    )

# 생성기가 문장에 직접 적은 정수·계보 값이 실제 데이터와 맞는가
proc = os.path.join(ROOT, "data", "processed")
lab = io.open(os.path.join(RES, "label_audit_summary.txt"), encoding="utf-8").read()
bcv = io.open(os.path.join(RES, "baseline_cv_summary.txt"), encoding="utf-8").read()
cens_s = io.open(os.path.join(RES, "censoring_summary.txt"), encoding="utf-8").read()
LIN = {
    "DILIrank": dict(
        f="dilirank2_with_year.csv", n=796, tr=640, te=156, pos=48, neg=108, ptr=0.6546875, scaf=542
    ),
    "DICTrank": dict(
        f="dictrank_processed.csv",
        n=919,
        tr=652,
        te=267,
        pos=164,
        neg=103,
        ptr=0.7699386503067485,
        scaf=638,
    ),
}
for d, c in LIN.items():
    df = pd.read_csv(os.path.join(proc, c["f"])).dropna(subset=["approval_year", "smiles", "y"])
    tr = df[df.approval_year < 2010]
    te = df[df.approval_year >= 2010]
    chk(len(df) == c["n"], "%s 분석집합 실측 %d != 문서 %d" % (d, len(df), c["n"]))
    chk(len(tr) == c["tr"], "%s 학습군 실측 %d != 문서 %d" % (d, len(tr), c["tr"]))
    chk(len(te) == c["te"], "%s 검증군 실측 %d != 문서 %d" % (d, len(te), c["te"]))
    chk(
        int(te.y.sum()) == c["pos"],
        "%s 검증 양성 실측 %d != 문서 %d" % (d, int(te.y.sum()), c["pos"]),
    )
    chk(
        int((1 - te.y).sum()) == c["neg"],
        "%s 검증 음성 실측 %d != 문서 %d" % (d, int((1 - te.y).sum()), c["neg"]),
    )
    chk(
        abs(tr.y.mean() - c["ptr"]) < 1e-12,
        "%s 학습군 양성률 실측 %.10f != 문서 %.10f" % (d, tr.y.mean(), c["ptr"]),
    )
    chk(
        "| %d | %d | %d/%d |" % (c["n"], c["te"], c["pos"], c["neg"]) in s0_doc
        or "%d (%d / %d)" % (c["n"], c["tr"], c["te"]) in s0_doc,
        "%s 계보 정수가 문서 표에 없다" % d,
    )
    chk("고유 스캐폴드 %d개" % c["scaf"] in bcv, "%s 스캐폴드 %d 가 원본에 없다" % (d, c["scaf"]))
    chk(
        "스캐폴드는 DILIrank 542개" in s0_doc or str(c["scaf"]) in s0_doc,
        "%s 스캐폴드 수가 문서에 없다" % d,
    )
# 제외 경로 숫자가 label_audit_summary.txt 에 실제로 있는지
for v in (
    "1336",
    "1318",
    "1292",
    "354",
    "107",
    "982",
    "1185",
    "568",
    "414",
    "868",
    "343",
    "186",
    "266",
):
    chk(v in lab.replace(",", ""), "제외 경로 값 %s 가 label_audit_summary.txt 에 없다" % v)
    chk(v in s0_doc.replace(",", ""), "제외 경로 값 %s 가 문서에 없다" % v)
# 노출 구간 양성률: 문서 표를 censoring_summary.txt 와 한 칸씩 대조
_blk = cens_s.split("[검열 근거]")[1:]
for d, b in zip(("DILIrank", "DICTrank"), _blk):
    bands = re.findall(r"([\d]+-\s*[\d]+년) \(n=\s*(\d+), 승인 ([\d\-]+)\): 양성 ([\d.]+)%", b)
    chk(len(bands) == 5, "%s 노출 구간 파싱 %d개" % (d, len(bands)))
    cells = " | ".join("%s%% (n=%s)" % (x[3], x[1]) for x in bands)
    chk(
        "| %s 양성률 | %s |" % (d, cells) in s0_doc,
        "%s 노출 구간 행이 원본과 다르다: 기대 %s" % (d, cells),
    )
    for x in bands:
        chk(x[2] in s0_doc, "%s 노출 구간 연도범위 %s 가 문서에 없다" % (d, x[2]))

# ATC 확보율
aa2 = pd.read_csv(os.path.join(RES, "atc_audit.csv"))
for d in ("DILIrank", "DICTrank"):
    got = int(aa2[aa2.dataset == d].atc.notna().sum())
    tot = int((aa2.dataset == d).sum())
    chk("%s %d/%d" % (d, got, tot) in s0_doc, "%s ATC 확보 %d/%d 가 문서와 다르다" % (d, got, tot))
# L 군 구성
al = pd.read_csv(os.path.join(RES, "atc_review_L.csv"))
comp = {k: int((al.atc.str[:3] == k).sum()) for k in ("L01", "L02", "L03", "L04")}
chk(
    comp == {"L01": 215, "L02": 30, "L03": 2, "L04": 36},
    "L 군 구성 실측 %s != 문서 215/30/2/36" % comp,
)
alt = al[al.approval_year >= 2010]
compt = {k: int((alt.atc.str[:3] == k).sum()) for k in ("L01", "L02", "L04")}
chk(compt == {"L01": 94, "L02": 8, "L04": 24}, "검증군 L 군 구성 실측 %s != 문서 94/8/24" % compt)

# ---- 5.9 재고 점검 — 내용이 조용히 사라지지 않았는가 -------------------
# 2026-10-19 사고: 한계 18번을 고치는 패치가 블록 끝을 'w()' 로 잡아
# 19~24번 항목을 함께 지웠다. 번호는 1..18 로 연속이어서 그 당시의
# 어떤 검증기도 잡지 못했다. 그래서 '연속성' 이 아니라 '개수가 줄지
# 않았는가' 를 센다. 기준치는 파일에 적어 두고 --save 로만 갱신한다.
_lim_i = s.find("\n## 36. 한계")
chk(_lim_i > 0, "§35 한계 절을 찾지 못했다")
_lim_body = s[_lim_i:] if _lim_i > 0 else ""
_lim_nums = [int(m.group(1)) for m in re.finditer(r"^(\d+)\. \*\*", _lim_body, re.M)]
chk(_lim_nums == list(range(1, len(_lim_nums) + 1)), "한계 항목 번호 불연속: %s" % _lim_nums)
INV = os.path.join(RES, "report_inventory.txt")
_inv_now = {
    "절": len(secs),
    "하위절": sum(len(v) for v in subs.values()),
    "표": ntab,
    "한계항목": len(_lim_nums),
}
if "--save" in sys.argv:
    io.open(INV, "w", encoding="utf-8").write(
        "\n".join("%s\t%d" % (k, v) for k, v in sorted(_inv_now.items())) + "\n"
    )
    print("재고 기준치 저장: %s" % INV)
elif os.path.exists(INV):
    _inv_old = dict(
        (ln.split("\t")[0], int(ln.split("\t")[1]))
        for ln in io.open(INV, encoding="utf-8").read().split("\n")
        if ln
    )
    for k, v0 in sorted(_inv_old.items()):
        v1 = _inv_now.get(k, -1)
        chk(
            v1 >= v0,
            "%s 개수가 줄었다: %d -> %d (내용이 사라졌을 수 있다. "
            "의도한 변경이면 verify9.py --save 로 기준치를 갱신할 것)" % (k, v0, v1),
        )
else:
    print("  [주의] 재고 기준치 파일이 없다 — verify9.py --save 로 한 번 만들 것")

# ---- 5.95 결과 파일 재고 — 계산했는데 아무도 안 읽는 것이 있는가 -------
# 왜 필요한가 — 2026-10-21. results 에 CSV 가 99개 있는데 생성기가
# 이름으로 읽는 것은 78개였다. 나머지 21개 중 대부분은 귀무분포 전체나
# 재개 캐시 같은 중간 파일이지만, 큐레이션 감사 다섯 개(연도·라벨·구조)와
# 결정성 두 개는 문서에 숫자가 한 줄도 안 들어가 있었다.
#
# 이 구멍은 verify9 의 다른 점검으로는 안 잡힌다. 그 점검들은 '문서에
# 있는 수치가 근거가 있는가'를 보기 때문에, 근거만 있고 문서에 안 들어간
# 것은 조용히 넘어간다. 그래서 여기서 반대 방향을 본다.
#
# 아래는 '생성기가 안 읽어도 되는 이유가 있는 파일'이다. 새 결과 파일이
# 생기면 생성기가 읽든가 이 목록에 이유와 함께 등록되든가 해야 하고,
# 둘 다 아니면 실패로 잡는다.
INTERMEDIATE = {
    # 귀무분포 전체 — 문서에는 평균·백분위·p 만 들어간다(요약 CSV 경유).
    "era_permutation_null_dilirank.csv": "귀무분포 2000개 원자료",
    "era_permutation_null_dictrank.csv": "귀무분포 2000개 원자료",
    "permutation_null_dilirank.csv": "귀무분포 원자료",
    "permutation_null_dictrank.csv": "귀무분포 원자료",
    # 재개 캐시 — 중단된 순열을 이어 돌리기 위한 것.
    "era_partial_dilirank.csv": "재개 캐시",
    "era_partial_dictrank.csv": "재개 캐시",
    "permutation_partial_dilirank.csv": "재개 캐시",
    "permutation_partial_dictrank.csv": "재개 캐시",
    "matched_n_era_permutation_quick100.csv": "앞 100개 대조 기록",
    # 분자별·반복별 원자료 — 문서에는 집계값만 들어간다.
    "prequential_retraining_pred.csv": "분자별 예측 원자료",
    "retrain_interval_pred.csv": "분자별 예측 원자료",
    "prevalence_sweep_reps.csv": "반복별 원자료",
    # 수동 검토용 목록 — 수치가 아니다.
    "atc_review_L.csv": "수동 검토 대상 목록",
}

# 아직 문서에 안 들어간 것. 들어가면 이 목록에서 지운다.
# 2026-10-21: 비었다. 큐레이션 감사 다섯 개는 §16.4 로, 결정성 두 개는
# §3.1 로 들어갔다. 이 칸이 비어 있는 것이 정상이고, 여기에 뭔가
# 쌓이기 시작하면 '계산은 했는데 안 쓰는 결과' 가 다시 생기는 중이다.
PENDING_DOC = {}

_gen = ""
for _g in ("make_report.py", "make_tables.py", "make_figures.py"):
    _gp = os.path.join(ROOT, "src", _g)
    if os.path.exists(_gp):
        _gen += io.open(_gp, encoding="utf-8").read()
_unread, _declared = [], []
for _f in sorted(glob.glob(os.path.join(RES, "*.csv"))):
    _b = os.path.basename(_f)
    if _b in _gen:
        continue
    if _b in INTERMEDIATE or _b in PENDING_DOC:
        _declared.append(_b)
        continue
    _unread.append(_b)
for _b in _unread:
    fails.append(
        "results/%s 를 생성기가 읽지 않는다 — 문서에 넣든가 "
        "verify9.py 의 INTERMEDIATE 에 이유를 적고 등록할 것" % _b
    )
_gone = [
    k for k in list(INTERMEDIATE) + list(PENDING_DOC) if not os.path.exists(os.path.join(RES, k))
]
_nowread = [k for k in PENDING_DOC if k in _gen]
print(
    "  [재고] 결과 CSV %d개 — 생성기가 읽음 %d / 중간·원자료 선언 %d / "
    "문서 편입 대기 %d / 설명 없음 %d"
    % (
        len(glob.glob(os.path.join(RES, "*.csv"))),
        len(glob.glob(os.path.join(RES, "*.csv"))) - len(_declared) - len(_unread),
        len([k for k in INTERMEDIATE if k in _declared]),
        len([k for k in PENDING_DOC if k in _declared]),
        len(_unread),
    )
)
for _b in sorted(PENDING_DOC):
    if _b in _declared:
        print("         편입 대기: %-34s %s" % (_b, PENDING_DOC[_b]))
for _b in _nowread:
    print("         [정리] %s 는 이제 생성기가 읽는다 — PENDING_DOC 에서 지울 것" % _b)
for _b in _gone:
    print("         [정리] %s 가 없다 — 목록에서 지우거나 만드는 스크립트를 볼 것" % _b)

# ---- 5.94 결과 CSV 에 완전히 같은 행이 중복되는가 (2026-10-22) ---------
# 왜 필요한가 — 재개 기능이 있는 스크립트는 결과를 append 모드로 덧붙인다.
# 같은 조합을 두 번 돌리면 행이 두 벌 쌓이는데, 파일은 멀쩡해 보이고
# 생성기도 .iloc[0] 으로 읽으면 눈치채지 못한다. 실제로
# mondrian_conformal.csv 가 48행 중 24행이 중복이었고, 그 파일을 읽는
# fdr_all_tests.py 가 검정 개수를 73개로 세어 BH 보정을 돌렸다
# (실제 고유 검정은 57개다). 다중비교 보정은 개수가 틀리면 q 가 전부
# 달라지므로 그냥 둘 수 없다.
#
# 순열 귀무분포는 예외다 — 유한 표본에서 AUC·ECE 가 같은 값을 여러 번
# 내는 것이 정상이고, 중복을 지우면 분포가 망가진다.
DUP_OK = {
    "era_permutation_null_dilirank.csv": "귀무분포 — 값 중복이 정상",
    "era_permutation_null_dictrank.csv": "귀무분포 — 값 중복이 정상",
    "permutation_null_dilirank.csv": "귀무분포 — 값 중복이 정상",
    "permutation_null_dictrank.csv": "귀무분포 — 값 중복이 정상",
    "era_partial_dilirank.csv": "귀무분포 재개 캐시",
    "era_partial_dictrank.csv": "귀무분포 재개 캐시",
    "permutation_partial_dilirank.csv": "귀무분포 재개 캐시",
    "permutation_partial_dictrank.csv": "귀무분포 재개 캐시",
    "matched_n_era_permutation_quick100.csv": "귀무분포 앞 100개 기록",
}
_dups = []
for _f in sorted(glob.glob(os.path.join(RES, "*.csv"))):
    _b = os.path.basename(_f)
    if _b in DUP_OK:
        continue
    try:
        _d = pd.read_csv(_f)
    except Exception:
        continue
    if len(_d) == 0:
        continue
    _n = int(_d.duplicated(keep="first").sum())
    if _n:
        _dups.append((_b, len(_d), _n))
print("  [중복] 결과 CSV 중 완전히 같은 행이 있는 파일 %d개" % len(_dups))
for _b, _tot, _n in _dups:
    fails.append(
        "results/%s 에 완전히 같은 행이 %d개 있다 (전체 %d행) — "
        "재개 캐시가 같은 조합을 두 번 덧붙였다. 그 스크립트를 "
        "--fresh 로 다시 돌릴 것" % (_b, _n, _tot)
    )

# ---- 5.95b §33 이동량 맞춤이 성공했는가 (2026-10-22) -------------------
# 왜 필요한가 — §33 은 한계 18 을 닫는 절이고, 그 주장은 비시간 분할의
# 이동 AUC 를 시간분할에 맞췄다는 전제 위에 선다. 맞춤은 2차 정밀
# alpha 탐색에서 성립하는데, 그 호출이 손으로만 돌아가 재현 경로에
# 없었다. 거친 격자만 돌리면 여섯 설정 전부 맞춤에 실패하고(차
# 0.063~0.081), 그래도 문서는 생성되고 verify9·verify10 은 통과한다 —
# 문서가 실패한 결과를 그대로 옮겨 적기 때문이다. 그래서 여기서 본다.
_sm2m9 = pd.read_csv(os.path.join(RES, "shift_matched_v2_matched.csv"))
_mt9 = _sm2m9["matched"].astype(str).str.strip().str.lower().isin(("true", "1"))
_nok9 = int(_mt9.sum())
print(
    "  [§33] 이동량 맞춤 성공 %d / %d   이동 AUC 차 %.4f~%.4f"
    % (_nok9, len(_sm2m9), _sm2m9.shift_auc_gap.abs().min(), _sm2m9.shift_auc_gap.abs().max())
)
chk(
    _nok9 == len(_sm2m9),
    "§33 이동량 맞춤이 %d/%d 설정에서 실패했다 (이동 AUC 차 최대 %.4f) — "
    "2차 정밀 탐색이 빠졌다. python src/shift_matched_v2.py "
    "--alphas 0.29,0.31,0.33,0.46,0.48,0.5 를 돌릴 것"
    % (len(_sm2m9) - _nok9, len(_sm2m9), _sm2m9.shift_auc_gap.abs().max()),
)

# ---- 5.96 결과가 코드보다 오래됐는가 (재개 캐시 감시) -------------------
# 왜 필요한가 — 2026-10-22. 재개 기능이 있는 스크립트 여덟 개는 이미 끝난
# 부분을 건너뛴다. 그래서 코드를 고친 뒤 재실행해도 **옛 값을 그대로
# 들고 있을 수 있다.** 실제로 10-20 의 정밀도 수정 뒤 두 번의 전체
# 재실행을 돌렸는데도 horizon_ratio·prequential_retraining·
# shift_matched_v2·tiebreak_averaged 네 파일이 수정 전 값이었다.
# verify9·verify10 은 문서와 결과 파일이 맞는지만 보기 때문에, 문서를
# 그 옛 파일에서 생성하면 둘 다 통과한다. 그래서 시각을 본다.
#
# mtime 은 복제·체크아웃으로 리셋된다. 여유를 크게 두면(300초로 해봤다)
# 실제 구멍을 놓친다 — shift_matched_v2 는 코드가 결과보다 60초 새로울
# 뿐인데 그 60초 사이에 고친 것이 결과에 안 들어가 있었다. 그래서 여유
# 대신 갓 받은 저장소인지를 본다. 복제 직후에는 src/*.py 의 시각이
# 거의 같은 순간에 몰리므로, 그 퍼짐이 SPREAD_MIN 초 미만이면 이 검사를
# 건너뛴다. 작업 중인 저장소에서는 퍼짐이 크므로 여유 없이 엄격히 본다.
import ast

STALE_MARGIN = 0
SPREAD_MIN = 120


def _producers(src_dir, res_dir):
    """결과 파일 -> 그 파일을 쓰는 스크립트들.

    ast 로 본다. to_csv / write_text / open(...,"w") 의 첫 인자에서
    문자열 리터럴을 모으고, 인자가 이름이면 모듈 수준 상수를 따라간다.
    'name.csv' 가 본문에 등장하는 것만으로는 생산자로 보지 않는다 —
    읽기만 하는 스크립트가 섞여 들어오기 때문이다.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(src_dir, "*.py"))):
        try:
            tree = ast.parse(io.open(f, encoding="utf-8").read())
        except SyntaxError:
            continue
        const = {}

        def lits(node):
            return {
                n.value
                for n in ast.walk(node)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and re.fullmatch(r"[A-Za-z0-9_.\-]+\.(csv|txt|json)", n.value)
            }

        for node in tree.body:  # 모듈 수준 상수만
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        v = lits(node.value)
                        if v:
                            const[t.id] = v
        names = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            label = (
                fn.attr
                if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name) else ""
            )
            if label not in ("to_csv", "write_text", "savefig", "open"):
                continue
            if label == "open":
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = str(kw.value.value)
                if "w" not in mode and "a" not in mode:
                    continue
            if not node.args:
                continue
            arg = node.args[0]
            names |= lits(arg)
            for n in ast.walk(arg):
                if isinstance(n, ast.Name) and n.id in const:
                    names |= const[n.id]
        for n in names:
            out.setdefault(n, set()).add(os.path.basename(f))

    # 보조: 쓰기 경로를 함수 인자로 넘기는 스크립트는 ast 로 못 따라간다
    # (shift_matched_v2.py 의 append(df, path) 처럼). 이 프로젝트는 결과
    # 파일 이름을 스크립트 이름으로 시작하게 붙이므로 그 규칙으로 메운다.
    stems = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(src_dir, "*.py"))}
    for f in glob.glob(os.path.join(res_dir, "*")):
        n = os.path.basename(f)
        if n in out:
            continue
        base = os.path.splitext(n)[0]
        cands = sorted(
            (st for st in stems if base == st or base.startswith(st + "_")), key=len, reverse=True
        )
        if cands:
            out[n] = {cands[0] + ".py"}
    return out


_GEN9 = {
    "make_report.py",
    "make_tables.py",
    "make_figures.py",
    "verify9.py",
    "verify10.py",
    "verify11.py",
    "verify12.py",
    "verify_state.py",
    "diff_results.py",
    "diff_ladder.py",
    "record_env.py",
}
# 보고서·표·그림이 읽는 파일 목록 (5.95 의 _gen 을 다시 쓴다)
need = set(re.findall(r'cv\("([a-z0-9_]+\.csv)"\)', _gen))
need |= set(re.findall(r'txt\("([a-z0-9_]+\.txt)"\)', _gen))
need |= set(re.findall(r'RES / "([a-z0-9_]+\.csv)"', _gen))

_prod9 = _producers(os.path.join(ROOT, "src"), RES)
_smt = [os.path.getmtime(x) for x in glob.glob(os.path.join(ROOT, "src", "*.py"))]
_spread = (max(_smt) - min(_smt)) if _smt else 0.0
_fresh_clone = _spread < SPREAD_MIN
_stale, _noprod = [], []
for _n in sorted(need):
    _fp = os.path.join(RES, _n)
    if not os.path.exists(_fp):
        continue
    _srcs = sorted(x for x in _prod9.get(_n, set()) if x not in _GEN9)
    if not _srcs:
        _noprod.append(_n)
        continue
    for _sc in _srcs:
        _sp = os.path.join(ROOT, "src", _sc)
        if not os.path.exists(_sp):
            continue
        _d = os.path.getmtime(_sp) - os.path.getmtime(_fp)
        if _d > STALE_MARGIN:
            _stale.append((_n, _sc, _d))
if _fresh_clone:
    _stale = []
# 정렬·주석만 바뀐 것은 결과를 바꾸지 않는다. state_manifest.json 에
# 기록된 구문 트리 해시와 지금이 같으면 코드가 실질적으로 그대로라는
# 뜻이므로 건너뛴다. 2026-10-22 에 black 으로 전체를 정렬했더니 시각만
# 보는 검사가 40건 넘게 오경보를 냈다.
_MANI = os.path.join(RES, "state_manifest.json")
_astmap = {}
if os.path.exists(_MANI):
    try:
        _astmap = json.load(io.open(_MANI, encoding="utf-8")).get("src_ast", {})
    except Exception:
        _astmap = {}


def _ast_same(scr):
    rec = _astmap.get("src/" + scr)
    if not rec:
        return False
    try:
        t = ast.parse(io.open(os.path.join(ROOT, "src", scr), encoding="utf-8").read())
    except SyntaxError:
        return False
    return hashlib.sha256(ast.dump(t).encode("utf-8")).hexdigest() == rec


_cosmetic = [x for x in _stale if _ast_same(x[1])]
_stale = [x for x in _stale if not _ast_same(x[1])]
if _cosmetic:
    print("         (정렬·주석만 바뀐 것 %d건은 건너뛴다 — 구문 트리가 같다)" % len(_cosmetic))
if not _astmap:
    print(
        "         (state_manifest.json 에 구문 트리 기록이 없다 — "
        "verify_state.py --save 를 한 번 돌릴 것)"
    )
    _stale = []
for _n, _sc, _d in _stale:
    fails.append(
        "results/%s 가 src/%s 보다 %.1f시간 오래됐다 — 그 스크립트를 "
        "다시 돌릴 것(재개 캐시가 건너뛰었을 수 있으니 --fresh)" % (_n, _sc, _d / 3600.0)
    )
print(
    "  [시각] 보고서가 읽는 파일 %d개 중 코드보다 오래된 것 %d개 / "
    "생산자 불명 %d개%s"
    % (
        len(need),
        len(_stale),
        len(_noprod),
        (
            "   (src 시각 퍼짐 %.0f초 — 갓 받은 저장소로 보고 검사를 건너뛴다)" % _spread
            if _fresh_clone
            else ""
        ),
    )
)
for _n in _noprod:
    print("         생산자 불명: %s" % _n)

# ---- 6. 생성기 재실행 대조 ---------------------------------------------
h0 = hashlib.sha256(io.open(DOC, "rb").read()).hexdigest()
rc = subprocess.run(
    [sys.executable, os.path.join(ROOT, "src", "make_report.py")], capture_output=True, text=True
)
chk(rc.returncode == 0, "생성기 재실행 실패: %s" % rc.stderr[-300:])
h1 = hashlib.sha256(io.open(DOC, "rb").read()).hexdigest()
chk(h0 == h1, "문서가 생성기 출력과 다르다 — 손으로 고친 부분이 있다 (재실행으로 덮였다)")

print("verify9 — 논문용 보고서 검증")
print("  절 %d개 / 하위절 %d개 / 표 %d개" % (len(secs), sum(len(v) for v in subs.values()), ntab))
print("  한계 항목 %d개 (기준치 대비 감소 없음)" % len(_lim_nums))
print("  2자리 이상 소수: 고유 %d개 (총 %d회)" % (len(found), sum(found.values())))
print("  백분율(소수1자리): 고유 %d개, 근거 불명 %d개" % (len(pcts), len(pct_bad)))
print(
    "  원본 직접 확인 %d / 파생 재계산 %d / 정의 상수 %d / 근거 불명 %d"
    % (
        sum(1 for k in found if k in tok),
        sum(1 for k in found if k not in tok and k in dtok),
        n_const,
        len(unknown),
    )
)
print("  생성기 재실행 해시 일치: %s" % ("예" if h0 == h1 else "아니오"))
print("실패 %d건" % len(fails))
for x in fails:
    print("  X", x)
sys.exit(1 if fails else 0)
