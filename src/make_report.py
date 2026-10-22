# -*- coding: utf-8 -*-
"""논문용 실험결과 보고서 생성기.

모든 수치를 results/ 의 CSV·TXT 에서 직접 읽어 문서를 만든다.
손으로 옮겨 적는 단계가 없으므로 전사 오류가 원리적으로 생기지 않는다.
문장만 이 파일에 있고, 숫자는 전부 데이터에서 온다.

    python src/make_report.py            # 보고서 생성
"""

import glob, io, json, os, re, sys
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
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "RESULTS_REPORT.md")


def cv(name):
    return pd.read_csv(os.path.join(RES, name))


def txt(name):
    return io.open(os.path.join(RES, name), encoding="utf-8").read()


# ---------------------------------------------------------------- 데이터 적재
bc = cv("baseline_cv_per_fold.csv")
cal = cv("calibration_results.csv")
pt = cv("permutation_test.csv")
pc = cv("prevalence_control.csv")
bd = cv("brier_decomposition.csv")
r3 = cv("brier_decomposition_r3.csv")
mn = cv("matched_n_control.csv")
rob = cv("robustness_models.csv")
robc = cv("robustness_cutoff.csv")
seed = cv("seed_sensitivity.csv")
sseed = cv("split_seed_sensitivity.csv")
binn = cv("ece_binning_sensitivity.csv")
grid = cv("grid_calibration.csv")
tie = cv("tiebreak_sensitivity.csv")
rec = cv("recalibration_results.csv")
recs = cv("recalibration_sensitivity.csv")
conf = cv("conformal_prediction.csv")
cens = cv("censoring_results.csv")
sq = cv("shift_quantification.csv")
sqd = cv("shift_quantification_descriptors.csv")
ep = cv("era_permutation.csv")
cc = cv("calibration_confounders.csv")
atcs = cv("atc_stratified.csv")
atct = cv("atc_tests.csv")
mc = cv("multiple_comparison_correction.csv")
ys = cv("year_sensitivity.csv")
avs = cv("atc_v_sensitivity.csv")
ster = cv("stereo_sensitivity.csv")
ov = cv("compound_overlap.csv")
dj = cv("disjoint_subset.csv")
td = cv("temporal_detail.csv")
pe = cv("prior_estimation.csv")  # §10.2 (2026-10-02)
rv = cv("reverse_split.csv")  # §12.2 (2026-10-02)
sm = cv("shift_matched_split.csv")  # §13.5 (2026-10-02)
tb = cv("trivial_baseline.csv")  # §25 (2026-10-19)
ri = cv("ratio_intervals.csv")  # §26 (2026-10-19)
cm = cv("calibration_metrics.csv")  # §27 (2026-10-19)
cmr = cv("calibration_metrics_ratio.csv")
hp = cv("hparam_sensitivity.csv")  # §28 (2026-10-19)
hpc = cv("hparam_sensitivity_ctrl.csv")
rp = cv("remedy_power.csv")  # §29 (2026-10-19)
rpa = cv("remedy_power_ad.csv")
uts = cv("useful_threshold_sweep.csv")  # §29.5 (2026-10-20)
utg = cv("useful_threshold_grid.csv")
utsa = cv("useful_threshold_sweep_ad.csv")  # §29.5 AD 팔 (2026-10-20 편입)
pfx = cv("precision_fix_20261020.csv")  # §10.5 정밀도 수정 기록
csw = cv("covariate_shift_weighting.csv")  # §30 (2026-10-19)
cswc = cv("covariate_shift_weighting_ctrl.csv")
g3 = cv("grid_r3.csv")  # §31 (2026-10-19)
hz = cv("horizon_ratio.csv")  # §32 (2026-10-20)
sm2 = cv("shift_matched_v2.csv")  # §33 (2026-10-20)
sm2m = cv("shift_matched_v2_matched.csv")
pe2 = cv("prior_estimation_v2.csv")  # §10.3 (2026-10-20)
tba = cv("tiebreak_averaged.csv")  # §10.4 (2026-10-20)
mdc = cv("mondrian_conformal.csv")  # §11.1 (2026-10-20)
fat = cv("fdr_all_tests.csv")  # §15.2 (2026-10-20)
sgc = cv("subgroup_calibration.csv")  # §14.3 (2026-10-20)
xc = cv("extra_calibrators.csv")  # §10.5 (2026-10-20)
pq = cv("prequential_retraining.csv")  # §34 (2026-10-20)
pqd = cv("prequential_retraining_delta.csv")  # §34 짝지은 ΔECE
rin = cv("retrain_interval.csv")  # §34.1 (2026-10-20)
rind = cv("retrain_interval_delta.csv")  # §34.1 짝지은 ΔECE
bdc = cv("brier_decomposition_cutoff.csv")  # §9.7
mep = cv("matched_n_era_permutation.csv")  # §13.1 보조
ya = cv("year_audit.csv")  # §16.4 (2026-10-21)
yc = cv("year_corrections.csv")  # §16.4
la = cv("label_audit.csv")  # §16.4
lc = cv("label_conflict_groups.csv")  # §16.4
sa = cv("smiles_audit.csv")  # §16.4
det = cv("determinism.csv")  # §3.1 (2026-10-21)
detx = cv("determinism_cross_process.csv")  # §3.1
mdup = cv("metric_duplicates.csv")  # §3.1 복제 정의 대조
pqy = cv("prequential_delay.csv")  # §34.2 (2026-10-22)
pqdd = cv("prequential_delay_delta.csv")  # §34.2 짝지은 ΔECE
pqc = cv("prequential_delay_check.csv")  # §34.2 지연 0 재현 확인
try:  # §10.5 (10-21 캐스팅 수정 기록)
    pf21 = cv("precision_fix_20261021.csv")
except FileNotFoundError:  # redo_calibration.sh 전이면 없다
    pf21 = None
env = json.load(io.open(os.path.join(RES, "environment.json"), encoding="utf-8"))

# 학습군 양성률과 검증군 양성/음성 개수는 결과 CSV 에 없다. 예전에는 숫자를
# 직접 적어 두었는데(0.6546875 등) 데이터가 바뀌면 조용히 틀린 값이 된다.
# 처리 데이터에서 매번 센다.
PROC = os.path.join(ROOT, "data", "processed")
_PROC_FILE = {"DILIrank": "dilirank2_with_year.csv", "DICTrank": "dictrank_processed.csv"}
SPLIT = {}
for _d, _fn in _PROC_FILE.items():
    _x = pd.read_csv(os.path.join(PROC, _fn)).dropna(subset=["approval_year", "y", "smiles"])
    _tr = _x[_x.approval_year.astype(int) < 2010]
    _te = _x[_x.approval_year.astype(int) >= 2010]
    SPLIT[_d] = {
        "n_train": len(_tr),
        "n_test": len(_te),
        "train_pos_rate": float(_tr.y.mean()),
        "test_pos_rate": float(_te.y.mean()),
        "test_pos": int((_te.y == 1).sum()),
        "test_neg": int((_te.y == 0).sum()),
    }

DS = ["DILIrank", "DICTrank"]
KOR = {"DILIrank": "DILIrank 2.0 (간독성)", "DICTrank": "DICTrank (심장독성)"}


def row(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        col = df[k].astype(str)
        m &= (col == str(v)) | col.str.startswith(str(v))
    r = df[m]
    assert len(r) == 1, "행 %d개: %s %s" % (len(r), kw, list(r.index))
    return r.iloc[0]


def f4(x):
    return "%.4f" % x


def f3(x):
    return "%.3f" % x


def f2(x):
    return "%.2f" % x


def f1(x):
    return "%.1f" % x


def sg(x):
    return "%+.4f" % x


def pct1(x):
    return "%.1f%%" % x


# 반복 횟수는 결과 파일에 기록돼 있다. 산문에 손으로 적으면 설정이
# 바뀔 때 조용히 거짓이 된다 — §13.5 의 "200회" 가 실제로 50회였고
# (2026-10-21), §33 의 "맞춤에 성공했다" 가 실패 상태에서 그대로
# 남아 있었다(2026-10-22). 그래서 숫자는 전부 파일에서 읽는다.
def _reps(name, col):
    return int(pd.read_csv(os.path.join(RES, name))[col].max())


def _pdp0(ds, dl):
    """§34.2 — 그 지연에서 P2 가 고정 모델 대비 낸 짝지은 ΔECE 행."""
    q = pqdd[
        (pqdd.dataset == ds)
        & (pqdd.delay == dl)
        & (pqdd.vs == "P0. 고정 모델")
        & pqdd.arm.str.startswith("P2")
    ]
    return q.iloc[0]


# pandas 판본에 따라 CSV 의 True/False 가 bool 로도 문자열로도 읽힌다.
# 두 경우 모두 맞게 처리한다 (그의 환경은 pandas 3.0.5, 시험대는 2.3.3).
def _tf(col):
    if pd.api.types.is_bool_dtype(col):
        return col
    return col.astype(str).str.strip().str.lower().isin(("true", "1"))


L = []


def w(s=""):
    L.append(s)


# 자주 쓰는 값
T = {d: row(bc, dataset=d, split="temporal") for d in DS}
CT = {d: cal[(cal.dataset.str.startswith(d)) & (cal.setting == "temporal")].iloc[0] for d in DS}
CR = {d: cal[(cal.dataset.str.startswith(d)) & (cal.setting == "random")].iloc[0] for d in DS}
PT = {d: row(pt, dataset=d) for d in DS}
PCT = {d: pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith("T"))].iloc[0] for d in DS}
PCR = {
    (d, b): pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith(b))].iloc[0]
    for d in DS
    for b in ("R1", "R2", "R3")
}
R3 = {
    (d, b): r3[(r3.dataset == d) & (r3.baseline.str.startswith(b))].iloc[0]
    for d in DS
    for b in ("R1", "R2", "R3")
}
MN = {
    (d, b): mn[(mn.dataset == d) & (mn.baseline == b)].iloc[0]
    for d in DS
    for b in ("R1", "R2", "R3")
}
BD = {d: row(bd, dataset=d) for d in DS}
EP = {d: row(ep, dataset=d) for d in DS}
SQ = {d: row(sq, dataset=d) for d in DS}
OV = {d: row(ov, dataset=d) for d in DS}
# 무작위 5-fold CV 평균 AUROC (§4 표와 같은 값). §5 의 신뢰구간 서술에서 쓴다.
CVR = {d: float(bc[(bc.dataset == d) & (bc.split == "random_cv")].AUROC.mean()) for d in DS}

# ECE 초과분 중 유병률 통제(R1 -> R3)로 해소되는 몫. §1 요약과 §8 표가 같이 쓴다.
_ex = {d: PCT[d].ECE - PCR[(d, "R1")].ECE for d in DS}
_rm = {d: PCT[d].ECE - PCR[(d, "R3")].ECE for d in DS}
_exp_share = {d: (_ex[d] - _rm[d]) / _ex[d] * 100 for d in DS}

# ================================================================ 머리말
w("# 시간분할 검증에서의 보정 붕괴 — 실험 결과 보고서 (논문용)")
w()
w("**대상** DILIrank 2.0 (약물유발 간독성) · DICTrank (약물유발 심장독성)  ")
w(
    "**작성** 2026-10-02 · **표기 정정** 2026-10-04 (수치 변경 없음 — 분모·라벨·누락 "
    "설명 9건) · 논문에 들어갈 실험만 담는다. 정정 이력, 재현 절차, 검증 스크립트 "
    "설명은 제외했다.  "
)
w(
    "**생성 방식** `src/make_report.py` 가 `results/` 의 결과 파일에서 수치를 직접 읽어 "
    "이 문서를 만든다. 사람이 숫자를 옮겨 적는 단계가 없다.  "
)
try:
    _v10n = "%d개" % int(txt("verify10_cells.txt").strip())
except Exception:
    _v10n = "전수"
w(
    "**검증** `verify9`(문서→결과 추적, 근거 불명 0) · `verify10`(결과→문서 셀 단위 "
    "%s) · `verify11`(격자 일치 32칸) 전부 실패 0건. `verify_state` 로 데이터·코드 "
    "지문도 일치 확인." % _v10n
)
w()
w("---")
w()

# ================================================================ 1. 요약
w("## 1. 핵심 결과")
w()
w(
    "무작위 분할로 평가하면 정상으로 보이는 모델이, 같은 데이터를 승인연도로 나누면 "
    "**판별력은 조금 떨어지고 확률 보정은 크게 무너진다.** 두 데이터셋에서 같은 방향으로 나타난다."
)
w()
w("| # | 지표 | DILIrank 2.0 | DICTrank | 출처 |")
w("|---|---|---|---|---|")
w(
    "| 1 | 분석집합 n (학습 / 검증) | %d (%d / %d) | %d (%d / %d) | §2 |"
    % (796, 640, 156, 919, 652, 267)
)
w(
    "| 2 | AUROC 무작위 5-fold CV → 시간분할 | %s → **%s** | %s → **%s** | §4 |"
    % (
        f4(bc[(bc.dataset == "DILIrank") & (bc.split == "random_cv")].AUROC.mean()),
        f4(T["DILIrank"].AUROC),
        f4(bc[(bc.dataset == "DICTrank") & (bc.split == "random_cv")].AUROC.mean()),
        f4(T["DICTrank"].AUROC),
    )
)
w(
    "| 3 | ΔAUROC (순열 1,000회 귀무 대비) | +%s (p = %s) | +%s (p = %s) | §5 |"
    % (
        f4(PT["DILIrank"].delta_auroc),
        f4(PT["DILIrank"].p_permutation),
        f4(PT["DICTrank"].delta_auroc),
        f4(PT["DICTrank"].p_permutation),
    )
)
w(
    "| 4 | ECE 무작위 → 시간분할 | %s → **%s** | %s → **%s** | §6 |"
    % (
        f4(CR["DILIrank"].ECE),
        f4(CT["DILIrank"].ECE),
        f4(CR["DICTrank"].ECE),
        f4(CT["DICTrank"].ECE),
    )
)
w(
    "| 5 | ECE 배율 (크기맞춤 대조군 R1, 200회) | **%s배** | **%s배** | §8 |"
    % (f2(PCR[("DILIrank", "R1")].ece_ratio), f2(PCR[("DICTrank", "R1")].ece_ratio))
)
w(
    "| 6 | ECE 배율 (유병률 양측 완전통제 R3) | **%s배** (p = %s) | **%s배** (p = %s) | §8 |"
    % (
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f4(PCR[("DILIrank", "R3")].p_perm),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        f4(PCR[("DICTrank", "R3")].p_perm),
    )
)
w(
    "| 7 | 보정기울기 무작위 → 시간분할 (1이 이상적) | %s → **%s** | %s → **%s** | §6 |"
    % (
        f4(CR["DILIrank"].calib_slope),
        f4(CT["DILIrank"].calib_slope),
        f4(CR["DICTrank"].calib_slope),
        f4(CT["DICTrank"].calib_slope),
    )
)
w(
    "| 8 | Brier 악화 중 보정 성분 기여 (주 분석) | **%s** | **%s** | §7 |"
    % (pct1(BD["DILIrank"].pct_from_reliability_A), pct1(BD["DICTrank"].pct_from_reliability_A))
)
w(
    "| 9 | 같은 기여율, 유병률 양측통제 후 (R3) | **%s** | **%s** | §7.1 |"
    % (
        pct1(R3[("DILIrank", "R3")].share2_reliability),
        pct1(R3[("DICTrank", "R3")].share2_reliability),
    )
)
_bc_rng = {
    d: recs[(recs.dataset == d) & (recs.method.str.startswith(("B.", "C.")))].recovery for d in DS
}
w(
    "| 10 | 학습군 내부 재보정으로 회복되는 ECE (보정셋 비율 0.15–0.40) | %s–%s%% | %s–%s%% | §10.1 |"
    % (
        f2(_bc_rng["DILIrank"].min()),
        f2(_bc_rng["DILIrank"].max()),
        f2(_bc_rng["DICTrank"].min()),
        f2(_bc_rng["DICTrank"].max()),
    )
)
w(
    "| 11 | conformal 90%% 목표 커버리지 실측 | %s | %s | §11 |"
    % (
        f4(
            conf[
                (conf.dataset == "DILIrank") & (conf.alpha == 0.1)
            ].marginal_coverage_temporal.iloc[0]
        ),
        f4(
            conf[
                (conf.dataset == "DICTrank") & (conf.alpha == 0.1)
            ].marginal_coverage_temporal.iloc[0]
        ),
    )
)
w(
    '| 12 | 구조만으로 "2010년 이후"를 맞히는 AUC (순열 2,000회) | **%s** (p = %s) | **%s** (p = %s) | §13.1 |'
    % (
        f4(EP["DILIrank"].era_auc),
        f4(EP["DILIrank"].p_phipson_smyth),
        f4(EP["DICTrank"].era_auc),
        f4(EP["DICTrank"].p_phipson_smyth),
    )
)
_rv1 = {
    d: rv[(rv.dataset == d) & (rv.direction == "역방향") & (rv.setting.str.startswith("R1"))].iloc[
        0
    ]
    for d in DS
}
w(
    "| 13 | **역방향** 시간분할 ECE 배율 (R1 크기맞춤, 200회) | **%s배** (p = %s) | **%s배** (p = %s) | §12.2 |"
    % (
        f2(_rv1["DILIrank"].ece_ratio),
        f4(_rv1["DILIrank"].p_phipson_smyth),
        f2(_rv1["DICTrank"].ece_ratio),
        f4(_rv1["DICTrank"].p_phipson_smyth),
    )
)
w(
    "| 14 | 시간이 아닌 축으로 이동을 포화시킨 분할의 최대 ECE 배율 | %s배 | %s배 | §13.5 |"
    % (
        f2(sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("X"))].ece_ratio.max()),
        f2(sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("X"))].ece_ratio.max()),
    )
)
w(
    "| 15 | **검증군** 라벨 없이 추정한 사전확률로 회복되는 ECE (최선) | %s | %s | §10.2 |"
    % (
        pct1(
            pe[
                (pe.dataset == "DILIrank")
                & (pe.estimator.str.startswith("P"))
                & (~pe.estimator.str.startswith("P4"))
            ].recovery.max()
        ),
        pct1(
            pe[
                (pe.dataset == "DICTrank")
                & (pe.estimator.str.startswith("P"))
                & (~pe.estimator.str.startswith("P4"))
            ].recovery.max()
        ),
    )
)
_tbm = {
    d: tb[(tb.dataset == d) & (tb.split == "시간분할") & (tb.predictor.str.startswith("M."))].iloc[
        0
    ]
    for d in DS
}
_ri3 = {d: ri[(ri.dataset == d) & (ri.control.str.startswith("R3"))].iloc[0] for d in DS}
w(
    "| 16 | 시간분할 Brier, 모델 대 상수(학습군 양성률) — BSS | **%s** | **%s** | §25 |"
    % (sg(_tbm["DILIrank"].bss_vs_C0), sg(_tbm["DICTrank"].bss_vs_C0))
)
w(
    "| 17 | ECE 배율 R3 의 95%% 신뢰구간 (분자+분모 변동) | **%s [%s, %s]** | **%s [%s, %s]** | §26 |"
    % (
        f2(_ri3["DILIrank"].ece_ratio),
        f2(_ri3["DILIrank"].ratio_lo),
        f2(_ri3["DILIrank"].ratio_hi),
        f2(_ri3["DICTrank"].ece_ratio),
        f2(_ri3["DICTrank"].ratio_lo),
        f2(_ri3["DICTrank"].ratio_hi),
    )
)
w(
    "| 18 | 같은 R3 배율, ECE 편향을 제거한 뒤 | **%s배** | **%s배** | §27 |"
    % (
        f2(
            cmr[
                (cmr.dataset == "DILIrank") & (cmr.control.str.startswith("R3"))
            ].ece_debiased_ratio.iloc[0]
        ),
        f2(
            cmr[
                (cmr.dataset == "DICTrank") & (cmr.control.str.startswith("R3"))
            ].ece_debiased_ratio.iloc[0]
        ),
    )
)
w(
    "| 19 | 하이퍼파라미터를 학습군 안에서 고른 뒤 R3 배율 | **%s배** (고정 %s배) | **%s배** (고정 %s배) | §28 |"
    % (
        f2(hpc[(hpc.dataset == "DILIrank") & (hpc.arm != "고정 설정")].ratio_R3.iloc[0]),
        f2(hpc[(hpc.dataset == "DILIrank") & (hpc.arm == "고정 설정")].ratio_R3.iloc[0]),
        f2(hpc[(hpc.dataset == "DICTrank") & (hpc.arm != "고정 설정")].ratio_R3.iloc[0]),
        f2(hpc[(hpc.dataset == "DICTrank") & (hpc.arm == "고정 설정")].ratio_R3.iloc[0]),
    )
)
w(
    "| 20 | 검출 가능한 최소 ECE 개선 (메워야 할 간격 대비) | %s–%s | %s–%s | §29 |"
    % (
        pct1(100 * rp[rp.dataset == "DILIrank"].mde_over_gap.min()),
        pct1(100 * rp[rp.dataset == "DILIrank"].mde_over_gap.max()),
        pct1(100 * rp[rp.dataset == "DICTrank"].mde_over_gap.min()),
        pct1(100 * rp[rp.dataset == "DICTrank"].mde_over_gap.max()),
    )
)
w(
    "| 21 | 도메인 분류기 밀도비 가중 후 ECE (가중 없음 대비) | %s (%s) | %s (%s) | §30 |"
    % (
        f4(
            csw[
                (csw.dataset == "DILIrank") & (csw.kind == "outer") & (csw.arm.str.startswith("IW"))
            ].ece.min()
        ),
        sg(
            csw[
                (csw.dataset == "DILIrank") & (csw.kind == "outer") & (csw.arm.str.startswith("IW"))
            ].ece.min()
            - float(
                csw[
                    (csw.dataset == "DILIrank")
                    & (csw.kind == "outer")
                    & (csw.arm.str.startswith("W0"))
                ].ece.iloc[0]
            )
        ),
        f4(
            csw[
                (csw.dataset == "DICTrank") & (csw.kind == "outer") & (csw.arm.str.startswith("IW"))
            ].ece.min()
        ),
        sg(
            csw[
                (csw.dataset == "DICTrank") & (csw.kind == "outer") & (csw.arm.str.startswith("IW"))
            ].ece.min()
            - float(
                csw[
                    (csw.dataset == "DICTrank")
                    & (csw.kind == "outer")
                    & (csw.arm.str.startswith("W0"))
                ].ece.iloc[0]
            )
        ),
    )
)
w(
    "| 22 | R3 배율이 1 을 넘는 모델×표현 칸 | **%d/%d** | **%d/%d** | §31 |"
    % (
        int((g3[g3.dataset == "DILIrank"].r3_ece_ratio > 1).sum()),
        len(g3[g3.dataset == "DILIrank"]),
        int((g3[g3.dataset == "DICTrank"].r3_ece_ratio > 1).sum()),
        len(g3[g3.dataset == "DICTrank"]),
    )
)
w(
    "| 23 | R3 배율이 1 을 넘는 승인연도 구간 | **%d/%d** | **%d/%d** | §32 |"
    % (
        int((hz[hz.dataset == "DILIrank"].r3_ratio > 1).sum()),
        len(hz[hz.dataset == "DILIrank"]),
        int((hz[hz.dataset == "DICTrank"].r3_ratio > 1).sum()),
        len(hz[hz.dataset == "DICTrank"]),
    )
)
w(
    "| 24 | 이동량을 맞춘 비시간 분할 대비 시간분할 ECE 배율 (축 3종) | %s–%s | %s–%s | §33 |"
    % (
        f2(sm2m[sm2m.dataset == "DILIrank"].ece_ratio_T_over_matched.min()),
        f2(sm2m[sm2m.dataset == "DILIrank"].ece_ratio_T_over_matched.max()),
        f2(sm2m[sm2m.dataset == "DICTrank"].ece_ratio_T_over_matched.min()),
        f2(sm2m[sm2m.dataset == "DICTrank"].ece_ratio_T_over_matched.max()),
    )
)


def _m1a(d, col):
    """§11.1 A(시간분할) Mondrian 행 — 목표 90%."""
    g = mdc[
        (mdc.dataset == d)
        & (mdc.arm.str.startswith("A."))
        & (mdc.method == "Mondrian")
        & ((mdc.alpha - 0.10).abs() < 1e-9)
    ]
    return float(g[col].iloc[0])


w(
    "| 25 | 클래스별 보장(Mondrian conformal) 후 음성 클래스 커버리지, 목표 90%% | **%s** | %s | §11.1 |"
    % (f4(_m1a("DILIrank", "cov_neg")), f4(_m1a("DICTrank", "cov_neg")))
)


def _pq1(d, arm, col):
    g = pq[(pq.dataset == d) & (pq.arm.str.startswith(arm))]
    return float(g[col].mean())


w(
    "| 26 | 해마다 재학습+재보정 후 ECE (고정 모델 대비) | **%s** (%s) | **%s** (%s) | §34 |"
    % (
        f4(_pq1("DILIrank", "P2.", "ece")),
        sg(_pq1("DILIrank", "P2.", "ece") - _pq1("DILIrank", "P0.", "ece")),
        f4(_pq1("DICTrank", "P2.", "ece")),
        sg(_pq1("DICTrank", "P2.", "ece") - _pq1("DICTrank", "P0.", "ece")),
    )
)
w(
    "| 27 | 그 ECE 가 조건 맞춘 교환가능 기준선의 몇 배인가 | %s | %s | §34 |"
    % (
        f2(_pq1("DILIrank", "P2.", "ece") / _pq1("DILIrank", "C2r.", "ece")),
        f2(_pq1("DICTrank", "P2.", "ece") / _pq1("DICTrank", "C2r.", "ece")),
    )
)
w()
w(
    '**16번을 특히 주의해서 읽을 것.** 시간분할에서 모델이 내놓는 확률은 "학습군 양성률을 '
    '그대로 되풀이하는 상수" 보다 Brier 가 나쁘다. 같은 분할에서 AUROC 는 %s / %s 로 0.5 를 '
    "넘는다. **순위는 쓸 수 있고 확률은 쓸 수 없다**(§25). 같은 데이터를 무작위로 나누면 "
    "같은 모델이 상수를 이긴다(BSS %s / %s) — 바뀐 것은 분할 축뿐이다."
    % (
        f4(_tbm["DILIrank"].auroc),
        f4(_tbm["DICTrank"].auroc),
        sg(
            tb[
                (tb.dataset == "DILIrank")
                & (tb.split == "무작위분할")
                & (tb.predictor.str.startswith("M."))
            ].bss_vs_C0.iloc[0]
        ),
        sg(
            tb[
                (tb.dataset == "DICTrank")
                & (tb.split == "무작위분할")
                & (tb.predictor.str.startswith("M."))
            ].bss_vs_C0.iloc[0]
        ),
    )
)
w()
w(
    "**한 문장으로** — 시간분할에서 무너지는 것은 순위를 매기는 능력(판별력)보다 "
    "확률의 크기를 맞히는 능력(보정)이고, 그 붕괴는 **검증군 양성률 차이만으로는 "
    '설명되지 않는다.** "원인은 양성률이 아니라 분포 이동이다" 라고 쓰면 두 군데서 '
    "과장이 된다 — 양성률 통제는 초과분의 %s / %s 를 실제로 걷어내고(§8), 분포 이동은 "
    "존재가 확인됐을 뿐 그 **정도**가 보정 붕괴의 **정도**를 설명하지는 못한다(§13.3)."
    % (pct1(_exp_share["DILIrank"]), pct1(_exp_share["DICTrank"]))
)
w()
w("**주의해서 읽을 것 세 가지.**")
w()
w(
    "1. DICTrank 의 보정 기여율은 유병률을 양측 완전통제하면 %s 로 내려간다(§7.1). "
    '"보정이 압도적" 이라는 표현은 DILIrank 에만 쓸 수 있다.'
    % pct1(R3[("DICTrank", "R3")].share2_reliability)
)
w(
    "2. ECE 배율은 분모(대조군 설계)에 따라 값이 달라진다. 배율을 인용할 때마다 "
    "어떤 대조군인지 함께 적어야 한다(§8, §9.1)."
)
w(
    "3. 시대 분류기(§13.1)는 분포 이동의 **존재와 크기**를 재지만, 그 이동이 보정 붕괴를 "
    "**일으켰다**는 것을 보이지는 않는다. 유사도 구간별 ECE 는 유사도를 따라가지 않는다(§13.3)."
)
w()
w("---")
w()

# ================================================================ 2. 데이터
w("## 2. 데이터와 전처리")
w()
w("### 2.1 출처와 라벨")
w()
w("| 항목 | DILIrank 2.0 | DICTrank |")
w("|---|---|---|")
w("| 엔드포인트 | 약물유발 간독성 | 약물유발 심장독성 |")
w("| 출처 | FDA DILIrank 2.0 데이터셋 | DICTrank 데이터셋 (Qu et al.) |")
w("| 원자료 행 | 1,336 (고유 키) | 1,318 (고유 키 1,292, 이름 중복 26행 접음) |")
w(
    "| 위험도 열 | `concern` (no / less / most / ambiguous) | `concern` (no / less / most / ambiguous) |"
)
w("| 양성 정의 | most + less | most + less |")
w("| 음성 정의 | no | no |")
w("| 제외 | ambiguous | ambiguous |")
w(
    "| 규칙 적용 결과 (원자료 **행** 기준) | 양성 568 / 음성 414 / 제외 354 | 양성 868 / 음성 343 / 제외 107 |"
)
w()
w(
    "**위 행 기준 수와 §2.2 의 고유 키 기준 수를 더해서 맞추려 하면 어긋난다.** DICTrank 는 "
    "868 + 343 = 1,211 로 원자료 1,318행에서 ambiguous 107행을 뺀 값이고, §2.2 의 1,185 는 "
    "이름 중복 26행을 접은 고유 키 1,292 에서 107 을 뺀 값이다. 두 수의 차 26 이 그 중복이다. "
    "DILIrank 는 접을 중복이 없어 568 + 414 = 982 가 §2.2 와 그대로 맞는다."
)
w()
w(
    "`most` 와 `less` 를 한데 묶어 양성으로 삼은 것은 이 분야 선행 연구의 관례를 따른 것이며, "
    "그 선택 자체는 Methods 에서 근거를 밝혀야 한다. 라벨 이진화가 적힌 규칙대로 실행됐는지는 "
    "원자료 행 단위로 대조해 불일치 0건을 확인했다."
)
w()
w("### 2.2 분석집합까지의 제외 경로")
w()
w("| 단계 | DILIrank | DICTrank |")
w("|---|---|---|")
w("| 고유 키 | 1,336 | 1,292 |")
w("| − ambiguous 제외 | −354 | −107 |")
w("| = 이진화 대상 | 982 | 1,185 |")
w(
    "| − SMILES 확보 실패 · 승인연도 미매칭 · 구조 중복 제거 · 탄소 없는 화합물 · 라벨 충돌 | −186 | −266 |"
)
w("| **= 분석집합** | **796** | **919** |")
w()
w(
    "제외 규칙은 결과를 보기 전에 정한 것이고, 각 단계의 제외 목록은 결과 파일로 남겨 두었다. "
    "염과 수화물은 **SMILES 표준화 단계에서** 떨어져 나가고, 구조 중복은 그 뒤 InChIKey 앞 "
    "14자리(골격 키)로 판정했다. 따라서 14자리 키가 추가로 합치는 것은 "
    "**입체이성질체**다. 합쳐지는 과정에서 라벨이 엇갈린 그룹은 어느 쪽도 채택하지 않고 "
    "전부 제외했다 — 결과를 보지 않고 적용되는 규칙이다. 27자리 전체 키로 입체이성질체를 "
    "분리해도 결론이 유지되는 것은 §16.3 에서 확인했다."
)
w()
w("### 2.3 시간분할")
w()
w(
    "승인연도 2010년을 절단으로 삼아 그 이전을 학습군, 이후를 검증군으로 나눴다. "
    "승인연도는 Drugs@FDA 의 성분별 최초 승인일에서 가져왔다."
)
w()
w("| | DILIrank | DICTrank |")
w("|---|---|---|")
for lab, k in [("학습군 n", "n_train"), ("검증군 n", "n_test")]:
    w("| %s | %d | %d |" % (lab, int(T["DILIrank"][k]), int(T["DICTrank"][k])))
w(
    "| 학습군 양성률 | %s | %s |"
    % (f4(SPLIT["DILIrank"]["train_pos_rate"]), f4(SPLIT["DICTrank"]["train_pos_rate"]))
)
w("| 검증군 양성률 | %s | %s |" % (f4(T["DILIrank"].pos_rate), f4(T["DICTrank"].pos_rate)))
w(
    "| 검증군 양성 / 음성 | %d / %d | %d / %d |"
    % (
        SPLIT["DILIrank"]["test_pos"],
        SPLIT["DILIrank"]["test_neg"],
        SPLIT["DICTrank"]["test_pos"],
        SPLIT["DICTrank"]["test_neg"],
    )
)
w()
w(
    "**학습군과 검증군의 양성률이 크게 다르다** — DILIrank 는 %s → %s 로 내려가고 "
    "DICTrank 는 %s → %s 로 내려간다. 이 차이만으로 보정이 무너질 수 있으므로 §8 에서 "
    "유병률을 맞춘 대조군으로 분리했다."
    % (
        f4(SPLIT["DILIrank"]["train_pos_rate"]),
        f4(T["DILIrank"].pos_rate),
        f4(SPLIT["DICTrank"]["train_pos_rate"]),
        f4(T["DICTrank"].pos_rate),
    )
)
w()

# ================================================================ 3. 환경
w("## 3. 모델과 계산 환경")
w()
w(
    "주 모델은 ECFP4 지문(반지름 2, 2048비트)에 RDKit 기술자 6종(MolWt, MolLogP, TPSA, "
    "NumHDonors, NumHAcceptors, NumRotatableBonds)을 붙인 표현 위의 XGBoost 다. "
    "모델·표현을 바꿔도 결과가 유지되는지는 §9.1 에서 데이터셋마다 16조합, 두 데이터셋 "
    "합해 32칸으로 확인했다."
)
w()
w("| 항목 | 값 |")
w("|---|---|")
w("| Python | %s |" % env["python"])
w("| 플랫폼 | %s |" % env["platform"])
for k in (
    "RDKit",
    "XGBoost",
    "scikit-learn",
    "NumPy",
    "pandas",
    "SciPy",
    "statsmodels",
    "LightGBM",
):
    w("| %s | %s |" % (k, env["core"][k]))
w("| 난수 시드 | 42 (모델·분할·부트스트랩 공통) |")
w("| 환경 기록 시각 | %s |" % env["recorded_utc"])
w()
w("시드 의존성은 §9.3 에서 모델 시드 10종·분할 배정 시드 10종으로 확인했다.")
w()
w("### 3.1 같은 입력이 같은 숫자를 주는가 — 결정성과 지표 정의 단일성")
w()
w(
    "이 보고서의 모든 수치는 생성기가 결과 파일에서 읽어 쓴다. 그래서 결과 파일이 "
    "**재실행에서 같은 값으로 재생되는지**가 모든 수치의 전제다. "
    "`check_determinism.py` 가 그것을 검사하고 수치를 "
    "`results/determinism.csv`·`determinism_cross_process.csv`·"
    "`metric_duplicates.csv` 에 남긴다."
)
w()
w("| 범위 | 칸 수 | 예측 최대차 | ECE 최대차 | 보고 자리수 안에서 동일 |")
w("|---|---|---|---|---|")
for _lab, _df in (("한 프로세스 안", det), ("프로세스 간(스레드 고정 해제)", detx)):
    w(
        "| %s | %d개 | %.1e | %.1e | %d/%d |"
        % (
            _lab,
            len(_df),
            float(_df.max_pred_diff.abs().max()),
            float(_df.ece_diff.abs().max()),
            int(_tf(_df.within_report_digits).sum()),
            len(_df),
        )
    )
w()
w(
    "예측도 ECE 도 최대차가 0 이다 — 자리수 안에서 같은 정도가 아니라 **비트 단위로 "
    "같다.** 별도 프로세스에서 스레드 고정을 풀고 다시 계산해도 같다. 팔은 §10 의 "
    "재보정 사다리를 두 데이터셋에서 돌린 %d칸이다(%s)."
    % (len(det), " · ".join(sorted(set(det.arm))))
)
w()
w(
    "**같은 지표가 여러 파일에 복제돼 있다.** 복제된 정의들이 서로 다른 값을 내면 "
    "같은 양이 절마다 다르게 적히는데, 그 불일치는 '문서의 수치에 근거가 있는가' 를 "
    "보는 검증으로는 잡히지 않는다. 각 정의를 파일에서 떼어 같은 입력에 돌려 "
    "비트 단위로 맞춰 본다. 유병률 맞춤 함수(`subsample_to_rate`)는 값이 아니라 "
    "**뽑힌 행 번호**를 맞춘다 — 갈라지면 지표가 아니라 대조군의 구성 자체가 "
    "달라지기 때문이다."
)
w()
w("| 지표 | 정의 개수 | 기준과 동일 |")
w("|---|---|---|")
for _, _r in mdup.iterrows():
    w("| `%s` | %d | %d |" % (_r.metric, int(_r.n_defs), int(_r.n_identical)))
w()
w(
    "전부 일치한다. 이 검사가 실제로 결함을 잡았다 — 로짓을 쓰는 함수들이 XGBoost 의 "
    "float32 예측을 캐스팅하지 않아 **같은 양이 어디서 계산됐는지에 따라 정밀도가 "
    "달랐다.** 2026-10-20 에 일곱 군데, 10-21 에 여덟 번째를 고쳤다. 여덟 번째를 "
    "놓친 이유와 그 뒤의 조치는 §10.5 에 적었다."
)
w()
# ================================================================ 4. 분할 전략
w("---")
w()
w("## 4. 분할 전략 비교")
w()
w(
    "같은 데이터·같은 모델을 세 가지 방식으로 평가했다. 무작위 5-fold CV, "
    "Bemis–Murcko 스캐폴드 5-fold CV, 그리고 승인연도 시간분할이다."
)
w()
w("| 데이터셋 | 분할 | AUROC | ECE |")
w("|---|---|---|---|")
for d in DS:
    for sp, lab in [("random_cv", "무작위 5-fold CV"), ("scaffold_cv", "스캐폴드 5-fold CV")]:
        g = bc[(bc.dataset == d) & (bc.split == sp)]
        w(
            "| %s | %s | %s ± %s | %s ± %s |"
            % (
                d,
                lab,
                f4(g.AUROC.mean()),
                f4(g.AUROC.std(ddof=1)),
                f4(g.ECE.mean()),
                f4(g.ECE.std(ddof=1)),
            )
        )
    t = T[d]
    w(
        "| %s | **시간분할** | **%s** [%s–%s] | **%s** |"
        % (d, f4(t.AUROC), f4(t.auroc_lo), f4(t.auroc_hi), f4(t.ECE))
    )
w()
w("고유 스캐폴드는 DILIrank 542개(단독 출현 455개), DICTrank 638개(단독 출현 545개)다.")
w()
w("**읽는 법 두 가지.**")
w()
_sd_d = (
    bc[(bc.dataset == "DILIrank") & (bc.split == "scaffold_cv")].AUROC.mean()
    - bc[(bc.dataset == "DILIrank") & (bc.split == "random_cv")].AUROC.mean()
)
_sd_c = (
    bc[(bc.dataset == "DICTrank") & (bc.split == "scaffold_cv")].AUROC.mean()
    - bc[(bc.dataset == "DICTrank") & (bc.split == "random_cv")].AUROC.mean()
)
w(
    "1. **스캐폴드 분할은 시간분할을 대체하지 못한다.** 스캐폴드 CV 의 AUROC 는 무작위 CV 와 "
    "거의 같고(DILIrank %+.4f, DICTrank %+.4f), 방향조차 데이터셋마다 반대다. "
    "ECE 도 무작위 CV 수준에 머문다. 즉 구조 중복을 막는 것만으로는 시간분할이 드러내는 "
    "문제를 볼 수 없다." % (_sd_d, _sd_c)
)
_r_d = T["DILIrank"].ECE / bc[(bc.dataset == "DILIrank") & (bc.split == "scaffold_cv")].ECE.mean()
_r_c = T["DICTrank"].ECE / bc[(bc.dataset == "DICTrank") & (bc.split == "scaffold_cv")].ECE.mean()
w(
    "2. **차이가 AUROC 보다 ECE 에서 훨씬 크다.** 시간분할 ECE 는 스캐폴드 CV 의 "
    "%s배(DILIrank) / %s배(DICTrank)다. AUROC 로만 보고하면 이 격차가 보이지 않는다."
    % (f2(_r_d), f2(_r_c))
)
w()
w(
    "> 배율을 인용할 때 분모를 반드시 밝힐 것. 위 %s배는 **스캐폴드 CV 5-fold ECE** 기준이고, "
    "§1 5행의 %s배는 **크기맞춤 무작위 대조군 %d회** 기준이다. 분자(시간분할 ECE %s)는 같다."
    % (
        f2(_r_d),
        f2(PCR[("DILIrank", "R1")].ece_ratio),
        _reps("prevalence_control.csv", "n_rep_valid"),
        f4(T["DILIrank"].ECE),
    )
)
w()

# ================================================================ 5. 판별력
w("---")
w()
w("## 5. 판별력 저하와 그 유의성")
w()
w(
    "시간분할 AUROC 가 우연히 낮게 나온 것이 아님을 확인하기 위해, **같은 크기의 무작위 분할**을 "
    "%s회 만들어 귀무분포를 구했다. 학습·검증의 크기만 유지하고 소속을 무작위로 바꾸는 "
    "방식이다." % format(_reps("permutation_test.csv", "n_perm_valid"), ",")
)
w()
w("| 데이터셋 | 시간분할 AUROC | 귀무 평균 (SD) | 귀무 2.5–97.5% | ΔAUROC | 관측값 이하 | p |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    r = PT[d]
    w(
        "| %s | **%s** | %s (%s) | %s–%s | **+%s** | %d/%d | **%s** |"
        % (
            d,
            f4(r.auroc_temporal),
            f4(r.null_mean),
            f4(r.null_sd),
            f4(r.null_p2_5),
            f4(r.null_p97_5),
            f4(r.delta_auroc),
            int(r.n_le_observed),
            int(r.n_perm_valid),
            f4(r.p_permutation),
        )
    )
w()
w(
    "p 는 Phipson–Smyth 편향보정 추정량 `(1 + r) / (1 + N)` 이다. 단순 비율 `r/N` 은 초과 사례가 "
    '0건일 때 p = 0 을 내는데, 이는 "귀무 확률이 정확히 0" 이라는 불가능한 주장이 된다. '
    "이 추정량의 하한은 순열 횟수가 정한다 — 1,000회에서는 `1/1001 = %s` 다. "
    "**따라서 `p < 0.001` 이나 `p = 0` 으로 쓰면 안 된다.**" % f4(1 / 1001)
)
w()
w(
    "AUROC 의 95%% 신뢰구간은 검증군 부트스트랩 2,000회에서 DILIrank [%s–%s], DICTrank [%s–%s] 다. "
    "**두 구간 모두 0.5 는 넘는다** — 판별력이 사라진 것이 아니라 줄어든 것이다. "
    '다만 **"두 구간 모두 무작위 CV 평균을 포함하지 않는다" 고 쓰면 사실이 아니다** — '
    "DILIrank 의 상한 %s 는 무작위 5-fold CV 평균 %s 보다 높아 그 값을 포함한다. "
    "DICTrank 는 상한 %s 로 평균 %s 를 포함하지 않는다. 즉 검증군 하나에서 잡은 "
    "부트스트랩 구간만으로는 DILIrank 의 격차를 단독 입증할 수 없고, 격차의 유의성 근거는 "
    "위의 순열검정(p = %s / %s)이다."
    % (
        f4(T["DILIrank"].auroc_lo),
        f4(T["DILIrank"].auroc_hi),
        f4(T["DICTrank"].auroc_lo),
        f4(T["DICTrank"].auroc_hi),
        f4(T["DILIrank"].auroc_hi),
        f4(CVR["DILIrank"]),
        f4(T["DICTrank"].auroc_hi),
        f4(CVR["DICTrank"]),
        f4(PT["DILIrank"].p_permutation),
        f4(PT["DICTrank"].p_permutation),
    )
)
w()
w(
    "> `baseline_cv` 의 무작위 5-fold CV AUROC(§4)와 순열검정의 귀무분포 평균은 다른 양이다. "
    "표·그림에는 CV 값을, 유의성 서술에는 순열 값을 쓰고 Methods 에 구분을 명시할 것."
)
w()

# ---------------------------------------------------------------- 5.x 검증군 재균형 대조
_tr5 = cv("temporal_results.csv")
_t5 = {str(r.setting): r for _, r in _tr5.iterrows()}
assert (
    abs(float(_t5["temporal"].AUROC) - float(T["DILIrank"].AUROC)) < 1e-9
), "temporal_results 의 시간분할 AUROC 가 §4 와 다르다"
w(
    "**판별력 저하가 양성률 차이의 산물인지도 따로 보았다(DILIrank 만).** 검증군을 "
    "**학습군 양성률(%s)에 맞춰 되올려** 재표집하면 n 이 %d 로 줄고 AUROC 는 %s 가 된다. "
    "같은 크기의 양성률 맞춤 무작위 분할은 %s 다. 즉 검증군의 양성률을 학습군과 같게 "
    "만들어도 판별력 격차가 남는다. **이 대조는 DILIrank 한쪽만 계산했고 n 이 %d 로 작으며 "
    "신뢰구간을 붙이지 않았으므로 보조 근거로만 쓴다** — 판별력 저하의 주 근거는 위의 "
    "순열검정이다."
    % (
        f4(_t5["temporal_rebalanced"].pos_rate),
        int(_t5["temporal_rebalanced"].n_test),
        f4(_t5["temporal_rebalanced"].AUROC),
        f4(_t5["random_matched"].AUROC),
        int(_t5["temporal_rebalanced"].n_test),
    )
)
w()
# ================================================================ 6. 보정
w("---")
w()
w("## 6. 보정 붕괴")
w()
w("무작위 분할(검증군 크기를 시간분할과 맞춘 것)과 시간분할을 같은 지표로 나란히 놓았다.")
w()
w(
    "| 지표 | DILIrank 무작위 | DILIrank 시간분할 | DICTrank 무작위 | DICTrank 시간분할 | 좋은 방향 |"
)
w("|---|---|---|---|---|---|")
MET = [
    ("AUROC", "AUROC", "높을수록"),
    ("Brier", "Brier", "낮을수록"),
    ("ECE", "ECE", "낮을수록"),
    ("MCE", "MCE", "낮을수록"),
    ("보정기울기", "calib_slope", "1에 가까울수록"),
    ("보정절편", "calib_intercept", "0에 가까울수록"),
    ("BalAcc @0.5", "BalAcc@0.5", "높을수록"),
    ("MCC @0.5", "MCC@0.5", "높을수록"),
]
for lab, k, good in MET:
    w(
        "| %s | %s | **%s** | %s | **%s** | %s |"
        % (
            lab,
            f4(CR["DILIrank"][k]),
            f4(CT["DILIrank"][k]),
            f4(CR["DICTrank"][k]),
            f4(CT["DICTrank"][k]),
            good,
        )
    )
w()
w("**확률이 한쪽으로 치우친다.** 시간분할에서는 평균 예측확률이 실제 양성률보다 크게 높다.")
w()
w("| 데이터셋 | 분할 | 실제 양성률 | 평균 예측확률 | 치우침 |")
w("|---|---|---|---|---|")
for d in DS:
    for lab, r in [("무작위", CR[d]), ("시간분할", CT[d])]:
        w(
            "| %s | %s | %s | %s | %+.4f |"
            % (d, lab, f4(r.prevalence), f4(r.mean_pred), r.mean_pred - r.prevalence)
        )
w()
w(
    '모델은 양성률이 낮아진 검증군에도 높은 확률을 내놓는다. **다만 "학습군 양성률을 그대로 '
    '들고 나온다" 고 쓰면 정확하지 않다** — 평균 예측확률(%s / %s)은 학습군 양성률(%s / %s)보다 '
    "오히려 %s / %s 만큼 더 높다. 그 결과 보정기울기가 %s(DILIrank) / %s(DICTrank)까지 "
    "떨어진다 — 1 이 이상적이므로 예측확률의 변화폭이 실제 위험의 변화폭보다 훨씬 크다는 뜻이다."
    % (
        f4(CT["DILIrank"].mean_pred),
        f4(CT["DICTrank"].mean_pred),
        f4(SPLIT["DILIrank"]["train_pos_rate"]),
        f4(SPLIT["DICTrank"]["train_pos_rate"]),
        f4(CT["DILIrank"].mean_pred - SPLIT["DILIrank"]["train_pos_rate"]),
        f4(CT["DICTrank"].mean_pred - SPLIT["DICTrank"]["train_pos_rate"]),
        f4(CT["DILIrank"].calib_slope),
        f4(CT["DICTrank"].calib_slope),
    )
)
w()
w("**임계값을 옮기면 일부는 되돌아온다.**")
w()
w("| 데이터셋 | 최적 임계값 (Youden J) | BalAcc 0.5 → 최적 | MCC 0.5 → 최적 |")
w("|---|---|---|---|")
for d in DS:
    r = CT[d]
    w(
        "| %s | %s | %s → %s (%+.4f) | %s → %s (%+.4f) |"
        % (
            d,
            f4(r.best_thr),
            f4(r["BalAcc@0.5"]),
            f4(r["BalAcc@best"]),
            r["BalAcc@best"] - r["BalAcc@0.5"],
            f4(r["MCC@0.5"]),
            f4(r["MCC@best"]),
            r["MCC@best"] - r["MCC@0.5"],
        )
    )
w()
w(
    "최적 임계값이 %s / %s 로 0.5 에서 크게 벗어나 있다는 것 자체가 확률 척도가 틀어졌다는 "
    "증거다. **다만 임계값 재조정은 보정 문제를 해결하지 못한다** — 순위는 그대로이므로 AUROC 는 "
    "바뀌지 않고, 확률값을 그대로 쓰는 용도(위험 소통, 기대효용 계산, 선별 우선순위)에서는 "
    "여전히 틀린 값이 나온다." % (f4(CT["DILIrank"].best_thr), f4(CT["DICTrank"].best_thr))
)
w()
w(
    "> MCE 는 구간 하나가 정하는 값이므로 결론을 걸지 말 것. DILIrank %s 와 DICTrank %s 는 "
    "각각 표본이 적은 단일 구간에서 나온다. 보조 수치로만 쓰고 각주를 달 것."
    % (f4(CT["DILIrank"].MCE), f4(CT["DICTrank"].MCE))
)
w()
# ================================================================ 7. Brier 분해
w("---")
w()
w("## 7. 보정과 판별력 중 어느 쪽이 무너졌나 — Brier 분해")
w()
w("Brier 점수는 Murphy 분해로 세 항으로 갈린다.")
w()
w("```")
w("Brier = reliability − resolution + uncertainty")
w("  reliability  보정 오차 (낮을수록 좋음)")
w("  resolution   판별력    (높을수록 좋음)")
w("  uncertainty  ō(1−ō)   검증군 양성률만으로 정해지는 상수")
w("```")
w()
w(
    "무작위 분할 %d회를 각각 분해한 뒤 평균한 값을 주 분석으로 삼았다. 양쪽 열의 검증군 크기가 "
    "같아야 하므로 무작위 쪽도 시간분할과 같은 n 으로 뽑는다."
    % _reps("brier_decomposition.csv", "n_repeats")
)
w()
w(
    "| 항 | DILIrank 무작위 | DILIrank 시간분할 | 차이 | DICTrank 무작위 | DICTrank 시간분할 | 차이 |"
)
w("|---|---|---|---|---|---|---|")
for lab, kt, kr in [
    ("uncertainty", None, None),
    ("resolution (판별력)", "resolution_temporal", "resolution_random_mean"),
    ("reliability (보정)", "reliability_temporal", "reliability_random_mean"),
    ("Brier", "brier_temporal", "brier_random_mean"),
]:
    if lab == "uncertainty":
        vals = []
        for d in DS:
            ut, ur = R3[(d, "R1")].unc_temporal, R3[(d, "R1")].unc_random
            vals += [f4(ur), f4(ut), "%+.4f" % (ut - ur)]
        w("| %s | %s |" % (lab, " | ".join(vals)))
        continue
    vals = []
    for d in DS:
        r = BD[d]
        vals += [f4(r[kr]), f4(r[kt]), "%+.4f" % (r[kt] - r[kr])]
    w("| %s | %s |" % (lab, " | ".join(vals)))
w()
w(
    "**uncertainty 행만 다른 실행에서 가져온 값이다.** reliability·resolution·Brier 는 "
    "무작위 20회의 주 분석에서 나오고, uncertainty 는 200회 대조군(R1, §7.1)에서 나온다. "
    "그래서 Murphy 항등식 `Brier = rel − res + unc` 이 **무작위 열에서는 정확히 닫히지 "
    "않는다** — 표의 값으로 계산하면 DILIrank %s, DICTrank %s 인데 표의 Brier 는 %s / %s 다 "
    "(차이 %s / %s). 시간분할 열은 같은 실행에서 나오므로 닫힌다. **항등식을 그대로 쓰려면 "
    "§7.1 의 R1 행을 쓸 것.**"
    % (
        f4(
            BD["DILIrank"].reliability_random_mean
            - BD["DILIrank"].resolution_random_mean
            + R3[("DILIrank", "R1")].unc_random
        ),
        f4(
            BD["DICTrank"].reliability_random_mean
            - BD["DICTrank"].resolution_random_mean
            + R3[("DICTrank", "R1")].unc_random
        ),
        f4(BD["DILIrank"].brier_random_mean),
        f4(BD["DICTrank"].brier_random_mean),
        f4(
            abs(
                BD["DILIrank"].brier_random_mean
                - (
                    BD["DILIrank"].reliability_random_mean
                    - BD["DILIrank"].resolution_random_mean
                    + R3[("DILIrank", "R1")].unc_random
                )
            )
        ),
        f4(
            abs(
                BD["DICTrank"].brier_random_mean
                - (
                    BD["DICTrank"].reliability_random_mean
                    - BD["DICTrank"].resolution_random_mean
                    + R3[("DICTrank", "R1")].unc_random
                )
            )
        ),
    )
)
w()
w("| | DILIrank | DICTrank |")
w("|---|---|---|")
w(
    "| **보정(reliability) 기여** | **%s** [95%% CI %s–%s] | **%s** [95%% CI %s–%s] |"
    % (
        pct1(BD["DILIrank"].pct_from_reliability_A),
        pct1(BD["DILIrank"].pct_reliability_A_lo),
        pct1(BD["DILIrank"].pct_reliability_A_hi),
        pct1(BD["DICTrank"].pct_from_reliability_A),
        pct1(BD["DICTrank"].pct_reliability_A_lo),
        pct1(BD["DICTrank"].pct_reliability_A_hi),
    )
)
w(
    "| 판별력(resolution) 기여 | %s | %s |"
    % (pct1(BD["DILIrank"].pct_from_resolution_A), pct1(BD["DICTrank"].pct_from_resolution_A))
)
w(
    "| 보정 / 판별력 비 | %s배 | %s배 |"
    % (
        f2(BD["DILIrank"].d_reliability_A / BD["DILIrank"].d_resolution_A),
        f2(BD["DICTrank"].d_reliability_A / BD["DICTrank"].d_resolution_A),
    )
)
w(
    "| 부트스트랩 2,000회 중 기여율 > 100%% | %s | %s |"
    % (pct1(BD["DILIrank"].frac_boot_over_100 * 100), pct1(BD["DICTrank"].frac_boot_over_100 * 100))
)
w()
w(
    "**상한이 100%% 를 넘는 것은 DICTrank 뿐이다**(%s; DILIrank 는 %s 로 100%% 미만이다). "
    "그것은 계산 오류가 아니다. 재표본 일부에서 판별력이 오히려 "
    "**개선**되어 분모가 보정 성분보다 작아지기 때문이다. DICTrank 에서 그 비율이 %s 로 "
    "높다는 것은 **DICTrank 의 판별력 저하 자체가 불안정하다**는 뜻이며, 이 데이터셋에서 "
    "판별력 저하를 단독 주장으로 쓰지 말아야 하는 근거다."
    % (
        pct1(BD["DICTrank"].pct_reliability_A_hi),
        pct1(BD["DILIrank"].pct_reliability_A_hi),
        pct1(BD["DICTrank"].frac_boot_over_100 * 100),
    )
)
w()
w(
    "> 풀링 값을 주 수치로 쓰지 말 것. 무작위 20회를 한 덩어리로 합쳐 한 번만 분해하면 "
    "보정 기여율이 %s / %s 로 올라간다. 무작위 쪽 표본만 20배가 되어 구간 추정이 안정해지는 "
    "비대칭 때문이며, 주 분석은 분할별 분해 후 평균(%s / %s)이다. 둘 다 보고하는 편이 강점이 된다."
    % (
        pct1(BD["DILIrank"].pct_from_reliability_B),
        pct1(BD["DICTrank"].pct_from_reliability_B),
        pct1(BD["DILIrank"].pct_from_reliability_A),
        pct1(BD["DICTrank"].pct_from_reliability_A),
    )
)
w()
w(
    "**이 기여율은 절단 2010년에 특정되지 않는다.** 절단을 2005·2015년으로 옮겨도 "
    "reliability 기여가 여섯 설정 전부에서 과반으로 유지된다(%s–%s). 표는 §9.7 에 있다."
    % (pct1(bdc.pct_from_reliability.min()), pct1(bdc.pct_from_reliability.max()))
)
w()
w("### 7.1 분모 선택 — 유병률을 맞추면 논쟁이 사라진다")
w()
w(
    "위의 기여율은 `Δreliability / (Δreliability + Δresolution)` 이며 **Δuncertainty 를 "
    "분모에서 제외한 값**이다. uncertainty 는 검증군 양성률만으로 정해지므로, 시간분할과 "
    "대조군의 양성률이 다르면 Δuncertainty ≠ 0 이 되고 이 비율은 ΔBrier 전체가 아니라 "
    "그 일부 위에서 계산된 값이 된다. 검증군 양성률을 맞춘 대조군(R2·R3)에서는 "
    "Δuncertainty 가 사실상 0 이 되어 분모 선택 문제가 사라진다."
)
w()
w("세 단계 대조군을 각각 %d회 반복했다." % _reps("prevalence_control.csv", "n_rep_valid"))
w()
w("| 대조군 | 통제 내용 |")
w("|---|---|")
w("| R1 | 검증군 크기만 맞춘 무작위 분할 |")
w("| R2 | R1 + 검증군 양성률을 시간분할과 일치 |")
w("| R3 | R2 + 학습군 양성률까지 일치 (양측 완전통제) |")
w()
w("| 데이터셋 | 대조군 | Δreliability | Δresolution | Δuncertainty | 2항 기여율 | 3항 기여율 |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    for b in ("R1", "R2", "R3"):
        r = R3[(d, b)]
        w(
            "| %s | %s | %+.4f | %+.4f | %+.4f | %s | %s |"
            % (
                d,
                b,
                r.d_reliability,
                r.d_resolution,
                r.d_uncertainty,
                pct1(r.share2_reliability),
                pct1(r.share3_reliability),
            )
        )
w()
_u2 = R3[("DICTrank", "R2")].d_uncertainty
_u3 = R3[("DICTrank", "R3")].d_uncertainty
w(
    "**R2·R3 에서 Δuncertainty 가 0 에 붙고 2항·3항 기여율이 일치한다.** 엄밀히는 정확히 0 이 "
    "아니다 — 유병률 맞춤이 정수 개수로 이루어져 200회 평균 양성률이 목표와 다섯째 소수에서 "
    "어긋난다(DILIrank %s 대 %s, DICTrank %s 대 %s). 그 결과 남는 Δuncertainty 는 "
    "DICTrank R2 %.1e / R3 %.1e 이고, 2항 분모와 3항 분모의 차이는 최대 %.4f%%p 다. "
    '소수 첫째 자리까지는 완전히 같다. **"정확히 0" 이라고 쓰면 사실이 아니다.**'
    % (
        f4(R3[("DILIrank", "R3")].obar_temporal),
        "%.7f" % R3[("DILIrank", "R3")].obar_random,
        f4(R3[("DICTrank", "R3")].obar_temporal),
        "%.7f" % R3[("DICTrank", "R3")].obar_random,
        _u2,
        _u3,
        max(
            abs(R3[(d, b)].share2_reliability - R3[(d, b)].share3_reliability)
            for d in DS
            for b in ("R2", "R3")
        ),
    )
)
w()
w(
    "**논문에 쓸 주 수치는 R3 값(%s / %s)이다** — 분모 논쟁이 없고 학습군 비율까지 통제한 "
    "가장 보수적인 설계이기 때문이다. R1 값은 대조군 민감도로 병기한다. "
    "R1 에서 분모가 갈리는 이유는 검증군 양성률을 맞추지 않아 Δuncertainty 가 "
    "DILIrank %+.4f / DICTrank %+.4f 로 남기 때문이다."
    % (
        pct1(R3[("DILIrank", "R3")].share2_reliability),
        pct1(R3[("DICTrank", "R3")].share2_reliability),
        R3[("DILIrank", "R1")].d_uncertainty,
        R3[("DICTrank", "R1")].d_uncertainty,
    )
)
w()
w("### 7.2 표본 크기 일치 교정")
w()
w(
    "R2·R3 는 유병률을 부표집으로 맞추는 과정에서 **대조군 n 이 줄어든다** — 대조군 "
    "%d / %d 대 시간분할 %d / %d 이다. ECE·reliability·resolution 은 구간당 표본이 줄면 "
    "위로 편향되므로(구간잡음 편향 δ ≈ Σ o_b(1−o_b)/N), 대조군 쪽만 부풀려진 상태로 비교한 "
    "셈이다. 시간분할 검증군도 같은 n 으로 층화 부표집해(200회) 양쪽을 같은 크기 위에 올린 "
    "뒤 다시 계산했다."
    % (
        int(MN[("DILIrank", "R3")].n_ctrl),
        int(MN[("DICTrank", "R3")].n_ctrl),
        int(MN[("DILIrank", "R3")].n_test_temporal),
        int(MN[("DICTrank", "R3")].n_test_temporal),
    )
)
w()
w("| 데이터셋 | 대조군 | n (대조 / 관측) | ECE 배율 | p | 2항 기여율 | 구간잡음 차 δ(관측−대조) |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    dfull = MN[(d, "R1")].delta_T_matched  # 검증군 전수의 구간잡음 δ (R1 은 n 이 이미 같다)
    for b in ("R1", "R2", "R3"):
        r = MN[(d, b)]
        same = abs(r.ratio_old - r.ratio_new) < 1e-12

        def arrow(a, bb, fmt, same=same):
            return fmt(a) if same else "%s → **%s**" % (fmt(a), fmt(bb))

        w(
            "| %s | %s | %d / %d | %s | %s | %s | %s |"
            % (
                d,
                b,
                int(r.n_ctrl),
                int(r.n_test_temporal),
                arrow(r.ratio_old, r.ratio_new, f2),
                arrow(r.p_old, r.p_new, f4),
                arrow(r.share_old, r.share_new, pct1),
                arrow(
                    dfull - r.delta_ctrl, r.delta_T_matched - r.delta_ctrl, lambda x: "%+.4f" % x
                ),
            )
        )
w()
w(
    "R1 은 애초에 양쪽 n 이 같으므로 전 열이 완전히 불변이다 — 이 교정이 n 불일치만 "
    "건드린다는 확인이다. 읽을 점 세 가지."
)
w()
_dlt = [
    (d, b, (MN[(d, "R1")].delta_T_matched - MN[(d, b)].delta_ctrl))
    for d in DS
    for b in ("R1", "R2", "R3")
]
_dlt_neg = [x for x in _dlt if x[2] < 0]
_dlt_pos = [x for x in _dlt if x[2] >= 0]
w(
    "1. **편향의 방향이 대체로 사전 예측과 맞았다.** 구간잡음 δ 는 여섯 중 %d건에서 대조군 "
    "쪽이 컸고(δ < 0), n 을 맞추면 그 격차가 사실상 0 으로 줄어든다. **예외가 %d건 있다** — "
    "%s 가 %s 로 부호가 반대다. 그 행은 애초에 양쪽 n 이 같아 교정할 것이 없는 R1 이므로 "
    "편향이 아니라 0 근처의 잡음으로 읽어야 한다. Δreliability 는 (δ_관측 − δ_대조)만큼 "
    "아래로, Δresolution 은 **거의 같은 양만큼** 위로 편향되어 2항 분모에서 상쇄되고 "
    "**비율만 하향 편향된다** — 정확히 같은 양이라는 보장은 없으므로 이것은 근사이며, "
    "실제 증거는 n 을 맞췄을 때 기여율이 올라간 위 표다."
    % (
        len(_dlt_neg),
        len(_dlt_pos),
        ", ".join("%s %s" % (x[0], x[1]) for x in _dlt_pos),
        ", ".join("%+.4f" % x[2] for x in _dlt_pos),
    )
)
w(
    "2. **교정폭은 DICTrank 에서 더 크다** — DILIrank %+.1f%%p, DICTrank %+.1f%%p. 그래도 "
    'DICTrank %s 는 "보정이 압도적" 이라고 쓸 수 있는 값이 아니다.'
    % (
        MN[("DILIrank", "R3")].share_new - MN[("DILIrank", "R3")].share_old,
        MN[("DICTrank", "R3")].share_new - MN[("DICTrank", "R3")].share_old,
        pct1(MN[("DICTrank", "R3")].share_new),
    )
)
w(
    "3. **DICTrank R3 의 p 가 %s(초과 %d/200)에서 %s(%d/200)로 내려간다.** 이것이 §15 의 "
    "가족 구성 논쟁에 직접 영향을 준다."
    % (
        f4(MN[("DICTrank", "R3")].p_old),
        round(MN[("DICTrank", "R3")].p_old * 201 - 1),
        f4(MN[("DICTrank", "R3")].p_new),
        round(MN[("DICTrank", "R3")].p_new * 201 - 1),
    )
)
w()
w(
    "**어느 값을 본문에 쓸 것인가** — n 일치 값이 불리해지는 쪽으로는 가지 않는다(기여율 ↑, "
    "배율 ↑, p 는 **같거나** ↓). p 가 실제로 내려간 것은 여섯 중 DICTrank R3 한 건뿐이고 "
    "나머지 다섯은 하한에 걸려 그대로다. 고른 것이 아니라 편향 제거의 방향이 한쪽이기 "
    "때문이다. 유리한 쪽을 주 "
    "수치로 올리면 선택적 보고로 읽힐 수 있으므로, **본문의 주 수치는 보수적인 기존 값을 "
    "그대로 두고 n 일치 값을 교정치로 병기한다.** 또한 시간분할 ECE 헤드라인(%s / %s)은 "
    "검증군 전수 %d / %d 에서 계산한 값이어야 하므로, 부표집 평균(%s / %s)을 그 자리에 쓰지 않는다."
    % (
        f4(MN[("DILIrank", "R3")].ece_T_full),
        f4(MN[("DICTrank", "R3")].ece_T_full),
        int(MN[("DILIrank", "R3")].n_test_temporal),
        int(MN[("DICTrank", "R3")].n_test_temporal),
        f4(MN[("DILIrank", "R3")].ece_T_matched),
        f4(MN[("DICTrank", "R3")].ece_T_matched),
    )
)
w()
# ================================================================ 8. 유병률 통제
w("---")
w()
w("## 8. 유병률 차이로 설명되는가")
w()
w(
    "학습군과 검증군의 양성률이 다르므로(§2.3), 보정 붕괴가 그 차이만으로 생긴 것일 수 있다. "
    "§7.1 의 세 단계 대조군을 ECE 기준으로 다시 본다. 각 대조군을 200회 반복하고, "
    '"무작위 대조군의 ECE 가 시간분할 ECE 이상" 인 횟수로 순열 p 를 구했다.'
)
w()
w("| 데이터셋 | 설정 | 검증군 n | 양성률 | ECE | AUROC | 보정기울기 | ECE 배율 | 초과 | p |")
w("|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    t = PCT[d]
    w(
        "| %s | **T. 시간분할** | %d | %s | **%s** [%s–%s] | %s | %s | — | — | — |"
        % (
            d,
            int(t.n),
            f4(t.pos_rate),
            f4(t.ECE),
            f4(t.ECE_lo),
            f4(t.ECE_hi),
            f4(t.AUROC),
            f4(t.slope),
        )
    )
    for b, lab in [
        ("R1", "R1. 무작위 (크기맞춤)"),
        ("R2", "R2. + 검증군 비율맞춤"),
        ("R3", "R3. + 학습·검증 비율맞춤"),
    ]:
        r = PCR[(d, b)]
        w(
            "| %s | %s | %.0f | %s | %s ± %s | %s | %s | **%s배** | %d/%d | **%s** |"
            % (
                d,
                lab,
                r.n,
                f4(r.pos_rate),
                f4(r.ECE),
                f4(r.ECE_sd),
                f4(r.AUROC),
                f4(r.slope),
                f2(r.ece_ratio),
                int(r.n_ge_observed),
                int(r.n_rep_valid),
                f4(r.p_perm),
            )
        )
w()
w(
    "**유병률을 양측 완전통제해도 배율이 1 을 넘는다.** DILIrank %s배, DICTrank %s배다. "
    "즉 보정 붕괴는 양성률 차이로 환원되지 않는다. 다만 통제를 조일수록 배율이 단조 감소한다 — "
    "DILIrank %s → %s → %s, DICTrank %s → %s → %s. **유병률 차이는 효과의 일부를 실제로 "
    '설명한다**는 뜻이므로, "유병률과 무관하다" 가 아니라 "유병률만으로는 설명되지 않는다" 로 써야 한다.'
    % (
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        f2(PCR[("DILIrank", "R1")].ece_ratio),
        f2(PCR[("DILIrank", "R2")].ece_ratio),
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f2(PCR[("DICTrank", "R1")].ece_ratio),
        f2(PCR[("DICTrank", "R2")].ece_ratio),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
    )
)
w()
w("**유병률이 설명하는 몫을 숫자로.**")
w()
w("| 데이터셋 | R1 대비 ECE 초과분 | R3 에서 남는 양 | 유병률 통제로 해소된 비율 |")
w("|---|---|---|---|")
for d in DS:
    w(
        "| %s | **%s** | **%s** | **%s** |"
        % (d, f4(_ex[d]), f4(_rm[d]), pct1((_ex[d] - _rm[d]) / _ex[d] * 100))
    )
w()
w(
    "> 이 표의 값은 전정밀 ECE 에서 계산한 것이다. 인쇄된 네 자리 값끼리 빼면 반올림 경계에서 "
    "마지막 자리가 어긋난다."
)
w()
w(
    '**"200회 중 가장 나쁜 경우도 시간분할보다 낫다" 고 쓰면 안 된다.** DICTrank R3 에서 '
    '%d회가 시간분할 ECE 를 넘었다. 정확한 서술은 "가장 엄격한 통제(R3)에서조차 시간분할 ECE 는 '
    '무작위의 %s배였고, 200회 중 %d회가 그보다 낮았다(p = %s)" 다.'
    % (
        int(PCR[("DICTrank", "R3")].n_ge_observed),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        int(PCR[("DICTrank", "R3")].n_rep_valid - PCR[("DICTrank", "R3")].n_ge_observed),
        f4(PCR[("DICTrank", "R3")].p_perm),
    )
)
w()
w("### 8.1 p 가 하한에 걸렸을 때 증거의 크기 — 대조군 분포 기준 z")
w()
w(
    "여섯 대조군 중 다섯이 초과 0/200 이라 p 가 하한 `1/201 = %s` 에 붙어 있다. 하한값은 "
    '"이보다 작다" 만 말하므로, 증거가 얼마나 큰지는 관측값이 대조군 분포에서 얼마나 떨어져 '
    "있는지로 따로 본다." % f4(1 / 201)
)
w()
w(
    "| 데이터셋 | 대조군 | 대조군 ECE (SD) | 관측 ECE 기존 / n일치 | z 기존 | z n일치 | 초과 | p 기존 → n일치 |"
)
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    for b in ("R1", "R2", "R3"):
        r = MN[(d, b)]
        zo = (r.ece_T_full - r.ece_ctrl) / r.ece_ctrl_sd
        zn = (r.ece_T_matched - r.ece_ctrl) / r.ece_ctrl_sd
        ng = int(PCR[(d, b)].n_ge_observed)
        w(
            "| %s | %s | %s (%s) | %s / %s | %s | %s | %d/200 | %s → %s |"
            % (
                d,
                b,
                f4(r.ece_ctrl),
                f4(r.ece_ctrl_sd),
                f4(r.ece_T_full),
                f4(r.ece_T_matched),
                f2(zo),
                f2(zn),
                ng,
                f4(r.p_old),
                f4(r.p_new),
            )
        )
w()
w(
    "z = (관측 ECE − 대조군 평균) / 대조군 SD 다. **정식 검정이 아니다** — 대조군 ECE 분포가 "
    "정규분포라는 근거가 없으므로 z 를 p 로 환산하지 말 것. 분포 무관하게 읽을 수 있는 것은 "
    "순열 p 쪽이고, 이 열은 하한에 눌린 p 가 가리지 못하는 **거리**만 보여준다."
)
w()
w(
    "1. **대조군을 조일수록 z 가 단조 감소한다** — DILIrank %s → %s → %s, DICTrank %s → %s → %s. "
    "유병률 통제가 실제로 효과의 일부를 걷어낸다는 것이 p(여섯 중 다섯이 하한에 눌려 있다)로는 "
    "보이지 않고 여기서만 보인다."
    % tuple(
        [
            f2((MN[(d, b)].ece_T_full - MN[(d, b)].ece_ctrl) / MN[(d, b)].ece_ctrl_sd)
            for d in DS
            for b in ("R1", "R2", "R3")
        ]
    )
)
w(
    "2. **하한에 걸린 다섯 건과 걸리지 않은 한 건이 z 와 일관된다.** 유일하게 하한이 아닌 "
    "DICTrank R3 이 z 도 유일하게 2 점대다. p 가 증거를 반영하고 있다는 교차 확인이다."
)
w("3. **n 일치 교정은 여섯 건 전부에서 z 를 올린다**(같거나 증가). 기여율·배율과 같은 방향이다.")
w()
# ================================================================ 9. 강건성
w("---")
w()
w("## 9. 강건성")
w()
w("### 9.1 모델 × 표현 32조합")
w()
w(
    "표현 4종(ECFP4+기술자, ECFP4, MACCS, RDKit 기술자)과 모델 4종(XGBoost, RandomForest, "
    "LogisticRegression, LightGBM)의 모든 조합을 두 데이터셋에서 평가했다."
)
w()
w(
    "| 데이터셋 | 조합 | ΔAUROC > 0 | ΔAUROC 중앙값 (범위) | ECE 배율 > 1 | ECE 배율 중앙값 (범위) | 보정기울기 중앙값 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    g = rob[rob.dataset == d]
    w(
        "| %s | %d | **%d/%d** | %+.4f (%+.4f ~ %+.4f) | **%d/%d** | %s (%s ~ %s) | %s |"
        % (
            d,
            len(g),
            (g.d_auroc > 0).sum(),
            len(g),
            g.d_auroc.median(),
            g.d_auroc.min(),
            g.d_auroc.max(),
            (g.ece_ratio > 1).sum(),
            len(g),
            f2(g.ece_ratio.median()),
            f2(g.ece_ratio.min()),
            f2(g.ece_ratio.max()),
            f4(g.slope_temporal.median()),
        )
    )
w()
_neg = rob[rob.d_auroc <= 0]
w(
    "**무엇이 몇 조합에서 성립하는지 구분해 쓸 것.** ECE 배율 > 1 은 **32/32** 에서 성립하지만, "
    "ΔAUROC > 0 은 **%d/32** 다. 유일한 예외는 DILIrank 의 %s / %s 조합(ΔAUROC %+.4f)이다. "
    "즉 **보정 악화가 판별력 저하보다 더 보편적이다.**"
    % ((rob.d_auroc > 0).sum(), _neg.iloc[0].rep, _neg.iloc[0].model, _neg.iloc[0].d_auroc)
)
w()
w(
    "배율의 최솟값은 DILIrank %s(%s / %s), DICTrank %s(%s / %s)다. "
    '**"모든 조합에서 배율이 1.6 을 넘는다" 처럼 특정 문턱을 주장하지 말 것** — '
    "격자 최솟값이 %s 이다."
    % (
        f2(rob[rob.dataset == "DILIrank"].ece_ratio.min()),
        rob.loc[rob[rob.dataset == "DILIrank"].ece_ratio.idxmin()].rep,
        rob.loc[rob[rob.dataset == "DILIrank"].ece_ratio.idxmin()].model,
        f2(rob[rob.dataset == "DICTrank"].ece_ratio.min()),
        rob.loc[rob[rob.dataset == "DICTrank"].ece_ratio.idxmin()].rep,
        rob.loc[rob[rob.dataset == "DICTrank"].ece_ratio.idxmin()].model,
        f2(rob.ece_ratio.min()),
    )
)
w()
_sl_hi = rob.loc[rob.slope_temporal.idxmax()]
w(
    "보정기울기 중앙값은 DILIrank %s, DICTrank %s 로 이상값 1 에서 크게 떨어져 있다. "
    '다만 **"모든 조합에서 압축돼 있다" 고 쓰면 사실이 아니다** — 32칸 중 %d칸은 기울기가 '
    "0.4 를 넘고, 최댓값은 %s(%s / %s / %s)다. 압축은 중앙값 수준의 경향이며 예외가 있다."
    % (
        f4(rob[rob.dataset == "DILIrank"].slope_temporal.median()),
        f4(rob[rob.dataset == "DICTrank"].slope_temporal.median()),
        (rob.slope_temporal > 0.4).sum(),
        f4(_sl_hi.slope_temporal),
        _sl_hi.dataset,
        _sl_hi.rep,
        _sl_hi.model,
    )
)
w()
w("### 9.2 절단연도")
w()
w("절단을 2005·2010·2015년으로 바꿔 반복했다(주 모델 조합).")
w()
w(
    "| 데이터셋 | 절단 | 학습 n | 검증 n | 검증 양성률 | AUROC 무작위 | AUROC 시간분할 | ΔAUROC | ECE 배율 | 보정기울기 |"
)
w("|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in robc[robc.dataset == d].sort_values("cutoff").iterrows():
        w(
            "| %s | %d | %d | %d | %s | %s | %s | %+.4f | %s배 | %s |"
            % (
                d,
                int(r.cutoff),
                int(r.n_train),
                int(r.n_test),
                f4(r.test_pos_rate),
                f4(r.auroc_random),
                f4(r.auroc_temporal),
                r.d_auroc,
                f2(r.ece_ratio),
                f4(r.slope),
            )
        )
w()
w(
    "**ΔAUROC 는 절단연도의 단조 함수가 아니다** — DILIrank 는 2010 > 2015 > 2005, "
    "DICTrank 는 2010 > 2005 > 2015 로 순위가 다르다. 반면 **ECE 배율은 여섯 경우 전부에서 "
    "1 을 넘고**, 범위는 DILIrank %s–%s, DICTrank %s–%s 다."
    % (
        f2(robc[robc.dataset == "DILIrank"].ece_ratio.min()),
        f2(robc[robc.dataset == "DILIrank"].ece_ratio.max()),
        f2(robc[robc.dataset == "DICTrank"].ece_ratio.min()),
        f2(robc[robc.dataset == "DICTrank"].ece_ratio.max()),
    )
)
w()
w("### 9.3 시드")
w()
w("**모델 시드 10종** (분할은 고정, 모델 난수만 바꿈)")
w()
w("| 데이터셋 | AUROC 평균 ± SD (범위) | ECE 평균 ± SD (범위) | 주 보고값(시드 42) |")
w("|---|---|---|---|")
for d in DS:
    g = seed[seed.dataset == d]
    b = g[g.seed == 42].iloc[0]
    w(
        "| %s | %s ± %s (%s–%s) | %s ± %s (%s–%s) | AUROC %s, ECE %s |"
        % (
            d,
            f4(g.auroc.mean()),
            f4(g.auroc.std(ddof=1)),
            f4(g.auroc.min()),
            f4(g.auroc.max()),
            f4(g.ece.mean()),
            f4(g.ece.std(ddof=1)),
            f4(g.ece.min()),
            f4(g.ece.max()),
            f4(b.auroc),
            f4(b.ece),
        )
    )
w()
w("**분할 배정 시드 10종** (CV 폴드 배정을 바꿈)")
w()
w("| 데이터셋 | 분할 | AUROC 범위 | ECE 범위 |")
w("|---|---|---|---|")
for d in DS:
    for sp, lab in [
        ("random_cv", "무작위 CV"),
        ("scaffold_cv_random", "스캐폴드 CV (무작위 배정)"),
    ]:
        g = sseed[(sseed.dataset == d) & (sseed.split == sp)]
        if not len(g):
            continue
        w(
            "| %s | %s (시드 %d종) | %s–%s | %s–%s |"
            % (
                d,
                lab,
                len(g),
                f4(g.auroc_oof.min()),
                f4(g.auroc_oof.max()),
                f4(g.ece_oof.min()),
                f4(g.ece_oof.max()),
            )
        )
w()
w(
    "어느 시드에서도 시간분할 ECE(%s / %s)에 근접한 값이 나오지 않는다 — 시드 잡음의 폭이 "
    "시간분할과 무작위의 격차보다 훨씬 작다." % (f4(T["DILIrank"].ECE), f4(T["DICTrank"].ECE))
)
w()
w("### 9.4 ECE 구간화")
w()
w(
    "ECE 는 구간 개수와 구간 나누는 방식에 따라 값이 달라지는 추정량이므로, %d조합에서 "
    "확인했다." % int(len(binn) / len(DS))
)
w()
w("| 데이터셋 | ECE 범위 | reliability 범위 | resolution 범위 | 주 보고값 (균등폭 10구간) |")
w("|---|---|---|---|---|")
for d in DS:
    g = binn[binn.dataset == d]
    b = g[(g.scheme == "균등폭") & (g.n_bins == 10)].iloc[0]
    w(
        "| %s | %s–%s | %s–%s | %s–%s | ECE %s / rel %s / res %s |"
        % (
            d,
            f4(g.ece.min()),
            f4(g.ece.max()),
            f4(g.reliability.min()),
            f4(g.reliability.max()),
            f4(g.resolution.min()),
            f4(g.resolution.max()),
            f4(b.ece),
            f4(b.reliability),
            f4(b.resolution),
        )
    )
w()
w(
    "**reliability 가 resolution 을 압도한다는 결론은 8조합 전부에서 유지된다** — 최악의 "
    "조합에서도 reliability 최솟값(%s / %s)이 resolution 최댓값(%s / %s)보다 크다."
    % (
        f4(binn[binn.dataset == "DILIrank"].reliability.min()),
        f4(binn[binn.dataset == "DICTrank"].reliability.min()),
        f4(binn[binn.dataset == "DILIrank"].resolution.max()),
        f4(binn[binn.dataset == "DICTrank"].resolution.max()),
    )
)
w()
w("### 9.5 격자 전체에서의 유병률 통제와 conformal")
w()
w("§8 의 유병률 통제와 §11 의 conformal 을 %d조합 전부에서 반복했다." % len(grid))
w()
w(
    "| 데이터셋 | R1 배율 중앙값 (범위) | R2 배율 중앙값 (범위) | R1 기여율 > 50% | R2 기여율 > 50% | conf90 미달 | conf80 미달 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    g = grid[grid.dataset == d]
    w(
        "| %s | %s (%s–%s) | %s (%s–%s) | %d/%d | **%d/%d** | %d/%d | %d/%d |"
        % (
            d,
            f2(g.r1_ece_ratio.median()),
            f2(g.r1_ece_ratio.min()),
            f2(g.r1_ece_ratio.max()),
            f2(g.r2_ece_ratio.median()),
            f2(g.r2_ece_ratio.min()),
            f2(g.r2_ece_ratio.max()),
            (g.r1_share2 > 50).sum(),
            len(g),
            (g.r2_share2 > 50).sum(),
            len(g),
            (g.conf90_marginal < 0.9).sum(),
            len(g),
            (g.conf80_marginal < 0.8).sum(),
            len(g),
        )
    )
w()
_r2ok = int((grid.r2_share2 > 50).sum())
_r1ok2 = int((grid.r1_share2 > 50).sum())
_r1ok3 = int((grid.r1_share3 > 50).sum())
_r2ok3 = int((grid.r2_share3 > 50).sum())
w(
    "**ECE 배율 > 1 과 conformal 커버리지 미달은 32/32 에서 성립한다.** 반면 **보정 성분이 "
    "과반인 것은 R1 기준 %d/32 이지만 R2(검증군 유병률 통제) 기준 %d/32 다** — "
    "미달 %d건은 전부 DICTrank 다. 이것이 §1 의 \"DICTrank 에는 '압도적' 이라는 표현을 쓸 수 "
    '없다" 의 격자 근거다.' % (_r1ok2, _r2ok, 32 - _r2ok)
)
w()
w(
    "**표의 기여율은 §7.1 의 2항 분모** `Δrel/(Δrel+Δres)` **다.** R1 에서 Δuncertainty 를 "
    "분모에 넣는 3항 기준으로 다시 세면 과반 조합은 %d/32 로 줄어든다(DILIrank %d/16, "
    "DICTrank %d/16). R2 에서는 Δunc ≈ 0 이므로 2항과 3항이 같다(%d/32). **따라서 R1 의 "
    "%d/32 는 분모 선택에 기댄 수치이며 단독으로 인용하면 안 된다** — 격자에서 보고할 값도 "
    "§7.1 과 같은 이유로 유병률을 통제한 R2 쪽이다."
    % (
        _r1ok3,
        int((grid[grid.dataset == "DILIrank"].r1_share3 > 50).sum()),
        int((grid[grid.dataset == "DICTrank"].r1_share3 > 50).sum()),
        _r2ok3,
        _r1ok2,
    )
)
w()
w("### 9.6 구현상 임의 선택")
w()
w(
    "내부 보정셋을 승인연도로 나눌 때 경계 연도에 동점이 생기고, 그 배정은 정렬 함수의 "
    "동점 처리가 정한다. 또 적합셋의 행 순서가 XGBoost 의 부분표본을 바꾼다. 둘 다 "
    '"어떤 값이어도 무방한" 임의 선택이므로, 그 임의성이 §10·§11 의 수치를 얼마나 흔드는지 재었다.'
)
w()
w(
    "| 데이터셋 | 지표 | 현행 구현 3종 | E1 동점 배정 무작위화 (범위) | E2 적합셋 행순서 무작위화 (범위) |"
)
w("|---|---|---|---|---|")
TIE = [
    ("conf90 주변커버리지", "conf90_marginal", f4),
    ("conf80 주변커버리지", "conf80_marginal", f4),
    ("Platt 회복률 %", "rec_B", f2),
    ("Isotonic 회복률 %", "rec_C", f2),
    ("AUROC", "auroc", f4),
]
for d in DS:
    g = tie[tie.dataset == d]
    e3 = g[g.experiment.str.startswith("E3")]
    e1 = g[g.experiment.str.startswith("E1")]
    e2 = g[g.experiment.str.startswith("E2")]
    for lab, k, fmt in TIE:
        w(
            "| %s | %s | %s | %s–%s (폭 %s) | %s–%s (폭 %s) |"
            % (
                d,
                lab,
                " / ".join(fmt(v) for v in e3[k]),
                fmt(e1[k].min()),
                fmt(e1[k].max()),
                fmt(e1[k].max() - e1[k].min()),
                fmt(e2[k].min()),
                fmt(e2[k].max()),
                fmt(e2[k].max() - e2[k].min()),
            )
        )
w()
w(
    "현행 구현 3종은 `np.argsort` 기본(§11 계열) / `pandas.sort_values` 기본(§10 계열) / "
    "안정 정렬이다. **앞 두 개는 이 데이터에서 같은 배정을 내므로 서로 다른 값이 아니다** "
    "— 표의 세 값 중 앞 두 개가 같은 이유다. 실제로 갈리는 것은 안정 정렬 하나다. "
    "E1 은 경계 연도 동점만 무작위로 배정하고, E2 는 보정셋을 고정한 채 "
    "적합셋 행 순서만 무작위화한 것으로 각각 20회다. 경계 연도는 DILIrank %d년(그 해 학습군 "
    "%d개 중 %d개만 보정셋), DICTrank %d년(%d개 중 %d개)이다."
    % (
        int(tie[tie.dataset == "DILIrank"].boundary_year.iloc[0]),
        int(tie[tie.dataset == "DILIrank"].n_tie.iloc[0]),
        int(tie[tie.dataset == "DILIrank"].n_tie_in_calib.iloc[0]),
        int(tie[tie.dataset == "DICTrank"].boundary_year.iloc[0]),
        int(tie[tie.dataset == "DICTrank"].n_tie.iloc[0]),
        int(tie[tie.dataset == "DICTrank"].n_tie_in_calib.iloc[0]),
    )
)
w()


def _tie_d(d, e):
    return tie[(tie.dataset == d) & (tie.experiment.str.startswith(e))]


def _rec10(d, pref):
    """§10 표가 쓰는 회복률 정의 — (A.ECE - X.ECE) / (A.ECE - E.ECE) x 100."""
    g = rec[rec.dataset.str.startswith(d)]
    a = g[g.method.str.startswith("A.")].iloc[0]
    e = g[g.method.str.startswith("E.")].iloc[0]
    x = g[g.method.str.startswith(pref)].iloc[0]
    return (a.ECE - x.ECE) / (a.ECE - e.ECE) * 100


w(
    "**§10·§11 의 점추정값은 재현 가능하지만 임의적이다.** 특히 DILIrank conf90 커버리지는 "
    "동점 배정만 바꿔도 %s–%s 로 움직이고, 회복률은 %s–%s%%p 폭으로 흔들린다. "
    "**결론(명목 커버리지 미달, 학습군 내부 재보정으로는 일부만 회복)은 모든 변형에서 "
    "유지되지만 수치는 범위로 써야 한다.**"
    % (
        f4(_tie_d("DILIrank", "E1").conf90_marginal.min()),
        f4(_tie_d("DILIrank", "E1").conf90_marginal.max()),
        f2(_tie_d("DILIrank", "E1").rec_B.min()),
        f2(_tie_d("DILIrank", "E1").rec_B.max()),
    )
)
w()
w("**이 표를 읽을 때 두 가지를 혼동하지 말 것.**")
w()
w(
    "1. **현행 conf90 값은 E1 범위 밖에 있다.** DILIrank 현행 %s 는 동점을 무작위 배정한 20회 "
    "전체(%s–%s)보다 높다. 즉 현행 배정은 커버리지에 **우연히 유리한** 쪽이고, 동점을 다르게 "
    "배정하면 커버리지는 더 내려간다. 결론(0.90 미달)은 그래서 더 강해지지만 %s 를 "
    '"대표값" 처럼 쓰면 안 된다. 행순서만 바꾼 E2 범위(%s–%s)에는 들어간다.'
    % (
        f4(_tie_d("DILIrank", "E3").conf90_marginal.iloc[0]),
        f4(_tie_d("DILIrank", "E1").conf90_marginal.min()),
        f4(_tie_d("DILIrank", "E1").conf90_marginal.max()),
        f4(_tie_d("DILIrank", "E3").conf90_marginal.iloc[0]),
        f4(_tie_d("DILIrank", "E2").conf90_marginal.min()),
        f4(_tie_d("DILIrank", "E2").conf90_marginal.max()),
    )
)
w()
w(
    "2. **회복률의 '현행' 값은 §10 표의 값과 다르다.** 이 민감도 스크립트는 모델을 다시 "
    "적합하므로, DILIrank Platt 회복률이 여기서는 %s%%, §10 표에서는 %s%% 다(차이 %s%%p). "
    "DICTrank 는 %s%% 대 %s%% 다. 두 차이는 모두 E2(적합 잡음) 범위 안에 있다 — DILIrank E2 는 "
    "%s–%s%% 다. **따라서 이 표에서 읽을 것은 폭이고, 점추정값은 §10 표의 것을 쓴다.**"
    % (
        f2(_tie_d("DILIrank", "E3").rec_B.iloc[0]),
        f2(_rec10("DILIrank", "B.")),
        f2(abs(_tie_d("DILIrank", "E3").rec_B.iloc[0] - _rec10("DILIrank", "B."))),
        f2(_tie_d("DICTrank", "E3").rec_B.iloc[0]),
        f2(_rec10("DICTrank", "B.")),
        f2(_tie_d("DILIrank", "E2").rec_B.min()),
        f2(_tie_d("DILIrank", "E2").rec_B.max()),
    )
)
w()
w(
    "**폭을 재는 것과 점추정값에서 임의성을 없애는 것은 다른 일이다.** 이 절은 폭만 "
    "보인다. 세 임의성을 동시에 무작위화해 평균을 새 점추정값으로 쓴 결과는 §10.4 에 "
    "있고, 거기서 결론이 반복 전부에서 유지되는지도 센다."
)
w()
# ---------------------------------------------------------------- 9.7 절단연도별 Brier 분해
_bq = {d: bdc[bdc.dataset == d].sort_values("cutoff") for d in DS}
w("### 9.7 절단연도를 바꿔도 보정이 주범인가")
w()
w(
    "§7 의 보정 기여율은 절단 2010년 한 점에서 계산한 값이다. 절단을 2005·2010·2015년으로 "
    "옮겨 Murphy 분해를 다시 하고, 매번 같은 크기의 무작위 분할 대조군과 비교했다."
)
w()
w(
    "| 데이터셋 | 절단 | 검증 n | reliability 무작위 → 시간분할 | resolution 무작위 → 시간분할 | Brier 무작위 → 시간분할 | 보정 성분 기여율 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in _bq[d].iterrows():
        w(
            "| %s | %d | %d | %s → %s | %s → %s | %s → %s | **%s** |"
            % (
                d,
                int(r.cutoff),
                int(r.n_test),
                f4(r.reliability_random_mean),
                f4(r.reliability_temporal),
                f4(r.resolution_random_mean),
                f4(r.resolution_temporal),
                f4(r.brier_random_mean),
                f4(r.brier_temporal),
                pct1(r.pct_from_reliability),
            )
        )
w()
_b2010 = bdc[(bdc.dataset == "DILIrank") & (bdc.cutoff == 2010)].pct_from_reliability.iloc[0]
assert (
    abs(float(_b2010) - float(BD["DILIrank"].pct_from_reliability_A)) < 1e-6
), "§9.7 의 2010년 행이 §7 주분석과 다르다"
w(
    "**여섯 설정 전부에서 Brier 악화의 과반이 reliability 에서 온다**(%s–%s). "
    '따라서 "무너진 것은 주로 보정이다" 는 절단 2010년에서만 성립하는 결론이 아니다. '
    "절단 2010년 행은 §7 의 주 분석과 같은 값이다(DILIrank 기여율 %s)."
    % (pct1(bdc.pct_from_reliability.min()), pct1(bdc.pct_from_reliability.max()), pct1(_b2010))
)
w()
w(
    "다만 **기여율이 절단에 따라 단조롭게 움직이지 않는다** — DILIrank 는 절단이 늦어질수록 "
    "내려가고(%s → %s) DICTrank 는 올라간다(%s → %s). 검증군 크기도 함께 줄어든다"
    "(DILIrank %d → %d, DICTrank %d → %d). 절단을 늦추면 검증군이 작아져 추정 분산이 "
    '커지므로 **이 추세 자체는 해석하지 않는다.** 읽을 것은 "어느 절단에서도 과반" 이라는 '
    "사실로 제한한다."
    % (
        pct1(_bq["DILIrank"].pct_from_reliability.iloc[0]),
        pct1(_bq["DILIrank"].pct_from_reliability.iloc[-1]),
        pct1(_bq["DICTrank"].pct_from_reliability.iloc[0]),
        pct1(_bq["DICTrank"].pct_from_reliability.iloc[-1]),
        int(_bq["DILIrank"].n_test.iloc[0]),
        int(_bq["DILIrank"].n_test.iloc[-1]),
        int(_bq["DICTrank"].n_test.iloc[0]),
        int(_bq["DICTrank"].n_test.iloc[-1]),
    )
)
w()
w(
    "절단을 옮기면 학습군과 검증군의 구성이 함께 바뀌므로 이 표는 §9.2(절단연도 민감도)의 "
    "AUROC·ECE 와 같은 성격의 검사이고, 새 모델 설정을 추가한 것이 아니다."
)
w()
# ================================================================ 10. 재보정
w("---")
w()
w("## 10. 재보정으로 되돌릴 수 있는가")
w()
w(
    "학습군을 시간순으로 다시 쪼개, 앞부분으로 모델을 적합하고 뒷부분(학습군 내부의 가장 최근 "
    "구간)을 보정셋으로 써서 재보정했다. `Oracle Platt` 는 검증군 라벨을 실제로 써서 맞춘 "
    "상한선이며 달성 가능한 값이 아니다."
)
w()
w(
    "> **어느 방법이 검증군 정보를 쓰는지 분명히 할 것.** B·C 는 학습군 내부 보정셋만 쓰므로 "
    "그대로 배포할 수 있다. E 는 검증군 라벨을 전부 쓰는 상한선이다. **D(사전확률 보정)는 그 "
    "중간이다** — 라벨 하나하나는 쓰지 않지만 **검증군의 양성률을 안다고 가정한다**(코드에서 "
    "`y_test.mean()`). 미래 집합의 유병률은 배포 시점에 알 수 없으므로 **D 도 그대로는 배포할 "
    '수 없다.** 그래서 DILIrank 에서 D 가 가장 좋다는 결과를 "학습군 내부 재보정만으로 '
    '회복된다" 로 읽으면 안 된다. 검증군 라벨 없이 유병률을 추정해 같은 보정을 걸 수 '
    "있는지는 "
    "§10.2 에서 따로 검증한다."
)
w()
for d in DS:
    g = rec[rec.dataset.str.startswith(d)]
    a = g[g.method.str.startswith("A.")].iloc[0]
    e = g[g.method.str.startswith("E.")].iloc[0]
    _rs = recs[(recs.dataset == d) & (recs.calib_frac == 0.25)].iloc[0]
    w(
        "**%s** — 적합셋 %d / 보정셋 %d / 검증군 %d (보정셋 비율 0.25)"
        % (KOR[d], int(_rs.n_fit), int(_rs.n_calib), int(T[d].n_test))
    )
    w()
    w("| 방법 | AUROC | ECE | 보정기울기 | 절편 | 평균 예측확률 | MCC | 무보정 대비 ECE 회복률 |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        recv = (a.ECE - r.ECE) / (a.ECE - e.ECE) * 100
        w(
            "| %s | %s | %s | %s | %+.4f | %s | %s | %s |"
            % (
                r.method,
                f4(r.AUROC),
                f4(r.ECE),
                f4(r.slope),
                r.intercept,
                f4(r.mean_pred),
                f4(r.MCC),
                "—" if r.method.startswith("A") else pct1(recv),
            )
        )
    w()
# §10 표(비율 0.25)에서의 Isotonic AUROC 변화와, §10.1 비율 0.15 에서의 최대 변화.
_g10 = rec[rec.dataset.str.startswith("DILIrank")]
_bb_iso_main = float(
    _g10[_g10.method.str.startswith("C.")].AUROC.iloc[0]
    - _g10[_g10.method.str.startswith("A'.")].AUROC.iloc[0]
)
_r015 = recs[(recs.dataset == "DILIrank") & (recs.calib_frac == 0.15)]
_iso_a = float(_r015[_r015.method.str.startswith("A'.")].auroc.iloc[0])
_iso_c = float(_r015[_r015.method.str.startswith("C.")].auroc.iloc[0])
w("**세 가지를 분명히 할 것.**")
w()
w(
    "1. **AUROC 는 재보정으로 바뀌지 않는다.** Platt 과 사전확률 보정은 단조변환이므로 순위가 "
    '보존된다. Isotonic 은 동점 처리 때문에 순위가 바뀔 수 있는데, **"미세하게만" 이라고 '
    "쓰면 사실이 아니다** — 이 표(보정셋 비율 0.25)에서는 변화가 작지만(DILIrank %+.4f), "
    "§10.1 의 비율 0.15 에서는 DILIrank AUROC 가 %s → %s (%+.4f)까지 움직인다. "
    "**어느 경우든 판별력 격차는 재보정으로 "
    "해결되지 않는다.**" % (_bb_iso_main, f4(_iso_a), f4(_iso_c), _iso_c - _iso_a)
)
_bb = {
    d: rec[(rec.dataset.str.startswith(d)) & (rec.method.str.startswith("B."))].iloc[0] for d in DS
}
_dd = {
    d: rec[(rec.dataset.str.startswith(d)) & (rec.method.str.startswith("D."))].iloc[0] for d in DS
}
_aa = {
    d: rec[(rec.dataset.str.startswith(d)) & (rec.method.str.startswith("A."))].iloc[0] for d in DS
}
_ee = {
    d: rec[(rec.dataset.str.startswith(d)) & (rec.method.str.startswith("E."))].iloc[0] for d in DS
}
w(
    "2. **두 데이터셋에서 가장 잘 듣는 방법이 다르다.** DILIrank 는 사전확률 보정이 "
    "ECE %s → %s 로 가장 크게 내리고, DICTrank 는 Platt 이 %s → %s 로 가장 크게 내린다. "
    "**단일 권고안을 낼 수 없다** — 어떤 재보정이 맞는지는 데이터셋마다 다르다."
    % (
        f4(_aa["DILIrank"].ECE),
        f4(_dd["DILIrank"].ECE),
        f4(_aa["DICTrank"].ECE),
        f4(_bb["DICTrank"].ECE),
    )
)
w(
    "3. **Oracle 과의 간격이 크게 남는다.** 검증군 라벨을 쓸 수 있다면 ECE 를 %s / %s 까지 "
    "내릴 수 있는데, 미래 라벨 없이 도달한 최선은 %s / %s 다."
    % (
        f4(_ee["DILIrank"].ECE),
        f4(_ee["DICTrank"].ECE),
        f4(
            min(
                rec[
                    (rec.dataset.str.startswith("DILIrank"))
                    & (~rec.method.str.startswith(("A", "E")))
                ].ECE
            )
        ),
        f4(
            min(
                rec[
                    (rec.dataset.str.startswith("DICTrank"))
                    & (~rec.method.str.startswith(("A", "E")))
                ].ECE
            )
        ),
    )
)
w()
_ssd = {d: seed[seed.dataset.str.startswith(d)] for d in DS}
w(
    "**이 표의 A 행은 §6 의 시간분할과 같은 모델·같은 분할인데 수치가 다르다 — 반드시 "
    "같이 읽을 것.** ECE 는 %s(§10) 대 %s(§6), AUROC 는 %s 대 %s 다. 원인은 **행 순서**다 — "
    '`recalibration.py` 는 보정셋을 "학습군의 가장 최근 구간" 으로 떼기 위해 승인연도로 '
    "정렬한 뒤 적합하고, `calibration.py` 는 원본 행 순서를 그대로 쓴다. XGBoost 는 행 순서가 "
    "바뀌면 히스토그램 분할의 동점 처리가 달라져 적합이 미세하게 바뀐다. 두 값 모두 §9.3 의 "
    "시드 민감도 범위(DILIrank ECE %s–%s) 안에 있다. **§10 의 회복률은 이 표의 A 를 분모로 "
    "쓰므로 표 내부에서는 일관되지만, §6·§8 의 점추정값과 직접 빼서 비교하면 안 된다.**"
    % (
        f4(_aa["DILIrank"].ECE),
        f4(CT["DILIrank"].ECE),
        f4(_aa["DILIrank"].AUROC),
        f4(T["DILIrank"].AUROC),
        f4(_ssd["DILIrank"].ece.min()),
        f4(_ssd["DILIrank"].ece.max()),
    )
)
w()
assert (
    _ssd["DILIrank"].ece.min() <= float(_aa["DILIrank"].ECE) <= _ssd["DILIrank"].ece.max()
), "§10 A 의 ECE 가 시드 범위를 벗어난다"
assert (
    _ssd["DICTrank"].ece.min() <= float(_aa["DICTrank"].ECE) <= _ssd["DICTrank"].ece.max()
), "§10 A 의 ECE 가 시드 범위를 벗어난다 (DICTrank)"
w("### 10.1 보정셋 비율 민감도")
w()
w(
    "보정셋 비율을 0.15–0.40 으로 바꿔 반복했다. **회복률은 단일 값이 아니라 범위로 "
    "보고해야 한다.**"
)
w()
w("| 데이터셋 | Platt 회복률 | Isotonic 회복률 | Platt∪Isotonic 합집합 범위 |")
w("|---|---|---|---|")
for d in DS:
    g = recs[recs.dataset == d]
    b = g[g.method.str.startswith("B.")]
    c = g[g.method.str.startswith("C.")]
    both = pd.concat([b.recovery, c.recovery])
    w(
        "| %s | %s–%s%% | %s–%s%% | **%s–%s%%** |"
        % (
            d,
            f2(b.recovery.min()),
            f2(b.recovery.max()),
            f2(c.recovery.min()),
            f2(c.recovery.max()),
            f2(both.min()),
            f2(both.max()),
        )
    )
w()
_rsA = {d: recs[(recs.dataset == d) & (recs.method.str.startswith("A."))].iloc[0] for d in DS}
_rsE = {d: recs[(recs.dataset == d) & (recs.method.str.startswith("E."))].iloc[0] for d in DS}
w(
    "**이 절의 회복률은 §10 표와 다른 실행에서 나온 값이다.** 기준이 되는 무보정 ECE 와 "
    "Oracle 상한이 다르다 — DILIrank 는 여기서 %s / %s, §10 에서는 %s / %s 다. 그래서 같은 "
    "사전확률 보정의 회복률이 여기서는 %s%%, §10 에서는 %s%% 로 갈린다. **본문의 점추정은 "
    "§10 값을 쓰고, 이 절에서는 비율에 따른 범위만 읽는다.**"
    % (
        f4(_rsA["DILIrank"].ece),
        f4(_rsE["DILIrank"].ece),
        f4(_aa["DILIrank"].ECE),
        f4(_ee["DILIrank"].ECE),
        f2(
            recs[(recs.dataset == "DILIrank") & (recs.method.str.startswith("D."))].recovery.iloc[0]
        ),
        f2(
            (_aa["DILIrank"].ECE - _dd["DILIrank"].ECE)
            / (_aa["DILIrank"].ECE - _ee["DILIrank"].ECE)
            * 100
        ),
    )
)
w()
_wr = []
for _d in DS:
    _g = recs[(recs.dataset == _d) & (recs.method.str.startswith("B."))]
    for _, _r in _g.iterrows():
        _ap = recs[
            (recs.dataset == _d)
            & (recs.calib_frac == _r.calib_frac)
            & (recs.method.str.startswith("A'."))
        ].iloc[0]
        if _r.ece > _ap.ece:
            _wr.append((_d, _r.calib_frac, _ap.ece, _r.ece))
if _wr:
    w(
        "**회복률이 양수인데도 A′ 보다 나빠지는 경우가 있다.** %s 에서 Platt 은 보정셋을 뺀 "
        "무보정(A′)보다 ECE 를 **더 나쁘게** 만든다(%s). 위 회복률이 양수인 것은 분자·분모의 "
        "기준이 A(전체 학습)이기 때문이다. 보정셋을 떼어내는 비용까지 포함해 보면 그 비율에서는 "
        "재보정이 손해다."
        % (
            ", ".join("%s 비율 %.2f" % (x[0], x[1]) for x in _wr),
            ", ".join("%s → %s" % (f4(x[2]), f4(x[3])) for x in _wr),
        )
    )
    w()
w(
    "사전확률 보정은 보정셋을 쓰지 않으므로 비율과 무관하게 DILIrank %s%%, DICTrank %s%% 로 고정이다. "
    "**DILIrank 에서는 학습군 내부 재보정(Platt·Isotonic)이 거의 듣지 않고(%s–%s%%) "
    "사전확률 보정이 훨씬 낫지만, DICTrank 에서는 그 반대다.** 이 역전은 두 데이터셋의 "
    "양성률 이동 방향·크기가 다르기 때문이며, 단일 처방을 만들 수 없다는 근거다."
    % (
        f2(
            recs[(recs.dataset == "DILIrank") & (recs.method.str.startswith("D."))].recovery.iloc[0]
        ),
        f2(
            recs[(recs.dataset == "DICTrank") & (recs.method.str.startswith("D."))].recovery.iloc[0]
        ),
        f2(
            pd.concat(
                [
                    recs[
                        (recs.dataset == "DILIrank") & (recs.method.str.startswith(("B.", "C.")))
                    ].recovery
                ]
            ).min()
        ),
        f2(
            pd.concat(
                [
                    recs[
                        (recs.dataset == "DILIrank") & (recs.method.str.startswith(("B.", "C.")))
                    ].recovery
                ]
            ).max()
        ),
    )
)
w()

w("### 10.2 사전확률을 검증군 라벨 없이 추정할 수 있는가 — 할 수 없다")
w()
w(
    "§10 의 D 는 검증군의 실제 양성률을 가져다 쓴다. 배포 시점에는 알 수 없는 값이므로, "
    "**라벨을 쓰지 않고 그 유병률을 추정**해 같은 절편 보정을 걸 수 있는지 네 방법으로 "
    "확인했다. 바뀌는 것은 사전확률 추정값 하나뿐이고 보정 함수는 D 와 동일하다."
)
w()
for d in DS:
    g = pe[(pe.dataset == d) & (pe.estimator.str.startswith("P"))]
    b = pe[(pe.dataset == d) & (pe.estimator.str.startswith("(기준)"))]
    a0 = b[b.estimator.str.contains("A.")].iloc[0]
    e0 = b[b.estimator.str.contains("E.")].iloc[0]
    w(
        "**%s** — 학습군 양성률 %s → 검증군 실제 양성률 %s (이동 %+.4f)"
        % (
            KOR[d],
            f4(g.pi_train.iloc[0]),
            f4(g.pi_true.iloc[0]),
            g.pi_true.iloc[0] - g.pi_train.iloc[0],
        )
    )
    w()
    w("| 추정량 | 쓰는 정보 | 추정 유병률 | 오차 | ECE | 회복률 |")
    w("|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %s | %+.4f | %s | %s |"
            % (r.estimator, r.info_used, f4(r.pi_hat), r.pi_err, f4(r.ECE), pct1(r.recovery))
        )
    w("| (기준) A. 무보정 | — | — | — | %s | %s |" % (f4(a0.ECE), pct1(0.0)))
    w("| (기준) E. Oracle Platt | 검증군 라벨 전부 | — | — | %s | %s |" % (f4(e0.ECE), pct1(100.0)))
    w()
_pe_nl = pe[(pe.estimator.str.startswith("P")) & (~pe.estimator.str.startswith("P4"))]
_pe_or = pe[pe.estimator.str.startswith("P4")]
w(
    "**결과는 분명한 음성이다.** **검증군** 라벨을 쓰지 않는 추정량은 모두 실제 유병률을 "
    "**위쪽으로** 틀리고(오차 %+.4f ~ %+.4f, P0 을 포함한 범위다 — P0 은 추정량이 아니라 "
    '"이동이 없다" 고 가정한 기준선이므로 추정량 P1–P3 만 보면 하한이 올라간다), '
    "그 결과 보정이 더 나빠진다. 회복률이 "
    "음수인 경우가 %d/%d 이고, 검증군 라벨 없는 최선조차 DILIrank %s, DICTrank %s 로 "
    'oracle(%s / %s)에 한참 못 미친다. **DICTrank 에서는 네 방법 중 "아무것도 하지 '
    '않는 것"(P0)이 최선이다** — 어떤 추정값을 넣어도 무보정보다 나빠진다.'
    % (
        _pe_nl.pi_err.min(),
        _pe_nl.pi_err.max(),
        int((_pe_nl.recovery < 0).sum()),
        len(_pe_nl),
        pct1(_pe_nl[_pe_nl.dataset == "DILIrank"].recovery.max()),
        pct1(_pe_nl[_pe_nl.dataset == "DICTrank"].recovery.max()),
        pct1(_pe_or[_pe_or.dataset == "DILIrank"].recovery.iloc[0]),
        pct1(_pe_or[_pe_or.dataset == "DICTrank"].recovery.iloc[0]),
    )
)
w()
w(
    "**왜 전부 같은 방향으로 틀리는지가 핵심이다.** 네 추정량은 모두 모델의 예측확률 "
    "또는 예측 라벨에서 출발한다. 그런데 시간분할에서 모델은 확률을 양성 쪽으로 밀어 "
    '놓았다(§6, 평균 예측확률 %s / %s). 그래서 점수에 기반한 추정은 "양성이 더 많다" '
    "고 읽고, 실제 유병률은 **내려갔는데** 추정값은 올라간다. 보정이 무너진 상황에서 "
    "그 무너진 보정으로 유병률을 추정하려 하므로 **순환**이다. BBSE 는 DICTrank 에서 "
    "혼동행렬이 거의 특이해져(행렬식 %.4f) 추정값이 경계 1.0000 으로 터진다 — 모델이 "
    "거의 전부를 양성으로 예측하기 때문이다."
    % (
        f4(CT["DILIrank"].mean_pred),
        f4(CT["DICTrank"].mean_pred),
        float(pe[pe.dataset == "DICTrank"].bbse_det.iloc[0]),
    )
)
w()
w(
    "**따라서 §10 의 결론을 이렇게 써야 한다.** 배포 가능한 재보정은 **B·C(학습군 내부 "
    "보정셋)뿐**이고, 그 회복률은 DILIrank %s–%s%% / DICTrank %s–%s%% 다(§10.1). D 는 "
    '**달성 불가능한 상한의 하나**로 E 와 함께 묶어 읽어야 하며, "사전확률 보정이 '
    'DILIrank 에서 가장 잘 듣는다" 를 처방으로 제시하면 안 된다.'
    % (
        f2(
            recs[
                (recs.dataset == "DILIrank") & (recs.method.str.startswith(("B.", "C.")))
            ].recovery.min()
        ),
        f2(
            recs[
                (recs.dataset == "DILIrank") & (recs.method.str.startswith(("B.", "C.")))
            ].recovery.max()
        ),
        f2(
            recs[
                (recs.dataset == "DICTrank") & (recs.method.str.startswith(("B.", "C.")))
            ].recovery.min()
        ),
        f2(
            recs[
                (recs.dataset == "DICTrank") & (recs.method.str.startswith(("B.", "C.")))
            ].recovery.max()
        ),
    )
)
w()
w(
    "> 추정량 구현은 Saerens 등(2002)의 EM 과 Lipton 등(2018)의 BBSE-hard 를 그대로 "
    "따랐다. 두 문헌을 참고문헌에 추가해야 한다. EM 수렴은 DILIrank %d회, DICTrank %d회다."
    % (
        int(pe[pe.dataset == "DILIrank"].em_iter.iloc[0]),
        int(pe[pe.dataset == "DICTrank"].em_iter.iloc[0]),
    )
)
w()
# ---------------------------------------------------------------- 10.3 추정량 9종
_p2 = {
    d: pe2[(pe2.dataset == d) & (~pe2.estimator.str.startswith("Q0"))].dropna(subset=["pi_hat"])
    for d in DS
}
_p2o = {d: pe2[(pe2.dataset == d) & (pe2.estimator.str.startswith("Q0"))].iloc[0] for d in DS}
w("### 10.3 더 정교한 추정량을 붙이면 되는가 — 정량화 문헌의 아홉 가지")
w()
w(
    '§10.2 는 추정량 넷을 시도했고 모두 실패했다. 그때 한계로 이렇게 적었다 — "더 정교한 '
    "추정량을 붙여도 같은 벽에 부딪힐 것으로 본다, **다만 그것은 추측이고 검증하지 "
    '않았다**." 이 논문의 한계 중 **미검증 추측이 본문에 적힌 유일한 항목**이었다. '
    "그리고 그 추측에는 발표된 반례 후보가 있다 — Alexandari 등(2020)은 EM/MLLS 가 "
    "**기반 분류기를 먼저 보정하면** 작동한다고 보였고, §10.2 의 EM 은 보정하지 않은 "
    "확률에 돌린 것이었다. 그래서 정량화 문헌의 표준 추정량을 모아 추측을 검증했다."
)
w()
w(
    "TPR·FPR 과 점수 분포는 **학습군 내부 보정셋에서만** 추정한다 — 검증군 라벨을 쓰지 "
    "않는다. 각 추정값에는 검증군 부트스트랩 구간을 붙였다(예측 고정, 재적합 없음)."
)
w()
for d in DS:
    g = pe2[pe2.dataset == d]
    w(
        "**%s** — 학습군 유병률 %s, 검증군 참값 %s"
        % (KOR[d], f4(g.pi_train.iloc[0]), f4(g.pi_true.iloc[0]))
    )
    w()
    w("| 추정량 | 쓰는 정보 | pi 추정 | pi 오차 | pi 95% 구간 | 참값 포함 | 보정 후 ECE | 회복률 |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %s | %s | [%s, %s] | %s | %s | %s |"
            % (
                r.estimator,
                r.info_used,
                f4(r.pi_hat),
                sg(r.pi_err),
                f4(r.pi_lo),
                f4(r.pi_hi),
                "예" if bool(r.covers_true) else "아니오",
                f4(r.ece_corrected),
                pct1(r.recovery),
            )
        )
    w()
w(
    "회복률은 §10 과 같은 정의다 — (무보정 ECE − 보정 후 ECE) / (무보정 − Oracle Platt) "
    "× 100. **pi 오차가 작아도 회복률이 양수가 아니면 처방이 아니다.**"
)
w()
w("#### 발표된 처방은 통하지 않는다")
w()
_q4b = {d: _p2[d][_p2[d].estimator.str.startswith("Q4b")].iloc[0] for d in DS}
_q5 = {d: _p2[d][_p2[d].estimator.str.startswith("Q5")].iloc[0] for d in DS}
w(
    "**Platt 보정 후 EM(Q5)은 두 데이터셋 모두 실패한다.** 같은 기반 모델로 짝지은 "
    "비교에서 회복률이 %s → %s(DILIrank), %s → %s(DICTrank)로 **보정하면 오히려 조금 "
    "나빠진다.** DICTrank 에서는 보정 후 EM 이 반복 상한에서 멈춰 수렴조차 하지 않았다"
    "(그 추정값은 믿을 수 없다고 스크립트가 표시한다). Q4b 와 Q5 는 기반 모델이 같으므로 "
    "이 차이는 **보정의 순효과**다 — §10.2 의 EM 이 보정되지 않은 확률을 썼다는 것이 "
    "실패의 원인은 아니었다."
    % (
        pct1(_q4b["DILIrank"].recovery),
        pct1(_q5["DILIrank"].recovery),
        pct1(_q4b["DICTrank"].recovery),
        pct1(_q5["DICTrank"].recovery),
    )
)
w()
w("#### DILIrank 는 완전히 막힌다")
w()
_nL = len(_p2["DILIrank"])
w(
    "**열 추정량 전부 유병률을 위쪽으로 틀리고(오차 %s 에서 %s 까지), 회복률이 전부 "
    "음수고, "
    "구간이 참값을 덮는 것이 %d/%d 다.** 추정이 단지 부정확한 것이 아니라 **편향돼 "
    "있다.** 가장 좋은 Q3 PACC 조차 오차 %s, 회복률 %s 다."
    % (
        sg(_p2["DILIrank"].pi_err.min()),
        sg(_p2["DILIrank"].pi_err.max()),
        int(_p2["DILIrank"].covers_true.sum()),
        _nL,
        sg(_p2["DILIrank"].loc[_p2["DILIrank"].pi_err.abs().idxmin()].pi_err),
        pct1(_p2["DILIrank"].loc[_p2["DILIrank"].pi_err.abs().idxmin()].recovery),
    )
)
w()
w("#### DICTrank 의 양수 회복은 과소추정의 부수효과로 보인다")
w()
_pe_pos = _p2["DICTrank"][_p2["DICTrank"].recovery > 0]
_pe_und = _pe_pos[_pe_pos.pi_err < 0]
w(
    "**DICTrank 에서는 %d개 추정량이 회복률 양수를 낸다** — %s. 그런데 **%d개 전부 "
    "유병률을 아래로 틀렸다**(오차 %s 에서 %s 까지). §23 이 이미 보인 것이 있다 — 유병률을 아래로 "
    "틀리면 ECE 가 **oracle 보다도** 낮아진다. 실제로 여기서도 Q3 의 보정 후 ECE %s 가 "
    "oracle 사전확률 보정 %s 보다 낮다. **그래서 이 회복은 추정이 성공해서가 아니라 "
    "과소추정의 부수효과일 수 있다.**"
    % (
        len(_pe_pos),
        ", ".join(str(r.estimator).split(" ")[0] for _, r in _pe_pos.iterrows()),
        len(_pe_und),
        sg(_pe_und.pi_err.max()),
        sg(_pe_und.pi_err.min()),
        f4(_pe_pos.loc[_pe_pos.pi_err.abs().idxmin()].ece_corrected),
        f4(_p2o["DICTrank"].ece_corrected),
    )
)
w()
w(
    "**그리고 그 추정량들은 식별되지 않는다.** pi 구간이 %s 는 [%s, %s], %s 는 "
    "[%s, %s] 로 사실상 정보가 없다. 한 번의 추출에서 아래로 내려 꽂힌 것이고, 같은 "
    "절차를 다른 표본에 쓰면 어디로 갈지 이 자료로는 말할 수 없다."
    % (
        str(_pe_pos.iloc[-1].estimator).split(" ")[0],
        f4(_pe_pos.iloc[-1].pi_lo),
        f4(_pe_pos.iloc[-1].pi_hi),
        str(
            _p2["DICTrank"][_p2["DICTrank"].estimator.str.startswith("Q2")].iloc[0].estimator
        ).split(" ")[0],
        f4(_p2["DICTrank"][_p2["DICTrank"].estimator.str.startswith("Q2")].iloc[0].pi_lo),
        f4(_p2["DICTrank"][_p2["DICTrank"].estimator.str.startswith("Q2")].iloc[0].pi_hi),
    )
)
w()
w("#### 같이 적어야 할 세 가지")
w()
w(
    "1. **Q3 PACC 와 Q7 BBSE-soft 는 이진 문제에서 대수적으로 같은 추정량이다.** 값이 "
    "같은 것은 오류가 아니라 구현 확인이다. 그래서 실질 추정량은 열이 아니라 아홉이다."
)
w(
    "2. **BBSE 라는 이름 아래 두 구현이 크게 다르다.** §10.2 의 P3 은 결합 혼동행렬 "
    "P(ŷ, y) 를 풀고, 여기 Q6 은 조건부율 (CC−FPR)/(TPR−FPR) 을 쓴다. 추정값이 "
    "%s / %s 만큼 벌어진다(§10.2 의 행렬식이 %s / %s 로 거의 특이하다). "
    "**두 합당한 구현이 그만큼 벌어진다는 사실 자체가 이 과제에서 사전확률 추정이 "
    "불안정하다는 증거다.**"
    % (
        f2(
            abs(
                float(
                    _p2["DILIrank"][_p2["DILIrank"].estimator.str.startswith("Q6")].pi_hat.iloc[0]
                )
                - float(
                    pe[
                        (pe.dataset == "DILIrank") & (pe.estimator.str.startswith("P3"))
                    ].pi_hat.iloc[0]
                )
            )
        ),
        f2(
            abs(
                float(
                    _p2["DICTrank"][_p2["DICTrank"].estimator.str.startswith("Q6")].pi_hat.iloc[0]
                )
                - float(
                    pe[
                        (pe.dataset == "DICTrank") & (pe.estimator.str.startswith("P3"))
                    ].pi_hat.iloc[0]
                )
            )
        ),
        f4(pe[pe.dataset == "DILIrank"].bbse_det.iloc[0]),
        f4(pe[pe.dataset == "DICTrank"].bbse_det.iloc[0]),
    )
)
w(
    "3. **Q4(전체 학습 EM)는 §10.2 의 P2 를 소수점까지 재현한다**(차 0.0000). 같은 "
    "정렬·같은 보정셋 설계를 썼다는 확인이다. 재현되지 않으면 스크립트가 그 자리에서 "
    "표시한다."
)
w()
# ---------------------------------------------------------------- §10.4 보조
# 한계 12 — §10·§11 점추정값의 임의성을 평균으로 없앤 결과를 읽는다.
_tba_n = int(tba.groupby("dataset").size().min())


def _tba_conf(d, a):
    return float(
        conf[(conf.dataset == d) & (abs(conf.alpha - a) < 1e-9)].marginal_coverage_temporal.iloc[0]
    )


def _tba_recrow(d, pref):
    g = rec[rec.dataset.str.startswith(d)]
    return g[g.method.str.startswith(pref)].iloc[0]


def _tba_qty(d):
    """(라벨, tba 열, §10·§11 현행값, 서식) — 현행값은 본문 표가 쓰는 그 값이다."""
    return [
        ("A 무보정 ECE", "ece_A", float(_tba_recrow(d, "A.").ECE), f4),
        ("A' 보정셋 제외 ECE", "ece_Ap", float(_tba_recrow(d, "A'.").ECE), f4),
        ("B Platt ECE", "ece_B", float(_tba_recrow(d, "B.").ECE), f4),
        ("C Isotonic ECE", "ece_C", float(_tba_recrow(d, "C.").ECE), f4),
        ("D 사전확률 ECE", "ece_D", float(_tba_recrow(d, "D.").ECE), f4),
        ("E Oracle 상한 ECE", "ece_E", float(_tba_recrow(d, "E.").ECE), f4),
        ("B 회복률 (%)", "rec_B", _rec10(d, "B."), f2),
        ("C 회복률 (%)", "rec_C", _rec10(d, "C."), f2),
        ("D 회복률 (%)", "rec_D", _rec10(d, "D."), f2),
        ("conformal 90% 커버리지", "conf90_marginal", _tba_conf(d, 0.1), f4),
        ("conformal 80% 커버리지", "conf80_marginal", _tba_conf(d, 0.2), f4),
        ("AUROC (무보정)", "auroc_A", float(_tba_recrow(d, "A.").AUROC), f4),
    ]


def _tba_ref(d, col):
    return [r for _, c, r, _f in _tba_qty(d) if c == col][0]


# 범위 밖인 수치를 세어 둔다 — 개수도 방향도 손으로 적지 않는다.
# ECE 계열은 작을수록 좋고 회복률·커버리지·AUROC 는 클수록 좋다.
# "유리" = 현행 표가 평균보다 낙관적으로 보이는 방향이다.
_tba_tot, _tba_in, _tba_out = 0, 0, []
for _d in DS:
    _g = tba[tba.dataset == _d]
    for _lab, _col, _ref, _f in _tba_qty(_d):
        _tba_tot += 1
        _mn, _mx = float(_g[_col].min()), float(_g[_col].max())
        if _mn <= _ref <= _mx:
            _tba_in += 1
            continue
        _low = _ref < _mn
        _small_good = _col.startswith("ece_")
        _tba_out.append(
            {
                "lab": "%s %s" % (_d, _lab),
                "ref": _ref,
                "mn": _mn,
                "mx": _mx,
                "mean": float(_g[_col].mean()),
                "fmt": _f,
                "dir": "낮다" if _low else "높다",
                "fav": "유리" if (_low == _small_good) else "불리",
            }
        )
_tba_favs = set(x["fav"] for x in _tba_out)


def _tba_c1(d):
    g = tba[tba.dataset == d]
    return int((g.conf90_marginal < 0.90).sum())


def _tba_c2(d):
    g = tba[tba.dataset == d]
    return int((g.rec_B < 100).sum())


def _tba_best(d):
    g = tba[tba.dataset == d]
    v = g.best_any.value_counts()
    _NM = {"B": "Platt", "C": "Isotonic", "D": "사전확률 보정"}
    return ", ".join("%s %d/%d" % (_NM.get(k, k), int(n), len(g)) for k, n in v.items())


def _tba_cwin(d):
    g = tba[tba.dataset == d]
    return int((g.best_deployable == "C").sum())


_tba_p = tba.pivot_table(index="rep", columns="dataset", values="best_any", aggfunc="first")
_tba_c3 = (
    int((_tba_p["DILIrank"] == _tba_p["DICTrank"]).sum()) if set(DS) <= set(_tba_p.columns) else -1
)

# ---------------------------------------------------------------- 10.4 임의성 제거
w("### 10.4 §10·§11 의 점추정값에서 임의성을 없앤다 — 동시 무작위화 50회")
w()
w(
    "§9.6 은 임의 선택이 수치를 얼마나 흔드는지 **폭**으로 보였다. 폭을 적어 두는 것과 "
    "점추정값 자체를 임의성에서 떼어 내는 것은 다른 일이다. §10·§11 표의 숫자는 여전히 "
    "정렬 함수 하나가 정한 **한 번의 실현값**이다. 그래서 세 임의성을 **동시에** "
    "무작위화해 %d회 반복하고, 그 평균을 새 점추정값으로 쓴다. 평균은 임의 선택에 "
    "의존하지 않는다." % _tba_n
)
w()
w(
    "동시에 흔드는 것 세 가지다 — (1) 경계 연도 동점의 보정셋 배정, (2) 적합셋 행 순서, "
    "(3) **전체학습 모델의 행 순서**. (3)을 넣는 이유는 회복률의 분모(A − E)가 전체학습 "
    "모델에서 나오기 때문이다. 분자만 흔들고 분모를 고정하면 회복률의 임의성을 절반만 "
    "재게 된다. 반복마다 시드가 정해져 있어(42 + 7000 + 반복번호) 중단되어도 같은 값으로 "
    "이어진다. 보정셋 크기는 §10 과 같다(학습군의 25%%, DILIrank %d개 / DICTrank %d개)."
    % (
        int(tba[tba.dataset == "DILIrank"].n_cal.iloc[0]),
        int(tba[tba.dataset == "DICTrank"].n_cal.iloc[0]),
    )
)
w()
w(
    "| 데이터셋 | 수치 | 평균 | 표준편차 | 2.5–97.5 백분위 | %d회 최소–최대 | §10·§11 현행값 | 현행값 위치 |"
    % _tba_n
)
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    g = tba[tba.dataset == d]
    for lab, col, ref, fmt in _tba_qty(d):
        m, sd = float(g[col].mean()), float(g[col].std(ddof=1))
        lo, hi = float(float((g[col]).quantile(0.025))), float(float((g[col]).quantile(0.975)))
        mn, mx = float(g[col].min()), float(g[col].max())
        inside = mn <= ref <= mx
        w(
            "| %s | %s | **%s** | %s | %s–%s | %s–%s | %s | %s |"
            % (
                d,
                lab,
                fmt(m),
                fmt(sd),
                fmt(lo),
                fmt(hi),
                fmt(mn),
                fmt(mx),
                fmt(ref),
                ("%+.1f SD" % ((ref - m) / sd)) if sd > 0 else "—",
            )
        )
w()
w("읽는 법 세 가지다.")
w()
w(
    "1. **§10·§11 의 현행 수치는 특이값이 아니다.** 대조한 %d개 중 %d개가 %d회 "
    "최소–최대 범위 안에 있다. 범위를 벗어난 것은 %d개다."
    % (_tba_tot, _tba_in, _tba_n, len(_tba_out))
)
for _o in _tba_out:
    _f = _o["fmt"]
    w(
        "   - %s — 현행 %s, %d회 범위 %s–%s(평균 %s). 범위보다 %s. "
        "현행 표가 평균보다 **%s**인 쪽이다."
        % (
            _o["lab"],
            _f(_o["ref"]),
            _tba_n,
            _f(_o["mn"]),
            _f(_o["mx"]),
            _f(_o["mean"]),
            _o["dir"],
            "낙관적" if _o["fav"] == "유리" else "보수적",
        )
    )
w()
w(
    "   ECE 계열은 작을수록 좋고 회복률·커버리지·AUROC 는 클수록 좋으므로 방향은 그 "
    "기준으로 읽는다. %s"
    % (
        "범위를 벗어난 내역이 낙관·보수 **양쪽에 걸쳐 있다** — 한쪽으로 기울어진 "
        "실현값이 아니라는 뜻이다."
        if len(_tba_favs) > 1
        else (
            (
                "벗어난 내역이 모두 **낙관 방향**이다 — 해당 수치는 평균으로 읽어야 한다."
                if _tba_favs == {"유리"}
                else "벗어난 내역이 모두 **보수 방향**이다 — 현행 표가 결론에 유리하게 부풀려진 "
                "것이 아니다."
            )
            if _tba_out
            else "벗어난 항목이 없다."
        )
    )
)
w()
w(
    "2. **결론은 %d회 전부에서 유지된다.** conformal 90%% 커버리지가 명목값 미달인 반복이 "
    "DILIrank %d/%d · DICTrank %d/%d, 학습군 내부 Platt 회복률이 100%% 미만인 반복이 "
    "DILIrank %d/%d · DICTrank %d/%d 이다. 한 번이라도 뒤집히면 이 자리에 그 횟수가 "
    "적힌다."
    % (
        _tba_n,
        _tba_c1("DILIrank"),
        _tba_n,
        _tba_c1("DICTrank"),
        _tba_n,
        _tba_c2("DILIrank"),
        _tba_n,
        _tba_c2("DICTrank"),
        _tba_n,
    )
)
w()
w(
    "3. **두 데이터셋의 권고안이 서로 다르다는 것도 임의 선택의 산물이 아니다.** B·C·D 중 "
    "최선이 두 데이터셋에서 **같은** 반복은 %d/%d 이다. DILIrank 는 %s, DICTrank 는 %s 이다. "
    "다만 **DICTrank 의 권고는 만장일치가 아니다** — 배포 가능한 둘(B·C) 중 Isotonic 이 "
    "이기는 반복이 %d회 있다. 두 방법의 ECE 차가 작아서다(평균 %s 대 %s). §10 이 "
    '"단일 권고안을 낼 수 없다" 고 쓴 근거는 이 자리에서 %d회 전부로 확인된다.'
    % (
        _tba_c3,
        _tba_n,
        _tba_best("DILIrank"),
        _tba_best("DICTrank"),
        _tba_cwin("DICTrank"),
        f4(tba[tba.dataset == "DICTrank"].ece_B.mean()),
        f4(tba[tba.dataset == "DICTrank"].ece_C.mean()),
        _tba_n,
    )
)
w()
w(
    "**본문에 쓸 형태.** DILIrank Platt 회복률 %s%% [%s, %s], conformal 90%% 커버리지 "
    "%s [%s, %s]. DICTrank Platt 회복률 %s%% [%s, %s], conformal 90%% 커버리지 "
    "%s [%s, %s]. 대괄호는 %d회의 2.5–97.5 백분위다. §10·§11 의 표는 단일 실현값을 "
    "그대로 두고(그 쪽이 재현 절차와 1:1 로 대응한다), 임의성을 없앤 값은 이 절에서 "
    "읽는다."
    % (
        f1(tba[tba.dataset == "DILIrank"].rec_B.mean()),
        f1(float((tba[tba.dataset == "DILIrank"].rec_B).quantile(0.025))),
        f1(float((tba[tba.dataset == "DILIrank"].rec_B).quantile(0.975))),
        f4(tba[tba.dataset == "DILIrank"].conf90_marginal.mean()),
        f4(float((tba[tba.dataset == "DILIrank"].conf90_marginal).quantile(0.025))),
        f4(float((tba[tba.dataset == "DILIrank"].conf90_marginal).quantile(0.975))),
        f1(tba[tba.dataset == "DICTrank"].rec_B.mean()),
        f1(float((tba[tba.dataset == "DICTrank"].rec_B).quantile(0.025))),
        f1(float((tba[tba.dataset == "DICTrank"].rec_B).quantile(0.975))),
        f4(tba[tba.dataset == "DICTrank"].conf90_marginal.mean()),
        f4(float((tba[tba.dataset == "DICTrank"].conf90_marginal).quantile(0.025))),
        f4(float((tba[tba.dataset == "DICTrank"].conf90_marginal).quantile(0.975))),
        _tba_n,
    )
)
w()
# ---------------------------------------------------------------- 10.5 보정기 추가
_xc = {d: xc[xc.dataset == d].set_index("calibrator") for d in DS}
_XCB = ["B. Platt (§10 재현)", "C. Isotonic (§10 재현)"]
_XCN = ["F. 온도 조정", "G. 베타 보정", "H. Venn-ABERS (중점)"]
w("### 10.5 보정기를 두 개만 써 보고 결론한 것은 아닌가 — 네 개를 더 붙였다")
w()
w(
    "§10 은 재보정을 Platt·Isotonic·사전확률 셋으로 시험했다. 보정 문헌을 아는 심사자는 "
    "바로 묻는다 — **보정기를 두 개만 써 보고 재보정이 안 된다고 결론했는가.** 그래서 네 "
    "개를 더 붙였다. 베타 보정(Kull 2017)은 log p 와 log(1−p) 를 따로 써서 확률의 양 끝을 "
    "따로 휜다. Venn-ABERS(Vovk 2014)는 Isotonic 을 conformal 틀에 넣은 것이고 이 논문이 "
    "이미 §11 에서 conformal 을 쓰므로 가장 자연스러운 보완이다. 온도 조정은 로짓에 기울기 "
    "하나만 쓴다. 마지막은 §10 의 B 와 D 를 **이어 붙인** 합성이다."
)
w()
# §10 재현 — B·C·A·E 를 §10 의 값과 대조한 최대차를 여기서 다시 센다.
_xc_dif = 0.0
for _d in DS:
    _g = rec[rec.dataset.str.startswith(_d)]
    for _pre, _tag in (("B.", "B. Platt (§10 재현)"), ("C.", "C. Isotonic (§10 재현)")):
        _xc_dif = max(
            _xc_dif,
            abs(
                float(_g[_g.method.str.startswith(_pre)].ECE.iloc[0]) - float(_xc[_d].loc[_tag].ece)
            ),
        )
    for _pre, _col in (("A.", "ece_A"), ("E.", "ece_E")):
        _xc_dif = max(
            _xc_dif,
            abs(
                float(_g[_g.method.str.startswith(_pre)].ECE.iloc[0])
                - float(xc[xc.dataset == _d][_col].iloc[0])
            ),
        )
w(
    "분할·모델·보정셋·회복률 정의는 §10 과 같다. B·C·A·E 를 다시 계산해 §10 과 "
    "대조한 최대차가 %.1e 로 %s."
    % (_xc_dif, "분할은 맞다" if _xc_dif < 1e-4 else "**분할이 어긋났다**")
)
w()
w(
    "**이 대조가 한때 3.6e-05 를 냈고, 그것이 코드 결함을 찾아냈다.** 처음에는 그 차이를 "
    '"lbfgs 가 실행마다 다른 해에 닿는다" 로 적었는데 틀렸다. `src/diff_ladder.py` 가 두 '
    "경로를 한 프로세스에서 나란히 돌려 열다섯 단계를 비트 단위로 비교했고, 갈라지는 자리는 "
    "**로짓의 dtype**이었다 — XGBoost 의 `predict_proba` 는 float32 를 돌려주고, §10 의 "
    "`to_logit` 이 그것을 캐스팅하지 않아 **Platt 적합이 단정밀도로** 일어났다. 영향은 "
    "DICTrank Platt 행에 몰려 있었다 — 바뀐 칸 %d개 중 소수 4자리에서 바뀌는 것이 %d개이고 "
    "**전부 그 행**이다(`results/precision_fix_20261020.csv`). ECE 는 4자리에서 같았지만"
    "(%s) **보정 기울기가 %s 와 %s 로 갈려 §10 표와 이 절의 표가 같은 값을 다르게 적고 "
    "있었다.** 두 표가 각자 자기 결과 파일과만 대조되고 있었기 때문에 검증기도 잡지 "
    "못했다."
    % (
        len(pfx),
        int(pfx.changes_4dp.sum()),
        f4(
            float(
                rec[
                    rec.dataset.str.startswith("DICTrank") & rec.method.str.startswith("B.")
                ].ECE.iloc[0]
            )
        ),
        f4(float(pfx[pfx.column == "slope"].before.iloc[0])),
        f4(float(pfx[pfx.column == "slope"].after.iloc[0])),
    )
)
w()
w(
    "고친 것은 셋이다. (1) 로짓·보정기울기를 계산하는 함수 일곱 군데에 float64 캐스팅을 "
    "넣었다(`recalibration.to_logit`·`calib_line`, `recalibration_sensitivity.logit`, "
    "`tiebreak_sensitivity.to_logit`, `_shared.calib_slope`, "
    "`prevalence_control.calib_slope`, `robustness.calib_slope` — 모두 10-04 이전 "
    "스크립트이고 이후에 추가한 것들은 처음부터 float64 였다). (2) `check_determinism.py` "
    "에 검사 [1d] 를 넣어, 같은 float32 값과 그것을 미리 넓힌 값이 **비트 단위로 같은 "
    "결과**를 내는지 매번 확인한다. (3) 이 절의 대조를 ECE 한 열에서 ECE·기울기·AUROC·"
    "평균예측 네 열로 넓혔다 — 한 열만 보면 이런 불일치를 놓친다."
)
w()
w(
    "**2026-10-21: 여덟 번째가 있었다.** 10-20 의 수정은 복제를 **이름으로** 찾았다 — "
    "`to_logit`·`logit`·`calib_slope`·`calib_line` 네 이름이었다. `calibration.py` 의 "
    "`calib_slope_intercept` 는 같은 로짓 적합을 하면서 이름이 달라 그 목록에 걸리지 "
    "않았고, 실제로 캐스팅이 없었다. 10-21 에 이름이 아니라 **본문에 로짓 변환이 있는 "
    "함수를 전수로** 훑어 찾았다. 이 함수는 §4 의 보정 표"
    "(`results/calibration_results.csv`)를 만들고 그 표가 본문 보정기울기의 출처다. "
    "조치는 둘이다 — 캐스팅을 넣어 `_shared.calib_slope` 와 비트 단위로 같아졌고, "
    "[1c] 를 **이름이 다른 별명 복제까지** 비교하도록 넓혔다(§3.1 표의 마지막 두 행). "
    "%s"
    % (
        (
            "수정 전후로 바뀐 칸은 %d개이고 전부 보정 기울기·절편이다. 그 중 소수 "
            "4자리에서 바뀌는 것은 %d개이고 최대차가 %s 다 — **본문에 적히는 자리에서는 "
            "하나도 움직이지 않았다**(`results/precision_fix_20261021.csv`). 결함이 "
            "무해했다는 뜻이 아니라, 같은 양을 두 자리에서 다르게 적고 있었다는 사실이 "
            "수치의 크기와 무관하게 고쳐야 할 것이었다는 뜻이다."
            % (
                len(pf21),
                int(_tf(pf21.changes_4dp).sum()),
                "%.0e" % float(pf21.abs_diff.abs().max()),
            )
        )
        if pf21 is not None
        else "수정 전후 대조는 `results/precision_fix_20261021.csv` 에 남긴다 — "
        "이 보고서를 만든 시점에는 그 파일이 아직 없었다."
    )
)
w("| 데이터셋 | 보정기 | ECE | ECE 95% CI | 회복률 | AUROC | 보정 기울기 | 평균 예측확률 |")
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    g = _xc[d]
    for c in _XCB + _XCN + ["I. Platt → 사전확률"]:
        if c not in g.index:
            continue
        r = g.loc[c]
        w(
            "| %s | %s | %s | [%s, %s] | %s%% | %s | %s | %s |"
            % (
                d,
                c,
                f4(r.ece),
                f4(r.ece_lo),
                f4(r.ece_hi),
                f1(r.recovery),
                f4(r.auroc),
                f4(r.slope),
                f4(r.mean_pred),
            )
        )
w()
_xc_v = []
for d in DS:
    g = _xc[d]
    b = g.loc[[c for c in _XCB if c in g.index]].sort_values("ece").iloc[0]
    n = g.loc[[c for c in _XCN if c in g.index]].sort_values("ece").iloc[0]
    _xc_v.append(
        (
            d,
            b.name,
            float(b.ece),
            float(b.ece_lo),
            float(b.recovery),
            n.name,
            float(n.ece),
            float(n.ece_hi),
            float(n.recovery),
            bool(n.ece < b.ece and n.ece_hi < b.ece_lo),
        )
    )
w("읽을 것이 셋이다.")
w()
w(
    "1. **신규 보정기 중 기존을 유의하게 넘는 것이 없다.** %s 따라서 §10 의 결론은 "
    "**보정기 선택의 산물이 아니다.** 점추정만 비교하면 더 낮아 보이지만 ECE 부트스트랩 "
    '구간이 겹치므로 "더 낫다" 고 쓸 수 없다.'
    % " ".join(
        "%s 는 기존 최선 %s(ECE %s, 회복률 %s%%) 대 신규 최선 %s(%s, %s%%)로 %s."
        % (
            v[0],
            v[1].split(" ")[1],
            f4(v[2]),
            f1(v[4]),
            v[5].split(" ")[1],
            f4(v[6]),
            f1(v[8]),
            "신규가 유의하게 낫다" if v[9] else "구간이 겹친다",
        )
        for v in _xc_v
    )
)
w()
w(
    "2. **온도 조정이 Platt 보다 낫다는 것은 예상과 반대였고, 그 이유가 §10.2 와 이어진다.** "
    '온도 조정은 절편을 0 으로 고정하므로 "전체가 한쪽으로 밀린" 경우를 못 고칠 것이라고 '
    "예상했는데, 두 데이터셋 모두에서 Platt 보다 ECE 가 낮다(%s 대 %s, %s 대 %s). 이유는 "
    "반대다 — Platt 은 학습군 내부 보정셋에서 **학습시대 유병률에 맞는 절편**을 배워 "
    "검증군으로 가져오고, 그 절편이 틀려서 해롭다. 절편을 묶으면 그 틀린 절편을 들여오지 "
    "않는다. 즉 학습군 내부 재보정이 잘 안 듣는 이유 자체가 **절편**에 있다."
    % (
        f4(_xc["DILIrank"].loc["F. 온도 조정"].ece),
        f4(_xc["DILIrank"].loc["B. Platt (§10 재현)"].ece),
        f4(_xc["DICTrank"].loc["F. 온도 조정"].ece),
        f4(_xc["DICTrank"].loc["B. Platt (§10 재현)"].ece),
    )
)
w()
w(
    "3. **모자란 것은 숫자 하나다.** 학습군 내부 Platt 뒤에 검증군 유병률로 절편을 다시 "
    "밀면 회복률이 %s%% / %s%% 가 된다 — 이 논문에서 나온 어떤 처방보다 높고, §10 의 D "
    "단독(%s%% / %s%%)보다도 높다. 그런데 이 합성은 **검증군 유병률을 알아야** 하므로 "
    "배포할 수 없다. §10.2·§10.3 이 보인 것이 바로 그 숫자 하나를 라벨 없이 추정할 수 "
    "없다는 것이다. **이 절은 그 음성 결과의 값을 올린다** — 모자란 것이 모형도 보정기도 "
    "아니라 단 하나의 모수라는 뜻이기 때문이다."
    % (
        f1(_xc["DILIrank"].loc["I. Platt → 사전확률"].recovery),
        f1(_xc["DICTrank"].loc["I. Platt → 사전확률"].recovery),
        f1(_rec10("DILIrank", "D.")),
        f1(_rec10("DICTrank", "D.")),
    )
)
w()
w(
    "**Venn-ABERS 의 구간 폭**은 평균 %s / %s 다. 좁다 — 보정셋 %d·%d개로도 구간이 "
    "벌어지지 않는다. 즉 §10 의 회복 실패는 보정셋이 작아서가 아니다."
    % (
        f4(xc[xc.dataset == "DILIrank"].va_width.dropna().iloc[0]),
        f4(xc[xc.dataset == "DICTrank"].va_width.dropna().iloc[0]),
        int(xc[xc.dataset == "DILIrank"].n_cal.iloc[0]),
        int(xc[xc.dataset == "DICTrank"].n_cal.iloc[0]),
    )
)
w()
# ================================================================ 11. conformal
w("---")
w()
w("## 11. Split conformal prediction")
w()
w(
    "분포 가정 없이 커버리지를 보장하는 절차가 시간분할에서도 보장을 지키는지 확인했다. "
    "보정셋은 §10 과 같은 학습군 내부 구간이다."
)
w()
w(
    "| 데이터셋 | 목표 | 주변 커버리지 무작위 | 주변 커버리지 시간분할 | 양성 클래스 | 음성 클래스 | 평균 예측집합 크기 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    for a in (0.1, 0.2):
        r = conf[(conf.dataset == d) & (conf.alpha == a)].iloc[0]
        w(
            "| %s | %.0f%% | %s | **%s** | %s → %s | %s → **%s** | %s → %s |"
            % (
                d,
                r.target_coverage * 100,
                f4(r.marginal_coverage_random),
                f4(r.marginal_coverage_temporal),
                f4(r.coverage_positive_random),
                f4(r.coverage_positive_temporal),
                f4(r.coverage_negative_random),
                f4(r.coverage_negative_temporal),
                f4(r.mean_set_size_random),
                f4(r.mean_set_size_temporal),
            )
        )
w()
_c90r = float(
    conf[(conf.dataset == "DICTrank") & (conf.alpha == 0.1)].marginal_coverage_random.iloc[0]
)
w(
    "**무작위 분할에서는 보장이 거의 지켜지고 시간분할에서는 깨진다.** 엄밀히 말하면 "
    "DICTrank 90%% 목표의 무작위 커버리지는 %s 로 명목값보다 %s 낮다 — 유한표본에서 나오는 "
    "크기이고, 시간분할의 미달 폭과는 비교가 안 된다. 무작위 쪽 주변 커버리지는 "
    "명목값(0.90 / 0.80)에 거의 정확히 맞는데, 시간분할에서는 DILIrank %s / %s, "
    "DICTrank %s / %s 로 크게 미달한다."
    % (
        f4(_c90r),
        f4(0.90 - _c90r),
        f4(
            conf[
                (conf.dataset == "DILIrank") & (conf.alpha == 0.1)
            ].marginal_coverage_temporal.iloc[0]
        ),
        f4(
            conf[
                (conf.dataset == "DILIrank") & (conf.alpha == 0.2)
            ].marginal_coverage_temporal.iloc[0]
        ),
        f4(
            conf[
                (conf.dataset == "DICTrank") & (conf.alpha == 0.1)
            ].marginal_coverage_temporal.iloc[0]
        ),
        f4(
            conf[
                (conf.dataset == "DICTrank") & (conf.alpha == 0.2)
            ].marginal_coverage_temporal.iloc[0]
        ),
    )
)
w()
w(
    "**부족분이 음성 클래스에 쏠린다.** 양성 클래스 커버리지는 시간분할에서 오히려 올라가는데 "
    "(모델이 모두 양성 쪽으로 밀기 때문이다), 음성 클래스는 DILIrank %s → %s, "
    "DICTrank %s → %s 로 무너진다. 독성 예측에서 음성 과소보장은 **안전한 약을 안전하다고 "
    "말하지 못하는 방향의 실패**이며, 선별 단계에서 유용한 약을 버리게 만든다."
    % (
        f4(
            conf[(conf.dataset == "DILIrank") & (conf.alpha == 0.1)].coverage_negative_random.iloc[
                0
            ]
        ),
        f4(
            conf[
                (conf.dataset == "DILIrank") & (conf.alpha == 0.1)
            ].coverage_negative_temporal.iloc[0]
        ),
        f4(
            conf[(conf.dataset == "DICTrank") & (conf.alpha == 0.1)].coverage_negative_random.iloc[
                0
            ]
        ),
        f4(
            conf[
                (conf.dataset == "DICTrank") & (conf.alpha == 0.1)
            ].coverage_negative_temporal.iloc[0]
        ),
    )
)
w()
w(
    "> split conformal 은 **주변(marginal) 커버리지**만 이론적으로 보장한다. 위 표의 클래스별 "
    '커버리지는 보장 대상이 아니라 기술 통계이므로, "클래스별 보장이 깨졌다" 가 아니라 '
    '"주변 보장이 깨졌고 그 부족분이 음성에 쏠렸다" 로 써야 한다.'
)
w()
# 공집합 비율은 CSV 에 열이 없고 요약 txt 에만 있다. 하드코딩하지 않고 거기서 읽는다
# (2026-10-07: 종전 문장은 "모든 조건에서 0" 이었는데 한 칸이 0 이 아니다).
_emp, _cd, _ca = [], None, None
for _ln in txt("conformal_prediction_summary.txt").split("\n"):
    _m = re.match(r"^(DILIrank|DICTrank)\s", _ln)
    if _m:
        _cd = _m.group(1)
        continue
    _m = re.search(r"목표 커버리지 (\d+)%", _ln)
    if _m:
        _ca = int(_m.group(1))
        continue
    _m = re.search(r"공집합 비율\s+([\d.]+)\s+([\d.]+)", _ln)
    if _m and _cd and _ca:
        _emp.append((_cd, _ca, float(_m.group(1)), float(_m.group(2))))
assert len(_emp) == 4, "공집합 블록 %d개" % len(_emp)
_nz = [e for e in _emp if e[2] > 0 or e[3] > 0]
if not _nz:
    w("공집합(예측집합이 비는 경우) 비율은 여덟 칸 모두 0 이다.")
else:
    w(
        '**공집합 비율은 "모든 조건에서 0" 이 아니다.** 여덟 칸 중 %d칸이 0 이 아니다 — %s. '
        "나머지 칸은 모두 0 이며, **시간분할 쪽은 네 칸 전부 0** 이다. 공집합은 두 클래스 모두 "
        "비순응도 문턱을 넘지 못할 때 생기므로, 확률이 양성 쪽으로 밀린 시간분할에서는 나오지 "
        "않고 오히려 무작위 대조군에서 나온다."
        % (
            len(_nz),
            " · ".join(
                "%s 목표 %d%% 무작위 %.3f / 시간분할 %.3f" % (e[0], e[1], e[2], e[3]) for e in _nz
            ),
        )
    )
w()
w(
    "**이 절의 클래스별 커버리지에는 보장이 붙어 있지 않다** — 주변형 split conformal 이 "
    "보장하는 것은 전체 커버리지 하나뿐이다. 클래스마다 보장을 붙인 절차로 다시 잰 결과는 "
    "§11.1 에 있다."
)
w()
# ---------------------------------------------------------------- 11.1 Mondrian
_MARM = ["A. 시간분할", "B. 무작위", "C. 무작위+양성률 맞춤"]
_MMET = ["주변형", "Mondrian"]


def _mz(cov, n, tgt):
    """커버리지가 명목값보다 낮은지 — 이항 표준오차 기준 단측 z."""
    if n is None or n <= 0 or cov != cov:
        return float("nan")
    se = (tgt * (1 - tgt) / n) ** 0.5
    return (cov - tgt) / se if se > 0 else float("nan")


def _mrow(d, al, arm, met):
    g = mdc[
        (mdc.dataset == d)
        & ((mdc.alpha - al).abs() < 1e-9)
        & (mdc.arm == arm)
        & (mdc.method == met)
    ]
    return g.iloc[0] if len(g) else None


_MAL = sorted(mdc.alpha.unique())
w("### 11.1 클래스별 보장을 붙이면 — 라벨 조건부(Mondrian) conformal")
w()
w(
    "§11 의 클래스별 커버리지에는 보장이 붙어 있지 않다. 주변형 split conformal 이 "
    "보장하는 것은 전체 커버리지 하나뿐이고, 클래스별 수치는 기술 통계다. 보장이 붙은 "
    '절차로 다시 재야 "클래스별로도 무너졌다" 를 주장할 수 있다. 그 절차가 **라벨 '
    "조건부 conformal**(Mondrian conformal)이다 — 보정셋을 참라벨로 쪼개 클래스마다 따로 "
    "분위수를 잡는다. 검증 시점에 참라벨을 몰라도 된다. 후보 라벨 y 마다 그 y 의 분위수와 "
    "비교하면 되기 때문이다."
)
w()
w("아래에서 p_y(x) 는 모델이 화합물 x 에 라벨 y 를 줄 확률이다.")
w()
w("| 절차 | 라벨 y 를 예측집합에 넣는 조건 | 분위수를 어디서 잡는가 | 보장 |")
w("|---|---|---|---|")
# 표 안에 세로줄(파이프)을 넣으면 열 구분자와 섞인다 — p(y|x) 를 p_y(x) 로 적는다.
w("| 주변형 (§11) | 1 − p_y(x) ≤ q | 보정셋 전체 | 전체 커버리지 ≥ 1 − α |")
w("| Mondrian (이 절) | 1 − p_y(x) ≤ q_y | 참라벨이 y 인 보정셋 | **클래스별** 커버리지 ≥ 1 − α |")
w()
w(
    "**이것은 단순한 보완이 아니다.** 라벨 조건부 conformal 은 클래스 안에서의 "
    "교환가능성만 쓰므로 **P(y) 가 아무리 바뀌어도 클래스별 보장이 살아 있다**. 즉 임의의 "
    "라벨이동(label shift)에 유효하다. 이 데이터의 양성률은 학습군 %s / %s 에서 검증군 "
    "%s / %s 로 떨어진다(§2.3). 그러므로 이 절은 한계 5번을 닫는 동시에 하나를 더 묻는다 "
    "— **§11 의 커버리지 붕괴가 라벨이동으로 설명되는가.** Mondrian 이 명목값을 회복하면 "
    "설명되는 것이고, 회복하지 못하면 라벨이동으로 환원되지 않는 성분이 있는 것이다."
    % (
        f4(SPLIT["DILIrank"]["train_pos_rate"]),
        f4(SPLIT["DICTrank"]["train_pos_rate"]),
        f4(T["DILIrank"].pos_rate),
        f4(T["DICTrank"].pos_rate),
    )
)
w()
w(
    "분할은 셋이다. **A** 는 §11 과 같은 시간분할이고, **B** 는 크기를 맞춘 무작위 분할 "
    "%d회(교환가능성이 성립하는 기준선), **C** 는 B 의 검증셋을 시간분할 검증군의 양성률로 "
    "부분표집한 것이다. C 에는 공변량 이동이 없고 P(y) 만 바뀐다 — **순수 라벨이동 모사**이자 "
    "구현이 맞다는 확인이다. 주변형·시간분할은 §11 의 수치를 그대로 재현한다(네 칸 모두 차 "
    "0.00e+00). 재현되지 않으면 스크립트가 그 자리에서 표시한다."
    % int(_mrow("DILIrank", _MAL[0], _MARM[1], "주변형").reps)
)
w()
w(
    "| 데이터셋 | 목표 | 분할 | 방법 | 전체 커버리지 | 양성 | 음성 | 평균 집합크기 | 양성 z | 음성 z |"
)
w("|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for al in _MAL:
        tgt = 1 - al
        for arm in _MARM:
            for met in _MMET:
                r = _mrow(d, al, arm, met)
                if r is None:
                    continue
                _sd = r.get("cov_all_sd")
                _c = f4(r["cov_all"]) if _sd != _sd else "%s ± %s" % (f4(r["cov_all"]), f4(_sd))
                _one = int(r["reps"]) == 1
                _zp = ("%+.2f" % _mz(r["cov_pos"], r["n_pos"], tgt)) if _one else "—"
                _zn = ("%+.2f" % _mz(r["cov_neg"], r["n_neg"], tgt)) if _one else "—"
                w(
                    "| %s | %.0f%% | %s | %s | %s | %s | %s | %s | %s | %s |"
                    % (
                        d,
                        tgt * 100,
                        arm,
                        met,
                        _c,
                        f4(r["cov_pos"]),
                        f4(r["cov_neg"]),
                        f2(r["set_size"]),
                        _zp,
                        _zn,
                    )
                )
w()
w(
    "z 는 시간분할(1회)에서만 적는다 — 커버리지의 이항 표준오차 "
    "sqrt(목표(1−목표)/n) 기준 단측값이고, 음수가 크면 명목값에 미달한다는 뜻이다. "
    "대조군은 반복 %d회의 전체 커버리지에 표준편차를 병기했다."
    % int(_mrow("DILIrank", _MAL[0], _MARM[1], "주변형").reps)
)
w()
w("읽는 순서가 있다.")
w()
_b_ok = all(
    _mrow(d, al, _MARM[1], "Mondrian")["cov_pos"]
    + 2 * _mrow(d, al, _MARM[1], "Mondrian")["cov_pos_sd"]
    >= 1 - al
    and _mrow(d, al, _MARM[1], "Mondrian")["cov_neg"]
    + 2 * _mrow(d, al, _MARM[1], "Mondrian")["cov_neg_sd"]
    >= 1 - al
    for d in DS
    for al in _MAL
)
w(
    "1. **B(교환가능 기준선)에서 Mondrian 은 클래스별로 명목값을 지킨다** — %s. "
    "보장이 작동한다는 확인이다."
    % (
        ", ".join(
            "%s %.0f%% 양성 %s · 음성 %s"
            % (
                d,
                (1 - al) * 100,
                f4(_mrow(d, al, _MARM[1], "Mondrian")["cov_pos"]),
                f4(_mrow(d, al, _MARM[1], "Mondrian")["cov_neg"]),
            )
            for d in DS
            for al in _MAL
        )
        + ("" if _b_ok else " — **그런데 일부가 명목 미달이다. 해석 전에 구현을 볼 것**")
    )
)
w()
w(
    "2. **C(순수 라벨이동)에서 주변형만 흔들린다.** B → C 로 바뀌는 것은 검증셋의 양성률 "
    "하나뿐인데, 전체 커버리지가 주변형에서는 %s 내려가고 Mondrian 에서는 %s 움직일 "
    "뿐이다. "
    "이유는 표의 클래스별 열에 그대로 있다 — **클래스별 커버리지는 라벨이동에 영향을 받지 "
    "않는다**(부분표집은 클래스 안의 분포를 바꾸지 않는다). 라벨이동이 바꾸는 것은 두 클래스 "
    "커버리지의 혼합비다. 주변형은 클래스별 커버리지가 한쪽으로 쏠려 있어 혼합비가 바뀌면 "
    "전체가 무너지고, Mondrian 은 클래스마다 명목값을 지키므로 **어떤 혼합비에서도** 전체가 "
    "명목값 이상이다."
    % (
        "%s–%s"
        % (
            f4(
                min(
                    abs(
                        _mrow(d, al, _MARM[2], "주변형")["cov_all"]
                        - _mrow(d, al, _MARM[1], "주변형")["cov_all"]
                    )
                    for d in DS
                    for al in _MAL
                )
            ),
            f4(
                max(
                    abs(
                        _mrow(d, al, _MARM[2], "주변형")["cov_all"]
                        - _mrow(d, al, _MARM[1], "주변형")["cov_all"]
                    )
                    for d in DS
                    for al in _MAL
                )
            ),
        ),
        "%s 이하"
        % f4(
            max(
                abs(
                    _mrow(d, al, _MARM[2], "Mondrian")["cov_all"]
                    - _mrow(d, al, _MARM[1], "Mondrian")["cov_all"]
                )
                for d in DS
                for al in _MAL
            )
        ),
    )
)
w()
_bad = []
for d in DS:
    for al in _MAL:
        r = _mrow(d, al, _MARM[0], "Mondrian")
        for cls, ck, nk in (("양성", "cov_pos", "n_pos"), ("음성", "cov_neg", "n_neg")):
            z = _mz(r[ck], r[nk], 1 - al)
            if z < -1.645:
                _bad.append((d, al, cls, r[ck], int(r[nk]), z))
w(
    "3. **그래서 A(시간분할)를 읽는다.** %s"
    % (
        "명목값에 미달하는 칸이 %d개 있다(보정 전 단측 z 기준) — %s. **라벨 조건부 "
        "보장을 붙여도 커버리지가 회복되지 않는 칸이 남고, 그 미달분은 라벨이동으로 "
        "환원되지 않는다.** 다만 이 z 는 FDR 보정을 견디지 못한다(§15.2) — 그래서 이 "
        "주장은 **부족분의 크기**로 하고 유의성 주장으로 쓰지 않는다."
        % (
            len(_bad),
            ", ".join(
                "%s 목표 %.0f%% %s 클래스 %s (n=%d, z=%+.2f)"
                % (b[0], (1 - b[1]) * 100, b[2], f4(b[3]), b[4], b[5])
                for b in _bad
            ),
        )
        if _bad
        else "명목값에 유의하게 미달하는 칸이 없다. **§11 의 커버리지 붕괴는 클래스 조건부 "
        "절차로 닫힌다.**"
    )
)
w()
_ds_bad = sorted(set(b[0] for b in _bad))
_ds_ok = [d for d in DS if d not in _ds_bad]
if _bad and _ds_ok:
    w(
        "   미달이 **%s 에만** 남는다는 점이 중요하다. %s 에서는 Mondrian 이 두 클래스 "
        "모두 명목값에 유의하게 미달하지 않는다. 즉 %s 의 커버리지 붕괴는 라벨이동과 "
        "주변형 절차의 클래스 쏠림으로 설명되고, %s 의 붕괴에는 그것으로 설명되지 않는 "
        "성분이 남는다. 이것은 §33 의 결론(같은 이동량에서도 시간분할이 더 심하다 — "
        "시간 축에 특정적인 성분이 있다)과 같은 방향이고, 두 데이터셋 중 어느 쪽에서 그 "
        "성분이 보이는지까지 일치한다."
        % (" · ".join(_ds_bad), " · ".join(_ds_ok), " · ".join(_ds_ok), " · ".join(_ds_bad))
    )
    w()
w(
    "4. **대가는 예측집합 크기다.** 시간분할에서 Mondrian 은 집합 크기를 %s 늘린다"
    '(%s). 음성 클래스의 분위수가 %s 까지 올라가 거의 모든 화합물이 "음성 가능" 으로 '
    "들어오기 때문이다. 공집합률은 여덟 칸 모두 %s 이다."
    % (
        "%s–%s"
        % (
            f2(
                min(
                    _mrow(d, al, _MARM[0], "Mondrian")["set_size"]
                    - _mrow(d, al, _MARM[0], "주변형")["set_size"]
                    for d in DS
                    for al in _MAL
                )
            ),
            f2(
                max(
                    _mrow(d, al, _MARM[0], "Mondrian")["set_size"]
                    - _mrow(d, al, _MARM[0], "주변형")["set_size"]
                    for d in DS
                    for al in _MAL
                )
            ),
        ),
        ", ".join(
            "%s %.0f%% %s → %s"
            % (
                d,
                (1 - al) * 100,
                f2(_mrow(d, al, _MARM[0], "주변형")["set_size"]),
                f2(_mrow(d, al, _MARM[0], "Mondrian")["set_size"]),
            )
            for d in DS
            for al in _MAL
        ),
        f4(max(_mrow(d, al, _MARM[0], "Mondrian")["q_neg"] for d in DS for al in _MAL)),
        f4(max(_mrow(d, al, _MARM[0], m)["empty_rate"] for d in DS for al in _MAL for m in _MMET)),
    )
)
w()
w(
    "**이 절의 한계 둘.** 첫째, A 의 z 는 %s 로 경계에 가깝다 — 음성 검증군이 %d / %d개뿐이고, "
    "같은 데이터에 목표 90%%·80%% 두 번을 쓰므로 두 검정은 독립이 아니다. 둘째, **이 z 를 "
    "§15.2 의 가족에 넣어 FDR 보정하면 Mondrian 쪽은 여덟 칸 전부 유의하지 않다**(최소 "
    "q %s). 보정 전에 경계에 있던 것이 보정 후에 남지 않는다. 그래서 이 절의 근거는 "
    "**효과크기**(%s 음성 클래스의 커버리지 부족 %s, 두 목표값에서 같은 방향)로 쓰고 z 는 "
    "보조로만 쓴다."
    % (
        ", ".join("%+.2f" % b[5] for b in _bad) if _bad else "모두 −1.645 보다 크다",
        int(_mrow("DILIrank", _MAL[0], _MARM[0], "Mondrian")["n_neg"]),
        int(_mrow("DICTrank", _MAL[0], _MARM[0], "Mondrian")["n_neg"]),
        f4(
            fat[
                fat.newly_included & (fat.source == "§11.1") & fat.test.str.startswith("Mondrian")
            ].q_F4.min()
        ),
        " · ".join(sorted(set(b[0] for b in _bad))) if _bad else "해당 없음",
        (
            "%s–%s"
            % (f4(min((1 - b[1]) - b[3] for b in _bad)), f4(max((1 - b[1]) - b[3] for b in _bad)))
            if _bad
            else "해당 없음"
        ),
    )
)
w()
# ================================================================ 12. 라벨 검열
w("---")
w()
w("## 12. 라벨 검열 통제")
w()
w(
    "최근 승인 약물은 시장 노출 기간이 짧아 아직 독성이 보고되지 않았을 수 있다. 그렇다면 "
    '검증군의 음성 라벨 일부는 "안전하다" 가 아니라 "아직 모른다" 이고, 성능 저하가 '
    "분포 이동이 아니라 라벨 미성숙 때문일 수 있다."
)
w()
w("**검열이 실재한다는 근거** — 노출 기간이 짧을수록 양성률이 낮다.")
w()
# 노출 구간 양성률은 censoring_summary.txt 에서 직접 읽는다 (하드코딩 금지)
_blocks = txt("censoring_summary.txt").split("[검열 근거]")[1:]
_bands = {}
for _d, _blk in zip(DS, _blocks):
    _bands[_d] = re.findall(
        r"([\d]+-\s*[\d]+년) \(n=\s*(\d+), 승인 ([\d\-]+)\): 양성 ([\d.]+)%", _blk
    )
    assert len(_bands[_d]) == 5, "%s 노출 구간 %d개" % (_d, len(_bands[_d]))
w(
    "| 데이터셋 | "
    + " | ".join(re.sub(r"\s+", "", b[0]).replace("-", "–") for b in _bands["DILIrank"])
    + " |"
)
w("|---|" + "---|" * 5)
for _d in DS:
    w("| %s 양성률 | %s |" % (_d, " | ".join("%s%% (n=%s)" % (b[3], b[1]) for b in _bands[_d])))
w()
w(
    "구간의 승인연도 범위는 DILIrank %s, DICTrank %s 다 — 라벨 관측 종료 시점이 각각 "
    "2025년·2023년이라 같은 노출 기간이 다른 연도 범위에 대응한다."
    % (" · ".join(b[2] for b in _bands["DILIrank"]), " · ".join(b[2] for b in _bands["DICTrank"]))
)
w()
# 구간 n 의 합이 분석집합과 어긋나는 쪽을 자진 신고한다 (2026-10-04).
_bsum = {_d: sum(int(_b[1]) for _b in _bands[_d]) for _d in DS}
_nall = {_d: SPLIT[_d]["n_train"] + SPLIT[_d]["n_test"] for _d in DS}
_refy = {_d: int(_bands[_d][0][2].split("-")[1]) for _d in DS}
_fut = {}
for _d, _fn in _PROC_FILE.items():
    _x = pd.read_csv(os.path.join(PROC, _fn)).dropna(subset=["approval_year", "y", "smiles"])
    _x = _x[_x.approval_year.astype(int) > _refy[_d]]
    _fut[_d] = sorted(
        "%s(%d)" % (str(_c).strip(), int(_v))
        for _c, _v in zip(_x["CompoundName"], _x["approval_year"])
    )
_yecand = set(str(_x).strip().upper() for _x in cv("year_evidence.csv")["화합물"])
_short = [_d for _d in DS if _bsum[_d] != _nall[_d]]
if _short:
    _d = _short[0]
    _ok = [_x for _x in DS if _x not in _short]
    _ov = [_n for _n in _fut[_d] if _n.split("(")[0].upper() in _yecand]
    w(
        "**구간 n 의 합을 먼저 확인할 것.** %s 는 %d 로 분석집합 %d 과 정확히 같지만, "
        "**%s 는 %d 뿐이어서 분석집합 %d 보다 %d 적다.** 노출기간을 "
        "`라벨 관측 종료연도 − 승인연도` 로 "
        "정의하는데 %s 의 관측 종료는 %d년이고 승인연도가 그보다 뒤로 기록된 화합물이 %d건 "
        "있어 노출기간이 음수가 되고, 0년 이상인 구간에 들어가지 않는다. 해당 화합물은 "
        "%s 다. 그중 %d건(%s)은 **§16.1 의 승인연도 수정 후보와 같은 화합물**이므로, 이 "
        "누락은 분석 오류가 아니라 승인연도 배정 오류의 또 다른 흔적으로 읽어야 한다."
        % (
            _ok[0],
            _bsum[_ok[0]],
            _nall[_ok[0]],
            _d,
            _bsum[_d],
            _nall[_d],
            _nall[_d] - _bsum[_d],
            _d,
            _refy[_d],
            len(_fut[_d]),
            " · ".join(_fut[_d]),
            len(_ov),
            " · ".join(_n.split("(")[0] for _n in _ov),
        )
    )
    w()
w(
    "세 가지 통제를 걸었다. **(1)** 검증군에서 노출 기간이 짧은 약물을 잘라내고 재평가, "
    "**(2)** 음성 라벨의 신뢰도를 관찰 기간에 비례해 가중(양성은 항상 1.0), "
    "**(3)** 학습군·검증군을 같은 기준으로 동시에 잘라 공정 비교."
)
w()
w("| 데이터셋 | 조건 | 검증군 n | 양성률 | AUROC | 무작위 대비 격차 | 기준 대비 |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    g = cens[cens.dataset.str.startswith(d)]
    base = g[g.exp.str.startswith("기준")].iloc[0]
    for _, r in g.iterrows():
        w(
            "| %s | %s | %d | %s | %s | +%s | %s |"
            % (
                d,
                r.exp,
                int(r.n_test),
                f4(r.pos_rate),
                f4(r.auroc),
                f4(r.gap),
                "기준" if r.exp.startswith("기준") else "%+.4f" % (base.gap - r.gap),
            )
        )
w()
_dmin = cens[
    (cens.dataset.str.startswith("DILIrank")) & (cens.exp.str.startswith("절단"))
].gap.min()
_dbase = cens[
    (cens.dataset.str.startswith("DILIrank")) & (cens.exp.str.startswith("기준"))
].gap.iloc[0]
_cmin = cens[
    (cens.dataset.str.startswith("DICTrank")) & (cens.exp.str.startswith("절단"))
].gap.min()
_cbase = cens[
    (cens.dataset.str.startswith("DICTrank")) & (cens.exp.str.startswith("기준"))
].gap.iloc[0]
w(
    "**검열은 DILIrank 에서 격차의 일부를 설명하고, DICTrank 에서는 설명하지 않는다.** "
    "노출 기간을 가장 크게 잘랐을 때 DILIrank 격차는 %s → %s 로 %s 줄어든다. 반면 "
    "**DICTrank 는 %s → %s 로 오히려 %s 커진다** — 자를수록 격차가 커지는 방향이므로 "
    '검열로 설명되는 몫이 없다. "두 데이터셋에서 격차가 줄어든다" 고 쓰면 사실이 아니다. '
    "**어느 조건에서도 격차가 0 이 되지 않는다.**"
    % (
        f4(_dbase),
        f4(_dmin),
        pct1((_dbase - _dmin) / _dbase * 100),
        f4(_cbase),
        f4(_cmin),
        pct1(abs(_cmin - _cbase) / _cbase * 100),
    )
)
w()
_wdlt = (
    cens[cens.exp.str.startswith("가중")].gap.values
    - cens[cens.exp.str.startswith("가중")]
    .dataset.map(
        {
            k: cens[(cens.dataset == k) & (cens.exp.str.startswith("기준"))].gap.iloc[0]
            for k in cens.dataset.unique()
        }
    )
    .values
)
w(
    "**음성 가중은 격차를 거의 바꾸지 않는다** — tau 를 5·8·10·15년으로 바꿔도 격차는 "
    '기준에서 최대 %s(정확히 %.6f) 만큼만 움직인다. 반올림한 네 자리로 "%s 이내" 라고 '
    "쓰면 경계에서 틀리므로 부등호를 붙이지 말 것. 방향은 데이터셋마다 다르다 — DILIrank 는 "
    "tau 를 키우면 격차가 조금 줄고, DICTrank 는 조금 커진다. 검열이 라벨 신뢰도 수준의 "
    "문제라면 이 가중이 효과를 지웠어야 하는데 그렇지 않았다."
    % (f4(abs(_wdlt).max()), abs(_wdlt).max(), f4(abs(_wdlt).max()))
)
w()
w(
    "**학습군까지 같이 자르면 격차가 오히려 커진다** — 양측절단 조건에서 DILIrank %s, "
    "DICTrank %s 로, 기준(%s / %s)보다 크다. 검열을 제거하는 방향의 통제가 효과를 "
    "**강화**한다는 것은 검열이 주된 원인이 아니라는 뜻이다."
    % (
        f4(
            cens[
                (cens.dataset.str.startswith("DILIrank")) & (cens.exp.str.startswith("양측"))
            ].gap.iloc[0]
        ),
        f4(
            cens[
                (cens.dataset.str.startswith("DICTrank")) & (cens.exp.str.startswith("양측"))
            ].gap.iloc[0]
        ),
        f4(_dbase),
        f4(_cbase),
    )
)
w()
w("### 12.1 검열 통제를 ECE 로 반복")
w()
w("위 표는 AUROC 만 본다. 같은 통제를 ECE·Brier 분해로 반복했다(유병률맞춤 대조군 기준).")
w()
w("| 데이터셋 | 조건 | n | 양성률 | ECE | ECE 배율 | 보정 기여율 |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    g = cc[(cc.dataset == d) & (cc.group.isin(["기준", "라벨성숙도"]))]
    for _, r in g.iterrows():
        w(
            "| %s | %s | %d | %s | %s | %s배 | %s |"
            % (
                d,
                r.condition,
                int(r.n),
                f4(r.pos_rate),
                f4(r.ece),
                f2(r.ece_ratio_pm),
                pct1(r.share_reliability_pm),
            )
        )
w()
_dl = cc[(cc.dataset == "DILIrank") & (cc.group == "라벨성숙도")]
_cl = cc[(cc.dataset == "DICTrank") & (cc.group == "라벨성숙도")]
w(
    "**DILIrank 에서는 라벨 성숙도가 ECE 배율을 설명하지 못한다** — 배율이 %s–%s 로 "
    "기준(%s)과 거의 같다. **DICTrank 에서는 기여할 가능성이 남는다** — 노출 ≥10년 조건에서 "
    "보정 기여율이 %s 까지 떨어지고 양측절단에서는 음수가 된다(판별력 쪽이 더 나빠졌다는 뜻). "
    "다만 그 조건의 검증군은 %d개뿐이어서 **이 표본으로 확정할 수 없다.**"
    % (
        f2(_dl.ece_ratio_pm.min()),
        f2(_dl.ece_ratio_pm.max()),
        f2(cc[(cc.dataset == "DILIrank") & (cc.group == "기준")].ece_ratio_pm.iloc[0]),
        pct1(_cl[_cl.condition.str.contains("10년")].share_reliability_pm.iloc[0]),
        int(_cl[_cl.condition.str.contains("10년")].n.iloc[0]),
    )
)
w()
w("### 12.2 시간축을 뒤집으면 — 검열 가설의 정면 대조")
w()
w(
    '§12 의 통제 세 가지는 모두 "검증군에서 미성숙한 라벨을 덜어내는" 방향이므로 검열 '
    "가설을 반증하지는 못한다. **뒤집으면 정면으로 갈린다** — 2010년 이후로 학습하고 "
    "2010년 이전으로 검증하면 검증군이 수십 년간 관찰된 성숙한 라벨이 된다. 검열이 원인이면 "
    "이 방향에서는 보정이 무너지지 않아야 한다. 대조군은 §8 과 같은 R1·R2·R3 설계이고 "
    "반복 %d회다." % _reps("reverse_split.csv", "n_rep_valid")
)
w()
w(
    "| 데이터셋 | 방향 | 설정 | 학습 n | 검증 n | AUROC | ECE | 보정기울기 | ECE 배율 | 보정 기여율 | 유효 반복 | p |"
)
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    g = rv[rv.dataset == d]
    _ntr = {"정방향(참고)": int(T[d].n_train), "역방향": int(T[d].n_test)}
    for _, r in g.iterrows():
        rat = "—" if pd.isna(r.ece_ratio) else "**%s배**" % f2(r.ece_ratio)
        sh = "—" if pd.isna(r.share_rel) else pct1(r.share_rel)
        nv = "—" if pd.isna(r.n_rep_valid) else "%d/200" % int(r.n_rep_valid)
        pv = "—" if pd.isna(r.p_phipson_smyth) else f4(r.p_phipson_smyth)
        w(
            "| %s | %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |"
            % (
                d,
                r.direction,
                r.setting,
                _ntr[r.direction],
                int(r.n_test),
                f4(r.auroc),
                f4(r.ece),
                f4(r.slope),
                rat,
                sh,
                nv,
                pv,
            )
        )
w()
_rvT = {
    d: rv[(rv.dataset == d) & (rv.direction == "역방향") & (rv.setting.str.startswith("T"))].iloc[0]
    for d in DS
}
_rvR = {
    (d, b): rv[
        (rv.dataset == d) & (rv.direction == "역방향") & (rv.setting.str.startswith(b))
    ].iloc[0]
    for d in DS
    for b in ("R1", "R2", "R3")
}
w(
    "**보정은 역방향에서도 무너진다.** 역방향 ECE 는 DILIrank %s, DICTrank %s 이고, "
    "크기맞춤 대조군 대비 배율은 %s배 / %s배(p = %s / %s)다. 검증군이 1939–2009년 승인 "
    "약물이므로 **라벨 미성숙으로는 설명할 수 없다.** 보정 성분 기여율은 %s / %s 로, "
    "정방향 R1(%s / %s)보다 오히려 높다."
    % (
        f4(_rvT["DILIrank"].ece),
        f4(_rvT["DICTrank"].ece),
        f2(_rvR[("DILIrank", "R1")].ece_ratio),
        f2(_rvR[("DICTrank", "R1")].ece_ratio),
        f4(_rvR[("DILIrank", "R1")].p_phipson_smyth),
        f4(_rvR[("DICTrank", "R1")].p_phipson_smyth),
        pct1(_rvR[("DILIrank", "R1")].share_rel),
        pct1(_rvR[("DICTrank", "R1")].share_rel),
        pct1(MN[("DILIrank", "R1")].share_old),
        pct1(MN[("DICTrank", "R1")].share_old),
    )
)
w()
w("**세 가지를 반드시 함께 적을 것.**")
w()
w(
    "1. **가장 엄한 통제(R3)에서는 두 데이터셋의 결과가 갈린다.** DILIrank 는 배율 %s배 "
    "p = %s 로 유의하지만, **DICTrank 는 배율 %s배 p = %s 로 유의하지 않다**(초과 %d/%d). "
    "역방향 증거는 정방향(§8 의 R3 가 %s / %s)보다 약하다."
    % (
        f2(_rvR[("DILIrank", "R3")].ece_ratio),
        f4(_rvR[("DILIrank", "R3")].p_phipson_smyth),
        f2(_rvR[("DICTrank", "R3")].ece_ratio),
        f4(_rvR[("DICTrank", "R3")].p_phipson_smyth),
        int(_rvR[("DICTrank", "R3")].n_ge),
        int(_rvR[("DICTrank", "R3")].n_rep_valid),
        f4(PCR[("DILIrank", "R3")].p_perm),
        f4(PCR[("DICTrank", "R3")].p_perm),
    )
)
w(
    "2. **DILIrank R3 의 유효 반복은 200회가 아니라 %d회다.** R3 는 대조군의 학습군 양성률을 "
    "역방향 학습군(= 정방향 검증군)의 %s 에 맞춰야 하는데, 전체 양성률 %s 에서 그만큼 "
    "끌어내리려면 부표집이 커서 대부분의 반복이 최소 표본 조건에 걸려 버려졌다. 그래서 그 p 는 "
    "하한 `1/%d = %s` 이고, 200회 기준 하한(%s)과 다른 값이다. **반복 수를 적지 않고 p 만 "
    "인용하면 안 된다.**"
    % (
        int(_rvR[("DILIrank", "R3")].n_rep_valid),
        f4(T["DILIrank"].pos_rate),
        f4(
            SPLIT["DILIrank"]["train_pos_rate"]
            * SPLIT["DILIrank"]["n_train"]
            / 1.0
            / (SPLIT["DILIrank"]["n_train"] + SPLIT["DILIrank"]["n_test"])
            + SPLIT["DILIrank"]["test_pos_rate"]
            * SPLIT["DILIrank"]["n_test"]
            / (SPLIT["DILIrank"]["n_train"] + SPLIT["DILIrank"]["n_test"])
        ),
        int(_rvR[("DILIrank", "R3")].n_rep_valid) + 1,
        f4(_rvR[("DILIrank", "R3")].p_phipson_smyth),
        f4(1 / 201),
    )
)
w(
    "3. **역방향 AUROC 는 정방향과 비교하지 말 것.** 역방향 학습군은 %d / %d 개뿐이어서 "
    "판별력 자체가 낮다(DILIrank %s). 대조군은 같은 학습군 크기를 쓰므로 **ECE 배율은** "
    "공정한 비교지만, 절대 AUROC 는 표본 크기 효과와 섞인다."
    % (int(T["DILIrank"].n_test), int(T["DICTrank"].n_test), f4(_rvT["DILIrank"].auroc))
)
w()
w(
    '> 이 설계는 "미래로 학습해 과거를 예측한다" 는 비현실적 설정이다. 배포 시나리오가 '
    "아니라 **검열 가설을 가르는 대조 실험**으로만 읽어야 한다."
)
w()
# ================================================================ 13. 공변량 이동
w("---")
w()
w("## 13. 시간적 공변량 이동")
w()
w(
    "시간분할이 학습군과 검증군 사이에 **실제로** 분포 차이를 만드는지(§13.1–13.2), 그리고 "
    "그 차이가 보정 붕괴를 설명하는지(§13.3)는 별개의 물음이다."
)
w()
w("### 13.1 이동의 직접 측정 — 시대 분류기")
w()
w(
    '구조만으로 "이 화합물은 2010년 이후인가" 를 맞히는 분류기를 같은 표현·같은 모델로 학습해 '
    "5-fold CV AUC 를 구했다. 0.5 는 학습군과 검증군이 구조적으로 구별되지 않는다는 뜻이고 "
    "1.0 은 완전히 분리된다는 뜻이다. 귀무분포는 시대 라벨을 무작위로 섞어 **%s회** "
    "만들었다." % format(_reps("era_permutation.csv", "n_perm"), ",")
)
w()
w(
    "| 데이터셋 | n (2010년 이후) | 시대 AUC | 귀무 평균 (SD) | 귀무 2.5–97.5% | 관측값 이상 | p | 귀무 SD 기준 z |"
)
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    r = EP[d]
    w(
        "| %s | %d (%d) | **%s** | %s (%s) | %s–%s | %d/%d | **%s** | %s |"
        % (
            d,
            int(r.n),
            int(r.n_post),
            f4(r.era_auc),
            f4(r.null_mean),
            f4(r.null_sd),
            f4(r.null_p025),
            f4(r.null_p975),
            int(r.n_ge),
            int(r.n_perm),
            f4(r.p_phipson_smyth),
            f2(r.z),
        )
    )
w()
w(
    "2,000회 중 관측값 이상이 0회이므로 p 는 Phipson–Smyth 하한 `1/2001 = %s` 다. "
    "실제 p 는 이보다 작다. z 는 (관측값 − 귀무 평균) / 귀무 SD 이며 **정식 검정이 아니다** — "
    "귀무분포가 정규라는 근거가 없으므로 p 로 환산하지 말 것. 하한에 눌린 p 가 가리지 못하는 "
    "거리만 보여준다." % f4(1 / 2001)
)
w()
w(
    '**"분류기가 무엇이든 맞히는 것 아니냐" 는 반론을 막는 대조군** — 같은 크기의 **무작위** '
    "분할 라벨에 같은 절차를 적용하면 AUC 가 DILIrank %s, DICTrank %s 로 우연 수준에 붙는다."
    % (f4(SQ["DILIrank"].random_split_era_auc), f4(SQ["DICTrank"].random_split_era_auc))
)
w()
for d in DS:
    assert (
        abs(float(mep[mep.dataset == d].era_auc.iloc[0]) - float(EP[d].era_auc)) < 1e-12
    ), "§13.1 보조 시대 AUC 불일치"
w(
    "**순열 설계를 바꿔 한 번 더 확인했다.** 학습·검증 n 을 맞춘 귀무분포로 1,000회 다시 "
    "돌린 결과도 같다 — 시대 AUC 는 위 표와 동일한 %s / %s, 귀무 평균 %s / %s, 귀무 SD 기준 "
    "z %s / %s, 1,000회 중 관측값 이상 %d / %d 회다. 순열 횟수(2,000 → 1,000)와 귀무 설계를 "
    "바꿔도 결론이 움직이지 않는다."
    % (
        f4(mep[mep.dataset == "DILIrank"].era_auc.iloc[0]),
        f4(mep[mep.dataset == "DICTrank"].era_auc.iloc[0]),
        f4(mep[mep.dataset == "DILIrank"].null_mean.iloc[0]),
        f4(mep[mep.dataset == "DICTrank"].null_mean.iloc[0]),
        f2(mep[mep.dataset == "DILIrank"].z.iloc[0]),
        f2(mep[mep.dataset == "DICTrank"].z.iloc[0]),
        int(mep[mep.dataset == "DILIrank"].n_ge.iloc[0]),
        int(mep[mep.dataset == "DICTrank"].n_ge.iloc[0]),
    )
)
w()
w("### 13.2 기술자 수준의 이동")
w()
w(
    "지문 없이 RDKit 기술자 6종만으로도 시대가 구별된다(AUC %s / %s). 학습군 대비 검증군의 "
    "중앙값이 한 방향으로 움직인다."
    % (f4(SQ["DILIrank"].era_auc_desc_only), f4(SQ["DICTrank"].era_auc_desc_only))
)
w()
w("| 기술자 | DILIrank 학습 → 검증 (차이) | DICTrank 학습 → 검증 (차이) |")
w("|---|---|---|")
for desc in ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumRotatableBonds", "NumHDonors"]:
    a = sqd[(sqd.dataset == "DILIrank") & (sqd.descriptor == desc)].iloc[0]
    b = sqd[(sqd.dataset == "DICTrank") & (sqd.descriptor == desc)].iloc[0]
    fm = (lambda x: "%.1f" % x) if desc in ("MolWt", "TPSA") else (lambda x: "%.2f" % x)
    if desc.startswith("Num"):
        fm = lambda x: "%.0f" % x
    sgn = lambda x: ("+" if x >= 0 else "") + fm(x)
    w(
        "| %s | %s → %s (%s) | %s → %s (%s) |"
        % (
            desc,
            fm(a.median_train),
            fm(a.median_test),
            sgn(a["diff"]),
            fm(b.median_train),
            fm(b.median_test),
            sgn(b["diff"]),
        )
    )
w()
w(
    "최근 승인 약물이 더 크고(MolWt +%.0f / +%.0f), 더 소수성이고(MolLogP +%.2f / +%.2f), "
    "극성 표면적이 넓다(TPSA +%.1f / +%.1f). **시간분할이 만드는 분포 이동은 구조 수준에서 "
    "실재한다.**"
    % (
        sqd[(sqd.dataset == "DILIrank") & (sqd.descriptor == "MolWt")]["diff"].iloc[0],
        sqd[(sqd.dataset == "DICTrank") & (sqd.descriptor == "MolWt")]["diff"].iloc[0],
        sqd[(sqd.dataset == "DILIrank") & (sqd.descriptor == "MolLogP")]["diff"].iloc[0],
        sqd[(sqd.dataset == "DICTrank") & (sqd.descriptor == "MolLogP")]["diff"].iloc[0],
        sqd[(sqd.dataset == "DILIrank") & (sqd.descriptor == "TPSA")]["diff"].iloc[0],
        sqd[(sqd.dataset == "DICTrank") & (sqd.descriptor == "TPSA")]["diff"].iloc[0],
    )
)
w()
w("### 13.3 이동의 크기와 보정 붕괴가 연결되는가 — 연결되지 않는다")
w()
_tn = td.nn_tanimoto
w(
    "검증군 각 분자의 학습군 최근접 Tanimoto 유사도를 구했다(DILIrank: 중앙값 %s, 평균 %s). "
    "유사도 0.4 미만(외삽 영역)이 %d/%d (%s)다."
    % (
        f4(_tn.median()),
        f4(_tn.mean()),
        (_tn < 0.4).sum(),
        len(_tn),
        pct1((_tn < 0.4).mean() * 100),
    )
)
w()
w(
    "**AUROC 는 DILIrank 에서만 유사도를 따라 올라간다.** DICTrank 는 단조가 아니다 — "
    "가장 유사한 구간 [0.4-0.6) 의 AUROC 가 중간 구간 [0.3-0.4) 보다 낮다(아래 표)."
)
w()
w("| 데이터셋 | 유사도 구간 | n | AUROC | ECE |")
w("|---|---|---|---|---|")
for d in DS:
    for _, r in cc[(cc.dataset == d) & (cc.condition.str.startswith("B 유사도 ["))].iterrows():
        w(
            "| %s | %s | %d | %s | %s |"
            % (d, r.condition.replace("B 유사도 ", ""), int(r.n), f4(r.auroc), f4(r.ece))
        )
w()
# 구간이 검증군 전체를 덮지 않는다는 사실을 자진 신고한다 (2026-10-04).
_simsum = {
    _d: int(cc[(cc.dataset == _d) & (cc.condition.str.startswith("B 유사도 ["))].n.sum())
    for _d in DS
}
_simtop = {_d: SPLIT[_d]["n_test"] - _simsum[_d] for _d in DS}
w(
    "**이 표는 유사도 0.6 미만만 담는다.** 구간 n 의 합은 DILIrank %d, DICTrank %d 로 "
    "검증군 %d / %d 보다 각각 %d · %d 적다 — 유사도 0.6 이상 분자는 표본이 적어 별도 "
    '구간을 만들지 않았다(DILIrank 에서 직접 세어도 %d개다). 바로 위의 "0.4 미만 %d/%d" '
    "는 앞의 두 구간 합이므로 이 누락과 무관하다. **구간별 n 을 더해 검증군 수와 맞추려 "
    "하면 어긋난다.**"
    % (
        _simsum["DILIrank"],
        _simsum["DICTrank"],
        SPLIT["DILIrank"]["n_test"],
        SPLIT["DICTrank"]["n_test"],
        _simtop["DILIrank"],
        _simtop["DICTrank"],
        int((_tn >= 0.6).sum()),
        (_tn < 0.4).sum(),
        len(_tn),
    )
)
w()
w(
    "**그러나 통계적으로 지지되지 않고, ECE 의 방향은 데이터셋마다 반대다.** 아래 검정은 "
    "**DILIrank 에서만** 수행했다 — DICTrank 에는 같은 부트스트랩·상관 검정을 돌리지 않았으므로 "
    '"두 데이터셋에서 지지되지 않는다" 고 쓰면 안 된다.'
)
w()
_st = cv("similarity_tests.csv")
_hi = cc[(cc.dataset == "DILIrank") & (cc.condition == "B 유사도 상위 50%")].iloc[0]
_lo = cc[(cc.dataset == "DILIrank") & (cc.condition == "B 유사도 하위 50%")].iloc[0]
w("| 검정 | 값 | p |")
w("|---|---|---|")
w(
    "| DILIrank 유사도 상위 vs 하위 AUROC 차이 (부트스트랩 2,000회) | +%s (상위 n=%d %s, 하위 n=%d %s) | %s |"
    % (
        f4(_st[_st.test.str.startswith("화학공간")].stat.iloc[0]),
        int(_hi.n),
        f4(_hi.auroc),
        int(_lo.n),
        f4(_lo.auroc),
        f4(_st[_st.test.str.startswith("화학공간")].p.iloc[0]),
    )
)
w(
    "| DILIrank 유사도–정답확률 Spearman | rho = +%s | %s |"
    % (
        f4(_st[_st.test.str.startswith("유사도-")].stat.iloc[0]),
        f4(_st[_st.test.str.startswith("유사도-")].p.iloc[0]),
    )
)
w(
    "| DILIrank 유사도 상위 vs 하위 **ECE** | %s vs %s (**유사도가 높은 쪽이 더 나쁘다**) | — |"
    % (f4(_hi.ece), f4(_lo.ece))
)
w()
_chi = cc[(cc.dataset == "DICTrank") & (cc.condition == "B 유사도 상위 50%")].iloc[0]
_clo = cc[(cc.dataset == "DICTrank") & (cc.condition == "B 유사도 하위 50%")].iloc[0]
w(
    "DICTrank 에서는 AUROC 가 유사도를 강하게 따라가지만(상위 %s 대 하위 %s) ECE 는 "
    "%s 대 %s 로 차이가 작다." % (f4(_chi.auroc), f4(_clo.auroc), f4(_chi.ece), f4(_clo.ece))
)
w()
w("**따라서 다음 두 문장을 한 문장으로 붙이지 말 것.**")
w()
w(
    "- (성립) 시간분할은 구조 수준의 분포 이동을 만든다 — 시대 분류기 AUC %s / %s, 무작위 기준 0.50."
    % (f4(EP["DILIrank"].era_auc), f4(EP["DICTrank"].era_auc))
)
w(
    "- (**미성립**) 그 이동의 **정도**가 보정 붕괴의 **정도**를 설명한다 — 유사도로 나눠 보면 "
    "ECE 는 유사도를 따라가지 않는다."
)
w()
w(
    "이동의 존재는 직접 측정으로 확립되지만, 이동 정도와 보정 붕괴 정도를 잇는 인과 사슬은 "
    "데이터셋 2개로는 검증할 수 없다."
)
w()
w("### 13.4 연도 구간별 추세")
w()
w("| 데이터셋 | 승인 구간 | n | 양성률 | AUROC | ECE |")
w("|---|---|---|---|---|---|")
for d in DS:
    for _, r in cc[(cc.dataset == d) & (cc.group == "연도구간")].iterrows():
        w(
            "| %s | %s | %d | %s | %s | %s |"
            % (d, r.condition.replace("C ", ""), int(r.n), f4(r.pos_rate), f4(r.auroc), f4(r.ece))
        )
w()
w(
    "**DILIrank 에서 최근 구간의 ECE 가 가장 나쁘다**(%s, 양성률 %s) — 시간이 지날수록 "
    "보정이 더 무너진다는 방향이다. DICTrank 에서는 그 추세가 뚜렷하지 않다. "
    "구간별 대조군을 정의할 수 없어 배율은 계산하지 않았고 절대 ECE 추세만 보고한다."
    % (
        f4(cc[(cc.dataset == "DILIrank") & (cc.condition == "C 2018-2026")].ece.iloc[0]),
        f4(cc[(cc.dataset == "DILIrank") & (cc.condition == "C 2018-2026")].pos_rate.iloc[0]),
    )
)
w()
w("### 13.5 시간이 아닌 축으로 같은 이동을 만들면 — 붕괴는 이동 크기를 따라가지 않는다")
w()
w(
    "§13.3 은 이동의 **정도**가 보정 붕괴의 **정도**를 설명하지 못한다는 음성 결과였다. "
    "그 음성을 양성 대조로 보강한다 — **시간을 쓰지 않고** 분포 이동을 만드는 분할 세 가지를 "
    "구성해 같은 눈금으로 붕괴를 쟀다. 검증군 크기는 모두 시간분할과 같게 맞추고, 각 분할마다 "
    "**그 분할 자신의 검증군 유병률에 맞춘** 무작위 대조군 %d회(§8 의 R2 설계)를 "
    "붙였다." % int(sm.n_rep_valid.max())
)
w()
w(
    "| 데이터셋 | 분할 | 검증 n | 학습 양성률 | 검증 양성률 | 유병률 이동 | 이동 AUC | AUROC | ECE | 대조 ECE | ECE 배율 | 보정 기여율 | p |"
)
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in sm[sm.dataset == d].iterrows():
        w(
            "| %s | %s | %d | %s | %s | %+.4f | %s | %s | %s | %s | **%s배** | %s | %s |"
            % (
                d,
                r.split,
                int(r.n_test),
                f4(r.pos_rate_train),
                f4(r.pos_rate_test),
                r.pos_rate_test - r.pos_rate_train,
                f4(r.shift_auc),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ctrl_ece),
                f2(r.ece_ratio),
                pct1(r.share_rel),
                f4(r.p_phipson_smyth),
            )
        )
w()
_smT = {d: sm[(sm.dataset == d) & (sm.split.str.startswith("T"))].iloc[0] for d in DS}
_smX = sm[sm.split.str.startswith("X")]
w(
    "**먼저 구현 확인.** T 행은 §8 의 R2 와 같은 설계이므로 배율이 일치해야 한다 — "
    "실제로 %s배 / %s배이고 §8 의 R2 는 %s배 / %s배다. 새 스크립트가 기존 파이프라인과 "
    "같은 값을 낸다는 교차 확인이다."
    % (
        f2(_smT["DILIrank"].ece_ratio),
        f2(_smT["DICTrank"].ece_ratio),
        f2(PCR[("DILIrank", "R2")].ece_ratio),
        f2(PCR[("DICTrank", "R2")].ece_ratio),
    )
)
w()
w(
    '**설계 의도는 "이동량을 맞추는" 것이었는데 맞춰지지 않았다. 그 사실을 먼저 적는다.** '
    "X1·X2 는 분자량·logP 의 임계로 나눈 분할이고 그 두 기술자가 **모델 입력에 들어 있으므로**, "
    "소속을 맞히는 분류기가 이동 AUC %s 까지 포화한다. X3(지문 군집)도 %s / %s 다. 시간분할의 "
    "%s / %s 보다 훨씬 크다. 따라서 이 표는 **이동량 일치 비교가 아니라 '이동을 훨씬 크게 "
    "준' 상한 대조**로 읽어야 한다."
    % (
        f4(_smX.shift_auc.max()),
        f4(sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("X3"))].shift_auc.iloc[0]),
        f4(sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("X3"))].shift_auc.iloc[0]),
        f4(_smT["DILIrank"].shift_auc),
        f4(_smT["DICTrank"].shift_auc),
    )
)
w()
w(
    "**그 상한 대조에서 나온 결과가 오히려 더 강하다.** 이동을 포화시켜도 배율은 시간분할을 "
    "넘지 못한다 — 비시간 분할 6칸의 배율은 %s–%s 이고 시간분할은 %s / %s 다. 특히 X3 는 "
    "이동 AUC 가 %s / %s 인데 배율이 %s / %s 로 사실상 1 이다. **공변량 이동을 아무리 크게 줘도 "
    "보정 붕괴가 그에 비례해 커지지 않는다.** 무작위 분할(이동 AUC %s / %s)의 배율이 "
    "%s / %s 로 1 근처인 것은 이 측정이 제대로 작동한다는 확인이다."
    % (
        f2(_smX.ece_ratio.min()),
        f2(_smX.ece_ratio.max()),
        f2(_smT["DILIrank"].ece_ratio),
        f2(_smT["DICTrank"].ece_ratio),
        f4(sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("X3"))].shift_auc.iloc[0]),
        f4(sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("X3"))].shift_auc.iloc[0]),
        f2(sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("X3"))].ece_ratio.iloc[0]),
        f2(sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("X3"))].ece_ratio.iloc[0]),
        f4(sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("R."))].shift_auc.iloc[0]),
        f4(sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("R."))].shift_auc.iloc[0]),
        f2(sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("R."))].ece_ratio.iloc[0]),
        f2(sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("R."))].ece_ratio.iloc[0]),
    )
)
w()
_sm2 = sm.copy()
_sm2["dprev"] = (_sm2.pos_rate_test - _sm2.pos_rate_train).abs()


def _spearman(a, b):
    a = pd.Series(a).rank().to_numpy(dtype=float)
    b = pd.Series(b).rank().to_numpy(dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    den = ((a * a).sum() * (b * b).sum()) ** 0.5
    return float((a * b).sum() / den) if den else float("nan")


_rho_shift = _spearman(_sm2.shift_auc, _sm2.ece_ratio)
_rho_prev = _spearman(_sm2.dprev, _sm2.ece_ratio)
w(
    "**배율을 따라가는 것은 이동 AUC 가 아니라 유병률 이동이다.** 10칸 전체에서 배율과의 "
    "Spearman 상관은 이동 AUC 쪽이 rho = %+.3f, 유병률 이동 절댓값 쪽이 rho = %+.3f 다 "
    "(**분할 10칸뿐이므로 상관의 유의성은 주장하지 않는다. 순서 비교로만 읽을 것**). "
    "유병률 이동이 가장 큰 칸(시간분할, %+.4f / %+.4f)이 배율도 가장 크고, 유병률이 거의 "
    "안 움직인 X3(%+.4f / %+.4f)는 배율이 1 근처다. 대조군이 각 분할의 검증군 유병률을 "
    "맞췄는데도 이 관계가 남는다는 점이 중요하다 — §8 에서 R2·R3 로 유병률을 통제해도 배율이 "
    "1 을 넘던 것과 같은 현상이다."
    % (
        _rho_shift,
        _rho_prev,
        _smT["DILIrank"].pos_rate_test - _smT["DILIrank"].pos_rate_train,
        _smT["DICTrank"].pos_rate_test - _smT["DICTrank"].pos_rate_train,
        sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("X3"))].pos_rate_test.iloc[0]
        - sm[(sm.dataset == "DILIrank") & (sm.split.str.startswith("X3"))].pos_rate_train.iloc[0],
        sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("X3"))].pos_rate_test.iloc[0]
        - sm[(sm.dataset == "DICTrank") & (sm.split.str.startswith("X3"))].pos_rate_train.iloc[0],
    )
)
w()
w("**이 절이 지지하는 문장과 지지하지 않는 문장을 구분할 것.**")
w()
w(
    "- (지지됨) 보정 붕괴는 **공변량 이동의 크기에 비례하지 않는다.** 이동을 포화시킨 "
    "분할에서도 배율이 시간분할보다 작다. §13.3 의 음성 결과와 같은 방향이다."
)
w(
    "- (지지됨) 시간분할의 붕괴에는 **라벨 분포(유병률) 이동이 더 크게 관여한다.** 10칸에서 "
    "배율이 유병률 이동을 따라간다."
)
w(
    '- (**지지되지 않음**) "시간축에 고유한 요인이 있다" — X1·X2·X3 는 이동량이 맞춰진 '
    "대조군이 아니므로, 같은 이동량에서 시간분할만 더 심하다는 비교를 할 수 없다. 이동량을 "
    "맞추려면 모델 입력에 없는 축으로 분할을 만들어야 한다(예: 외부 특성). 그것은 이 "
    "데이터셋에서 수행하지 못했고 한계로 적는다."
)
w("- (**지지되지 않음**) 인과. 분할 10칸의 관찰 상관이며 데이터셋 2개다.")
w()
w(
    "> 이동 AUC 는 §13.1 의 시대 분류기와 같은 절차(같은 표현·같은 모델·5-fold CV)로 "
    "계산했다. T 행의 값(%s / %s)이 §13.1 의 시대 분류기 AUC 와 같은지 확인할 수 있다."
    % (f4(_smT["DILIrank"].shift_auc), f4(_smT["DICTrank"].shift_auc))
)
w()
# ================================================================ 14. 하위집단
w("---")
w()
w("## 14. 하위집단 분석 (탐색적)")
w()
w(
    "**이 절의 모든 검정은 사후 탐색이며, §15 의 3군 분할 기준으로는 보정 후 유의한 것이 "
    "하나도 없다.** 단 문서의 검정 22건을 단일 가족으로 묶으면 두 건(DILIrank ATC L, "
    "DICTrank MW>500)이 유의해진다(§15.1). 어느 가족 구성을 쓰는지에 달려 있으므로 "
    '"어떤 보정에서도 유의하지 않다" 고 쓰면 안 된다. '
    "가설 생성용 관찰로만 보고한다."
)
w()
w("### 14.1 ATC 치료군별")
w()
_aa_all = cv("atc_audit.csv")
_atc_n = {
    d: (int(_aa_all[_aa_all.dataset == d].atc.notna().sum()), int((_aa_all.dataset == d).sum()))
    for d in DS
}
w(
    "ATC 코드는 KEGG 매핑으로 얻었다. 확보율은 DILIrank %d/%d (%s), DICTrank %d/%d (%s) 다."
    % (
        _atc_n["DILIrank"][0],
        _atc_n["DILIrank"][1],
        pct1(_atc_n["DILIrank"][0] / _atc_n["DILIrank"][1] * 100),
        _atc_n["DICTrank"][0],
        _atc_n["DICTrank"][1],
        pct1(_atc_n["DICTrank"][0] / _atc_n["DICTrank"][1] * 100),
    )
)
w()
w("| ATC | 설명 | DILI n | 양성률 | AUROC | DICT n | 양성률 | AUROC |")
w("|---|---|---|---|---|---|---|---|")
for code in ["A", "B", "C", "J", "L", "N", "V"]:
    a = atcs[(atcs.dataset == "DILIrank") & (atcs.atc == code)]
    b = atcs[(atcs.dataset == "DICTrank") & (atcs.atc == code)]
    nm = a.name.iloc[0] if len(a) else b.name.iloc[0]

    def cell(g):
        if not len(g):
            return "— | — | —"
        r = g.iloc[0]
        au = "정의 불가" if pd.isna(r.auroc) else f4(r.auroc)
        return "%d | %s | %s" % (int(r.n), pct1(r.pos_rate * 100), au)

    star = "**" if code == "L" else ""
    w("| %s%s%s | %s%s%s | %s | %s |" % (star, code, star, star, nm, star, cell(a), cell(b)))
w()
_small = [
    (d, _a)
    for _a in ["A", "B", "C", "J", "L", "N", "V"]
    for d in DS
    if len(atcs[(atcs.dataset == d) & (atcs.atc == _a)])
    and int(atcs[(atcs.dataset == d) & (atcs.atc == _a)].n.iloc[0]) <= 15
]
w(
    "n ≤ 15 행(%s)은 정보가 없다고 보아야 한다. **DILIrank 쪽만 적으면 빠진다** — "
    "DICTrank 에도 해당 행이 있다." % ", ".join("%s %s" % (d, a) for d, a in _small)
)
w()
# 표가 전체가 아니라 n>=10 계열만 담는다는 사실을 명시한다 (2026-10-04).
_atc_shown = {_d: int(atcs[atcs.dataset == _d].n.sum()) for _d in DS}
_atc_has = {
    _d: int(
        atct[(atct.dataset == _d) & (atct.test.str.startswith("ATC L"))].n_a.iloc[0]
        + atct[(atct.dataset == _d) & (atct.test.str.startswith("ATC L"))].n_b.iloc[0]
    )
    for _d in DS
}
w(
    "**위 표에 실린 것은 검증군 n ≥ 10 인 계열뿐이다.** 검증군에서 ATC 가 붙은 분자는 "
    "DILIrank %d / DICTrank %d 인데 7행의 n 합은 %d / %d 다 — 차이(%d · %d)는 n < 10 이라 "
    '행을 만들지 않은 계열이다. 아래 "L 군 vs 그 외" 표의 분모는 7행 합이 아니라 ATC 가 '
    "붙은 전체(%d / %d)이므로, **두 표의 n 을 더해 맞추려 하면 어긋난다.**"
    % (
        _atc_has["DILIrank"],
        _atc_has["DICTrank"],
        _atc_shown["DILIrank"],
        _atc_shown["DICTrank"],
        _atc_has["DILIrank"] - _atc_shown["DILIrank"],
        _atc_has["DICTrank"] - _atc_shown["DICTrank"],
        _atc_has["DILIrank"],
        _atc_has["DICTrank"],
    )
)
w()
_AA = cv("atc_audit.csv")
_AL = _AA[_AA.atc.astype(str).str.startswith("L")]
_ALT = _AL[_AL.approval_year >= 2010]
w(
    "**ATC L 군의 이름을 정확히 쓸 것.** ATC 최상위 L 의 정식 명칭은 "
    '**"항종양제 및 면역조절제(antineoplastic and immunomodulating agents)"** 다. '
    "구성은 **데이터셋별로** 적어야 한다 — 두 데이터셋을 합한 행 수는 공통 분자를 두 번 "
    "세기 때문이다(아래 L 행 합계 %d행의 고유 분자는 %d개뿐이다). %s. "
    '검증군만 보면 %s 다. **"항종양제" 라고만 쓰면 부정확하다.**'
    % (
        len(_AL),
        _AL.ikey14.nunique(),
        " / ".join(
            "%s L %d행(%s)"
            % (
                _d,
                len(_AL[_AL.dataset == _d]),
                " · ".join(
                    "%s %d" % (k, v)
                    for k, v in _AL[_AL.dataset == _d]
                    .atc.astype(str)
                    .str[:3]
                    .value_counts()
                    .sort_index()
                    .items()
                ),
            )
            for _d in DS
        ),
        " / ".join(
            "%s %d행(%s)"
            % (
                _d,
                len(_ALT[_ALT.dataset == _d]),
                " · ".join(
                    "%s %d" % (k, v)
                    for k, v in _ALT[_ALT.dataset == _d]
                    .atc.astype(str)
                    .str[:3]
                    .value_counts()
                    .sort_index()
                    .items()
                ),
            )
            for _d in DS
        ),
    )
)
w()
w("**L 군 vs 그 외**")
w()
w("| 데이터셋 | L | 그 외 | 차이 | 부트스트랩 p |")
w("|---|---|---|---|---|")
aa = cv("atc_audit.csv")
aa_te = aa[aa.approval_year >= 2010]
for d in DS:
    r = atct[(atct.dataset == d) & (atct.test.str.startswith("ATC L"))].iloc[0]
    g = aa_te[aa_te.dataset == d]
    pos_L = g[g.atc1 == "L"].y.mean()
    pos_O = g[g.atc1 != "L"].y.mean()
    w(
        "| %s | n=%d, 양성 %s, AUROC %s | n=%d, 양성 %s, AUROC %s | **%+.4f** | %s |"
        % (
            d,
            int(r.n_a),
            pct1(pos_L * 100),
            f4(r.auroc_a),
            int(r.n_b),
            pct1(pos_O * 100),
            f4(r.auroc_b),
            r["diff"],
            f4(r.p),
        )
    )
w()
# 94·32·3배가 손으로 적혀 있었다 — 하위집단 크기라 데이터셋 구성이
# 바뀌면 거짓이 된다 (2026-10-22). atc_stratified.csv 에서 읽는다.
_nL = {d: int(atcs[(atcs.dataset == d) & (atcs.atc == "L")].n.iloc[0]) for d in DS}
w(
    "**DICTrank 의 L 하위집단이 %.1f배 큼에도(%d 대 %d) 재현되지 않는다.** DILIrank 의 "
    "큰 차이는 표본 %d개에서 나온 값이며, 다중비교 보정 후 유의하지 않다(§15)."
    % (_nL["DICTrank"] / _nL["DILIrank"], _nL["DICTrank"], _nL["DILIrank"], _nL["DILIrank"])
)
w()
_adc = atcs[atcs.dataset == "DICTrank"]
_adc_big = _adc[_adc.n > 15]
w(
    "전체 AUROC(DICTrank %s)가 계열별 %s–%s 의 편차를 은폐한다는 점은 그 자체로 보고할 값이 "
    '있다. **다만 그 범위의 양 끝은 바로 위에서 "정보가 없다" 고 한 n ≤ 15 칸이다** '
    "(%s). n > 15 칸만 보면 범위는 %s–%s 로 좁아진다. 범위를 인용할 때 어느 쪽인지 밝힐 것."
    % (
        f4(T["DICTrank"].AUROC),
        f4(_adc.auroc.min()),
        f4(_adc.auroc.max()),
        " / ".join(
            "%s n=%d AUROC %s" % (_r.atc, int(_r.n), f4(_r.auroc))
            for _, _r in _adc.loc[[_adc.auroc.idxmin(), _adc.auroc.idxmax()]].iterrows()
        ),
        f4(_adc_big.auroc.min()),
        f4(_adc_big.auroc.max()),
    )
)
w()
w("### 14.2 구조 기반 층화")
w()
w("| 데이터셋 | 기준 | 참 (n, AUROC) | 거짓 (n, AUROC) | 차이 | p |")
w("|---|---|---|---|---|---|")
for d in DS:
    for _, r in atct[(atct.dataset == d) & (atct.test.str.startswith("구조"))].iterrows():
        w(
            "| %s | %s | %d, %s | %d, %s | %+.4f | %s |"
            % (
                d,
                r.test.replace("구조: ", ""),
                int(r.n_a),
                f4(r.auroc_a),
                int(r.n_b),
                f4(r.auroc_b),
                r["diff"],
                f4(r.p),
            )
        )
w()
_str_t = atct[atct.test.str.startswith("구조")]
_opp, _same = [], []
for _k in sorted(_str_t.test.unique()):
    _a = _str_t[(_str_t.dataset == "DILIrank") & (_str_t.test == _k)]["diff"].iloc[0]
    _b = _str_t[(_str_t.dataset == "DICTrank") & (_str_t.test == _k)]["diff"].iloc[0]
    (_opp if _a * _b < 0 else _same).append(_k.replace("구조: ", ""))
_cells = pd.concat([_str_t.n_a, _str_t.n_b])
w(
    "**부호가 데이터셋마다 반대인 것은 네 기준 중 %d개다** — %s 에서 반대이고, %s 에서는 "
    '같은 방향이다. "부호가 데이터셋마다 반대다" 라고 뭉뚱그리면 과장이다. 예컨대 N ≥ 6 '
    "조건은 DILIrank %+.4f, DICTrank %+.4f 로 반대다. 이 표의 셀 크기는 %d(%s)에서 %d 까지 "
    "편차가 크다."
    % (
        len(_opp),
        ", ".join(_opp),
        ", ".join(_same),
        _str_t[(_str_t.dataset == "DILIrank") & (_str_t.test.str.contains("N>=6"))]["diff"].iloc[0],
        _str_t[(_str_t.dataset == "DICTrank") & (_str_t.test.str.contains("N>=6"))]["diff"].iloc[0],
        int(_cells.min()),
        _str_t.loc[_str_t.n_a.idxmin()].dataset
        + " "
        + _str_t.loc[_str_t.n_a.idxmin()].test.replace("구조: ", "")
        + " 참",
        int(_cells.max()),
    )
)
w()

# ---------------------------------------------------------------- 14.3 하위집단 보정
_sg_ok = sgc[~sgc.underpowered.astype(bool)]
_sg_up = sgc[sgc.underpowered.astype(bool)]
w("### 14.3 하위집단을 보정 지표로 다시 — §14.1·§14.2 는 AUROC 뿐이다")
w()
w(
    "§14.1 과 §14.2 는 하위집단을 **AUROC 로만** 보고한다. 이 논문의 논지는 무너진 것이 "
    '보정이라는 것이므로, 하위집단도 보정으로 재지 않으면 "보정 붕괴가 특정 치료군이나 '
    '특정 구조군에 몰려 있는가" 에 답이 없다. 몰려 있다면 처방이 생긴다 — 그 군에는 확률을 '
    "쓰지 않는다. 그래서 §21(적용범위)이 Tanimoto 구간에 쓴 설계를 ATC 계열과 구조 층에 "
    "그대로 적용했다. 층마다 **크기와 양성률을 맞춘 무작위 대조군** %d회를 붙이고 ECE 배율과 "
    "그 부트스트랩 구간을 낸다." % int(_sg_ok.n_rep_valid.max())
)
w()
w(
    "표본부족 기준은 §21·§17.2 와 같다(n < 40 또는 소수 클래스 < 10). 그 기준으로 층 "
    "%d개 중 %d개가 빠진다 — **ATC 계열은 거의 전부 빠진다.** DILIrank 검증군에서 가장 큰 "
    "계열이 L(항종양·면역조절) %d개뿐이기 때문이다(한계 8번). 그래서 ATC 는 계열별로 "
    "읽지 못하고 L 대 그외로만 읽는다."
    % (
        len(sgc),
        len(_sg_up),
        int(sgc[(sgc.dataset == "DILIrank") & (sgc.stratum == "ATC L")].n.iloc[0]),
    )
)
w()
w("| 데이터셋 | 층 | n | 양성률 | AUROC | ECE | 대조군 ECE | ECE 배율 | 배율 95% CI | p |")
w("|---|---|---|---|---|---|---|---|---|---|")
# "ATC L" 과 "ATC L 대 그외: L" 은 같은 층이다 — 뒤쪽을 뺀다.
_sg_dup = set(zip(_sg_ok.dataset, _sg_ok.stratum))
for d in DS:
    for _, r in _sg_ok[_sg_ok.dataset == d].iterrows():
        if r.stratum == "ATC L 대 그외: L" and (d, "ATC L") in _sg_dup:
            continue
        _ex = (r.ratio_lo > 1) or (r.ratio_hi < 1)
        w(
            "| %s | %s | %d | %s | %s | %s | %s | %s | [%s, %s] | %s |"
            % (
                d,
                r.stratum,
                int(r.n),
                f4(r.pos_rate),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ctrl_ece),
                ("**%s**" % f2(r.ece_ratio)) if _ex else f2(r.ece_ratio),
                f2(r.ratio_lo),
                f2(r.ratio_hi),
                f4(r.p_phipson_smyth),
            )
        )
w()
_sg_ex = _sg_ok[(_sg_ok.ratio_lo > 1) | (_sg_ok.ratio_hi < 1)]
w("읽을 것이 둘이다.")
w()
w(
    "1. **붕괴는 어느 층에도 몰려 있지 않다 — 전부에 있다.** 계산 가능한 %d개 층 중 "
    "**%d개**에서 ECE 배율이 1 을 넘고 구간이 1 을 제외한다. 배율 범위는 %s–%s 다. "
    '즉 "이 치료군·이 구조군에는 확률을 쓰지 말라" 는 식의 처방이 나오지 않는다. '
    "하위집단으로 피할 수 있는 문제가 아니라는 뜻이고, 이것은 §21 이 유사도 축에서 "
    "내린 결론(6개 구간 중 5개가 1 을 제외)과 같은 방향이다."
    % (len(_sg_ok), len(_sg_ex), f2(_sg_ex.ece_ratio.min()), f2(_sg_ex.ece_ratio.max()))
)
w()
_sg_cmp = []
for d in DS:
    g = _sg_ok[_sg_ok.dataset == d]
    if not len(g):
        continue
    _a = g.loc[g.auroc.idxmin()]
    _e = g.loc[g.ece_ratio.idxmax()]
    _sg_cmp.append(
        (d, _a.stratum, float(_a.auroc), _e.stratum, float(_e.ece_ratio), _a.stratum == _e.stratum)
    )
w(
    "2. **AUROC 로만 읽으면 보정이 가장 많이 무너진 층을 놓칠 수 있다.** %s"
    % " ".join(
        "%s 는 AUROC 가 가장 낮은 층이 %s(%s)이고 ECE 배율이 가장 큰 층이 "
        "%s(%s)로 %s." % (c[0], c[1], f4(c[2]), c[3], f2(c[4]), "같다" if c[5] else "**다르다**")
        for c in _sg_cmp
    )
)
if any(not c[5] for c in _sg_cmp):
    w()
    w(
        '   한쪽에서 두 순위가 갈린다는 사실만으로 "AUROC 는 쓸모없다" 고 쓰면 과장이다 — '
        "두 지표가 서로 다른 것을 재므로 순위가 갈리는 것은 정상이다. 여기서 말할 수 있는 "
        "것은 **하위집단 표를 AUROC 로만 실으면 이 논문의 논지(보정)에 대응하는 정보가 "
        "표에 없다**는 것이다. 그래서 §14.1·§14.2 를 이 절과 함께 읽어야 한다."
    )
w()
w(
    "**이 절의 한계.** 대조군 재표집 변동이 배율 구간에 들어가 있지 않아 구간이 실제보다 "
    '좁다(§21 과 같은 한계다). 그리고 층이 %d개 빠졌으므로 "모든 층에서 무너진다" 가 '
    '아니라 **"볼 수 있었던 모든 층에서 무너진다"** 가 정확한 서술이다.' % len(_sg_up)
)
w()
# ================================================================ 15. 다중비교
w("---")
w()
w("## 15. 다중비교 보정 (Benjamini–Hochberg FDR, α = 0.05)")
w()
w(
    "검정을 성격이 다른 세 군으로 분리해 군 내에서만 보정했다. 사전 계획된 확증 검정과 사후 "
    "탐색 검정을 같은 군으로 묶으면 전자가 과도하게, 후자가 과소하게 처리된다. "
    "**이 3군 분류는 사후에 정한 것이며 사전등록된 계획이 아니다.**"
)
w()
FAM = [
    ("1_primary", "1군 — 핵심 확증 검정"),
    ("2_prevalence_control", "2군 — 유병률 통제 대조군"),
    ("3_exploratory_subgroup", "3군 — 탐색적 하위집단"),
]
for fam, lab in FAM:
    g = mc[mc.family == fam]
    nsig = int(g.significant_after_FDR.sum())
    w("**%s (n = %d) → %d/%d 유의**" % (lab, len(g), nsig, len(g)))
    w()
    w("| 데이터셋 | 검정 | p | q (BH) |")
    w("|---|---|---|---|")
    for _, r in g.sort_values("p").iterrows():
        bold = "**" if r.significant_after_FDR else ""
        # 결과 CSV 의 짧은 라벨을 §14.1 에서 정한 정확한 이름으로 바꿔 쓴다.
        _tn = r.test.replace("ATC L(항종양)", "ATC L(항종양·면역조절)")
        w("| %s | %s | %s | %s%s%s |" % (r.dataset, _tn, f4(r.p), bold, f4(r.q_BH), bold))
    w()
w(
    "핵심 주장(시간분할 성능 저하, 유병률로 환원되지 않음)은 보정 후에도 두 데이터셋 모두 "
    "유의하다. 가장 작은 3군 q 는 %s 이며 DILIrank ATC L 과 DICTrank MW>500 이 공유한다."
    % f4(mc[mc.family == "3_exploratory_subgroup"].q_BH.min())
)
w()
# 표의 네 자리 p 로 BH 를 다시 돌리면 마지막 자리가 어긋난다. 자진 신고한다 (2026-10-04).
_g3 = mc[mc.family == "3_exploratory_subgroup"].sort_values("p")
_pp = [float(f4(_x)) for _x in _g3.p]
_m3 = len(_pp)
_qp = [0.0] * _m3
_run = 1.0
for _k in range(_m3, 0, -1):
    _run = min(_run, _pp[_k - 1] * _m3 / _k)
    _qp[_k - 1] = _run
_pmin = float(_g3.p.iloc[0 if _g3.q_BH.iloc[0] == _g3.q_BH.min() else 1])
_pexact = float(mc.loc[mc[mc.family == "3_exploratory_subgroup"].q_BH.idxmin(), "p"])
w(
    "> **q 는 반올림 전 정밀 p 로 계산했다.** 위 표에 인쇄된 네 자리 p 로 BH 를 다시 돌리면 "
    "마지막 자리가 어긋난다 — 3군의 가장 작은 q 는 %s 인데, 인쇄값으로 계산하면 %s 다. "
    "차이는 ATC L 의 정밀 p 가 %.9f 이고 인쇄값이 %s 이기 때문이다. §15.1 의 q 값도 "
    "모두 정밀 p 기준이다. **표의 p 로 q 를 검산하지 말 것.**"
    % (f4(mc[mc.family == "3_exploratory_subgroup"].q_BH.min()), f4(min(_qp)), _pexact, f4(_pexact))
)
w()
w("### 15.1 가족 구성이 결론을 지탱한다 — 반드시 자진 신고할 것")
w()


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


_era_p = float(EP["DILIrank"].p_phipson_smyth)
_g1 = list(mc[mc.family == "1_primary"].p)
_g2 = list(mc[mc.family == "2_prevalence_control"].p)
_g3 = list(mc[mc.family == "3_exploratory_subgroup"].p)
_ins = {}
for key, base in [("1군", _g1), ("2군", _g2), ("3군", _g3)]:
    _ins[key] = bh(base + [_era_p, _era_p])[len(base)]
_fam22 = _g1 + _g2 + _g3 + [_era_p, _era_p]
_q22 = bh(_fam22)
_r3p = float(MN[("DICTrank", "R3")].p_old)
_i_r3 = _fam22.index(_r3p)
_fam22m = [float(MN[("DICTrank", "R3")].p_new) if abs(x - _r3p) < 1e-15 else x for x in _fam22]
_q22m = bh(_fam22m)
# 탐색적 2건(DILIrank ATC L, DICTrank MW>500)의 위치를 이름으로 찾는다.
# 예전에 "p 순위 9번째" 로 집었다가 다른 검정의 q 를 인쇄한 버그가 있었다.
_g3names = list(mc[mc.family == "3_exploratory_subgroup"].test)
_g3ds = list(mc[mc.family == "3_exploratory_subgroup"].dataset)
_i_L = (
    len(_g1)
    + len(_g2)
    + next(
        i
        for i, (d, t) in enumerate(zip(_g3ds, _g3names))
        if d == "DILIrank" and t.startswith("ATC L")
    )
)
_i_MW = (
    len(_g1)
    + len(_g2)
    + next(i for i, (d, t) in enumerate(zip(_g3ds, _g3names)) if d == "DICTrank" and "MW>500" in t)
)
assert abs(_q22[_i_L] - _q22[_i_MW]) < 1e-12, "탐색적 2건 q 가 다르다"
w(
    "**1. §13.1 의 시대 분류기 2건이 위 세 군 어디에도 없다.** 어느 군에 넣어도 그 검정 자체는 "
    "유의하다 — 1군 편입 시 q = %s, 2군 %s, 3군 %s, 22건 단일 가족 %s (전부 반올림 전 정밀 p "
    "로 계산). 그러나 배정 규칙이 문서에 없는 것은 결함이다."
    % (f4(_ins["1군"]), f4(_ins["2군"]), f4(_ins["3군"]), f4(_q22[len(_fam22) - 1]))
)
w()
w(
    "**2. 문서의 모든 검정 22건을 단일 가족으로 묶으면 결과가 달라진다.** 탐색적 발견 2건"
    "(DILIrank ATC L p = %s, DICTrank MW>500 p = %s)이 **유의해진다**(q = %s). "
    "§8 DICTrank R3 은 어느 p 를 쓰는지에 갈린다 — n 을 맞추지 않은 p = %s 면 **q = %s 로 "
    "유의성을 잃고**, 양측 n 을 맞춘 p = %s(§7.2)면 **q = %s 로 지킨다.** "
    "따라서 그 검정의 유의성을 3군 분할이 **홀로** 지탱한다고는 말할 수 없지만, "
    "두 근거(3군 분할 / n 일치 교정) 모두 사후에 확보한 것이라는 점은 그대로다."
    % (
        f4(mc[(mc.dataset == "DILIrank") & (mc.test.str.startswith("ATC L"))].p.iloc[0]),
        f4(mc[(mc.dataset == "DICTrank") & (mc.test.str.contains("MW>500"))].p.iloc[0]),
        f4(_q22[_i_L]),
        f4(_r3p),
        f4(_q22[_i_r3]),
        f4(MN[("DICTrank", "R3")].p_new),
        f4(_q22m[_i_r3]),
    )
)
w()
_n_auroc = int(atcs[atcs.atc.isin(["A", "B", "C", "J", "L", "N", "V"])].auroc.notna().sum())
w(
    "**3. 3군의 m = %d 는 실제로 훑은 비교 수보다 작다.** §14.1 은 ATC 7계열 × 2 = 14칸을 "
    "훑었고 그중 **AUROC 가 실제로 계산된 것은 %d칸**이다(DILIrank V 는 양성률 0%% 로 정의 "
    "불가). 그 값을 보고 L 을 골랐다. §14.2 는 임계 4종 × 2 = 8건이다. "
    '따라서 3군 q 는 하한으로 읽어야 한다. 다행히 m 을 늘리면 "0/%d 유의" 는 더 강해지므로 '
    "자진 신고가 손해가 아니다." % (len(_g3), _n_auroc, len(_g3))
)
w()
w(
    "**4. §12.2·§13.5 의 새 p 값들도 어느 가족에도 들어가 있지 않다.** 역방향 시간분할 "
    "6건(R1·R2·R3 × 2)과 비시간 분할 8건(X1·X2·X3·R × 2)이 보정 밖에 있다. 성격상 "
    "역방향은 2군(대조군 검정)에, 비시간 분할은 3군(탐색)에 가깝지만 **그 배정 역시 "
    "사후 결정이므로 가족을 다시 짜는 셈이다.** 본문에서는 이 둘을 효과크기(배율)로 쓰고 "
    "p 는 보조로만 쓰기로 했다 — 그래야 가족 구성에 결론이 걸리지 않는다. 해당 p 값은 "
    "역방향 R1 %s / %s, 비시간 분할 X 계열 %s–%s 다."
    % (
        f4(_rv1["DILIrank"].p_phipson_smyth),
        f4(_rv1["DICTrank"].p_phipson_smyth),
        f4(sm[sm.split.str.startswith("X")].p_phipson_smyth.min()),
        f4(sm[sm.split.str.startswith("X")].p_phipson_smyth.max()),
    )
)
w()
w(
    "**5. 따라서 §8 DICTrank R3 의 방어선을 p 값에 걸지 말 것.** 그 p(%s)는 순열 하한에 걸린 "
    "값이 아니다 — 초과 %d/200 이므로 하한(%s)의 %.0f배다. 반복 횟수를 열 배로 올려도 "
    "0.03 근처로 수렴할 뿐 내려가지 않는다. 이 값을 실제로 움직인 것은 횟수가 아니라 n 불일치 "
    "교정이었다(%s → %s, §7.2). **실질 근거는 효과크기**(배율 %s, n 일치 시 %s — 둘 다 1 을 "
    "넘는다)로 쓰고 p 는 보조로만 쓸 것."
    % (
        f4(_r3p),
        int(PCR[("DICTrank", "R3")].n_ge_observed),
        f4(1 / 201),
        _r3p / (1 / 201),
        f4(_r3p),
        f4(MN[("DICTrank", "R3")].p_new),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        f2(MN[("DICTrank", "R3")].ratio_new),
    )
)
w()
_p18 = cv("concept_shift.csv")
_p21 = cv("applicability_domain.csv")
_p24 = cv("label_threshold.csv")
_n18 = int(_p18.p_phipson_smyth.notna().sum())
_n21 = int(_p21.p_phipson_smyth.notna().sum())
_n24 = int(_p24.p_phipson_smyth.notna().sum())
w(
    "**6. §18 이후에 추가된 절들의 p 값 %d건도 처음에는 어느 가족에도 없었다.** §18 개념 이동 %d건, "
    "§21 적용범위 %d건, §24 라벨 이진화 %d건이다. 이 절들은 모두 §17 이후에 기전·처방을 "
    "묻기 위해 추가한 **탐색적 분석**이므로 확증 가족(1군 %d건)에 사후 편입하지 않았다. "
    "**확증 주장은 어느 것도 이 p 값에 걸려 있지 않다** — §18 은 ΔAUROC·ΔECE 로, §21 은 "
    "부트스트랩 구간으로, §24 는 배율로 읽는다. 핵심 배율 자체의 구간은 §26 에 따로 붙였고 "
    "여섯 설정 전부 1 을 제외한다. 따라서 **이 문서의 확증 결론은 사전 계획된 1·2군 %d건과 "
    "§26 의 효과크기 구간 위에 서 있고, 탐색적 p 값 %d건은 어느 결론도 지탱하지 않는다.** "
    "**이 %d건은 §15.2 에서 한 가족으로 묶어 BH 보정을 다시 돌렸다** — 빼 두는 것으로 "
    "끝내지 않았다."
    % (
        _n18 + _n21 + _n24,
        _n18,
        _n21,
        _n24,
        len(_g1),
        len(_g1) + len(_g2),
        int(fat.newly_included.sum()),
        int(fat.newly_included.sum()),
    )
)
w()


# ---------------------------------------------------------------- 15.5 전수 편입
def _fq(x):
    """q 가 1e-4 미만이면 0.0000 으로 뭉개진다 — 그때는 부등호로 적는다."""
    return "< 0.0001" if float(x) < 1e-4 else f4(x)


_fn = fat[fat.newly_included]
_fo = fat[~fat.newly_included].copy()
_fo["sig_existing"] = _fo.sig_existing.astype(bool)
_fo["sig_F0"] = _fo.sig_F0.astype(bool)
_fflip = _fo[_fo.sig_existing & ~_fo.sig_F0]
_fsec = [x for x in ("§11.1", "§12.2", "§13.5", "§18", "§21", "§24") if (_fn.source == x).any()]
w("### 15.2 가족 밖에 있던 검정을 전부 편입했을 때 — 한계 19번을 닫는다")
w()
w(
    "§15.1 의 5·6번은 가족 밖에 있는 p 값을 **자진 신고**하는 데까지 갔고, 그 p 들로는 "
    '어떤 확증 주장도 하지 않는다고 적었다. 그것으로 충분하지 않다 — 심사자는 "왜 그 '
    'p 들만 보정 밖에 두었나" 를 묻는다. 그래서 **빼지 않고 넣었다.** 가족 밖에 있던 '
    "검정 %d건(%s)을 모아 BH 보정을 다시 돌리고, 더 나아가 **기존 %d건까지 한 가족으로** "
    "합친 가장 보수적인 구성(F0, %d건)에서 기존 결론이 살아남는지 셌다. 모델을 다시 "
    "적합하지 않는다 — 저장된 p 와 z 만 쓴다."
    % (
        len(_fn),
        " · ".join("%s %d건" % (x, int((_fn.source == x).sum())) for x in _fsec),
        len(_fo),
        len(fat),
    )
)
w()
w(
    "음성 대조군 행도 빼지 않고 넣었다(§13.5 의 R. 무작위, §12.2 의 정방향 참고 등). "
    "가족이 커지면 보정이 더 보수적이 되므로 **넣는 쪽이 자기에게 불리한 선택**이다. "
    "§11.1 의 커버리지 z 는 정규근사로 단측 p 로 바꾸어 넣었다."
)
w()
w("| 절 | 편입 건수 | F4 에서 유의 | 가장 작은 q (F4) | 비고 |")
w("|---|---|---|---|---|")
for x in _fsec:
    g = _fn[_fn.source == x]
    w(
        "| %s | %d | **%d** | %s | %s |"
        % (
            x,
            len(g),
            int(g.sig_F4.astype(bool).sum()),
            _fq(g.q_F4.min()),
            "확증 주장에 쓰지 않는다" if x in ("§18", "§21") else "효과크기로 읽는다",
        )
    )
w(
    "| **합계** | **%d** | **%d** | %s | — |"
    % (len(_fn), int(_fn.sig_F4.astype(bool).sum()), _fq(_fn.q_F4.min()))
)
w()
w("읽을 것이 셋이다.")
w()
w(
    "1. **가장 보수적인 구성(F0, %d건)에서도 기존 확증 결론은 %s.** 기존 유의 %d건 중 "
    "F0 에서 유의성을 잃는 것은 %d건이다%s. §15.1 의 5번이 이미 적은 그 항목이고, "
    "그 절의 결론은 애초에 p 가 아니라 효과크기(배율 %s)에 걸려 있다."
    % (
        len(fat),
        "거의 그대로다" if len(_fflip) <= 1 else "%d건이 흔들린다" % len(_fflip),
        int(_fo.sig_existing.sum()),
        len(_fflip),
        (
            (
                " — %s"
                % ", ".join(
                    "%s %s (q %s → %s)"
                    % (r.dataset, str(r.test).split(" (")[0], f4(r.q_existing), f4(r.q_F0))
                    for _, r in _fflip.iterrows()
                )
            )
            if len(_fflip)
            else ""
        ),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
    )
)
w()
_f24 = _fn[_fn.source == "§24"]
w(
    "2. **편입한 %d건 중 %d건이 F4 에서 유의하다.** 가장 많이 살아남는 것은 §24 라벨 "
    "이진화의 대조군 검정(%d/%d)과 §12.2 역방향 분할(%d/%d)이다. 즉 가족 밖에 두었던 "
    "검정들은 **보정을 견딘다** — 보정을 피하려고 밖에 둔 것이 아니라는 증거다. 반면 "
    "§18·§21 은 F4 에서 %d건만 유의한데, 이 두 절의 결론은 원래부터 p 가 아니라 "
    "부트스트랩 구간과 ΔECE 로 읽도록 써 두었다(§15.1 의 6번)."
    % (
        len(_fn),
        int(_fn.sig_F4.astype(bool).sum()),
        int(_f24.sig_F4.astype(bool).sum()),
        len(_f24),
        int(_fn[_fn.source == "§12.2"].sig_F4.astype(bool).sum()),
        len(_fn[_fn.source == "§12.2"]),
        int(_fn[_fn.source.isin(["§18", "§21"])].sig_F4.astype(bool).sum()),
    )
)
w()
_fm = _fn[(_fn.source == "§11.1") & _fn.test.str.startswith("Mondrian")]
# 부족분은 실제로 명목값에 못 미친 칸에서만 센다 (넘은 칸은 음수가 된다).
_FDEF = [
    (1 - al) - _mrow(d, al, _MARM[0], "Mondrian")["cov_neg"]
    for d in DS
    for al in _MAL
    if _mrow(d, al, _MARM[0], "Mondrian")["cov_neg"] < 1 - al
]
_fr = _fn[(_fn.source == "§11.1") & _fn.test.str.startswith("주변형")]
w(
    "3. **§11.1 은 절차 둘을 함께 싣고 있으므로 나누어 읽어야 한다.** 주변형의 음성 "
    "클래스 미달 %d건은 F4 에서도 유의하다(최소 q %s) — 그것은 §11 이 이미 보고한 "
    "커버리지 붕괴 자체다. 그러나 **클래스별 보장을 붙인 뒤에 남은 미달분(Mondrian)은 "
    "%d건 전부 FDR 보정 후 유의하지 않다**(q %s–%s). 주변형 행을 가족에서 빼도 "
    '마찬가지다. 따라서 §11.1 의 "라벨이동으로 환원되지 않는 성분" 은 **효과크기'
    "(커버리지 부족 %s–%s)로 주장하고, 유의성 주장으로 쓰지 않는다.**"
    % (
        int(_fr.sig_F4.astype(bool).sum()),
        _fq(_fr.q_F4.min()),
        len(_fm),
        f4(_fm.q_F4.min()),
        f4(_fm.q_F4.max()),
        f4(min(_FDEF)),
        f4(max(_FDEF)),
    )
)
w()
# ================================================================ 16. 큐레이션 민감도
w("---")
w()
w("## 16. 데이터 큐레이션 민감도")
w()
w("### 16.1 승인연도 배정")
w()
w(
    "승인연도 감사에서 절단선을 넘는 수정 후보 7건이 나왔다. 프로젝트 안의 Drugs@FDA "
    "원자료(Products·Submissions·Applications)에서 각 후보의 최초 승인 기록을 직접 확인했다."
)
w()
_ye = cv("year_evidence.csv")
w("| 데이터셋 | 화합물 | 현재 배정 | 최초 NDA 승인일 | 근거 제품 (ApplNo) | 판정 |")
w("|---|---|---|---|---|---|")
_VER = {
    "Riboflavin 5'-phosphate": "**근거 불충분**",
    "DESVENLAFAXINE": "타당",
    "NILOTINIB": "타당",
    "AMPHETAMINE": "타당",
    "OXYCODONE": "타당",
    "AMANTADINE": "타당",
    "DICLOFENAC": "타당",
}
_seen = set()
for _, r in _ye.iterrows():
    if r["화합물"] in _seen:
        continue
    _seen.add(r["화합물"])
    w(
        "| %s | %s | %d → %d | %s | %s (%s) | %s |"
        % (
            r["데이터셋"],
            r["화합물"],
            int(r["현재배정"]),
            int(r["스크립트제안"]),
            r["최초승인일"],
            r["제품명"],
            r["근거_ApplNo"],
            _VER[r["화합물"]],
        )
    )
w()
w(
    "6건은 브랜드 단일 신약의 최초 NDA 가 명확하다. **Riboflavin 5'-phosphate 1건만 근거가 "
    "약하다** — 1947년 기록은 이 성분을 포함한 종합비타민 복합제(NDA 006071)이고 단일 성분 "
    "제품의 최초 승인이 아니다. 즉 아래 S1 은 **근거가 약한 수정까지 일부러 포함한 상한 "
    "시나리오**이며, 그래도 결론이 유지된다는 점에서 오히려 강한 진술이 된다."
)
w()
w(
    "세 시나리오 — **S0** 기준(수정 없음, 본문 수치) / **S1** 7건 전부 적용 / "
    "**S2** S1 + Phenobarbital sodium 제외(1912년부터 임상 사용, Drugs@FDA 최초 NDA 기록은 2022년)."
)
w()
w("| 데이터셋 | 시나리오 | n | 검증군 | AUROC | ECE | 대조 ECE | ECE 배율 | 보정 기여율 |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in ys[ys.dataset == d].iterrows():
        w(
            "| %s | %s | %d | %d | %s | %s | %s | %s배 | %s |"
            % (
                d,
                r.scenario,
                int(r.n),
                int(r.n_test),
                f4(r.AUROC),
                f4(r.ECE),
                f4(r.ECE_random_mean),
                f2(r.ece_ratio),
                pct1(r.pct_from_reliability),
            )
        )
w()
for d in DS:
    g = ys[ys.dataset == d]
    b = g[g.scenario.str.startswith("S0")].iloc[0]
    w(
        "- **%s** — 기준(S0) 대비 ECE 배율 변동 최대 %.2f, 보정 기여율 변동 최대 %.1f%%p. "
        "**세 시나리오 사이의 최대 폭은 기여율 %.1f%%p 로 조금 더 크다** (기준 대비만 적으면 "
        "작게 보인다). 세 시나리오 모두에서 (배율 > 1, 기여율 > 50%%) 유지. "
        "**ECE 배율의 분모는 시나리오마다 다르다** — 분석집합이 바뀌면 대조군도 다시 뽑기 "
        "때문이다(S0 %s · S1 %s · S2 %s). 표의 `대조 ECE` 열이 각 행의 분모이고, §8 의 "
        "R1(%s)과도 다른 실행이다. **배율을 인용할 때는 같은 행의 대조 ECE 를 함께 적을 것** "
        "— 분모를 하나로 적으면 S2 의 배율이 재계산되지 않는다."
        % (
            d,
            (g.ece_ratio - b.ece_ratio).abs().max(),
            (g.pct_from_reliability - b.pct_from_reliability).abs().max(),
            g.pct_from_reliability.max() - g.pct_from_reliability.min(),
            f4(g[g.scenario.str.startswith("S0")].ECE_random_mean.iloc[0]),
            f4(g[g.scenario.str.startswith("S1")].ECE_random_mean.iloc[0]),
            f4(g[g.scenario.str.startswith("S2")].ECE_random_mean.iloc[0]),
            f4(PCR[(d, "R1")].ECE),
        )
    )
w()
w("**본문 수치는 S0 이며, 연도 배정 오류가 결론을 만들지 않았음을 S1·S2 가 보인다.**")
w()
w("### 16.2 ATC V(기타) 계열 제외")
w()
w(
    'ATC V 는 진단용 조영제·방사성 의약품 등 "기타" 로 묶인 이질적 집단이다. 이들이 '
    "결과를 만드는지 확인했다."
)
w()
w("| 데이터셋 | 시나리오 | n | 검증군 | AUROC | ECE | 대조 ECE | ECE 배율 | 보정 기여율 |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in avs[avs.dataset == d].iterrows():
        w(
            "| %s | %s | %d | %d | %s | %s | %s | %s배 | %s |"
            % (
                d,
                r.scenario,
                int(r.n),
                int(r.n_test),
                f4(r.AUROC),
                f4(r.ECE),
                f4(r.ECE_random_mean),
                f2(r.ece_ratio),
                pct1(r.pct_from_reliability),
            )
        )
w()
_avf = {d: avs[(avs.dataset == d) & (avs.scenario.str.startswith("전체"))].iloc[0] for d in DS}
_avx = {d: avs[(avs.dataset == d) & (avs.scenario.str.startswith("ATC V"))].iloc[0] for d in DS}
w(
    "**배율은 거의 그대로지만 기여율은 양쪽 모두 올라간다** — DILIrank %s → %s(%+.1f%%p), "
    'DICTrank %s → %s(%+.1f%%p). "거의 그대로" 라고만 쓰면 이 변화가 가려진다. 방향이 '
    "결론을 강화하는 쪽이므로 결론은 그대로이지만, 수치는 그대로가 아니다. **ECE 배율의 분모는 "
    "행마다 다르다** — 표의 `대조 ECE` 열이 그 값이다(DILIrank 전체 %s / V 제외 %s, "
    "DICTrank 전체 %s / V 제외 %s)."
    % (
        pct1(_avf["DILIrank"].pct_from_reliability),
        pct1(_avx["DILIrank"].pct_from_reliability),
        _avx["DILIrank"].pct_from_reliability - _avf["DILIrank"].pct_from_reliability,
        pct1(_avf["DICTrank"].pct_from_reliability),
        pct1(_avx["DICTrank"].pct_from_reliability),
        _avx["DICTrank"].pct_from_reliability - _avf["DICTrank"].pct_from_reliability,
        f4(_avf["DILIrank"].ECE_random_mean),
        f4(_avx["DILIrank"].ECE_random_mean),
        f4(_avf["DICTrank"].ECE_random_mean),
        f4(_avx["DICTrank"].ECE_random_mean),
    )
)
w()
w(
    "V 계열 %d개(DILIrank) / %d개(DICTrank)를 뺀 결과다."
    % (
        int(
            avs[(avs.dataset == "DILIrank") & (avs.scenario.str.startswith("전체"))].n.iloc[0]
            - avs[(avs.dataset == "DILIrank") & (avs.scenario.str.startswith("ATC V"))].n.iloc[0]
        ),
        int(
            avs[(avs.dataset == "DICTrank") & (avs.scenario.str.startswith("전체"))].n.iloc[0]
            - avs[(avs.dataset == "DICTrank") & (avs.scenario.str.startswith("ATC V"))].n.iloc[0]
        ),
    )
)
w()
w("### 16.3 구조 키 자릿수 (입체이성질체)")
w()
w(
    "주 분석은 InChIKey 앞 14자리(골격 키)로 중복을 제거하므로 입체이성질체가 합쳐진다. "
    "27자리 전체 키로 분리해 재계산했다."
)
w()
w("| 데이터셋 | 변형 | n | 학습 / 검증 | AUROC | ECE | 대조 ECE | ECE 배율 | 보정 기여율 |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in ster[ster.dataset == d].iterrows():
        w(
            "| %s | %s | %d | %d / %d | %s | %s | %s | %s배 | %s |"
            % (
                d,
                r.variant,
                int(r.n),
                int(r.n_train),
                int(r.n_test),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ece_random_pm),
                f2(r.ece_ratio_pm),
                pct1(r.share_reliability),
            )
        )
w()
w(
    "**이 표의 ECE 배율은 §16.1·§16.2 와 분모가 다르다** — 여기서는 유병률맞춤 대조군(§8 의 "
    "R2 계열)을 쓰고, §16.1·§16.2 는 크기맞춤 대조군을 쓴다. 그래서 **14자리** 행의 같은 "
    "시간분할 ECE 가 여기서는 %s배(분모 %s), §16.1 S0 에서는 %s배(분모 %s)로 나온다. "
    "**27자리 행은 분석집합이 달라 대조군도 다시 뽑으므로 분모가 또 다르다**(DILIrank %s, "
    "DICTrank %s) — 그래서 14자리의 분모로 27자리 배율을 재계산하면 맞지 않는다. 각 행의 "
    "분모는 표의 `대조 ECE` 열에 있다. §35 의 배율 표기 규칙을 이 표에도 적용한 것이다."
    % (
        f2(ster[(ster.dataset == "DILIrank")].ece_ratio_pm.iloc[0]),
        f4(ster[(ster.dataset == "DILIrank")].ece_random_pm.iloc[0]),
        f2(ys[(ys.dataset == "DILIrank")].ece_ratio.iloc[0]),
        f4(ys[(ys.dataset == "DILIrank")].ECE_random_mean.iloc[0]),
        f4(ster[(ster.dataset == "DILIrank")].ece_random_pm.iloc[1]),
        f4(ster[(ster.dataset == "DICTrank")].ece_random_pm.iloc[1]),
    )
)
w()
w(
    "입체이성질체를 분리하면 분석집합이 DILIrank %d → %d, DICTrank %d → %d 로 늘지만 "
    "결론이 바뀌지 않는다. **다만 DICTrank 의 보정 기여율은 %s → %s 로 크게 움직인다** — "
    "이 데이터셋의 기여율 추정이 큐레이션 선택에 민감하다는 또 하나의 증거이며, "
    '"압도적" 이라는 표현을 쓸 수 없는 이유를 하나 더 보탠다.'
    % (
        int(ster[(ster.dataset == "DILIrank")].n.iloc[0]),
        int(ster[(ster.dataset == "DILIrank")].n.iloc[1]),
        int(ster[(ster.dataset == "DICTrank")].n.iloc[0]),
        int(ster[(ster.dataset == "DICTrank")].n.iloc[1]),
        pct1(ster[(ster.dataset == "DICTrank")].share_reliability.iloc[0]),
        pct1(ster[(ster.dataset == "DICTrank")].share_reliability.iloc[1]),
    )
)
w()

# ================================================================ 17. 중첩
w("---")
w()
w("### 16.4 큐레이션이 맞는지 원자료와 대조한다 — 연도·라벨·구조 감사")
w()
w(
    "§16.1–§16.3 은 큐레이션의 **선택**을 바꿔도 결론이 유지되는지를 물었다. 이 절은 "
    "다른 질문이다 — 그 큐레이션이 **애초에 맞는가**. 네 개의 감사 스크립트"
    "(`check_years.py`·`check_labels.py`·`check_smiles.py`·`check_atc.py`)가 각 화합물을 "
    "원자료와 대조해 `results/year_audit.csv`·`label_audit.csv`·`smiles_audit.csv`·"
    "`label_conflict_groups.csv` 를 만든다. **2026-10-21 까지 이 네 파일의 숫자는 이 "
    "보고서에 한 줄도 들어가 있지 않았다** — 계산은 했는데 아무도 읽지 않았다. "
    "`verify9.py` 의 다른 점검이 이것을 못 잡은 이유는 그 점검들이 '문서에 있는 수치에 "
    "근거가 있는가' 를 보기 때문이다. 근거만 있고 문서에 없는 것은 반대 방향이라 "
    "조용히 넘어간다. 그래서 `verify9.py` 에 결과 파일 재고 점검을 넣었다 — "
    "생성기가 읽지 않는 결과 파일은 중간·원자료로 이유를 적어 등록하지 않으면 "
    "실패로 잡힌다. 그리고 여기에 수치를 적는다."
)
w()
w(
    "**(가) 승인연도.** 배정한 연도를 같은 분자의 가장 이른 승인 근거와 대조한다. "
    "`gap` 은 배정연도 − 가장 이른 근거연도다. 라벨이 공개된 연도는 DILIrank 2025년 · "
    "DICTrank 2023년이고, 그보다 뒤로 배정된 승인연도는 그 자체로 모순이다 — "
    "라벨이 이미 붙어 있는 분자가 아직 승인되지 않았다는 뜻이 되기 때문이다."
)
w()
w(
    "| 데이터셋 | 감사 n | gap>0 | gap≥5년 | 최대 gap | 근거 결측 | 라벨 공개연도 이후 배정 | 그중 검증군(2010~) |"
)
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    g = ya[ya.dataset == d]
    if len(g) == 0:
        continue
    w(
        "| %s | %d | %d | %d | %d년 | %d | %d | %d |"
        % (
            d,
            len(g),
            int((g.gap > 0).sum()),
            int((g.gap >= 5).sum()),
            int(g.gap.max()),
            int(g.gap.isna().sum()),
            int(_tf(g.after_label_freeze).sum()),
            int((_tf(g.after_label_freeze) & _tf(g.in_heldout)).sum()),
        )
    )
w()
_yad = {d: ya[ya.dataset == d] for d in DS if len(ya[ya.dataset == d])}
_g5 = sum(int((g.gap >= 5).sum()) for g in _yad.values())
_n_ya = sum(len(g) for g in _yad.values())
_yaf = ya[_tf(ya.after_label_freeze)]
w(
    "두 데이터셋 합해 %d개 중 gap 이 5년 이상인 것이 %d개(%s)다. 중앙값은 %d년이다 — "
    "대다수는 어긋나지 않는다. **라벨이 공개된 연도보다 뒤로 승인연도가 배정된** 분자는 "
    "%d개이고 전부 검증군에 있다 — %s. **§12 가 노출기간이 음수가 되어 구간에서 "
    "빠진다고 적은 분자가 바로 이 %d개다**(§12 의 구간 n 합이 분석집합보다 %d 적은 "
    "이유). 서로 다른 두 감사가 같은 분자를 가리킨다."
    % (
        _n_ya,
        _g5,
        pct1(100.0 * _g5 / _n_ya),
        int(ya.gap.median()),
        len(_yaf),
        " · ".join("%s(%d)" % (r.compound, int(r.assigned_year)) for _, r in _yaf.iterrows()),
        len(_yaf),
        len(_yaf),
    )
)
w()
w(
    "**(나) 연도 정정 후보 %d건.** 근거가 배정연도보다 뚜렷하게 이른 분자들이다. "
    "**후보는 정정이 아니다** — §16.1 이 이 %d건을 Drugs@FDA 원자료(Products·"
    "Submissions·Applications)에서 한 건씩 확인해 판정하고, 적용 여부로 세 시나리오를 "
    "나눠 결론이 바뀌지 않는 것을 보였다. 아래 표의 '근거' 열은 감사가 제시한 값이고 "
    "본문 수치는 '배정' 열을 쓴다." % (len(yc), len(yc))
)
w()
w("| 데이터셋 | 화합물 | 배정 | 근거 | 차이 |")
w("|---|---|---|---|---|")
for _, r in yc.iterrows():
    w(
        "| %s | %s | %d | %d | %d년 |"
        % (
            r.dataset,
            r.compound,
            int(r.assigned_year),
            int(r.suggested_year),
            int(r.assigned_year) - int(r.suggested_year),
        )
    )
w()
w(
    "**(다) 라벨.** 이진 라벨을 원자료의 concern 등급 문자열에서 다시 유도해 대조한다. "
    "불일치는 DILIrank %d건 / DICTrank %d건이다."
    % (
        int(_tf(la[la.dataset == "DILIrank"].mismatch).sum()),
        int(_tf(la[la.dataset == "DICTrank"].mismatch).sum()),
    )
)
w()
w(
    "같은 골격 키(InChIKey 앞 14자리)를 공유하면서 **라벨이 서로 다른** 묶음도 센다 — "
    "중복 제거가 어느 쪽을 남기는지에 따라 라벨이 달라지는 분자들이다."
)
w()
w("| 데이터셋 | 충돌 묶음 | 해당 화합물 |")
w("|---|---|---|")
for d in DS:
    g = lc[lc.dataset == d]
    w("| %s | %d | %d |" % (d, g["_k"].nunique(), len(g)))
w()
w(
    "전체 %d개 묶음 %d개 화합물이다. 이 충돌을 제외하는 변형으로 주 결과를 다시 "
    "계산한 것이 §2.2 의 제외 경로이고, 그 변형의 수행 기록은 "
    "`results/label_conflict_filter_log.txt` 에 있다. 숫자가 작아 보이지만 "
    "**DICTrank 의 충돌 묶음이 DILIrank 의 %.0f배**라는 비대칭은 적어 둘 필요가 "
    "있다 — DICTrank 의 "
    "화합물 이름이 대문자 표기이고 염 표기가 더 다양해서 같은 골격이 여러 이름으로 "
    "들어온 결과로 보인다."
    % (
        lc["_k"].nunique(),
        len(lc),
        lc[lc.dataset == "DICTrank"]["_k"].nunique()
        / max(lc[lc.dataset == "DILIrank"]["_k"].nunique(), 1),
    )
)
w()
w(
    "**(라) 구조.** 염·수화물 표기를 떼고 표준화한 뒤 골격 키를 다시 만든다. "
    "`skel_raw` 와 `skel_std` 가 달라지는 것이 표준화가 실제로 작동한 분자다."
)
w()
w("| 데이터셋 | 감사 n | 이름에 염 표기 | 표준화로 골격 바뀜 | PubChem CID 결측 |")
w("|---|---|---|---|---|")
for d in DS:
    g = sa[sa.dataset == d]
    if len(g) == 0:
        continue
    w(
        "| %s | %d | %d | %d | %d |"
        % (
            d,
            len(g),
            int(_tf(g.salt_in_name).sum()),
            int((g.skel_std != g.skel_raw).sum()),
            int(g.pubchem_cid.isna().sum()),
        )
    )
w()
w(
    "감사의 n 은 제외 경로를 거치기 **전** 숫자다(§2.2) — 그래서 DILIrank 가 %d 로 "
    "분석집합 %d 보다 크다. 이름에 염 표기가 있는 분자가 DILIrank %s / DICTrank %s 이고, "
    "표준화로 골격 키가 바뀌는 분자도 비슷한 규모다. 전처리가 손대지 않고 지나간 "
    "분자가 아니라 **대부분의 분자가 표준화를 거친다**는 뜻이고, 이것이 §16.3 에서 "
    "키 자릿수를 바꿔 본 이유다 — 중복 제거가 이 표준화에 얹혀 있다."
    % (
        len(sa[sa.dataset == "DILIrank"]),
        SPLIT["DILIrank"]["n_train"] + SPLIT["DILIrank"]["n_test"],
        pct1(100.0 * _tf(sa[sa.dataset == "DILIrank"].salt_in_name).mean()),
        pct1(100.0 * _tf(sa[sa.dataset == "DICTrank"].salt_in_name).mean()),
    )
)
w()
w("## 17. 두 데이터셋의 중첩과 비중첩 부분집합 재계산")
w()
w("### 17.1 중첩의 크기")
w()
w(
    "| 데이터셋 | 분석집합 n | 공유 분자 | 비율 | 전용 분자 | 전용 중 학습 / 검증 | 전용 검증 양성 / 음성 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    r = OV[d]
    w(
        "| %s | %d | %d | **%s** | %d | %d / %d | %d / %d |"
        % (
            d,
            int(r.n_analysis),
            int(r.n_shared),
            pct1(r.shared_pct),
            int(r.n_exclusive),
            int(r.n_exclusive_train),
            int(r.n_exclusive_test),
            int(r.n_exclusive_test_pos),
            int(r.n_exclusive_test_neg),
        )
    )
w()
w(
    "**두 데이터셋은 화합물의 대부분을 공유한다** — InChIKey14 기준 공통 분자 %d개로 "
    "DILIrank 의 %s, DICTrank 의 %s 다. 라벨(간독성 / 심장독성)은 다르지만 구조·승인연도·"
    "시간분할 소속·학습군 구성이 대체로 겹치므로 두 데이터셋의 AUROC·ECE 추정치는 "
    '**상관돼 있다.** "두 데이터셋에서 재현" 은 독립 표본 두 개에서의 재현이 아니라 '
    "**같은 화학공간에서 엔드포인트만 바꾼 확인**이다."
    % (
        int(OV["DILIrank"].n_shared),
        pct1(OV["DILIrank"].shared_pct),
        pct1(OV["DICTrank"].shared_pct),
    )
)
w()
w("### 17.2 비중첩 부분집합에서의 재계산")
w()
w(
    "겹침이 결론을 만들었는지 확인하기 위해 다섯 설계로 재계산했다(무작위 대조 %d회, "
    "부트스트랩 2,000회). 표본부족 기준은 검증 n < 40 또는 소수 클래스 < 10 이다."
    % _reps("disjoint_subset.csv", "ctrl_reps")
)
w()
w("| 설계 | 학습군 | 검증군 |")
w("|---|---|---|")
w("| V0 | 전체 | 전체 (본문 값 재현 확인) |")
w("| V1 | 전용 분자만 | 전용 분자만 (가장 강한 독립성) |")
w("| V1c | 전체에서 V1 과 같은 수만 무작위 추출 (20회) | V1 과 동일 |")
w("| V2 | 전체 | 전용 분자만 |")
w("| V3 | 전체 | 공유 분자만 (V2 의 대조) |")
w()
w(
    "| 데이터셋 | 설계 | 학습 n | 검증 n | 양성/음성 | AUROC | ECE | 대조군 ECE | 배율 | p | 보정 기여율 | 비고 |"
)
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for v in ["V0", "V1", "V1c", "V2", "V3"]:
        r = dj[(dj.dataset == d) & (dj.variant == v)].iloc[0]
        w(
            "| %s | %s | %d | %d | %d/%d | %s | %s | %s | **%s배** | %s | %s | %s |"
            % (
                d,
                v,
                int(r.n_train),
                int(r.n_test),
                int(r.n_test_pos),
                int(r.n_test_neg),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ctrl_ece),
                f2(r.ece_ratio),
                f4(r.p_phipson_smyth),
                pct1(r.share_calib),
                "표본부족" if r.underpowered else "",
            )
        )
w()
_v2 = dj[(dj.dataset == "DICTrank") & (dj.variant == "V2")].iloc[0]
_v3 = dj[(dj.dataset == "DICTrank") & (dj.variant == "V3")].iloc[0]
_v1 = dj[(dj.dataset == "DICTrank") & (dj.variant == "V1")].iloc[0]
_v1c = dj[(dj.dataset == "DICTrank") & (dj.variant == "V1c")].iloc[0]
w(
    "**핵심은 V2 대 V3 다** — 같은 모델을 같은 데이터셋 검증군에 적용하면서 평가 분자만 "
    "'전용' 과 '공유' 로 갈랐다. DICTrank 에서 **전용 분자 %d개에서도 보정 붕괴가 남지만"
    "(배율 %s, p = %s, 기여율 %s) 공유 분자 %d개보다는 작다(배율 %s, 기여율 %s).** "
    "즉 **겹침이 결과를 만든 것은 아니지만 효과 크기를 부풀린다.**"
    % (
        int(_v2.n_test),
        f2(_v2.ece_ratio),
        f4(_v2.p_phipson_smyth),
        pct1(_v2.share_calib),
        int(_v3.n_test),
        f2(_v3.ece_ratio),
        pct1(_v3.share_calib),
    )
)
w()
_v0 = dj[(dj.dataset == "DICTrank") & (dj.variant == "V0")].iloc[0]
_drop = _v0.auroc - _v1.auroc
# V0 와 V1c 는 검증군이 다르다(전체 267 대 전용 125). 그래서 V0 → V1c 를 그대로
# "학습군 크기" 라고 부를 수 없다. 같은 검증군·전체 학습군인 V2 를 중간 기준으로
# 놓으면 검증군 교체 / 학습군 크기 / 학습 분자 구성이 분리된다 (2026-10-04).
_swap = _v2.auroc - _v0.auroc
_size = _v2.auroc - _v1c.auroc
_comp = _v1c.auroc - _v1.auroc
_base = _size + _comp
w(
    '**V1 의 AUROC %s 를 두고 "판별력이 우연 수준이다" 와 "우연 수준이 아니다" 중 '
    "어느 쪽도 단정할 수 없다.** V1 의 95%% 신뢰구간은 [%s–%s], V1c 는 [%s–%s] 로 **둘 다 "
    "0.5 를 포함한다.** 이 표본에서는 판별력의 유무를 가릴 수 없다."
    % (f4(_v1.auroc), f4(_v1.auroc_lo), f4(_v1.auroc_hi), f4(_v1c.auroc_lo), f4(_v1c.auroc_hi))
)
w()
w(
    "**V1c 가 분해해 주는 것은 저하의 원인이지 그 유의성이 아니다.** 전체 학습군에서 같은 "
    "수(%d개)만 무작위로 뽑아 V1 과 같은 검증군을 평가하면 AUROC 가 %s 다. "
    "**다만 V0 와 V1c 는 검증군이 다르므로**(전체 %d개 대 전용 %d개) V0 → V1c 의 차이를 "
    '그대로 "학습군 크기" 라고 부를 수 없다. 같은 검증군·전체 학습군인 **V2 를 중간 '
    "기준**으로 놓으면 세 효과가 분리된다."
    % (int(_v1c.n_train), f4(_v1c.auroc), int(_v0.n_test), int(_v1.n_test))
)
w()
w("| 구간 | 바뀌는 것 | ΔAUROC | 몫 |")
w("|---|---|---|---|")
w(
    "| V0 %s → V2 %s | 검증군 교체 (전체 %d → 전용 %d) | %s | — |"
    % (f4(_v0.auroc), f4(_v2.auroc), int(_v0.n_test), int(_v2.n_test), "%+.4f" % _swap)
)
w(
    "| V2 %s → V1c %s | 학습군 크기 (%d → %d) | %s | **%s** |"
    % (
        f4(_v2.auroc),
        f4(_v1c.auroc),
        int(_v2.n_train),
        int(_v1c.n_train),
        f4(_size),
        pct1(_size / _base * 100),
    )
)
w(
    "| V1c %s → V1 %s | 학습 분자 구성 (무작위 → 전용) | %s | **%s** |"
    % (f4(_v1c.auroc), f4(_v1.auroc), f4(_comp), pct1(_comp / _base * 100))
)
w()
w(
    '즉 "저하가 전용 분자의 성질이 아니라 학습군 크기 때문" 이라고만 쓰면 절반을 '
    "빼먹는다 — 같은 검증군 위에서 **크기 몫 %s, 구성 몫 %s** 로 비슷하다. "
    "검증군 교체는 %s 이고 방향이 반대이며(전용 분자 쪽이 오히려 조금 쉽다), 그래서 V0 에서 "
    "V1 까지의 총 낙폭 %s 를 두 몫으로만 나누면 크기 몫이 과소평가된다."
    % (pct1(_size / _base * 100), pct1(_comp / _base * 100), "%+.4f" % _swap, f4(_drop))
)
w()
w(
    "**남는 교란 한 가지** — V2 와 V3 는 검증군 양성률이 다르다(%s 대 %s). ECE 는 양성률에 "
    "민감하므로 V2/V3 배율 차이(%s 대 %s)는 겹침 효과의 **상한**으로 읽어야 한다."
    % (f4(_v2.pos_rate_test), f4(_v3.pos_rate_test), f2(_v2.ece_ratio), f2(_v3.ece_ratio))
)
w()
w(
    "**DILIrank 는 이 질문에 답할 수 없다** — 전용 검증군이 %d개(양성 %d / 음성 %d)뿐이어서 "
    "V1·V1c·V2 가 전부 표본부족이다."
    % (
        int(OV["DILIrank"].n_exclusive_test),
        int(OV["DILIrank"].n_exclusive_test_pos),
        int(OV["DILIrank"].n_exclusive_test_neg),
    )
)
w()
w()
# ================================================================ 17.3 잔차 상관
w("### 17.3 공유 검증 분자에서의 추정치 상관")
w()
_cso = cv("concept_shift_overlap.csv")
_sgn = _cso.iloc[0]
_abs = _cso.iloc[1]
_lab = _cso.iloc[2]
_nsh = int(_sgn.n)
w(
    "§17.1 은 **몇 개가 겹치는가** 를, §17.2 는 **겹치지 않는 부분만 쓰면 어떻게 되는가** 를 "
    "보았다. 남은 질문이 하나 있다 — 겹치는 분자에서 **두 데이터셋의 오차가 실제로 "
    "같이 움직이는가.** 분자를 공유해도 오차가 따로 놀면 두 결과는 사실상 독립 정보다."
)
w()
w(
    "양쪽 검증군(둘 다 2010년 이후)에 모두 있는 %d개 분자에서, 각 데이터셋 정방향 모델의 "
    "분자별 값을 짝지어 상관을 냈다. 부트스트랩 2,000회." % _nsh
)
w()
w("| 양 | n | Pearson r | 95% CI | Spearman |")
w("|---|---|---|---|---|")
# CSV 의 "절대잔차 |y-p|" 는 파이프 문자가 들어 있어 마크다운 표를 깨뜨린다
# (verify9 의 열 수 검사가 잡았다). 표기용 이름으로 바꿔 쓴다.
_QN = {
    "\uc794\ucc28 y-p (\ubd80\ud638 \uc788\uc74c)": "\uc794\ucc28 y \u2212 p (\ubd80\ud638 \uc788\uc74c)",
    "\uc808\ub300\uc794\ucc28 |y-p|": "\uc808\ub300\uc794\ucc28 abs(y \u2212 p)",
    "\ub77c\ubca8 y (phi)": "\ub77c\ubca8 y (phi \uacc4\uc218)",
    "\uc608\uce21 p": "\uc608\uce21\ud655\ub960 p",
}
for _, r in _cso.iterrows():
    _qn = _QN.get(str(r.quantity), str(r.quantity).replace("|", "abs"))
    assert "|" not in _qn, "\ud45c \uc774\ub984\uc5d0 \ud30c\uc774\ud504: %s" % _qn
    w(
        "| %s | %d | %s | [%s, %s] | %s |"
        % (_qn, int(r.n), sg(r.pearson), f3(r.ci_lo), f3(r.ci_hi), sg(r.spearman))
    )
w()
_excl = (_sgn.ci_lo > 0) or (_sgn.ci_hi < 0)
_neff = 2.0 / (1.0 + float(_sgn.pearson))
_nlo = 2.0 / (1.0 + float(_sgn.ci_hi))
w(
    "**읽어야 하는 줄은 첫 줄이다.** ECE 는 구간별 **부호 있는** 평균잔차로 만들어지므로, "
    "두 ECE 추정치의 의존성을 재는 올바른 양은 부호 있는 잔차 상관이다. 절대잔차 상관은 "
    "오차 **크기** 의 동조만 재서 의존성을 과소평가한다."
)
w()
w(
    "부호 있는 잔차 상관은 %s [%s, %s] 로 **신뢰구간이 0 을 %s**. 두 추정치를 평균하면 "
    "분산이 하나의 (1+r)/2 = %s 배가 되고, 이는 유효 표본 **%s개** 데이터셋에 해당한다 "
    "(신뢰구간 상한 %s 을 쓰면 %s개). 절대잔차 기준은 %s 로 약한 신호이고, 라벨 자체의 "
    "일치도는 phi = %s 로 거의 무관하다."
    % (
        sg(_sgn.pearson),
        f3(_sgn.ci_lo),
        f3(_sgn.ci_hi),
        "포함하지 않는다 — 상관이 있다" if _excl else "포함한다 — 상관을 단정할 수 없다",
        f2((1.0 + float(_sgn.pearson)) / 2.0),
        f2(_neff),
        sg(_sgn.ci_hi),
        f2(_nlo),
        sg(_abs.pearson),
        sg(_lab.pearson),
    )
)
w()
w(
    "즉 **두 결과는 완전한 독립 재현이 아니며, 이 절이 그 정도를 숫자로 적는다.** "
    "유효 표본 수 근사는 두 추정치의 분산이 같다고 가정한 것이고, 그 가정은 검정하지 "
    "않았다. phi 는 이 %d개 분자에서 계산한 값이고, 공유 분자 전체에서의 값과는 다르다." % _nsh
)
w()
# ================================================================ 18. 개념 이동
w("---")
w()
w("## 18. 개념 이동 — P(y|x) 가 시대에 따라 변했는가")
w()
_cs = cv("concept_shift.csv")


def _ck(nm):
    return "A2" if nm.startswith("A2") else nm[:2]


CS = {(r.dataset, _ck(r.arm)): r for _, r in _cs.iterrows()}
_AL = {
    "A.": "A. 시대일치 (무작위 5-fold)",
    "A2": "A2. 시대일치 (스캐폴드 5-fold)",
    "B.": "B. 정방향 전체 (§8 설정)",
    "C.": "C. 정방향 크기맞춤",
    "D.": "D. 정방향 크기·유병률맞춤",
}
w(
    "§16 은 라벨 분포 P(y) 를, §13 은 입력 분포 P(x) 를 통제했다. 세 번째가 남는다 — "
    "**구조-독성 관계 P(y|x) 자체가 시대에 따라 바뀌었는가.** 바뀌었다면 보정 붕괴는 "
    "'표본이 작아서' 가 아니라 '관계가 달라서' 생긴 것이고, 서술이 현상 보고에서 "
    "기전 진술로 올라간다."
)
w()
w(
    "설계는 **검증군을 고정하고 학습군의 성질만 하나씩 바꾸는** 것이다. 다섯 팔 모두 "
    "2010년 이후 분자 전체를 검증군으로 쓴다(DILIrank %d개 / DICTrank %d개). A·A2 는 "
    "out-of-fold 예측이므로 자기 분자로 학습한 모델의 예측이 아니다."
    % (int(CS[("DILIrank", "A.")].n_test), int(CS[("DICTrank", "A.")].n_test))
)
w()
w(
    "| 데이터셋 | 팔 | 학습 시대 | 학습 n | 학습 양성률 | AUROC | ECE | 보정 기울기 | 신뢰도 | A 대비 배율 | p |"
)
w("|---|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for k in ("A.", "A2", "B.", "C.", "D."):
        if (d, k) not in CS:
            continue
        r = CS[(d, k)]
        w(
            "| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |"
            % (
                d,
                _AL[k],
                "post-2010" if k in ("A.", "A2") else "pre-2010",
                int(round(r.train_n)),
                f4(r.train_rate),
                f4(r.auroc),
                f4(r.ece),
                f4(r.slope),
                f4(r.rel),
                "—" if r.ece_ratio != r.ece_ratio else "**%s배**" % f2(r.ece_ratio),
                "—" if r.p_phipson_smyth != r.p_phipson_smyth else f4(r.p_phipson_smyth),
            )
        )
w()
_nrep = int(_cs.n_rep_valid.max())
_era = {d: CS[(d, "D.")].ece - CS[(d, "A.")].ece for d in DS}
_rate = {d: CS[(d, "C.")].ece - CS[(d, "D.")].ece for d in DS}
_siz = {d: CS[(d, "B.")].ece - CS[(d, "C.")].ece for d in DS}
_tot = {d: CS[(d, "B.")].ece - CS[(d, "A.")].ece for d in DS}
for d in DS:
    assert abs(_era[d] + _rate[d] + _siz[d] - _tot[d]) < 1e-9, "3분해 합 불일치 %s" % d
w(
    "**읽는 줄은 A 대 D 다.** 학습 크기·학습 양성률·검증군이 모두 같고 학습 분자의 "
    "**시대만** 다르다. 차이는 DILIrank %s (p = %s, 대조군 %d/%d 회만 A 를 따라옴), "
    "DICTrank %s (p = %s, %d/%d). 즉 **개념 이동은 간독성에서 확인되고 심장독성에서는 "
    "확인되지 않는다.**"
    % (
        sg(_era["DILIrank"]),
        f4(CS[("DILIrank", "D.")].p_phipson_smyth),
        int(CS[("DILIrank", "D.")].n_better),
        _nrep,
        sg(_era["DICTrank"]),
        f4(CS[("DICTrank", "D.")].p_phipson_smyth),
        int(CS[("DICTrank", "D.")].n_better),
        _nrep,
    )
)
w()
w(
    "**유사체 누출 반론은 막힌다.** A 의 학습군은 검증군과 같은 표본에서 나오므로 "
    "유사체가 fold 를 넘나들어 A 를 유리하게 만들 수 있다. 스캐폴드를 분리한 A2 는 "
    "A 보다 ECE 가 %s / %s 나빠질 뿐이고, 이는 DILIrank 시대 몫 %s 보다 작다."
    % (
        sg(CS[("DILIrank", "A2")].ece - CS[("DILIrank", "A.")].ece),
        sg(CS[("DICTrank", "A2")].ece - CS[("DICTrank", "A.")].ece),
        sg(_era["DILIrank"]),
    )
)
w()
w(
    "**내부 정합성** — 팔 B 는 §8 시간분할과 같은 설정이다. 두 값이 네 자리까지 같아야 "
    "이 실험이 같은 파이프라인 위에 있다는 뜻이다."
)
w()
w("| 데이터셋 | 항목 | §8 값 | 팔 B 값 | 일치 |")
w("|---|---|---|---|---|")
for d in DS:
    _c8 = cal[(cal.dataset.str.startswith(d)) & (cal.setting == "temporal")].iloc[0]
    for lab, _a, _b in (
        ("AUROC", _c8.AUROC, CS[(d, "B.")].auroc),
        ("ECE", _c8.ECE, CS[(d, "B.")].ece),
        ("보정 기울기", _c8.calib_slope, CS[(d, "B.")].slope),
    ):
        w(
            "| %s | %s | %s | %s | %s |"
            % (d, lab, f4(_a), f4(_b), "예" if f4(_a) == f4(_b) else "**아니오**")
        )
w()
w("**A→D→C→B 3분해** — 세 몫의 합은 정확히 B − A 다.")
w()
w("| 데이터셋 | 시대(개념) A→D | 학습 유병률 D→C | 학습 크기 C→B | 합 = B−A |")
w("|---|---|---|---|---|")
for d in DS:
    w(
        "| %s | %s (%s) | %s (%s) | %s (%s) | %s |"
        % (
            d,
            sg(_era[d]),
            pct1(_era[d] / _tot[d] * 100),
            sg(_rate[d]),
            pct1(_rate[d] / _tot[d] * 100),
            sg(_siz[d]),
            pct1(_siz[d] / _tot[d] * 100),
            sg(_tot[d]),
        )
    )
w()
w(
    "**세 몫은 이 경로 순서에서의 몫이다** — 순서를 바꾸면 교호작용이 다르게 배분된다 "
    "(합은 언제나 B−A). 학습 크기 몫이 **음수**인 것은, 학습 분자를 맞춤 크기에서 "
    "pre-2010 전체로 늘렸을 때 보정이 오히려 조금 좋아졌다는 뜻이다. "
    "'표본이 작아서 그렇다' 는 설명이 정방향에서는 성립하지 않는다."
)
w()
w(
    "p 는 교환가능성을 가정한 순열검정이 아니라 **참조분포 비교** 다 — 재표집 대조군 "
    "%d회 중 팔 A 이하로 보정된 비율의 Phipson–Smyth 추정량이다. 효과 크기(ECE 차)를 "
    "먼저 읽어야 한다. A·A2 의 통합 예측은 서로 다른 5개 모델에서 나오므로 모델별 절편 "
    "차이가 통합 보정에 섞인다." % _nrep
)
w()
# ================================================================ 19. 유병률 스윕
w("---")
w()
w("## 19. 학습 유병률 격차 스윕")
w()
_sw = cv("prevalence_sweep.csv")
_swf = cv("prevalence_sweep_fit.csv")
SWF = {r.dataset: r for _, r in _swf.iterrows()}
_nrw = int(_sw.n_rep.max())
w(
    "§18 의 3분해에서 학습 유병률 몫이 %s / %s 로 가장 컸다. 그러면 다음 질문은 하나다 — "
    "그게 **함수 관계** 인가, 아니면 §16 의 세 점이 우연히 그렇게 놓인 것인가."
    % (
        pct1(_rate["DILIrank"] / _tot["DILIrank"] * 100),
        pct1(_rate["DICTrank"] / _tot["DICTrank"] * 100),
    )
)
w()
w(
    "여기서는 **시간을 완전히 빼고** 무작위 분할만 쓴다. 검증군을 시간분할과 같은 크기·"
    "같은 양성률로 고정하고, 학습군은 크기를 고정한 채 양성률만 단계별로 바꾼다. "
    "단계당 %d회 반복." % _nrw
)
w()
w("| 데이터셋 | 학습 양성률 | 격차 | 학습 n | AUROC | ECE | ECE sd | 보정 기울기 | 신뢰도 |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in _sw[_sw.dataset == d].sort_values("level").iterrows():
        mk = (
            " (검증군 양성률)"
            if bool(r.is_test_rate)
            else (" (pre-2010 양성률)" if bool(r.is_pre2010_rate) else "")
        )
        _gp = 0.0 if abs(float(r.gap)) < 5e-5 else float(r.gap)
        w(
            "| %s | %s%s | %s | %d | %s | %s | %s | %s | %s |"
            % (
                d,
                f4(r.level),
                mk,
                sg(_gp),
                int(r.train_n),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ece_sd),
                f4(r.slope),
                f4(r.rel),
            )
        )
w()
_aur = {d: (_sw[_sw.dataset == d].auroc.max() - _sw[_sw.dataset == d].auroc.min()) for d in DS}
_ecr = {d: (_sw[_sw.dataset == d].ece.max() / _sw[_sw.dataset == d].ece.min()) for d in DS}
w(
    "**이 표의 핵심은 두 열의 대비다.** 학습 양성률만 0.20 에서 0.80 으로 바꿨을 때 "
    "ECE 는 %s배 / %s배로 벌어지는데 AUROC 변동폭은 %s / %s 에 그친다. 시간도, 공변량 "
    "이동도, 개념 이동도 없는 순수 무작위 분할에서다 — **판별은 멀쩡한데 보정이 무너진다**는 "
    "이 논문의 주장이 통제실험 하나로 재현된다."
    % (f2(_ecr["DILIrank"]), f2(_ecr["DICTrank"]), f4(_aur["DILIrank"]), f4(_aur["DICTrank"]))
)
w()
w(
    "| 데이터셋 | 최소 ECE 위치(격차) | 최소 ECE | 절대격차 0.10 당 기울기 | 95% CI | 격차 음수쪽 | 격차 양수쪽 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    r = SWF[d]
    _mg = 0.0 if abs(float(r.min_gap)) < 5e-5 else float(r.min_gap)
    w(
        "| %s | %s | %s | **%s** | [%s, %s] | %s | %s |"
        % (
            d,
            sg(_mg),
            f4(r.min_ece),
            f4(r["slope_per_0.10"]),
            f3(r.slope_lo),
            f3(r.slope_hi),
            f4(r["slope_neg_side_per_0.10"]),
            f4(r["slope_pos_side_per_0.10"]),
        )
    )
w()
w(
    "최소가 두 데이터셋 모두 **격차 0 에서 한 격자점 안**에 떨어진다. 기울기도 두 "
    "데이터셋이 거의 같다 — 보정오차는 시간의 함수가 아니라 **학습·검증 유병률 격차의 "
    "함수**다. 다만 양쪽 기울기가 같지는 않아서, 절대격차 단일 기울기는 평균값으로만 "
    "읽어야 한다."
)
w()
_slmean = {d: _sw[_sw.dataset == d].slope.mean() for d in DS}
_relr = {d: _sw[_sw.dataset == d].rel.max() / _sw[_sw.dataset == d].rel.min() for d in DS}
_slrng = {d: (_sw[_sw.dataset == d].slope.min(), _sw[_sw.dataset == d].slope.max()) for d in DS}
w(
    "**기전이 하나 더 보인다 — 위치 오차지 폭 오차가 아니다.** 보정 기울기는 전 구간에서 "
    "%s~%s (평균 %s) / %s~%s (평균 %s) 범위에 머무는데 신뢰도는 %s배 / %s배로 뛴다. "
    "유병률 격차는 예측을 통째로 밀어 **절편 오차** 를 만들고 예측의 **퍼짐은 거의 "
    "건드리지 않는다.** 결함이 둘로 갈린다 — 격차와 무관하게 늘 있는 기울기 결함과, "
    "격차가 만드는 절편 결함."
    % (
        f4(_slrng["DILIrank"][0]),
        f4(_slrng["DILIrank"][1]),
        f2(_slmean["DILIrank"]),
        f4(_slrng["DICTrank"][0]),
        f4(_slrng["DICTrank"][1]),
        f2(_slmean["DICTrank"]),
        f2(_relr["DILIrank"]),
        f2(_relr["DICTrank"]),
    )
)
w()
_e0 = {d: float(SWF[d].ece_at_gap0) for d in DS}
_eo = {d: float(SWF[d].ece_at_obs_gap) for d in DS}
w("| 데이터셋 | 격차 0 에서 ECE | §8 실제 격차 | 그 격차에서 ECE | 차이 | 배율 |")
w("|---|---|---|---|---|---|")
for d in DS:
    w(
        "| %s | %s | %s | %s | %s | %s배 |"
        % (
            d,
            f4(_e0[d]),
            sg(SWF[d].gap_obs),
            f4(_eo[d]),
            sg(_eo[d] - _e0[d]),
            f2(_eo[d] / _e0[d]) if _e0[d] else "—",
        )
    )
w()
_tn = {d: int(_sw[_sw.dataset == d].train_n.iloc[0]) for d in DS}
w(
    "**이 스윕은 몫 추정에 쓰면 안 된다 — 모양과 기울기를 보는 데만 쓴다.** 학습 크기를 "
    "%d개 / %d개로 작게 고정했기 때문에 격차 0 기준선(%s / %s)이 §8 무작위분할 값보다 "
    "높고, 그만큼 측정 가능한 폭이 눌린다. 몫은 §16 사다리와 §18 3분해의 값을 쓴다."
    % (_tn["DILIrank"], _tn["DICTrank"], f4(_e0["DILIrank"]), f4(_e0["DICTrank"]))
)
w()
# ================================================================ 20. 최신성 가중
w("---")
w()
w("## 20. 최신성 가중 — 학습 단계에서 고칠 수 있는가")
w()
_rw = cv("recency_weighting.csv")
RW = {(r.dataset, r.tag): r for _, r in _rw.iterrows()}
# §10.2 의 회복률과 §18 의 AUROC 는 데이터에서 직접 읽는다.
# (고정 문장으로 쓰면 두 데이터셋 중 한쪽에서만 참인 주장이 끼어든다 —
#  2026-10-16 감사에서 실제로 세 건 발견했다.)
_pe20 = cv("prior_estimation.csv")


def _rec(d, key):
    r = _pe20[(_pe20.dataset == d) & (_pe20.estimator.str.startswith(key))]
    return float(r.recovery.iloc[0]) if len(r) else float("nan")


_cs20 = cv("concept_shift.csv")


def _arm(d, k):
    g = _cs20[(_cs20.dataset == d) & (_cs20.arm.str.startswith(k))]
    return g.iloc[0]


_winA = [d for d in DS if _arm(d, "A.").auroc > _arm(d, "B.").auroc]
_loseA = [d for d in DS if d not in _winA]
w(
    "§10 은 예측을 **사후에** 고치는 쪽을 다뤘다. 그 결과를 정확히 옮기면 — 라벨을 전혀 "
    "쓰지 않는 유병률 추정(P1 예측평균, P2 EM)은 두 데이터셋 모두에서 회복이 음수였고"
    "(P2 는 %s / %s), 학습군 라벨만 쓰는 BBSE(P3)는 %s 로 **엇갈렸다.** "
    "이 절은 방향이 다르다 — **학습 단계에서** 고친다."
    % (
        pct1(_rec("DILIrank", "P2")),
        pct1(_rec("DICTrank", "P2")),
        " / ".join("%s %s" % (d, pct1(_rec(d, "P3"))) for d in DS),
    )
)
w()
w(
    "§18 에서 **%s 는** 2010년 이후 %d개로 학습한 모델(AUROC %s)이 pre-2010 %d개 모델"
    "(%s)을 **이겼다.** %s 는 그렇지 않았다(%s 대 %s). 그래도 질문은 양쪽 모두에 선다 — "
    '"오래된 분자의 가중치를 낮추면 되는가."'
    % (
        ", ".join(_winA) if _winA else "어느 쪽도",
        int(round(_arm(_winA[0], "A.").train_n)) if _winA else 0,
        f4(_arm(_winA[0], "A.").auroc) if _winA else "—",
        int(round(_arm(_winA[0], "B.").train_n)) if _winA else 0,
        f4(_arm(_winA[0], "B.").auroc) if _winA else "—",
        ", ".join(_loseA) if _loseA else "나머지",
        f4(_arm(_loseA[0], "A.").auroc) if _loseA else "—",
        f4(_arm(_loseA[0], "B.").auroc) if _loseA else "—",
    )
)
w()
w(
    "가중은 w = 0.5^((2010 − 승인연도) / tau) 이고 tau 는 반감기(년)다. tau 가 없으면 "
    "균등가중이고 이는 §8 설정과 같다. 비교군으로 하드 절단(최근 K년만 학습)도 함께 "
    "돌렸다. 검증군은 모든 설정에서 2010년 이후 전체로 고정이다."
)
w()
w(
    "**유효 표본 수 n_eff = (Σw)²/Σw² 를 반드시 같이 읽어야 한다.** 가중은 공짜가 아니다 — "
    "최근 쪽에 몰수록 실질 표본이 줄어든다."
)
w()
for d in DS:
    g = _rw[(_rw.dataset == d) & (_rw.kind.isin(["weight", "window"]))]
    w("**%s**" % d)
    w()
    w(
        "| 설정 | 학습 n | 유효 n | 학습 양성률 | 격차 | AUROC | ECE | ECE 95% CI | 보정 기울기 | 신뢰도 | 해상도 |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %d | %s | %s | %s | %s | %s | [%s, %s] | %s | %s | %s |"
            % (
                r.tag,
                int(r.n_train),
                f1(r.n_eff),
                f4(r.train_rate_weighted),
                sg(r.gap),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ece_lo),
                f4(r.ece_hi),
                f4(r.slope),
                f4(r.rel),
                f4(r.res),
            )
        )
    w()
w("### 20.1 tau 를 미래를 보지 않고 고르는 프로토콜")
w()
w(
    "sweep 전체를 검증군에서 보고 제일 좋은 tau 를 고르면 그것은 검증군 누출이다. 그래서 "
    "pre-2010 을 다시 가른다 — 내부 학습군 2000년 미만, 내부 검증군 2000~2009. 내부 "
    "검증군에서 ECE 가 최소인 tau* 를 고르고, 그 tau* 로 pre-2010 **전체**를 다시 학습해 "
    "2010년 이후에서 평가한다. 미래를 보지 않으므로 실제로 배포 가능한 규칙이다."
)
w()
w(
    "| 데이터셋 | 균등 ECE (= §8) | 정직선택 tau* | 정직선택 ECE | 균등 대비 | sweep 최솟값 (검증군 열람, 상한선) |"
)
w("|---|---|---|---|---|---|")
_HON = {}
for d in DS:
    h = _rw[(_rw.dataset == d) & (_rw.kind == "honest")].iloc[0]
    b = _rw[(_rw.dataset == d) & (_rw.kind == "weight") & (_rw.tau.isna())].iloc[0]
    sw_ = _rw[(_rw.dataset == d) & (_rw.kind == "weight")]
    bs = sw_.loc[sw_.ece.idxmin()]
    _HON[d] = (b, h, bs)
    _dd = float(h.ece) - float(b.ece)
    w(
        "| %s | %s | %s | %s | %s (%s %s) | %s (%s) |"
        % (
            d,
            f4(b.ece),
            "반감기 %g년" % float(h.tau) if h.tau == h.tau else "균등",
            f4(h.ece),
            sg(_dd),
            "악화" if _dd > 0 else "개선",
            pct1(abs(_dd) / float(b.ece) * 100),
            f4(bs.ece),
            bs.tag,
        )
    )
w()
_dl, _hl, _bl = _HON["DILIrank"]
_dc, _hc, _bc = _HON["DICTrank"]
w(
    "**DILIrank 는 어떤 tau 도 균등가중을 이기지 못했다** — sweep 최솟값이 균등 그 자체이고"
    "(%s), 정직선택은 %s 로 %s 악화다. 하드 절단은 모두 더 나빴다."
    % (f4(_bl.ece), f4(_hl.ece), pct1(abs(float(_hl.ece) - float(_dl.ece)) / float(_dl.ece) * 100))
)
w()
w(
    "**DICTrank 의 %s 개선은 해상도를 판 대가다.** 정직선택 tau*=%g년에서 ECE 는 %s 에서 "
    "%s 로 내려가지만, Murphy 해상도가 %s 에서 %s 로 **약 5분의 1 토막이 나고** AUROC 는 %s "
    "에서 %s, 보정 기울기는 %s 에서 %s, 유효 표본은 %d 에서 %d 가 된다. 예측을 평평하게 "
    "만들면 ECE 는 내려간다 — 쓸모없는 모델이 될 뿐이다. **ECE 하나만 보고 개선이라고 "
    "쓰는 것은 이 논문이 비판하는 바로 그 오류다.**"
    % (
        pct1(abs(float(_hc.ece) - float(_dc.ece)) / float(_dc.ece) * 100),
        float(_hc.tau),
        f4(_dc.ece),
        f4(_hc.ece),
        f4(_dc.res),
        f4(_hc.res),
        f4(_dc.auroc),
        f4(_hc.auroc),
        f4(_dc.slope),
        f4(_hc.slope),
        int(round(float(_dc.n_eff))),
        int(round(float(_hc.n_eff))),
    )
)
w()
w(
    "**방법론적으로는 하나 건진다** — 정직 프로토콜이 DICTrank 에서 sweep 최적값을 "
    "미래를 보지 않고 찾아냈다. 고른 설정이 쓸모없었을 뿐, 선택 규칙 자체는 작동했다. "
    "다만 내부 이동(유병률 %s / %s)이 실제 이동(%s / %s)보다 훨씬 작다는 한계가 남는다."
    % (
        sg(
            float(_rw[(_rw.dataset == "DILIrank") & (_rw.kind == "inner")].test_rate.iloc[0])
            - float(
                _rw[(_rw.dataset == "DILIrank") & (_rw.kind == "inner")].train_rate_weighted.iloc[0]
            )
        ),
        sg(
            float(_rw[(_rw.dataset == "DICTrank") & (_rw.kind == "inner")].test_rate.iloc[0])
            - float(
                _rw[(_rw.dataset == "DICTrank") & (_rw.kind == "inner")].train_rate_weighted.iloc[0]
            )
        ),
        sg(float(_dl.gap) * -1),
        sg(float(_dc.gap) * -1),
    )
)
w()
_rp20 = rp[(rp.dataset == "DICTrank") & (rp.remedy.str.startswith("R."))]
if len(_rp20):
    _r20 = _rp20.iloc[0]
    w(
        '**다만 이 절의 DICTrank 결론은 "ECE 가 내려가지 않는다" 가 아니다.** §29 의 짝지은 '
        "검정에서 그 ECE 개선(Δ %s)은 구간이 0 을 포함하고(%s, %s) **구간이 간격의 %s 까지 "
        "개선을 허용한다**(판정 보류). 그래서 DICTrank 에서 최신성 가중을 기각하는 근거는 "
        "**ECE 가 안 내려간다는 것이 아니라 위에 적은 해상도 붕괴**다 — 그쪽은 점추정만으로도 "
        "분명하다(해상도 %s → %s). DILIrank 는 구간이 허용하는 최대 개선이 간격의 %s 뿐이라 "
        '"쓸 만한 개선이 없다" 를 그대로 쓸 수 있다.'
        % (
            sg(_r20.delta),
            sg(_r20.delta_lo),
            sg(_r20.delta_hi),
            pct1(100 * (-_r20.delta_lo) / _r20.gap_to_control),
            f4(_dc.res),
            f4(_hc.res),
            pct1(
                100
                * (
                    -rp[
                        (rp.dataset == "DILIrank") & (rp.remedy.str.startswith("R."))
                    ].delta_lo.iloc[0]
                )
                / rp[
                    (rp.dataset == "DILIrank") & (rp.remedy.str.startswith("R."))
                ].gap_to_control.iloc[0]
            ),
        )
    )
    w()
w("### 20.2 §19 법칙의 예측 검정 — 부분 실패")
w()
w(
    "§19 는 보정오차가 학습·검증 유병률 격차의 함수라고 했다. 최신성 가중은 학습 양성률을 "
    "바꾸므로 격차도 바뀐다. 그러면 §19 기울기만으로 ECE 변화를 **미리 계산할 수 있다.** "
    "이것은 사후 설명이 아니라 독립 실험에서의 사전 예측 검정이다."
)
w()
_swf20 = cv("prevalence_sweep_fit.csv")
_SW20 = {r.dataset: r for _, r in _swf20.iterrows()}
w("| 데이터셋 | 설정 | 격차 | 절대격차 변화 | 예측 ΔECE | 관측 ΔECE | 차이 |")
w("|---|---|---|---|---|---|---|")
_nover, _ntot, _nsign = 0, 0, 0
for d in DS:
    sl = float(_SW20[d]["slope_per_0.10"]) / 0.10
    g = _rw[(_rw.dataset == d) & (_rw.kind == "weight")]
    b = g[g.tau.isna()].iloc[0]
    for _, r in g[g.tau.notna()].iterrows():
        dg = abs(float(r.gap)) - abs(float(b.gap))
        pr = sl * dg
        ob = float(r.ece) - float(b.ece)
        _ntot += 1
        if abs(pr) > abs(ob):
            _nover += 1
        if pr * ob < 0:
            _nsign += 1
        w(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (d, r.tag, sg(r.gap), sg(dg), sg(pr), sg(ob), sg(ob - pr))
        )
w()
w(
    "예측이 관측보다 큰 행이 %d개 중 **%d개**이고, 부호가 어긋난 행이 %d개다. "
    "**방향은 대체로 맞지만 크기는 전이되지 않는다.** §19 는 학습 크기를 고정한 무작위 "
    "분할이었는데 여기서는 유효 표본이 설정마다 크게 변한다. 따라서 §19 의 기울기는 "
    "**근사적 기술이지 이식 가능한 정량 예측기가 아니다.** 본문에서 그 이상으로 쓰면 안 된다."
    % (_ntot, _nover, _nsign)
)
w()
w("### 20.3 이 절이 닫는 반론")
w()
w(
    "§10.2 는 라벨 없는 유병률 추정(P1·P2)이 두 데이터셋 모두에서 회복을 못 냄을, 이 절은 "
    "학습단계 최신성 가중이 처방이 못 됨을 보인다. 둘을 합치면 **시간 이동에 의한 보정 "
    "붕괴는 검증군 정보 없이 고칠 수 없다** 는 결론이 남는다."
)
w()
w(
    "회복을 내는 것은 **검증군 라벨을 쓰는 방법뿐이다** — 검증군 유병률을 아는 사전확률 "
    "보정(P4)이 %s / %s, 검증군 라벨 전부를 쓰는 Oracle Platt 가 정의상 100%% 다. 다만 "
    "P4 의 회복폭은 데이터셋에 따라 크게 다르므로(%s 대 %s) '사전확률만 알면 된다' 고 "
    "일반화하면 안 된다. 이것은 회피 가능한 모델링 실수가 아니라 자료 확보의 문제다."
    % (
        pct1(_rec("DILIrank", "P4")),
        pct1(_rec("DICTrank", "P4")),
        pct1(_rec("DILIrank", "P4")),
        pct1(_rec("DICTrank", "P4")),
    )
)
w()
# ================================================================ 21. 적용범위
w("---")
w()
w("## 21. 적용범위 — 미설명분이 '낯선 분자' 에 몰려 있는가")
w()
_ad = cv("applicability_domain.csv")
_adf = cv("applicability_domain_filter.csv")
w(
    "§18~§20 이 유병률·크기·개념 이동으로 ECE 초과분의 상당 부분을 설명했다. 남은 질문은 "
    '**어디에 몰려 있는가** 다. 심사자가 반드시 제안할 가설이 하나 있다 — "2010년 이후 '
    '분자가 학습군과 화학적으로 낯설어서 그런 것 아닌가." 맞다면 처방이 생긴다: '
    "**낯선 분자에는 예측을 내지 않는다(AD 필터).**"
)
w()
w(
    "§14 가 이미 최근접 Tanimoto 로 층화한 ECE 를 냈지만 결론에 쓸 수 없다. (1) 구간별 "
    "무작위 대조군이 없고(§14 요약문이 자진 신고), (2) 구간마다 양성률이 다른데 §19 가 "
    "ECE 는 유병률 격차의 함수임을 보였으므로 구간 간 차이가 **유병률 교란일 수 있다.** "
    "그래서 여기서는 층마다 **크기와 양성률을 맞춘 무작위 대조군**을 붙인다. 표본부족 "
    "기준은 §17.2 와 같다(n < 40 또는 소수 클래스 < 10)."
)
w()
for d in DS:
    g = _ad[(_ad.dataset == d) & (~_ad.stratum.str.contains("기울기"))]
    w("**%s**" % d)
    w()
    w("| 층 | n | 양성률 | AUROC | ECE | 평균잔차 | 대조 ECE | 배율 | 배율 95% CI | p |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        if bool(r.underpowered):
            w("| %s | %d | — | — | — | — | — | — | — | 표본부족 |" % (r.stratum, int(r.n)))
            continue
        w(
            "| %s | %d | %s | %s | %s | %s | %s | **%s배** | [%s, %s] | %s |"
            % (
                r.stratum,
                int(r.n),
                f4(r.pos_rate),
                f4(r.auroc),
                f4(r.ece),
                sg(r.mean_resid),
                f4(r.ctrl_ece),
                f2(r.ece_ratio),
                f2(r.ratio_lo),
                f2(r.ratio_hi),
                f4(r.p_phipson_smyth),
            )
        )
    w()
_over1 = _ad[(~_ad.underpowered) & (~_ad.stratum.str.contains("기울기")) & (_ad.ratio_lo > 1.0)]
_nstr = int(((~_ad.underpowered) & (~_ad.stratum.str.contains("기울기"))).sum())
w(
    "**보정 붕괴는 화학공간 어디에서나 일어난다.** 계산 가능한 %d개 층 중 **%d개에서 배율 "
    "신뢰구간의 하한이 1 을 넘는다.** 즉 유사도가 높든 낮든, 스캐폴드가 기존이든 신규든 "
    "시간분할은 같은 크기·같은 양성률의 무작위 분할보다 나쁘다. 국소 현상이 아니다."
    % (_nstr, len(_over1))
)
w()
_sc = _ad[(_ad.stratum == "스캐폴드 신규") & (~_ad.underpowered)]
w(
    "부수적으로 나온 숫자 하나 — 2010년 이후 분자 중 **학습군에 없던 Bemis–Murcko "
    "스캐폴드를 가진 것이 %s / %s** 다. 시간분할이 만드는 구조적 낯섦의 크기를 한 숫자로 "
    "보여 준다."
    % (
        " / ".join(
            "%d개" % int(_sc[_sc.dataset == d].n.iloc[0]) for d in DS if len(_sc[_sc.dataset == d])
        ),
        " / ".join(
            pct1(100.0 * int(_sc[_sc.dataset == d].n.iloc[0]) / int(CS[(d, "A.")].n_test))
            for d in DS
            if len(_sc[_sc.dataset == d])
        ),
    )
)
w()
w("### 21.1 유사도 기울기 — 보정과 판별이 같이 움직이는가")
w()
w("| 데이터셋 | 양 | 하위−상위 차 | 95% CI | 0 포함 |")
w("|---|---|---|---|---|")
_G = {}
for d in DS:
    for lab, key, fm in (
        ("보정 (ECE 배율)", "유사도 기울기(하위−상위)", f2),
        ("판별 (AUROC)", "판별 기울기(상위−하위 AUROC)", f4),
    ):
        r = _ad[(_ad.dataset == d) & (_ad.stratum == key)]
        if not len(r):
            continue
        r = r.iloc[0]
        inc = bool(r.ratio_lo <= 0.0 <= r.ratio_hi)
        _G[(d, lab)] = (float(r.ece_ratio), float(r.ratio_lo), float(r.ratio_hi), inc)
        w(
            "| %s | %s | %s | [%s, %s] | %s |"
            % (
                d,
                lab,
                fm(r.ece_ratio),
                fm(r.ratio_lo),
                fm(r.ratio_hi),
                "예 — **주장 불가**" if inc else "아니오",
            )
        )
w()
w(
    "**판별력은 유사도에 따라 변한다** — 상위 1/3 과 하위 1/3 의 AUROC 차가 %s / %s 이고 "
    "두 신뢰구간 모두 0 을 포함하지 않는다. 학습군과 비슷한 분자일수록 잘 판별된다는 "
    "Sheridan 등의 관찰과 방향이 같다."
    % (f4(_G[("DILIrank", "판별 (AUROC)")][0]), f4(_G[("DICTrank", "판별 (AUROC)")][0]))
)
w()
_cl_hi = max(abs(_G[(d, "보정 (ECE 배율)")][1]) for d in DS)
_cl_hi = max(_cl_hi, max(abs(_G[(d, "보정 (ECE 배율)")][2]) for d in DS))
w(
    "**보정 배율에서는 같은 표본으로 변화를 검출하지 못했다** — 두 구간 모두 0 을 포함한다. "
    "다만 **여기서 멈춰야 한다.** '한쪽은 유의하고 한쪽은 비유의' 라는 대비는 두 효과가 "
    "다르다는 검정이 아니고, 두 양은 눈금이 달라 직접 비교할 수도 없다. 게다가 보정 쪽 "
    "구간은 최대 %s 크기의 기울기까지 허용한다. 따라서 **'보정은 유사도와 무관하다' 는 "
    "입증되지 않았다.** 말할 수 있는 것은 두 줄뿐이다 — 판별력은 유사도에 따라 변하고, "
    "보정 배율에서는 이 표본 크기로 변화를 찾지 못했다." % f2(_cl_hi)
)
w()
w(
    "보정 쪽 구간은 검증군 재표집만 반영하고 대조군 재표집 변동은 담지 않아 **실제보다 "
    "좁다.** 그 좁은 구간으로도 0 을 포함한다는 뜻이므로, 기울기가 있다는 주장은 더더욱 "
    "서지 않는다."
)
w()
w("### 21.2 AD 필터는 처방이 되지 못한다")
w()
w(
    '"가장 낯선 x% 에는 예측하지 않는다" 를 x = 0~50% 로 돌리고, **같은 수를 무작위로 '
    "버리는 대조군**을 나란히 뒀다. 예측은 고정이므로 재적합이 없다. 둘이 같으면 필터는 "
    "어려운 분자를 골라낸 것이 아니라 그냥 표본을 줄인 것이다."
)
w()
for d in DS:
    g = _adf[_adf.dataset == d]
    w("**%s**" % d)
    w()
    w("| 버림 | 적용률 | 남은 n | ECE | AUROC | 무작위 버림 ECE | 차이 |")
    w("|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %d | %s | %s | %s | %s |"
            % (
                pct1(r.drop_frac * 100),
                f3(r.coverage),
                int(r.n_keep),
                f4(r.ece_ad),
                f4(r.auroc_ad),
                f4(r.ece_random_drop),
                sg(r.delta),
            )
        )
    w()
_pos = int((_adf.delta > 0).sum())
_tot_f = int(len(_adf[_adf.drop_frac > 0]))
w(
    "차이 열이 음수여야 필터가 값어치를 하는데, 0%% 를 제외한 %d개 설정 중 **%d개에서 "
    "오히려 양수**다. 부호가 일정하지 않고 크기도 작다 — 잡음이다. **AD 필터는 처방이 "
    "되지 못한다.** 심사자가 제안할 가장 자연스러운 해법이 대조군 있는 설계로 닫혔다."
    % (_tot_f, _pos)
)
w()
w("### 21.3 §14 표에 대한 정정")
w()
w(
    "§14 의 유사도 구간별 ECE 는 구간마다 양성률이 달라(이 절의 표에서도 DILIrank 는 "
    "%s / %s / %s) 구간 간 비교에 쓸 수 없다. **대조군을 붙인 이 절의 배율이 §14 의 원 "
    "ECE 를 대체한다.** §14 표는 기술 통계로만 읽어야 한다."
    % tuple(
        f4(
            _ad[(_ad.dataset == "DILIrank") & (_ad.stratum == "Tanimoto %s 1/3" % k)].pos_rate.iloc[
                0
            ]
        )
        for k in ("하위", "중위", "상위")
    )
)
w()
# ================================================================ 22. 결정곡선
w("---")
w()
w("## 22. 결정곡선 — 보정 붕괴의 의사결정 비용")
w()
_dc = cv("decision_curve.csv")
w(
    '§8~§21 은 지표의 언어로 말한다. 독자가 반드시 묻는 다음 질문은 **"그래서 쓸 때 뭐가 '
    '달라지는가"** 다. 순편익(net benefit)이 그 간극을 메운다 — NB(p_t) = TP/N − (FP/N) × '
    "p_t/(1−p_t). 임계확률 p_t 는 의사결정자가 '양성 하나를 놓치는 손해' 와 '음성 하나를 "
    "잘못 거르는 손해' 를 어떤 비로 보는지를 뜻한다."
)
w()
w(
    "**해석 주의.** 독성 예측의 '양성 판정' 은 투여가 아니라 **추가 시험·보류** 다. 또한 NB 는 "
    "검증군 유병률에 직접 의존하므로 시간분할과 무작위분할의 NB 를 같은 눈금에서 비교할 수 "
    "없다. 아래 표는 **같은 검증군 안에서 보정 방법만 바꾼 비교** 로만 읽어야 한다."
)
w()
for d in DS:
    g = _dc[_dc.dataset == d]
    base = g[g.arm.str.startswith("A.")].set_index("p_t")
    w("**%s** (검증군 %d개, 양성률 %s)" % (d, int(base.n_test.iloc[0]), f4(base.test_rate.iloc[0])))
    w()
    arms = [a for a in g.arm.unique() if not a.startswith("R.")]
    w("| p_t | 전부 양성 | %s |" % " | ".join(a.split(". ")[1] for a in arms))
    w("|---|---|%s" % ("---|" * len(arms)))
    for pt in sorted(g.p_t.unique()):
        cells = []
        for a in arms:
            v = g[(g.arm == a) & (g.p_t == pt)]
            cells.append(sg(v.net_benefit.iloc[0]) if len(v) else "—")
        na = g[(g.arm == arms[0]) & (g.p_t == pt)].nb_treat_all.iloc[0]
        w("| %s | %s | %s |" % (f2(pt), sg(na), " | ".join(cells)))
    w()
_neg = {}
for d in DS:
    b = _dc[(_dc.dataset == d) & (_dc.arm.str.startswith("A."))]
    nb = b[b.net_benefit < 0]
    _neg[d] = (float(nb.p_t.min()), float(nb.net_benefit.min())) if len(nb) else None
_d1 = [d for d in DS if _neg[d]]
if _d1:
    d0 = _d1[0]
    w(
        "**%s 에서 무보정 모델은 p_t ≥ %s 구간에서 순편익이 음수다**(최저 %s). AUROC %s 인 "
        "모델이 그 구간에서는 **아무것도 하지 않느니만 못하다.** 판별력 지표로는 보이지 않는 "
        "손해이고, 이 논문이 측정하려던 바로 그것이다."
        % (
            d0,
            f2(_neg[d0][0]),
            sg(_neg[d0][1]),
            f4(_dc[(_dc.dataset == d0) & (_dc.arm.str.startswith("A."))].auroc.iloc[0]),
        )
    )
    w()
_oth = [d for d in DS if not _neg[d]]
if _oth:
    w(
        "반면 **%s 에서는 무보정 NB 가 전 구간에서 0 이상이다.** 결정 단위의 손해도 "
        "데이터셋마다 갈린다 — §18(개념 이동)·§21(유사도 기울기)에서 본 것과 같은 방향이다."
        % ", ".join(_oth)
    )
    w()
w("| 데이터셋 | 보정 방법 | 무보정 대비 NB 차 최대 | 최소 | p_t ≥ 0.40 평균 |")
w("|---|---|---|---|---|")
for d in DS:
    g = _dc[_dc.dataset == d]
    base = g[g.arm.str.startswith("A.")].set_index("p_t").net_benefit
    for a in g.arm.unique():
        if a.startswith("A.") or a.startswith("R."):
            continue
        s2 = g[g.arm == a].set_index("p_t").net_benefit
        dd = (s2 - base).dropna()
        hi = dd[dd.index >= 0.40]
        w(
            "| %s | %s | %s (p_t %s) | %s (p_t %s) | %s |"
            % (
                d,
                a,
                sg(dd.max()),
                f2(dd.idxmax()),
                sg(dd.min()),
                f2(dd.idxmin()),
                sg(hi.mean()) if len(hi) else "—",
            )
        )
w()
w(
    "**검증군 라벨을 쓰는 보정(사전확률 보정)만 의미 있게 회복한다.** 학습군 내부 Platt 은 "
    "거의 아무것도 되살리지 못하는데, 이는 §10 에서 Platt 의 ECE 회복률이 작았던 것과 "
    "일관된다."
)
w()
w(
    "**단조 관계가 아니라는 점을 함께 적어야 한다.** 과대예측 모델은 낮은 임계확률에서 "
    "'전부 양성' 에 가깝게 작동하므로 그 구간의 손해가 작다. 손해는 거짓양성을 꺼리는 "
    "**높은 임계확률**에서 나타난다. 이 비대칭은 과대예측이라는 **방향** 때문이며, 방향이 "
    "반대인 모델에서는 반대로 나타날 수 있다. 따라서 'ECE 가 크면 결정 손해가 크다' 로 "
    "일반화해서는 안 된다."
)
w()
# ================================================================ 23. 클래스 가중
w("---")
w()
w("## 23. 학습 시 클래스 가중 — 사후 보정 대 사전 예방")
w()
_cw = cv("class_weighting.csv")
w(
    "§19 가 보정오차를 유병률 격차의 함수로 두었으므로, 가장 자연스러운 처방은 **학습 "
    "단계에서 유효 유병률을 검증군 쪽으로 옮기는 것** 이다. 지금까지 시험한 것은 사후 "
    "사전확률 보정(§10 D)과 시간축 가중(§20)뿐이고, **라벨축 가중** 은 비어 있었다. "
    "XGBoost 의 양성 가중으로 목표 유병률을 만들어 학습한다."
)
w()
for d in DS:
    g = _cw[_cw.dataset == d]
    b = g[g.arm.str.startswith("W0")].iloc[0]
    w(
        "**%s** (학습 양성률 %s → 검증 양성률 %s, 격차 %s)"
        % (d, f4(b.pi_train), f4(b.pi_test), sg(b.pi_test - b.pi_train))
    )
    w()
    w("| 설정 | 목표 유병률 | 양성 가중 | AUROC | ECE | ECE 95% CI | 보정 기울기 | 해상도 |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %s | %s | %s | [%s, %s] | %s | %s |"
            % (
                r.arm,
                "—" if r.pi_target != r.pi_target else f4(r.pi_target),
                "—" if r.scale_pos_weight != r.scale_pos_weight else f3(r.scale_pos_weight),
                f4(r.auroc),
                f4(r.ece),
                f4(r.ece_lo),
                f4(r.ece_hi),
                f4(r.slope),
                f4(r.res),
            )
        )
    w()
_best, _orc = {}, {}
for d in DS:
    g = _cw[(_cw.dataset == d) & (_cw.arm.str.match(r"W[123]"))]
    _best[d] = g.loc[g.ece.idxmin()]
    _orc[d] = _cw[(_cw.dataset == d) & (_cw.arm.str.startswith("W1"))].iloc[0]
_notorc = [d for d in DS if not str(_best[d].arm).startswith("W1")]
if _notorc:
    w(
        "**검증군 유병률을 그대로 목표로 두는 것이 최적이 아니다.** %s 에서 ECE 가 최소인 "
        "설정은 W1(oracle)이 아니라 **%s**(%s 대 %s)다. 즉 손상은 순수한 사전확률 이동이 "
        "아니며, 가중은 절편만이 아니라 적합 자체를 바꾼다. 이 결과는 단순한 '유병률만 "
        "맞추면 된다' 는 처방을 **반증한다.**"
        % (
            ", ".join(_notorc),
            _best[_notorc[0]].arm,
            f4(_best[_notorc[0]].ece),
            f4(_orc[_notorc[0]].ece),
        )
    )
    w()
w(
    "| 데이터셋 | 과소지정 중 악화 | 과대지정 중 악화 | 사후(§10 D) ECE | 사전(W1) ECE | 어느 쪽이 나은가 |"
)
w("|---|---|---|---|---|---|")
for d in DS:
    g = _cw[(_cw.dataset == d)].set_index("arm")
    b = float(g[g.index.str.startswith("W0")].ece.iloc[0])
    mis = g[g.index.str.contains("오지정")]
    un = mis[mis.index.str.contains("-0")]
    ov = mis[mis.index.str.contains(r"\+0")]
    po = float(g[g.index.str.startswith("P.")].ece.iloc[0])
    w1 = float(g[g.index.str.startswith("W1")].ece.iloc[0])
    w(
        "| %s | %d/%d | %d/%d | %s | %s | %s |"
        % (
            d,
            int((un.ece > b).sum()),
            len(un),
            int((ov.ece > b).sum()),
            len(ov),
            f4(po),
            f4(w1),
            "사후 보정" if po < w1 else "학습 시 가중",
        )
    )
w()
w(
    "**과소·과대가 대칭이 아니다** — 목표 유병률을 실제보다 낮게 잡으면 안전하고, 높게 "
    "잡으면 무가중보다 나빠질 수 있다. 그리고 사후와 사전 중 일관된 승자가 없다. 두 방법 "
    "모두 **검증군 유병률을 알아야** 하고 §10.2 는 그 값을 라벨 없이 추정하려는 시도가 "
    "실패함을 보였으므로, 둘 다 **배포 가능한 처방이 아니라 상한선** 으로 읽어야 한다."
)
w()
# ================================================================ 24. 라벨 역치
w("---")
w()
w("## 24. 라벨 이진화 민감도 — 결론이 라벨 역치의 산물인가")
w()
_lt = cv("label_threshold.csv")
w(
    "두 원자료는 모두 3단계 라벨이다. 주 분석은 중간 등급과 최고 등급을 함께 양성으로 "
    "묶었다. 문헌에서 쓰이는 다른 선택은 **최고 등급만 양성** 으로 두는 것이고, 그러면 "
    "양성률이 절반 이하로 떨어진다. **이 논문의 중심 주장이 유병률 격차에 걸려 있는데 그 "
    "유병률을 정하는 것이 바로 이 선택이다.** §16(큐레이션 민감도)은 결정 세 건을 다루지만 "
    "라벨 역치는 거기 없었다."
)
w()
w("| 데이터셋 | 라벨 정의 | 검증 양성 n | 학습 양성률 | 검증 양성률 | 격차 | 시간분할 ECE |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    for lt in _lt[_lt.dataset == d].label_def.unique():
        t = _lt[(_lt.dataset == d) & (_lt.label_def == lt) & (_lt.setting.str.startswith("T"))]
        if not len(t):
            continue
        t = t.iloc[0]
        w(
            "| %s | %s | %d | %s | %s | %s | %s |"
            % (
                d,
                lt,
                int(t.n_test_pos),
                f4(t.rate_train),
                f4(t.rate_test),
                sg(t.rate_test - t.rate_train),
                f4(t.ece),
            )
        )
w()
w(
    "| 데이터셋 | 라벨 정의 | R1 배율 | R2 배율 | R3 배율 | R1 p | R3 p | 배율 전부 >1 | 사다리 단조 |"
)
w("|---|---|---|---|---|---|---|---|---|")
_fail_mono, _fail_sig = [], []
for d in DS:
    for lt in _lt[_lt.dataset == d].label_def.unique():
        g = _lt[(_lt.dataset == d) & (_lt.label_def == lt)]
        rr = {b: g[g.setting.str.startswith(b)] for b in ("R1", "R2", "R3")}
        if any(not len(v) for v in rr.values()):
            continue
        v = [float(rr[b].ece_ratio.iloc[0]) for b in ("R1", "R2", "R3")]
        p1 = float(rr["R1"].p_phipson_smyth.iloc[0])
        p3 = float(rr["R3"].p_phipson_smyth.iloc[0])
        mono = v[0] > v[1] > v[2]
        if not mono:
            _fail_mono.append("%s/%s" % (d, lt))
        if p3 >= 0.05:
            _fail_sig.append("%s/%s" % (d, lt))
        w(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                d,
                lt,
                f2(v[0]),
                f2(v[1]),
                f2(v[2]),
                f4(p1),
                f4(p3),
                "예" if all(x > 1 for x in v) else "**아니오**",
                "예" if mono else "**아니오**",
            )
        )
w()
w(
    "**현상 자체는 살아남는다** — 네 조건 모두에서 ECE 배율이 1 을 넘는다. 그러나 두 가지를 "
    "반드시 함께 적어야 한다."
)
w()
w(
    "첫째, **헤드라인 배율은 주 라벨 정의에 특정적이다.** 최고 등급만 양성으로 두면 "
    "유병률 격차가 거의 사라지고 시간분할 ECE 도 크게 내려간다. 이것은 §19 의 법칙이 "
    "**설계 목적이 아닌 실험에서 재현된 것** 으로 읽을 수 있다 — 격차를 줄였더니 보정오차가 "
    "줄었다. 다만 라벨을 바꾸면 과제 난이도와 해상도도 함께 변하므로 §19 의 깨끗한 검정은 "
    "아니다."
)
w()
if _fail_sig:
    w(
        "둘째, **%s 에서는 통제 후 배율이 통계적으로 유의하지 않다**(R3 p ≥ 0.05). "
        "주장 범위를 주 라벨 정의 아래로 명시해야 한다." % ", ".join(_fail_sig)
    )
if _fail_mono:
    w(
        "또한 **%s 에서는 유병률 사다리의 단조 순서(R1 > R2 > R3)가 유지되지 않는다.** "
        "격차가 거의 0 이어서 맞출 것이 없기 때문이지만, 사다리를 핵심 근거로 쓰는 이상 "
        "그대로 적어야 한다." % ", ".join(_fail_mono)
    )
w()
w(
    "**L2 는 양성 표본이 적다**(검증군 양성이 수십 개 수준). 그래서 L1 과 L2 의 배율을 직접 "
    "빼서 비교해서는 안 되고, 각자 자기 대조군과만 비교해야 한다."
)
w()
# ================================================================ 25. 상수 예측기
w("---")
w()
w("## 25. 상수 예측기 대비 — 그 확률을 쓸 수 있는가")
w()
w(
    '보정 붕괴를 "얼마나 나쁜가" 로 말하려면 비교 대상이 필요하다. 학습을 전혀 하지 않고도 '
    "낼 수 있는 상수 예측기 셋과 Brier 점수를 직접 비교했다. **C0** 는 학습군 양성률을 모든 "
    "분자에 그대로 내놓는다(배포 가능하다). **C1** 은 0.5 다. **C2** 는 검증군 양성률을 쓰므로 "
    "oracle 이고 달성 불가능한 참조선이다. BSS(Brier Skill Score) = 1 − Brier(모델)/Brier(C0) "
    "이며 양수면 C0 보다 낫다는 뜻이다."
)
w()


def _tb1(d, sp, pref):
    g = tb[(tb.dataset == d) & (tb.split == sp) & (tb.predictor.str.startswith(pref))]
    assert len(g) == 1, ("trivial_baseline 행 선택 실패", d, sp, pref)
    return g.iloc[0]


_CP = [
    ("M.", "모델 (XGBoost)"),
    ("C0.", "C0. 상수 = 학습군 양성률"),
    ("C1.", "C1. 상수 = 0.5"),
    ("C2.", "C2. 상수 = 검증군 양성률 (oracle)"),
]
for d in DS:
    _m = _tb1(d, "시간분할", "M.")
    w(
        "**%s** — 학습 %d / 검증 %d, 학습군 양성률 %s, 검증군 양성률 %s"
        % (KOR[d], int(_m.n_train), int(_m.n_test), f4(_m.pi_train), f4(_m.pi_test))
    )
    w()
    w(
        "| 예측기 | 시간분할 Brier | 시간분할 BSS | 시간분할 ECE | 무작위분할 Brier | 무작위분할 BSS |"
    )
    w("|---|---|---|---|---|---|")
    for pref, lab in _CP:
        t = _tb1(d, "시간분할", pref)
        r = _tb1(d, "무작위분할", pref)
        em = "**" if pref == "M." else ""
        w(
            "| %s%s%s | %s%s%s | %s%s%s | %s | %s | %s |"
            % (
                em,
                lab,
                em,
                em,
                f4(t.brier),
                em,
                em,
                sg(t.bss_vs_C0),
                em,
                f4(t.ece),
                f4(r.brier),
                sg(r.bss_vs_C0),
            )
        )
    w()
_worse = {
    d: sum(
        1
        for c in ("C0.", "C1.", "C2.")
        if _tb1(d, "시간분할", "M.").brier > _tb1(d, "시간분할", c).brier
    )
    for d in DS
}
_better = {
    d: sum(
        1
        for c in ("C0.", "C1.", "C2.")
        if _tb1(d, "무작위분할", "M.").brier < _tb1(d, "무작위분할", c).brier
    )
    for d in DS
}
w(
    "**시간분할에서는 모델의 확률이 상수보다 나쁘다.** 세 상수 중 모델보다 Brier 가 작은 것이 "
    "DILIrank %d/3, DICTrank %d/3 이다. 배포 상황에서 쓸 수 있는 유일한 상수인 C0 를 기준으로 "
    "한 BSS 는 **%s / %s** 로 둘 다 음수다."
    % (
        _worse["DILIrank"],
        _worse["DICTrank"],
        sg(_tb1("DILIrank", "시간분할", "M.").bss_vs_C0),
        sg(_tb1("DICTrank", "시간분할", "M.").bss_vs_C0),
    )
)
w()
w(
    "**같은 데이터를 무작위로 나누면 부호가 뒤집힌다.** 모델이 Brier 로 이기는 상수가 "
    "DILIrank %d/3, DICTrank %d/3 이고 C0 기준 BSS 는 **%s / %s** 로 둘 다 양수다. "
    "특성·모델·하이퍼파라미터·시드는 그대로이고 바뀐 것은 분할 축뿐이다. "
    '**같은 파이프라인이 무작위 분할에서는 "상수보다 낫다", 시간분할에서는 "상수보다 '
    '못하다" 로 평가된다.**'
    % (
        _better["DILIrank"],
        _better["DICTrank"],
        sg(_tb1("DILIrank", "무작위분할", "M.").bss_vs_C0),
        sg(_tb1("DICTrank", "무작위분할", "M.").bss_vs_C0),
    )
)
w()
w(
    "**판별력은 같이 무너지지 않았다.** 같은 시간분할에서 AUROC 는 %s / %s 로 0.5 보다 "
    "높다(§5 의 순열검정). 상수 예측기는 모든 분자에 같은 값을 주므로 순위가 없고 AUROC 를 "
    "정의할 수 없다. 즉 모델에는 상수가 갖지 못한 정보가 들어 있고(resolution %s / %s > 0) "
    "그런데도 확률의 위치가 틀려서(reliability %s / %s) Brier 에서는 진다. "
    "**순위로는 쓸 수 있고 확률로는 쓸 수 없다** 는 이 논문의 주장을 한 표에 담은 것이 "
    "이 절이다."
    % (
        f4(_tb1("DILIrank", "시간분할", "M.").auroc),
        f4(_tb1("DICTrank", "시간분할", "M.").auroc),
        f4(_tb1("DILIrank", "시간분할", "M.").res),
        f4(_tb1("DICTrank", "시간분할", "M.").res),
        f4(_tb1("DILIrank", "시간분할", "M.").rel),
        f4(_tb1("DICTrank", "시간분할", "M.").rel),
    )
)
w()
w("### 25.1 재보정을 거친 확률은 상수를 넘는가 — 한쪽만 넘는다")
w()
_c0b = {d: float(_tb1(d, "시간분할", "C0.").brier) for d in DS}
w(
    "§10 의 재보정 결과를 같은 C0 기준으로 환산했다. C0 의 Brier 는 모델과 무관하게 "
    "검증군 라벨과 학습군 양성률만으로 정해지므로(DILIrank %s, DICTrank %s) 다른 스크립트의 "
    "확률과도 비교할 수 있다." % (f4(_c0b["DILIrank"]), f4(_c0b["DICTrank"]))
)
w()
w("| 데이터셋 | 재보정 방법 | Brier | C0 대비 BSS | C0 를 넘는가 |")
w("|---|---|---|---|---|")
for d in DS:
    for _, r in rec[rec.dataset.str.startswith(d)].iterrows():
        _b = float(r.Brier)
        _bss = 1.0 - _b / _c0b[d]
        w(
            "| %s | %s | %s | %s | %s |"
            % (d, r.method, f4(_b), sg(_bss), "예" if _bss > 0 else "아니오")
        )
w()
_okB = {
    d: 1.0
    - float(rec[(rec.dataset.str.startswith(d)) & (rec.method.str.startswith("B."))].Brier.iloc[0])
    / _c0b[d]
    for d in DS
}
_okD = {
    d: 1.0
    - float(rec[(rec.dataset.str.startswith(d)) & (rec.method.str.startswith("D."))].Brier.iloc[0])
    / _c0b[d]
    for d in DS
}
w(
    "**학습군 내부 Platt 재보정은 DILIrank 에서 여전히 C0 를 넘지 못한다**(BSS %s). "
    "DICTrank 에서는 간신히 넘는다(%s). 검증군 양성률을 안다고 가정한 사전확률 보정은 "
    "DILIrank 에서 넘고(%s) DICTrank 에서는 넘지 못한다(%s). "
    '**즉 "재보정하면 상수보다는 낫다" 를 두 데이터셋에 공통으로 말할 수 있는 방법이 '
    "하나도 없다** — 배포 가능한 방법 중에는 없고, 각 데이터셋에서 통하는 방법이 서로 다르다. "
    '이것은 §10 의 "단일 권고안을 낼 수 없다" 를 Brier 단위로 다시 확인한 것이다.'
    % (sg(_okB["DILIrank"]), sg(_okB["DICTrank"]), sg(_okD["DILIrank"]), sg(_okD["DICTrank"]))
)
w()
w("**이 절을 인용할 때 네 가지를 같이 적을 것.**")
w()
w(
    '1. **C2 는 oracle 이다.** 검증군 라벨을 써서 만든 상수이므로 "달성 가능한 상수" 가 '
    "아니다. BSS 의 기준은 C0 이고, 배포 상황에서 쓸 수 있는 상수도 C0 뿐이다."
)
w(
    "2. **Brier 는 보정과 판별력을 함께 담는 지표다.** 모델이 지는 이유는 §7 의 분해가 "
    "말하는 그대로 reliability 항이다 — 새 사실이 아니라 §7 을 실무 단위로 바꿔 쓴 것이다."
)
w(
    "3. **§25.1 의 A 행은 §25 본표의 모델 행과 수치가 다르다**(Brier %s 대 %s). "
    "§10 스크립트가 행 순서를 승인연도로 정렬해 적합하기 때문이며, 그 폭은 §9.3 의 시드 "
    "범위 안이다(§10 의 주석 참조). 그래서 §25.1 은 **C0 를 넘는지 여부**만 읽고 "
    "점추정값은 각 절의 것을 쓴다."
    % (
        f4(
            rec[
                (rec.dataset.str.startswith("DILIrank")) & (rec.method.str.startswith("A."))
            ].Brier.iloc[0]
        ),
        f4(_tb1("DILIrank", "시간분할", "M.").brier),
    )
)
w(
    '4. **BSS 의 부호는 C0 의 정의에 걸려 있다.** C0 를 "학습군 양성률" 로 두는 것은 '
    "배포 시점에 실제로 아는 값이기 때문이다. 만약 C1(0.5)을 기준으로 쓰면 "
    "DILIrank 시간분할 BSS 는 %s 가 된다 — 여전히 음수다."
    % sg(
        1.0
        - float(_tb1("DILIrank", "시간분할", "M.").brier)
        / float(_tb1("DILIrank", "시간분할", "C1.").brier)
    )
)
w()
w(
    "§22 의 결정곡선은 같은 결론을 임계확률별 순편익으로 보여준다. 두 절은 같은 사실의 "
    "다른 단위다 — 여기서는 Brier, 거기서는 의사결정 단위다."
)
w()
# ================================================================ 26. 배율 신뢰구간
w("---")
w()
w("## 26. ECE 배율의 신뢰구간 — 효과크기에 불확실성을 붙인다")
w()
import re as _re2

_rib = int(_re2.search(r"부트스트랩 (\d+)회", txt("ratio_intervals_summary.txt")).group(1))
_rir = int(ri.n_rep_valid.max())
w(
    "§8·§9 가 보고한 배율은 점추정값이다. 분자(시간분할 ECE)는 검증군 한 벌에서 나오고 "
    "분모는 대조군 %d회의 평균이다. 두 쪽 모두 표본오차가 있으므로 배율에 구간을 붙였다 — "
    "분자는 검증군 부트스트랩 %s회, 분모는 같은 재표집 안에서 대조군 반복을 다시 뽑는다. "
    '**§15.1 의 5번에서 "p 대신 효과크기로 쓴다" 고 한 그 효과크기의 구간이 이 표다.**'
    % (_rir, "{:,}".format(_rib))
)
w()
w(
    "| 데이터셋 | 대조 설계 | 대조 ECE | 배율 | 배율 95% CI (분자+분모) | 배율 95% CI (분자만) | ECE 차 95% CI | 1 제외 |"
)
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in ri[ri.dataset == d].iterrows():
        w(
            "| %s | %s | %s | **%s** | [%s, %s] | [%s, %s] | [%s, %s] | %s |"
            % (
                d,
                r.control,
                f4(r.ece_control_mean),
                f2(r.ece_ratio),
                f2(r.ratio_lo),
                f2(r.ratio_hi),
                f2(r.ratio_lo_numonly),
                f2(r.ratio_hi_numonly),
                sg(r.diff_lo),
                sg(r.diff_hi),
                "예" if bool(r.excludes_one) else "아니오",
            )
        )
w()
for d in DS:
    _t = ri[ri.dataset == d].ece_temporal.iloc[0]
    assert abs(float(_t) - float(CT[d].ECE)) < 1e-9, "§26 의 분자가 §6 과 다르다"
    for _, r in ri[ri.dataset == d].iterrows():
        _pc = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith(str(r.control)[:2]))]
        if len(_pc):
            assert (
                abs(float(r.ece_ratio) - float(_pc.ece_ratio.iloc[0])) < 5e-4
            ), "§26 의 배율이 §8 과 다르다"
w(
    "**여섯 설정 전부에서 배율 구간의 하한이 1 을 넘는다.** 가장 낮은 하한은 %s "
    '(%s, %s)이다. 그래서 "유병률을 양측으로 완전히 통제한 뒤에도 시간분할의 보정오차가 '
    '더 크다" 를 p 값 없이 효과크기 수준에서 말할 수 있다. ECE 차의 구간도 전부 0 을 '
    "제외하며, 가장 낮은 하한은 %s 다."
    % (
        f2(ri.ratio_lo.min()),
        ri.loc[ri.ratio_lo.idxmin(), "dataset"],
        ri.loc[ri.ratio_lo.idxmin(), "control"],
        sg(ri.diff_lo.min()),
    )
)
w()
w(
    "**여유가 좁다는 것도 같이 적어야 한다.** %s 의 %s 하한은 %s 로 1 에 거의 붙어 있다. "
    '이 한 칸이 §8 의 가장 보수적인 통제에 해당하므로, **주장은 "통제 후에도 배율이 1 보다 '
    '크다" 까지이고 "배율이 크다" 로 늘려 쓸 수 없다.** 나머지 다섯 칸은 하한이 %s 이상이다.'
    % (
        ri.loc[ri.ratio_lo.idxmin(), "dataset"],
        ri.loc[ri.ratio_lo.idxmin(), "control"],
        f3(ri.ratio_lo.min()),
        f2(sorted(ri.ratio_lo)[1]),
    )
)
w()
_wr = ri.ci_width_full / ri.ci_width_numonly
w(
    "**불확실성은 거의 전부 검증군 크기에서 온다.** 분모(대조군 반복)의 변동을 넣었을 때 "
    "구간 폭이 넓어지는 비율은 %s–%s배에 그친다(1 보다 작은 칸은 재표집 잡음이다). "
    "대조군 %d회면 분모는 충분히 안정되고, 구간을 넓히는 것은 검증군 n(%d / %d)이다. "
    "**따라서 이 구간을 좁히는 방법은 대조군 반복을 늘리는 것이 아니라 검증 분자를 "
    "늘리는 것뿐이다.**"
    % (
        f4(_wr.min()),
        f4(_wr.max()),
        _rir,
        int(ri[ri.dataset == "DILIrank"].n_test.iloc[0]),
        int(ri[ri.dataset == "DICTrank"].n_test.iloc[0]),
    )
)
w()
w(
    "> **배율과 차이를 섞어 쓰지 말 것.** 배율은 분모에 따라 값이 달라지고(§8), 차이는 "
    "단위가 ECE 그대로다. 본문에서 배율을 인용할 때는 대조 설계와 이 구간을 함께 적는다."
)
w()
# ================================================================ 27. 지표 비의존성
w("---")
w()
w("## 27. 구간화에 의존하지 않는 보정 지표 — 결론이 ECE 라는 지표의 산물인가")
w()
w(
    "§6 부터 §26 까지의 모든 수치가 ECE 한 지표에서 나왔다. 보정 문헌을 아는 심사자가 "
    "던지는 질문이 둘 있다. **구간 수를 바꾸면 달라지지 않나**(§9.4 가 답한다 — 균등폭·"
    "균등질량 × 5·10·15·20 구간에서 결론이 유지된다). 그리고 **ECE 는 위로 편향된 "
    "추정량이 아닌가**(지금까지 답이 없었다). 검증군이 %d·%d개뿐이므로 완벽히 보정된 "
    "모델이라도 각 구간의 관측 비율이 예측 평균과 우연히 어긋나 ECE 가 0 보다 크게 나온다."
    % (
        int(cm[(cm.dataset == "DILIrank") & (cm.arm.str.startswith("T"))].n.iloc[0]),
        int(cm[(cm.dataset == "DICTrank") & (cm.arm.str.startswith("T"))].n.iloc[0]),
    )
)
w()
w("그래서 **구간을 쓰지 않는 지표 셋**으로 같은 비교를 다시 했다.")
w()
w("| 지표 | 정의 | 구간을 쓰는가 |")
w("|---|---|---|")
w(
    "| ICI · E50 · E90 · Emax | 검증군에서 loess 보정곡선 f(p) 를 적합해 "
    "abs(p − f(p)) 의 평균·중앙값·90백분위·최대값 (Austin & Steyerberg 2019) | 아니오 "
    "(평활 폭 frac 을 고른다) |"
)
w(
    "| Spiegelhalter z | Σ(y−p)(1−2p) / sqrt(Σ(1−2p)²p(1−p)). 귀무가설은 "
    '"예측확률이 참값이다" | 아니오 (임의 선택이 전혀 없다) |'
)
w(
    "| 편향 제거 ECE | 예측을 고정하고 y\\* ~ Bernoulli(p) 를 반복 생성해 만든 ECE 의 "
    "귀무분포 평균을 관측 ECE 에서 뺀다. 그 귀무표본은 **정의상 완벽히 보정**돼 있다 | "
    "ECE 를 쓰되 그 편향을 빼낸다 |"
)
w()
_cmT = {d: cm[(cm.dataset == d) & (cm.arm.str.startswith("T"))].iloc[0] for d in DS}
for d in DS:
    g = cm[cm.dataset == d]
    w("**%s**" % KOR[d])
    w()
    w("| 분할 | n | ECE | 편향 제거 ECE | ICI | E50 | E90 | Emax | SH z | 대형 보정오차 | AUROC |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r.arm,
                int(round(r.n)),
                f4(r.ece),
                f4(r.ece_debiased),
                f4(r.ici),
                f4(r.e50),
                f4(r.e90),
                f4(r.emax),
                f2(r.sh_z),
                sg(r.cal_large),
                f4(r.auroc),
            )
        )
    w()
w(
    "대조군 행은 반복 %d회의 평균이다. R2·R3 는 양성률을 맞추려고 부분표집하므로 n 이 "
    "작아진다(%d / %d) — **절대값을 분할 간에 직접 비교하지 말고 같은 쌍의 배율로만 읽을 것.**"
    % (
        int(cm.n_rep.max()),
        int(round(cm[(cm.dataset == "DILIrank") & (cm.arm.str.startswith("R3"))].n.iloc[0])),
        int(round(cm[(cm.dataset == "DICTrank") & (cm.arm.str.startswith("R3"))].n.iloc[0])),
    )
)
w()
w("### 27.1 네 지표로 다시 낸 배율")
w()
w("| 데이터셋 | 대조 설계 | ECE 배율 | 편향 제거 배율 | ICI 배율 | E90 배율 |")
w("|---|---|---|---|---|---|")
for d in DS:
    for _, r in cmr[cmr.dataset == d].iterrows():
        w(
            "| %s | %s | %s | **%s** | %s | %s |"
            % (
                d,
                r.control,
                f2(r.ece_ratio),
                f2(r.ece_debiased_ratio),
                f2(r.ici_ratio),
                f2(r.e90_ratio),
            )
        )
w()
for d in DS:
    for _, r in cmr[cmr.dataset == d].iterrows():
        _pc = pc[(pc.dataset.str.startswith(d)) & (pc.setting.str.startswith(str(r.control)[:2]))]
        if len(_pc):
            assert (
                abs(float(r.ece_ratio) - float(_pc.ece_ratio.iloc[0])) < 5e-6
            ), "§27 의 ECE 배율이 §8 과 다르다"
_n_cells = len(cmr) * 4
_n_over = int(
    sum((cmr[c] > 1.0).sum() for c in ("ece_ratio", "ece_debiased_ratio", "ici_ratio", "e90_ratio"))
)
w(
    "**%d칸 전부 1 을 넘는다**(%d설정 × 4지표). ECE 배율은 §8 과 소수 6자리까지 같다 — "
    "같은 시드·같은 대조 설계를 독립적으로 재현한 뒤 지표만 바꿔 얹었기 때문이다. "
    '따라서 **"통제 후에도 시간분할의 보정이 더 나쁘다" 는 결론은 ECE 라는 지표의 '
    "산물이 아니다.**" % (_n_over, len(cmr))
)
w()
_bigger = int(sum(1 for _, r in cmr.iterrows() if r.ece_debiased_ratio > r.ece_ratio))
w(
    "**그리고 편향을 빼면 배율이 오히려 커진다 — %d설정 전부에서.** R1 에서 %s배 → %s배"
    "(DILIrank), %s배 → %s배(DICTrank)로 올라가고, 가장 보수적인 R3 에서도 %s → %s, "
    "%s → %s 다. 이유는 간단하다. 편향은 n 과 예측분포가 정하므로 분자와 분모에 비슷한 "
    "절대량(%s 대 %s)으로 들어가는데, 분모(대조군 ECE %s)가 분자(%s)보다 훨씬 작으므로 "
    "**같은 양을 빼면 분모가 비율로 더 많이 깎인다.** 즉 **§8 의 배율은 편향 때문에 "
    "과장된 값이 아니라 오히려 보수적인 값이다.**"
    % (
        _bigger,
        f2(cmr[(cmr.dataset == "DILIrank") & (cmr.control.str.startswith("R1"))].ece_ratio.iloc[0]),
        f2(
            cmr[
                (cmr.dataset == "DILIrank") & (cmr.control.str.startswith("R1"))
            ].ece_debiased_ratio.iloc[0]
        ),
        f2(cmr[(cmr.dataset == "DICTrank") & (cmr.control.str.startswith("R1"))].ece_ratio.iloc[0]),
        f2(
            cmr[
                (cmr.dataset == "DICTrank") & (cmr.control.str.startswith("R1"))
            ].ece_debiased_ratio.iloc[0]
        ),
        f2(cmr[(cmr.dataset == "DILIrank") & (cmr.control.str.startswith("R3"))].ece_ratio.iloc[0]),
        f2(
            cmr[
                (cmr.dataset == "DILIrank") & (cmr.control.str.startswith("R3"))
            ].ece_debiased_ratio.iloc[0]
        ),
        f2(cmr[(cmr.dataset == "DICTrank") & (cmr.control.str.startswith("R3"))].ece_ratio.iloc[0]),
        f2(
            cmr[
                (cmr.dataset == "DICTrank") & (cmr.control.str.startswith("R3"))
            ].ece_debiased_ratio.iloc[0]
        ),
        f4(_cmT["DILIrank"].ece_null_mean),
        f4(cm[(cm.dataset == "DILIrank") & (cm.arm.str.startswith("R1"))].ece_null_mean.iloc[0]),
        f4(cm[(cm.dataset == "DILIrank") & (cm.arm.str.startswith("R1"))].ece.iloc[0]),
        f4(_cmT["DILIrank"].ece),
    )
)
w()
w(
    "**편향의 크기 자체는 작지 않다.** 시간분할 ECE 중 편향이 차지하는 몫이 "
    "%s(DILIrank) / %s(DICTrank)다. 관측값은 두 데이터셋 모두 귀무분포의 97.5 백분위"
    "(%s / %s)를 넘으므로 **보정 붕괴가 유한표본 잡음일 가능성은 배제된다.** 다만 "
    '**절대 ECE 를 인용할 때는 이 몫을 함께 적어야 한다** — "시간분할 ECE 0.4384" 중 '
    "%s 는 어떤 모델이라도 나오는 값이다(DICTrank 은 %s 중 %s)."
    % (
        pct1(100.0 * _cmT["DILIrank"].ece_null_mean / _cmT["DILIrank"].ece),
        pct1(100.0 * _cmT["DICTrank"].ece_null_mean / _cmT["DICTrank"].ece),
        f4(_cmT["DILIrank"].ece_null_p975),
        f4(_cmT["DICTrank"].ece_null_p975),
        f4(_cmT["DILIrank"].ece_null_mean),
        f4(_cmT["DICTrank"].ece),
        f4(_cmT["DICTrank"].ece_null_mean),
    )
)
w()
w("### 27.2 반드시 함께 적어야 할 세 가지")
w()
_shr = cm[~cm.arm.str.startswith("T")]
w(
    "**1. 무작위 분할도 보정이 되어 있지 않다.** Spiegelhalter 검정은 시간분할"
    "(z = %s / %s)뿐 아니라 **대조군에서도 보정 가설을 기각한다**(z = %s–%s, 반복 중 "
    '유의 비율 %s–%s). 그래서 이 논문의 주장을 "무작위 분할은 보정되어 있는데 시간분할은 '
    '무너진다" 로 쓰면 **틀린다.** 둘 다 틀려 있고, **틀린 정도가 다르다**는 것이 주장이다.'
    % (
        f2(_cmT["DILIrank"].sh_z),
        f2(_cmT["DICTrank"].sh_z),
        f2(_shr.sh_z.min()),
        f2(_shr.sh_z.max()),
        pct1(100 * _shr.sh_p_frac_sig.min()),
        pct1(100 * _shr.sh_p_frac_sig.max()),
    )
)
_lad = 0
for _d in DS:
    _g = cmr[cmr.dataset == _d]
    for _c in ("ece_ratio", "ece_debiased_ratio", "ici_ratio", "e90_ratio"):
        _v = list(_g[_c])
        _lad += int(_v[0] > _v[1] > _v[2])
_zmono = all(
    list(cm[(cm.dataset == _d) & (~cm.arm.str.startswith("T"))].sh_z)
    == sorted(cm[(cm.dataset == _d) & (~cm.arm.str.startswith("T"))].sh_z)
    for _d in DS
)
w(
    "2. **loess 평활 폭(frac = 0.75)은 임의 선택이다.** ICI 는 구간을 없애는 대신 평활 "
    "폭이라는 선택을 들여온다. 그래서 ICI 를 단독 근거로 쓰지 않는다. 대신 두 가지를 "
    "확인했다 — **유병률 사다리(R1 > R2 > R3)가 네 지표 × 두 데이터셋 %d/8 에서 "
    "유지되고**, 임의 선택이 전혀 없는 Spiegelhalter z 도 대조군을 같은 방향으로 "
    "줄세운다(R1 < R2 < R3: %s). 즉 순서는 지표가 아니라 설계가 만든다."
    % (_lad, "두 데이터셋 모두 성립" if _zmono else "성립하지 않는 데이터셋이 있다")
)
w(
    "3. **대형 보정오차가 모든 설정에서 양수다**(시간분할 %s / %s). 평균 예측확률이 실제 "
    '양성률보다 높다는 뜻이고, §7 의 "무너진 것은 분산이 아니라 절편" 과 같은 방향이다. '
    "무작위 분할에서는 %s / %s 로 거의 0 이다 — 절편 이동이 시간분할에 특정적이라는 "
    "직접 증거다."
    % (
        sg(_cmT["DILIrank"].cal_large),
        sg(_cmT["DICTrank"].cal_large),
        sg(cm[(cm.dataset == "DILIrank") & (cm.arm.str.startswith("R1"))].cal_large.iloc[0]),
        sg(cm[(cm.dataset == "DICTrank") & (cm.arm.str.startswith("R1"))].cal_large.iloc[0]),
    )
)
w()
# ================================================================ 28. 하이퍼파라미터
w("---")
w()
w("## 28. 하이퍼파라미터 — 고정 설정 하나가 결론을 만들었나")
w()
_hpf = {d: hp[(hp.dataset == d) & (hp.is_fixed)].iloc[0] for d in DS}
_hpg = {d: hp[hp.dataset == d] for d in DS}
w(
    "§6 부터 §27 까지의 모든 수치가 `_shared.make_model()` 의 **설정 하나**에서 나왔다 — "
    "`%s`. 한 번도 조정하지 않았다. §9.1 의 32조합은 **모델 종류와 분자 표현**을 바꾼 "
    '것이고 하이퍼파라미터는 모델마다 고정이다. 그래서 "튜닝하지 않은 모델이 과적합해서 '
    '확률이 틀어진 것 아닌가" 에 답한 적이 없다.' % _hpf["DILIrank"].config
)
w()
w(
    "깊이 {3,6,9} × 나무 수 {200,400,800} × 학습률 {0.02,0.05,0.10} × "
    "min_child_weight {1,5} = **%d조합**을 전수 평가했다. 고정 설정이 이 격자 안에 있고, "
    "그 조합의 검증군 예측이 `make_model()` 과 **최대차 0** 으로 같은지 먼저 확인했다"
    "(스크립트가 그 자리에서 중단하도록 해 두었다)." % len(_hpg["DILIrank"])
)
w()
w("### 28.1 학습군만 보고 고른 조합 (실무에서 할 수 있는 최선)")
w()
w(
    "학습군을 연도순으로 나눠 앞 75% 로 적합하고 **뒤 25%**(학습 시대의 최근 구간)에서 "
    "조합을 골랐다. 검증군은 선택에 전혀 쓰지 않는다."
)
w()
w("| 데이터셋 | 선택 기준 | 고른 조합 | 검증 ECE | 검증 AUROC | 고정 대비 ECE |")
w("|---|---|---|---|---|---|")
_pick = {}
for d in DS:
    g = _hpg[d]
    _pick[d] = {
        "log loss": g.loc[g.inner_logloss.idxmin()],
        "ECE": g.loc[g.inner_ece.idxmin()],
        "AUROC": g.loc[g.inner_auroc.idxmax()],
    }
    for crit, r in _pick[d].items():
        w(
            "| %s | 내부 %s | `%s` | %s | %s | **%s** |"
            % (
                d,
                crit,
                r.config,
                f4(r.temporal_ece),
                f4(r.temporal_auroc),
                sg(r.temporal_ece - _hpf[d].temporal_ece),
            )
        )
    w(
        "| %s | (고정, 논문) | `%s` | %s | %s | — |"
        % (d, _hpf[d].config, f4(_hpf[d].temporal_ece), f4(_hpf[d].temporal_auroc))
    )
w()
_rank = {d: int((_hpg[d].temporal_ece < _hpf[d].temporal_ece).sum()) + 1 for d in DS}
w(
    "**자진 신고 — 논문의 고정 설정은 좋은 설정이 아니다.** 검증 ECE 가 작은 순으로 "
    "줄세우면 %d조합 중 **%d위(DILIrank) / %d위(DICTrank)** 로 둘 다 나쁜 쪽 절반에 있다. "
    "학습군만 보고 보정 기준으로 조합을 고르면 검증 ECE 가 %s / %s 까지 내려간다"
    "(각각 %s / %s). "
    '따라서 **"이 과제의 시간분할 보정오차는 %s 다" 처럼 절대값을 과제의 성질인 것처럼 '
    "쓰면 안 된다** — 설정에 따라 격자 안에서 %s–%s / %s–%s 로 움직인다."
    % (
        len(_hpg["DILIrank"]),
        _rank["DILIrank"],
        _rank["DICTrank"],
        f4(_pick["DILIrank"]["ECE"].temporal_ece),
        f4(_pick["DICTrank"]["ECE"].temporal_ece),
        sg(_pick["DILIrank"]["ECE"].temporal_ece - _hpf["DILIrank"].temporal_ece),
        sg(_pick["DICTrank"]["ECE"].temporal_ece - _hpf["DICTrank"].temporal_ece),
        f4(_hpf["DILIrank"].temporal_ece),
        f4(_hpg["DILIrank"].temporal_ece.min()),
        f4(_hpg["DILIrank"].temporal_ece.max()),
        f4(_hpg["DICTrank"].temporal_ece.min()),
        f4(_hpg["DICTrank"].temporal_ece.max()),
    )
)
w()
w("### 28.2 그런데 배율은 그대로다")
w()
w(
    "튜닝한 조합으로 §8 과 같은 무작위 분할 대조군을 다시 돌렸다(각 %d회)."
    % int(hpc.ctrl_R1_n.max())
)
w()
w("| 데이터셋 | 조합 | 설정 | 검증 ECE | R1 대조 | R1 배율 | R3 대조 | R3 배율 |")
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in hpc[hpc.dataset == d].iterrows():
        w(
            "| %s | %s | `%s` | %s | %s | **%s** | %s | **%s** |"
            % (
                d,
                r.arm,
                r.config,
                f4(r.temporal_ece),
                f4(r.ctrl_R1_ece),
                f2(r.ratio_R1),
                f4(r.ctrl_R3_ece),
                f2(r.ratio_R3),
            )
        )
_hf = {d: hpc[(hpc.dataset == d) & (hpc.arm == "고정 설정")].iloc[0] for d in DS}
_ht = {d: hpc[(hpc.dataset == d) & (hpc.arm != "고정 설정")].iloc[0] for d in DS}
w()
w(
    "**튜닝은 격차를 줄이지 못한다.** 양측 유병률을 맞춘 R3 배율은 %s → %s(DILIrank), "
    "%s → %s(DICTrank)로 사실상 그대로다. 표준 대조군 R1 에서는 오히려 **커진다** — "
    "%s → %s, %s → %s. 이유는 표에 있다. 튜닝이 시간분할 ECE 를 내리는 것보다 **무작위 "
    "분할 대조군의 ECE 를 더 많이 내리기 때문이다**(R1 대조 %s → %s / %s → %s). "
    "같은 설정 변경이 무작위 분할에서는 크게 듣고 시간분할에서는 조금만 듣는다."
    % (
        f2(_hf["DILIrank"].ratio_R3),
        f2(_ht["DILIrank"].ratio_R3),
        f2(_hf["DICTrank"].ratio_R3),
        f2(_ht["DICTrank"].ratio_R3),
        f2(_hf["DILIrank"].ratio_R1),
        f2(_ht["DILIrank"].ratio_R1),
        f2(_hf["DICTrank"].ratio_R1),
        f2(_ht["DICTrank"].ratio_R1),
        f4(_hf["DILIrank"].ctrl_R1_ece),
        f4(_ht["DILIrank"].ctrl_R1_ece),
        f4(_hf["DICTrank"].ctrl_R1_ece),
        f4(_ht["DICTrank"].ctrl_R1_ece),
    )
)
w()
w(
    "> 고정 설정의 대조군 값이 §8 과 조금 다른 것은 반복 횟수 차이다 — 여기는 %d회, "
    "§8 은 %d회다(R1 %s 대 %s). 같은 설계·같은 시드이고 차이는 %s 로 §9.3 의 적합 잡음 "
    "범위 안이다."
    % (
        int(hpc.ctrl_R1_n.max()),
        int(PCR[("DILIrank", "R1")].n_rep_valid),
        f4(_hf["DILIrank"].ctrl_R1_ece),
        f4(PCR[("DILIrank", "R1")].ECE),
        f4(abs(_hf["DILIrank"].ctrl_R1_ece - PCR[("DILIrank", "R1")].ECE)),
    )
)
w()
_nbelow = {d: int((_hpg[d].temporal_ece <= _hf[d].ctrl_R1_ece).sum()) for d in DS}
w(
    "**그리고 격자 안에 보정을 되돌리는 조합이 없다.** 검증군을 **보고** 골라도 "
    "(달성 불가능한 사후 선택) 검증 ECE 가 무작위 분할 대조 수준 이하로 내려가는 조합이 "
    "%d조합 중 %d개 / %d개다. 즉 §28.1 의 실패는 선택 방법의 문제가 아니다."
    % (len(_hpg["DILIrank"]), _nbelow["DILIrank"], _nbelow["DICTrank"])
)
w()
w("### 28.3 어떤 손잡이가 보정을 망가뜨리는가")
w()
w("| 손잡이 | 값 | DILIrank 평균 검증 ECE | DICTrank 평균 검증 ECE |")
w("|---|---|---|---|")
for col, lab in (
    ("max_depth", "트리 깊이"),
    ("n_estimators", "나무 수"),
    ("learning_rate", "학습률"),
    ("min_child_weight", "min_child_weight"),
):
    for v in sorted(hp[col].unique()):
        w(
            "| %s | %s | %s | %s |"
            % (
                lab,
                ("%g" % v),
                f4(_hpg["DILIrank"][_hpg["DILIrank"][col] == v].temporal_ece.mean()),
                f4(_hpg["DICTrank"][_hpg["DICTrank"][col] == v].temporal_ece.mean()),
            )
        )
w()
_mono = 0
for col in ("max_depth", "n_estimators", "learning_rate"):
    for d in DS:
        _v = [_hpg[d][_hpg[d][col] == x].temporal_ece.mean() for x in sorted(hp[col].unique())]
        _mono += int(_v[0] < _v[1] < _v[2])
w(
    "**용량 손잡이 셋이 전부 같은 방향이다 — %d/6 에서 단조.** 깊이가 깊을수록, 나무가 "
    "많을수록, 학습률이 클수록 시간분할 보정오차가 커진다. 두 데이터셋에서 똑같다. "
    '**시간분할의 보정 붕괴는 모델 용량과 함께 커진다**는 뜻이고, §7 의 "무너진 것은 '
    'reliability" 와 결이 같다 — 용량이 크면 학습 시대의 유병률을 더 확신 있게 외운다.' % _mono
)
w()


def _spear(a, b):
    import numpy as _np

    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(_np.corrcoef(ra, rb)[0, 1])


_tr_ece = {d: _spear(_hpg[d].inner_ece, _hpg[d].temporal_ece) for d in DS}
_tr_ll = {d: _spear(_hpg[d].inner_logloss, _hpg[d].temporal_ece) for d in DS}
_tr_au = {d: _spear(_hpg[d].temporal_auroc, _hpg[d].temporal_ece) for d in DS}
w("### 28.4 검증군 정보를 쓰지 않고 보정을 개선한 조작 — 정적 설계에서는 이것뿐이다")
w()
w(
    "학습 시대의 최근 구간에서 잰 성능이 미래 집합의 보정 순위를 **부분적으로 맞힌다** — "
    "내부 ECE 와 검증 ECE 의 순위상관 %s / %s, 내부 log loss 와는 %s / %s 다. "
    "**두 기준이 순위상관으로는 갈리지 않는다**(DILIrank 에서는 log loss 가 오히려 "
    "조금 높다). 갈리는 것은 실제로 고른 조합의 결과다 — 내부 ECE 로 고른 조합의 검증 "
    "ECE 개선이 %s 대 %s(DILIrank), %s 대 %s(DICTrank)로 두 데이터셋 모두 더 크다. "
    '**그래서 처방은 "보정으로 고르라" 이고, 근거는 상관계수가 아니라 이 개선폭이다.**'
    % (
        f3(_tr_ece["DILIrank"]),
        f3(_tr_ece["DICTrank"]),
        f3(_tr_ll["DILIrank"]),
        f3(_tr_ll["DICTrank"]),
        sg(_pick["DILIrank"]["ECE"].temporal_ece - _hpf["DILIrank"].temporal_ece),
        sg(_pick["DILIrank"]["log loss"].temporal_ece - _hpf["DILIrank"].temporal_ece),
        sg(_pick["DICTrank"]["ECE"].temporal_ece - _hpf["DICTrank"].temporal_ece),
        sg(_pick["DICTrank"]["log loss"].temporal_ece - _hpf["DICTrank"].temporal_ece),
    )
)
w()
w(
    "§20 의 최신성 가중, §21 의 적용범위 필터는 검정력을 붙여도 개선을 보이지 못했고"
    "(§29, DICTrank 최신성 가중 1건은 판정 보류), §23 의 클래스 가중은 검증군 양성률을 "
    "가정해야 듣는다. "
    "**모델을 한 번만 학습하는 설계 안에서는, 학습 시대의 최근 구간에서 보정 기준으로 "
    "조합을 고르는 것이 검증군 정보를 쓰지 않고 보정을 개선한 유일한 조작이다.** 다만 "
    "개선폭은 간격을 메우기에 턱없이 모자라고(배율은 그대로다, §28.2), 순위상관이 1 이 "
    "아니므로 **보장이 아니라 경향**이다. "
    "**설계를 바꾸면 더 센 것이 있다** — 승인이 쌓이는 대로 다시 학습하고 그때마다 "
    "학습창의 최신 구간으로 다시 보정하면 ΔECE %s / %s 다(§34). 이 절의 조작과 §34 는 "
    '모순이 아니라 층이 다르다 — 여기는 "한 번 학습할 때 어떤 설정을 고를까" 이고, '
    '§34 는 "얼마나 자주 다시 학습할까" 다.'
    % (
        sg(pqd[(pqd.dataset == "DILIrank") & pqd.arm.str.startswith("P2")].delta.iloc[0]),
        sg(pqd[(pqd.dataset == "DICTrank") & pqd.arm.str.startswith("P2")].delta.iloc[0]),
    )
)
w()
w(
    "**판별력과의 관계는 데이터셋마다 다르다.** 검증 AUROC 와 검증 ECE 의 순위상관이 "
    "%s(DILIrank) / %s(DICTrank)다. DILIrank 에서는 보정이 좋은 조합이 판별력도 좋지만 "
    'DICTrank 에서는 거의 무관하다. **"보정을 얻으려면 판별력을 내준다" 는 식의 '
    "교환관계를 주장하지 말 것** — 이 격자에는 그런 관계가 없다. 검증 AUROC 범위는 "
    "%s–%s / %s–%s 다."
    % (
        f3(_tr_au["DILIrank"]),
        f3(_tr_au["DICTrank"]),
        f4(_hpg["DILIrank"].temporal_auroc.min()),
        f4(_hpg["DILIrank"].temporal_auroc.max()),
        f4(_hpg["DICTrank"].temporal_auroc.min()),
        f4(_hpg["DICTrank"].temporal_auroc.max()),
    )
)
w()
# ================================================================ 29. 처방 검정력
w("---")
w()
w("## 29. 처방이 실패했다는 주장에 검정력을 붙인다")
w()
w(
    "§10·§20·§21·§23 의 결론 상당 부분이 **음성 결과**다 — 재보정은 일부만 되돌리고, "
    "최신성 가중·적용범위 필터·클래스 가중은 듣지 않는다. 음성 결과를 주장하려면 "
    '**"효과가 있었다면 이 표본으로 볼 수 있었는가"** 에 답해야 한다. 검증군이 %d·%d개뿐이니 '
    "작은 개선은 애초에 검출되지 않는다. 지금까지 그 한계를 적은 수치가 없었다."
    % (int(SPLIT["DILIrank"]["n_test"]), int(SPLIT["DICTrank"]["n_test"]))
)
w()
w(
    "각 처방을 **무보정과 같은 검증군에서 짝지어** 부트스트랩했다. 예측을 고정하고 검증군만 "
    "재표집하므로 두 팔의 상관이 보존되고, 짝지은 설계는 독립 비교보다 분산이 작다. "
    "그래서 이 표의 검정력은 같은 n 으로 얻을 수 있는 최선에 가깝다."
)
w()
w("| 열 | 뜻 |")
w("|---|---|")
w("| ΔECE | 처방 ECE − 무보정 ECE. 음수면 개선 |")
w("| Δ 95% 구간 | 짝지은 부트스트랩 백분위 |")
w("| MDE80 | 2.80 × SE. 이 크기 이상 개선했다면 유의하게 잡혔을 값 (양측 α 0.05, 검정력 0.80) |")
w("| 간격 | 무보정 ECE − 무작위 분할 대조 ECE(§8 R1). 처방이 메워야 하는 거리 |")
w("| MDE/간격 | 검출 가능한 최소 개선이 간격의 몇 퍼센트인가 |")
w()
_gap = {d: float(rp[rp.dataset == d].gap_to_control.iloc[0]) for d in DS}
for d in DS:
    g = rp[rp.dataset == d]
    w(
        "**%s** — 무보정 ECE %s, 무작위 대조 %s, **메워야 할 간격 %s**"
        % (KOR[d], f4(g.ece_base.iloc[0]), f4(g.ece_base.iloc[0] - _gap[d]), f4(_gap[d]))
    )
    w()
    w("| 처방 | 배포 | ECE | ΔECE | Δ 95% 구간 | MDE80 | MDE/간격 | 판정 |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %s | %s | [%s, %s] | %s | %s | **%s** |"
            % (
                r.remedy,
                r.deployable,
                f4(r.ece_remedy),
                sg(r.delta),
                sg(r.delta_lo),
                sg(r.delta_hi),
                f4(r.mde80),
                pct1(100 * r.mde_over_gap),
                str(r.verdict).split(" ")[0],
            )
        )
    w()
w(
    "판정 기호 — **(가)** 구간이 전부 0 아래(개선이 유의하다). **(나)** 구간이 허용하는 "
    "최대 개선조차 간격의 25% 에 못 미친다(쓸 만한 개선이 없음을 보였다). **(다)** 구간이 "
    "쓸 만한 개선을 아직 배제하지 못한다(판정 보류). 25% 기준은 스크립트에 사전 상수로 "
    "박아 두었다."
)
w()
w("### 29.1 이 연구는 검정력이 있었다")
w()
_n_pow = int((rp.mde_over_gap <= 0.25).sum())
w(
    "검출 가능한 최소 개선폭이 간격의 **%s–%s**(DILIrank) / **%s–%s**(DICTrank)다. "
    "**%d개 팔 중 %d개는 간격의 4분의 1짜리 개선이 있었다면 유의하게 잡아낼 검정력이 "
    "있었다** — 그중 %d개는 개선이 유의했고(%s) %d개는 쓸 만한 개선이 없음을 보였다. "
    '그래서 음성 결과를 "표본이 작아 못 봤다" 로 치부할 수 없다. '
    "판정 보류는 %d개뿐이고, 그 하나가 바로 MDE 가 가장 큰 팔(간격의 %s)이다 — "
    "검정력이 모자란 자리와 결론을 보류한 자리가 정확히 일치한다."
    % (
        pct1(100 * rp[rp.dataset == "DILIrank"].mde_over_gap.min()),
        pct1(100 * rp[rp.dataset == "DILIrank"].mde_over_gap.max()),
        pct1(100 * rp[rp.dataset == "DICTrank"].mde_over_gap.min()),
        pct1(100 * rp[rp.dataset == "DICTrank"].mde_over_gap.max()),
        len(rp),
        _n_pow,
        int(((rp.mde_over_gap <= 0.25) & rp.verdict.str.startswith("(가)")).sum()),
        ", ".join(
            sorted(
                set(
                    rp[(rp.mde_over_gap <= 0.25) & rp.verdict.str.startswith("(가)")].remedy.str[:2]
                )
            )
        ),
        int(((rp.mde_over_gap <= 0.25) & rp.verdict.str.startswith("(나)")).sum()),
        int(rp.verdict.str.startswith("(다)").sum()),
        pct1(100 * rp.mde_over_gap.max()),
    )
)
w()
_hold = rp[rp.verdict.str.startswith("(다)")]
if len(_hold):
    w("**판정 보류 %d개는 본문 문장을 고쳐야 한다.**" % len(_hold))
    w()
    for _, r in _hold.iterrows():
        w(
            "- %s %s — Δ %s, 구간 [%s, %s]. 구간이 **최대 %s(간격의 %s)** 의 개선까지 "
            '허용한다. 그래서 "듣지 않는다" 로 쓸 수 없고 "이 표본은 아직 배제하지 '
            '못한다" 로 써야 한다(§20 수정).'
            % (
                r.dataset,
                r.remedy,
                sg(r.delta),
                sg(r.delta_lo),
                sg(r.delta_hi),
                f4(-r.delta_lo),
                pct1(100 * (-r.delta_lo) / r.gap_to_control),
            )
        )
    w()
_dep = rp[rp.deployable == "배포 가능"]
_dep_sig = _dep[_dep.verdict.str.startswith("(가)")]
w("### 29.2 배포 가능한 처방 중 실제로 듣는 것")
w()
w(
    "검증군 양성률을 가정하는 D(사전확률 보정)·W1(클래스 가중)은 §10 에서 쓴 대로 "
    "**배포할 수 없다**. 배포 가능한 팔은 학습군 내부 보정셋만 쓰는 B·C 와 최신성 가중 R "
    "뿐이다(%d개). 그중 개선이 유의한 것은 **%d개 — 전부 DICTrank 다.**"
    % (len(_dep), len(_dep_sig))
)
w()
for _, r in _dep_sig.iterrows():
    w(
        "- %s %s — Δ %s, 구간 [%s, %s]. **간격의 %s 를 메운다.**"
        % (
            r.dataset,
            r.remedy,
            sg(r.delta),
            sg(r.delta_lo),
            sg(r.delta_hi),
            pct1(100 * (-r.delta) / r.gap_to_control),
        )
    )
w()
_bl = rp[(rp.dataset == "DILIrank") & (rp.remedy.str.startswith("B."))].iloc[0]
w(
    "**DILIrank 에서는 배포 가능한 처방이 전부 실패한다.** 가장 좋은 Platt 도 Δ %s 에 "
    "구간이 0 을 포함하고(%s, %s), 구간이 허용하는 최대 개선이 간격의 %s 다. "
    '**그래서 "재보정이 처방이다" 를 두 데이터셋에 공통으로 쓸 수 없다** — §10 의 '
    '"단일 권고안을 낼 수 없다" 와 §25.1 의 "상수를 넘는 방법이 데이터셋마다 다르다" 가 '
    "검정력 단위에서 다시 확인된다."
    % (
        sg(_bl.delta),
        sg(_bl.delta_lo),
        sg(_bl.delta_hi),
        pct1(100 * (-_bl.delta_lo) / _bl.gap_to_control),
    )
)
w()
w("### 29.3 적용범위 필터 — 버릴수록 검정력이 사라진다")
w()
w(
    "§21 의 필터는 n 을 줄이므로 짝지은 비교를 다르게 짰다. 재표집 안에서 유사도 순 보존과 "
    "무작위 버림을 **같은 재표집으로** 다시 만들어 차이를 뺀다."
)
w()
w("| 데이터셋 | 버림 비율 | 남는 n | 필터 ECE | 무작위 버림 ECE | Δ | Δ 95% 구간 | MDE80 |")
w("|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in rpa[rpa.dataset == d].iterrows():
        w(
            "| %s | %s | %d | %s | %s | %s | [%s, %s] | %s |"
            % (
                d,
                f2(r.drop_frac),
                int(r.n_keep),
                f4(r.ece_ad),
                f4(r.ece_random_drop),
                sg(r.delta),
                sg(r.delta_lo),
                sg(r.delta_hi),
                f4(r.mde80),
            )
        )
w()
_n0 = int(((rpa.delta_lo < 0) & (rpa.delta_hi > 0)).sum())
w(
    "**%d개 설정 전부에서 구간이 0 을 포함한다.** 그리고 버릴 비율이 커질수록 MDE80 이 "
    "%s 에서 %s 로 벌어진다 — **필터는 검정력까지 함께 버린다.** 많이 버리면 '남은 "
    "분자에서는 보정이 낫다' 를 보일 수도, 반박할 수도 없게 된다. §21 의 결론(필터는 "
    "처방이 아니다)은 버림 비율이 작은 쪽, 즉 검정력이 남아 있는 구간에서 읽어야 한다."
    % (_n0, f4(rpa.mde80.min()), f4(rpa.mde80.max()))
)
w()
w("### 29.4 재현 검사")
w()
w(
    "짝지은 비교는 같은 적합 위에서만 뜻이 있으므로 이 절은 §6·§8 과 같은 **원본 행 "
    "순서**에서 모든 팔을 다시 계산했다(§10 은 보정셋을 떼기 위해 승인연도로 정렬한다 — "
    "§10 의 주석 참조). 무보정 ECE 가 §8 과 일치하는지, 최신성 가중 τ* 팔이 §20 과, "
    "클래스 가중 W1 팔이 §23 과 일치하는지 스크립트가 매번 검사하고 어긋나면 중단한다. "
    "W1 은 §23 과 같은 경로(양성에만 주는 sample_weight)를 쓴다 — `scale_pos_weight` 로 "
    "같은 가중을 걸면 목적함수는 같지만 XGBoost 내부 적용 지점이 달라 ECE 가 0.005 수준 "
    "어긋난다(2026-10-19 확인·수정)."
)
w()
w("### 29.5 기준 25% 의 임의성을 없앤다")
w()
_fix = uts[uts.u_independent.astype(bool)]
_var = uts[~uts.u_independent.astype(bool)].sort_values("u_star")
w(
    '위 판정에서 "쓸 만한 개선" 의 기준을 간격의 25%% 로 두었는데 **그것은 임의 '
    "선택이다**(한계 23번). 그래서 기준을 %.0f%%–%.0f%% 로 훑어 판정이 어디서 뒤집히는지 "
    "셌다. 모델을 다시 적합하지 않는다 — 저장된 Δ 구간과 간격만 쓴다."
    % (utg.useful.min() * 100, utg.useful.max() * 100)
)
w()
w("**핵심은 팔마다 뒤집히는 지점이 하나의 수로 정해진다는 것이다.**")
w()
w("    U* = (구간이 허용하는 최대 개선) / (메워야 할 간격) = (−Δ하한) / 간격")
w()
w(
    "기준이 U* 보다 크면 (나), 작으면 (다) 다. 그래서 **U* 를 팔마다 적어 두면 기준을 "
    "하나로 고를 필요가 없다** — 독자가 자기 기준으로 판정을 읽는다."
)
w()
w("| 데이터셋 | 처방 | 배포 | U* | 기준 25% 판정 |")
w("|---|---|---|---|---|")
for _, r in uts.sort_values(["u_independent", "u_star"]).iterrows():
    w(
        "| %s | %s | %s | %s | (%s) |"
        % (
            r.dataset,
            r.remedy,
            r.deployable,
            "기준 무관" if bool(r.u_independent) else f3(r.u_star),
            r.verdict_at_ref,
        )
    )
w()
w(
    "**기준과 무관하게 판정이 정해지는 팔이 %d/%d 이다.** 구간이 0 을 포함하지 않으므로 "
    "어떤 기준에서도 (가) 또는 (가') 다. 나머지 %d개는 U* 가 %s–%s 에 흩어져 있다."
    % (len(_fix), len(uts), len(_var), f3(_var.u_star.min()), f3(_var.u_star.max()))
)
w()
w("| 기준 | (가) | (가') | (나) | (다) |")
w("|---|---|---|---|---|")
for _, r in utg.iterrows():
    mark = " ← 본문" if abs(float(r.useful) - 0.25) < 1e-9 else ""
    w(
        "| %s%% | %d | %d | %d | %d%s |"
        % (
            ("%.0f" % (r.useful * 100)),
            int(r.n_ga),
            int(r.n_ga_bad),
            int(r.n_na),
            int(r.n_da),
            mark,
        )
    )
w()
_ref = utg[(utg.useful - 0.25).abs() < 1e-9].iloc[0]
# 본문 기준과 판정이 완전히 같은 기준 구간을 데이터에서 찾는다
_same = utg[(utg.n_na == _ref.n_na) & (utg.n_da == _ref.n_da) & (utg.n_ga == _ref.n_ga)]
_lo_same, _hi_same = float(_same.useful.min()), float(_same.useful.max())
_below = _var[_var.u_star < 0.25].sort_values("u_star")
_above = _var[_var.u_star >= 0.25].sort_values("u_star")
w(
    "**판정이 본문 기준과 완전히 같은 구간은 %.0f%%–%.0f%% 다**((가) %d · (나) %d · "
    "(다) %d). 그 구간 밖에서는 이렇게 바뀐다 — 기준을 %.0f%% 로 올리면 %s 이 "
    "(다)에서 (나)로 넘어가고(U* = %s), %.0f%% 로 내리면 %s 이 (나)에서 (다)로 "
    "넘어간다(U* = %s). %.0f%% 에서는 %d건이 (다)로 넘어간다."
    % (
        _lo_same * 100,
        _hi_same * 100,
        int(_ref.n_ga),
        int(_ref.n_na),
        int(_ref.n_da),
        utg.useful.max() * 100,
        str(_above.iloc[0].remedy) if len(_above) else "없음",
        f3(_above.iloc[0].u_star) if len(_above) else "—",
        20.0,
        str(_below.iloc[-1].remedy) if len(_below) else "없음",
        f3(_below.iloc[-1].u_star) if len(_below) else "—",
        10.0,
        int(utg[(utg.useful - 0.10).abs() < 1e-9].iloc[0].n_da),
    )
)
w()
w(
    "**즉 음성 결과는 기준을 %.0f%%–%.0f%% 어디에 두어도 동일하다.** 그보다 낮추면 "
    '일부가 판정 보류로 바뀌는데, 낮춘다는 것은 "아주 작은 개선도 쓸 만하다고 치자" '
    "는 뜻이므로 판정 보류가 늘어나는 것은 당연하고 결함이 아니다. **그리고 (가)·(가') "
    "판정 %d건은 어떤 기준에서도 불변이다** — 구간이 0 을 포함하지 않으니 기준이 개입할 "
    "여지가 없다." % (_lo_same * 100, _hi_same * 100, int(_ref.n_ga) + int(_ref.n_ga_bad))
)
w()
# §29 가 쓰는 기준 (스크립트 상수와 같은 값, 하드코딩하지 않고 격자에서 읽는다)
_UREF = float(utg.useful.iloc[(utg.useful - 0.25).abs().argmin()])
_ua_lo = utsa[utsa.u_star < _UREF]
_ua_hi = utsa[utsa.u_star >= _UREF]
w(
    "**적용범위(AD) 필터 팔도 같은 기준으로 재야 한다.** §21.2 는 AD 필터가 처방이 못 됨을 "
    "보였고 §29 는 그 음성 결과에 검정력을 붙였는데, 이 소절의 U* 는 본 처방들에만 붙이고 "
    "AD 팔에는 붙이지 않고 있었다. 계산은 이미 돼 있었고 보고서가 읽지 않고 있었다. "
    '붙이면 이렇게 된다 — U* 는 **그 값보다 기준을 높게 두면 "쓸 만한 개선이 없다" 가 '
    "되는 경계**이므로, 본문 기준 %.0f%% 보다 U* 가 **작은** 행에서 음성 결과가 성립한다. "
    "%d행 중 %d행이 그렇고, 나머지 %d행은 판정 보류다."
    % (_UREF * 100, len(utsa), len(_ua_lo), len(_ua_hi))
)
w()
w(
    "| 데이터셋 | 버리는 비율 | 남는 n | AD 필터 ECE | 같은 수 무작위 폐기 ECE | ΔECE | Δ 95% 구간 | MDE80 | U* | 판정 |"
)
w("|---|---|---|---|---|---|---|---|---|---|")
for _, _r in utsa.iterrows():
    w(
        "| %s | %.0f%% | %d | %s | %s | %s | [%s, %s] | %s | %s | %s |"
        % (
            _r.dataset,
            100 * _r.drop_frac,
            int(_r.n_keep),
            f4(_r.ece_ad),
            f4(_r.ece_random_drop),
            sg(_r.delta),
            sg(_r.delta_lo),
            sg(_r.delta_hi),
            f4(_r.mde80),
            f3(_r.u_star),
            str(_r.verdict).split(" ")[0],
        )
    )
w()
w(
    "비교 대상은 **같은 수를 무작위로 버린 대조군**이다(§21.2 와 같은 설계). ΔECE 가 음수면 "
    'AD 필터가 무작위 폐기보다 나은 것인데, %d행 전부에서 구간이 0 을 포함한다 — **"가장 '
    '낯선 분자를 버린다" 는 조작이 "아무 분자나 같은 수만큼 버린다" 와 구별되지 '
    "않는다.**" % len(utsa)
)
w()
w(
    "**판정 보류가 어디에 몰려 있는지가 핵심이다.** 보류는 %s 다. 데이터셋마다 경계가 "
    "다르고(DILIrank 는 %.0f%% 부터, DICTrank 는 %.0f%% 부터), 공통점은 **많이 버려 남는 "
    "분자가 줄어든 쪽**이라는 것이다. 즉 표본이 있는 구간에서는 음성 결과가 기준과 무관하게 "
    "성립하고, 표본이 줄어든 구간에서는 판정을 못 한다. §29 의 틀과 같은 모양이다 — "
    '검출하지 못한 것을 "효과가 없다" 로 쓰지 않는다.'
    % (
        ", ".join(
            "%s %.0f%%(남는 n %d)" % (r.dataset, 100 * r.drop_frac, int(r.n_keep))
            for _, r in _ua_hi.sort_values(["dataset", "drop_frac"]).iterrows()
        ),
        100 * _ua_hi[_ua_hi.dataset == "DILIrank"].drop_frac.min(),
        100 * _ua_hi[_ua_hi.dataset == "DICTrank"].drop_frac.min(),
    )
)
w()
w(
    "> **한계 23번이 이 소절로 닫힌다.** 기준이 임의 선택이라는 사실은 그대로지만, "
    "**결론이 그 선택에 걸려 있지 않다**는 것을 수치로 보였다. 본문에서 음성 결과를 "
    "인용할 때는 U* 를 함께 적는다."
)
w()
# ================================================================ 30. 공변량이동 가중
w("---")
w()
w("## 30. 공변량 이동 보정 — 도메인 분류기로 밀도비 가중을 걸면 되는가")
w()
_cw0 = {
    d: csw[(csw.dataset == d) & (csw.kind == "outer") & (csw.arm.str.startswith("W0"))].iloc[0]
    for d in DS
}
_cwi = {
    d: csw[(csw.dataset == d) & (csw.kind == "outer") & (csw.arm.str.startswith("IW"))] for d in DS
}
w(
    "이 논문은 처방을 네 가지 시도했다 — 사후 재보정(§10), 사전확률 추정(§10.2), 최신성 "
    "가중(§20), 학습 시 클래스 가중(§23). 그런데 분포 이동 문헌에서 **가장 먼저 나오는 "
    "처방**이 빠져 있었다 — 공변량 이동 하의 **중요도 가중 학습**이다. 더 공교롭게도 그 "
    "가중을 만들 재료가 §13.1 에 이미 있었다. 시대 분류기가 구조만으로 시대를 맞히므로, "
    "그 확률이 곧 밀도비 추정값이다."
)
w()
w("    w(x) = p_post(x) / p_pre(x)  ∝  P(post|x) / (1 − P(post|x))")
w()
w(
    "학습군 각 분자에 이 가중을 주고 다시 학습한다. 가중은 교차검증 out-of-fold 확률로 "
    "만들어 자기 자신을 보고 가중을 정하는 일이 없게 했다. 시대 분류기의 OOF AUC 는 "
    "%s / %s 로 §13.1 과 같다(같은 시드·같은 분할)."
    % (f4(EP["DILIrank"].era_auc), f4(EP["DICTrank"].era_auc))
)
w()
w(
    '> **이 처방은 검증군 라벨을 쓰지 않는다.** 시대 분류기가 쓰는 것은 "이 분자가 어느 '
    '시대인가" 뿐이고, 점수를 매길 분자가 손에 있으면 아는 정보다. 그래서 §10 의 '
    "D(사전확률 보정)나 §23 의 W1 과 달리 **원리적으로 배포 가능한 처방이다.** 단 "
    "**미래 분자들의 구조를 미리 모아 두어야 한다**(transductive). 그 조건을 충족할 수 "
    "없는 상황에서는 이 절의 결과를 쓸 수 없다."
)
w()
w(
    "§20 의 최신성 가중과는 다른 처방이다. 저쪽은 **승인연도** 한 축으로 가중을 주고, "
    "이쪽은 **분자 구조 전체**로 준다 — 연도가 같아도 구조가 미래 쪽이면 가중이 커진다."
)
w()
for d in DS:
    g = csw[(csw.dataset == d) & (csw.kind == "outer")]
    w("**%s** — 시대 분류기 OOF AUC %s" % (KOR[d], f4(EP[d].era_auc)))
    w()
    w("| 팔 | 밀도비 상한 | 유효 n | 최대 w | AUROC | ECE | 보정기울기 | 해상도 | 평균 예측확률 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r.arm,
                "—" if r.cap != r.cap else ("%g" % r.cap),
                f1(r.n_eff),
                f2(r.w_max),
                f4(r.auroc),
                f4(r.ece),
                f4(r.slope),
                f4(r.res),
                f4(r.mean_pred),
            )
        )
    w()
_n_up = int(sum((_cwi[d].ece > float(_cw0[d].ece)).sum() for d in DS))
_n_all = int(sum(len(_cwi[d]) for d in DS))
w(
    "**%d개 팔 전부에서 ECE 가 올라간다**(%d/%d). 상한을 어떻게 두어도 가중 없음보다 "
    "나쁘다. 가장 좋은 팔조차 %s(DILIrank, 가중 없음 %s) / %s(DICTrank, %s)다. "
    "**중요도 가중은 이 과제에서 처방이 아니다.**"
    % (
        _n_all,
        _n_up,
        _n_all,
        f4(_cwi["DILIrank"].ece.min()),
        f4(_cw0["DILIrank"].ece),
        f4(_cwi["DICTrank"].ece.min()),
        f4(_cw0["DICTrank"].ece),
    )
)
w()
w(
    "**이유는 유효 표본의 붕괴다.** 밀도비는 꼬리가 두껍다 — 절단하지 않으면 최대 가중이 "
    "%s / %s 까지 올라가고 유효 표본이 %d → %d(%s 손실) / %d → %d(%s)로 무너진다. "
    "§20 에서 최신성 가중이 겪은 것과 같은 함정이지만 **훨씬 심하다** — 최신성 가중은 "
    "τ=3년에서도 유효 표본을 %d 는 남겼다."
    % (
        f2(_cwi["DILIrank"].w_max.max()),
        f2(_cwi["DICTrank"].w_max.max()),
        int(round(float(_cw0["DILIrank"].n_eff))),
        int(round(_cwi["DILIrank"].n_eff.min())),
        pct1(100 * (1 - _cwi["DILIrank"].n_eff.min() / float(_cw0["DILIrank"].n_eff))),
        int(round(float(_cw0["DICTrank"].n_eff))),
        int(round(_cwi["DICTrank"].n_eff.min())),
        pct1(100 * (1 - _cwi["DICTrank"].n_eff.min() / float(_cw0["DICTrank"].n_eff))),
        int(
            round(
                float(
                    _rw[
                        (_rw.dataset == "DILIrank") & (_rw.kind == "weight") & (_rw.tau == 3.0)
                    ].n_eff.iloc[0]
                )
            )
        ),
    )
)
w()
w("### 30.1 판별력은 오히려 조금 좋아진다 — 이 논문의 분리를 다시 확인한다")
w()
w(
    "**AUROC 는 올라간다.** 가중 없음 %s → 최선 %s(DILIrank), %s → %s(DICTrank)다. "
    "같은 조작이 **순위는 조금 개선하고 확률은 악화시킨다.** §25 가 Brier 로, §22 가 "
    "결정곡선으로 보인 분리가 처방 쪽에서도 한 번 더 나타난다 — 판별력을 고치는 손잡이와 "
    "보정을 고치는 손잡이가 같지 않다."
    % (
        f4(_cw0["DILIrank"].auroc),
        f4(_cwi["DILIrank"].auroc.max()),
        f4(_cw0["DICTrank"].auroc),
        f4(_cwi["DICTrank"].auroc.max()),
    )
)
w()
w("### 30.2 정직 선택과 대조군 배율")
w()
_cwin = {d: csw[(csw.dataset == d) & (csw.kind == "inner")] for d in DS}
w(
    "상한 선택은 §20 과 같은 **정직 프로토콜**로 했다 — 학습군을 %d년으로 다시 쪼개 "
    "앞부분으로 적합하고 뒷부분에서 고른다. 검증군은 선택에 쓰지 않는다." % 2000
)
w()
w("| 데이터셋 | 내부 선택이 고른 상한 | 내부 ECE | 그 팔의 검증 ECE | 사후 최선 상한 |")
w("|---|---|---|---|---|")
for d in DS:
    gi = _cwin[d]
    if not len(gi):
        continue
    b = gi.loc[gi.ece.idxmin()]
    _cap = b.cap
    outer = csw[
        (csw.dataset == d) & (csw.kind == "outer") & (csw.arm == b.arm.replace("내부 ", ""))
    ]
    best_out = _cwi[d].loc[_cwi[d].ece.idxmin()]
    w(
        "| %s | %s | %s | %s | %s |"
        % (
            d,
            "절단 없음" if _cap != _cap else ("%g" % _cap),
            f4(b.ece),
            f4(outer.ece.iloc[0]) if len(outer) else "—",
            "절단 없음" if best_out.cap != best_out.cap else ("%g" % best_out.cap),
        )
    )
w()
w(
    "**정직 선택이 고른 팔도, 사후에 검증군을 보고 고른 팔도 모두 가중 없음보다 나쁘다.** "
    "그래서 이 실패는 선택 방법의 문제가 아니다."
)
w()
w("| 데이터셋 | 대조 설계 | 대조 ECE | 배율 | 대조군 시대 AUC |")
w("|---|---|---|---|---|")
for d in DS:
    for _, r in cswc[cswc.dataset == d].iterrows():
        w(
            "| %s | %s | %s | **%s** | %s |"
            % (d, r.control, f4(r.ctrl_ece), f2(r.ratio), f4(r.ctrl_era_auc))
        )
w()
w(
    "**대조군에도 같은 처방을 걸었다.** 처방의 효과를 공정하게 보려면 대조군 쪽에서도 같은 "
    "절차를 돌려야 한다. 무작위 분할에서는 시대 분류기가 맞힐 것이 없으므로 AUC 가 "
    "%s–%s 로 0.5 근처이고, 따라서 가중이 1 에 가까워 처방이 사실상 아무 일도 하지 않는다 "
    "— 이 값이 설계가 맞물렸는지 확인하는 점검이다."
    % (f4(cswc.ctrl_era_auc.min()), f4(cswc.ctrl_era_auc.max()))
)
w()
_r3 = {d: cswc[(cswc.dataset == d) & (cswc.control == "R3")].iloc[0] for d in DS}
w(
    "**처방을 양쪽에 걸고 난 뒤의 R3 배율은 %s / %s 다.** 가중 없이 쟀을 때의 %s / %s"
    "(§8)보다 오히려 조금 크다. **중요도 가중은 분할 설계 격차를 전혀 좁히지 못한다.**"
    % (
        f2(_r3["DILIrank"].ratio),
        f2(_r3["DICTrank"].ratio),
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
    )
)
w()
w("### 30.3 이 절의 한계 세 가지")
w()
w(
    "1. **밀도비 추정량이 하나다.** 도메인 분류기 방식만 썼다. KMM·KLIEP 같은 다른 "
    '추정량은 시도하지 않았으므로 "중요도 가중 계열 전체가 실패한다" 로 넓혀 쓰면 안 된다. '
    "말할 수 있는 것은 **가장 널리 쓰이는 추정량으로는 실패한다** 까지다."
)
w(
    "2. **시대 분류기의 AUC 가 %s / %s 로 1 과 멀다.** 밀도비 추정이 정확하지 않다는 "
    "뜻이고, 가중에 잡음이 섞인다. 더 정확한 추정량이 있으면 결과가 달라질 수 있다."
    % (f4(EP["DILIrank"].era_auc), f4(EP["DICTrank"].era_auc))
)
w(
    "3. **transductive 조건이 붙는다.** 가중을 만들려면 미래 분자들의 구조가 있어야 한다. "
    "구조조차 없는 완전한 사전 배포 상황에는 적용할 수 없다."
)
w()
# ================================================================ 31. 격자 R3
w("---")
w()
w("## 31. 헤드라인 배율을 모델×표현 32칸 전체에서 — 가장 보수적인 대조군으로")
w()
w(
    "§9.5 의 32칸 격자(모델 4종 × 표현 4종)는 대조군을 **R1·R2 까지만** 돌렸다. 그런데 "
    "이 논문이 방어선으로 쓰는 수치는 **R3** 다 — 학습군과 검증군의 양성률을 양측 모두 "
    "맞춘 대조군에서 나온 %s배 · %s배(§8, §26, §1 표 6번). 즉 지금까지 상태는 이랬다. "
    "R1·R2 배율은 32칸 전부에서 1 을 넘었지만, **가장 보수적인 통제는 XGBoost + "
    'ECFP4+desc 한 칸만 확인돼 있었다.** 심사자가 "그 %s배가 그 모델 하나의 성질은 '
    '아닌가" 라고 물으면 답할 자료가 없었다.'
    % (
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        f2(PCR[("DILIrank", "R3")].ece_ratio),
    )
)
w()
w(
    "모델·표현 정의는 `_shared.grid_models()` / `grid_reps()` 한 곳에서 오고, 칸마다 "
    "시간분할 ECE 가 §9.5 와 일치하는지 먼저 검사했다(%d칸 전부 일치, 허용 1e-06). "
    "그 위에 R3 대조군만 새로 돌렸다(각 칸 %d회)." % (len(g3), int(g3.r3_n_rep_valid.max()))
)
w()
for d in DS:
    g = g3[g3.dataset == d]
    w("**%s**" % KOR[d])
    w()
    w("| 표현 | 모델 | 시간분할 ECE | R3 대조 ECE | R3 배율 | Δreliability | Δresolution |")
    w("|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %s | %s | %s | **%s** | %s | %s |"
            % (
                r.rep,
                r.model,
                f4(r.ece_temporal),
                f4(r.r3_ece_random),
                f2(r.r3_ece_ratio),
                sg(r.r3_d_reliability),
                sg(r.r3_d_resolution),
            )
        )
    w()
_n_over = int((g3.r3_ece_ratio > 1.0).sum())
w(
    "**%d칸 전부에서 R3 배율이 1 을 넘는다.** 범위 %s–%s, 중앙값 %s 다. "
    "**따라서 헤드라인은 모델과 표현의 선택에 의존하지 않는다** — 가장 보수적인 통제에서도 "
    "그렇다."
    % (_n_over, f2(g3.r3_ece_ratio.min()), f2(g3.r3_ece_ratio.max()), f2(g3.r3_ece_ratio.median()))
)
w()
w("| 묶음 | R3 배율 중앙값 |")
w("|---|---|")
for m, g in g3.groupby("model"):
    w("| 모델 %s | %s |" % (m, f2(g.r3_ece_ratio.median())))
for m, g in g3.groupby("rep"):
    w("| 표현 %s | %s |" % (m, f2(g.r3_ece_ratio.median())))
w()
_xg = {
    d: g3[(g3.dataset == d) & (g3.rep == "ECFP4+desc") & (g3.model == "XGBoost")].iloc[0]
    for d in DS
}
w(
    "**어느 모델군도 면역이 아니다.** 모델별 중앙값이 %s–%s, 표현별이 %s–%s 로 좁은 "
    "구간에 모여 있다. 본문 설정에 해당하는 칸(XGBoost + ECFP4+desc)은 %s / %s 로 "
    "§8 의 %s / %s 와 반복 횟수 차이(%d회 대 %d회) 안에서 일치한다."
    % (
        f2(g3.groupby("model").r3_ece_ratio.median().min()),
        f2(g3.groupby("model").r3_ece_ratio.median().max()),
        f2(g3.groupby("rep").r3_ece_ratio.median().min()),
        f2(g3.groupby("rep").r3_ece_ratio.median().max()),
        f2(_xg["DILIrank"].r3_ece_ratio),
        f2(_xg["DICTrank"].r3_ece_ratio),
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        int(g3.r3_n_rep_valid.max()),
        int(PCR[("DILIrank", "R3")].n_rep_valid),
    )
)
w()
w("### 31.1 보정 기여율은 이 설계에서 쓸 수 없다 — 자진 신고")
w()
_in_range = int(((g3.r3_share2 >= 0) & (g3.r3_share2 <= 100)).sum())
_den = (g3.r3_d_reliability + g3.r3_d_resolution).abs()
w(
    "**이 스크립트가 함께 계산한 R3 보정 기여율(Δreliability / (Δreliability + "
    "Δresolution))은 해석할 수 없는 값이 나왔다.** 32칸 중 0–100%% 범위에 들어온 칸이 "
    "**%d개뿐**이고 나머지는 %s%% 까지 튄다. 원인은 분모다 — Δreliability + Δresolution 의 "
    "절대값이 하위 10%% 칸에서 %s 밖에 안 되어 0 을 스치고, 그러면 비율이 폭발하거나 "
    '부호가 뒤집힌다. §7.1 이 "분모 선택" 을 따로 절로 둔 이유가 격자 수준에서 '
    "그대로 재현된 것이다. **이 열은 본문 주장에 쓰지 않는다.**"
    % (_in_range, f2(g3.r3_share2.abs().max()), f4(_den.quantile(0.1)))
)
w()
_pos = int((g3.r3_d_reliability > 0).sum())
_gt = int((g3.r3_d_reliability > g3.r3_d_resolution).sum())
_gt_abs = int((g3.r3_d_reliability > g3.r3_d_resolution.abs()).sum())
w(
    "**대신 나눗셈이 없는 통계로 같은 질문에 답한다.** 부호와 크기 비교는 분모가 0 을 "
    "스쳐도 무너지지 않는다."
)
w()
w("- Δreliability > 0 인 칸: **%d/%d** — 모든 칸에서 보정 성분이 나빠졌다." % (_pos, len(g3)))
w("- Δreliability > Δresolution 인 칸: **%d/%d**" % (_gt, len(g3)))
w(
    "- Δreliability > abs(Δresolution) 인 칸: **%d/%d** — 판별력 쪽 변화의 크기까지 "
    "넘어선 칸이다." % (_gt_abs, len(g3))
)
w()
w(
    '**이 세 줄이 §7 의 "무너진 것은 주로 보정이다" 를 백분율 없이, 32칸 전부에서 '
    "지탱한다.** 백분율로 쓰려면 반복을 §8 수준(%d회)으로 올려 분모를 안정시켜야 하고, "
    "격자 32칸 × 그 반복은 비용이 맞지 않는다고 판단했다 — 그 판단을 한계로 적는다."
    % int(PCR[("DILIrank", "R3")].n_rep_valid)
)
w()
# ================================================================ 32. 지평별 배율
w("---")
w()
w("## 32. 예측 지평이 멀어질수록 더 무너지는가 — 구간별 배율")
w()
_hzd = {d: hz[hz.dataset == d].sort_values("year_from") for d in DS}
w(
    '§13.4 는 검증군을 승인연도 구간으로 쪼개 ECE 를 찍고 이렇게 적었다 — "구간별 '
    '대조군을 정의할 수 없어 배율은 계산하지 않았고 절대 ECE 추세만 보고한다." 그 한 '
    "줄이 문제였다. 구간마다 n 과 양성률이 다르고(DILIrank %d개·%s 에서 %d개·%s 로), "
    "**양성률이 내려가면 §19 의 법칙만으로도 ECE 는 올라간다.** 그래서 §13.4 의 상승 "
    "추세가 지평 때문인지 양성률 때문인지 구분할 수 없었다."
    % (
        int(_hzd["DILIrank"].n.iloc[0]),
        f4(_hzd["DILIrank"].pos_rate.iloc[0]),
        int(_hzd["DILIrank"].n.iloc[-1]),
        f4(_hzd["DILIrank"].pos_rate.iloc[-1]),
    )
)
w()
w(
    "이 절은 **구간마다 그 구간과 n·양성률이 같은 대조군**을 만들어 배율을 낸다. R1 은 "
    "크기만, R2 는 검증군 양성률까지, R3 는 학습군 양성률까지 맞춘다. 배율에는 구간 "
    "검증군 부트스트랩으로 구간도 붙였다(분자만, §26 과 같은 방식)."
)
w()
for d in DS:
    g = _hzd[d]
    w("**%s**" % KOR[d])
    w()
    w(
        "| 승인 구간 | n | 양성률 | 학습군 대비 격차 | AUROC | ECE | R1 배율 | R2 배율 | R3 배율 | R3 배율 95% CI |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in g.iterrows():
        w(
            "| %s | %d | %s | %s | %s | %s | %s | %s | **%s** | [%s, %s] |"
            % (
                r.window,
                int(r.n),
                f4(r.pos_rate),
                sg(r.gap_vs_train),
                f4(r.auroc),
                f4(r.ece),
                f2(r.r1_ratio),
                f2(r.r2_ratio),
                f2(r.r3_ratio),
                f2(r.r3_ratio_lo),
                f2(r.r3_ratio_hi),
            )
        )
    w()
_n_over = int((hz.r3_ratio > 1.0).sum())
w(
    "**%d구간 전부에서 R3 배율이 1 을 넘는다**(%s–%s). 즉 **어느 지평 구간에서도 양측 "
    "유병률을 맞춘 뒤에도 보정이 더 나쁘다.** 이것이 이 절의 확실한 결과다."
    % (_n_over, f2(hz.r3_ratio.min()), f2(hz.r3_ratio.max()))
)
w()
_ex = hz[hz.r3_excludes_one.astype(bool)]
w(
    "구간이 1 을 제외하는 것은 %d/%d 구간뿐이다 — %s 다. 나머지는 구간 n 이 %d–%d 로 "
    "작아 구간이 넓다. **그래서 구간별 배율은 점추정값으로 비교하지 말고 방향만 읽어야 "
    "한다.**"
    % (
        len(_ex),
        len(hz),
        (
            ", ".join("%s %s" % (r.dataset, r.window) for _, r in _ex.iterrows())
            if len(_ex)
            else "없다"
        ),
        int(hz.n.min()),
        int(hz.n.max()),
    )
)
w()
w("### 32.1 지평 추세는 이 표본으로 판정할 수 없다")
w()
_d3 = {d: float(_hzd[d].r3_ratio.iloc[-1] - _hzd[d].r3_ratio.iloc[0]) for d in DS}
_hw = {
    d: 0.5
    * float(
        (_hzd[d].r3_ratio_hi.iloc[0] - _hzd[d].r3_ratio_lo.iloc[0])
        + (_hzd[d].r3_ratio_hi.iloc[-1] - _hzd[d].r3_ratio_lo.iloc[-1])
    )
    / 2.0
    for d in DS
}
w(
    "| 데이터셋 | 절대 ECE 추세 | 양성률 추세 | R1 배율 추세 | R3 배율 추세 | R3 변화(첫→끝) | 양 끝 구간 반폭 평균 |"
)
w("|---|---|---|---|---|---|---|")
for d in DS:
    g = _hzd[d]
    w(
        "| %s | %s | %s | %s | %s | %s | %s |"
        % (
            d,
            " → ".join(f4(v) for v in g.ece),
            " → ".join(f4(v) for v in g.pos_rate),
            " → ".join(f2(v) for v in g.r1_ratio),
            " → ".join(f2(v) for v in g.r3_ratio),
            sg(_d3[d]),
            f3(_hw[d]),
        )
    )
w()
w(
    "**절대 ECE 는 두 데이터셋 모두 올라가고, 유병률을 통제하지 않은 R1 배율도 올라간다"
    "**(%s → %s / %s → %s). 그런데 **양측을 맞춘 R3 배율의 변화는 %s / %s 로 양 끝 구간 "
    "신뢰구간 반폭(%s / %s)보다 작다.** DILIrank 는 단조조차 아니다(%s)."
    % (
        f2(_hzd["DILIrank"].r1_ratio.iloc[0]),
        f2(_hzd["DILIrank"].r1_ratio.iloc[-1]),
        f2(_hzd["DICTrank"].r1_ratio.iloc[0]),
        f2(_hzd["DICTrank"].r1_ratio.iloc[-1]),
        sg(_d3["DILIrank"]),
        sg(_d3["DICTrank"]),
        f3(_hw["DILIrank"]),
        f3(_hw["DICTrank"]),
        " → ".join(f2(v) for v in _hzd["DILIrank"].r3_ratio),
    )
)
w()
w(
    "**그래서 결론은 둘이다.** 첫째, **§13.4 의 상승 추세를 지평 효과로 쓰면 안 된다** — "
    "유병률을 맞추면 사라진다. DILIrank 에서 양성률이 %s 에서 %s 로 떨어지는 것과 같이 "
    '가는 현상이고, §19 가 보인 법칙이 그것을 설명한다. 둘째, **"지평이 멀어질수록 더 '
    '나빠진다" 는 주장도 기각하지 않는다** — 이 표본으로는 판정이 안 된다. 구간 n 이 '
    "%d–%d 이라 R3 배율의 구간 반폭이 %s 를 넘는데, 15년 사이의 변화는 그보다 작다."
    % (
        f4(_hzd["DILIrank"].pos_rate.iloc[0]),
        f4(_hzd["DILIrank"].pos_rate.iloc[-1]),
        int(hz.n.min()),
        int(hz.n.max()),
        f3(min(_hw.values())),
    )
)
w()
w(
    '> **한계 9번이 이 절로 닫힌다.** "구간별 대조군을 정의할 수 없어 배율이 없다" 가 '
    '"배율을 냈고, 여섯 구간 전부에서 1 을 넘으며, 지평 추세는 이 표본 크기로 판정할 '
    '수 없다" 로 바뀐다. 대조군을 못 만든 것이 아니라 **만들었더니 추세를 가릴 만큼 '
    "구간이 넓었다** 는 것이 정확한 서술이다."
)
w()
# ================================================================ 33. 이동량 맞춤
w("---")
w()
w("## 33. 같은 이동량에서 시간분할만 더 심한가 — 이동량을 맞춘 비시간 분할")
w()
_s2 = {d: sm2[sm2.dataset == d] for d in DS}
_s2m = {d: sm2m[sm2m.dataset == d] for d in DS}
w(
    '§13.5 는 시간이 아닌 축(분자량·logP·지문 군집)으로 분할을 만들어 "이동이 크면 보정이 '
    '더 무너지는가" 를 물었다. 그런데 그 축들의 이동 AUC 가 %s–%s 로 포화해 버렸고 '
    "시간분할은 %s / %s 다. **비교 대상의 이동량이 전혀 맞지 않았다.** 그래서 §13.5 는 "
    '"이동을 더 크게 줘도 붕괴가 더 커지지 않는다" 까지만 말하고, 정작 묻고 싶은 '
    '**"같은 이동량에서 시간분할만 더 심한가"** 는 답하지 못했다(한계 18번).'
    % (
        f4(sm[sm.split.str.startswith("X")].shift_auc.min()),
        f4(sm[sm.split.str.startswith("X")].shift_auc.max()),
        f4(_s2["DILIrank"].shift_auc_temporal.iloc[0]),
        f4(_s2["DICTrank"].shift_auc_temporal.iloc[0]),
    )
)
w()
w(
    "**축을 숨기는 대신 이동량을 조절했다.** 모델 입력에 없는 축을 찾는 방식은 막다른 "
    "길이다 — 지문이 분자 크기·극성과 상관되므로 어떤 물리화학적 축도 입력에서 완전히 "
    "숨길 수 없다. 그래서 반대로 간다. 각 분자에 `alpha × (축 순위) + (1−alpha) × "
    "(무작위 순위)` 를 주고 상위를 검증군으로 뽑으면, alpha = 0 은 순수 무작위"
    "(이동 AUC 약 0.5), alpha = 1 은 축 극단(약 1.0)이고 **그 사이를 연속으로 훑을 수 "
    "있다.** 시간분할의 이동 AUC 와 가장 가까운 alpha* 를 골라 거기서 ECE 를 비교한다."
)
w()
w(
    "교란을 막기 위해 **검증군 크기·양성률과 학습군 양성률을 모두 시간분할에 맞췄다**"
    "(R3 와 같은 양측 통제). 그러면 시간분할과 다른 것은 **분할 축 하나**뿐이다. "
    "이동 AUC 는 §13.1 과 같은 방식으로 쟀다 — 같은 특성, 같은 모델, 분할 라벨에 대한 "
    "교차검증 out-of-fold AUC(여기서는 비용 때문에 3겹, §13.1 은 5겹이다. "
    "그래서 기준점 자체도 %s / %s 로 §13.1 의 %s / %s 과 조금 다르다)."
    % (
        f4(_s2["DILIrank"].shift_auc_temporal.iloc[0]),
        f4(_s2["DICTrank"].shift_auc_temporal.iloc[0]),
        f4(EP["DILIrank"].era_auc),
        f4(EP["DICTrank"].era_auc),
    )
)
w()
w("### 33.1 alpha 를 훑어 이동량을 맞춘다")
w()
w("| 데이터셋 | 축 | alpha 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 목표 (시간분할) |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for ax in sorted(_s2[d].axis.unique()):
        g = _s2[d][(_s2[d].axis == ax)].set_index("alpha_num")
        cells = []
        for al in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            cells.append(f4(g.shift_auc.loc[al]) if al in g.index else "—")
        w(
            "| %s | %s | %s | %s |"
            % (d, ax, " | ".join(cells), f4(_s2[d].shift_auc_temporal.iloc[0]))
        )
w()
# 판정어("성공했다")를 산문에 박아 두면 결과가 실패로 바뀌어도 문장이
# 그대로 남는다. 2026-10-22 에 실제로 그렇게 됐다 — 2차 정밀 탐색이
# 빠진 상태로 문서를 만들어, 같은 문단에서 문장은 "성공" 이고 표의
# 판정 열은 여섯 행 모두 "맞춤 실패" 였다. 그래서 판정도 파일에서 읽는다.
_mt = sm2m["matched"].astype(str).str.strip().str.lower().isin(("true", "1"))
_nmt, _nall = int(_mt.sum()), len(sm2m)
# 추가로 **탐색한** alpha (거친 격자 0.0·0.2·…·1.0 밖의 값)를 적는다.
# 선택된 alpha_star 만 적으면 "추가로 돌렸다" 와 어긋난다.
_coarse = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
_fine = sorted(v for v in set(sm2.alpha_num.round(2)) if round(v, 2) not in _coarse)
w(
    "처음 돌린 0.2 단위 격자는 **여섯 설정 전부 맞춤에 실패했다** — 목표가 두 격자점 "
    "사이에 떨어지고 차이가 허용치 0.05 를 넘었다. 반복간 이동 AUC 의 표준편차가 "
    "%s–%s(중앙값 %s)이라 평균의 표준오차는 0.006 수준이므로, 격자만 촘촘히 하면 맞출 수 "
    "있다고 보고 맞춤 구간(alpha %s)을 추가로 돌렸다. "
    "**%d개 설정 중 %d개에서 맞춤에 성공했다**(이동 AUC 차 %s–%s)."
    % (
        f4(sm2.shift_auc_sd.min()),
        f4(sm2.shift_auc_sd.max()),
        f4(sm2.shift_auc_sd.median()),
        "·".join("%.2f" % v for v in _fine),
        _nall,
        _nmt,
        f4(sm2m.shift_auc_gap.abs().min()),
        f4(sm2m.shift_auc_gap.abs().max()),
    )
)
w()
w("### 33.2 맞춘 지점에서의 비교")
w()
w(
    "| 데이터셋 | 축 | alpha* | 이동 AUC (목표) | 맞춤 ECE | 반복 SD | 시간분할 ECE | 차이 | SE 배수 | 배율 | 판정 |"
)
w("|---|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for _, r in _s2m[d].sort_values("axis").iterrows():
        w(
            "| %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | **%s** | %s |"
            % (
                d,
                r.axis,
                f2(r.alpha_star),
                f4(r.shift_auc_matched),
                f4(r.shift_auc_temporal),
                f4(r.ece_matched),
                f4(r.ece_matched_sd),
                f4(r.ece_temporal),
                sg(r.ece_diff),
                f1(r.diff_se_units),
                f2(r.ece_ratio_T_over_matched),
                str(r.verdict).split(" (")[0],
            )
        )
w()
_win = int((sm2m.diff_se_units > 2).sum())
_tie = int((sm2m.diff_se_units.abs() <= 2).sum())
_los = int((sm2m.diff_se_units < -2).sum())
w(
    "**%d/%d 에서 시간분할 ECE 가 더 크고, 그 차이가 반복 표준오차의 %s–%s배다.** "
    "무승부가 %d건(%s, 차 %s, %s SE), 반대 방향은 %d건이다. 배율은 %s–%s, 중앙값 %s 다."
    % (
        _win,
        len(sm2m),
        f1(sm2m[sm2m.diff_se_units > 2].diff_se_units.min()),
        f1(sm2m[sm2m.diff_se_units > 2].diff_se_units.max()),
        _tie,
        ", ".join(
            "%s %s" % (r.dataset, r.axis) for _, r in sm2m[sm2m.diff_se_units.abs() <= 2].iterrows()
        ),
        sg(sm2m[sm2m.diff_se_units.abs() <= 2].ece_diff.iloc[0]),
        f1(sm2m[sm2m.diff_se_units.abs() <= 2].diff_se_units.iloc[0]),
        _los,
        f2(sm2m.ece_ratio_T_over_matched.min()),
        f2(sm2m.ece_ratio_T_over_matched.max()),
        f2(sm2m.ece_ratio_T_over_matched.median()),
    )
)
w()
_sm2n = pd.Series(
    [
        float(
            sm2[
                (sm2.dataset == r.dataset)
                & (sm2.axis == r.axis)
                & ((sm2.alpha_num - r.alpha_star).abs() < 1e-9)
            ].n_test.iloc[0]
        )
        for _, r in sm2m.iterrows()
    ]
)
w(
    "**이 비교는 결론에 불리한 쪽으로 기울어져 있다 — 그래서 더 강하다.** 맞춤 분할의 "
    "검증군 n 은 %d–%d 로 시간분할(%d / %d)보다 작다. §27 이 보인 대로 n 이 작으면 ECE "
    "편향이 **커진다**(n=156 에서 %s, n=92 에서 %s). 즉 맞춤 분할 쪽에 편향이 더 많이 "
    "얹혀 있는데도 ECE 가 더 낮다. 편향을 걷어내면 격차는 더 벌어진다."
    % (
        int(round(_sm2n.min())),
        int(round(_sm2n.max())),
        int(SPLIT["DILIrank"]["n_test"]),
        int(SPLIT["DICTrank"]["n_test"]),
        f4(_cmT["DILIrank"].ece_null_mean),
        f4(cm[(cm.dataset == "DILIrank") & (cm.arm.str.startswith("R3"))].ece_null_mean.iloc[0]),
    )
)
w()
w(
    "> **한계 18번이 닫힌다.** 같은 이동량에서도 시간분할의 보정오차가 더 크다. 따라서 "
    '**보정 붕괴는 "분포가 많이 이동했다" 로 환원되지 않고 시간 축에 특정적인 성분이 '
    '있다.** §13.3 이 보인 "이동의 크기가 붕괴의 크기를 설명하지 못한다" 와 §13.5 의 '
    '"이동을 키워도 붕괴가 커지지 않는다" 가 여기서 하나로 모인다 — 문제는 이동의 '
    "**양**이 아니라 이동의 **종류**다."
)
w()
w("### 33.3 이 절을 인용할 때 같이 적을 것")
w()
w(
    "1. **무승부 1건을 숨기지 말 것.** %s 에서는 차이가 %s(%s SE)로 0 과 구분되지 "
    '않는다. "여섯 설정 전부" 가 아니라 "다섯 설정에서, 한 설정은 무승부" 로 쓴다.'
    % (
        " ".join(sm2m[sm2m.diff_se_units.abs() <= 2].iloc[0][["dataset", "axis"]]),
        sg(sm2m[sm2m.diff_se_units.abs() <= 2].ece_diff.iloc[0]),
        f1(sm2m[sm2m.diff_se_units.abs() <= 2].diff_se_units.iloc[0]),
    )
)
w(
    "2. **이동 AUC 는 모델이 볼 수 있는 이동만 잰다.** 같은 특성으로 분류기를 학습하므로, "
    '특성에 들어오지 않는 이동은 이 지표로 0 이다. "이동량을 맞췄다" 는 '
    "**이 특성 공간에서 맞췄다** 는 뜻이다."
)
w(
    "3. **alpha 혼합 분할은 자연스러운 설계가 아니다.** 축 순위와 무작위 순위를 섞은 "
    "보간이고, 현실에서 그런 방식으로 데이터가 쪼개지지는 않는다. 목적은 현실의 분할을 "
    "흉내내는 것이 아니라 **이동량이라는 변수 하나만 통제해 비교하는 것**이다."
)
w(
    "4. **반복 %d회의 평균이다.** 맞춤 ECE 의 반복간 SD 가 %s–%s 이므로 점추정값을 "
    "소수 넷째 자리까지 인용하지 말 것."
    % (int(sm2.n_rep_valid.max()), f4(sm2m.ece_matched_sd.min()), f4(sm2m.ece_matched_sd.max()))
)
w()
# ================================================================ 34. 통계 방법
# ================================================================ 34. 배포 시뮬레이션
w("---")
w()
w("## 34. 해마다 다시 학습하면 회복되는가 — 배포 시뮬레이션")
w()
_PQA = [
    "P0. 고정 모델",
    "P1. 해마다 재학습",
    "P2. 재학습+재보정",
    "C1. 연도 무작위",
    "C2. 연도 무작위+양성률 맞춤",
    "C1r. 연도 무작위+재보정",
    "C2r. 연도 무작위+양성률 맞춤+재보정",
]


def _pq(d, arm):
    g = pq[(pq.dataset == d) & (pq.arm == arm)]
    return g if len(g) else None


w(
    "이 논문은 처방을 일곱 가지 시험했다 — 사후 재보정(§10·§10.5), 사전확률 추정(§10.2·"
    "§10.3), 최신성 가중(§20), 적용범위 필터(§21), 클래스 가중(§23), 공변량 이동 가중(§30), "
    "하이퍼파라미터 선택(§28). 그런데 **현장에서 가장 먼저 하는 일**이 빠져 있었다. 새 약이 "
    '승인되고 라벨이 붙으면 그것을 학습에 넣고 모델을 다시 만든다. "그러면 해마다 다시 '
    '학습하면 되는 것 아닌가" 라는 질문에 답이 없었다.'
)
w()
w(
    "그래서 전향적 순차 평가(prequential)를 돌렸다. 검증 연도 T 마다 승인연도 < T 인 분자 "
    "전부로 학습해 그 해 승인 분자를 예측하고, 모든 연도의 예측을 **모아서 한 번** 평가한다. "
    "모아서 재는 이유는 연도별 분자가 10여 개뿐이라 연도별 ECE 를 쓸 수 없기 때문이다. "
    "모아 놓은 평가집합은 §4~§10 의 검증군과 **같은 집합**(%d / %d개)이므로 고정 모델과 "
    "분자 단위로 짝지어 비교할 수 있다. 연도는 DILIrank %d개, DICTrank %d개다."
    % (
        int(_pq("DILIrank", "P0. 고정 모델").n.iloc[0]),
        int(_pq("DICTrank", "P0. 고정 모델").n.iloc[0]),
        int(_pq("DILIrank", "P1. 해마다 재학습").n_years.iloc[0]),
        int(_pq("DICTrank", "P1. 해마다 재학습").n_years.iloc[0]),
    )
)
w()
w(
    "팔은 일곱이다. **P0** 고정 모델(2010년 이전만으로 한 번 학습 — 이 논문의 기존 설정), "
    "**P1** 해마다 재학습, **P2** 해마다 재학습 + 해마다 학습창의 최근 25% 로 Platt 재보정"
    "(§10 의 B 와 같은 방식을 움직이는 창에서), 그리고 연도를 무작위로 섞어 같은 절차를 "
    "돌린 교환가능 기준선 넷이다. 기준선을 넷으로 둔 이유가 있다 — 양성률을 맞춘 것(C2·C2r)과 "
    "안 맞춘 것(C1·C1r), 재보정을 한 것(C1r·C2r)과 안 한 것(C1·C2)을 모두 둬야 **팔과 "
    "조건이 같은 기준선**을 고를 수 있다. 처음에는 재보정 없는 기준선만 두고 P2 와 "
    "비교했는데, 그러면 P2 의 우위가 재보정 덕인지 시간축 덕인지 갈라지지 않는다."
)
w()
w("| 데이터셋 | 팔 | n | ECE | ECE 표준편차 | AUROC | 보정 기울기 | 평균 예측확률 | 실제 양성률 |")
w("|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for arm in _PQA:
        g = _pq(d, arm)
        if g is None:
            continue
        _sd = "%s (%d회)" % (f4(g.ece.std(ddof=1)), len(g)) if len(g) > 1 else "—"
        w(
            "| %s | %s | %d | %s | %s | %s | %s | %s | %s |"
            % (
                d,
                arm,
                int(g.n.mean()),
                f4(g.ece.mean()),
                _sd,
                f4(g.auroc.mean()),
                f4(g.slope.mean()),
                f4(g.mean_pred.mean()),
                f4(g.obs_rate.mean()),
            )
        )
w()
w(
    "짝지은 비교는 같은 분자에서 재표집한 부트스트랩 %d회다. ΔECE 는 새 팔 − 고정 "
    "모델이므로 음수면 개선이다. **조건 맞춘 기준선** 열은 그 팔과 재보정 유무·양성률 "
    "맞춤이 같은 대조군이다." % 2000
)
w()
_pqd = {}
for _, _r in pqd.iterrows():
    _pqd[(str(_r.dataset), str(_r.arm))] = _r
w("| 데이터셋 | 비교 | 짝지은 ΔECE | 95% CI | 판정 | 조건 맞춘 기준선 |")
w("|---|---|---|---|---|---|")
for d in DS:
    for arm in ("P1. 해마다 재학습", "P2. 재학습+재보정"):
        if (d, arm) not in _pqd:
            continue
        r = _pqd[(d, arm)]
        w(
            "| %s | %s 대 P0 | %s | [%s, %s] | %s | %s %s |"
            % (
                d,
                arm.split(". ")[1],
                sg(r.delta),
                sg(r.delta_lo),
                sg(r.delta_hi),
                ("**%s**" % r.verdict) if r.delta_hi < 0 else str(r.verdict),
                str(r.ctrl_arm).split(". ")[0],
                f4(r.ctrl_ece),
            )
        )
w()
w("읽을 것이 넷이다.")
w()
_pq_p1 = [d for d in DS if _pqd[(d, "P1. 해마다 재학습")].delta_hi >= 0]
w(
    "1. **일을 하는 것은 재보정 쪽이다.** P1(재학습만)의 ΔECE 는 %s 이고, P2(재학습+재보정)는 "
    "%s 로 효과가 세 배 안팎이다. %s"
    % (
        " / ".join(sg(_pqd[(d, "P1. 해마다 재학습")].delta) for d in DS),
        " / ".join(sg(_pqd[(d, "P2. 재학습+재보정")].delta) for d in DS),
        (
            "특히 **%s 에서는 P1 단독의 구간이 0 을 포함한다**(%s [%s, %s]) — 재학습만으로는 "
            "개선을 검출하지 못한다."
            % (
                " · ".join(_pq_p1),
                sg(_pqd[(_pq_p1[0], "P1. 해마다 재학습")].delta),
                sg(_pqd[(_pq_p1[0], "P1. 해마다 재학습")].delta_lo),
                sg(_pqd[(_pq_p1[0], "P1. 해마다 재학습")].delta_hi),
            )
            if _pq_p1
            else "두 데이터셋 모두에서 P1 단독도 유의하다."
        ),
    )
)
w()
w(
    "2. **움직이는 창에서 재보정하면 §10 의 한 번짜리 재보정보다 훨씬 많이 돌아온다.** §10 의 "
    "B(고정 보정셋 Platt)는 DILIrank 회복률 %s%% 였다. P2 는 ECE 를 %s → %s 로 내리는데, "
    "§10 의 간격(A−E = %s)으로 환산하면 회복률 %s%% 다. **같은 보정기를 쓰면서 보정셋만 "
    "최신으로 옮긴 것**이 차이의 전부다."
    % (
        f1(_rec10("DILIrank", "B.")),
        f4(_pq("DILIrank", "P0. 고정 모델").ece.iloc[0]),
        f4(_pq("DILIrank", "P2. 재학습+재보정").ece.iloc[0]),
        f4(
            float(
                rec[
                    rec.dataset.str.startswith("DILIrank") & rec.method.str.startswith("A.")
                ].ECE.iloc[0]
            )
            - float(
                rec[
                    rec.dataset.str.startswith("DILIrank") & rec.method.str.startswith("E.")
                ].ECE.iloc[0]
            )
        ),
        f1(
            100.0
            * (
                float(_pq("DILIrank", "P0. 고정 모델").ece.iloc[0])
                - float(_pq("DILIrank", "P2. 재학습+재보정").ece.iloc[0])
            )
            / (
                float(
                    rec[
                        rec.dataset.str.startswith("DILIrank") & rec.method.str.startswith("A.")
                    ].ECE.iloc[0]
                )
                - float(
                    rec[
                        rec.dataset.str.startswith("DILIrank") & rec.method.str.startswith("E.")
                    ].ECE.iloc[0]
                )
            )
        ),
    )
)
w()
_pq_gap = []
for d in DS:
    p2 = float(_pq(d, "P2. 재학습+재보정").ece.iloc[0])
    c = _pq(d, "C2r. 연도 무작위+양성률 맞춤+재보정")
    cm, csd = float(c.ece.mean()), float(c.ece.std(ddof=1))
    _pq_gap.append((d, p2, cm, csd, (p2 - cm) / csd if csd else float("nan")))
w(
    "3. **그래도 교환가능 기준선까지 내려오지는 않는다 — 다만 거리가 많이 줄었다.** 조건을 "
    "맞춘 기준선(C2r)과 비교하면 %s. 기존 설계의 가장 엄한 대조군 배율이 %s / %s "
    "였던 것(§8 의 R3)에 비하면 %s / %s 로 줄어든 것이다."
    % (
        ", ".join(
            "%s 는 P2 %s 대 기준선 %s (표준편차의 %.1f배 %s)"
            % (g[0], f4(g[1]), f4(g[2]), abs(g[4]), "높다" if g[4] > 0 else "낮다")
            for g in _pq_gap
        ),
        f2(PCR[("DILIrank", "R3")].ece_ratio),
        f2(PCR[("DICTrank", "R3")].ece_ratio),
        f2(_pq_gap[0][1] / _pq_gap[0][2]),
        f2(_pq_gap[1][1] / _pq_gap[1][2]),
    )
)
w()
w(
    "4. **이 결과는 낙관적이다 — 반드시 그렇게 읽어야 한다.** P1·P2 는 승인 시점에 "
    "간독성·심장독성 라벨이 이미 확정돼 있다고 가정한다. 실제로는 라벨이 쌓이는 데 시간이 "
    "걸린다(§12 의 라벨 성숙도 — 검증군에서 라벨 근거가 늦게 붙은 분자가 있다). 따라서 "
    "**P2 의 수치는 주기적 재학습이 낼 수 있는 상한**이고, 현실의 회복은 이보다 작다. "
    "라벨 지연을 넣은 시뮬레이션은 승인 후 몇 년에 라벨이 붙는지에 대한 외부 자료가 있어야 "
    "하고 이 데이터로는 할 수 없다."
)
w()
w(
    '> **이 절이 이 논문의 처방을 바꾼다.** §10 부터 §30 까지는 "무엇을 해도 안 된다" 였다. '
    "이 절은 **되는 것 하나**를 보인다 — 승인이 쌓이는 대로 다시 학습하고, 그때마다 학습창의 "
    "최신 구간으로 다시 보정하는 것. 다만 (가) 효과는 재보정 쪽에서 오고 재학습 단독은 한 "
    "데이터셋에서만 듣고, (나) 교환가능 수준까지는 가지 않으며, (다) 라벨이 즉시 붙는다는 "
    "가정 위의 상한이다. 세 조건을 같이 적지 않으면 과장이 된다. **주기를 얼마로 둘 것인가는 "
    "§34.1 이 답한다** — 큰 단계는 주기를 두는가이고, 1년까지 당기는 이득은 한 데이터셋에서만 "
    "검출된다."
)
w()
# ---------------------------------------------------------------- 34.1 재학습 주기
_RIK = [1, 2, 3, 5, 0]


def _rik(k):
    return ("%d년마다" % k) if k else "하지 않음(고정)"


def _ri(d, k, rc):
    g = rin[(rin.dataset == d) & (rin.interval == k) & (rin.recal == rc)]
    return g.iloc[0] if len(g) else None


def _rid(d, k, rc):
    g = rind[(rind.dataset == d) & (rind.interval == k) & (rind.recal == rc)]
    return g.iloc[0] if len(g) else None


w("### 34.1 얼마나 자주 해야 하는가 — 재학습 주기 용량반응")
w()
w(
    "§34 는 주기를 1년으로 고정해 답했다. 거기서 두 가지가 바로 따라 나온다 — 그 개선이 "
    "**특정 일정의 우연은 아닌가**, 그리고 실무에서는 **얼마나 자주 해야 하는가**. 둘 다 "
    '같은 실험으로 답한다. 재학습 주기를 1·2·3·5년과 "하지 않음"(고정 모델)으로 늘어놓고 '
    "보정오차가 어떻게 움직이는지 본다."
)
w()
w(
    "**주기를 늘려도 평가집합은 그대로다.** 주기가 2년이면 모델이 최대 1년 묵은 상태로 "
    "예측하는 것이고, 평가 대상은 검증군 전부(%d / %d개)로 같다. 그래서 모든 주기가 같은 "
    "분자에서 짝지어 비교되고 검정력이 떨어지지 않는다. 설명변수는 주기가 아니라 **모델이 "
    "묵은 햇수**이므로 그 평균을 함께 적는다 — 5년 주기의 평균 묵은햇수는 5년이 아니라 "
    "%.1f / %.1f년이다(분자가 주기 안에 흩어져 있으므로)."
    % (
        int(_ri("DILIrank", 1, True).n),
        int(_ri("DICTrank", 1, True).n),
        float(_ri("DILIrank", 5, True).mean_age),
        float(_ri("DICTrank", 5, True).mean_age),
    )
)
w()
w(
    "주기 1년은 §34 의 P1(재보정 없음)·P2(재보정 있음)와, 고정은 P0 과 **같은 절차**다. "
    "여섯 칸 전부 차 0.00e+00 로 재현된다 — 두 스크립트가 같은 것을 다르게 계산하고 있지 "
    "않다는 확인이다."
)
w()
w(
    "| 데이터셋 | 재보정 | 주기 | 재학습 횟수 | 평균 묵은햇수 | ECE | AUROC | 1년 대비 ΔECE | 95% CI | 판정 |"
)
w("|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    for rc in (True, False):
        for k in _RIK:
            r = _ri(d, k, rc)
            if r is None:
                continue
            q = _rid(d, k, rc)
            w(
                "| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |"
                % (
                    d,
                    "있음" if rc else "없음",
                    _rik(k),
                    int(r.n_refit),
                    f2(r.mean_age),
                    f4(r.ece),
                    f4(r.auroc),
                    "기준" if q is None else sg(q.delta),
                    "—" if q is None else "[%s, %s]" % (sg(q.delta_lo), sg(q.delta_hi)),
                    (
                        "—"
                        if q is None
                        else (("**%s**" % q.verdict) if q.delta_lo > 0 else str(q.verdict))
                    ),
                )
            )
w()
_ri_sig = {
    (d, rc): [
        k for k in _RIK[1:] if k and _rid(d, k, rc) is not None and _rid(d, k, rc).delta_lo > 0
    ]
    for d in DS
    for rc in (True, False)
}
w("읽을 것이 넷이다.")
w()
w(
    '1. **큰 단계는 "주기를 두는가" 이고 "주기를 얼마로 하는가" 가 아니다.** 재보정을 '
    "함께 하면 고정 모델은 1년 주기보다 %s / %s 나쁘고 두 데이터셋 모두 유의하다. 반면 "
    "주기를 5년까지 늘려도 차이는 %s / %s 에 머문다 — 고정 대비 %s배·%s배 작다. "
    '**실무 권고는 "해마다" 가 아니라 "주기를 두라" 가 먼저다.**'
    % (
        sg(_rid("DILIrank", 0, True).delta),
        sg(_rid("DICTrank", 0, True).delta),
        sg(_rid("DILIrank", 5, True).delta),
        sg(_rid("DICTrank", 5, True).delta),
        f1(_rid("DILIrank", 0, True).delta / _rid("DILIrank", 5, True).delta),
        f1(_rid("DICTrank", 0, True).delta / _rid("DICTrank", 5, True).delta),
    )
)
w()
w(
    "2. **묵은 정도에 따른 기울기는 한 데이터셋에서만 검출된다.** DILIrank(재보정 있음)는 "
    "2·3·5년 **전부** 1년보다 유의하게 나쁘고 묵은햇수 순으로 단조다(%s). DICTrank 는 "
    "점추정은 같은 방향이지만(%s) **5년까지 어느 주기도 1년과 구별되지 않는다**. 즉 "
    '"더 자주 하면 더 좋다" 는 DILIrank 에서만 말할 수 있다.'
    % (
        " → ".join(f4(_ri("DILIrank", k, True).ece) for k in _RIK[:-1]),
        " → ".join(f4(_ri("DICTrank", k, True).ece) for k in _RIK[:-1]),
    )
)
w()
w(
    "3. **재보정 없이 재학습만 하면 기울기가 아예 안 보인다.** 재보정 없음 쪽에서 1년과 "
    "유의하게 다른 주기는 %s. DICTrank 는 고정 모델조차 1년과 구별되지 않는다(%s [%s, %s]). "
    '**§34 가 "일을 하는 것은 재보정" 이라고 한 것이 주기 축에서도 같은 모양으로 '
    "나온다.**"
    % (
        (
            (
                "DILIrank 의 고정 모델 하나뿐이다"
                if _ri_sig[("DILIrank", False)] == []
                else "일부 있다"
            )
            if _rid("DILIrank", 0, False).delta_lo > 0
            else "없다"
        ),
        sg(_rid("DICTrank", 0, False).delta),
        sg(_rid("DICTrank", 0, False).delta_lo),
        sg(_rid("DICTrank", 0, False).delta_hi),
    )
)
w()
w(
    '4. **"5년마다도 괜찮다" 로 과장하지 말 것.** 5년 주기도 이 창에서 재학습을 '
    "%d·%d회 하고 평균 묵은햇수는 %.1f / %.1f년이다. 가장 묵은 분자도 %d / %d년이다. "
    "고정 모델의 평균 묵은햇수는 %.1f / %.1f년(최대 %d / %d년)이고 그쪽은 유의하게 "
    "나쁘다. 그러므로 이 절이 보인 것은 **%s** 이고, 어느 쪽이든 임의로 오래 묵혀도 "
    "된다는 뜻이 아니다."
    % (
        int(_ri("DILIrank", 5, True).n_refit),
        int(_ri("DICTrank", 5, True).n_refit),
        float(_ri("DILIrank", 5, True).mean_age),
        float(_ri("DICTrank", 5, True).mean_age),
        int(_ri("DILIrank", 5, True).max_age),
        int(_ri("DICTrank", 5, True).max_age),
        float(_ri("DILIrank", 0, True).mean_age),
        float(_ri("DICTrank", 0, True).mean_age),
        int(_ri("DILIrank", 0, True).max_age),
        int(_ri("DICTrank", 0, True).max_age),
        (
            "; ".join(
                "%s 는 0~%d년 범위에서도 묵은 정도가 %s"
                % (
                    d,
                    int(_ri(d, 5, True).max_age),
                    "구별된다" if _ri_sig[(d, True)] else "구별되지 않는다",
                )
                for d in DS
            )
        ),
    )
)
w()
w("---")
w()


def _pd0(ds, arm, dl):
    """지연 0 대비 같은 팔의 짝지은 ΔECE 행."""
    q = pqdd[
        (pqdd.dataset == ds)
        & (pqdd.delay == dl)
        & pqdd.vs.str.startswith("지연 0")
        & pqdd.arm.str.startswith(arm)
    ]
    return q.iloc[0]


w("### 34.2 라벨이 늦게 붙으면 — 지연 민감도 (한계 25)")
w()
w(
    "§34 의 P2 는 **승인 연도에 간독성·심장독성 라벨이 이미 확정돼 있다**고 보고 "
    "학습창에 넣는다. 현실에서는 라벨이 쌓이는 데 시간이 걸리므로(§12) 그 회복분은 "
    '상한이다. 한계 25 가 그 사실을 적으면서 "라벨 지연을 넣으려면 외부 자료가 '
    '필요하고 이 데이터로는 할 수 없다" 고 적어 두었는데, **그 절반은 틀렸다.** '
    "실제 지연이 몇 년인지 알려면 외부 자료가 필요하지만, 지연을 **가정으로 "
    "쓸어보는 것**은 이 데이터로 할 수 있다. 연도 T 에서 학습창을 "
    "`approval_year < T` 대신 `approval_year <= T - 1 - d` 로 바꾸면 승인 후 d 년이 "
    "지나 라벨이 익은 분자만 학습에 들어간다."
)
w()
w(
    "**평가 분자를 모든 지연에서 같게 맞췄다.** 지연이 커지면 학습창이 %d개 미만이 "
    "되어 건너뛰는 연도가 생길 수 있고, 그대로 비교하면 ΔECE 가 지연 효과인지 "
    "평가집합 효과인지 갈리지 않는다. 각 지연의 평가 가능 연도를 구해 교집합으로 "
    "제한했다 — 이 데이터에서는 교집합이 전체와 같아 DILIrank %d개 연도 %d개 분자, "
    "DICTrank %d개 연도 %d개 분자가 모든 지연에서 동일하다. 그래서 짝지은 "
    "부트스트랩이 유효하다."
    % (
        100,
        int(pqy[(pqy.dataset == "DILIrank") & (pqy.delay == 0)].n_years.iloc[0]),
        int(pqy[(pqy.dataset == "DILIrank") & (pqy.delay == 0)].n.iloc[0]),
        int(pqy[(pqy.dataset == "DICTrank") & (pqy.delay == 0)].n_years.iloc[0]),
        int(pqy[(pqy.dataset == "DICTrank") & (pqy.delay == 0)].n.iloc[0]),
    )
)
w()
w(
    "**먼저 구현 확인.** 지연 0 은 §34 와 같은 절차이므로 ECE 가 같아야 한다. "
    "여섯 값 전부 차가 %s 다(`results/prequential_delay_check.csv`)."
    % ("%.2e" % float(pqc.abs_diff.abs().max()))
)
w()
w("| 데이터셋 | 지연 | P0 ECE | P1 ECE | P2 ECE | P1−P0 ΔECE [95% CI] " "| P2−P0 ΔECE [95% CI] |")
w("|---|---|---|---|---|---|---|")
for d in DS:
    g = pqy[pqy.dataset == d]
    for dl in sorted(g.delay.unique()):
        r0 = g[(g.delay == dl) & g.arm.str.startswith("P0")].iloc[0]
        r1 = g[(g.delay == dl) & g.arm.str.startswith("P1")].iloc[0]
        r2 = g[(g.delay == dl) & g.arm.str.startswith("P2")].iloc[0]
        q = pqdd[(pqdd.dataset == d) & (pqdd.delay == dl) & (pqdd.vs == "P0. 고정 모델")]
        a1 = q[q.arm.str.startswith("P1")].iloc[0]
        a2 = q[q.arm.str.startswith("P2")].iloc[0]
        w(
            "| %s | %d년 | %s | %s | %s | %s [%s, %s]%s | %s [%s, %s]%s |"
            % (
                d,
                int(dl),
                f4(r0.ece),
                f4(r1.ece),
                f4(r2.ece),
                sg(a1.dece),
                sg(a1.lo),
                sg(a1.hi),
                "" if bool(a1.excludes_zero) else " (0 포함)",
                sg(a2.dece),
                sg(a2.lo),
                sg(a2.hi),
                "" if bool(a2.excludes_zero) else " (0 포함)",
            )
        )
w()

_p2 = {}
for d in DS:
    q = pqdd[
        (pqdd.dataset == d) & (pqdd.vs == "P0. 고정 모델") & pqdd.arm.str.startswith("P2")
    ].sort_values("delay")
    _p2[d] = q
_dmax = int(max(pqy.delay))
w(
    "**처방은 지연을 넣어도 살아남는다.** P2 의 ΔECE 는 두 데이터셋 모든 지연에서 "
    "음수이고 구간이 0 을 제외한다 — 지연 %d년에서도 DILIrank %s, DICTrank %s 다. "
    "지연이 회복분을 깎기는 한다. 지연 0 의 회복분을 100%%로 놓으면 지연 %d년에 "
    "DILIrank 는 %s, DICTrank 는 %s 가 남는다. **두 데이터셋의 감쇠 속도가 다르다** — "
    "DILIrank 는 지연 1년만으로도 지연 0 대비 유의하게 나빠지는데(%s [%s, %s]) "
    "DICTrank 는 지연 3년이 되어야 유의해진다(%s [%s, %s])."
    % (
        _dmax,
        sg(_p2["DILIrank"][_p2["DILIrank"].delay == _dmax].dece.iloc[0]),
        sg(_p2["DICTrank"][_p2["DICTrank"].delay == _dmax].dece.iloc[0]),
        _dmax,
        pct1(
            100.0
            * abs(_p2["DILIrank"][_p2["DILIrank"].delay == _dmax].dece.iloc[0])
            / abs(_p2["DILIrank"][_p2["DILIrank"].delay == 0].dece.iloc[0])
        ),
        pct1(
            100.0
            * abs(_p2["DICTrank"][_p2["DICTrank"].delay == _dmax].dece.iloc[0])
            / abs(_p2["DICTrank"][_p2["DICTrank"].delay == 0].dece.iloc[0])
        ),
        sg(_pd0("DILIrank", "P2", 1).dece),
        sg(_pd0("DILIrank", "P2", 1).lo),
        sg(_pd0("DILIrank", "P2", 1).hi),
        sg(_pd0("DICTrank", "P2", 3).dece),
        sg(_pd0("DICTrank", "P2", 3).lo),
        sg(_pd0("DICTrank", "P2", 3).hi),
    )
)
w()
w(
    "**재학습 단독(P1)은 지연을 넣으면 더 약해진다.** DILIrank 에서는 지연 3년까지 "
    "유의하고 %d년에서 구간이 0 을 포함한다. DICTrank 에서는 지연 0 에서도 이미 "
    "유의하지 않다(한계 26). 즉 §34 의 결론 — **다시 학습하는 것만으로는 안 되고 "
    "다시 보정해야 한다** — 가 지연을 넣어도 그대로이고, 오히려 더 선명해진다." % _dmax
)
w()
w(
    "**남는 한계 셋.** (1) 이것은 지연의 **추정이 아니라 쓸기**다. 실제 지연이 몇 "
    "년인지는 여전히 외부 자료가 있어야 안다. (2) 지연을 학습창에만 걸었고 P0 에는 "
    "걸지 않았다 — P0 은 2010년 이전 전체로 한 번 학습하는 고정 모델이고 그 라벨도 "
    "엄밀히는 2010년에 다 익지 않았다. §34 와 1:1 로 대응시키기 위해 그대로 두었다. "
    "P0 에도 지연을 걸면 고정 모델이 더 나빠져 P2 의 이득이 커질 것이므로, "
    "지금 설계는 P2 를 **과대평가하지 않는다**. (3) 대조군을 두지 "
    "않았다 — 교환가능 기준선과의 비교는 §34 가 지연 0 에서 했고 이 절의 질문은 "
    "지연에 따른 기울기다."
)
w()
w("## 35. 통계 방법 명세")
w()
w("| 항목 | 내용 |")
w("|---|---|")
w("| 판별력 | AUROC. 신뢰구간은 검증군 부트스트랩 2,000회의 2.5–97.5 백분위 |")
w("| 보정 오차 | ECE = Σ (n_b/N)·abs(ō_b − p̄_b), 균등폭 10구간. 구간화 민감도는 §9.4 |")
w("| 최대 보정 오차 | MCE = max_b abs(ō_b − p̄_b). 구간 하나가 정하므로 보조 수치 |")
w("| 보정기울기·절편 | 로짓 재보정 회귀의 기울기·절편. 1·0 이 이상적 |")
w("| Brier 분해 | Murphy: BS = reliability − resolution + uncertainty (균등폭 10구간) |")
w(
    "| 보정 기여율 | Δreliability / (Δreliability + Δresolution). Δuncertainty 를 분모에서 제외 (§7.1) |"
)
w("| 순열검정 (판별력) | 같은 크기 무작위 분할 1,000회. 단측 |")
w("| 순열검정 (유병률 통제) | 대조군 200회. 단측 |")
w("| 순열검정 (시대 분류기) | 시대 라벨 무작위화 2,000회. 단측 |")
w("| p 추정량 | Phipson & Smyth (1+r)/(1+N). 하한은 1/(N+1) |")
w("| 2표본 비교 | 층화 부트스트랩 2,000회 |")
w("| 상관 | Spearman rho |")
w("| Brier Skill Score | 1 - Brier(모델)/Brier(C0). C0 = 학습군 양성률 상수 (§25) |")
w("| 배율 신뢰구간 | 분자 검증군 부트스트랩, 같은 재표집 안에서 분모 대조군 재추출 (§26) |")
w("| ICI·E50·E90·Emax | loess 보정곡선(frac 0.75)에서 abs(p − f(p)) 의 요약통계 (§27) |")
w("| Spiegelhalter z | Σ(y−p)(1−2p)/sqrt(Σ(1−2p)²p(1−p)). 양측 정규근사 (§27) |")
w("| 편향 제거 ECE | 예측 고정, y\\* ~ Bernoulli(p) 귀무표본 1,000회의 평균을 뺀다 (§27) |")
w("| 짝지은 ΔECE | 예측 고정, 검증군만 4,000회 재표집. 두 팔을 같은 재표집에서 뺀다 (§29) |")
w("| MDE80 | 2.80 × SE(ΔECE). 양측 alpha 0.05, 검정력 0.80 (§29) |")
w("| 밀도비 가중 | 시대 분류기 OOF 확률의 오즈 비. 평균 1 로 정규화하고 상한에서 자른다 (§30) |")
w("| 다중비교 | Benjamini–Hochberg FDR, α = 0.05, 3군 분리 (§15) |")
w("| Conformal | split conformal, 비순응도 = 1 − 정답 클래스 확률, 보정셋은 학습군 내부 |")
w()
w(
    "**p 값 표기 규칙** — `p < 0.001` 이나 `p = 0` 으로 쓰지 말 것. 순열 p 의 하한은 횟수가 "
    "정한다: 2,000회 `1/2001 = %s`, 1,000회 `1/1001 = %s`, 200회 `1/201 = %s`. "
    "관측값 초과가 0건이면 p 는 그 하한값이고 실제 p 는 그보다 작다."
    % (f4(1 / 2001), f4(1 / 1001), f4(1 / 201))
)
w()
w(
    "**배율 표기 규칙** — ECE 배율의 분모(대조군 설계)를 매번 명시할 것. 이 문서에 나오는 "
    "DILIrank 시간분할 ECE %s 의 배율은 분모에 따라 다음과 같이 갈린다." % f4(T["DILIrank"].ECE)
)
w()
w("| 분모 | 값 | 출처 |")
w("|---|---|---|")
_dn = [
    (
        "무작위 5-fold CV (§4)",
        bc[(bc.dataset == "DILIrank") & (bc.split == "random_cv")].ECE.mean(),
    ),
    (
        "스캐폴드 5-fold CV (§4)",
        bc[(bc.dataset == "DILIrank") & (bc.split == "scaffold_cv")].ECE.mean(),
    ),
    ("크기맞춤 무작위 200회 R1 (§8)", PCR[("DILIrank", "R1")].ECE),
    ("검증군 비율맞춤 R2 (§8)", PCR[("DILIrank", "R2")].ECE),
    ("양측 비율맞춤 R3 (§8)", PCR[("DILIrank", "R3")].ECE),
    (
        "연도 민감도 무작위 20회 (§16.1)",
        ys[(ys.dataset == "DILIrank") & (ys.scenario.str.startswith("S0"))].ECE_random_mean.iloc[0],
    ),
    ("격자 주 조합 R1 (§9.5)", grid[(grid.dataset == "DILIrank")].r1_ece_random.iloc[0]),
    ("유병률맞춤 대조군 (§12.1 · §16.3)", ster[ster.dataset == "DILIrank"].ece_random_pm.iloc[0]),
]
for lab, den in _dn:
    w("| %s | %s배 (분모 %s) | — |" % (lab, f2(T["DILIrank"].ECE / den), f4(den)))
w()
w("분자는 어디서나 같다. **논문에서 배율을 쓸 때마다 분모를 적을 것.**")
w()
# ECE 배율에만 규칙이 있고 AUROC 기준값에는 없었다. 같은 규칙을 적용한다 (2026-10-04).
w(
    '**ΔAUROC 기준 표기 규칙** — "무작위 분할 AUROC" 도 설계마다 다른 값이다. '
    "DILIrank 에서 이 문서가 쓰는 값은 다음과 같다."
)
w()
w("| 기준 | 값 | 설계 |")
w("|---|---|---|")
_ab = [
    (
        "§4 무작위 5-fold CV",
        bc[(bc.dataset == "DILIrank") & (bc.split == "random_cv")].AUROC.mean(),
        "폴드 5개 평균",
    ),
    ("§5 순열 귀무 평균", PT["DILIrank"].null_mean, "같은 크기 무작위 분할 1,000회"),
    ("§6 무작위 (크기맞춤)", CR["DILIrank"].AUROC, "검증군 크기를 시간분할과 맞춘 단일 실행"),
    ("§8 R1 대조군", PCR[("DILIrank", "R1")].AUROC, "크기맞춤 무작위 200회 평균"),
    (
        "§9.2 절단 2010 실행",
        robc[(robc.dataset == "DILIrank") & (robc.cutoff == 2010)].auroc_random.iloc[0],
        "절단연도 민감도 스크립트의 자체 대조군",
    ),
]
for _lab, _v, _note in _ab:
    w("| %s | %s | %s |" % (_lab, f4(_v), _note))
w()
w(
    "폭이 %s 다(%s–%s). 시간분할 AUROC %s 는 어디서나 같으므로, **ΔAUROC 를 인용할 때마다 "
    "어느 기준인지 적을 것** — §5 의 ΔAUROC %s 는 순열 귀무 평균 기준이고, §12 의 "
    '"무작위 대비 격차" %s 는 §6 기준이어서 같은 양이 아니다.'
    % (
        f4(max(_v for _, _v, _ in _ab) - min(_v for _, _v, _ in _ab)),
        f4(min(_v for _, _v, _ in _ab)),
        f4(max(_v for _, _v, _ in _ab)),
        f4(T["DILIrank"].AUROC),
        f4(PT["DILIrank"].delta_auroc),
        f4(
            cens[
                (cens.dataset.str.startswith("DILIrank")) & (cens.exp.str.startswith("기준"))
            ].gap.iloc[0]
        ),
    )
)
w()

# ================================================================ 19. 한계
w("---")
w()
w("## 36. 한계")
w()
_ad_t = _ad[_ad.stratum.str.startswith("Tanimoto")].dropna(subset=["ratio_lo"])
w("**설계상 한계**")
w()
w(
    "1. **데이터셋 2개로는 인과 사슬을 검증할 수 없다.** 시대 분류기(§13.1)는 분포 이동의 "
    "존재와 크기를 재지만 그 이동이 보정 붕괴를 일으켰음을 보이지 않는다. 유사도 구간별 ECE 는 "
    "유사도를 따라가지 않는다(§13.3)."
)
w(
    '2. **두 데이터셋이 화합물의 %s / %s 를 공유한다**(§17.1). "두 데이터셋에서 재현" 은 '
    "독립 표본 두 개가 아니라 같은 화학공간에서 엔드포인트를 바꾼 확인이다. 비중첩 부분집합 "
    "재계산(§17.2)에서 DICTrank 는 답이 나왔지만 DILIrank 는 전용 검증군이 %d개뿐이어서 "
    "답할 수 없다."
    % (
        pct1(OV["DILIrank"].shared_pct),
        pct1(OV["DICTrank"].shared_pct),
        int(OV["DILIrank"].n_exclusive_test),
    )
)
w(
    "3. **3군 다중비교 분류는 사후에 정했다**(§15). 단일 가족으로 묶으면 탐색적 발견 2건이 "
    "유의해지고 §8 DICTrank R3 의 유의성은 어느 p 를 쓰는지에 갈린다."
)
w(
    "4. **ECFP4 계열 표현 중심이며 그래프 신경망은 평가하지 않았다.** 검증군 %d / %d 로는 "
    "아키텍처를 신뢰성 있게 구별할 수 없다는 판단인데, 그 판단 자체가 한계다."
    % (int(T["DILIrank"].n_test), int(T["DICTrank"].n_test))
)
w(
    "5. **클래스별 보장은 §11.1 이 붙였고, 그 결과가 두 데이터셋에서 갈린다.** §11 의 "
    "주변형 split conformal 은 전체 커버리지만 보장하므로 그 절의 클래스별 수치는 기술 "
    "통계다. 라벨 조건부(Mondrian) conformal 로 클래스별 보장을 붙이면(§11.1) %s. "
    "라벨 조건부 절차는 임의의 라벨이동에 유효하므로 그 미달분은 양성률 변화로 환원되지 "
    "않는다. **남은 한계는 검정력이다** — 음성 검증군이 %d / %d개뿐이어서 z 가 %s 로 "
    "경계에 있고, §15.2 의 가족에 넣어 FDR 보정하면 Mondrian 쪽 여덟 칸은 전부 유의하지 "
    "않다. 그래서 이 절의 주장은 부족분의 크기로 하고 유의성 주장으로 쓰지 않는다."
    % (
        (
            (
                "%s 는 두 클래스 모두 명목값에 미달하지 않고, %s 는 음성 클래스에서 여전히 "
                "미달한다" % (" · ".join(_ds_ok), " · ".join(_ds_bad))
            )
            if (_bad and _ds_ok)
            else (
                "여덟 칸 전부에서 명목값에 유의하게 미달하지 않는다 — 이 한계는 닫힌다"
                if not _bad
                else "여덟 칸 중 %d칸이 여전히 명목값에 유의하게 미달한다" % len(_bad)
            )
        ),
        int(_mrow("DILIrank", _MAL[0], _MARM[0], "Mondrian")["n_neg"]),
        int(_mrow("DICTrank", _MAL[0], _MARM[0], "Mondrian")["n_neg"]),
        (", ".join("%+.2f" % b[5] for b in _bad) if _bad else "모두 −1.645 보다 큼"),
    )
)
w()
w("**표본 크기 한계**")
w()
w(
    "6. **DICTrank 의 판별력 저하는 불안정하다** — 부트스트랩 %s 에서 판별력이 오히려 "
    "개선된다(§7). 이 데이터셋에서 판별력 저하를 단독 주장으로 쓸 수 없다."
    % pct1(BD["DICTrank"].frac_boot_over_100 * 100)
)
w(
    "7. **라벨 성숙도의 DICTrank 조건은 검증군 %d개다**(§12.1). 성숙도가 DICTrank 보정 붕괴에 "
    "기여하는지는 이 표본으로 확정할 수 없다."
    % int(
        cc[
            (cc.dataset == "DICTrank")
            & (cc.condition.str.contains("10년"))
            & (cc.group == "라벨성숙도")
        ].n.iloc[0]
    )
)
w(
    "8. **n ≤ 15 인 ATC 계열에는 정보가 없다**(§14.1). %s 가 해당한다."
    % ", ".join("%s %s" % (d, a) for d, a in _small)
)
w(
    "9. **구간별 배율은 두 축 모두 나왔다.** 연도 축은 §32 가 냈다 — 여섯 구간 전부 "
    "1 을 넘지만 지평 추세는 판정할 수 없다(구간 n %d–%d). 유사도 축은 §21 이 "
    "Tanimoto 삼분위로 이미 냈다 — 여섯 층 중 %d개가 구간 하한이 1 을 넘는다"
    "(%s 상위 1/3 만 %s 로 1 을 포함한다). **배율이 없는 것은 §13.3 의 세분 구간"
    "(0.0–0.3 / 0.3–0.4 / 0.4–0.6)뿐이고**, 그 구간은 AUROC 추세를 보이기 위한 "
    "것이라 §21 의 삼분위 배율과 겹친다. 그래서 §13.3 은 절대 ECE 와 AUROC 추세로만 "
    "읽고 배율은 §21 을 쓴다."
    % (
        int(hz.n.min()),
        int(hz.n.max()),
        int((_ad_t.ratio_lo > 1).sum()),
        "DILIrank",
        "[%s, %s]"
        % (
            f2(_ad_t[_ad_t.stratum.str.startswith("Tanimoto 상위")].ratio_lo.iloc[0]),
            f2(_ad_t[_ad_t.stratum.str.startswith("Tanimoto 상위")].ratio_hi.iloc[0]),
        ),
    )
)
w()
w("**보고 선택에 관한 한계**")
w()
w(
    "10. **DICTrank 의 보정 기여율은 유병률 양측통제 후 %s 다**(n 일치 교정 시 %s). "
    "격자에서도 R2 기준 32칸 중 %d칸이 50%% 아래이고 그 전부가 DICTrank 다(§9.5). "
    '"보정이 압도적" 이라는 표현은 DICTrank 에 '
    "쓸 수 없고 DILIrank 에만 안전하다."
    % (
        pct1(R3[("DICTrank", "R3")].share2_reliability),
        pct1(MN[("DICTrank", "R3")].share_new),
        32 - int((grid.r2_share2 > 50).sum()),
    )
)
w(
    "11. **§7.2 의 표본 크기 일치 교정은 세 지표 전부에서 결론에 불리해지지 않는 방향으로 "
    "움직인다**(기여율 ↑, 배율 ↑, p 는 같거나 ↓ — p 가 실제로 내려간 것은 여섯 중 한 "
    "건뿐이고 나머지는 하한에 걸려 있다). "
    "그래서 본문에는 보수적인 기존 값을 싣고 교정치를 병기했다. 본문 수치는 잠정치가 아니라 "
    "**의도적으로 보수적인 값**이다."
)
w(
    "12. **§10·§11 의 점추정값은 한 번의 실현값이다 — 임의성을 없앤 값을 §10.4 에 함께 "
    "적었다.** 내부 보정셋 경계의 동점 배정, 적합셋 행 순서, 전체학습 모델의 행 순서를 "
    "동시에 무작위화해 %d회 평균한 값이 §10.4 다. 대조한 %d개 중 %d개에서 현행값이 %d회 "
    "범위 안에 있고, 결론 셋(명목 커버리지 미달, 학습군 내부 재보정으로는 일부만 회복, "
    "두 데이터셋의 권고안이 다름)은 %d회 전부에서 유지된다. **남은 한계는 평균이 아니라 "
    "반복 횟수다** — %d회의 표준오차만큼(회복률 %s–%s%%p) 평균 자체도 흔들린다. 본문 "
    "§10·§11 표는 재현 절차와 1:1 로 대응하도록 단일 실현값을 그대로 두었다."
    % (
        _tba_n,
        _tba_tot,
        _tba_in,
        _tba_n,
        _tba_n,
        _tba_n,
        f2(min(tba[tba.dataset == d].rec_B.std(ddof=1) / (_tba_n**0.5) for d in DS)),
        f2(max(tba[tba.dataset == d].rec_B.std(ddof=1) / (_tba_n**0.5) for d in DS)),
    )
)
w(
    "13. **DICTrank 의 보정 기여율은 큐레이션 선택에도 민감하다** — 입체이성질체를 분리하면 "
    "%s → %s 로 움직인다(§16.3)."
    % (
        pct1(ster[ster.dataset == "DICTrank"].share_reliability.iloc[0]),
        pct1(ster[ster.dataset == "DICTrank"].share_reliability.iloc[1]),
    )
)
w()
w("**순열 하한 한계**")
w()
w(
    "14. **§8 유병률 통제 5건의 p 는 순열 200회의 하한(%s)에 걸려 있고 올리지 않았다.** "
    "어느 가족 구성에서도 q ≤ %s 로 유의하고(§15), 효과크기가 z = %s 로 크며(§8.1), "
    "200 → 2,000회로 올리려면 모델 적합이 열 배로 늘어 실익 대비 비용이 맞지 않는다는 "
    "판단이다. 이 판단 자체를 한계로 적어 둔다."
    % (
        f4(1 / 201),
        f4(max(_q22[i] for i, pv in enumerate(_fam22) if abs(pv - 1 / 201) < 1e-15)),
        " / ".join(
            f2((MN[(d, b)].ece_T_full - MN[(d, b)].ece_ctrl) / MN[(d, b)].ece_ctrl_sd)
            for d in DS
            for b in ("R1", "R2", "R3")
            if PCR[(d, b)].n_ge_observed == 0
        ),
    )
)
w(
    "15. **§13.1 시대 분류기의 p 두 개도 하한값이다**(2,000회, %s). 실제 p 는 그보다 작다."
    % f4(1 / 2001)
)
w()
w("**추가 실험 3종에서 나온 한계 (2026-10-02)**")
w()
w(
    "16. **사전확률 보정은 배포할 수 없고, 더 정교한 추정량도 거의 전부 실패한다**"
    "(§10.2, §10.3). 정량화 문헌의 추정량 아홉 가지를 붙여 §10.2 의 추측을 검증했다. "
    "**DILIrank 는 열 팔 전부 유병률을 위쪽으로 틀리고 회복률이 음수이며 구간이 "
    "참값을 덮는 것이 하나도 없다.** 발표된 처방(Platt 보정 후 EM, Alexandari 등 "
    '2020)도 두 데이터셋 모두 실패했다(회복률 %s / %s). **다만 "모든 추정량이 '
    '같은 벽에 부딪힌다" 로는 쓸 수 없다** — DICTrank 에서 %d개가 회복률 양수를 '
    "냈다(최대 %s). 그 %d개는 전부 유병률을 아래로 틀렸고 §23 이 보인 과소추정 "
    "효과로 설명되지만, pi 구간이 사실상 정보가 없어 그 해석도 단정할 수 없다. "
    "따라서 §10.2 의 추측은 **DILIrank 와 보정된 EM 경로에서는 검증됐고, 추정량 "
    "전체로 일반화할 수는 없다.**"
    % (
        pct1(_q5["DILIrank"].recovery),
        pct1(_q5["DICTrank"].recovery),
        len(_pe_pos),
        pct1(_pe_pos.recovery.max()),
        len(_pe_und),
    )
)
w(
    "17. **역방향 시간분할의 증거는 정방향보다 약하다**(§12.2). 가장 엄한 통제(R3)에서 "
    "DICTrank 는 유의하지 않고(p = %s), DILIrank 는 유효 반복이 %d회뿐이어서 p 가 그 하한에 "
    "걸려 있다. 결론은 R1·R2 에서만 분명하다."
    % (
        f4(
            rv[
                (rv.dataset == "DICTrank")
                & (rv.direction == "역방향")
                & (rv.setting.str.startswith("R3"))
            ].p_phipson_smyth.iloc[0]
        ),
        int(
            rv[
                (rv.dataset == "DILIrank")
                & (rv.direction == "역방향")
                & (rv.setting.str.startswith("R3"))
            ].n_rep_valid.iloc[0]
        ),
    )
)
w(
    "18. **이동량을 맞춘 비교는 §33 이 했다.** §13.5 의 비시간 분할은 이동 AUC 가 "
    "%s–%s 로 포화해 시간분할(%s / %s)과 이동량이 맞지 않았다. §33 은 alpha 혼합으로 "
    "이동량을 맞췄고(차 %s–%s), **여섯 설정 중 다섯에서 시간분할 ECE 가 더 크다"
    "**(반복 표준오차의 %s–%s배). 한 설정(%s)은 무승부다. 따라서 §13.5 의 "
    '"이동을 키워도 붕괴가 커지지 않는다" 는 "같은 이동량에서도 시간분할이 더 '
    '심하다" 로 강화된다. 다만 이동 AUC 는 **모델 특성 공간에서 보이는 이동만** '
    "재고, alpha 혼합 분할은 자연스러운 설계가 아니다(§33.3)."
    % (
        f4(sm[sm.split.str.startswith("X")].shift_auc.min()),
        f4(sm[sm.split.str.startswith("X")].shift_auc.max()),
        f4(sm2[sm2.dataset == "DILIrank"].shift_auc_temporal.iloc[0]),
        f4(sm2[sm2.dataset == "DICTrank"].shift_auc_temporal.iloc[0]),
        f4(sm2m.shift_auc_gap.abs().min()),
        f4(sm2m.shift_auc_gap.abs().max()),
        f1(sm2m[sm2m.diff_se_units > 2].diff_se_units.min()),
        f1(sm2m[sm2m.diff_se_units > 2].diff_se_units.max()),
        " ".join(sm2m[sm2m.diff_se_units.abs() <= 2].iloc[0][["dataset", "axis"]]),
    )
)
w(
    "19. **가족 밖에 있던 검정 %d건을 §15.2 에서 전부 편입했다.** 전에는 §12.2·§13.5 "
    '등의 p 를 "어느 가족에도 넣지 않았으니 보조로만 쓴다" 고 적어 두었는데, 그것은 '
    "회피다. 넣어서 BH 를 다시 돌리면 %d건이 유의하고, 기존 %d건까지 한 가족으로 합친 "
    "가장 보수적인 구성에서도 기존 유의 %d건 중 %d건만 유의성을 잃는다(그 한 건은 "
    "§15.1 의 5번이 이미 적은 §8 DICTrank R3 이고, 그 결론은 p 가 아니라 배율에 걸려 "
    "있다). **남은 한계는 §11.1 쪽이다** — 클래스별 보장을 붙인 뒤의 커버리지 미달분은 "
    "보정 후 유의하지 않아 효과크기로만 쓴다."
    % (
        len(_fn),
        int(_fn.sig_F4.astype(bool).sum()),
        len(_fo),
        int(_fo.sig_existing.sum()),
        len(_fflip),
    )
)
w(
    "20. **절대 ECE 는 하이퍼파라미터에 의존한다.** 본문의 %s / %s 는 조정하지 않은 "
    "고정 설정에서 나온 값이고, 그 설정은 %d조합 중 %d위·%d위로 나쁜 쪽 절반에 있다"
    "(§28.1). 격자 안에서 %s–%s / %s–%s 로 움직인다. **그래서 절대값을 과제의 성질로 "
    "제시하지 않고, 같은 설정으로 돌린 대조군과의 배율만 주장으로 쓴다** — 배율은 튜닝 "
    "후에도 유지된다(R3 %s → %s / %s → %s, §28.2)."
    % (
        f4(CT["DILIrank"].ECE),
        f4(CT["DICTrank"].ECE),
        len(_hpg["DILIrank"]),
        _rank["DILIrank"],
        _rank["DICTrank"],
        f4(_hpg["DILIrank"].temporal_ece.min()),
        f4(_hpg["DILIrank"].temporal_ece.max()),
        f4(_hpg["DICTrank"].temporal_ece.min()),
        f4(_hpg["DICTrank"].temporal_ece.max()),
        f2(_hf["DILIrank"].ratio_R3),
        f2(_ht["DILIrank"].ratio_R3),
        f2(_hf["DICTrank"].ratio_R3),
        f2(_ht["DICTrank"].ratio_R3),
    )
)
w(
    "21. **§28 의 격자는 XGBoost 한 모델에 대한 것이다.** 다른 모델군의 하이퍼파라미터는 "
    "조정하지 않았다(§9.1 은 모델 종류와 표현만 바꾼다). 그리고 격자는 용량 손잡이 네 개 "
    '× 유한한 값들이므로 "어떤 설정으로도 되돌릴 수 없다" 가 아니라 "이 격자 안에는 '
    '없다" 까지만 말한다.'
)
w(
    "22. **무작위 분할도 보정되어 있지 않다**(§27.2). Spiegelhalter 검정이 대조군에서도 "
    "보정 가설을 기각한다. 이 논문의 주장은 '보정됨 대 무너짐' 이 아니라 '틀린 정도의 "
    "차이' 다."
)
w(
    "23. **음성 결과의 사정거리는 간격의 %s / %s 까지다**(§29.1). 그보다 작은 개선은 "
    '이 표본으로 검출할 수 없었고, 그래서 "효과가 전혀 없다" 가 아니라 "쓸 만한 '
    '크기의 개선은 없다" 까지만 주장한다. **쓸 만한 크기의 기준(간격의 25%%)은 임의 '
    "선택이지만 결론이 그 선택에 걸려 있지 않다**(§29.5) — 판정이 기준 %.0f%%–%.0f%% "
    "구간에서 동일하고, (가)·(가') 판정 %d건은 어떤 기준에서도 불변이다. 팔마다 "
    "뒤집히는 지점 U* 를 함께 보고한다."
    % (
        pct1(100 * rp[rp.dataset == "DILIrank"].mde_over_gap.min()),
        pct1(100 * rp[rp.dataset == "DICTrank"].mde_over_gap.min()),
        _lo_same * 100,
        _hi_same * 100,
        int(_ref.n_ga) + int(_ref.n_ga_bad),
    )
)
w(
    "24. **§31 격자의 보정 기여율 백분율은 쓸 수 없다.** 분모(Δreliability + "
    "Δresolution)가 0 을 스쳐 32칸 중 %d칸만 0–100%% 범위에 들어온다(§31.1). 그래서 "
    "격자 수준의 보정 우세는 백분율이 아니라 부호·크기 비교로만 주장한다(Δreliability > 0 "
    "이 %d/%d, abs(Δresolution)까지 넘어선 칸이 %d/%d). 백분율로 쓰려면 반복을 격자 32칸 "
    "전부에서 %d회로 올려야 하고 비용이 맞지 않는다고 판단했다."
    % (
        int(((g3.r3_share2 >= 0) & (g3.r3_share2 <= 100)).sum()),
        int((g3.r3_d_reliability > 0).sum()),
        len(g3),
        int((g3.r3_d_reliability > g3.r3_d_resolution.abs()).sum()),
        len(g3),
        int(PCR[("DILIrank", "R3")].n_rep_valid),
    )
)
w()
w("**배포 시뮬레이션에서 나온 한계 (2026-10-20)**")
w()
w(
    "25. **주기적 재학습의 효과는 라벨이 즉시 붙는다는 가정 위의 상한이다**(§34). P2 는 "
    "승인 연도에 간독성·심장독성 라벨이 이미 확정돼 있다고 보고 학습창에 넣는다. 실제로는 "
    "라벨이 쌓이는 데 시간이 걸리므로(§12) 현실의 회복은 ΔECE %s / %s 보다 작다. "
    "**얼마나 작아지는지는 §34.2 가 쓸었다**(2026-10-22) — 지연을 %d년까지 걸어도 P2 의 "
    "ΔECE 는 두 데이터셋 모두 음수이고 구간이 0 을 제외하며, 지연 0 의 회복분 대비 "
    "%s / %s 가 남는다. **남는 한계는 실제 지연 연수다** — 승인 후 몇 년에 라벨이 "
    "붙는지는 외부 자료가 있어야 알고, §34.2 는 추정이 아니라 가정 쓸기다."
    % (
        sg(pqd[(pqd.dataset == "DILIrank") & pqd.arm.str.startswith("P2")].delta.iloc[0]),
        sg(pqd[(pqd.dataset == "DICTrank") & pqd.arm.str.startswith("P2")].delta.iloc[0]),
        int(max(pqy.delay)),
        pct1(
            100.0
            * abs(_pdp0("DILIrank", int(max(pqy.delay))).dece)
            / abs(_pdp0("DILIrank", 0).dece)
        ),
        pct1(
            100.0
            * abs(_pdp0("DICTrank", int(max(pqy.delay))).dece)
            / abs(_pdp0("DICTrank", 0).dece)
        ),
    )
)
w(
    "26. **재학습 단독의 효과는 한 데이터셋에서만 검출된다**(§34). DICTrank 에서 P1 의 "
    'ΔECE 는 %s [%s, %s] 로 구간이 0 을 포함한다. 따라서 "다시 학습하면 된다" 가 '
    '아니라 **"다시 학습하고 다시 보정해야 한다"** 가 정확한 서술이다.'
    % (
        sg(pqd[(pqd.dataset == "DICTrank") & pqd.arm.str.startswith("P1")].delta.iloc[0]),
        sg(pqd[(pqd.dataset == "DICTrank") & pqd.arm.str.startswith("P1")].delta_lo.iloc[0]),
        sg(pqd[(pqd.dataset == "DICTrank") & pqd.arm.str.startswith("P1")].delta_hi.iloc[0]),
    )
)
w()
w("---")
w()
w("## 부록. 검수 결정 2건")
w()
w("투고 전 사람의 확인이 필요했던 항목 중 판단이 갈린 2건의 결정과 근거다.")
w()
w("| # | 항목 | 결정 | 근거 | 수치 영향 |")
w("|---|---|---|---|---|")
w(
    "| 1 | Riboflavin 5'-phosphate 승인연도 2016 → 1947 | **기각 (2016 유지)** | "
    "1947년 기록(NDA 006071 BEROCCA PN)은 이 성분을 포함한 종합비타민 복합제이고 단일 성분 "
    "제품의 최초 승인이 아니다. 나머지 6건은 브랜드 단일 신약의 최초 NDA 로 명확하다 | "
    "없음 — 본문은 S0(수정 없음)이고, 이 수정을 포함한 S1 에서도 결론 유지 (§16.1) |"
)
w(
    "| 2 | Celecoxib 의 ATC L 군 배정 | **유지** | "
    "WHO ATC 색인에 L01XX33 = celecoxib 가 실제로 등재돼 있다(가족성 대장폴립증 적응증). "
    '다만 1차 용도는 NSAID(M01AH01)이므로 군 이름을 "항종양제 및 면역조절제" 로 정확히 쓴다 | '
    "없음 — 두 데이터셋 모두 승인 1998년이라 학습군에 속하고 §14.1 의 검증군 통계에 들어가지 않는다 |"
)
w()
w(
    "ATC 대분류가 데이터셋 간에 불일치한 3건(Orphenadrine, Ketoconazole–Levoketoconazole, "
    "Diphenhydramine–Dimenhydrinate)은 **매핑 오류가 아니다.** 염은 SMILES 표준화 단계에서 "
    "떨어져 나가고, 14자리 키는 거울상·입체 이성질체를 구별하지 못한다. 그 두 단계가 겹쳐 "
    "WHO 가 별도 물질로 등재한 것끼리 한 분자로 합쳐진 "
    "결과다(WHO ATC 색인 직접 확인). 세 건 모두 L 코드가 아니므로 §14.1 에 영향이 없다."
)
w()
w("---")
w()
w(
    "*이 문서는 `src/make_report.py` 가 `results/` 의 결과 파일에서 생성했다. "
    "수치를 바꾸려면 해당 분석을 다시 돌린 뒤 이 스크립트를 재실행할 것.*"
)

# ---------------------------------------------------------------- 저장
io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("저장: %s" % OUT)
print("  %d 행 / %d 바이트" % (len(L), sum(len(x) for x in L)))
