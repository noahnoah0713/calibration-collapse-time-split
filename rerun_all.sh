#!/bin/bash
# =====================================================================
# 전체 재실행 — 논문의 모든 수치를 data/processed 에서 다시 만든다.
#
#   bash rerun_all.sh
#
# 주의 — 이 스크립트는 results/ 의 파일을 **덮어쓴다.** 논문에 실린 값을
#   보존하려면 먼저 사본을 떠 둘 것:  cp -r results results_논문값
#   전체 실행은 순열·대조군 반복(대부분 200~2,000회) 때문에 수 시간이 걸린다.
#   일부만 빠르게 보려면 해당 스크립트를 개별로 돌리고, 반복 수 옵션을 받는
#   스크립트에는 --reps 를 직접 넘길 것 (예: python src/prevalence_control.py --reps 10).
#
# 재현에 필요한 것
#   - data/processed/*.csv  (저장소에 포함. 원자료 data/raw 는 불필요)
#   - requirements.txt 의 패키지
#   원자료를 직접 처리하려면 README 의 '원자료에서 시작하기' 를 볼 것.
#     data/raw 를 읽는 스크립트는 5개다 — map_approval_year.py,
#     dictrank_pipeline.py, ev_years.py, check_years.py, check_labels.py.
#     map_approval_year.py 와 dictrank_pipeline.py 는 이 스크립트가 부르지
#     않는다(원자료 전처리 단계). ev_years.py / check_years.py /
#     check_labels.py 는 data/raw 가 없으면 자동으로 건너뛴다 — run_raw.
#     그래서 results/year_evidence.csv 와 year_corrections.csv 는 raw 없이는
#     다시 만들 수 없고, 저장소에 들어 있는 사본을 그대로 쓴다(§16.1 근거표).
#
# 2026-10-19 전면 개정: 이전 판은 26개만 돌려 논문 결과의 절반이
#   생성되지 않았다. 보고서·표가 읽는 결과를 만드는 스크립트를 전부
#   넣고, 마지막에 누락 파일을 자동 점검하도록 했다.
# =====================================================================
set -u
# tee 로 파이프하면 파이프라인의 종료코드가 tee 의 것이 된다. pipefail 이 없으면
# 아래 run() 의 실패 감지와 마지막 누락 점검이 **절대 작동하지 않는다.**
set -o pipefail
cd "$(dirname "$0")"
LOG="results/rerun_all_log.txt"
mkdir -p results
start=$(date +%s)
# (이전 판에 있던 --quick 플래그는 실제로 어느 run 에도 전달되지 않아
#  전체 실행과 동일하게 동작했다. 거짓 약속이므로 제거했다.)
if [ "${1:-}" != "" ]; then
  echo "이 스크립트는 인자를 받지 않는다: ${1}"
  echo "반복 수를 줄이려면 개별 스크립트에 --reps 를 직접 넘길 것."
  exit 2
fi
: > "$LOG"

run () {
  local script="$1"; shift
  if [ ! -f "src/$script" ]; then
    echo "!!! src/$script 없음 - 건너뜀" | tee -a "$LOG"; return 0
  fi
  echo "" | tee -a "$LOG"
  echo "=====================================================" | tee -a "$LOG"
  echo ">>> $script $*   시작 $(date '+%H:%M:%S')" | tee -a "$LOG"
  echo "=====================================================" | tee -a "$LOG"
  local t0=$(date +%s)
  if python "src/$script" "$@" 2>&1 | tee -a "$LOG"; then
    echo "<<< $script 완료 ($(( $(date +%s) - t0 ))초)" | tee -a "$LOG"
  else
    echo "!!! $script 실패 - 중단" | tee -a "$LOG"; exit 1
  fi
}

# data/raw 가 있어야만 돌아가는 원자료 대조 감사. 저장소에는 raw 가 없으므로
# 없을 때는 실패가 아니라 건너뛴다. (이전 판은 여기서 exit 1 로 멈췄다.)
run_raw () {
  if [ ! -d "data/raw" ]; then
    echo "" | tee -a "$LOG"
    echo ">>> $1 건너뜀 — data/raw 없음 (원자료 대조 감사는 선택 단계다)" | tee -a "$LOG"
    return 0
  fi
  run "$@"
}

# ---------------------------------------------------------- 0 전처리 필터
run apply_structure_filter.py
run apply_label_conflict_filter.py

# ---------------------------------------------------------- 1 주 분석
run temporal_split.py
run calibration.py
run baseline_cv.py
run brier_decomposition.py
run brier_decomposition_cutoff.py
run brier_decomposition_r3.py
run prevalence_control.py
run recalibration.py
run recalibration_sensitivity.py
run extra_calibrators.py     # 보정기 4종 추가 (베타·Venn-ABERS·온도·합성)
run prior_estimation.py
run conformal_prediction.py
run mondrian_conformal.py    # 라벨 조건부 conformal (한계 5)
run prequential_retraining.py  # 해마다 재학습하는 배포 시뮬레이션
run retrain_interval.py      # 재학습 주기 용량반응 (1·2·3·5년·고정)
run prequential_delay.py     # 라벨 지연 민감도 (한계 25, 2026-10-22)
run censoring.py

# ---------------------------------------------------------- 2 강건성
run robustness.py
run grid_calibration.py
run seed_sensitivity.py
run split_seed_sensitivity.py
run ece_binning_sensitivity.py
run tiebreak_sensitivity.py
run stereo_sensitivity.py
run year_sensitivity.py
run calibration_confounders.py

# ---------------------------------------------------------- 3 분포 이동
run shift_quantification.py
# 2026-10-21: 인자를 안 주면 기본 50회로 떨어진다. 논문 수치는 200회
#   대조군에서 나온 것이었는데 10-20 전체 재실행이 50회로 덮어써
#   p 하한이 0.0050 에서 0.0196 으로 올라가 있었다. 명시한다.
run shift_matched_split.py --reps 200
run era_permutation.py
run matched_n_control.py
run reverse_split.py
run overlap_count.py
run disjoint_subset.py

# ---------------------------------------------------------- 4 기전·처방 (신규)
run concept_shift.py
run prevalence_sweep.py
run recency_weighting.py
run applicability_domain.py
run trivial_baseline.py
run decision_curve.py
run class_weighting.py
run label_threshold.py
run ratio_intervals.py
run calibration_metrics.py   # 구간화 비의존 지표 (ICI·Spiegelhalter·편향제거)
run hparam_sensitivity.py    # 하이퍼파라미터 54조합
run remedy_power.py          # 음성 결과의 최소 검출 가능 개선폭
run grid_r3.py               # 헤드라인 배율을 32칸 격자 전체에서
run horizon_ratio.py         # 예측 지평별 배율 (지평 효과 대 양성률 효과)
run covariate_shift_weighting.py   # 도메인 분류기 밀도비 가중
# 2026-10-22: 거친 격자(0.0~1.0 여섯 단)만으로는 이동 AUC 맞춤이
#   여섯 설정 전부 실패한다(차 0.063~0.081). 논문 §33 의 수치는
#   맞춤 구간을 촘촘히 다시 돌린 2차 탐색에서 나왔는데, 그 호출이
#   손으로만 돌아가 재현 경로에 없었다. 두 줄로 명시한다.
run shift_matched_v2.py      # 1차: 거친 alpha 격자 (한계 18)
run shift_matched_v2.py --alphas 0.29,0.31,0.33,0.46,0.48,0.5  # 2차: 맞춤 구간 정밀 탐색
run useful_threshold_sweep.py  # 쓸 만한 개선 기준의 민감도 (한계 23)
run prior_estimation_v2.py   # 정량화 추정량 9종 (한계 16)
run tiebreak_averaged.py     # 점추정값의 임의성 제거, 평균 50회 (한계 12)

# ---------------------------------------------------------- 5 감사·검정
run check_determinism.py     # 결정성 5종 (모델·표현·지표 정의 단일성 + 프로세스 내·간 재현)
run check_smiles.py
run_raw ev_years.py
run_raw check_years.py
run_raw check_labels.py
run check_atc.py
run atc_v_sensitivity.py
run atc_kegg.py
run subgroup_calibration.py  # 하위집단을 보정 지표로 (§14 가 AUROC 뿐)
run stats_analysis.py
run permutation_test.py
run multiple_comparison_correction.py
run fdr_all_tests.py         # 가족 밖 검정 전수 편입 (한계 19)

# ---------------------------------------------------------- 6 문서 생성
# verify10 이 셀 수를 파일로 남기고 보고서 머리글이 그것을 읽으므로
# 보고서를 두 번 만든다. 첫 번째는 verify10 을 돌리기 위한 것이다.
# 2026-10-22: 이 호출이 전에는 두 번째 verify10 뒤(78번)에 있었다.
#   make_report 가 precision_fix_20261020.csv 를 **읽는데** 그 파일을
#   만드는 스크립트가 보고서보다 뒤에 있었다 — 갓 받은 저장소에서
#   돌리면 보고서가 저장소에 들어 있던 사본을 쓰고, 재생성된 값은
#   다음 실행에서야 반영된다. 보고서 앞으로 옮긴다.
run record_precision_fix.py  # 2026-10-20 정밀도 수정 전·후 기록 (백업 대조)

run make_filter_lists.py   # 큐레이션 필터가 제거한 화합물 목록 (보충 표 S13·S15)
run make_report.py
run make_curation_chain.py    # 제외 경로 집계 -> results/curation_chain.csv
                              # (make_figures 의 Fig S2 가 이 CSV 를 읽는다)
run make_tables.py
run make_figures.py
run make_fig_floats.py        # figures.tex -> manuscript/floats/fig_*.tex

# ---------------------------------------------------------- 7 검증
run verify9.py
run verify10.py
run verify11.py
run make_report.py          # verify10 셀 수를 머리글에 반영
run verify9.py              # 반영 후 재검증
run verify10.py
run make_manuscript.py      # 투고본 한 파일 조립 (main.tex -> submission.tex)
run verify12.py             # 원고 산문 수치 대조
run verify13.py             # 표가 저널 판면에 들어가는지 pdflatex 로 실측
run verify14.py             # 그림이 저널 판면 폭으로 그려졌는지 확인
run verify_state.py --save

# ---------------------------------------------------------- 8 누락 점검
echo "" | tee -a "$LOG"
echo "=====================================================" | tee -a "$LOG"
echo ">>> 누락 점검 — 보고서·표가 읽는 파일이 모두 있는가" | tee -a "$LOG"
echo "=====================================================" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import io, os, re
need = set()
for f in ("src/make_report.py", "src/make_tables.py"):
    t = io.open(f, encoding="utf-8").read()
    need |= set(re.findall(r'cv\("([a-z0-9_]+\.csv)"\)', t))
    need |= set(re.findall(r'txt\("([a-z0-9_]+\.txt)"\)', t))
t = io.open("src/make_figures.py", encoding="utf-8").read()
need |= set(re.findall(r'RES / "([a-z0-9_]+\.csv)"', t))
miss = sorted(n for n in need if not os.path.exists(os.path.join("results", n)))
print("보고서·표가 읽는 결과 파일 %d개" % len(need))
if miss:
    print("!!! 누락 %d개:" % len(miss))
    for m in miss:
        print("   ", m)
    raise SystemExit(1)
print("누락 없음 — 재현 완료")
PY
rc=${PIPESTATUS[0]}

echo "" | tee -a "$LOG"
echo "총 소요 $(( $(date +%s) - start ))초" | tee -a "$LOG"
[ $rc -ne 0 ] && echo "!!! 누락 파일이 있다. 위 목록을 확인할 것." | tee -a "$LOG"
exit $rc
