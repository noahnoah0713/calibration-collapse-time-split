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
LOG="results/rerun_rest_log.txt"
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

# ====== 이어돌리기: matched_n_control.py 까지(31개) 는 2026-10-20 21:05 시작한
#        전체 재실행에서 이미 끝났다. M3(시대 분류기 순열 1000회) 는
#        era_permutation.py 가 같은 검정을 2000회로 돌려 두었고 관측값이
#        비트 단위로 같으므로(0.683523637820513 / 0.7917040049631212)
#        생략한다. matched_n_era_permutation.csv 는 기존 사본을 쓴다.

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
run shift_matched_v2.py      # 이동량을 맞춘 비시간 분할 (한계 18)
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
run make_report.py
run make_tables.py
run make_figures.py

# ---------------------------------------------------------- 7 검증
run verify9.py
run verify10.py
run verify11.py
run make_report.py          # verify10 셀 수를 머리글에 반영
run verify9.py              # 반영 후 재검증
run verify10.py
run record_precision_fix.py  # 2026-10-20 정밀도 수정 전·후 기록 (백업 대조)
run make_manuscript.py      # 투고본 한 파일 조립 (main.tex -> submission.tex)
run verify12.py             # 원고 산문 수치 대조
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
