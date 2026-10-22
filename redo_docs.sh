#!/usr/bin/env bash
# 문서·표·그림을 다시 만들고 검증기를 순서대로 돌린다. 몇 분이면 끝난다.
# verify10 이 셀 수를 파일로 남기고 보고서 머리글이 그것을 읽으므로
# 보고서를 두 번 만든다 (rerun_all.sh 의 꼬리와 같은 순서다).
set -u
cd "$(dirname "$0")"
LOGF="results/redo_docs_log.txt"
exec > >(tee "$LOGF") 2>&1
for step in make_report make_tables make_figures verify9 verify10 verify11 \
            make_report verify9 verify10 make_manuscript verify12; do
  printf "%-18s " "$step"
  if python "src/$step.py" > "/tmp/redo_$step.out" 2>&1; then
    echo "OK   $(grep -hE '실패 [0-9]+건|대조한 행·칸|절 [0-9]+개|옛 수치 합계' "/tmp/redo_$step.out" | tr '\n' ' ')"
  else
    echo "실패"; tail -25 "/tmp/redo_$step.out"; exit 1
  fi
done
python src/verify_state.py --save > /dev/null 2>&1 || true
echo
grep -h "재고" /tmp/redo_verify9.out || true
echo "끝났다."
