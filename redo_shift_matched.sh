#!/usr/bin/env bash
# 2026-10-21. shift_matched_split 을 200회로 다시 돌리고 문서를 다시 만든다.
#
# 왜 — 10-20 전체 재실행이 이 스크립트를 인자 없이 불러 반복이 기본 50회로
# 떨어졌다. 논문 수치는 200회에서 나온 것이었고, 50회에서는 순열 p 하한이
# 0.0050 에서 0.0196 으로 올라가 §13.5 의 열 칸 중 p 가 하한에 걸린 것이
# 세 개가 된다. rerun_all.sh 에는 --reps 200 을 명시해 두었다.
# 약 70분 (50회에서 17분이었다).
set -u
cd "$(dirname "$0")"
LOGF="results/redo_shift_matched_log.txt"
exec > >(tee "$LOGF") 2>&1
die () { echo "!!! [$1] 실패 (종료코드 $2). 멈춘다."; exit 1; }

B=_backup/shift_matched_50rep_20261021
mkdir -p "$B"
[ -f "$B/shift_matched_split.csv" ] || cp results/shift_matched_split.csv "$B/"

echo "---- shift_matched_split.py --reps 200 ----"
python src/shift_matched_split.py --reps 200 || die shift_matched_split $?

echo "---- 50회판과 비교 ----"
python - <<'PY' || true
import pandas as pd
a = pd.read_csv("_backup/shift_matched_50rep_20261021/shift_matched_split.csv")
b = pd.read_csv("results/shift_matched_split.csv")
k = ["dataset", "split"]
m = a.merge(b, on=k, suffixes=("_50", "_200"))
for c in ("n_rep_valid", "n_ge", "p_phipson_smyth", "ece_ratio", "share_rel"):
    if c + "_50" not in m:
        continue
    print("\n[%s]" % c)
    for _, r in m.iterrows():
        print("  %-10s %-24s %12.6f -> %12.6f"
              % (r.dataset, r.split, r[c + "_50"], r[c + "_200"]))
PY

echo "---- 문서·표·그림 재생성과 검증 ----"
for step in make_report make_tables make_figures verify9 verify10 verify11 \
            make_report verify9 verify10 verify12; do
  echo "-- $step"
  python "src/$step.py" > /tmp/redo_$step.out 2>&1 || {
    tail -25 /tmp/redo_$step.out; die "$step" $?; }
  tail -4 /tmp/redo_$step.out
done
python src/verify_state.py --save > /dev/null 2>&1 || true
echo
echo "끝났다. 50회판은 $B 에 있다."
