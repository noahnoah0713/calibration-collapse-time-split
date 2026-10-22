#!/usr/bin/env bash
# 2026-10-21. calibration.calib_slope_intercept 의 float64 캐스팅 누락을
# 고친 뒤 보정 표를 다시 만들고 바뀐 칸을 자리별로 보여준다. 약 1분.
#
# 왜 이렇게 번거롭게 하나 — rerun_rest.sh 가 돌고 있고 그 안의
# prevalence_sweep.py(5번), remedy_power.py(15번) 등이
# results/calibration_results.csv 를 읽는다. calibration.py 를 그냥
# 돌리면 그 파일을 덮어쓰는 순간에 저쪽이 읽어 깨질 수 있고, 깨지면
# run 함수가 exit 1 로 재실행 전체를 중단시킨다. 그래서 맥의 /tmp 에
# 작업용 사본을 만들어 거기서 돌리고, 결과는 같은 디렉터리 안에서
# mv 로 바꿔 끼운다(같은 디렉터리 rename 은 원자적이다).
set -u
cd "$(dirname "$0")"

# 2026-10-21 2차: 1차 실행이 백업만 만들고 중간에 멈췄다. 어디서
# 멈췄는지 알 수 있게 전 과정을 results/redo_calibration_log.txt 에
# 남긴다. set -e 는 빼고 단계마다 종료코드를 직접 확인한다.
P="/tmp/fda_calib_$$"
LOGF="results/redo_calibration_log.txt"
exec > >(tee "$LOGF") 2>&1
step () { echo; echo "---- [$1] ----"; }
die  () {
  echo "!!! [$1] 에서 실패 (종료코드 $2). 여기서 멈춘다."
  echo "작업 사본 상태:"
  ls -l "${P:-/없음}/data/processed" 2>/dev/null | head -8 || echo "  data/processed 없음"
  ls -l "${P:-/없음}/results" 2>/dev/null || echo "  results 비어 있음"
  exit 1
}

step "환경"
echo "python: $(command -v python || echo '없음')"
python -c "import sys; print('버전', sys.version.split()[0])" || die 환경 $?
python -c "import sklearn, xgboost, rdkit, numpy, pandas; print('sklearn', sklearn.__version__, '/ xgboost', xgboost.__version__, '/ numpy', numpy.__version__)" || die 패키지 $?

B="_backup/calib_cast_20261021"
step "작업 사본 준비"
mkdir -p "$P/src" "$P/data" "$P/results" "$B" || die 디렉터리 $?
[ -f "$B/calibration_results.csv" ] || cp results/calibration_results.csv "$B/"
[ -f "$B/calibration_summary.txt" ] || cp results/calibration_summary.txt "$B/" 2>/dev/null || true

cp src/*.py "$P/src/" || die src복사 $?
cp -R data/processed "$P/data/" || die data복사 $?
echo "작업 사본: $P  (src $(ls "$P/src" | wc -l)개, processed $(ls "$P/data/processed" | wc -l)개)"

step "calibration.py 실행"
python "$P/src/calibration.py" || die calibration.py $?
ls -l "$P/results/" || true

step "전후 비교"
python - "$B/calibration_results.csv" "$P/results/calibration_results.csv" <<'PY' || die 비교 $?
import sys, pandas as pd, numpy as np
a = pd.read_csv(sys.argv[1]); b = pd.read_csv(sys.argv[2])
print("\n" + "=" * 72)
print("calibration_results.csv   캐스팅 수정 전 -> 후")
print("=" * 72)
if list(a.columns) != list(b.columns) or len(a) != len(b):
    print("모양이 다르다", a.shape, b.shape); raise SystemExit(1)
# pandas 3 은 문자열 열을 StringDtype 으로 읽는데 np.issubdtype 이
# 그것을 dtype 으로 해석하지 못해 TypeError 가 난다 (2026-10-21).
isnum = pd.api.types.is_numeric_dtype
key = [c for c in a.columns if not isnum(a[c])]
nd = []
for c in a.columns:
    if not isnum(a[c]): continue
    for i in range(len(a)):
        x, y = a[c][i], b[c][i]
        if pd.isna(x) and pd.isna(y): continue
        if x != y:
            nd.append((i, c, float(x), float(y), abs(float(y) - float(x)),
                       ("%.4f" % x) != ("%.4f" % y)))
print("바뀐 칸 %d개   그 중 소수 4자리에서 바뀐 것 %d개"
      % (len(nd), sum(1 for r in nd if r[5])))
for i, c, x, y, d, d4 in sorted(nd, key=lambda r: -r[4])[:40]:
    k = " / ".join(str(a[q][i]) for q in key)[:46]
    print("  %-48s %-14s %.6f -> %.6f  차 %.2e%s"
          % (k, c, x, y, d, "   <-- 4자리에서 바뀜" if d4 else ""))
if not nd:
    print("  바뀐 칸 없음 — 보고서 수치는 그대로다.")

# 보고서가 산문으로 숫자를 적지 않도록 전후 기록을 CSV 로 남긴다
# (10-20 수정의 precision_fix_20261020.csv 와 같은 방식).
rows = []
for i, c, x, y, d, d4 in sorted(nd, key=lambda r: -r[4]):
    rows.append({"row": i,
                 "key": " / ".join(str(a[q][i]) for q in key),
                 "column": c, "before": x, "after": y,
                 "abs_diff": d, "changes_4dp": bool(d4)})
out = pd.DataFrame(rows, columns=["row", "key", "column", "before", "after",
                                  "abs_diff", "changes_4dp"])
out.to_csv("results/precision_fix_20261021.csv", index=False)
print("\n기록: results/precision_fix_20261021.csv  (%d행)" % len(out))
PY

step "바꿔 끼우기"
cp "$P/results/calibration_results.csv" results/.calib_new.csv || die 복사 $?
mv results/.calib_new.csv results/calibration_results.csv
if [ -f "$P/results/calibration_summary.txt" ]; then
  cp "$P/results/calibration_summary.txt" results/.calib_new.txt
  mv results/.calib_new.txt results/calibration_summary.txt
fi
rm -rf "$P"
echo
echo "바꿔 끼웠다. 이전 판은 $B 에 있다."
