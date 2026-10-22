#!/usr/bin/env bash
# 2026-10-22. 재개 캐시가 건너뛴 네 스크립트를 --fresh 로 다시 돌린다.
#
# 왜 — 10-20 정밀도 수정 뒤 두 번의 전체 재실행을 돌렸는데도 보고서가
# 읽는 네 파일(horizon_ratio·prequential_retraining·shift_matched_v2·
# tiebreak_averaged)이 수정 전 값이었다. 재개 기능이 "이미 끝났다" 고
# 보고 통째로 건너뛰었기 때문이다. verify9 에 5.96 절(결과가 코드보다
# 오래됐는가)을 넣어 이제는 잡히고, 그래서 지금 verify9 는 4건 실패로
# 나온다. 이 스크립트가 그 4건을 없앤다.
#
# 실측 소요: tiebreak_averaged 4분, shift_matched_v2 75분,
# prequential_retraining 74분. horizon_ratio 는 기록이 없어 모른다.
# 전부 합쳐 3~5시간으로 본다.
set -u
cd "$(dirname "$0")"
LOGF="results/redo_stale_log.txt"
exec > >(tee "$LOGF") 2>&1
die () { echo "!!! [$1] 실패 (종료코드 $2). 멈춘다."; exit 1; }

B=_backup/stale_cache_20261022
mkdir -p "$B"
for f in horizon_ratio prequential_retraining shift_matched_v2 tiebreak_averaged; do
  for ext in csv; do
    [ -f "$B/$f.$ext" ] || cp "results/$f.$ext" "$B/" 2>/dev/null || true
  done
done

for s in tiebreak_averaged shift_matched_v2 prequential_retraining horizon_ratio; do
  echo
  echo "================ $s --fresh  ($(date '+%H:%M:%S')) ================"
  python "src/$s.py" --fresh || die "$s" $?
  echo "-- $s 끝 ($(date '+%H:%M:%S')) --"
  python - "$s" <<'PY' || true
import sys, os
import pandas as pd
import numpy as np
n = sys.argv[1]
a_p = os.path.join("_backup/stale_cache_20261022", n + ".csv")
b_p = os.path.join("results", n + ".csv")
if not os.path.exists(a_p):
    print("   (이전 사본이 없어 비교 못 함)"); raise SystemExit
a = pd.read_csv(a_p); b = pd.read_csv(b_p)
if list(a.columns) != list(b.columns) or len(a) != len(b):
    print("   모양이 달라졌다", a.shape, "->", b.shape); raise SystemExit
isnum = pd.api.types.is_numeric_dtype
nd = []
for c in a.columns:
    if not isnum(a[c]):
        continue
    for i in range(len(a)):
        x, y = a[c][i], b[c][i]
        if pd.isna(x) and pd.isna(y):
            continue
        if pd.isna(x) or pd.isna(y) or x != y:
            nd.append((c, float(x), float(y), abs(float(y) - float(x)),
                       ("%.4f" % x) != ("%.4f" % y)))
n4 = sum(1 for r in nd if r[4])
print("   바뀐 칸 %d개   소수 4자리에서 바뀐 칸 %d개" % (len(nd), n4))
for c, x, y, d, d4 in sorted(nd, key=lambda r: -r[3])[:10]:
    print("     %-18s %.6f -> %.6f  차 %.2e%s"
          % (c, x, y, d, "   <-- 4자리" if d4 else ""))
PY
done

echo
echo "================ 문서 재생성과 검증 ================"
bash redo_docs.sh
