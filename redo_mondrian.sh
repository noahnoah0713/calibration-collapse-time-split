#!/usr/bin/env bash
# 2026-10-22. mondrian_conformal 의 중복 행을 없애고 FDR 보정을 다시 돌린다.
#
# 왜 — 재개 기능이 결과를 append 로 덧붙이는데 같은 조합을 두 번 돌려
# mondrian_conformal.csv 가 48행 중 24행 중복이 됐다. 그 파일을 읽는
# fdr_all_tests.py 가 새로 편입되는 검정을 57개가 아니라 73개로 세어
# BH 보정을 돌렸고, 그 결과 유의 건수가 F4 에서 16 대신 20, F0 에서
# 23 대신 27 로 나왔다. 논문의 핵심 결론(기존 유의 8건 중 1건만
# 유의성을 잃는다)은 중복을 지워도 그대로였지만, 검정 개수가 틀린
# 다중비교 보정을 그대로 둘 수는 없다.
#
# verify9 에 중복 행 점검(5.94)을 넣어 두었으니 다시 쌓이면 잡힌다.
# mondrian 은 재개 없이 돌아도 몇 분이면 끝난다.
set -u
cd "$(dirname "$0")"
LOGF="results/redo_mondrian_log.txt"
exec > >(tee "$LOGF") 2>&1
die () { echo "!!! [$1] 실패 (종료코드 $2). 멈춘다."; exit 1; }

B=_backup/dup_rows_20261022
mkdir -p "$B"
for f in mondrian_conformal fdr_all_tests; do
  [ -f "$B/$f.csv" ] || cp "results/$f.csv" "$B/" 2>/dev/null || true
done

echo "---- mondrian_conformal.py --fresh ($(date '+%H:%M:%S')) ----"
python src/mondrian_conformal.py --fresh || die mondrian_conformal $?

echo
echo "---- fdr_all_tests.py ----"
python src/fdr_all_tests.py || die fdr_all_tests $?

echo
echo "---- 중복 확인과 전후 비교 ----"
python - <<'PY' || true
import pandas as pd
for n in ("mondrian_conformal", "fdr_all_tests"):
    a = pd.read_csv("_backup/dup_rows_20261022/%s.csv" % n)
    b = pd.read_csv("results/%s.csv" % n)
    print("%-22s 전 %d행(중복 %d) → 후 %d행(중복 %d)"
          % (n, len(a), int(a.duplicated().sum()),
             len(b), int(b.duplicated().sum())))
f = pd.read_csv("results/fdr_all_tests.csv")
new = f[f.newly_included]
print("\n새로 편입된 검정 %d개   F4 유의 %d건   F0 유의 %d건"
      % (len(new), int((new.q_F4 <= 0.05).sum()), int((f.q_F0 <= 0.05).sum())))
old = f[~f.newly_included]
print("기존 유의 %d건 중 F0 에서 유의성을 잃는 것 %d건"
      % (int((old.sig_existing == True).sum()),
         int(((old.sig_existing == True) & (old.q_F0 > 0.05)).sum())))
PY

echo
echo "---- 문서 재생성과 검증 ----"
bash redo_docs.sh
