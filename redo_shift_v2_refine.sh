#!/usr/bin/env bash
# 2026-10-22. §33 이동량 맞춤의 2차 정밀 탐색. 약 75분.
#
# 왜 — redo_stale.sh 가 shift_matched_v2 를 --fresh 로 돌리면서 거친
# alpha 격자(0.0~1.0 여섯 단)만 다시 만들었다. 그 격자로는 이동 AUC
# 맞춤이 여섯 설정 전부 실패한다(차 0.063~0.081). 논문 §33 의 수치
# (차 0.0011~0.0134)는 맞춤 구간을 촘촘히 돌린 2차 탐색에서 나왔고,
# 그 호출은 손으로만 돌려 재현 경로에 없었다. alpha 목록은 이전 결과
# 파일에서 꺼냈다 — 여섯 설정 모두 0.29·0.31·0.33·0.46·0.48·0.50 이다.
# rerun_all.sh 에도 두 줄로 명시해 두었다.
set -u
cd "$(dirname "$0")"
LOGF="results/redo_shift_v2_refine_log.txt"
exec > >(tee "$LOGF") 2>&1
die () { echo "!!! [$1] 실패 (종료코드 $2). 멈춘다."; exit 1; }

echo "---- 2차 정밀 탐색 ($(date '+%H:%M:%S')) ----"
python src/shift_matched_v2.py --alphas 0.29,0.31,0.33,0.46,0.48,0.5 || die refine $?

echo
echo "---- 맞춤 결과 ----"
python - <<'PY' || true
import pandas as pd
m = pd.read_csv("results/shift_matched_v2_matched.csv")
cols = [c for c in ("dataset", "axis", "alpha_star", "shift_auc_gap",
                    "matched", "ece_diff", "diff_se_units", "verdict")
        if c in m.columns]
print(m[cols].to_string(index=False))
ok = int(m["matched"].astype(str).str.lower().isin(("true", "1")).sum())
print("\n맞춤 성공 %d / %d   이동 AUC 차 %.4f ~ %.4f"
      % (ok, len(m), m.shift_auc_gap.abs().min(), m.shift_auc_gap.abs().max()))
PY

echo
echo "---- 문서 재생성과 검증 ----"
bash redo_docs.sh
