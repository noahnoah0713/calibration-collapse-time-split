# -*- coding: utf-8 -*-
"""헤드라인 배율을 모델×표현 32칸 전체에서 — 가장 보수적인 대조군으로.

왜 필요한가 (범위의 구멍)
------------------------
§9.5 의 32칸 격자(모델 4종 × 표현 4종)는 대조군을 R1·R2 까지만 돌렸다
(`results/grid_calibration.csv` 에 r3 열이 없다, 2026-10-19 확인). 그런데
논문이 방어선으로 쓰는 수치는 R3 다 — 학습군과 검증군의 양성률을 양측
모두 맞춘 대조군에서 나온 1.37배·1.21배(§8, §26).

즉 지금 상태는 이렇다.
  R1·R2 배율: 32칸 전부에서 1 을 넘는다 (이미 확인됨)
  R3 배율:    XGBoost + ECFP4+desc 한 칸만 확인됨

심사자가 묻는다 — "가장 보수적인 통제에서의 1.37 배가 그 모델 하나의
성질은 아닌가." 이 스크립트가 그 한 칸을 32칸으로 넓힌다.

설계
----
grid_calibration.py 와 같은 모델·같은 표현·같은 시드를 쓰고(정의는
`_shared.grid_models()` / `grid_reps()` 한 곳에서 온다), 시간분할 ECE 가
기존 결과와 칸마다 일치하는지 먼저 검사한다. 그 다음 R3 대조군만 새로 돈다.
보정 기여율(share)도 R3 기준으로 함께 낸다 — §7.1 의 "DICTrank 기여율이
양측통제 후 내려간다" 가 모델 특정적인지 보기 위해서다.

사용법
------
    python src/grid_r3.py                  # 대조군 15회 (§9.5 와 동일)
    python src/grid_r3.py --reps 50        # 더 안정적으로
    python src/grid_r3.py --reps 3         # 배관 점검

출력
----
    results/grid_r3.csv
    results/grid_r3_summary.txt
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()
PROG = os.path.join(RES, "grid_r3_progress.txt")

N_REP_DEFAULT = 15  # grid_calibration.py 와 동일
TOL = 1e-6  # 칸별 시간분할 ECE 재현 허용 오차


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))


def run(name, fn, n_rep, ref):
    df = S.load(os.path.join(PROC, fn))
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"].tolist()]
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    n_test = len(te)
    rate_te, rate_tr = float(y[te].mean()), float(y[tr].mean())

    log("=" * 112)
    log(
        "%s   학습 %d / 검증 %d   양성률 학습 %.4f / 검증 %.4f   대조군 %d회"
        % (name, len(tr), n_test, rate_tr, rate_te, n_rep)
    )
    log("=" * 112)
    log(
        "%-12s %-14s %9s %9s %9s %9s %9s"
        % ("표현", "모델", "시간 ECE", "R3 대조", "R3 배율", "R3 기여율", "기존 일치")
    )
    log("-" * 112)

    REPS = S.grid_reps()
    MODELS = S.grid_models()
    rows = []
    for rname, rfn in REPS.items():
        X = rfn(mols)
        for mname, mk in MODELS.items():
            m = mk().fit(X[tr], y[tr])
            p_t = m.predict_proba(X[te])[:, 1]
            ece_T = S.ece(y[te], p_t)
            dt = S.decompose(y[te], p_t)

            ok = "—"
            if ref is not None:
                q = ref[(ref.dataset == name) & (ref.rep == rname) & (ref.model == mname)]
                if len(q):
                    d0 = abs(float(q.ece_temporal.iloc[0]) - ece_T)
                    ok = "일치" if d0 < TOL else "**불일치 %.1e**" % d0
                    assert d0 < TOL, "§9.5 재현 실패 %s/%s/%s: %.6f 대 %.6f" % (
                        name,
                        rname,
                        mname,
                        float(q.ece_temporal.iloc[0]),
                        ece_T,
                    )

            acc = {k: [] for k in ("reliability", "resolution", "uncertainty")}
            E = []
            for i in range(n_rep):
                r = np.random.default_rng(S.SEED + i)
                idx = r.permutation(len(y))
                ti, ri = idx[:n_test], idx[n_test:]
                ti = S.subsample_to_rate(y, ti, rate_te, r)
                ri = S.subsample_to_rate(y, ri, rate_tr, r)
                if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                    continue
                if len(ti) < 30 or len(ri) < 100:
                    continue
                mm = mk().fit(X[ri], y[ri])
                pp = mm.predict_proba(X[ti])[:, 1]
                E.append(S.ece(y[ti], pp))
                dd = S.decompose(y[ti], pp)
                for k in acc:
                    acc[k].append(dd[k])
            if not E:
                log("%-12s %-14s 유효 반복 0회 — 건너뜀" % (rname, mname))
                continue
            ctrl = float(np.mean(E))
            d_rel = dt["reliability"] - float(np.mean(acc["reliability"]))
            d_res = dt["resolution"] - float(np.mean(acc["resolution"]))
            share = d_rel / (d_rel + d_res) * 100.0 if (d_rel + d_res) != 0 else float("nan")
            rows.append(
                {
                    "dataset": name,
                    "rep": rname,
                    "model": mname,
                    "auroc_temporal": float(roc_auc_score(y[te], p_t)),
                    "ece_temporal": ece_T,
                    "r3_ece_random": ctrl,
                    "r3_ece_sd": float(np.std(E, ddof=1)) if len(E) > 1 else float("nan"),
                    "r3_n_rep_valid": len(E),
                    "r3_ece_ratio": ece_T / ctrl if ctrl > 0 else float("nan"),
                    "r3_d_reliability": d_rel,
                    "r3_d_resolution": d_res,
                    "r3_share2": share,
                }
            )
            log(
                "%-12s %-14s %9.4f %9.4f %9.2f %8.1f%% %9s"
                % (rname, mname, ece_T, ctrl, ece_T / ctrl, share, ok)
            )
            progress("%s %s/%s 배율 %.2f" % (name, rname, mname, ece_T / ctrl))
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP_DEFAULT)
    a = ap.parse_args()
    open(PROG, "w", encoding="utf-8").write(
        "grid_r3 진행 기록 — %s\n" % time.strftime("%Y-%m-%d %H:%M")
    )
    log("헤드라인 배율을 모델×표현 32칸 전체에서 — R3(양측 유병률 통제)")
    log("§9.5 는 R1·R2 까지만 돌렸다. 논문이 방어선으로 쓰는 R3 를 격자 전체로 넓힌다.")
    log("대조군 %d회, 시드 %d" % (a.reps, S.SEED))
    log("")
    ref = None
    _p = os.path.join(RES, "grid_calibration.csv")
    if os.path.exists(_p):
        ref = pd.read_csv(_p)
        log("기존 격자 %d칸과 시간분할 ECE 를 칸마다 대조한다 (허용 %.0e)." % (len(ref), TOL))
        log("")

    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.reps, ref)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "grid_r3.csv"), index=False, encoding="utf-8")

    log("=" * 112)
    log("판정")
    log("=" * 112)
    log("[읽는 법] 32칸 전부에서 R3 배율이 1 을 넘으면 헤드라인은 모델·표현의")
    log("          선택과 무관하다. 1 이하인 칸이 있으면 그 칸을 본문에 적어야 한다.")
    log("")
    bad = out[~(out.r3_ece_ratio > 1.0)]
    log(
        "  칸 %d개 / R3 배율 %.2f–%.2f (중앙값 %.2f)"
        % (len(out), out.r3_ece_ratio.min(), out.r3_ece_ratio.max(), out.r3_ece_ratio.median())
    )
    if len(bad):
        log("  **1 이하인 칸 %d개 — 본문에 명시할 것**" % len(bad))
        for _, r in bad.iterrows():
            log("      %-10s %-12s %-14s 배율 %.2f" % (r.dataset, r.rep, r.model, r.r3_ece_ratio))
    else:
        log("  **32칸 전부 1 을 넘는다 — 가장 보수적인 통제에서도 모델·표현에")
        log("    의존하지 않는다.**")
    log("")
    for d, g in out.groupby("dataset"):
        log(
            "  %s 배율 %.2f–%.2f / 보정 기여율 %.1f%%–%.1f%% (기여율이 50%% 미만인 칸 %d개)"
            % (
                d,
                g.r3_ece_ratio.min(),
                g.r3_ece_ratio.max(),
                g.r3_share2.min(),
                g.r3_share2.max(),
                int((g.r3_share2 < 50).sum()),
            )
        )
    log("")
    log("  모델별 배율 중앙값")
    for m, g in out.groupby("model"):
        log("    %-14s %.2f" % (m, g.r3_ece_ratio.median()))
    log("  표현별 배율 중앙값")
    for m, g in out.groupby("rep"):
        log("    %-14s %.2f" % (m, g.r3_ece_ratio.median()))
    log("")
    log("저장: %s" % os.path.join(RES, "grid_r3.csv"))
    log.save(Path(RES) / "grid_r3_summary.txt")


if __name__ == "__main__":
    main()
