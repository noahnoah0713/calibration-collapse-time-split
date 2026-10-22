# -*- coding: utf-8 -*-
"""예측 지평이 멀어질수록 보정이 더 무너지는가 — 구간별 배율.

왜 필요한가 (§13.4 의 미완성)
---------------------------
§13.4 는 검증군을 승인연도 구간으로 쪼개 ECE 를 찍는다. DILIrank 는
2010-2013 에서 0.3136, 2018 이후에서 0.6158 로 두 배 가까이 나빠진다.
그런데 그 절은 스스로 이렇게 적고 있다 — "구간별 대조군을 정의할 수 없어
배율은 계산하지 않았고 절대 ECE 추세만 보고한다."

그 한 줄이 이 논문에서 가장 큰 미완성이다. 절대 ECE 는 구간마다 n 과
양성률이 다르기 때문에 그대로 비교할 수 없다. 2018 이후 구간은 n=57,
양성률 0.1404 로 2010-2013(n=50, 0.5200)과 과제 자체가 다르다. 양성률이
낮으면 ECE 는 자동으로 커진다(§19 의 법칙). 즉 추세가 지평 때문인지
양성률 때문인지 지금은 구분이 안 된다.

심사자가 묻는다 — "시간이 갈수록 더 나빠지는가, 아니면 그냥 양성률이
내려간 것인가." 이 스크립트가 구간마다 대조군을 만들어 답한다.

설계
----
구간마다 그 구간과 n·양성률이 같은 무작위 분할 대조군을 만든다.
  R1  같은 n, 양성률 맞추지 않음
  R2  같은 n + 같은 양성률 (검증군 쪽만)
  R3  같은 n + 같은 양성률 + 학습군 양성률도 맞춤
그러면 구간별 배율이 나오고, 구간 간 비교가 가능해진다. 배율에는 검증군
부트스트랩으로 구간도 붙인다(§26 과 같은 방식, 분자만).

지평과 양성률을 분리하는 핵심은 R2·R3 다. R2·R3 배율이 구간을 따라
올라가면 "지평이 멀어질수록 더 나빠진다" 가 양성률과 별개로 성립한다.
올라가지 않으면 §13.4 의 추세는 양성률 하락의 결과이고, 그렇게 적어야 한다.

사용법
------
    python src/horizon_ratio.py                 # 대조군 200회
    python src/horizon_ratio.py --reps 20       # 짧게
    python src/horizon_ratio.py --edges 2014,2018

출력
----
    results/horizon_ratio.csv
    results/horizon_ratio_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

N_BOOT = 4000
MIN_N = 30  # 이보다 작은 구간은 배율을 내지 않는다
OUT = os.path.join(RES, "horizon_ratio.csv")
PROG = os.path.join(RES, "horizon_ratio_progress.txt")

# 2026-10-20: 이 스크립트는 구간 3개 x 대조 3종 x 반복 N 회 x 데이터셋 2개라
# 가장 무겁다. 첫 실행이 중간에 끊겼을 때 맨 끝에서 한 번만 저장하는 설계
# 때문에 끝난 구간까지 전부 날아갔다. 구간 하나가 끝날 때마다 바로 덧붙이고,
# 다시 돌리면 이미 있는 (데이터셋, 구간) 은 건너뛴다.


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))


def append_row(row):
    """구간 하나를 즉시 CSV 에 덧붙인다. 중단되어도 여기까지는 남는다."""
    df = pd.DataFrame([row])
    if os.path.exists(OUT):
        df.to_csv(OUT, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(OUT, index=False, encoding="utf-8")


def already_done():
    """이미 끝난 (데이터셋, 구간) 집합."""
    if not os.path.exists(OUT):
        return set()
    try:
        d = pd.read_csv(OUT)
    except Exception:
        return set()
    if "dataset" not in d.columns or "window" not in d.columns:
        return set()
    return set(zip(d.dataset.astype(str), d.window.astype(str)))


def boot_ratio_num(y, p, ctrl, n_boot, seed):
    """분자(구간 검증군)만 재표집한 배율 구간. 예측은 고정한다."""
    if not (ctrl == ctrl) or ctrl <= 0:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p[i]) / ctrl)
    if not v:
        return float("nan"), float("nan")
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def run(name, fn, edges, reps, done):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rate_tr = float(y[tr].mean())

    m = S.make_model().fit(X[tr], y[tr])
    p_all = np.full(len(y), np.nan)
    p_all[te] = m.predict_proba(X[te])[:, 1]

    # 구간 나누기 — [CUTOFF, e1), [e1, e2), [e2, inf)
    bounds = [S.CUTOFF] + list(edges) + [int(yr.max()) + 1]
    wins = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = te[(yr[te] >= a) & (yr[te] < b)]
        wins.append(("%d-%d" % (a, b - 1), a, b, idx))

    log("=" * 122)
    log(
        "%s   학습 %d (양성률 %.4f) / 검증 %d   대조군 %d회 · 부트스트랩 %d회"
        % (name, len(tr), rate_tr, len(te), reps, N_BOOT)
    )
    log("=" * 122)
    log(
        "%-12s %6s %9s %9s %9s %9s %9s %9s %-18s"
        % ("구간", "n", "양성률", "AUROC", "ECE", "R1 배율", "R2 배율", "R3 배율", "R3 배율 95%CI")
    )
    log("-" * 122)

    rows = []
    for lab, a, b, idx in wins:
        if (name, lab) in done:
            log("%-12s 이미 계산돼 있다 — 건너뜀 (재개)" % lab)
            continue
        if len(idx) < MIN_N or len(np.unique(y[idx])) < 2:
            log("%-12s %6d  표본 부족 또는 한 클래스뿐 — 건너뜀" % (lab, len(idx)))
            continue
        n_w = len(idx)
        rate_w = float(y[idx].mean())
        ece_w = S.ece(y[idx], p_all[idx])
        auc_w = float(roc_auc_score(y[idx], p_all[idx]))

        ctrl = {}
        for tag, mt, mr in (("R1", False, False), ("R2", True, False), ("R3", True, True)):
            E = []
            for i in range(reps):
                r = np.random.default_rng(S.SEED + i)
                perm = r.permutation(len(y))
                ti, ri = perm[:n_w], perm[n_w:]
                if mt:
                    ti = S.subsample_to_rate(y, ti, rate_w, r)
                if mr:
                    ri = S.subsample_to_rate(y, ri, rate_tr, r)
                if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                    continue
                if len(ti) < 20 or len(ri) < 100:
                    continue
                mm = S.make_model().fit(X[ri], y[ri])
                E.append(S.ece(y[ti], mm.predict_proba(X[ti])[:, 1]))
            ctrl[tag] = (float(np.mean(E)) if E else float("nan"), len(E))
        lo3, hi3 = boot_ratio_num(y[idx], p_all[idx], ctrl["R3"][0], N_BOOT, S.SEED + 3)
        row = {
            "dataset": name,
            "window": lab,
            "year_from": a,
            "year_to": b - 1,
            "n": n_w,
            "pos_rate": rate_w,
            "gap_vs_train": rate_w - rate_tr,
            "auroc": auc_w,
            "ece": ece_w,
        }
        for tag in ("R1", "R2", "R3"):
            c, nv = ctrl[tag]
            row["%s_ece" % tag.lower()] = c
            row["%s_n_rep" % tag.lower()] = nv
            row["%s_ratio" % tag.lower()] = ece_w / c if c == c and c > 0 else float("nan")
        row["r3_ratio_lo"] = lo3
        row["r3_ratio_hi"] = hi3
        row["r3_excludes_one"] = bool(lo3 == lo3 and lo3 > 1.0)
        rows.append(row)
        append_row(row)
        progress("%s %s 완료 (n=%d, R3 배율 %.2f)" % (name, lab, n_w, row["r3_ratio"]))
        log(
            "%-12s %6d %9.4f %9.4f %9.4f %9.2f %9.2f %9.2f [%5.2f, %5.2f]"
            % (
                lab,
                n_w,
                rate_w,
                auc_w,
                ece_w,
                row["r1_ratio"],
                row["r2_ratio"],
                row["r3_ratio"],
                lo3,
                hi3,
            )
        )
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--edges", type=str, default="2014,2018")
    ap.add_argument(
        "--fresh", action="store_true", help="기존 결과를 무시하고 처음부터 다시 계산한다"
    )
    ap.add_argument(
        "--judge-only",
        action="store_true",
        help="이미 저장된 CSV 로 판정만 다시 찍는다 (모델 적합 없음)",
    )
    a = ap.parse_args()
    if a.fresh and os.path.exists(OUT):
        os.remove(OUT)
    edges = [int(x) for x in a.edges.split(",") if x.strip()]

    log("예측 지평이 멀어질수록 보정이 더 무너지는가 — 구간별 배율")
    log("§13.4 는 절대 ECE 추세만 보고했다. 구간마다 n·양성률을 맞춘 대조군을")
    log("만들어 지평 효과와 양성률 효과를 분리한다.")
    log("구간 경계 %s, 대조군 %d회, 부트스트랩 %d회, 시드 %d" % (edges, a.reps, N_BOOT, S.SEED))
    log("")

    done = already_done()
    if done:
        log("이미 끝난 구간 %d개를 건너뛴다 (재개). 처음부터 다시 하려면 --fresh." % len(done))
        log("")
    if a.judge_only:
        log("판정만 다시 계산한다 (모델 적합 없음).")
        log("")
    else:
        progress("실행 시작 (대조군 %d회, 이미 끝난 구간 %d개)" % (a.reps, len(done)))
        for name, fn in S.DATASETS:
            run(name, fn, edges, a.reps, done)
    # 판정은 증분 저장된 CSV 전체를 다시 읽어서 한다
    if not os.path.exists(OUT):
        log("계산된 구간이 없다 — 중단한다.")
        return
    out = pd.read_csv(OUT)

    log("=" * 122)
    log("판정")
    log("=" * 122)
    log("[읽는 법] 절대 ECE 가 구간을 따라 올라가는 것만으로는 '지평이 멀어져서'")
    log("          라고 말할 수 없다. 양성률이 함께 내려가면 §19 의 법칙만으로도")
    log("          ECE 는 올라간다. **양성률을 맞춘 R2·R3 배율이 올라가야** 지평")
    log("          효과를 따로 주장할 수 있다.")
    log("")
    for d, g in out.groupby("dataset", sort=False):
        g = g.sort_values("year_from")
        log("  %s" % d)
        log("    절대 ECE   %s" % " → ".join("%.4f" % v for v in g.ece))
        log("    양성률     %s" % " → ".join("%.4f" % v for v in g.pos_rate))
        log("    R1 배율    %s" % " → ".join("%.2f" % v for v in g.r1_ratio))
        log("    R2 배율    %s" % " → ".join("%.2f" % v for v in g.r2_ratio))
        log("    R3 배율    %s" % " → ".join("%.2f" % v for v in g.r3_ratio))
        # 2026-10-20 수정: 전에는 단조성만 보고 "지평 효과가 따로 있다" 고 찍었다.
        # 틀렸다. 변화량이 구간 폭보다 작으면 추세를 판정할 수 없다. DICTrank 의
        # R3 배율 변화는 +0.035 인데 구간 반폭이 0.3~0.6 이다 — 잡음 안이다.
        # 그래서 변화량을 양 끝 구간의 신뢰구간 반폭과 비교한다.
        _e_up = bool(g.ece.iloc[-1] > g.ece.iloc[0])
        _d3 = float(g.r3_ratio.iloc[-1] - g.r3_ratio.iloc[0])
        _hw = (
            0.5
            * float(
                (g.r3_ratio_hi.iloc[0] - g.r3_ratio_lo.iloc[0])
                + (g.r3_ratio_hi.iloc[-1] - g.r3_ratio_lo.iloc[-1])
            )
            / 2.0
        )
        _mono3 = bool(all(x < yv for x, yv in zip(g.r3_ratio, g.r3_ratio[1:])))
        log(
            "    절대 ECE 증가 %s / R3 배율 변화 %+.3f (단조 %s) / 양 끝 구간 반폭 평균 %.3f"
            % ("예" if _e_up else "아니오", _d3, "예" if _mono3 else "아니오", _hw)
        )
        if abs(_d3) < _hw:
            log(
                "    → **R3 배율의 변화(%+.3f)가 구간 반폭(%.3f)보다 작다. 이 표본으로" % (_d3, _hw)
            )
            log("      지평 추세를 판정할 수 없다.** 절대 ECE 상승은 양성률 하락과")
            log("      같이 가므로 §13.4 의 추세를 지평 효과로 쓰면 안 된다.")
        elif _d3 > 0:
            log("    → **양성률을 맞춘 뒤에도 배율이 구간 폭을 넘어 올라간다.**")
            log("      지평 효과를 따로 주장할 수 있다.")
        else:
            log("    → **양성률을 맞추면 배율이 오히려 내려간다.** §13.4 의 추세는")
            log("      양성률 하락의 결과다.")
        log("    R3 구간이 1 을 제외하는 구간 수: %d/%d" % (int(g.r3_excludes_one.sum()), len(g)))
        log("")
    log(
        "  **구간 n 이 작다**(최소 %d). 구간별 배율의 구간이 넓으므로 추세의 "
        "방향만 읽고 점추정값을 비교하지 말 것." % int(out.n.min())
    )
    log("")
    log("저장: %s" % os.path.join(RES, "horizon_ratio.csv"))
    log.save(Path(RES) / "horizon_ratio_summary.txt")


if __name__ == "__main__":
    main()
