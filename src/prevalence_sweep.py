# -*- coding: utf-8 -*-
"""학습 유병률 스윕 — 보정오차가 학습·검증 유병률 격차의 함수인가.

왜 필요한가
----------
§16 유병률 사다리는 세 점(R1·R2·R3)만 보여 준다. concept_shift.py 의
A→D→C→B 3분해는 ECE 초과분의 82.0%(DILIrank)·98.1%(DICTrank)가
**학습 유병률 몫**임을 가리켰다. 그러면 다음 질문은 하나다 —
그게 **함수 관계**인가, 아니면 세 점이 우연히 그렇게 놓였을 뿐인가.

여기서는 시간을 완전히 빼고(무작위 분할만 쓴다) 학습군 양성률만
여러 단계로 바꿔 ECE 곡선을 그린다. 곡선이 격차 0 에서 최소를 이루고
양쪽으로 단조 증가하면, 보정오차는 분포 이동의 **라벨 성분**이
만들어내는 체계적 함수다. 시간분할의 실제 격차가 곡선 위 어디에
떨어지는지를 표시하면 §8 의 값이 설명된다.

설계 (검증군을 고정하고 학습군의 양성률만 바꾼다)
--------------------------------------------------
  검증군 : 시간분할과 **같은 크기·같은 양성률** 로 무작위 표집
           (DILIrank n=156 양성률 0.3077 / DICTrank n=267 0.6142)
  학습군 : 남은 풀에서 **고정 크기**, 양성률만 단계별로 바꿔 표집
  단계   : 0.20~0.80 등간 7단계 + 검증군 양성률 + pre-2010 실제 양성률
           (실제 값 두 개를 눈금에 끼워 §8 과 바로 비교된다)

학습 크기는 데이터가 허용하는 최대로 자동 결정한다. 가장 빡빡한
단계(양성률 0.20 → 음성이 많이 필요하다)에서 풀이 감당할 수 있는
수의 95%다. 데이터셋마다 다르므로 로그에 적어 둔다.

읽는 법
  - 최소 ECE 가 **격차 0 근처**에 오는가 → 라벨 이동이 원인이라는 증거
  - |격차| 에 대한 기울기 → "유병률 격차 0.10 당 ECE 가 얼마" 라는
    인용 가능한 한 문장
  - 시간분할 실제 격차에서의 곡선 높이 대 격차 0 에서의 높이
    → §8 ECE 중 라벨 이동만으로 설명되는 부분

한계
  - 학습 크기를 고정했으므로 크기 효과는 들어 있지 않다 (의도한 것).
  - 양성률을 맞추려 비복원 재표집하므로 단계마다 화학공간 대표성이
    조금씩 달라진다.
  - 무작위 분할이므로 공변량 이동·개념 이동은 없다. 이 곡선은
    **라벨 이동 단독의 상한** 이 아니라 **단독 효과**를 보여 준다.

사용법
------
    python src/prevalence_sweep.py                 # 단계당 100회
    python src/prevalence_sweep.py --reps 10       # 짧게 점검

출력
----
    results/prevalence_sweep.csv        단계별 집계
    results/prevalence_sweep_reps.csv   반복별 원자료 (그림 오차막대용)
    results/prevalence_sweep_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

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
PROGRESS = os.path.join(RES, "prevalence_sweep_progress.txt")


def prog(msg):
    """단계가 끝날 때마다 즉시 파일에 덧붙인다 (실행 중 확인용)."""
    import datetime

    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (datetime.datetime.now().strftime("%H:%M:%S"), msg))


BASE_LEVELS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
SAFETY = 0.95  # 풀 소진을 막는 여유
N_BOOT = 2000


def metrics(y, p):
    d = S.decompose(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "ece": S.ece(y, p),
        "slope": S.calib_slope(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
    }


def feasible_train_n(n_pool_pos, n_pool_neg, levels):
    """모든 단계에서 표집 가능한 최대 학습 크기."""
    caps = []
    for L in levels:
        cap_by_pos = n_pool_pos / L if L > 0 else np.inf
        cap_by_neg = n_pool_neg / (1 - L) if L < 1 else np.inf
        caps.append(min(cap_by_pos, cap_by_neg))
    return int(np.floor(min(caps) * SAFETY))


def ols(x, y):
    """절편 포함 단순 최소제곱. (기울기, 절편) 반환."""
    A = np.c_[np.ones(len(x)), np.asarray(x, dtype=float)]
    b, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return float(b[1]), float(b[0])


def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_mask, te_mask = S.temporal_masks(df)

    n_test = int(te_mask.sum())
    rate_test = float(y[te_mask].mean())  # 시간분할 검증군 양성률
    rate_pre = float(y[tr_mask].mean())  # 시간분할 학습군 양성률
    gap_obs = rate_pre - rate_test  # §8 의 실제 격차

    # 검증군 구성은 고정이다 → 풀 구성도 결정적이다
    n_te_pos = int(round(n_test * rate_test))
    n_te_neg = n_test - n_te_pos
    all_pos = np.where(y == 1)[0]
    all_neg = np.where(y == 0)[0]
    pool_pos = len(all_pos) - n_te_pos
    pool_neg = len(all_neg) - n_te_neg

    levels = sorted(set([round(v, 4) for v in BASE_LEVELS + [rate_test, rate_pre]]))
    n_train = feasible_train_n(pool_pos, pool_neg, levels)

    log("=" * 104)
    log("%s   전체 %d개 (양성 %d / 음성 %d)" % (name, len(y), len(all_pos), len(all_neg)))
    log(
        "  검증군 고정: n=%d, 양성률 %.4f (양성 %d / 음성 %d) — 시간분할과 동일"
        % (n_test, rate_test, n_te_pos, n_te_neg)
    )
    log("  학습 풀: 양성 %d / 음성 %d  →  고정 학습 크기 %d" % (pool_pos, pool_neg, n_train))
    log("  단계 %d개: %s" % (len(levels), ", ".join("%.4f" % v for v in levels)))
    log("  §8 실제 격차: 학습 %.4f − 검증 %.4f = %+.4f" % (rate_pre, rate_test, gap_obs))
    prog(
        "%s 시작 — 단계 %d개 × 반복 %d회 = 적합 %d회 (학습 n %d)"
        % (name, len(levels), reps, len(levels) * reps, n_train)
    )
    log("=" * 104)
    log(
        "%-9s %7s %8s %7s %8s %8s %8s %8s %6s"
        % ("학습률", "격차", "학습n", "AUROC", "ECE", "ECE sd", "기울기", "신뢰도", "유효")
    )
    log("-" * 104)

    agg, per_rep = [], []
    for li, L in enumerate(levels):
        n_pos = int(round(n_train * L))
        n_neg = n_train - n_pos
        if n_pos > pool_pos or n_neg > pool_neg or n_pos < 2 or n_neg < 2:
            log("%-9.4f  표집 불가 (양성 %d / 음성 %d 필요) — 건너뜀" % (L, n_pos, n_neg))
            continue
        acc = {k: [] for k in ("auroc", "ece", "slope", "rel", "res", "train_rate")}
        for i in range(reps):
            r = np.random.default_rng(S.SEED + 20000 + li * 1000 + i)
            te = np.concatenate(
                [
                    r.choice(all_pos, n_te_pos, replace=False),
                    r.choice(all_neg, n_te_neg, replace=False),
                ]
            )
            te_set = set(te.tolist())
            p_pos = np.array([v for v in all_pos if v not in te_set])
            p_neg = np.array([v for v in all_neg if v not in te_set])
            tr = np.concatenate(
                [r.choice(p_pos, n_pos, replace=False), r.choice(p_neg, n_neg, replace=False)]
            )
            m = S.make_model().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            mm = metrics(y[te], p)
            for k, v in mm.items():
                acc[k].append(v)
            acc["train_rate"].append(float(y[tr].mean()))
            per_rep.append({"dataset": name, "level": L, "rep": i, "gap": L - rate_test, **mm})
        if not acc["ece"]:
            continue
        E = np.array(acc["ece"])
        log(
            "%-9.4f %+7.4f %8d %7.4f %8.4f %8.4f %8.4f %8.4f %6d"
            % (
                L,
                L - rate_test,
                n_train,
                float(np.mean(acc["auroc"])),
                float(E.mean()),
                float(E.std(ddof=1)) if len(E) > 1 else 0.0,
                float(np.mean(acc["slope"])),
                float(np.mean(acc["rel"])),
                len(E),
            )
        )
        prog(
            "  %s 단계 %d/%d  학습률 %.4f  ECE %.4f  (반복 %d회 완료)"
            % (name, li + 1, len(levels), L, float(E.mean()), len(E))
        )
        agg.append(
            {
                "dataset": name,
                "level": L,
                "gap": L - rate_test,
                "train_n": n_train,
                "n_test": n_test,
                "test_rate": rate_test,
                "train_rate_actual": float(np.mean(acc["train_rate"])),
                "auroc": float(np.mean(acc["auroc"])),
                "ece": float(E.mean()),
                "ece_sd": float(E.std(ddof=1)) if len(E) > 1 else 0.0,
                "ece_se": (float(E.std(ddof=1)) / np.sqrt(len(E)) if len(E) > 1 else 0.0),
                "slope": float(np.mean(acc["slope"])),
                "rel": float(np.mean(acc["rel"])),
                "res": float(np.mean(acc["res"])),
                "n_rep": len(E),
                "is_test_rate": abs(L - round(rate_test, 4)) < 1e-9,
                "is_pre2010_rate": abs(L - round(rate_pre, 4)) < 1e-9,
            }
        )
    log("")
    return (
        agg,
        per_rep,
        dict(rate_test=rate_test, rate_pre=rate_pre, gap_obs=gap_obs, n_train=n_train),
    )


def judge(name, A, R, meta):
    """곡선이 실제로 격차의 함수인지 세 가지로 읽는다."""
    a = pd.DataFrame(A)
    r = pd.DataFrame(R)
    log("  %s" % name)

    # (1) 최소가 어디인가
    i = int(a.ece.values.argmin())
    log(
        "    최소 ECE %.4f 는 학습률 %.4f (격차 %+.4f) 에서 — "
        "격차 0 에서 %.4f 떨어져 있다"
        % (a.ece.iloc[i], a.level.iloc[i], a.gap.iloc[i], abs(a.gap.iloc[i]))
    )

    # (2) |격차| 기울기 (반복별 원자료로 회귀, 반복 단위 부트스트랩 CI)
    g = np.abs(r.gap.values)
    e = r.ece.values
    b, a0 = ols(g, e)
    rng = np.random.default_rng(S.SEED)
    bs = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(g), len(g))
        if np.std(g[k]) == 0:
            continue
        bs.append(ols(g[k], e[k])[0])
    bs = np.sort(np.array(bs))
    lo, hi = (
        (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        if len(bs)
        else (np.nan, np.nan)
    )
    log(
        "    |격차| 기울기: 격차 0.10 당 ECE %+.4f "
        "[%+.4f, %+.4f]  (절편 %.4f, 반복 %d개)" % (b * 0.10, lo * 0.10, hi * 0.10, a0, len(g))
    )

    # (2b) 대칭 가정을 쓰지 않는 양쪽 기울기. |격차| 하나로 묶으면
    #      곡선이 한쪽으로 기울어 있어도 보이지 않는다.
    sides = {}
    for lab, sel in (
        ("학습률 < 검증률 (격차 음)", r.gap.values < 0),
        ("학습률 > 검증률 (격차 양)", r.gap.values > 0),
    ):
        gg, ee = r.gap.values[sel], r.ece.values[sel]
        if len(gg) < 10 or np.std(gg) == 0:
            log("    %s: 점이 부족해 기울기를 내지 않는다 (%d개)" % (lab, len(gg)))
            sides[lab] = float("nan")
            continue
        bb = ols(np.abs(gg), ee)[0]
        log("    %s: 격차 0.10 당 ECE %+.4f (반복 %d개)" % (lab, bb * 0.10, len(gg)))
        sides[lab] = bb * 0.10
    vals = [v for v in sides.values() if v == v]
    if len(vals) == 2:
        log("      → 양쪽 기울기 차 %+.4f. 차이가 크면 곡선이 비대칭이고," % (vals[1] - vals[0]))
        log("        |격차| 단일 기울기는 평균값으로만 읽어야 한다.")

    # (3) 실제 격차에서의 높이 대 격차 0 에서의 높이
    e_at = float(np.interp(meta["gap_obs"], a.gap.values, a.ece.values))
    e_zero = float(np.interp(0.0, a.gap.values, a.ece.values))
    log(
        "    곡선 보간: 격차 0 → ECE %.4f / §8 실제 격차 %+.4f → ECE %.4f "
        "(차이 %+.4f, 배율 %.2f)"
        % (e_zero, meta["gap_obs"], e_at, e_at - e_zero, e_at / e_zero if e_zero else float("nan"))
    )
    log("      [해석] 무작위 분할에서 학습 유병률만 §8 만큼 어긋나게 해도")
    log("             ECE 가 %.4f 까지 올라간다. §8 시간분할의 실제 ECE 와" % e_at)
    log("             비교하면 라벨 이동만으로 설명되는 몫을 읽을 수 있다")
    log("             (§8 값은 calibration_results.csv 의 temporal 행).")
    log("")
    sl = list(sides.values()) if len(sides) == 2 else [float("nan")] * 2
    return {
        "dataset": name,
        "min_level": float(a.level.iloc[i]),
        "min_gap": float(a.gap.iloc[i]),
        "min_ece": float(a.ece.iloc[i]),
        "slope_per_0.10": b * 0.10,
        "slope_lo": lo * 0.10,
        "slope_hi": hi * 0.10,
        "intercept": a0,
        "slope_neg_side_per_0.10": sl[0],
        "slope_pos_side_per_0.10": sl[1],
        "ece_at_gap0": e_zero,
        "ece_at_obs_gap": e_at,
        "gap_obs": meta["gap_obs"],
        "n_rep_total": int(len(g)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100, help="단계당 반복 수 (기본 100)")
    a = ap.parse_args()

    if os.path.exists(PROGRESS):
        os.remove(PROGRESS)
    prog("실행 시작 (반복 %d회)" % a.reps)
    log("학습 유병률 스윕 — 보정오차는 유병률 격차의 함수인가")
    log("단계당 반복 %d회, 시드 %d, 부트스트랩 %d회" % (a.reps, S.SEED, N_BOOT))
    log("시간은 쓰지 않는다 — 전부 무작위 분할이다.")
    log("")

    AGG, REP, JUD = [], [], []
    for name, fn in S.DATASETS:
        A, R, meta = run(name, fn, a.reps)
        AGG += A
        REP += R
        if A:
            JUD.append((name, A, R, meta))

    pd.DataFrame(AGG).to_csv(
        os.path.join(RES, "prevalence_sweep.csv"), index=False, encoding="utf-8"
    )
    pd.DataFrame(REP).to_csv(
        os.path.join(RES, "prevalence_sweep_reps.csv"), index=False, encoding="utf-8"
    )

    log("=" * 104)
    log("판정")
    log("=" * 104)
    rows = [judge(*j) for j in JUD]
    pd.DataFrame(rows).to_csv(
        os.path.join(RES, "prevalence_sweep_fit.csv"), index=False, encoding="utf-8"
    )

    log("저장: %s" % os.path.join(RES, "prevalence_sweep.csv"))
    log("      %s" % os.path.join(RES, "prevalence_sweep_reps.csv"))
    log("      %s" % os.path.join(RES, "prevalence_sweep_fit.csv"))
    log("      %s" % os.path.join(RES, "prevalence_sweep_summary.txt"))
    prog("완료 — 모든 파일 저장됨")
    log.save(Path(RES) / "prevalence_sweep_summary.txt")


if __name__ == "__main__":
    main()
