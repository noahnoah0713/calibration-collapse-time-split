# -*- coding: utf-8 -*-
r"""이동량을 맞춘 비시간 분할 — 한계 18번을 닫는다.

왜 필요한가 (한계가 스스로 적어놓은 숙제)
---------------------------------------
§13.5 는 시간이 아닌 축(분자량·logP·지문 군집)으로 분할을 만들어 "이동이
크면 보정이 더 무너지는가" 를 물었다. 그런데 그 축들이 모델 입력에 들어 있어
이동 AUC 가 0.98–1.00 으로 포화했다. 시간분할의 이동 AUC 는 0.6835 /
0.7917 이므로 비교 대상의 이동량이 전혀 맞지 않는다. 그래서 §13.5 는
"이동을 더 크게 줘도 붕괴가 더 커지지 않는다" 까지만 말하고, 정작 묻고 싶은
"같은 이동량에서 시간분할만 더 심한가" 는 답하지 못했다. 한계 18번이
그것이고, 해법도 거기 적혀 있다 — 이동량을 맞춘 대조군을 다시 설계하라.

설계 — 축을 숨기는 게 아니라 이동량을 조절한다
--------------------------------------------
모델 입력에 없는 축을 찾는 방식은 막다른 길이다. 지문은 분자 크기·극성과
상관되므로 어떤 물리화학적 축을 써도 입력에서 완전히 숨길 수 없다.

그래서 반대로 간다 — 이동량을 연속적으로 조절해 시간분할과 같게 맞춘다.

  alpha = 0  순수 무작위 분할 (이동 AUC 약 0.5)
  alpha = 1  축 극단 분할 (이동 AUC 약 1.0, = §13.5)
  그 사이    각 분자를 확률 alpha 로 축 순서에 따라, 1-alpha 로 무작위로 배정

alpha 를 훑으면 이동 AUC 가 0.5 에서 1.0 까지 연속으로 올라간다. 그중
시간분할의 이동 AUC 와 가장 가까운 alpha\* 를 고르고, 그 분할의 ECE 를
시간분할 ECE 와 비교한다.

교란을 막는 두 가지
------------------
  (1) 검증군 크기와 양성률을 시간분할과 같게 맞춘다. §19 가 보인 대로
      양성률 격차만으로도 ECE 는 움직인다. 맞추지 않으면 비교가 성립하지 않는다.
  (2) 학습군 양성률도 맞춘다. 즉 R3 와 같은 양측 통제 위에서 비교한다.
      그러면 시간분할과 다른 것은 분할 축 하나뿐이다.

이동 AUC 는 §13.1 과 같은 방식으로 잰다 — 같은 특성, 같은 모델, 분할 라벨에
대한 교차검증 out-of-fold AUC.

사용법
------
    python src/shift_matched_v2.py                 # 축 3종 x alpha 6단 x 10회
    python src/shift_matched_v2.py --reps 20       # 더 안정적으로
    python src/shift_matched_v2.py --fresh         # 처음부터 다시

중단되어도 (데이터셋, 축, alpha) 단위로 즉시 저장하므로 같은 명령을 다시
넣으면 남은 것만 이어서 돈다.

출력
----
    results/shift_matched_v2.csv
    results/shift_matched_v2_matched.csv
    results/shift_matched_v2_summary.txt
    results/shift_matched_v2_progress.txt
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

OUT = os.path.join(RES, "shift_matched_v2.csv")
OUT_M = os.path.join(RES, "shift_matched_v2_matched.csv")
PROG = os.path.join(RES, "shift_matched_v2_progress.txt")

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_FOLD_SHIFT = 3  # 이동 AUC 교차검증 (비용 때문에 3겹)
N_BOOT = 2000
TOL = 5e-4


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))


def append_row(row, path):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def done_set(path, keys):
    """이미 끝난 설정의 키 집합.

    2026-10-20: alpha 를 "%.3f" 로 저장하는데 pandas 가 그 열을 수치로 읽어
    astype(str) 하면 "0.400" 이 "0.4" 가 되어 재개가 작동하지 않았다.
    수치로 읽히는 열은 다시 같은 형식으로 포맷해 맞춘다.
    """
    if not os.path.exists(path):
        return set()
    try:
        d = pd.read_csv(path)
    except Exception:
        return set()
    if any(k not in d.columns for k in keys):
        return set()
    cols = []
    for k in keys:
        c = d[k]
        if c.dtype.kind in "fi":
            cols.append(c.astype(float).map(lambda v: "%.3f" % v))
        else:
            cols.append(c.astype(str))
    return set(zip(*cols))


def shift_auc(X, lab, seed=S.SEED):
    """§13.1 과 같은 방식 — 분할 라벨에 대한 OOF AUC. 라벨 y 는 쓰지 않는다."""
    lab = np.asarray(lab, dtype=int)
    if len(np.unique(lab)) < 2:
        return float("nan")
    p = np.zeros(len(lab), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLD_SHIFT, shuffle=True, random_state=seed)
    for tri, tei in skf.split(X, lab):
        m = S.make_model().fit(X[tri], lab[tri])
        p[tei] = m.predict_proba(X[tei])[:, 1]
    return float(roc_auc_score(lab, p))


def axes(X, df):
    """분할 축 3종. 값이 큰 쪽이 '검증군' 이 된다."""
    mw = df["smiles"].map(len).values.astype(float)  # 자리표시자, 아래에서 교체
    desc = X[:, -6:]  # featurize 가 뒤에 붙이는 기술자 6종
    r = np.random.default_rng(S.SEED)
    # 무작위 방향: 지문 공간의 임의 선형결합. 어떤 물리화학적 해석도 없다.
    v = r.normal(size=X.shape[1])
    v = v / np.linalg.norm(v)
    return {
        "A1. 분자량": desc[:, 0],
        "A2. logP": desc[:, 1],
        "A3. 무작위 방향": X.astype(float) @ v,
    }


def blended_split(score, n_test, alpha, r):
    """alpha 로 축 순서와 무작위 배정을 섞는다.

    각 분자에 점수 = alpha * (축 순위) + (1-alpha) * (무작위 순위) 를 주고
    상위 n_test 개를 검증군으로 뽑는다. alpha=0 이면 순수 무작위,
    alpha=1 이면 축 극단이다.
    """
    n = len(score)
    rank_axis = np.argsort(np.argsort(score)) / max(n - 1, 1)
    rank_rand = r.permutation(n) / max(n - 1, 1)
    blend = alpha * rank_axis + (1.0 - alpha) * rank_rand
    order = np.argsort(-blend)
    te = order[:n_test]
    tr = order[n_test:]
    return tr, te


def run(name, fn, reps, done):
    # ALPHAS 는 main 에서 --alphas 로 바뀔 수 있다
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr0, te0 = np.where(tr_m)[0], np.where(te_m)[0]
    n_test = len(te0)
    rate_te, rate_tr = float(y[te0].mean()), float(y[tr0].mean())

    # 기준점 — 시간분할
    m0 = S.make_model().fit(X[tr0], y[tr0])
    p0 = m0.predict_proba(X[te0])[:, 1]
    ece_T = S.ece(y[te0], p0)
    auc_T = shift_auc(X, te_m.astype(int))

    log("=" * 124)
    log(
        "%s   학습 %d / 검증 %d   시간분할 ECE %.4f   이동 AUC %.4f (%d겹)"
        % (name, len(tr0), n_test, ece_T, auc_T, N_FOLD_SHIFT)
    )
    log("  목표: 이동 AUC 를 %.4f 에 맞춘 비시간 분할을 만들어 ECE 를 비교한다." % auc_T)
    log("=" * 124)

    AX = axes(X, df)
    log(
        "%-16s %7s %10s %10s %10s %8s %8s"
        % ("축", "alpha", "이동 AUC", "ECE", "AUROC", "검증 n", "양성률")
    )
    log("-" * 124)

    for aname, score in AX.items():
        for al in ALPHAS:
            if (name, aname, "%.3f" % al) in done:
                log("%-16s %7.1f  이미 계산돼 있다 — 건너뜀" % (aname, al))
                continue
            E, A, NT, RT, AU = [], [], [], [], []
            for i in range(reps):
                r = np.random.default_rng(S.SEED + i)
                tr_i, te_i = blended_split(score, n_test, al, r)
                # 양측 유병률·크기 맞춤 (R3 와 같은 통제)
                te_i = S.subsample_to_rate(y, te_i, rate_te, r)
                tr_i = S.subsample_to_rate(y, tr_i, rate_tr, r)
                if len(np.unique(y[te_i])) < 2 or len(np.unique(y[tr_i])) < 2:
                    continue
                if len(te_i) < 30 or len(tr_i) < 100:
                    continue
                lab = np.zeros(len(y), dtype=int)
                lab[te_i] = 1
                sub = np.concatenate([tr_i, te_i])
                AU.append(shift_auc(X[sub], lab[sub], seed=S.SEED + i))
                mm = S.make_model().fit(X[tr_i], y[tr_i])
                pp = mm.predict_proba(X[te_i])[:, 1]
                E.append(S.ece(y[te_i], pp))
                A.append(float(roc_auc_score(y[te_i], pp)))
                NT.append(len(te_i))
                RT.append(float(y[te_i].mean()))
            if not E:
                log("%-16s %7.1f  유효 반복 0회 — 건너뜀" % (aname, al))
                continue
            row = {
                "dataset": name,
                "axis": aname,
                "alpha": "%.3f" % al,
                "alpha_num": al,
                "n_rep_valid": len(E),
                "shift_auc": float(np.mean(AU)),
                "shift_auc_sd": float(np.std(AU, ddof=1)) if len(AU) > 1 else float("nan"),
                "ece": float(np.mean(E)),
                "ece_sd": float(np.std(E, ddof=1)) if len(E) > 1 else float("nan"),
                "auroc": float(np.mean(A)),
                "n_test": float(np.mean(NT)),
                "pos_rate": float(np.mean(RT)),
                "ece_temporal": ece_T,
                "shift_auc_temporal": auc_T,
            }
            append_row(row, OUT)
            progress(
                "%s %s alpha=%.3f 이동AUC %.4f ECE %.4f"
                % (name, aname, al, row["shift_auc"], row["ece"])
            )
            log(
                "%-16s %7.1f %10.4f %10.4f %10.4f %8.1f %8.4f"
                % (
                    aname,
                    al,
                    row["shift_auc"],
                    row["ece"],
                    row["auroc"],
                    row["n_test"],
                    row["pos_rate"],
                )
            )
    log("")


def judge(reps):
    if not os.path.exists(OUT):
        log("계산된 설정이 없다 — 중단한다.")
        return
    d = pd.read_csv(OUT)
    rows = []
    log("=" * 124)
    log("이동량을 맞춘 지점에서의 비교")
    log("=" * 124)
    log(
        "%-10s %-16s %7s %10s %10s %10s %10s %-10s"
        % ("데이터셋", "축", "alpha*", "이동 AUC", "시간 AUC", "맞춤 ECE", "시간 ECE", "판정")
    )
    log("-" * 124)
    for (ds, ax), g in d.groupby(["dataset", "axis"]):
        tgt = float(g.shift_auc_temporal.iloc[0])
        ece_T = float(g.ece_temporal.iloc[0])
        k = (g.shift_auc - tgt).abs().idxmin()
        b = g.loc[k]
        # 맞춤이 실제로 됐는지 — 이동 AUC 차가 0.05 를 넘으면 맞춤 실패다
        matched = abs(float(b.shift_auc) - tgt) <= 0.05
        # 2026-10-20: 전에는 점추정값 대소만 보고 판정했다. 반복간 표준오차를
        # 넣지 않으면 0.01 차이도 "더 나쁘다" 로 찍힌다. 차이가 2 SE 를
        # 넘지 않으면 무승부로 적는다.
        _sd = float(b.ece_sd) if b.ece_sd == b.ece_sd else float("nan")
        _se = (
            _sd / np.sqrt(float(b.n_rep_valid))
            if _sd == _sd and float(b.n_rep_valid) > 0
            else float("nan")
        )
        _dif = ece_T - float(b.ece)
        _z = _dif / _se if _se == _se and _se > 0 else float("nan")
        if abs(_z) <= 2.0:
            verdict = "무승부 (차 %+.4f, %.1f SE)" % (_dif, _z)
        elif _dif > 0:
            verdict = "시간분할이 더 나쁘다 (%.1f SE)" % _z
        else:
            verdict = "맞춤 분할이 더 나쁘다 (%.1f SE)" % abs(_z)
        if not matched:
            verdict = "맞춤 실패 (이동 AUC 차 %.3f)" % abs(float(b.shift_auc) - tgt)
        rows.append(
            {
                "dataset": ds,
                "axis": ax,
                "alpha_star": float(b.alpha_num),
                "shift_auc_matched": float(b.shift_auc),
                "shift_auc_temporal": tgt,
                "shift_auc_gap": float(b.shift_auc) - tgt,
                "matched": bool(matched),
                "ece_matched": float(b.ece),
                "ece_temporal": ece_T,
                "ece_diff": ece_T - float(b.ece),
                "ece_ratio_T_over_matched": (
                    ece_T / float(b.ece) if float(b.ece) > 0 else float("nan")
                ),
                "n_rep_valid": int(b.n_rep_valid),
                "ece_matched_sd": _sd,
                "ece_matched_se": _se,
                "diff_se_units": _z,
                "verdict": verdict,
            }
        )
        log(
            "%-10s %-16s %7.1f %10.4f %10.4f %10.4f %10.4f %-10s"
            % (ds, ax, float(b.alpha_num), float(b.shift_auc), tgt, float(b.ece), ece_T, verdict)
        )
    M = pd.DataFrame(rows)
    M.to_csv(OUT_M, index=False, encoding="utf-8")
    log("")
    log("[읽는 법] alpha* 는 이동 AUC 가 시간분할에 가장 가까운 지점이다.")
    log("          그 지점의 ECE 가 시간분할 ECE 보다 **작으면**, 같은 이동량에서")
    log("          시간분할이 더 심하다는 뜻이다 — §13.5 가 못 했던 비교다.")
    log("")
    ok = M[M.matched]
    log("  맞춤에 성공한 설정 %d/%d (이동 AUC 차 0.05 이내)" % (len(ok), len(M)))
    if len(ok):
        _w = int((ok.diff_se_units > 2).sum())
        _tie = (
            int(ok.diff_se_units.abs() <= 2).__int__()
            if False
            else int((ok.diff_se_units.abs() <= 2).sum())
        )
        _l = int((ok.diff_se_units < -2).sum())
        log(
            "  그중 시간분할이 2 SE 를 넘어 더 나쁜 설정: **%d/%d** "
            "(무승부 %d, 반대 %d)" % (_w, len(ok), _tie, _l)
        )
        log(
            "  시간/맞춤 ECE 배율 %.2f–%.2f (중앙값 %.2f)"
            % (
                ok.ece_ratio_T_over_matched.min(),
                ok.ece_ratio_T_over_matched.max(),
                ok.ece_ratio_T_over_matched.median(),
            )
        )
        if _w + _tie == len(ok) and _w > 0:
            log("  → **같은 이동량에서도 시간분할이 더 심하다. 한계 18번이 닫힌다.**")
            log("    즉 보정 붕괴는 '이동이 크다' 로만 설명되지 않고 **시간 축에")
            log("    특정적인 성분**이 있다.")
        elif _w == 0 and _l == 0:
            log("  → **이동량을 맞추면 차이가 사라진다.** 그렇다면 보정 붕괴는")
            log("    이동량의 함수이고 시간 축이 특별하지 않다는 뜻이다. 본문의")
            log("    서술을 그 방향으로 고쳐야 한다.")
        else:
            log("  → **축에 따라 갈린다. 본문에 축별로 적어야 한다.**")
    else:
        log("  → **어느 축에서도 이동량을 맞추지 못했다.** alpha 격자를 더 촘촘히")
        log("    하거나 축을 바꿔야 한다. 현재 결과로는 한계 18번을 닫을 수 없다.")
    log("")
    for _, r in M.iterrows():
        log(
            "  %-10s %-16s alpha* %.1f, 이동 AUC %.4f (목표 %.4f, 차 %+.4f), "
            "ECE %.4f 대 %.4f"
            % (
                r.dataset,
                r.axis,
                r.alpha_star,
                r.shift_auc_matched,
                r.shift_auc_temporal,
                r.shift_auc_gap,
                r.ece_matched,
                r.ece_temporal,
            )
        )
    log("")
    log("저장: %s" % OUT)
    log("      %s" % OUT_M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument(
        "--alphas",
        type=str,
        default="",
        help="alpha 격자를 직접 준다 (예: 0.3,0.4,0.5,0.6,0.7). "
        "맞춤에 실패했을 때 그 구간만 촘촘히 다시 돌리면 된다 "
        "— 이미 끝난 설정은 건너뛴다.",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument(
        "--judge-only",
        action="store_true",
        help="저장된 CSV 로 판정만 다시 찍는다 (모델 적합 없음)",
    )
    a = ap.parse_args()
    global ALPHAS
    if a.alphas.strip():
        ALPHAS = [float(x) for x in a.alphas.split(",") if x.strip()]
    if a.fresh:
        for p in (OUT, OUT_M):
            if os.path.exists(p):
                os.remove(p)
    log("이동량을 맞춘 비시간 분할 — 한계 18번")
    log("alpha 로 축 순서와 무작위 배정을 섞어 이동 AUC 를 시간분할에 맞춘다.")
    log(
        "alpha 격자 %s, 축 3종, 반복 %d회, 이동 AUC %d겹, 시드 %d"
        % (ALPHAS, a.reps, N_FOLD_SHIFT, S.SEED)
    )
    log("검증군 크기·양성률과 학습군 양성률을 시간분할에 맞춘다(R3 와 같은 통제).")
    log("")
    done = done_set(OUT, ["dataset", "axis", "alpha"])
    if done:
        log("이미 끝난 설정 %d개를 건너뛴다 (재개). 처음부터는 --fresh." % len(done))
        log("")
    if a.judge_only:
        log("판정만 다시 계산한다 (모델 적합 없음).")
        log("")
    else:
        progress("실행 시작 (반복 %d회, 이미 끝난 설정 %d개)" % (a.reps, len(done)))
        for name, fn in S.DATASETS:
            run(name, fn, a.reps, done)
    judge(a.reps)
    log.save(Path(RES) / "shift_matched_v2_summary.txt")


if __name__ == "__main__":
    main()
