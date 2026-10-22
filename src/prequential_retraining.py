# -*- coding: utf-8 -*-
"""해마다 다시 학습하면 보정이 회복되는가 — 배포 시뮬레이션.

왜 이것이 빠져 있었나
-------------------
이 논문은 처방을 여럿 시험하고 전부 또는 거의 전부 실패했다 — 사후
재보정(§10), 사전확률 추정(§10.2·§10.3), 최신성 가중(§20), 적용범위
필터(§21), 클래스 가중(§23), 공변량 이동 가중(§30). 그런데 **현장에서 가장
먼저 하는 일**이 빠져 있다. 새 약이 승인되고 라벨이 붙으면 그것을 학습에
넣고 모델을 다시 만든다. 심사자가 "그러면 해마다 다시 학습하면 되는 것
아닌가" 라고 물으면 지금 원고에는 답이 없다.

설계 — 전향적 순차 평가(prequential)
----------------------------------
검증 연도 T 마다

    학습 = 승인연도 < T 인 분자 전부 (해가 갈수록 커진다)
    평가 = 승인연도 = T 인 분자

로 예측하고, 모든 연도의 예측을 **모아서 한 번** 평가한다. 모아서 재는
이유는 연도별 분자가 10여 개뿐이라 연도별 ECE 가 쓸 수 없기 때문이다.
모아 놓은 검증집합은 §4~§10 의 검증군(승인 2010년 이후)과 **같은 집합**
이므로 고정 모델과 직접 짝지어 비교할 수 있다.

팔 넷
----
    P0  고정 모델 (2010년 이전만으로 한 번 학습) — 이 논문의 기존 설정
    P1  해마다 재학습
    P2  해마다 재학습 + 해마다 학습창 내부에서 Platt 재보정
        (학습창의 최근 25% 를 보정셋으로 — §10 의 B 와 같은 방식)
    C   연도를 무작위로 섞은 뒤 같은 절차 (교환가능 기준선, R회)

P1·P2 대 P0 는 **같은 분자**에 대한 비교이므로 짝지은 부트스트랩으로 ΔECE
구간을 낸다. 점추정 비교만으로 판정하지 않는다.

정직하게 적어야 하는 것
---------------------
P1·P2 는 **낙관적이다.** 승인 시점에 간독성·심장독성 라벨이 이미 확정돼
있다고 가정하기 때문이다. 실제로는 라벨이 쌓이는 데 시간이 걸린다(§12 의
라벨 성숙도). 그래서 이 실험이 "회복된다" 고 나오면 그것은 **상한**이고,
"회복되지 않는다" 고 나오면 현실에서는 더 나쁘다는 뜻이다.

사용법
------
    python src/prequential_retraining.py            # 대조군 10회
    python src/prequential_retraining.py --reps 20
    python src/prequential_retraining.py --fresh
    python src/prequential_retraining.py --judge-only

연도마다 모델을 적합하므로 적합 횟수는 (연도 수 × 2) + (연도 수 × R) 이다.
반복마다 즉시 저장하므로 중단되어도 이어서 돈다.

출력
----
    results/prequential_retraining.csv
    results/prequential_retraining_pred.csv
    results/prequential_retraining_summary.txt
    results/prequential_retraining_progress.txt
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

OUT = os.path.join(RES, "prequential_retraining.csv")
PRED = os.path.join(RES, "prequential_retraining_pred.csv")
DELTA = os.path.join(RES, "prequential_retraining_delta.csv")
PROG = os.path.join(RES, "prequential_retraining_progress.txt")
CALIB_FRAC = 0.25
N_BOOT = 2000
MIN_FIT = 100  # 학습창이 이보다 작으면 그 연도는 건너뛴다


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
    return np.log(p / (1 - p))


def platt(p_cal, y_cal, p_te):
    """§10 의 B 와 같은 구현."""
    if len(np.unique(y_cal)) < 2:
        return np.asarray(p_te, dtype=float)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    out = {
        "ece": S.ece(y, p),
        "mean_pred": float(p.mean()),
        "obs_rate": float(y.mean()),
        "n": int(len(y)),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    try:
        out["slope"] = S.calib_slope(y, p)
    except Exception:
        out["slope"] = float("nan")
    return out


def prequential(X, y, yr, years, recal):
    """연도별 확장창 예측. recal=True 면 창 내부 Platt 재보정까지 한다.

    반환: (예측, 해당 분자 인덱스, 연도별 학습창 크기)
    """
    ps, idxs, sizes = [], [], []
    for T in years:
        te = np.where(yr == T)[0]
        tr = np.where(yr < T)[0]
        if len(te) == 0 or len(tr) < MIN_FIT or len(np.unique(y[tr])) < 2:
            continue
        if not recal:
            m = S.make_model().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
        else:
            srt = tr[np.lexsort((tr, yr[tr]))]
            n_cal = int(len(srt) * CALIB_FRAC)
            fit_i, cal_i = srt[:-n_cal], srt[-n_cal:]
            if len(fit_i) < MIN_FIT or len(np.unique(y[cal_i])) < 2:
                m = S.make_model().fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
            else:
                m = S.make_model().fit(X[fit_i], y[fit_i])
                p = platt(m.predict_proba(X[cal_i])[:, 1], y[cal_i], m.predict_proba(X[te])[:, 1])
        ps.append(p)
        idxs.append(te)
        sizes.append((int(T), len(tr), len(te)))
    if not ps:
        return np.array([]), np.array([], dtype=int), sizes
    return np.concatenate(ps), np.concatenate(idxs), sizes


def pick_ctrl(g, arm):
    """팔의 재보정 유무와 **같은** 대조군을 고른다. 양성률 맞춤을 우선한다."""
    rc = arm.startswith("P2")
    order = ["C2r.", "C1r."] if rc else ["C2.", "C1."]
    for pre in order:
        q = g[g.arm.str.startswith(pre)]
        if len(q):
            return q
    return g.iloc[0:0]


def paired_delta(y, p_new, p_old, n_boot=N_BOOT, seed=S.SEED):
    """같은 분자에서 ΔECE = 새 − 기존. 음수면 개선이다."""
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p_new[i]) - S.ece(y[i], p_old[i]))
    if not v:
        return float("nan"), float("nan"), float("nan")
    a = np.sort(np.array(v))
    return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def append_row(row):
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)


def done_set():
    if not os.path.exists(OUT):
        return set()
    d = pd.read_csv(OUT)
    return set(zip(d.dataset, d.arm, d.rep.astype(int)))


def run(name, fn, reps, done):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    pre, post = np.where(tr_m)[0], np.where(te_m)[0]
    years = sorted(set(int(v) for v in yr[post]))

    log("=" * 118)
    log(
        "%s   2010년 이전 %d개 / 이후 %d개   검증 연도 %d개 (%d–%d)"
        % (name, len(pre), len(post), len(years), years[0], years[-1])
    )
    log("=" * 118)

    # ---- P1·P2 와 P0 — 결정적이므로 rep 0 하나
    if (name, "P1. 해마다 재학습", 0) not in done:
        p1, i1, sizes = prequential(X, y, yr, years, recal=False)
        p2, i2, _ = prequential(X, y, yr, years, recal=True)
        assert np.array_equal(i1, i2), "P1 과 P2 의 평가 분자가 다르다"
        m0 = S.make_model().fit(X[pre], y[pre])
        p0 = m0.predict_proba(X[i1])[:, 1]
        ys = y[i1]
        log("  연도별 학습창 — %s" % ", ".join("%d년 학습 %d 평가 %d" % s for s in sizes[:4]))
        log("    ... 마지막 %d년 학습 %d 평가 %d" % sizes[-1] if sizes else "")
        log(
            "  평가에 쓴 분자 %d개 (검증군 %d개 중 — 학습창이 %d개 미만인 "
            "연도는 건너뛴다)" % (len(ys), len(post), MIN_FIT)
        )
        for arm, p in (("P0. 고정 모델", p0), ("P1. 해마다 재학습", p1), ("P2. 재학습+재보정", p2)):
            r = metrics(ys, p)
            append_row(dict(dataset=name, arm=arm, rep=0, n_years=len(sizes), **r))
        pd.DataFrame(
            dict(
                dataset=name,
                idx=i1,
                year=yr[i1],
                y=ys,
                p_fixed=p0,
                p_retrain=p1,
                p_retrain_recal=p2,
            )
        ).to_csv(PRED, mode="a", header=not os.path.exists(PRED), index=False)
        progress("%s P0·P1·P2 완료" % name)

    # ---- C — 연도를 섞은 교환가능 기준선
    for i in range(reps):
        if (name, "C1. 연도 무작위", i) in done and (name, "C1r. 연도 무작위+재보정", i) in done:
            continue
        rng = np.random.default_rng(S.SEED + 11000 + i)
        yr_s = yr.copy()
        rng.shuffle(yr_s)
        yrs_s = sorted(set(int(v) for v in yr_s[yr_s >= S.CUTOFF]))
        # **재보정 유무를 맞춘다.** 처음에는 대조군을 recal=False 로만
        # 돌려서 P2(재보정 있음)를 재보정 없는 대조군과 비교했다. 그
        # 비교로는 "P2 가 교환가능 기준선보다 낫다" 가 재보정 덕인지
        # 시간축 덕인지 갈라지지 않는다 (2026-10-20 수정).
        rate = float(y[post].mean())
        for tag, rc in (("", False), ("r", True)):
            pc, ic, _ = prequential(X, y, yr_s, yrs_s, recal=rc)
            if len(ic) == 0:
                continue
            append_row(
                dict(
                    dataset=name,
                    arm="C1%s. 연도 무작위%s" % (tag, "+재보정" if rc else ""),
                    rep=i,
                    n_years=len(yrs_s),
                    **metrics(y[ic], pc)
                )
            )
            pos = {v: k for k, v in enumerate(ic)}
            keep = S.subsample_to_rate(y, ic, rate, rng)
            if len(keep) >= 40 and len(np.unique(y[keep])) > 1:
                sel = np.array([pos[v] for v in keep])
                append_row(
                    dict(
                        dataset=name,
                        arm="C2%s. 연도 무작위+양성률 맞춤%s" % (tag, "+재보정" if rc else ""),
                        rep=i,
                        n_years=len(yrs_s),
                        **metrics(y[keep], pc[sel])
                    )
                )
        if (i + 1) % 5 == 0 or i == 0:
            progress("%s 대조군 %d/%d" % (name, i + 1, reps))
    log("  대조군 %d회까지 완료" % reps)
    log("")


def judge():
    if not os.path.exists(OUT):
        log("계산된 결과가 없다 — 중단한다.")
        return
    d = pd.read_csv(OUT)
    pr = pd.read_csv(PRED) if os.path.exists(PRED) else None
    drows = []

    for ds in d.dataset.unique():
        g = d[d.dataset == ds]
        log("=" * 118)
        log("%s" % ds)
        log("=" * 118)
        log(
            "%-20s %6s %8s %10s %8s %8s %9s %9s"
            % ("팔", "n", "ECE", "ECE 표준편차", "AUROC", "기울기", "평균예측", "실제양성률")
        )
        log("-" * 118)
        for arm in (
            "P0. 고정 모델",
            "P1. 해마다 재학습",
            "P2. 재학습+재보정",
            "C1. 연도 무작위",
            "C2. 연도 무작위+양성률 맞춤",
            "C1r. 연도 무작위+재보정",
            "C2r. 연도 무작위+양성률 맞춤+재보정",
        ):
            a = g[g.arm == arm]
            if not len(a):
                continue
            sd = float(a.ece.std(ddof=1)) if len(a) > 1 else float("nan")
            log(
                "%-20s %6d %8.4f %10s %8.4f %8.4f %9.4f %9.4f"
                % (
                    arm,
                    int(a.n.mean()),
                    float(a.ece.mean()),
                    ("%.4f (%d회)" % (sd, len(a))) if len(a) > 1 else "—",
                    float(a.auroc.mean()),
                    float(a.slope.mean()),
                    float(a.mean_pred.mean()),
                    float(a.obs_rate.mean()),
                )
            )
        log("")

        if pr is None or not (pr.dataset == ds).any():
            continue
        q = pr[pr.dataset == ds]
        for arm, col in (
            ("P1. 해마다 재학습", "p_retrain"),
            ("P2. 재학습+재보정", "p_retrain_recal"),
        ):
            m, lo, hi = paired_delta(q.y.values, q[col].values, q.p_fixed.values)
            # 보고서가 산문을 파싱하지 않도록 CSV 로 남긴다 (2026-10-20).
            _c = pick_ctrl(g, arm)
            drows.append(
                dict(
                    dataset=ds,
                    arm=arm,
                    delta=m,
                    delta_lo=lo,
                    delta_hi=hi,
                    verdict=(
                        "개선이 유의하다"
                        if hi < 0
                        else "오히려 유의하게 나빠진다" if lo > 0 else "개선을 검출하지 못했다"
                    ),
                    own_ece=float(g[g.arm == arm].ece.iloc[0]),
                    ctrl_arm=(str(_c.arm.iloc[0]) if len(_c) else ""),
                    ctrl_ece=(float(_c.ece.mean()) if len(_c) else float("nan")),
                    ctrl_sd=(float(_c.ece.std(ddof=1)) if len(_c) > 1 else float("nan")),
                    ctrl_reps=(len(_c) if len(_c) else 0),
                )
            )
            ctrl = pick_ctrl(g, arm)
            cm = float(ctrl.ece.mean()) if len(ctrl) else float("nan")
            own = float(g[g.arm == arm].ece.iloc[0])
            v = (
                "개선이 유의하다"
                if hi < 0
                else "오히려 유의하게 나빠진다" if lo > 0 else "개선을 검출하지 못했다"
            )
            log(
                "  %s 대 P0 — 짝지은 ΔECE %+.4f [%+.4f, %+.4f] → %s"
                % (arm.split(". ")[1], m, lo, hi, v)
            )
            if cm == cm:
                log(
                    "      이 팔의 ECE %.4f 대 교환가능 기준선 %.4f (%s, 배율 %.2f)"
                    % (
                        own,
                        cm,
                        str(ctrl.arm.iloc[0]).split(". ")[0],
                        own / cm if cm else float("nan"),
                    )
                )
        log("")

    # ---- 판정
    log("=" * 118)
    log("판정")
    log("=" * 118)
    log("[읽는 법] P1·P2 는 승인 시점에 라벨이 이미 확정돼 있다고 가정하므로")
    log("          **낙관적이다**(§12 라벨 성숙도). 그래서 '회복되지 않는다' 가")
    log("          나오면 현실에서는 더 나쁘고, '회복된다' 가 나오면 그것은 상한이다.")
    log("")
    for ds in d.dataset.unique():
        g = d[d.dataset == ds]
        pr2 = pr[pr.dataset == ds] if pr is not None else None
        if pr2 is None or not len(pr2):
            continue
        best, bl, bh, bname = None, None, None, None
        for arm, col in (("P1", "p_retrain"), ("P2", "p_retrain_recal")):
            m, lo, hi = paired_delta(pr2.y.values, pr2[col].values, pr2.p_fixed.values)
            if best is None or m < best:
                best, bl, bh, bname = m, lo, hi, arm
        ctrl = pick_ctrl(g, "P2. 재학습+재보정" if bname == "P2" else "P1. 해마다 재학습")
        cm = float(ctrl.ece.mean()) if len(ctrl) else float("nan")
        csd = float(ctrl.ece.std(ddof=1)) if len(ctrl) > 1 else float("nan")
        own = float(
            g[g.arm == ("P1. 해마다 재학습" if bname == "P1" else "P2. 재학습+재보정")].ece.iloc[0]
        )
        log(
            "  %s — 가장 좋은 팔은 %s, 고정 모델 대비 ΔECE %+.4f [%+.4f, %+.4f]"
            % (ds, bname, best, bl, bh)
        )
        if bh < 0:
            log("      **해마다 다시 학습하면 보정이 유의하게 개선된다.**")
        elif bl > 0:
            log("      **해마다 다시 학습하면 오히려 유의하게 나빠진다.**")
        else:
            log("      **개선을 검출하지 못했다 — 구간이 0 을 포함한다.**")
        if cm == cm:
            _above = (own - cm) / csd if csd == csd and csd > 0 else float("nan")
            _d = own - cm
            log(
                "      교환가능 기준선(%s %.4f)보다 %.4f %s"
                % (str(ctrl.arm.iloc[0]).split(". ")[0], cm, abs(_d), "높다" if _d > 0 else "낮다")
                + ("" if _above != _above else " — 대조군 표준편차의 %.1f배다" % abs(_above))
            )
        log("")
    if drows:
        pd.DataFrame(drows).to_csv(DELTA, index=False)
        log("저장: %s" % DELTA)
    log("저장: %s" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()
    if a.fresh:
        for p in (OUT, PRED):
            if os.path.exists(p):
                os.remove(p)
    log("해마다 다시 학습하면 보정이 회복되는가 — 배포 시뮬레이션")
    log(
        "연도별 확장창으로 순차 예측하고 모아서 한 번 평가한다. 대조군 %d회, 시드 %d"
        % (a.reps, S.SEED)
    )
    log("")
    done = done_set()
    if done and not a.judge_only:
        log("이미 끝난 조합 %d개를 건너뛴다 (재개). 처음부터는 --fresh." % len(done))
        log("")
    if not a.judge_only:
        progress("실행 시작 (대조군 %d회, 이미 끝난 %d개)" % (a.reps, len(done)))
        for name, fn in S.DATASETS:
            run(name, fn, a.reps, done)
    judge()
    log.save(Path(RES) / "prequential_retraining_summary.txt")


if __name__ == "__main__":
    main()
