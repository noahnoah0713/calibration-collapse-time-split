"""얼마나 자주 다시 학습해야 하는가 — 재학습 주기 용량반응.

왜 필요한가
----------
§34 는 "해마다 재학습하고 해마다 재보정하면 보정이 유의하게 개선된다" 를
보였다. 거기서 바로 두 가지가 따라 나온다.

  (1) 심사자: 그 개선이 특정 일정(해마다)의 우연이 아닌가.
  (2) 실무자: 그러면 얼마나 자주 해야 하는가.

둘 다 같은 실험으로 답한다. 재학습 주기를 1·2·3·5년과 '하지 않음'(고정
모델)으로 늘어놓고 보정오차가 주기에 따라 어떻게 움직이는지 본다. 단일
대조가 아니라 용량반응이 나오면 §34 의 결론이 훨씬 세진다.

왜 검정력이 떨어지지 않는가
-------------------------
주기를 늘려도 평가집합은 그대로다. 주기가 2년이면 모델이 최대 1년 묵은
상태로 예측하는 것이고, 평가 대상 분자는 검증군 전부(%d / %d개)로 같다.
그래서 모든 주기가 같은 분자에서 짝지어 비교된다. (처음에는 주기를
늘리면 구간이 못 쓸 만큼 넓어질 것이라고 판단했는데 틀렸다 — 평가집합이
줄어드는 설계로 잘못 생각한 것이다. 2026-10-20)

설계
----
주기 k 에 대해 재학습 시점은 CUTOFF, CUTOFF+k, CUTOFF+2k, ... 이고, 승인연도
Y 인 분자는 Y 이하의 가장 최근 재학습 시점에 만들어진 모델이 예측한다.
그 모델의 학습창은 그 시점 이전에 승인된 분자 전부다. 따라서 모델이
묵은 정도는 0~k-1년이고, 그 평균을 함께 적는다 — 주기가 아니라 묵은
정도가 실제 설명변수다.

팔마다 재보정 유무를 둔다. 재보정은 §34 의 P2 와 같다 — 재학습 시점마다
학습창의 최근 25%% 로 Platt 을 다시 적합한다.

판정에 표준오차를 쓴다
-------------------
주기 k 대 주기 1 의 ECE 차를 같은 분자에서 부트스트랩해 구간을 낸다.
점추정이 단조라는 것만으로 추세를 주장하지 않는다 — 1년과 5년의 차이가
구간으로 0 을 제외해야 "주기가 길면 나빠진다" 를 쓴다.

사용법
------
    python src/retrain_interval.py
    python src/retrain_interval.py --intervals 1,2,3,5
    python src/retrain_interval.py --judge-only

출력
----
    results/retrain_interval.csv
    results/retrain_interval_pred.csv
    results/retrain_interval_summary.txt
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

OUT = os.path.join(RES, "retrain_interval.csv")
PRED = os.path.join(RES, "retrain_interval_pred.csv")
DELTA = os.path.join(RES, "retrain_interval_delta.csv")
PROG = os.path.join(RES, "retrain_interval_progress.txt")
CALIB_FRAC = 0.25
N_BOOT = 2000
MIN_FIT = 100


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(np.clip(p, S.EPS, 1 - S.EPS) / np.clip(1 - p, S.EPS, 1 - S.EPS))


def platt(p_cal, y_cal, p_te):
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


def fit_at(X, y, yr, T, recal):
    """승인연도 < T 인 분자로 학습한 예측함수. recal 이면 창 내부 Platt 까지."""
    tr = np.where(yr < T)[0]
    if len(tr) < MIN_FIT or len(np.unique(y[tr])) < 2:
        return None
    if not recal:
        m = S.make_model().fit(X[tr], y[tr])
        return lambda idx: m.predict_proba(X[idx])[:, 1]
    srt = tr[np.lexsort((tr, yr[tr]))]
    n_cal = int(len(srt) * CALIB_FRAC)
    fit_i, cal_i = srt[:-n_cal], srt[-n_cal:]
    if len(fit_i) < MIN_FIT or len(np.unique(y[cal_i])) < 2:
        m = S.make_model().fit(X[tr], y[tr])
        return lambda idx: m.predict_proba(X[idx])[:, 1]
    m = S.make_model().fit(X[fit_i], y[fit_i])
    p_cal = m.predict_proba(X[cal_i])[:, 1]
    return lambda idx: platt(p_cal, y[cal_i], m.predict_proba(X[idx])[:, 1])


def run_interval(X, y, yr, years, k, recal):
    """주기 k. 반환: (예측, 인덱스, 묵은 햇수, 재학습 횟수)."""
    refits = list(range(S.CUTOFF, max(years) + 1, k)) if k > 0 else [S.CUTOFF]
    cache, ps, idxs, age, used = {}, [], [], [], 0
    for T in years:
        r = max([t for t in refits if t <= T], default=S.CUTOFF)
        if r not in cache:
            cache[r] = fit_at(X, y, yr, r, recal)
            if cache[r] is not None:
                used += 1
        f = cache[r]
        te = np.where(yr == T)[0]
        if f is None or len(te) == 0:
            continue
        ps.append(f(te))
        idxs.append(te)
        age.append(np.full(len(te), T - r, dtype=float))
    if not ps:
        return np.array([]), np.array([], dtype=int), np.array([]), used
    return (np.concatenate(ps), np.concatenate(idxs), np.concatenate(age), used)


def paired_delta(y, p_new, p_old, seed=S.SEED):
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    v = []
    for _ in range(N_BOOT):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p_new[i]) - S.ece(y[i], p_old[i]))
    if not v:
        return float("nan"), float("nan"), float("nan")
    a = np.sort(np.array(v))
    return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def run(name, fn, ks):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    post = np.where(te_m)[0]
    years = sorted(set(int(v) for v in yr[post]))

    log("=" * 122)
    log(
        "%s   검증 연도 %d개 (%d–%d), 평가 분자 %d개 — 주기와 무관하게 같은 집합"
        % (name, len(years), years[0], years[-1], len(post))
    )
    log("=" * 122)
    log(
        "%-16s %-7s %7s %8s %8s %8s %9s %9s"
        % ("주기", "재보정", "재학습", "평균 묵은햇수", "ECE", "AUROC", "보정기울기", "평균예측")
    )
    log("-" * 122)

    rows, preds = [], {}
    for recal in (False, True):
        for k in ks:
            p, idx, age, used = run_interval(X, y, yr, years, k, recal)
            if len(idx) == 0:
                continue
            tag = ("%d년마다" % k) if k > 0 else "하지 않음(고정)"
            m = metrics(y[idx], p)
            log(
                "%-16s %-7s %7d %8.2f %8.4f %8.4f %9.4f %9.4f"
                % (
                    tag,
                    "예" if recal else "아니오",
                    used,
                    float(age.mean()),
                    m["ece"],
                    m["auroc"],
                    m["slope"],
                    m["mean_pred"],
                )
            )
            rows.append(
                dict(
                    dataset=name,
                    interval=k,
                    recal=recal,
                    n_refit=used,
                    mean_age=float(age.mean()),
                    max_age=float(age.max()),
                    **m
                )
            )
            preds[(k, recal)] = (idx, p)
            progress("%s k=%d recal=%s 완료" % (name, k, recal))
        log("-" * 122)
    # 예측 저장 — 짝지은 비교는 같은 분자에서 한다
    base_idx = preds[(ks[0], False)][0]
    d = {"dataset": name, "idx": base_idx, "year": yr[base_idx], "y": y[base_idx]}
    for (k, rc), (idx, p) in preds.items():
        assert np.array_equal(idx, base_idx), "주기 %d 의 평가 분자가 다르다" % k
        d["p_k%d_%s" % (k, "recal" if rc else "raw")] = p
    pd.DataFrame(d).to_csv(PRED, mode="a", header=not os.path.exists(PRED), index=False)
    log("")
    return rows


def judge(ks):
    if not os.path.exists(OUT):
        log("계산된 결과가 없다 — 중단한다.")
        return
    d = pd.read_csv(OUT)
    pr = pd.read_csv(PRED)
    drows = []
    log("=" * 122)
    log("판정 — 주기 1년을 기준으로 한 짝지은 ΔECE")
    log("=" * 122)
    log("[읽는 법] 점추정이 단조라는 것만으로 추세를 주장하지 않는다. 1년과 가장 긴")
    log("          주기의 차이가 구간으로 0 을 제외해야 '주기가 길면 나빠진다' 를 쓴다.")
    log("")
    for ds in d.dataset.unique():
        q = pr[pr.dataset == ds]
        g = d[d.dataset == ds]
        for rc in (True, False):
            sfx = "recal" if rc else "raw"
            b = "p_k%d_%s" % (ks[0], sfx)
            if b not in q.columns:
                continue
            log("  %s — 재보정 %s" % (ds, "있음" if rc else "없음"))
            log("    %-16s %9s %-22s %s" % ("주기", "ΔECE", "95% CI", "판정"))
            res = {}
            for k in ks[1:]:
                c = "p_k%d_%s" % (k, sfx)
                if c not in q.columns:
                    continue
                m, lo, hi = paired_delta(q.y.values, q[c].values, q[b].values)
                res[k] = (m, lo, hi)
                # 보고서가 산문을 파싱하지 않도록 CSV 로 남긴다.
                _a = g[(g.interval == k) & (g.recal == rc)]
                drows.append(
                    dict(
                        dataset=ds,
                        interval=k,
                        recal=rc,
                        base_interval=ks[0],
                        delta=m,
                        delta_lo=lo,
                        delta_hi=hi,
                        ece=(float(_a.ece.iloc[0]) if len(_a) else float("nan")),
                        mean_age=(float(_a.mean_age.iloc[0]) if len(_a) else float("nan")),
                        n_refit=(int(_a.n_refit.iloc[0]) if len(_a) else 0),
                        verdict=(
                            "1년보다 유의하게 나쁘다"
                            if lo > 0
                            else "1년보다 유의하게 좋다" if hi < 0 else "1년과 구별되지 않는다"
                        ),
                    )
                )
                log(
                    "    %-16s %+9.4f [%+.4f, %+.4f] %s"
                    % (
                        ("%d년마다" % k) if k > 0 else "하지 않음(고정)",
                        m,
                        lo,
                        hi,
                        (
                            "**1년보다 유의하게 나쁘다**"
                            if lo > 0
                            else ("1년보다 유의하게 좋다" if hi < 0 else "1년과 구별되지 않는다")
                        ),
                    )
                )
            _long = [k for k in ks[1:] if k > 0]
            if _long and max(_long) in res:
                m, lo, hi = res[max(_long)]
                log(
                    "    → 가장 긴 주기(%d년)와 1년의 차이: %s"
                    % (
                        max(_long),
                        (
                            "유의하다 — 주기가 길면 나빠진다"
                            if lo > 0
                            else (
                                "유의하지 않다 — **%d년까지는 1년과 구별되지 않는다**" % max(_long)
                            )
                        ),
                    )
                )
            if 0 in res:
                m, lo, hi = res[0]
                log(
                    "    → 고정 모델과 1년의 차이: %s (%+.4f [%+.4f, %+.4f])"
                    % ("유의하다" if lo > 0 else "유의하지 않다", m, lo, hi)
                )
            log("")
    # ---- §34 재현 확인: k=1 은 §34 의 P1·P2, k=0 은 P0 과 같아야 한다
    _p = os.path.join(RES, "prequential_retraining.csv")
    if os.path.exists(_p):
        pq = pd.read_csv(_p)
        log("=" * 122)
        log("§34 재현 확인 (k=1 은 P1·P2, k=0(고정)은 P0 과 같은 절차다)")
        log("=" * 122)
        worst = 0.0
        for ds in d.dataset.unique():
            for k, rc, arm in ((ks[0], False, "P1."), (ks[0], True, "P2."), (0, False, "P0.")):
                a9 = d[(d.dataset == ds) & (d.interval == k) & (d.recal == rc)]
                b9 = pq[(pq.dataset == ds) & pq.arm.str.startswith(arm)]
                if not (len(a9) and len(b9)):
                    continue
                dif = abs(float(a9.ece.iloc[0]) - float(b9.ece.iloc[0]))
                worst = max(worst, dif)
                log(
                    "  %s %-4s k=%d recal=%-5s  §34 %.6f  여기 %.6f  차 %.2e"
                    % (ds, arm, k, str(rc), float(b9.ece.iloc[0]), float(a9.ece.iloc[0]), dif)
                )
        log(
            "  최대차 %.2e — %s"
            % (
                worst,
                (
                    "같다"
                    if worst < 1e-9
                    else (
                        "보고 자리수 안에서 같다 (부동소수 수준)"
                        if worst < 1e-4
                        else "**다르다. 두 스크립트가 어긋났다**"
                    )
                ),
            )
        )
        log("")
    log("=" * 122)
    log("요약")
    log("=" * 122)
    for ds in d.dataset.unique():
        g = d[(d.dataset == ds)]
        gr = g[g.recal]
        if not len(gr):
            continue
        _o = gr.sort_values("mean_age")
        log(
            "  %s (재보정 있음) — 묵은햇수 평균 %s 에서 ECE %s"
            % (
                ds,
                " → ".join("%.2f" % v for v in _o.mean_age),
                " → ".join("%.4f" % v for v in _o.ece),
            )
        )
    log("")
    if drows:
        pd.DataFrame(drows).to_csv(DELTA, index=False)
        log("저장: %s" % DELTA)
    log("저장: %s" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--intervals", default="1,2,3,5,0", help="재학습 주기(년). 0 은 '하지 않음(고정 모델)'"
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()
    ks = [int(v) for v in a.intervals.split(",")]
    if a.fresh:
        for p in (OUT, PRED):
            if os.path.exists(p):
                os.remove(p)
    log("얼마나 자주 다시 학습해야 하는가 — 재학습 주기 용량반응")
    log(
        "주기 %s / 평가집합은 주기와 무관하게 같다. 시드 %d"
        % (", ".join(("%d년" % k) if k else "고정" for k in ks), S.SEED)
    )
    log("")
    if not a.judge_only:
        if os.path.exists(OUT):
            log("기존 결과를 지우고 다시 돈다 (주기 조합이 바뀌면 섞이면 안 된다).")
            for p in (OUT, PRED):
                if os.path.exists(p):
                    os.remove(p)
            log("")
        rows = []
        for name, fn in S.DATASETS:
            rows += run(name, fn, ks)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    judge(ks)
    log.save(Path(RES) / "retrain_interval_summary.txt")


if __name__ == "__main__":
    main()
