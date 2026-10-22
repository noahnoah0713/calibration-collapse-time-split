# -*- coding: utf-8 -*-
"""라벨 조건부(Mondrian) conformal — 한계 5번.

왜 필요한가
----------
한계 5번은 이렇게 적혀 있다.

    "Split conformal 은 주변 커버리지만 보장한다. §11 의 클래스별
     커버리지는 기술 통계다."

§11 은 클래스별 커버리지를 표에 적었지만, 그 숫자에는 보장이 붙어 있지
않다. 보장이 붙은 절차로 다시 재야 "클래스별로도 무너졌다" 를 주장할 수
있다. 그 절차가 라벨 조건부 conformal(Mondrian conformal, Vovk 2012)
이다 — 보정셋을 참라벨로 쪼개 클래스마다 따로 분위수를 잡는다. 검증 시점에
참라벨을 몰라도 된다. 후보 라벨 y 마다 그 y 의 분위수와 비교하면 된다.

    주변형   y ∈ C(x)  ⇔  1 − p(y|x) ≤ q           (q 는 보정셋 전체)
    Mondrian y ∈ C(x)  ⇔  1 − p(y|x) ≤ q_y         (q_y 는 라벨 y 보정셋)

왜 이것이 단순한 보완이 아닌가
----------------------------
라벨 조건부 conformal 은 순수 라벨이동(label shift)에서는 그대로
유효하다 — 클래스 안에서의 교환가능성만 쓰기 때문에 P(y) 가 바뀌어도
클래스별 커버리지 보장이 살아 있다(Podkopaev & Ramdas 2021). 이 데이터는
학습군 양성률 0.65/0.77 에서 검증군 0.31/0.61 로 떨어진다. 그러므로

    Mondrian 이 명목값을 회복한다  → §11 의 붕괴는 라벨이동으로 설명된다
    Mondrian 도 명목값에 못 미친다 → 라벨이동으로 환원되지 않는 성분이 있다

둘 중 어느 쪽인지가 §8(유병률 통제)·§33(이동량 맞춤)의 결론과 직접
맞물린다. 그래서 이 스크립트는 한계 5번을 닫는 동시에 하나를 더 묻는다.

세 가지 분할
-----------
    A. 시간분할                     (§11 과 같은 분할, 1회)
    B. 무작위 분할                  (크기 일치, %d회 — 교환가능성이 성립하는 기준선)
    C. 무작위 분할 + 양성률만 맞춤  (순수 라벨이동 모사, %d회)

C 는 B 의 검증셋을 시간분할 검증군의 양성률로 부분표집한 것이다.
공변량 이동은 없고 P(y) 만 바뀌므로, 이론대로면 주변형은 무너지고
Mondrian 은 버틴다. C 는 구현이 맞다는 확인이자 A 를 읽는 기준이 된다.

판정에는 표준오차를 쓴다
----------------------
시간분할은 1회뿐이므로 커버리지의 이항 표준오차
sqrt(target(1-target)/n_y) 로 단측 z 를 만든다. 대조군은 반복 간
표준편차를 함께 적는다. 점추정값 비교만으로 판정하지 않는다.

사용법
------
    python src/mondrian_conformal.py
    python src/mondrian_conformal.py --reps 20
    python src/mondrian_conformal.py --fresh
    python src/mondrian_conformal.py --judge-only

출력
----
    results/mondrian_conformal.csv
    results/mondrian_conformal_summary.txt
    results/mondrian_conformal_progress.txt
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

OUT = os.path.join(RES, "mondrian_conformal.csv")
PROG = os.path.join(RES, "mondrian_conformal_progress.txt")
CALIB_FRAC = 0.25
ALPHAS = (0.10, 0.20)
ARMS = ("A. 시간분할", "B. 무작위", "C. 무작위+양성률 맞춤")


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))


def qhat_of(scores, alpha):
    """split conformal 분위수. n 이 작아 1 을 넘으면 최대값으로 잘린다."""
    n = len(scores)
    if n == 0:
        return float("nan"), True
    lv = np.ceil((n + 1) * (1 - alpha)) / n
    capped = lv > 1.0
    lv = min(lv, 1.0)
    return float(np.quantile(scores, lv, method="higher")), bool(capped)


def scores_true(p1, y):
    """참라벨에 대한 비적합도 1 - p(참라벨)."""
    return 1.0 - np.where(y == 1, p1, 1.0 - p1)


def eval_sets(in0, in1, y):
    size = in0.astype(int) + in1.astype(int)
    cov = np.where(y == 1, in1, in0)
    pm, nm = (y == 1), (y == 0)
    # 열 이름에 cov / size / empty 를 쓰면 pandas Series 의 메서드·속성과
    # 충돌한다 (r.cov 가 공분산 메서드가 되어 조용히 엉뚱한 값이 된다).
    return dict(
        cov_all=float(cov.mean()),
        cov_pos=float(cov[pm].mean()) if pm.sum() else float("nan"),
        cov_neg=float(cov[nm].mean()) if nm.sum() else float("nan"),
        n_pos=int(pm.sum()),
        n_neg=int(nm.sum()),
        set_size=float(size.mean()),
        empty_rate=float((size == 0).mean()),
    )


def marginal_arm(p_cal, y_cal, p_te, y_te, alpha):
    q, capped = qhat_of(scores_true(p_cal, y_cal), alpha)
    r = eval_sets((1.0 - (1.0 - p_te)) <= q, (1.0 - p_te) <= q, y_te)
    r.update(q_neg=q, q_pos=q, capped_neg=capped, capped_pos=capped)
    return r


def mondrian_arm(p_cal, y_cal, p_te, y_te, alpha):
    """라벨마다 따로 분위수를 잡는다 — 검증 라벨은 쓰지 않는다."""
    s = scores_true(p_cal, y_cal)
    q1, c1 = qhat_of(s[y_cal == 1], alpha)
    q0, c0 = qhat_of(s[y_cal == 0], alpha)
    r = eval_sets(p_te <= q0, (1.0 - p_te) <= q1, y_te)
    r.update(q_neg=q0, q_pos=q1, capped_neg=c0, capped_pos=c1)
    return r


METHODS = (("주변형", marginal_arm), ("Mondrian", mondrian_arm))


def temporal_split(df):
    """§11 과 같은 식으로 나눈다 — 값이 재현되는지 확인할 수 있도록."""
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    tr_idx = np.where(tr_m)[0]
    srt = tr_idx[np.argsort(yr[tr_idx])]
    n_cal = int(len(srt) * CALIB_FRAC)
    return srt[:-n_cal], srt[-n_cal:], np.where(te_m)[0]


# 열 집합을 고정한다. A(1회) 행에는 반복 표준편차가 없는데, 덧붙이기
# 모드에서 행마다 열이 다르면 CSV 가 깨진다(드라이런에서 실제로 깨졌다).
COLS = [
    "dataset",
    "arm",
    "method",
    "alpha",
    "reps",
    "cov_all",
    "cov_all_sd",
    "cov_pos",
    "cov_pos_sd",
    "cov_neg",
    "cov_neg_sd",
    "n_pos",
    "n_neg",
    "set_size",
    "set_size_sd",
    "empty_rate",
    "empty_rate_sd",
    "q_neg",
    "q_pos",
    "capped_neg",
    "capped_pos",
]


def append_row(row):
    unknown = [k for k in row if k not in COLS]
    if unknown:
        raise SystemExit("COLS 에 없는 열: %s" % unknown)
    d = pd.DataFrame([{c: row.get(c, float("nan")) for c in COLS}], columns=COLS)
    d.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)


def done_set():
    if not os.path.exists(OUT):
        return set()
    d = pd.read_csv(OUT)
    return set(zip(d.dataset, d.arm, d.method, d.alpha.round(3).astype(str)))


def run(name, fn, reps, done):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    fit_i, cal_i, te_i = temporal_split(df)
    rate_te = float(y[te_i].mean())

    log("=" * 118)
    log(
        "%s   적합 %d / 보정 %d / 검증 %d   보정셋 양성 %d·음성 %d   검증 양성률 %.4f"
        % (
            name,
            len(fit_i),
            len(cal_i),
            len(te_i),
            int(y[cal_i].sum()),
            int((y[cal_i] == 0).sum()),
            rate_te,
        )
    )
    log("=" * 118)

    # ---- A. 시간분할 (1회)
    m = S.make_model().fit(X[fit_i], y[fit_i])
    p_cal = m.predict_proba(X[cal_i])[:, 1]
    p_te = m.predict_proba(X[te_i])[:, 1]
    for mname, f in METHODS:
        for al in ALPHAS:
            k = (name, ARMS[0], mname, "%.3f" % al)
            if k in done:
                continue
            r = f(p_cal, y[cal_i], p_te, y[te_i], al)
            append_row(
                dict(
                    dataset=name,
                    arm=ARMS[0],
                    method=mname,
                    alpha=al,
                    reps=1,
                    **{kk: vv for kk, vv in r.items()}
                )
            )
    progress("%s A 완료" % name)

    # ---- B·C. 무작위 분할 (적합/보정 크기 일치)
    n_fit, n_cal, n_te = len(fit_i), len(cal_i), len(te_i)
    need = {}
    for mname, _f in METHODS:
        for al in ALPHAS:
            for arm in ARMS[1:]:
                if (name, arm, mname, "%.3f" % al) not in done:
                    need[(arm, mname, al)] = []
    if not need:
        log("  대조군은 이미 계산되어 있다 — 건너뛴다.")
        log("")
        return
    for i in range(reps):
        rng = np.random.default_rng(S.SEED + 9100 + i)
        idx = rng.permutation(len(y))
        f_i, c_i = idx[:n_fit], idx[n_fit : n_fit + n_cal]
        t_i = idx[n_fit + n_cal : n_fit + n_cal + n_te]
        mm = S.make_model().fit(X[f_i], y[f_i])
        pc = mm.predict_proba(X[c_i])[:, 1]
        pt = mm.predict_proba(X[t_i])[:, 1]
        # C — 검증셋만 시간분할 양성률로 부분표집 (공변량 이동 없음)
        t_c = S.subsample_to_rate(y, t_i, rate_te, rng)
        pt_c = mm.predict_proba(X[t_c])[:, 1]
        for (arm, mname, al), acc in need.items():
            f = dict(METHODS)[mname]
            if arm == ARMS[1]:
                acc.append(f(pc, y[c_i], pt, y[t_i], al))
            else:
                acc.append(f(pc, y[c_i], pt_c, y[t_c], al))
        if (i + 1) % 5 == 0 or i == 0:
            progress("%s 대조군 %d/%d" % (name, i + 1, reps))
    for (arm, mname, al), acc in need.items():
        row = dict(dataset=name, arm=arm, method=mname, alpha=al, reps=len(acc))
        for kk in acc[0]:
            vals = np.array([a[kk] for a in acc], dtype=float)
            row[kk] = float(np.nanmean(vals))
            if kk in ("cov_all", "cov_pos", "cov_neg", "set_size", "empty_rate"):
                row[kk + "_sd"] = float(np.nanstd(vals, ddof=1))
        append_row(row)
    log("  대조군 %d회 완료" % reps)
    log("")


def z_below(cov, n, target):
    """커버리지가 명목값보다 낮은지 — 이항 표준오차 기준 단측 z."""
    if n <= 0 or cov != cov:
        return float("nan")
    se = (target * (1 - target) / n) ** 0.5
    return (cov - target) / se if se > 0 else float("nan")


def judge():
    if not os.path.exists(OUT):
        log("계산된 결과가 없다 — 중단한다.")
        return
    d = pd.read_csv(OUT)
    ref = os.path.join(RES, "conformal_prediction.csv")
    cp = pd.read_csv(ref) if os.path.exists(ref) else None

    for ds in d.dataset.unique():
        for al in sorted(d[d.dataset == ds].alpha.unique()):
            tgt = 1 - al
            log("=" * 118)
            log("%s   목표 커버리지 %.0f%%" % (ds, tgt * 100))
            log("=" * 118)
            log(
                "%-22s %-9s %15s %9s %9s %8s %8s %8s %8s"
                % (
                    "분할",
                    "방법",
                    "전체",
                    "양성",
                    "음성",
                    "집합크기",
                    "공집합률",
                    "양성 z",
                    "음성 z",
                )
            )
            log("-" * 118)
            for arm in ARMS:
                for mname, _f in METHODS:
                    g = d[
                        (d.dataset == ds) & (d.alpha == al) & (d.arm == arm) & (d.method == mname)
                    ]
                    if not len(g):
                        continue
                    r = g.iloc[0]
                    zp = z_below(r["cov_pos"], r["n_pos"], tgt)
                    zn = z_below(r["cov_neg"], r["n_neg"], tgt)
                    _s = r.get("cov_all_sd")
                    _c = ("%.4f±%.3f" % (r["cov_all"], _s)) if _s == _s else "%.4f" % r["cov_all"]
                    log(
                        "%-22s %-9s %15s %9.4f %9.4f %8.3f %8.4f %8.1f %8.1f"
                        % (
                            arm,
                            mname,
                            _c,
                            r["cov_pos"],
                            r["cov_neg"],
                            r["set_size"],
                            r["empty_rate"],
                            zp,
                            zn,
                        )
                    )
            log("")

    # ---- §11 재현 확인
    if cp is not None:
        log("§11 재현 확인 (주변형·시간분할이 conformal_prediction.csv 와 같은가)")
        worst = 0.0
        for ds in d.dataset.unique():
            for al in sorted(d[d.dataset == ds].alpha.unique()):
                q = cp[(cp.dataset == ds) & ((cp.alpha - al).abs() < 1e-9)]
                g = d[
                    (d.dataset == ds)
                    & (d.alpha == al)
                    & (d.arm == ARMS[0])
                    & (d.method == "주변형")
                ]
                if not (len(q) and len(g)):
                    continue
                dif = abs(float(q.marginal_coverage_temporal.iloc[0]) - float(g["cov_all"].iloc[0]))
                worst = max(worst, dif)
                log(
                    "  %s α=%.2f  §11 %.6f  여기 %.6f  차 %.2e"
                    % (
                        ds,
                        al,
                        float(q.marginal_coverage_temporal.iloc[0]),
                        float(g["cov_all"].iloc[0]),
                        dif,
                    )
                )
        log(
            "  최대차 %.2e — %s"
            % (worst, "같다" if worst < 1e-9 else "**다르다. 분할이 어긋났다**")
        )
        log("")

    # ---- 판정
    log("=" * 118)
    log("판정")
    log("=" * 118)
    log("[읽는 법] **클래스별 커버리지는 라벨이동에 영향을 받지 않는다** — 부분표집은")
    log("          클래스 안의 분포를 바꾸지 않는다. 라벨이동이 바꾸는 것은 두 클래스")
    log("          커버리지의 **혼합비**, 즉 전체 커버리지다. 주변형은 클래스별")
    log("          커버리지가 한쪽으로 쏠려 있어(양성 1.0, 음성 낮음) 혼합비가 바뀌면")
    log("          전체 커버리지가 무너진다. Mondrian 은 클래스마다 명목값을 보장하므로")
    log("          **어떤 혼합비에서도** 전체 커버리지가 명목값 이상이다 — 임의의")
    log("          라벨이동에 유효하다(Vovk 2012; Podkopaev & Ramdas 2021).")
    log("          그래서 A 에서 Mondrian 이 명목값에 못 미치면 그 미달분은")
    log("          라벨이동으로 환원되지 않는 성분이다.")
    log("")

    def get(ds, al, arm, mname):
        g = d[(d.dataset == ds) & (d.alpha == al) & (d.arm == arm) & (d.method == mname)]
        return g.iloc[0] if len(g) else None

    for ds in d.dataset.unique():
        for al in sorted(d[d.dataset == ds].alpha.unique()):
            tgt = 1 - al
            b_m, b_d = get(ds, al, ARMS[1], "주변형"), get(ds, al, ARMS[1], "Mondrian")
            c_m, c_d = get(ds, al, ARMS[2], "주변형"), get(ds, al, ARMS[2], "Mondrian")
            a_m, a_d = get(ds, al, ARMS[0], "주변형"), get(ds, al, ARMS[0], "Mondrian")
            log("  %s   목표 %.2f" % (ds, tgt))

            # 구현 확인 1 — 교환가능 기준선에서 Mondrian 이 클래스별로 명목값을 지키는가
            if b_d is not None:
                _sdp = b_d.get("cov_pos_sd")
                _sdn = b_d.get("cov_neg_sd")
                _sdp = 0.0 if _sdp != _sdp else float(_sdp)
                _sdn = 0.0 if _sdn != _sdn else float(_sdn)
                _ok = b_d["cov_pos"] + 2 * _sdp >= tgt and b_d["cov_neg"] + 2 * _sdn >= tgt
                log(
                    "    확인1 B(교환가능) Mondrian 클래스별 %.4f / %.4f → %s"
                    % (
                        b_d["cov_pos"],
                        b_d["cov_neg"],
                        "보장대로 명목값 수준" if _ok else "**명목 미달 — 해석 전에 구현을 볼 것**",
                    )
                )

            # 구현 확인 2 — 라벨이동만 주면 주변형의 전체 커버리지가 떨어지는가
            if all(x is not None for x in (b_m, c_m, b_d, c_d)):
                _dm = c_m["cov_all"] - b_m["cov_all"]
                _dd = c_d["cov_all"] - b_d["cov_all"]
                _sd = c_m.get("cov_all_sd")
                _sd = 0.0 if _sd != _sd else float(_sd)
                log(
                    "    확인2 B→C(라벨이동만) 전체 커버리지 주변형 %+.4f, Mondrian %+.4f"
                    "  (C 주변형 %.4f, Mondrian %.4f) → %s"
                    % (
                        _dm,
                        _dd,
                        c_m["cov_all"],
                        c_d["cov_all"],
                        (
                            "이론대로다 — 주변형만 흔들린다"
                            if abs(_dm) > abs(_dd)
                            else "이론과 다르다 — 해석 전에 구현을 볼 것"
                        ),
                    )
                )

            # 본 판정 — A 에서 클래스별로 명목값에 미달하는가
            if a_m is None or a_d is None:
                log("")
                continue
            for lab, r in (("주변형", a_m), ("Mondrian", a_d)):
                for cls, ck, nk in (("양성", "cov_pos", "n_pos"), ("음성", "cov_neg", "n_neg")):
                    z = z_below(r[ck], r[nk], tgt)
                    v = (
                        "명목 미달 (유의, 단측 z < -1.645)"
                        if z < -1.645
                        else (
                            "명목 이상"
                            if r[ck] >= tgt
                            else "미달이지만 이 표본으로는 유의하지 않다"
                        )
                    )
                    log(
                        "    A %-8s %s 커버리지 %.4f (n=%d, z=%+.2f) → %s"
                        % (lab, cls, r[ck], int(r[nk]), z, v)
                    )
            _zp = z_below(a_d["cov_pos"], a_d["n_pos"], tgt)
            _zn = z_below(a_d["cov_neg"], a_d["n_neg"], tgt)
            _bad = [c for c, z in (("양성", _zp), ("음성", _zn)) if z < -1.645]
            log(
                "    → %s"
                % (
                    "**Mondrian 도 %s 클래스에서 명목 미달이다. 보장을 붙여도 "
                    "커버리지가 회복되지 않는다 — 라벨이동으로 환원되지 않는 "
                    "성분이 있다.**" % "·".join(_bad)
                    if _bad
                    else "**Mondrian 은 두 클래스 모두 명목값에 유의하게 미달하지 "
                    "않는다. 한계 5번의 '주변 커버리지만 보장' 은 클래스 조건부 "
                    "절차로 닫힌다. 예측집합 크기 변화는 %+.3f 다.**"
                    % (a_d["set_size"] - a_m["set_size"])
                )
            )
            log(
                "    주변형은 클래스별 커버리지가 %.4f / %.4f 로 쏠려 있다 — "
                "전체 %.4f 는 검증군 양성률을 가중치로 한 그 둘의 가중평균이다."
                % (a_m["cov_pos"], a_m["cov_neg"], a_m["cov_all"])
            )
            log("")
    log("저장: %s" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()
    if a.fresh and os.path.exists(OUT):
        os.remove(OUT)
    log("라벨 조건부(Mondrian) conformal — 한계 5번")
    log(
        "보정셋을 참라벨로 쪼개 클래스마다 분위수를 잡는다. 대조군 %d회, 시드 %d" % (a.reps, S.SEED)
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
    log.save(Path(RES) / "mondrian_conformal_summary.txt")


if __name__ == "__main__":
    main()
