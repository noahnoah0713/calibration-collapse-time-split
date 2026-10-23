"""§10·§11 의 점추정값에서 임의성을 없앤다 — 한계 12번.

왜 필요한가
----------
한계 12번은 이렇게 적혀 있다.

    "§10·§11 의 점추정값은 재현 가능하지만 임의적이다(§9.6). 내부
     보정셋 경계의 동점 배정과 적합셋 행 순서가 값을 흔든다. 결론은 모든
     변형에서 유지되지만 수치는 범위로 써야 한다."

§9.6 은 그 흔들림의 폭을 이미 보여 준다(E1 경계 동점 20회, E2 행 순서
20회). 그런데 본문 §10·§11 의 표에 실린 숫자는 여전히 임의 선택 하나의
결과다 — 어떤 정렬 함수를 쓰느냐로 정해진 한 번의 실현값이다. 폭을 적어
두는 것과, 점추정값 자체를 임의성에서 떼어내는 것은 다른 일이다.

이 스크립트가 하는 것
--------------------
두 임의성을 같이 무작위화해 R회 반복하고(E4), 그 평균을 새 점추정값
으로 쓴다. 평균은 임의 선택에 의존하지 않는다 — 그것이 표준적인 해법이다.

  E1(§9.6)  경계 동점 배정만 무작위
  E2(§9.6)  적합셋 행 순서만 무작위
  E4(여기)  둘을 동시에 무작위 + 전체학습 모델의 행 순서도 무작위

전체학습 모델(§10 의 A 행)도 행 순서에 흔들리므로 그것까지 넣는다. 그러면
§10 의 회복률 분모(A − E)도 반복마다 달라지고, 회복률이 두 끝 모두
임의성을 반영한다.

그리고 결론이 모든 반복에서 유지되는지를 센다. 폭만 보고하는 것으로는
"결론은 유지된다" 를 확인할 수 없다 — 반복마다 부등호를 직접 확인해야 한다.
  C1  conformal 90%% 커버리지가 명목값 미달인가
  C2  학습군 내부 재보정(B)의 회복률이 100%% 미만인가
  C3  두 데이터셋에서 가장 잘 듣는 방법이 서로 다른가 (§10 의 결론)

C3 이 특히 중요하다 — §10 은 "DILIrank 는 사전확률 보정, DICTrank 는 Platt"
이라고 적었는데, 그것이 한 번의 실현값에서 나온 순서라면 결론이 아니다.

사용법
------
    python src/tiebreak_averaged.py              # 50회
    python src/tiebreak_averaged.py --reps 100
    python src/tiebreak_averaged.py --fresh

반복 하나가 끝날 때마다 즉시 저장하므로 중단되어도 이어서 돈다.

출력
----
    results/tiebreak_averaged.csv          반복별 원자료
    results/tiebreak_averaged_summary.txt
    results/tiebreak_averaged_progress.txt
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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

OUT = os.path.join(RES, "tiebreak_averaged.csv")
PROG = os.path.join(RES, "tiebreak_averaged_progress.txt")
CALIB_FRAC = 0.25
ALPHAS = (0.10, 0.20)


def progress(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))


def append_row(row):
    df = pd.DataFrame([row])
    if os.path.exists(OUT):
        df.to_csv(OUT, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(OUT, index=False, encoding="utf-8")


def done_set():
    if not os.path.exists(OUT):
        return set()
    try:
        d = pd.read_csv(OUT)
    except Exception:
        return set()
    if "dataset" not in d.columns or "rep" not in d.columns:
        return set()
    return set(zip(d.dataset.astype(str), d.rep.astype(int)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
    return np.log(p / (1 - p))


def platt(p_cal, y_cal, p_te):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_te):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(p_cal, np.asarray(y_cal))
    return np.clip(ir.predict(p_te), S.EPS, 1 - S.EPS)


def prior_shift(p, src, tgt):
    odds = (tgt / (1 - tgt)) / (src / (1 - src))
    return 1.0 / (1.0 + np.exp(-(logit(p) + np.log(odds))))


def conformal_eval(p_cal, y_cal, p_te, y_te, alpha):
    """conformal_prediction.py 와 동일한 구현."""
    s_cal = 1 - np.where(np.asarray(y_cal) == 1, p_cal, 1 - p_cal)
    n = len(s_cal)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(s_cal, q_level, method="higher")
    in1 = (1 - p_te) <= qhat
    in0 = p_te <= qhat
    size = in1.astype(int) + in0.astype(int)
    cov = np.where(np.asarray(y_te) == 1, in1, in0)
    pos, neg = np.asarray(y_te) == 1, np.asarray(y_te) == 0
    return {
        "marginal": float(cov.mean()),
        "pos": float(cov[pos].mean()) if pos.sum() else float("nan"),
        "neg": float(cov[neg].mean()) if neg.sum() else float("nan"),
        "empty": float((size == 0).mean()),
    }


def canonical(yr, idx):
    """연도 → 원본 index 로 완전히 결정되는 순서."""
    idx = np.asarray(idx)
    return idx[np.lexsort((idx, yr[idx]))]


def joint_split(yr, tr_idx, n_cal, rng):
    """E4 — 경계 동점 배정과 적합셋 행 순서를 동시에 무작위화한다."""
    srt = canonical(yr, tr_idx)
    boundary = yr[srt[-n_cal]]
    above = srt[yr[srt] > boundary]
    tie = srt[yr[srt] == boundary]
    k = n_cal - len(above)
    pick = rng.choice(tie, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    cal = np.concatenate([np.sort(pick), above])
    fit = np.setdiff1d(tr_idx, cal, assume_unique=False).copy()
    rng.shuffle(fit)  # 행 순서도 무작위
    return fit, cal


def run(name, fn, reps, done):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    tr_idx, te = np.where(tr_m)[0], np.where(te_m)[0]
    n_cal = int(round(len(tr_idx) * CALIB_FRAC))
    pi_tr, pi_te = float(y[tr_idx].mean()), float(y[te].mean())

    _srt = canonical(yr, tr_idx)
    boundary = int(yr[_srt[-n_cal]])
    n_tie = int((yr[tr_idx] == boundary).sum())

    log("=" * 124)
    log(
        "%s   학습 %d / 검증 %d   보정셋 %d   경계연도 %d (그 해 학습군 %d개)"
        % (name, len(tr_idx), len(te), n_cal, boundary, n_tie)
    )
    log("  E4: 경계 동점 배정 + 적합셋 행 순서 + 전체학습 행 순서를 동시에 무작위화")
    log("=" * 124)

    for r_i in range(reps):
        if (name, r_i) in done:
            continue
        rng = np.random.default_rng(S.SEED + 7000 + r_i)
        # A 행 — 전체 학습, 행 순서만 무작위
        tr_sh = tr_idx.copy()
        rng.shuffle(tr_sh)
        m_all = S.make_model().fit(X[tr_sh], y[tr_sh])
        p_A = m_all.predict_proba(X[te])[:, 1]
        ece_A = S.ece(y[te], p_A)
        # E 행 — 검증군 라벨을 쓴 Oracle Platt 상한
        p_E = platt(p_A, y[te], p_A)
        ece_E = S.ece(y[te], p_E)
        # 보정셋 분할 — 두 임의성 동시
        fit_i, cal_i = joint_split(yr, tr_idx, n_cal, rng)
        m_fit = S.make_model().fit(X[fit_i], y[fit_i])
        p_cal = m_fit.predict_proba(X[cal_i])[:, 1]
        p_tef = m_fit.predict_proba(X[te])[:, 1]
        p_B = platt(p_cal, y[cal_i], p_tef)
        p_C = isotonic(p_cal, y[cal_i], p_tef)
        p_D = prior_shift(p_A, pi_tr, pi_te)
        den = ece_A - ece_E
        row = {
            "dataset": name,
            "rep": r_i,
            "n_fit": len(fit_i),
            "n_cal": len(cal_i),
            "auroc_A": float(roc_auc_score(y[te], p_A)),
            "ece_A": ece_A,
            "ece_E": ece_E,
            "ece_Ap": S.ece(y[te], p_tef),
            "ece_B": S.ece(y[te], p_B),
            "ece_C": S.ece(y[te], p_C),
            "ece_D": S.ece(y[te], p_D),
        }
        for k in ("B", "C", "D"):
            row["rec_%s" % k] = (
                100.0 * (ece_A - row["ece_%s" % k]) / den if den != 0 else float("nan")
            )
        for a in ALPHAS:
            c = conformal_eval(p_cal, y[cal_i], p_tef, y[te], a)
            t = "conf%d" % int(round((1 - a) * 100))
            for kk in ("marginal", "pos", "neg", "empty"):
                row["%s_%s" % (t, kk)] = c[kk]
        # 이 반복에서 가장 잘 듣는 배포 가능 방법 (B·C 중) 과 전체 중 최선
        _dep = {"B": row["ece_B"], "C": row["ece_C"]}
        row["best_deployable"] = min(_dep, key=_dep.get)
        _all = {"B": row["ece_B"], "C": row["ece_C"], "D": row["ece_D"]}
        row["best_any"] = min(_all, key=_all.get)
        append_row(row)
        if (r_i + 1) % 10 == 0 or r_i == 0:
            progress(
                "%s %d/%d 완료 (ece_B %.4f, conf90 %.4f)"
                % (name, r_i + 1, reps, row["ece_B"], row["conf90_marginal"])
            )
    log("  반복 완료")
    log("")


def ref_values():
    """§10·§11 본문 표가 쓰는 그 값을 읽는다.

    처음에는 tiebreak_sensitivity.csv 의 '현행 구현' 행을 참조값으로 썼는데,
    그 스크립트는 모델을 따로 적합하므로 본문 표의 숫자와 다르다(§9.6 의 두
    번째 읽는 법). 참조값이 두 가지면 보고서와 이 요약이 서로 다른 숫자를
    말하게 되므로, 본문이 읽는 파일에서 직접 가져온다.
    """
    rp = os.path.join(RES, "recalibration_results.csv")
    cp = os.path.join(RES, "conformal_prediction.csv")
    if not (os.path.exists(rp) and os.path.exists(cp)):
        return None
    rec = pd.read_csv(rp)
    conf = pd.read_csv(cp)

    def rr(d, pref):
        g = rec[rec.dataset.str.startswith(d)]
        return g[g.method.str.startswith(pref)].iloc[0]

    def rc(d, pref):
        a, e, x = rr(d, "A."), rr(d, "E."), rr(d, pref)
        return (a.ECE - x.ECE) / (a.ECE - e.ECE) * 100.0

    def cf(d, al):
        q = conf[(conf.dataset == d) & ((conf.alpha - al).abs() < 1e-9)]
        return float(q.marginal_coverage_temporal.iloc[0])

    out = {}
    for d, _fn in S.DATASETS:
        out[d] = {
            "ece_A": float(rr(d, "A.").ECE),
            "ece_Ap": float(rr(d, "A'.").ECE),
            "ece_B": float(rr(d, "B.").ECE),
            "ece_C": float(rr(d, "C.").ECE),
            "ece_D": float(rr(d, "D.").ECE),
            "ece_E": float(rr(d, "E.").ECE),
            "rec_B": rc(d, "B."),
            "rec_C": rc(d, "C."),
            "rec_D": rc(d, "D."),
            "conf90_marginal": cf(d, 0.1),
            "conf80_marginal": cf(d, 0.2),
            "auroc_A": float(rr(d, "A.").AUROC),
        }
    return out


def judge(reps, ref):
    if not os.path.exists(OUT):
        log("계산된 반복이 없다 — 중단한다.")
        return
    d = pd.read_csv(OUT)
    QTY = [
        ("ece_A", "A 무보정 ECE"),
        ("ece_Ap", "A' 보정셋 제외 ECE"),
        ("ece_B", "B Platt ECE"),
        ("ece_C", "C Isotonic ECE"),
        ("ece_D", "D 사전확률 ECE"),
        ("ece_E", "E Oracle 상한 ECE"),
        ("rec_B", "B 회복률 (%)"),
        ("rec_C", "C 회복률 (%)"),
        ("rec_D", "D 회복률 (%)"),
        ("conf90_marginal", "conformal 90 커버리지"),
        ("conf80_marginal", "conformal 80 커버리지"),
        ("conf90_empty", "conformal 90 공집합률"),
        ("auroc_A", "A AUROC"),
    ]
    for ds, g in d.groupby("dataset", sort=False):
        log("=" * 124)
        log("%s   반복 %d회 — 평균을 새 점추정값으로 쓴다" % (ds, len(g)))
        log("=" * 124)
        log(
            "%-26s %10s %10s %10s %-20s %12s"
            % ("수치", "평균", "표준편차", "표준오차", "2.5-97.5 백분위", "현행값 거리")
        )
        log("-" * 124)
        for col, lab in QTY:
            if col not in g.columns:
                continue
            v = g[col].dropna()
            if not len(v):
                continue
            m, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            se = sd / np.sqrt(len(v)) if len(v) else float("nan")
            lo, hi = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            cur, dist = "—", "—"
            if ref is not None and ds in ref and col in ref[ds]:
                cur = float(ref[ds][col])
                # 표즜편트가 사실상 0 이면 SD 단위가 무의미하다.
                dist = "%+.1f SD" % ((cur - m) / sd) if sd > 1e-9 else "Δ%+.4f" % (cur - m)
                if not (float(v.min()) <= cur <= float(v.max())):
                    dist += " *범위밖*"
            log("%-26s %10.4f %10.4f %10.4f [%8.4f, %8.4f] %12s" % (lab, m, sd, se, lo, hi, dist))
        log("")
        # 결론이 모든 반복에서 유지되는가
        n = len(g)
        c1 = int((g.conf90_marginal < 0.90).sum())
        c2 = int((g.rec_B < 100).sum())
        log("  결론 점검 (반복 %d회 중)" % n)
        log("    C1 conformal 90%% 커버리지가 명목값 미달: %d/%d" % (c1, n))
        log("    C2 학습군 내부 Platt 회복률이 100%% 미만: %d/%d" % (c2, n))
        _bd = g.best_deployable.value_counts()
        _ba = g.best_any.value_counts()
        log(
            "    배포 가능(B·C) 중 최선: %s"
            % ", ".join("%s %d회" % (k, int(vv)) for k, vv in _bd.items())
        )
        log("    B·C·D 중 최선: %s" % ", ".join("%s %d회" % (k, int(vv)) for k, vv in _ba.items()))
        log("")
    # C3 — 두 데이터셋의 최선 방법이 다른가
    log("=" * 124)
    log("판정")
    log("=" * 124)
    log("[읽는 법] 평균은 임의 선택에 의존하지 않는다. **본문 표의 숫자를 이 평균과")
    log("          백분위 범위로 바꾸면 한계 12번의 '임의적' 이 사라진다.**")
    log("          결론 점검이 반복 전부에서 성립해야 '결론은 유지된다' 고 쓸 수 있다.")
    log("")
    dsl = list(d.dataset.unique())
    if len(dsl) == 2:
        a, b = dsl
        ga, gb = d[d.dataset == a], d[d.dataset == b]
        same = 0
        nn = min(len(ga), len(gb))
        for i in range(nn):
            if ga.sort_values("rep").best_any.iloc[i] == gb.sort_values("rep").best_any.iloc[i]:
                same += 1
        log("  C3 두 데이터셋의 최선 방법이 **같은** 반복: %d/%d" % (same, nn))
        if same == 0:
            log("     → **모든 반복에서 다르다. §10 의 '단일 권고안을 낼 수 없다' 가")
            log("       임의 선택의 산물이 아니다.**")
        elif same == nn:
            log("     → **모든 반복에서 같다. §10 의 서술을 고쳐야 한다.**")
        else:
            log(
                "     → **반복에 따라 갈린다(%d/%d). §10 의 서술을 그 비율로 바꿔야 한다.**"
                % (same, nn)
            )
    log("")
    for ds, g in d.groupby("dataset", sort=False):
        log(
            "  %s — 본문에 쓸 형태: B 회복률 %.1f%% [%.1f, %.1f], "
            "conformal 90 커버리지 %.4f [%.4f, %.4f]"
            % (
                ds,
                g.rec_B.mean(),
                np.percentile(g.rec_B, 2.5),
                np.percentile(g.rec_B, 97.5),
                g.conf90_marginal.mean(),
                np.percentile(g.conf90_marginal, 2.5),
                np.percentile(g.conf90_marginal, 97.5),
            )
        )
    log("")
    log("저장: %s" % OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()
    if a.fresh and os.path.exists(OUT):
        os.remove(OUT)
    log("§10·§11 의 점추정값에서 임의성을 없앤다 — 한계 12번")
    log("경계 동점 배정·적합셋 행 순서·전체학습 행 순서를 **동시에** 무작위화하고")
    log("그 평균을 새 점추정값으로 쓴다. 반복 %d회, 시드 %d" % (a.reps, S.SEED))
    log("")
    ref = ref_values()
    done = done_set()
    if done and not a.judge_only:
        log("이미 끝난 반복 %d개를 건너뛴다 (재개). 처음부터는 --fresh." % len(done))
        log("")
    if not a.judge_only:
        progress("실행 시작 (반복 %d회, 이미 끝난 %d개)" % (a.reps, len(done)))
        for name, fn in S.DATASETS:
            run(name, fn, a.reps, done)
    judge(a.reps, ref)
    log.save(Path(RES) / "tiebreak_averaged_summary.txt")


if __name__ == "__main__":
    main()
