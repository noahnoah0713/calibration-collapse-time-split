"""라벨 지연을 넣어도 주기적 재학습이 남는가 — 한계 25를 닫는다 (2026-10-21).

왜 필요한가
-----------
§34 는 해마다 다시 학습하고 다시 보정하면 ECE 가 0.1655 / 0.1759 내려간다고
냈고, 그것이 이 논문에서 유일하게 작동하는 처방이다. 그런데 §34 는 **승인
연도에 간독성·심장독성 라벨이 이미 확정돼 있다**고 보고 학습창에 넣는다.
현실에서는 라벨이 쌓이는 데 시간이 걸린다(§12 라벨 성숙도). 그래서 §34 의
회복분은 상한이고, 한계 25 가 그 사실을 적으면서 "라벨 지연을 넣으려면
외부 자료가 필요하고 이 데이터로는 할 수 없다" 고 적었다.

**그 절반은 틀렸다.** 실제 지연이 몇 년인지 알려면 외부 자료가 필요하지만,
지연을 **가정으로 쓸어보는 것**은 이 데이터로 할 수 있다. 연도 T 에서
학습창을 만들 때 approval_year < T 대신

    approval_year <= T - 1 - d

로 바꾸면, 승인 후 d 년이 지나 라벨이 익은 분자만 학습에 들어간다. d 를
0·1·2·3·5 로 쓸면 §34 의 회복분이 어느 지연에서 사라지는지 나온다.
결론이 "상한이다" 에서 "지연 d 년까지는 이만큼 남는다" 로 바뀐다.

설계
----
  P0        고정 모델. 2010년 이전 전체로 한 번 학습한다. d 와 무관하므로
            모든 지연의 공통 기준선이다. (P0 의 학습 라벨도 엄밀히는
            2010년에 다 익지 않았지만, §34 와 1:1 로 대응시키기 위해
            그대로 둔다 — 이 점은 결과 요약에 적는다.)
  P1(d)     해마다 재학습, 재보정 없음, 지연 d.
  P2(d)     해마다 재학습 + 창 내부 Platt 재보정, 지연 d.

**평가 분자를 모든 d 에서 같게 맞춘다.** d 가 커지면 학습창이 MIN_FIT 에
못 미쳐 건너뛰는 연도가 생긴다. 그대로 비교하면 분모가 달라져 ΔECE 가
지연 효과인지 평가집합 효과인지 갈리지 않는다. 그래서 먼저 각 d 의 평가
가능 연도를 구하고 **교집합**으로 제한한 뒤 모든 팔을 그 분자에서만 잰다.
짝지은 부트스트랩이 유효해진다.

**정확성 검사.** d=0 을 연도 제한 없이 한 번 더 돌려 §34 의 P1·P2·P0 ECE
와 대조한다. 같은 절차이므로 차가 0 이어야 한다.

대조군은 두지 않는다. 교환가능 기준선과의 비교는 §34 가 d=0 에서 이미
했고, 이 절의 질문은 **d 에 따른 기울기**라서 d=0 과의 짝지은 차로 답한다.

쓰기
----
    python src/prequential_delay.py                # 지연 0,1,2,3,5
    python src/prequential_delay.py --delays 0,2,5
    python src/prequential_delay.py --quick        # 배관 점검 (지연 0,2)
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

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

OUT = RES / "prequential_delay.csv"
DELTA = RES / "prequential_delay_delta.csv"
SUMM = RES / "prequential_delay_summary.txt"
CHECK = RES / "prequential_delay_check.csv"

CALIB_FRAC = 0.25  # §34 와 같다
N_BOOT = 2000  # §34 와 같다
MIN_FIT = 100  # §34 와 같다
DELAYS_DEFAULT = (0, 1, 2, 3, 5)

_lines = []


def log(m=""):
    print(m)
    _lines.append(m)


def logit(p):
    # §34 와 같은 식이고 float64 로 올린다 (2026-10-20 정밀도 수정).
    p = np.clip(np.asarray(p, dtype=np.float64), S.EPS, 1 - S.EPS)
    return np.log(p / (1 - p))


def platt(p_cal, y_cal, p_te):
    if len(np.unique(y_cal)) < 2:
        return np.asarray(p_te, dtype=np.float64)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def prequential(X, y, yr, years, recal, delay):
    """연도별 확장창 예측. 학습창은 approval_year <= T - 1 - delay 다."""
    ps, idxs, sizes = [], [], []
    for T in years:
        te = np.where(yr == T)[0]
        tr = np.where(yr <= T - 1 - delay)[0]
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


def paired_delta(y, p_new, p_old, n_boot=N_BOOT, seed=S.SEED):
    """같은 분자에서 ΔECE = 새 − 기존. 음수면 개선이다. §34 와 같은 절차."""
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
        return float("nan"), float("nan"), float("nan"), 0
    a = np.sort(np.array(v))
    return (float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), len(a))


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


def ref34():
    """§34 의 P0·P1·P2 ECE. 정확성 검사의 기준이다."""
    f = RES / "prequential_retraining.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    d = d[d.rep == 0]
    out = {}
    for _, r in d.iterrows():
        if str(r.arm).startswith(("P0", "P1", "P2")):
            out[(r.dataset, str(r.arm)[:2])] = float(r.ece)
    return out


def run(name, fn, delays, ref):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    pre, post = np.where(tr_m)[0], np.where(te_m)[0]
    years_all = sorted(set(int(v) for v in yr[post]))

    log("=" * 110)
    log(
        "%s   2010년 이전 %d개 / 이후 %d개   검증 연도 %d개 (%d–%d)"
        % (name, len(pre), len(post), len(years_all), years_all[0], years_all[-1])
    )
    log("=" * 110)

    # ---- 정확성 검사: d=0 이 §34 를 재현하는가 (연도 제한 없이)
    p1f, i1f, szf = prequential(X, y, yr, years_all, False, 0)
    p2f, i2f, _ = prequential(X, y, yr, years_all, True, 0)
    assert np.array_equal(i1f, i2f), "P1 과 P2 의 평가 분자가 다르다"
    m0 = S.make_model().fit(X[pre], y[pre])
    p0f = m0.predict_proba(X[i1f])[:, 1]
    chk = [("P0", S.ece(y[i1f], p0f)), ("P1", S.ece(y[i1f], p1f)), ("P2", S.ece(y[i1f], p2f))]
    log("  [정확성] d=0 이 §34 를 재현하는가")
    worst = 0.0
    for arm, v in chk:
        r0 = ref.get((name, arm))
        if r0 is None:
            log("    %s  여기 %.6f   §34 값 없음" % (arm, v))
            continue
        worst = max(worst, abs(v - r0))
        log("    %s  §34 %.6f  여기 %.6f  차 %.2e" % (arm, r0, v, abs(v - r0)))
    log("    최대차 %.2e — %s" % (worst, "같다" if worst < 1e-12 else "**어긋났다**"))
    log()
    # 보고서가 이 수치를 산문에 적지 않도록 CSV 로 남긴다.
    chk_rows = []
    for arm, v in chk:
        r0 = ref.get((name, arm))
        chk_rows.append(
            dict(
                dataset=name,
                arm=arm,
                ref_s34=(float(r0) if r0 is not None else float("nan")),
                here=float(v),
                abs_diff=(abs(v - r0) if r0 is not None else float("nan")),
            )
        )

    # ---- 각 지연의 평가 가능 연도, 그리고 교집합
    year_sets, cache = {}, {}
    for d in delays:
        a1, j1, s1 = prequential(X, y, yr, years_all, False, d)
        a2, j2, _ = prequential(X, y, yr, years_all, True, d)
        assert np.array_equal(j1, j2), "지연 %d 에서 평가 분자가 다르다" % d
        cache[d] = (a1, a2, j1, s1)
        year_sets[d] = set(int(t) for t, _, _ in s1)
        log("  지연 %d년 — 평가 가능 연도 %d개, 분자 %d개" % (d, len(s1), len(j1)))
    common = sorted(set.intersection(*year_sets.values())) if year_sets else []
    log(
        "  모든 지연에서 공통인 연도 %d개%s"
        % (len(common), (" (%d–%d)" % (common[0], common[-1])) if common else "")
    )
    if not common:
        log("  **공통 연도가 없다 — 비교할 수 없다.**")
        return [], [], chk_rows
    log()

    # ---- 공통 연도로 제한해 다시 잰다
    keep_mask = np.isin(yr, common)
    rows = []
    store = {}
    for d in delays:
        a1, a2, j1, _ = cache[d]
        sel = np.isin(j1, np.where(keep_mask)[0])
        idx = j1[sel]
        ys = y[idx]
        q1, q2 = a1[sel], a2[sel]
        p0 = m0.predict_proba(X[idx])[:, 1]
        store[d] = (idx, ys, p0, q1, q2)
        for arm, p in (("P0. 고정 모델", p0), ("P1. 해마다 재학습", q1), ("P2. 재학습+재보정", q2)):
            rows.append(dict(dataset=name, delay=d, arm=arm, n_years=len(common), **metrics(ys, p)))

    # 모든 지연에서 평가 분자가 같아야 짝지은 비교가 유효하다
    base_idx = store[delays[0]][0]
    for d in delays[1:]:
        assert np.array_equal(store[d][0], base_idx), (
            "지연 %d 의 평가 분자가 다르다 — 교집합 제한이 안 먹었다" % d
        )

    log(
        "  공통 연도 기준 — 평가 분자 %d개 (양성률 %.4f)"
        % (len(base_idx), float(store[delays[0]][1].mean()))
    )
    log()
    log("  %-6s %9s %9s %9s %11s %11s" % ("지연", "P0 ECE", "P1 ECE", "P2 ECE", "P1−P0", "P2−P0"))
    log("  " + "-" * 62)
    deltas = []
    for d in delays:
        idx, ys, p0, q1, q2 = store[d]
        d1 = paired_delta(ys, q1, p0)
        d2 = paired_delta(ys, q2, p0)
        log(
            "  %-6d %9.4f %9.4f %9.4f %11.4f %11.4f"
            % (d, S.ece(ys, p0), S.ece(ys, q1), S.ece(ys, q2), d1[0], d2[0])
        )
        for arm, dd in (("P1. 해마다 재학습", d1), ("P2. 재학습+재보정", d2)):
            deltas.append(
                dict(
                    dataset=name,
                    delay=d,
                    arm=arm,
                    vs="P0. 고정 모델",
                    dece=dd[0],
                    lo=dd[1],
                    hi=dd[2],
                    n_boot=dd[3],
                    excludes_zero=bool(dd[1] * dd[2] > 0),
                )
            )
    log()
    log("  지연 0 대비 — 회복분이 얼마나 깎이는가 (양수면 나빠진 것)")
    for d in delays[1:]:
        for ai, arm in ((3, "P1. 해마다 재학습"), (4, "P2. 재학습+재보정")):
            _, ys, _, _, _ = store[d]
            new = store[d][ai]
            old = store[delays[0]][ai]
            dd = paired_delta(ys, new, old)
            log(
                "    지연 %d년  %-18s ΔECE %+.4f [%+.4f, %+.4f]  %s"
                % (d, arm, dd[0], dd[1], dd[2], "유의" if dd[1] * dd[2] > 0 else "구간이 0 을 포함")
            )
            deltas.append(
                dict(
                    dataset=name,
                    delay=d,
                    arm=arm,
                    vs="지연 %d년 같은 팔" % delays[0],
                    dece=dd[0],
                    lo=dd[1],
                    hi=dd[2],
                    n_boot=dd[3],
                    excludes_zero=bool(dd[1] * dd[2] > 0),
                )
            )
    log()
    return rows, deltas, chk_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", type=str, default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.delays:
        delays = tuple(int(v) for v in a.delays.split(","))
    elif a.quick:
        delays = (0, 2)
    else:
        delays = DELAYS_DEFAULT
    delays = tuple(sorted(set(delays)))
    assert delays[0] == 0, "지연 0 이 기준이므로 반드시 넣어야 한다"

    t0 = time.time()
    log("라벨 지연 민감도 — 한계 25")
    log(
        "지연 %s년, 보정셋 비율 %.2f, 부트스트랩 %d회, 최소 학습창 %d, 시드 %d"
        % (",".join(str(d) for d in delays), CALIB_FRAC, N_BOOT, MIN_FIT, S.SEED)
    )
    log()
    ref = ref34()
    rows, deltas, checks = [], [], []
    for name, fn in S.DATASETS:
        r, dl, ck = run(name, fn, delays, ref)
        rows += r
        deltas += dl
        checks += ck
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(deltas).to_csv(DELTA, index=False)
    pd.DataFrame(checks).to_csv(CHECK, index=False)
    log("=" * 110)
    log("읽는 법")
    log("=" * 110)
    log("P2−P0 가 음수이고 구간이 0 을 제외하면 그 지연에서도 처방이 작동한다.")
    log("지연 0 대비 ΔECE 가 양수로 커지면 지연이 회복분을 깎는 것이다.")
    log("P0 은 지연과 무관한 공통 기준선이다 — 2010년 이전 전체로 한 번")
    log("학습하며, 그 라벨도 엄밀히는 2010년에 다 익지 않았다. §34 와")
    log("1:1 로 대응시키기 위해 그대로 두었고 이 점은 한계로 남는다.")
    log()
    log("소요 %.1f분" % ((time.time() - t0) / 60))
    SUMM.write_text("\n".join(_lines), encoding="utf-8")
    print("\n저장: %s\n      %s\n      %s\n      %s" % (OUT, DELTA, CHECK, SUMM))


if __name__ == "__main__":
    main()
