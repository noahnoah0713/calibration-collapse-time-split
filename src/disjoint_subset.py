"""
비중첩 부분집합에서 주 결과 재계산 (2026-09-28)

왜 필요한가
-----------
§18 '방법론적 선택' 은 이렇게 적어 뒀다:

    두 데이터셋은 화합물의 대부분을 공유한다. (...) "두 데이터셋에서
    재현" 은 독립 표본 두 개에서의 재현이 아니라 같은 화학공간에서
    엔드포인트만 바꾼 확인이다. 진짜 독립 확인을 원하면 비중첩
    부분집합에서 주 결과를 재계산해야 한다 — 하지 않았다.

이 스크립트가 그 "하지 않았다" 를 실행한다. InChIKey14 기준으로 상대
데이터셋에 **없는** 분자만 남기고 주 결과(AUROC·ECE·Murphy 분해)를
다시 계산한다. 그러면 두 데이터셋의 추정치가 공유하는 분자가 0개가
되므로, 재현이 독립 표본에서의 재현이 된다.

설계 3종
--------
V1  완전 분리   : 학습군·검증군 모두 전용 분자만. 두 데이터셋이 공유하는
                  분자가 학습에도 검증에도 없다. 가장 강한 독립성이지만
                  n 이 가장 작다.
V2  검증군 분리 : 학습군은 전체(<2010) 를 쓰고 검증군만 전용 분자.
                  학습 표본을 잃지 않으면서 '평가에 쓰인 분자가 겹치지
                  않는다' 를 확보한다.
V3  검증군 공유 : V2 의 대조군. 학습군 전체 + 검증군은 **공유** 분자.
                  V2 와 V3 의 차이가 곧 '겹침이 결과를 만들었는가' 다.
                  (V2 와 V3 는 같은 모델을 쓰고 평가 분자만 갈린다)
V1c 학습군 축소 : V1 의 교란 통제. V1 은 학습군이 전용 분자로 줄면서
                  **표본이 작아지는 효과**와 **전용 분자만 쓰는 효과**가 섞인다.
                  V1c 는 전체 학습군에서 같은 크기(180 / 199)를 무작위로
                  뽑아 같은 검증군에 적용한 것을 N_SHRINK 회 평균한다.
                  V1 ≈ V1c 이면 V1 의 성능 저하는 단순히 학습 표본이
                  작아서 생긴 것이고, V1 < V1c 이면 전용 분자 쪽에
                  추가 요인이 있다는 뜻이다.

각 설계마다 무작위 분할 대조군(R1)을 같은 (학습 n, 검증 n) 으로
reps 회 돌려 ECE 배율과 Phipson–Smyth p 를 낸다.

표본 부족의 정직한 처리
-----------------------
DILIrank 전용 분자의 검증군은 21개(양성 6 / 음성 15)뿐이다. ECE 를
10구간으로 계산하면 구간당 2개꼴이라 수치에 의미가 없다. 그래서
  - 검증군 n, 양성/음성 개수를 항상 같이 인쇄하고
  - ECE·AUROC 에 부트스트랩 2.5–97.5 구간을 붙이고
  - n_test < MIN_N(=40) 또는 소수 클래스 < 10 이면 행에 '표본부족'
    표시를 달아 **결론에 쓰지 말라고 명시**한다.
표본이 부족하다는 사실 자체가 결과다 — 숫자를 지우지 않고 남긴다.

실행
----
    python src/disjoint_subset.py              # reps 200 (기본)
    python src/disjoint_subset.py --reps 30    # 점검용
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MIN_N = 40  # 이보다 작으면 표본부족 표시
MIN_MINORITY = 10  # 소수 클래스가 이보다 작으면 표본부족 표시
N_BOOT = 2000
N_SHRINK = 20  # V1c 학습군 축소 대조의 반복 횟수

log = S.Log()


def boot_ci(y, p, fn, n_boot=N_BOOT, seed=S.SEED):
    """관측 표본 부트스트랩 2.5–97.5 구간. 한쪽 클래스만 나온 재표본은 버린다."""
    r = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = r.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(fn(y[idx], p[idx]))
        except Exception:
            continue
    if len(vals) < 100:
        return (np.nan, np.nan, len(vals))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals))


def evaluate(X, y, tr_idx, te_idx):
    """학습 인덱스로 적합, 검증 인덱스에서 지표. 지표는 _shared 구현 그대로."""
    m = S.make_model().fit(X[tr_idx], y[tr_idx])
    p = m.predict_proba(X[te_idx])[:, 1]
    yt = y[te_idx]
    d = S.decompose(yt, p)
    return dict(
        auroc=float(roc_auc_score(yt, p)),
        ece=S.ece(yt, p),
        slope=S.calib_slope(yt, p),
        rel=d["reliability"],
        res=d["resolution"],
        unc=d["uncertainty"],
        p_hat=p,
        y_true=yt,
    )


def random_control(X, y, n_train, n_test, reps):
    """같은 (학습 n, 검증 n) 의 무작위 분할 대조군. 시간 정보를 쓰지 않는다."""
    eces, aucs, rels, ress = [], [], [], []
    N = len(y)
    for i in range(reps):
        r = np.random.default_rng(S.SEED + i)
        perm = r.permutation(N)
        te_i, tr_i = perm[:n_test], perm[n_test : n_test + n_train]
        if len(np.unique(y[te_i])) < 2 or len(np.unique(y[tr_i])) < 2:
            continue
        m = S.make_model().fit(X[tr_i], y[tr_i])
        pp = m.predict_proba(X[te_i])[:, 1]
        d = S.decompose(y[te_i], pp)
        eces.append(S.ece(y[te_i], pp))
        aucs.append(float(roc_auc_score(y[te_i], pp)))
        rels.append(d["reliability"])
        ress.append(d["resolution"])
    return (np.array(eces), np.array(aucs), np.array(rels), np.array(ress))


def flag(n_test, n_pos, n_neg):
    if n_test < MIN_N or min(n_pos, n_neg) < MIN_MINORITY:
        return "표본부족"
    return ""


def run_shrunk(name, X, y, tr_mask, te_idx, n_train_cap, reps):
    """V1c — 전체 학습군에서 n_train_cap 개를 무작위로 뽑아 같은 검증군 평가.

    V1 의 '학습 표본이 작다' 는 효과만 따로 떼어 보는 교란 통제다.
    N_SHRINK 회 평균하며, 대조군(R1)은 V1 과 같은 (학습 n, 검증 n) 이므로
    같은 함수를 다시 쓴다.
    """
    pool = np.where(tr_mask)[0]
    n_pos = int(y[te_idx].sum())
    n_neg = len(te_idx) - n_pos
    fl = flag(len(te_idx), n_pos, n_neg)
    log(
        f"\n  --- V1c. 학습군 축소 대조 (전체 학습군에서 {n_train_cap}개 무작위, "
        f"{N_SHRINK}회 평균)"
    )
    log(
        f"      학습 n={n_train_cap} (전체 {len(pool)}개 중)   "
        f"검증 n={len(te_idx)} (양성 {n_pos} / 음성 {n_neg})" + (f"   <-- {fl}" if fl else "")
    )

    A, E, R, S_, SL = [], [], [], [], []
    for i in range(N_SHRINK):
        r = np.random.default_rng(20_000 + S.SEED + i)
        tri = r.choice(pool, n_train_cap, replace=False)
        if len(np.unique(y[tri])) < 2:
            continue
        m = S.make_model().fit(X[tri], y[tri])
        pp = m.predict_proba(X[te_idx])[:, 1]
        d = S.decompose(y[te_idx], pp)
        A.append(float(roc_auc_score(y[te_idx], pp)))
        E.append(S.ece(y[te_idx], pp))
        R.append(d["reliability"])
        S_.append(d["resolution"])
        SL.append(S.calib_slope(y[te_idx], pp))
    if not A:
        log("      계산 불가")
        return None
    A, E, R, S_ = map(np.array, (A, E, R, S_))
    log(
        f"      시간분할  AUROC {A.mean():.4f} ± {A.std(ddof=1):.4f}   "
        f"ECE {E.mean():.4f} ± {E.std(ddof=1):.4f}   "
        f"보정기울기 {np.mean(SL):.4f}"
    )
    log(f"                rel {R.mean():.4f}  res {S_.mean():.4f}")

    ec, ac, rc, sc = random_control(X, y, n_train_cap, len(te_idx), reps)
    if len(ec) == 0:
        log("      대조군 계산 불가")
        return None
    ratio = E.mean() / ec.mean()
    r_ge = int((ec >= E.mean()).sum())
    pv = (1 + r_ge) / (1 + len(ec))
    d_rel = R.mean() - rc.mean()
    d_res = sc.mean() - S_.mean()
    share = 100 * d_rel / (d_rel + d_res) if (d_rel + d_res) else np.nan
    log(
        f"      무작위 대조 {len(ec)}회   AUROC {ac.mean():.4f}   "
        f"ECE {ec.mean():.4f} ± {ec.std(ddof=1):.4f}"
    )
    log(
        f"      ΔAUROC {ac.mean()-A.mean():+.4f}   ECE 배율 {ratio:.2f}   "
        f"이상 {r_ge}/{len(ec)}   p {pv:.4f}"
    )
    log(f"      Brier 보정 기여율 {share:.1f}%  " f"(Δrel {d_rel:+.4f} / Δres {d_res:+.4f})")
    return dict(
        dataset=name,
        variant="V1c",
        desc=f"학습군 축소 대조 ({n_train_cap}개 무작위, {N_SHRINK}회 평균)",
        n_train=n_train_cap,
        n_test=len(te_idx),
        n_test_pos=n_pos,
        n_test_neg=n_neg,
        pos_rate_train=float(np.nan),
        pos_rate_test=float(y[te_idx].mean()),
        auroc=float(A.mean()),
        auroc_lo=float(A.min()),
        auroc_hi=float(A.max()),
        ece=float(E.mean()),
        ece_lo=float(E.min()),
        ece_hi=float(E.max()),
        slope=float(np.mean(SL)),
        rel=float(R.mean()),
        res=float(S_.mean()),
        unc=float(np.nan),
        ctrl_reps=len(ec),
        ctrl_auroc=float(ac.mean()),
        ctrl_ece=float(ec.mean()),
        ctrl_ece_sd=float(ec.std(ddof=1)),
        ctrl_rel=float(rc.mean()),
        ctrl_res=float(sc.mean()),
        delta_auroc=float(ac.mean() - A.mean()),
        ece_ratio=float(ratio),
        n_ge=r_ge,
        p_phipson_smyth=pv,
        d_rel=float(d_rel),
        d_res=float(d_res),
        share_calib=float(share),
        underpowered=bool(fl),
    )


def run_one(name, X, y, tr_mask, te_mask, variant, desc, sel_te, reps):
    """variant 하나를 계산. sel_te 는 검증군으로 쓸 불리언 마스크."""
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask & sel_te)[0]
    n_pos = int(y[te_idx].sum())
    n_neg = len(te_idx) - n_pos
    fl = flag(len(te_idx), n_pos, n_neg)

    log(f"\n  --- {variant}. {desc}")
    log(
        f"      학습 n={len(tr_idx)} (양성 {y[tr_idx].mean()*100:.1f}%)   "
        f"검증 n={len(te_idx)} (양성 {n_pos} / 음성 {n_neg}, "
        f"{y[te_idx].mean()*100:.1f}%)" + (f"   <-- {fl}" if fl else "")
    )
    if len(te_idx) < 10 or min(n_pos, n_neg) < 2:
        log("      계산 불가 (검증군이 너무 작거나 한쪽 클래스만 있다)")
        return None

    ev = evaluate(X, y, tr_idx, te_idx)
    a_lo, a_hi, _ = boot_ci(ev["y_true"], ev["p_hat"], lambda yy, pp: float(roc_auc_score(yy, pp)))
    e_lo, e_hi, _ = boot_ci(ev["y_true"], ev["p_hat"], lambda yy, pp: S.ece(yy, pp))
    log(
        f"      시간분할  AUROC {ev['auroc']:.4f} [{a_lo:.3f}–{a_hi:.3f}]   "
        f"ECE {ev['ece']:.4f} [{e_lo:.3f}–{e_hi:.3f}]   "
        f"보정기울기 {ev['slope']:.4f}"
    )
    log(f"                rel {ev['rel']:.4f}  res {ev['res']:.4f}  " f"unc {ev['unc']:.4f}")

    ec, ac, rc, sc = random_control(X, y, len(tr_idx), len(te_idx), reps)
    if len(ec) == 0:
        log("      대조군 계산 불가")
        return None
    ratio = ev["ece"] / ec.mean()
    r_ge = int((ec >= ev["ece"]).sum())
    p = (1 + r_ge) / (1 + len(ec))
    d_rel = ev["rel"] - rc.mean()
    d_res = sc.mean() - ev["res"]
    share = 100 * d_rel / (d_rel + d_res) if (d_rel + d_res) else np.nan
    log(
        f"      무작위 대조 {len(ec)}회   AUROC {ac.mean():.4f}   "
        f"ECE {ec.mean():.4f} ± {ec.std(ddof=1):.4f}"
    )
    log(
        f"      ΔAUROC {ac.mean()-ev['auroc']:+.4f}   "
        f"ECE 배율 {ratio:.2f}   이상 {r_ge}/{len(ec)}   p {p:.4f}"
    )
    log(f"      Brier 보정 기여율 {share:.1f}%  " f"(Δrel {d_rel:+.4f} / Δres {d_res:+.4f})")

    return dict(
        dataset=name,
        variant=variant,
        desc=desc,
        n_train=len(tr_idx),
        n_test=len(te_idx),
        n_test_pos=n_pos,
        n_test_neg=n_neg,
        pos_rate_train=float(y[tr_idx].mean()),
        pos_rate_test=float(y[te_idx].mean()),
        auroc=ev["auroc"],
        auroc_lo=a_lo,
        auroc_hi=a_hi,
        ece=ev["ece"],
        ece_lo=e_lo,
        ece_hi=e_hi,
        slope=ev["slope"],
        rel=ev["rel"],
        res=ev["res"],
        unc=ev["unc"],
        ctrl_reps=len(ec),
        ctrl_auroc=float(ac.mean()),
        ctrl_ece=float(ec.mean()),
        ctrl_ece_sd=float(ec.std(ddof=1)),
        ctrl_rel=float(rc.mean()),
        ctrl_res=float(sc.mean()),
        delta_auroc=float(ac.mean() - ev["auroc"]),
        ece_ratio=float(ratio),
        n_ge=r_ge,
        p_phipson_smyth=p,
        d_rel=float(d_rel),
        d_res=float(d_res),
        share_calib=float(share),
        underpowered=bool(fl),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    log("비중첩 부분집합에서 주 결과 재계산")
    log(f"무작위 대조 {args.reps}회, seed {S.SEED}, 절단 {S.CUTOFF}, " f"부트스트랩 {N_BOOT}회")
    log(f"표본부족 기준: 검증 n < {MIN_N} 또는 소수 클래스 < {MIN_MINORITY}")

    frames = {n: S.load(DATA / f) for n, f in S.DATASETS}
    keys = {n: set(d["ikey14"].dropna()) for n, d in frames.items()}
    names = [n for n, _ in S.DATASETS]
    shared = keys[names[0]] & keys[names[1]]

    log("\n" + "=" * 96)
    log("화합물 중첩 (InChIKey14 앞 14자리 기준, 현행 분석집합)")
    log("=" * 96)
    for n in names:
        log(
            f"  {n:10s} n={len(frames[n])}   공유 {len(shared)}개 "
            f"= {len(shared)/len(frames[n])*100:.1f}%   "
            f"전용 {len(frames[n]) - frames[n]['ikey14'].isin(shared).sum()}개"
        )

    rows = []
    for name in names:
        df = frames[name]
        X = S.featurize(df["smiles"].tolist())
        y = df["y"].values
        tr_mask, te_mask = S.temporal_masks(df)
        is_shared = df["ikey14"].isin(shared).values
        excl = ~is_shared

        log("\n" + "=" * 96)
        log(f"{name}   n={len(df)}   학습군 {int(tr_mask.sum())} / " f"검증군 {int(te_mask.sum())}")
        log(
            f"  전용 분자 {int(excl.sum())}개 "
            f"(학습군 {int((tr_mask & excl).sum())} / "
            f"검증군 {int((te_mask & excl).sum())})"
        )
        log("=" * 96)

        # V1 완전 분리 — 학습군도 전용 분자만
        r = run_one(
            name,
            X,
            y,
            tr_mask & excl,
            te_mask,
            "V1",
            "완전 분리 (학습·검증 모두 전용 분자)",
            excl,
            args.reps,
        )
        if r:
            rows.append(r)
        # V1c 학습군 축소 대조 — V1 의 '표본이 작다' 효과를 분리한다
        r = run_shrunk(
            name, X, y, tr_mask, np.where(te_mask & excl)[0], int((tr_mask & excl).sum()), args.reps
        )
        if r:
            rows.append(r)
        # V2 검증군만 분리 — 학습군은 전체
        r = run_one(
            name, X, y, tr_mask, te_mask, "V2", "검증군만 분리 (학습군 전체)", excl, args.reps
        )
        if r:
            rows.append(r)
        # V3 검증군 공유 — V2 의 대조
        r = run_one(
            name, X, y, tr_mask, te_mask, "V3", "검증군 공유분자 (V2 의 대조)", is_shared, args.reps
        )
        if r:
            rows.append(r)
        # V0 전체 — 본문 값 재현 확인
        r = run_one(
            name,
            X,
            y,
            tr_mask,
            te_mask,
            "V0",
            "전체 검증군 (본문 값 재현 확인)",
            np.ones(len(df), dtype=bool),
            args.reps,
        )
        if r:
            rows.append(r)

    if not rows:
        log("\n결과 없음")
        return
    out = pd.DataFrame(rows)
    out.to_csv(RES / "disjoint_subset.csv", index=False)

    log("\n" + "=" * 96)
    log("요약")
    log("=" * 96)
    log(
        f"{'데이터셋':9s} {'설계':4s} {'검증n':>6s} {'양성/음성':>9s} "
        f"{'AUROC':>7s} {'ECE':>7s} {'배율':>6s} {'p':>7s} {'기여율':>7s} 비고"
    )
    log("-" * 96)
    for r in rows:
        log(
            f"{r['dataset']:9s} {r['variant']:4s} {r['n_test']:6d} "
            f"{str(r['n_test_pos'])+'/'+str(r['n_test_neg']):>9s} "
            f"{r['auroc']:7.4f} {r['ece']:7.4f} {r['ece_ratio']:6.2f} "
            f"{r['p_phipson_smyth']:7.4f} {r['share_calib']:6.1f}% "
            f"{'표본부족' if r['underpowered'] else ''}"
        )
    log("-" * 96)
    log("[읽는 법]")
    log("  V0 는 본문 값(§8·§7.1)과 맞아야 한다 — 구현 확인용이다.")
    log("  V2 대 V3 가 핵심이다: 같은 모델을 같은 데이터셋 검증군에")
    log("  적용하면서 평가 분자만 '전용' 과 '공유' 로 갈랐다. 두 행의")
    log("  배율·기여율이 비슷하면 겹침이 결론을 만든 것이 아니다.")
    log("  V1 은 학습군까지 분리해 가장 강한 독립성을 주지만 n 이 작다.")
    log("  V1 과 V1c 를 나란히 보라: 비슷하면 V1 의 저하는 학습 표본이")
    log("  작아서 생긴 것이고, V1 이 더 나쁘면 전용 분자 쪽에 추가 요인이 있다.")
    log("  '표본부족' 행의 수치는 결론에 쓰지 말 것 — n 과 구간 폭이 답이다.")
    log()
    log(f"저장: {RES/'disjoint_subset.csv'}")
    log(f"      {RES/'disjoint_subset_summary.txt'}")
    log.save(RES / "disjoint_subset_summary.txt")


if __name__ == "__main__":
    main()
