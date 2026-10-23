"""§12.2 — 시간축을 거꾸로 돌린 분할 (미래로 학습, 과거로 검증).

왜 필요한가
----------
§12 는 라벨 검열(최근 승인 약물은 노출 기간이 짧아 아직 독성이 보고되지
않았을 수 있다)을 노출기간 절단·음성 가중·양측 절단으로 통제했다. 그런데
그 셋은 모두 "검증군에서 미성숙한 라벨을 덜어내는" 방향이고, 검열 가설을
정면으로 반증하지는 못한다.

거꾸로 돌리면 정면으로 갈린다. 2010년 이후로 학습하고 2010년
이전으로 검증하면, 검증군은 수십 년간 관찰된 성숙한 라벨이다. 검열
가설이 맞다면 이 방향에서는 보정이 무너지지 않아야 한다. 그래도 무너지면
원인은 라벨 성숙도가 아니라 분포 이동 쪽이다.

정방향과 같은 대조군(R1 크기맞춤 / R2 검증군 유병률맞춤 / R3 양측맞춤)을
같은 반복 수로 붙여 ECE 배율과 순열 p 를 §8 과 같은 눈금으로 읽을 수 있게
했다. 구현은 prevalence_control.py 와 같은 _shared 함수를 쓴다.

주의: 이것은 "미래 데이터로 과거를 예측한다" 는 비현실적 설정이다. 배포
시나리오가 아니라 검열 가설을 가르는 대조 실험으로만 읽어야 한다.

사용법
------
    python src/reverse_split.py                # 대조군 200회
    python src/reverse_split.py --reps 50      # 짧게 점검

출력
----
    results/reverse_split.csv
    results/reverse_split_summary.txt
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


def phipson_smyth(n_ge, n_valid):
    return (1.0 + n_ge) / (1.0 + n_valid)


def evaluate(y_tr, X_tr, y_te, X_te):
    m = S.make_model().fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    d = S.decompose(y_te, p)
    return {
        "auroc": roc_auc_score(y_te, p),
        "ece": S.ece(y_te, p),
        "slope": S.calib_slope(y_te, p),
        "rel": d["reliability"],
        "res": d["resolution"],
        "unc": d["uncertainty"],
    }


def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    fwd_tr, fwd_te = S.temporal_masks(df)  # 정방향: 학습 <2010 / 검증 >=2010
    rev_tr, rev_te = fwd_te, fwd_tr  # 역방향: 학습 >=2010 / 검증 <2010

    n_test = int(rev_te.sum())
    rate_t = float(y[rev_te].mean())
    rate_tr = float(y[rev_tr].mean())

    log("=" * 100)
    log(f"{name}   역방향: 학습 {int(rev_tr.sum())}(>=2010) / " f"검증 {n_test}(<2010)")
    log(f"         정방향 참고: 학습 {int(fwd_tr.sum())} / 검증 {int(fwd_te.sum())}")
    log("=" * 100)
    log(
        f"학습군 양성률 {rate_tr:.4f} / 검증군 양성률 {rate_t:.4f} "
        f"(이동 {rate_t - rate_tr:+.4f})"
    )
    log("  정방향과 이동 **방향이 반대**인 것이 정상이다 — 같은 두 집단의 역할만 바꿨다.")
    log("")

    obs = evaluate(y[rev_tr], X[rev_tr], y[rev_te], X[rev_te])
    fwd = evaluate(y[fwd_tr], X[fwd_tr], y[fwd_te], X[fwd_te])

    rows = [
        {
            "dataset": name,
            "direction": "정방향(참고)",
            "setting": "T. 시간분할",
            "n_test": int(fwd_te.sum()),
            "pos_rate": float(y[fwd_te].mean()),
            "n_rep_valid": np.nan,
            "n_ge": np.nan,
            "p_phipson_smyth": np.nan,
            "ece_ratio": np.nan,
            "share_rel": np.nan,
            **fwd,
        },
        {
            "dataset": name,
            "direction": "역방향",
            "setting": "T. 시간분할",
            "n_test": n_test,
            "pos_rate": rate_t,
            "n_rep_valid": np.nan,
            "n_ge": np.nan,
            "p_phipson_smyth": np.nan,
            "ece_ratio": np.nan,
            "share_rel": np.nan,
            **obs,
        },
    ]

    log(
        f"{'설정':30s} {'검증n':>6s} {'양성률':>7s} {'AUROC':>7s} {'ECE':>8s} "
        f"{'기울기':>8s} {'배율':>6s} {'기여율':>7s} {'초과':>9s} {'p':>8s}"
    )
    log("-" * 100)
    log(
        f"{'정방향 T (참고)':30s} {int(fwd_te.sum()):6d} {float(y[fwd_te].mean()):7.4f} "
        f"{fwd['auroc']:7.4f} {fwd['ece']:8.4f} {fwd['slope']:8.4f} "
        f"{'—':>6s} {'—':>7s} {'—':>9s} {'—':>8s}"
    )
    log(
        f"{'역방향 T':30s} {n_test:6d} {rate_t:7.4f} "
        f"{obs['auroc']:7.4f} {obs['ece']:8.4f} {obs['slope']:8.4f} "
        f"{'—':>6s} {'—':>7s} {'—':>9s} {'—':>8s}"
    )

    configs = [
        ("R1. 무작위 (표준)", False, False),
        ("R2. 무작위 + 검증군 비율맞춤", True, False),
        ("R3. 무작위 + 학습·검증 비율맞춤", True, True),
    ]

    for tag, match_test, match_train in configs:
        E, A, SL, REL, RES_, N = [], [], [], [], [], []
        for i in range(reps):
            r = np.random.default_rng(S.SEED + i)
            idx = r.permutation(len(y))
            ti, ri = idx[:n_test], idx[n_test:]
            if match_test:
                ti = S.subsample_to_rate(y, ti, rate_t, r)
            if match_train:
                ri = S.subsample_to_rate(y, ri, rate_tr, r)
            if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                continue
            if len(ti) < 30 or len(ri) < 100:
                continue
            e = evaluate(y[ri], X[ri], y[ti], X[ti])
            E.append(e["ece"])
            A.append(e["auroc"])
            SL.append(e["slope"])
            REL.append(e["rel"])
            RES_.append(e["res"])
            N.append(len(ti))
        if not E:
            log(f"{tag:30s} 유효 반복 0회 — 건너뜀")
            continue
        E = np.array(E)
        n_ge = int((E >= obs["ece"]).sum())
        p = phipson_smyth(n_ge, len(E))
        ratio = obs["ece"] / float(E.mean())
        d_rel = obs["rel"] - float(np.mean(REL))
        d_res = float(np.mean(RES_)) - obs["res"]  # 손실 방향을 양수로
        share = d_rel / (d_rel + d_res) * 100 if (d_rel + d_res) != 0 else np.nan
        log(
            f"{tag:30s} {int(np.mean(N)):6d} {rate_t:7.4f} "
            f"{float(np.mean(A)):7.4f} {float(E.mean()):8.4f} "
            f"{float(np.mean(SL)):8.4f} {ratio:6.2f} {share:6.1f}% "
            f"{n_ge:4d}/{len(E):<4d} {p:8.4f}"
        )
        rows.append(
            {
                "dataset": name,
                "direction": "역방향",
                "setting": tag,
                "n_test": int(np.mean(N)),
                "pos_rate": rate_t,
                "auroc": float(np.mean(A)),
                "ece": float(E.mean()),
                "slope": float(np.mean(SL)),
                "rel": float(np.mean(REL)),
                "res": float(np.mean(RES_)),
                "unc": np.nan,
                "n_rep_valid": len(E),
                "n_ge": n_ge,
                "p_phipson_smyth": p,
                "ece_ratio": ratio,
                "share_rel": share,
            }
        )
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()

    log("역방향 시간분할 — 미래로 학습하고 과거로 검증 (§12.2)")
    log(f"대조군 반복 {a.reps}회, 시드 {S.SEED}")
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.reps)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "reverse_split.csv"), index=False, encoding="utf-8")

    log("=" * 100)
    log("판정")
    log("=" * 100)
    log("[읽는 법] 역방향에서도 ECE 배율이 1 을 넘고 보정 기여율이 과반이면,")
    log("          보정 붕괴는 '최근 라벨이 미성숙해서' 생기는 것이 아니다 —")
    log("          역방향의 검증군은 수십 년 관찰된 성숙한 라벨이기 때문이다.")
    log("          반대로 역방향에서 붕괴가 사라지면 §12 의 검열 통제가 불충분했다는")
    log("          뜻이고, 그 경우 본문의 인과 서술을 약하게 고쳐야 한다.")
    log("")
    for d in out.dataset.unique():
        g = out[(out.dataset == d) & (out.direction == "역방향")]
        t = g[g.setting.str.startswith("T")]
        for b in ("R1", "R2", "R3"):
            rr = g[g.setting.str.startswith(b)]
            if not len(t) or not len(rr):
                continue
            log(
                f"  {d} 역방향 {b}: ECE {float(t.ece.iloc[0]):.4f} 대 "
                f"{float(rr.ece.iloc[0]):.4f} → 배율 {float(rr.ece_ratio.iloc[0]):.2f}, "
                f"기여율 {float(rr.share_rel.iloc[0]):.1f}%, "
                f"p {float(rr.p_phipson_smyth.iloc[0]):.4f}"
            )
    log("")
    log(f"저장: {os.path.join(RES, 'reverse_split.csv')}")
    log(f"      {os.path.join(RES, 'reverse_split_summary.txt')}")
    log.save(Path(RES) / "reverse_split_summary.txt")


if __name__ == "__main__":
    main()
