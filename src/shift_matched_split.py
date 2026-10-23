"""§13.5 — 이동량을 맞춘 '시간이 아닌' 분할에서도 보정이 무너지는가.

왜 필요한가
----------
§13.1 은 시간분할이 구조 수준의 분포 이동을 만든다는 것을 직접 측정했다
(시대 분류기 AUC 0.6835 / 0.7917). §13.3 은 그 이동의 정도가 보정 붕괴의
정도를 설명하지 못한다는 음성 결과를 냈다. 그래서 "그럼 무엇이 붕괴를
일으키는가" 가 열린 채로 남는다. 심사자가 가장 먼저 찌를 자리다.

이 실험이 그 틈을 메운다. 시간을 쓰지 않고 같은 크기의 분포 이동을 만드는
분할을 몇 가지 구성해, 같은 눈금(대조군 대비 ECE 배율)으로 붕괴 정도를 잰다.

  - 이동량이 비슷한데 붕괴도 비슷하면 → 원인은 분포 이동 자체다.
    시간은 이동을 만드는 여러 축 중 하나일 뿐이라는 뜻이고, 그러면 결론이
    시간분할을 넘어 일반화된다(논문의 주장이 넓어진다).
  - 이동량이 비슷한데 시간분할만 더 심하게 무너지면 → 시간축에 고유한 무언가가
    있다는 뜻이다(라벨 관행 변화, 적응증 변화 등). 그 경우 §13.3 의 음성 결과와
    합쳐 "이동량으로 환원되지 않는다" 를 더 강하게 쓸 수 있다.

어느 쪽이 나와도 보고할 값이 있다. 그래서 결과를 보기 전에 설계를 고정한다.

분할 구성 (검증군 크기는 모두 시간분할과 같게 맞춘다)
---------------------------------------------------
  T   시간분할              승인연도 2010년 기준 (기준점)
  X1  분자량 상위           MolWt 가 큰 쪽 n_test 개를 검증군으로
  X2  logP 상위             MolLogP 가 큰 쪽 n_test 개
  X3  지문 군집 홀드아웃      ECFP4 k-means 군집을 통째로 빼서 n_test 근처까지
  R   무작위                이동이 거의 없는 기준점

이동량 측정은 §13.1 과 같은 방식이다 — "이 분자가 검증군 쪽인가" 를 같은 표현·
같은 모델로 맞히는 5-fold CV AUC. T 에 적용하면 시대 분류기와 같은 값이 나온다.

각 분할마다 그 분할 자신의 검증군 유병률에 맞춘 무작위 대조군(§8 의 R2 와
같은 설계)을 붙인다. 그래야 분할 간 ECE 배율을 유병률 차이에 오염되지 않은
상태로 비교할 수 있다.

사용법
------
    python src/shift_matched_split.py                # 대조군 50회
    python src/shift_matched_split.py --reps 200     # 본 실행
    python src/shift_matched_split.py --reps 5       # 5분 내 점검

출력
----
    results/shift_matched_split.csv
    results/shift_matched_split_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
N_FOLD = 5
log = S.Log()


def phipson_smyth(n_ge, n_valid):
    return (1.0 + n_ge) / (1.0 + n_valid)


def shift_auc(X, is_test):
    """'검증군 쪽인가' 를 맞히는 5-fold CV AUC. §13.1 과 같은 방식."""
    lab = is_test.astype(int)
    if len(np.unique(lab)) < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=S.SEED)
    oof = np.zeros(len(lab))
    for tr, va in skf.split(X, lab):
        m = S.make_model().fit(X[tr], lab[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(lab, oof))


def evaluate(X, y, tr, te):
    m = S.make_model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    d = S.decompose(y[te], p)
    return {
        "auroc": roc_auc_score(y[te], p),
        "ece": S.ece(y[te], p),
        "slope": S.calib_slope(y[te], p),
        "rel": d["reliability"],
        "res": d["resolution"],
    }


def cluster_holdout(X, n_test, k):
    """ECFP k-means 군집을 통째로 빼서 n_test 에 가장 가깝게 채운다."""
    km = KMeans(n_clusters=k, random_state=S.SEED, n_init=10).fit(X)
    lab = km.labels_
    sizes = sorted(((int((lab == c).sum()), int(c)) for c in range(k)), reverse=True)
    chosen, tot = [], 0
    for sz, c in sizes:  # 큰 군집부터 넣되 n_test 를 넘지 않게
        if tot + sz <= n_test:
            chosen.append(c)
            tot += sz
    for sz, c in reversed(sizes):  # 아직 부족하면 가장 작은 것부터 보탠다
        # (큰 것부터 보태면 n_test 를 크게 넘긴다)
        if c in chosen:
            continue
        if tot >= n_test:
            break
        chosen.append(c)
        tot += sz
    te = np.isin(lab, chosen)
    return te, tot, k


def controls(X, y, n_test, rate_t, obs, reps):
    """해당 분할의 검증군 유병률에 맞춘 무작위 대조군 (§8 의 R2 설계)."""
    E, A, REL, RES_, N = [], [], [], [], []
    for i in range(reps):
        r = np.random.default_rng(S.SEED + i)
        idx = r.permutation(len(y))
        ti, ri = idx[:n_test], idx[n_test:]
        ti = S.subsample_to_rate(y, ti, rate_t, r)
        if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
            continue
        if len(ti) < 30 or len(ri) < 100:
            continue
        e = evaluate(X, y, ri, ti)
        E.append(e["ece"])
        A.append(e["auroc"])
        REL.append(e["rel"])
        RES_.append(e["res"])
        N.append(len(ti))
    if not E:
        return None
    E = np.array(E)
    n_ge = int((E >= obs["ece"]).sum())
    d_rel = obs["rel"] - float(np.mean(REL))
    d_res = float(np.mean(RES_)) - obs["res"]
    share = d_rel / (d_rel + d_res) * 100 if (d_rel + d_res) != 0 else np.nan
    return {
        "ctrl_n": int(np.mean(N)),
        "ctrl_ece": float(E.mean()),
        "ctrl_ece_sd": float(E.std(ddof=1)) if len(E) > 1 else np.nan,
        "ctrl_auroc": float(np.mean(A)),
        "n_rep_valid": len(E),
        "n_ge": n_ge,
        "p_phipson_smyth": phipson_smyth(n_ge, len(E)),
        "ece_ratio": obs["ece"] / float(E.mean()),
        "share_rel": share,
    }


def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_t, te_t = S.temporal_masks(df)
    n_test = int(te_t.sum())

    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    mw = np.array([Descriptors.MolWt(m) for m in mols])
    lp = np.array([Descriptors.MolLogP(m) for m in mols])

    def top_mask(v, n):
        te = np.zeros(len(v), dtype=bool)
        te[np.argsort(-v, kind="stable")[:n]] = True
        return te

    # X3 의 군집 수: 평균 군집 크기가 n_test 의 1/5 정도가 되게 잡는다.
    k = max(5, int(round(len(y) / max(n_test / 5.0, 1.0))))
    te_cl, cl_n, k_used = cluster_holdout(X, n_test, k)

    rng = np.random.default_rng(S.SEED)
    te_rand = np.zeros(len(y), dtype=bool)
    te_rand[rng.permutation(len(y))[:n_test]] = True

    splits = [
        ("T. 시간분할 (기준점)", te_t),
        ("X1. 분자량 상위", top_mask(mw, n_test)),
        ("X2. logP 상위", top_mask(lp, n_test)),
        ("X3. 지문 군집 홀드아웃", te_cl),
        ("R. 무작위 (기준점)", te_rand),
    ]

    log("=" * 112)
    log(
        f"{name}   n={len(y)}  검증군 크기 {n_test} 로 모든 분할을 맞춤  "
        f"(X3 군집 {k_used}개 중 선택분 n={cl_n})"
    )
    log("=" * 112)
    log(
        f"{'분할':26s} {'검증n':>6s} {'학습양성률':>9s} {'검증양성률':>9s} "
        f"{'이동AUC':>8s} {'AUROC':>7s} {'ECE':>8s} {'대조ECE':>8s} "
        f"{'배율':>6s} {'기여율':>7s} {'p':>8s}"
    )
    log("-" * 112)

    rows = []
    for tag, te in splits:
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            log(f"{tag:26s} 한쪽 클래스만 있어 건너뜀")
            continue
        rate_t = float(y[te].mean())
        rate_tr = float(y[tr].mean())
        sa = shift_auc(X, te)
        obs = evaluate(X, y, tr, te)
        c = controls(X, y, int(te.sum()), rate_t, obs, reps)
        if c is None:
            log(f"{tag:26s} 유효 대조군 0회 — 건너뜀")
            continue
        log(
            f"{tag:26s} {int(te.sum()):6d} {rate_tr:9.4f} {rate_t:9.4f} "
            f"{sa:8.4f} {obs['auroc']:7.4f} {obs['ece']:8.4f} "
            f"{c['ctrl_ece']:8.4f} {c['ece_ratio']:6.2f} {c['share_rel']:6.1f}% "
            f"{c['p_phipson_smyth']:8.4f}"
        )
        rows.append(
            {
                "dataset": name,
                "split": tag,
                "n_test": int(te.sum()),
                "pos_rate_train": rate_tr,
                "pos_rate_test": rate_t,
                "shift_auc": sa,
                **obs,
                **c,
            }
        )
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()

    log("이동량 일치 비시간 분할 — 시간을 쓰지 않고 같은 크기의 이동을 만들면? (§13.5)")
    log(f"대조군 반복 {a.reps}회 (각 분할의 검증군 유병률에 맞춤), 시드 {S.SEED}")
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.reps)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "shift_matched_split.csv"), index=False, encoding="utf-8")

    log("=" * 112)
    log("판정")
    log("=" * 112)
    log("[읽는 법] '이동AUC' 는 §13.1 의 시대 분류기와 같은 눈금이다(0.5 = 이동 없음).")
    log("          '배율' 은 각 분할의 검증군 유병률에 맞춘 대조군 대비 ECE 배율이므로")
    log("          분할 간 비교가 가능하다. 두 열을 짝으로 읽을 것.")
    log("")
    for d in out.dataset.unique():
        g = out[out.dataset == d]
        t = g[g.split.str.startswith("T")]
        if not len(t):
            continue
        t = t.iloc[0]
        log(
            f"  {d}  시간분할: 이동AUC {t.shift_auc:.4f}, 배율 {t.ece_ratio:.2f}, "
            f"기여율 {t.share_rel:.1f}%"
        )
        for _, r in g[~g.split.str.startswith("T")].iterrows():
            rel = (
                "비슷"
                if abs(r.ece_ratio - t.ece_ratio) < 0.15
                else ("더 큼" if r.ece_ratio > t.ece_ratio else "더 작음")
            )
            log(
                f"      {r.split:26s} 이동AUC {r.shift_auc:.4f} "
                f"(차 {r.shift_auc - t.shift_auc:+.4f}) / 배율 {r.ece_ratio:.2f} "
                f"({rel}) / 기여율 {r.share_rel:.1f}%"
            )
    log("")
    log("이동AUC 가 시간분할과 비슷한 비시간 분할에서 배율도 비슷하면, 보정 붕괴는")
    log("시간이 아니라 분포 이동 자체의 결과다. 시간분할만 유독 크면 시간축에 고유한")
    log("요인(라벨 관행·적응증 변화 등)이 남아 있다는 뜻이다. 둘 중 어느 쪽이든")
    log("본문 §13 의 인과 서술을 그 결과에 맞춰 고칠 것.")
    log("")
    log(f"저장: {os.path.join(RES, 'shift_matched_split.csv')}")
    log(f"      {os.path.join(RES, 'shift_matched_split_summary.txt')}")
    log.save(Path(RES) / "shift_matched_split_summary.txt")


if __name__ == "__main__":
    main()
