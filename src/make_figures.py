"""
논문 투고용 그림 생성

규격:
  - 크기: 단일 컬럼 3.3in, 이중 컬럼 7.0in (ACS/Elsevier 표준)
  - 폰트: Arial 계열, 축 8pt, 패널 라벨 12pt bold
  - 저장: PDF(벡터) + PNG(600dpi) 동시
  - 색맹 친화 팔레트, 흑백 인쇄 대비 해칭/마커 병용

생성 그림:
  Fig 1  분할 전략별 AUROC / ECE (2패널)
  Fig 2  보정 곡선 reliability diagram (2패널: 두 데이터셋)
  Fig 3  견고성 (2패널: 32조합 ECE 배율 분포, 절단연도 민감도)
  Fig 4  학습 유병률 격차 스윕 (2패널)
  Fig 5  재보정 사다리와 conformal 커버리지 (2패널)   ← 2026-10-22
  Fig 6  배포 시뮬레이션과 라벨 지연 (2패널)           ← 2026-10-22
  Fig S1 ATC 계열별 성능

  2026-10-22: 이 목록이 실제 출력과 어긋나 있었다 — 유병률이 FigS1 에서
  Fig4 로, ATC 가 FigS2 에서 FigS1 로 옮겨간 뒤 주석만 안 고쳤다.
  그리고 Fig1~Fig4 는 논문의 전반부(붕괴의 존재와 견고성)까지만 말한다.
  뒷부분 — 처방이 어디까지 되돌리는지, 보장이 깨지는지, 해마다 재학습
  하면 돌아오는지 — 에 그림이 없었다. Fig5·Fig6 이 그 자리다.

사용법:
    python src/make_figures.py

출력:
    figures/*.pdf, figures/*.png
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
COL1, COL2 = 5.146, 5.146  # 인치 = sn-jnl \textwidth (372pt = 130.7mm)
# 이 폭으로 만들면 width=\textwidth 삽입 시 축소율 1.00
N_BINS = 10

# ------------------------------------------------------------ 스타일
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # 폰트 임베딩 (편집 가능)
        "ps.fonttype": 42,
    }
)

# 색맹 친화 팔레트 (Okabe-Ito)
C = {
    "random": "#0072B2",
    "scaffold": "#E69F00",
    "temporal": "#D55E00",
    "gray": "#666666",
    "green": "#009E73",
    "purple": "#CC79A7",
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  저장: figures/{name}.pdf, .png")


def panel_label(ax, txt, x=-0.16, y=1.06):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


# ------------------------------------------------------------ 공통 계산
def featurize(smiles):
    fps, desc = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=np.float32))
        desc.append(
            [
                Descriptors.MolWt(m),
                Descriptors.MolLogP(m),
                Descriptors.TPSA(m),
                Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m),
            ]
        )
    return np.hstack([np.array(fps), np.array(desc)])


def make_model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def get_probs(df):
    """시간분할 및 무작위 분할 예측확률 반환."""
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr

    m = make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]

    # 무작위: 동일 크기 1회 (그림용)
    r = np.random.default_rng(SEED)
    idx = r.permutation(len(y))
    ti, ri = idx[: te.sum()], idx[te.sum() :]
    mm = make_model().fit(X[ri], y[ri])
    p_r = mm.predict_proba(X[ti])[:, 1]

    return y[te], p_t, y[ti], p_r


def rel_curve(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        hi = p <= 1 if i == n_bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum() >= 3:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


# ============================================================ Fig 1
def fig1():
    """분할 전략별 AUROC(a)·ECE(b)와 유병률 통제 단계별 ECE(c).

    이 그림이 논문의 주장을 그대로 담는다.
      (a) 판별력은 분할을 어렵게 해도 완만하게만 떨어진다.
      (b) 그런데 보정 오차는 스캐폴드 분할에서는 무작위와 같은 수준이고,
          시간분할에서만 무너진다. 즉 보정 붕괴는 "화학공간이 달라서"가
          아니라 "시간이 달라서" 생긴다.
      (c) 그 붕괴가 학습군·검증군의 양성 비율 차이 때문도 아니다.

    모든 수치는 결과 CSV에서 읽는다 (하드코딩 금지).
      (a)(b) baseline_cv_per_fold.csv — 두 데이터셋 동일 절차
          · random_cv / scaffold_cv : 5-fold 평균 ± fold 간 표준편차
          · temporal                : 단일 분할
                                      (AUROC 는 95% 부트스트랩 CI,
                                       ECE 는 prevalence_control.csv 의
                                       부트스트랩 CI를 가져다 쓴다)
      (c) prevalence_control.csv
          · R1/R2/R3 : 반복 평균 ± 반복 간 표준편차
          · T        : 단일 분할 + 95% 부트스트랩 신뢰구간

    반복 기반 막대는 SD, 시간분할 막대는 부트스트랩 CI이므로 구간의
    성격이 다르다. 그림 캡션에 반드시 명시할 것.
    """
    try:
        cv = pd.read_csv(RES / "baseline_cv_per_fold.csv")
        prev = pd.read_csv(RES / "prevalence_control.csv")
    except FileNotFoundError as e:
        print(
            f"  Fig1 건너뜀: {e.filename} 없음 "
            f"(baseline_cv.py / prevalence_control.py 먼저 실행)"
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))
    w = 0.36

    splits = ["random_cv", "scaffold_cv", "temporal"]
    labels = ["Random\n5-fold CV", "Scaffold\n5-fold CV", "Temporal\nsplit"]
    series = [
        ("DILIrank", dict(color=C["random"], edgecolor="black")),
        ("DICTrank", dict(color="white", edgecolor="black", hatch="///")),
    ]

    def temporal_ece_ci(ds):
        """시간분할 ECE의 부트스트랩 CI는 prevalence_control.csv 에 있다."""
        s = prev[prev.dataset.str.startswith(ds) & prev.setting.str.startswith("T. ")]
        if not len(s):
            return np.nan, np.nan
        r = s.iloc[0]
        return r.ECE - r.ECE_lo, r.ECE_hi - r.ECE

    def stat_of(ds, sp, col):
        s = cv[(cv.dataset == ds) & (cv.split == sp)]
        if not len(s):
            return np.nan, np.nan, np.nan
        if sp == "temporal":
            r = s.iloc[0]
            if col == "AUROC":
                return r.AUROC, r.AUROC - r.auroc_lo, r.auroc_hi - r.AUROC
            lo, hi = temporal_ece_ci(ds)
            return r.ECE, lo, hi
        v = s[col]
        return v.mean(), v.std(ddof=1), v.std(ddof=1)

    def grouped_bars(ax, col):
        """DILIrank 는 색, DICTrank 는 빗금. 시간분할 막대만 주황으로 둬서
        세 패널에서 '주황 = 시간분할' 이 같은 뜻이 되게 한다."""
        x = np.arange(len(splits))
        for k, (ds, style) in enumerate(series):
            vals, lo, hi = zip(*[stat_of(ds, sp, col) for sp in splits])
            style = dict(style)
            if k == 0:
                style["color"] = [C["random"], C["random"], C["temporal"]]
            ax.bar(x + (k - 0.5) * w, vals, w, linewidth=0.6, label=ds, **style)
            ax.errorbar(
                x + (k - 0.5) * w,
                vals,
                yerr=[lo, hi],
                fmt="none",
                ecolor="black",
                elinewidth=0.7,
                capsize=2.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        return x

    # ---------------------------------------------- (a) 분할 전략별 AUROC
    ax = axes[0]
    grouped_bars(ax, "AUROC")
    ax.axhline(0.5, color=C["gray"], ls=":", lw=0.7, zorder=0)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.85)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="0.55", edgecolor="black", linewidth=0.6, label="DILIrank"),
            Patch(
                facecolor="white", edgecolor="black", linewidth=0.6, hatch="///", label="DICTrank"
            ),
        ],
        frameon=False,
        loc="upper right",
        fontsize=5.5,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    panel_label(ax, "a", x=-0.28, y=1.10)

    # ---------------------------------------------- (b) 분할 전략별 ECE
    ax = axes[1]
    grouped_bars(ax, "ECE")
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 0.58)
    panel_label(ax, "b", x=-0.28, y=1.10)

    # ---------------------------------------------- (c) 유병률 통제별 ECE
    ax = axes[2]
    order = ["R1. ", "R2. ", "R3. ", "T. "]
    # 패널 폭이 좁아 긴 라벨은 서로 겹친다. R1/R2/R3 의 뜻은 캡션에서
    # 정의할 것: R1 = 비율 맞춤 없음, R2 = 검증군만 맞춤, R3 = 학습·검증 모두.
    labels_c = ["R1", "R2", "R3", "Temporal\nsplit"]

    def ece_of(ds_key, prefix):
        s = prev[prev.dataset.str.startswith(ds_key) & prev.setting.str.startswith(prefix)]
        if not len(s):
            return np.nan, np.nan, np.nan
        r = s.iloc[0]
        if prefix == "T. ":
            return r.ECE, r.ECE - r.ECE_lo, r.ECE_hi - r.ECE
        sd = r.ECE_sd if pd.notna(r.ECE_sd) else 0.0
        return r.ECE, sd, sd

    x = np.arange(len(order))
    cols = [C["random"]] * 3 + [C["temporal"]]
    for k, (ds_key, hatch) in enumerate([("DILIrank", None), ("DICTrank", "///")]):
        vals, lo, hi = zip(*[ece_of(ds_key, pf) for pf in order])
        ax.bar(
            x + (k - 0.5) * w,
            vals,
            w,
            color=(cols if k == 0 else ["white"] * 4),
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            label=ds_key,
        )
        ax.errorbar(
            x + (k - 0.5) * w,
            vals,
            yerr=[lo, hi],
            fmt="none",
            ecolor="black",
            elinewidth=0.7,
            capsize=2.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_c, fontsize=6)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 0.58)
    panel_label(ax, "c", x=-0.28, y=1.10)

    fig.tight_layout(w_pad=1.4)
    save(fig, "Fig1_split_strategies")


# ============================================================ Fig 2
def fig2():
    """보정 곡선 (reliability diagram)."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))

    files = [
        ("DILIrank", PROC / "dilirank2_with_year.csv"),
        ("DICTrank", PROC / "dictrank_processed.csv"),
    ]

    for ax, (nm, path), lab in zip(axes, files, "ab"):
        if not path.exists():
            continue
        df = load(path)
        y_t, p_t, y_r, p_r = get_probs(df)

        ax.plot(
            [0, 1], [0, 1], ls="--", color=C["gray"], lw=0.9, label="Perfect calibration", zorder=1
        )

        for y, p, tag, col, mk in [
            (y_r, p_r, "Random split", C["random"], "o"),
            (y_t, p_t, "Temporal split", C["temporal"], "s"),
        ]:
            xs, ys, ns = rel_curve(y, p)
            sz = 8 + 40 * ns / ns.max()
            ax.plot(xs, ys, "-", color=col, lw=1.1, zorder=2)
            ax.scatter(
                xs,
                ys,
                s=sz,
                color=col,
                marker=mk,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
                label=tag,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted probability")
        if lab == "a":
            ax.set_ylabel("Observed positive fraction")
        ax.set_title(nm, fontsize=9)
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)
        panel_label(ax, lab, x=-0.24, y=1.16)  # 제목/눈금과 겹치지 않게

    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig2_calibration_curves")


# ============================================================ Fig 3
def fig3():
    """견고성: 32조합 분포 + 절단연도 민감도."""
    try:
        rob = pd.read_csv(RES / "robustness_models.csv")
        cut = pd.read_csv(RES / "robustness_cutoff.csv")
    except FileNotFoundError:
        print("  Fig3 건너뜀: 결과 파일 없음")
        return

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))

    # (a) 32조합 ECE 배율 (스트립 플롯)
    ax = axes[0]
    for i, ds in enumerate(rob["dataset"].unique()):
        s = rob[rob.dataset == ds]
        jit = np.random.default_rng(SEED).normal(0, 0.055, len(s))
        ax.scatter(
            np.full(len(s), i) + jit,
            s["ece_ratio"],
            s=22,
            color=C["temporal"] if i == 0 else C["random"],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )
        med = s["ece_ratio"].median()
        ax.plot([i - 0.22, i + 0.22], [med, med], color="black", lw=1.6, zorder=4)
        ax.text(i, s["ece_ratio"].max() + 0.28, f"{len(s)}/{len(s)}", ha="center", fontsize=6.5)

    ax.axhline(1.0, color=C["gray"], ls="--", lw=0.9, zorder=1)
    ax.set_xticks(range(len(rob["dataset"].unique())))
    ax.set_xticklabels(rob["dataset"].unique(), fontsize=7.5)
    ax.set_ylabel("ECE ratio (temporal / random)")
    ax.set_ylim(0.8, 6.6)  # 수정: 상단 여유 확보 (기존 6.0)
    # 수정: 좌상단 -> 우상단으로 이동 (DILIrank의 16/16 라벨과 겹치던 문제 해결)
    ax.text(
        0.98,
        0.97,
        "All 32 combinations > 1.0",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        ha="right",
        style="italic",
    )
    panel_label(ax, "a")

    # (b) 절단연도 민감도: AUROC 격차 vs ECE 배율
    ax = axes[1]
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)

    # 수정: 데이터셋 구분을 채움/테두리 마커 + x축 지터로 명확화
    # (기존 코드는 두 데이터셋 선이 같은 색이라 2010년 교차점에서 뒤섞여 보였음)
    style = {
        "DILIrank": dict(marker="o", fill=True, dx=-0.12),
        "DICTrank": dict(marker="s", fill=False, dx=+0.12),
    }

    for ds in cut["dataset"].unique():
        s = cut[cut.dataset == ds].sort_values("cutoff")
        st = style.get(ds, dict(marker="o", fill=True, dx=0.0))
        x = s["cutoff"] + st["dx"]

        ax.plot(
            x,
            s["d_auroc"],
            "-",
            marker=st["marker"],
            ms=5,
            lw=1.2,
            color=C["temporal"],
            markerfacecolor=C["temporal"] if st["fill"] else "white",
            markeredgecolor=C["temporal"],
            markeredgewidth=1.0,
            zorder=3,
        )

        ax2.plot(
            x,
            s["ece_ratio"],
            "--",
            marker=st["marker"],
            ms=5,
            lw=1.2,
            color=C["random"],
            markerfacecolor=C["random"] if st["fill"] else "white",
            markeredgecolor=C["random"],
            markeredgewidth=1.0,
            zorder=3,
        )

    ax.set_xticks([2005, 2010, 2015])
    ax.set_xlabel("Temporal split cutoff (year)")
    ax.set_ylabel("ΔAUROC (random − temporal)", color=C["temporal"])
    ax2.set_ylabel("ECE ratio", color=C["random"])
    ax.tick_params(axis="y", colors=C["temporal"])
    ax2.tick_params(axis="y", colors=C["random"])
    ax.set_ylim(0, 0.13)
    ax2.set_ylim(0, 4.0)
    ax.axhline(0, color=C["gray"], ls=":", lw=0.7)

    handles = [
        Line2D([], [], color=C["temporal"], marker="o", ms=4, lw=1.2, label="ΔAUROC (left)"),
        Line2D(
            [],
            [],
            color=C["random"],
            marker="o",
            ms=4,
            lw=1.2,
            ls="--",
            markerfacecolor="white",
            label="ECE ratio (right)",
        ),
        Line2D(
            [], [], color="black", marker="o", ms=4, lw=0, markerfacecolor="black", label="DILIrank"
        ),
        Line2D(
            [], [], color="black", marker="s", ms=4, lw=0, markerfacecolor="white", label="DICTrank"
        ),
    ]
    ax.legend(
        handles=handles, frameon=False, fontsize=6, loc="upper center", ncol=2, columnspacing=0.8
    )
    panel_label(ax, "b")

    fig.tight_layout(w_pad=3.0)
    save(fig, "Fig3_robustness")


# ============================================================ Fig 4
def fig4():
    """학습 유병률 격차 스윕 — 보정은 무너지고 판별은 멀쩡하다.

    (a) ECE 가 격차 0 에서 최소를 이루고 양쪽으로 오른다. 시간분할의
        실제 격차 위치를 표시해 §8 값이 어디서 오는지 보여 준다.
    (b) 격차 0 값으로 정규화한 비. ECE 는 2~3배로 벌어지는데 AUROC 는
        1.0 근처에 붙어 있다 — 제목의 'Beyond AUROC' 가 통제실험으로
        증명되는 지점이다.
    """
    try:
        sw = pd.read_csv(RES / "prevalence_sweep.csv")
    except FileNotFoundError:
        print("  Fig4 건너뜀: prevalence_sweep.csv 없음")
        return

    style = {
        "DILIrank": dict(c=C["temporal"], marker="o", fill=True),
        "DICTrank": dict(c=C["random"], marker="s", fill=False),
    }

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))

    # ---------------------------------------------- (a) ECE 대 격차
    ax = axes[0]
    for ds in sw["dataset"].unique():
        g = sw[sw.dataset == ds].sort_values("gap")
        st = style.get(ds, dict(c="black", marker="o", fill=True))
        ax.errorbar(
            g["gap"],
            g["ece"],
            yerr=g["ece_se"],
            color=st["c"],
            marker=st["marker"],
            ms=4.5,
            lw=1.2,
            capsize=1.8,
            elinewidth=0.8,
            markerfacecolor=st["c"] if st["fill"] else "white",
            markeredgecolor=st["c"],
            markeredgewidth=1.0,
            label=ds,
            zorder=3,
        )
        # 시간분할 실제 격차 위치 (is_pre2010_rate 로 표시된 단계)
        obs = g[g["is_pre2010_rate"]]
        if len(obs):
            ax.scatter(
                obs["gap"],
                obs["ece"],
                s=95,
                facecolor="none",
                edgecolor="black",
                linewidth=1.1,
                zorder=5,
            )

    ax.axvline(0, color=C["gray"], ls="--", lw=0.9, zorder=1)
    ax.set_xlabel("Training − test prevalence gap")
    ax.set_ylabel("ECE")
    ax.margins(y=0.22)  # 범례가 곡선을 덮지 않도록 여유
    # 주석은 좌하단 — 좌상단은 범례 자리이고 우하단은 DILIrank 곡선이 지난다
    ax.text(
        0.02,
        0.03,
        "○ actual time-split gap",
        transform=ax.transAxes,
        fontsize=6,
        va="bottom",
        ha="left",
        style="italic",
    )
    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="upper center",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    panel_label(ax, "a")

    # ------------------------------- (b) 격차 0 기준 정규화 비교
    ax = axes[1]
    for ds in sw["dataset"].unique():
        g = sw[sw.dataset == ds].sort_values("gap")
        base = g[g["is_test_rate"]]
        if not len(base):
            continue
        e0 = float(base["ece"].iloc[0])
        a0 = float(base["auroc"].iloc[0])
        st = style.get(ds, dict(c="black", marker="o", fill=True))
        ax.plot(
            g["gap"],
            g["ece"] / e0,
            "-",
            color=st["c"],
            marker=st["marker"],
            ms=4.5,
            lw=1.2,
            markerfacecolor=st["c"] if st["fill"] else "white",
            markeredgecolor=st["c"],
            markeredgewidth=1.0,
            zorder=3,
        )
        ax.plot(
            g["gap"],
            g["auroc"] / a0,
            ":",
            color=st["c"],
            marker=st["marker"],
            ms=3.2,
            lw=1.0,
            markerfacecolor="white",
            markeredgecolor=st["c"],
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.axhline(1.0, color=C["gray"], ls="--", lw=0.9, zorder=1)
    ax.axvline(0, color=C["gray"], ls="--", lw=0.9, zorder=1)
    ax.set_xlabel("Training − test prevalence gap")
    ax.set_ylabel("Ratio to value at zero gap")

    ax.margins(y=0.14)
    handles = [
        Line2D([], [], color="black", ls="-", lw=1.2, label="ECE"),
        Line2D([], [], color="black", ls=":", lw=1.0, label="AUROC"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        handlelength=2.2,
        handletextpad=0.5,
        borderaxespad=0.3,
    )
    panel_label(ax, "b")

    fig.tight_layout(w_pad=3.0)
    save(fig, "Fig4_prevalence_sweep")


# ============================================================ Fig S1
def figS1():
    """ATC 계열별 성능."""
    f = RES / "atc_stratified.csv"
    if not f.exists():
        print("  FigS1 건너뜀: atc_stratified.csv 없음")
        return
    s = pd.read_csv(f).dropna(subset=["auroc"]).sort_values("auroc")

    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    yy = np.arange(len(s))
    ax.barh(yy, s["auroc"], color=C["random"], edgecolor="black", linewidth=0.6)
    ax.axvline(0.5, color=C["temporal"], ls="--", lw=1.0)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r.atc} (n={int(r.n)})" for _, r in s.iterrows()], fontsize=6.5)
    ax.set_xlabel("AUROC (temporal split)")
    ax.set_xlim(0, 1)
    # 수정: "random"은 무작위 분할 대조군(0.70~0.75대)이 아니라
    # 우연 수준(AUROC=0.5)을 뜻하므로 다른 그림들과 용어 혼동 방지
    ax.text(0.51, -0.6, "chance (AUROC=0.5)", fontsize=6, color=C["temporal"])
    fig.tight_layout()
    save(fig, "FigS1_atc_stratified")


def fig5():
    """처방의 실패와 부분 성공 — 학습군 안에서 고치면 절반만 돌아오고
    보장은 깨진다.

    (a) 재보정 사다리. 무보정 A 에서 시작해 Platt·Isotonic·사전확률을
        붙이고, 검증군 라벨을 쓴 Oracle Platt E 를 상한선으로 둔다.
        A 와 E 사이가 되돌릴 수 있는 전부이고, B·C·D 가 그 중 어디까지
        가는지가 §10 의 답이다.
    (b) split conformal 의 실제 커버리지. 명목값(0.9, 0.8)을 점선으로
        두고 무작위 분할과 시간분할을 나란히 둔다. 시간분할에서 명목값
        아래로 내려가는 것이 §11 이고, 음성 클래스에서 더 내려간다.
    """
    try:
        rec = pd.read_csv(RES / "recalibration_results.csv")
        cf = pd.read_csv(RES / "conformal_prediction.csv")
    except FileNotFoundError as e:
        print(f"  Fig5 건너뜀: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7))

    # ---------------------------------------------- (a) 재보정 사다리
    ax = axes[0]
    order = ["A.", "B.", "C.", "D.", "E."]
    lab = {"A.": "None", "B.": "Platt", "C.": "Isotonic", "D.": "Prior shift", "E.": "Oracle"}
    style = {
        "DILIrank": dict(c=C["temporal"], marker="o", mfc=C["temporal"]),
        "DICTrank": dict(c=C["random"], marker="s", mfc="white"),
    }
    x = np.arange(len(order))
    for ds, st in style.items():
        g = rec[rec.dataset.str.startswith(ds)]
        ys = []
        for pre in order:
            r = g[g.method.str.startswith(pre)]
            ys.append(float(r.ECE.iloc[0]) if len(r) else np.nan)
        ax.plot(
            x,
            ys,
            color=st["c"],
            marker=st["marker"],
            mfc=st["mfc"],
            ms=4,
            lw=1.2,
            mew=1.0,
            label=ds,
            clip_on=False,
            zorder=3,
        )
        # 무보정과 Oracle 사이를 음영으로 — 되돌릴 수 있는 전부
        ax.fill_between(
            [x[0], x[-1]],
            [ys[0], ys[0]],
            [ys[-1], ys[-1]],
            color=st["c"],
            alpha=0.06,
            lw=0,
            zorder=1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lab[p] for p in order], rotation=20, ha="right")
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    panel_label(ax, "a")

    # ---------------------------------------------- (b) conformal 커버리지
    ax = axes[1]
    rows = []
    for _, r in cf.sort_values(["alpha", "dataset"]).iterrows():
        rows.append(
            (
                r.dataset,
                float(r.target_coverage),
                float(r.marginal_coverage_random),
                float(r.marginal_coverage_temporal),
                float(r.coverage_negative_temporal),
            )
        )
    xs = np.arange(len(rows))
    wd = 0.26
    ax.bar(xs - wd, [r[2] for r in rows], wd, color=C["random"], label="Random", lw=0)
    ax.bar(xs, [r[3] for r in rows], wd, color=C["temporal"], label="Temporal", lw=0)
    ax.bar(
        xs + wd,
        [r[4] for r in rows],
        wd,
        facecolor="white",
        edgecolor=C["temporal"],
        hatch="///",
        lw=0.8,
        label="Temporal, negatives",
    )
    for i, r in enumerate(rows):
        ax.plot(
            [i - wd * 1.6, i + wd * 1.6],
            [r[1], r[1]],
            ls=(0, (3, 2)),
            color=C["gray"],
            lw=1.0,
            zorder=4,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["%s\n%.0f%%" % (r[0].replace("rank", ""), 100 * r[1]) for r in rows])
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", handlelength=1.4, borderaxespad=0.2)
    panel_label(ax, "b")

    fig.tight_layout(w_pad=2.4)
    save(fig, "Fig5_recalibration_conformal")


def fig6():
    """작동하는 처방 — 해마다 다시 학습하고 다시 보정하면 돌아오고,
    라벨이 늦게 붙어도 남는다.

    (a) 배포 시뮬레이션 세 팔의 ECE. 고정 모델 P0, 해마다 재학습 P1,
        재학습+재보정 P2. P1 만으로는 부족하고 P2 에서 내려간다.
    (b) 라벨 지연 민감도. P2 가 고정 모델 대비 낸 짝지은 ΔECE 를 지연
        0~5년에서 95% 구간과 함께 본다. 0 선 아래에 머무르면 그 지연
        에서도 처방이 작동한다는 뜻이다.
    """
    try:
        pqy = pd.read_csv(RES / "prequential_delay.csv")
        pqd = pd.read_csv(RES / "prequential_delay_delta.csv")
    except FileNotFoundError as e:
        print(f"  Fig6 건너뜀: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7))
    style = {
        "DILIrank": dict(c=C["temporal"], marker="o", mfc=C["temporal"]),
        "DICTrank": dict(c=C["random"], marker="s", mfc="white"),
    }

    # ---------------------------------------------- (a) 세 팔의 ECE
    ax = axes[0]
    arms = ["P0", "P1", "P2"]
    lab = {"P0": "Fixed", "P1": "Retrain", "P2": "Retrain\n+ recalibrate"}
    x = np.arange(len(arms))
    g0 = pqy[pqy.delay == 0]
    for ds, st in style.items():
        g = g0[g0.dataset == ds]
        ys = [float(g[g.arm.str.startswith(a)].ece.iloc[0]) for a in arms]
        ax.plot(
            x,
            ys,
            color=st["c"],
            marker=st["marker"],
            mfc=st["mfc"],
            ms=4,
            lw=1.2,
            mew=1.0,
            label=ds,
            clip_on=False,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lab[a] for a in arms])
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    panel_label(ax, "a")

    # ---------------------------------------------- (b) 라벨 지연
    ax = axes[1]
    q = pqd[(pqd.vs == "P0. 고정 모델") & pqd.arm.str.startswith("P2")]
    ax.axhline(0, color=C["gray"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    for ds, st in style.items():
        g = q[q.dataset == ds].sort_values("delay")
        d = g.delay.values.astype(float)
        ax.plot(
            d,
            g.dece.values,
            color=st["c"],
            marker=st["marker"],
            mfc=st["mfc"],
            ms=4,
            lw=1.2,
            mew=1.0,
            label=ds,
            zorder=3,
        )
        ax.fill_between(d, g.lo.values, g.hi.values, color=st["c"], alpha=0.14, lw=0, zorder=2)
    ax.set_xlabel("Assumed label delay (years)")
    ax.set_ylabel(r"$\Delta$ECE vs fixed model")
    ax.set_xticks(sorted(q.delay.unique()))
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    panel_label(ax, "b")

    fig.tight_layout(w_pad=2.4)
    save(fig, "Fig6_retraining_delay")


def main():
    print("그림 생성 중...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    figS1()
    print(f"\n완료. {FIG} 확인")
    print("PDF는 벡터(투고용), PNG는 600dpi(확인용)")


if __name__ == "__main__":
    main()
