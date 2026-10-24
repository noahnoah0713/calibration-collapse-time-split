"""
논문 투고용 그림 생성

규격:
  - 크기: 폭 4.685in = sn-jnl 단일단 textwidth (119mm). sn-jnl.cls 의
    geometry 가 text={119mm,194.25mm} 이다. 전에 130.7mm 로 적혀 있었는데
    그 값은 어디서 온 것인지 근거가 없고, 그대로 두면 width=\textwidth 로
    넣을 때 0.91 로 줄어들어 7pt 글자가 6.4pt 로 실렸다 (2026-10-24
    sn-jnl.cls 원문에서 확인). bbox_inches="tight" 가 여백을 조금 깎으므로
    실제 축척은 1.0 을 조금 넘는다 — 글자가 작아지는 쪽이 아니라 커지는
    쪽이라 안전하다. BMC 는 production 에서 85mm 또는 170mm 로 다시
    앉히는데, 119mm 로 그려 두면 170mm 에서는 글자가 커지고 85mm 에서도
    5pt 밑으로 내려가지 않는다.
  - 폰트: Arial 계열, 축 7pt, 눈금 6.5pt, 패널 라벨 8pt bold
  - 저장: PDF(벡터) + PNG(600dpi) 동시
  - Nature 계열 관행: 눈금선을 깔지 않고 축선은 왼쪽·아래만, 눈금은
    바깥으로. 빗금은 쓰지 않는다 (인쇄에서 뭉치고 화면에서 시끄럽다).

색이 뜻하는 것 — 이 규칙을 그림마다 바꾸지 않는다
--------------------------------------------------
  파랑  #0072B2 = DILIrank
  주황  #D55E00 = DICTrank
  회색  #666666 = 무작위 분할(대조군), 기준선, 우연 수준

  2026-10-23 이전에는 같은 두 색이 Fig1~3 에서 '시간분할 / 무작위분할',
  Fig4~7 에서 'DILIrank / DICTrank' 를 뜻했다. Fig1 에서 '주황 = 시간
  분할' 을 배운 독자가 Fig4 부터 속는다. 그래서 색은 언제나 데이터셋이고,
  분할 종류는 위치·패널·표식 모양으로 구분한다. 한 데이터셋 안에서
  분할을 대비할 때만(Fig2 등) 무작위 쪽을 회색으로 둔다 — 회색은 두
  데이터셋 색 어느 쪽과도 겹치지 않는다.

생성 그림:
  Fig 1  분할 전략별 AUROC / ECE (2패널)
  Fig 2  보정 곡선 reliability diagram (2패널: 두 데이터셋)
  Fig 3  학습 유병률 격차 스윕 (2패널)
  Fig 4  이동량을 맞춘 비시간 분할 (2패널)
  Fig 5  재보정 사다리와 conformal 커버리지 (2패널)   ← 2026-10-22
  Fig 6  배포 시뮬레이션과 라벨 지연 (2패널)           ← 2026-10-22
  Fig 7  견고성 (3패널: 32조합 ECE 배율, 절단연도 민감도 둘)
  Fig S1 ATC 계열별 성능 (Additional file 1)

  2026-10-24: 그림 번호를 본문 인용 순서에 맞춰 다시 매겼다. 학술지는
  "본문에 처음 언급되는 순서로 번호 매기고 그 순서로 올리라" 고 요구한다.
  옛 Fig4 -> Fig3, 옛 Fig7 -> Fig4, 옛 Fig3 -> Fig7 이다.

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

import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
# 출력 폴더. 기본은 figures/ 지만 FIG_OUT 으로 바꿀 수 있다.
# 왜 — Arial 이 없는 기계에서 시험 삼아 돌리면 대체 폰트로 그려진
# 그림이 투고용 파일을 덮어쓴다. 실제로 한 번 그랬다 (2026-10-22,
# git show 로 되살렸다). 미리보기는 FIG_OUT 을 주고 돌린다.
FIG = Path(os.environ.get("FIG_OUT", ROOT / "figures"))
FIG.mkdir(parents=True, exist_ok=True)

SEED, CUTOFF = 42, 2010
COL1, COL2 = 4.685, 4.685  # 인치 = sn-jnl 단일단 \textwidth (119mm = 339pt)
# 이 폭으로 만들면 width=\textwidth 삽입 시 축척이 1.0 근처가 된다.
# src/verify14.py 가 만들어진 PDF 의 실제 폭을 재서 확인한다.
N_BINS = 10

# ------------------------------------------------------------ 스타일
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#000000",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
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

# 색맹 친화 팔레트 (Wong B. Points of view: Color blindness.
# Nat Methods 2011;8:441 — Nature 계열 그림이 표준으로 쓰는 8색).
# 두 데이터셋 색 조합은 검증을 거쳤다: 적록색약 dE 21.9, 정상 31.2,
# 배경 대비 3:1 이상.
C = {
    "dili": "#0072B2",  # DILIrank — 파랑
    "dict": "#D55E00",  # DICTrank — 주황
    "ref": "#666666",  # 무작위 분할 대조군·기준선·우연 수준
    "gray": "#666666",
    "accent": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
}
# 데이터셋 이름 -> 색·표식. 그림마다 손으로 고르지 말고 이것만 쓴다.
DS_COLOR = {"DILIrank": C["dili"], "DICTrank": C["dict"]}
DS_MARKER = {"DILIrank": "o", "DICTrank": "s"}


def ds_color(name):
    """'DILIrank 2.0' 처럼 뒤에 말이 붙어도 맞춘다."""
    for k, v in DS_COLOR.items():
        if str(name).startswith(k):
            return v
    return C["gray"]


def ds_marker(name):
    for k, v in DS_MARKER.items():
        if str(name).startswith(k):
            return v
    return "o"


def bare(ax):
    """Nature 계열 축 — 눈금선 없음, 축선은 왼쪽·아래만, 눈금 바깥."""
    ax.tick_params(axis="y", pad=1.5)
    ax.tick_params(axis="x", pad=2.5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  저장: {FIG.name}/{name}.pdf, .png")


def panel_label(ax, txt, x=-0.16, y=1.08):
    """패널 글자는 소문자 굵게 8pt — Nature 계열 관행. 전에는 12pt 라
    축 글자(7pt)보다 한참 커서 그림보다 글자가 먼저 보였다."""
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


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
          시간분할에서만 무너진다.
      (c) 그 붕괴가 학습군·검증군의 양성 비율 차이 때문도 아니다.

    (a) 를 막대가 아니라 점으로 그린다 — y축이 0.45 에서 시작하는데
    막대는 길이가 크기를 뜻하므로 축을 자르면 차이가 과장된다. 점은
    길이로 말하지 않으니 잘라도 된다 (2026-10-23 수정).

    시간분할 칸은 한 칸 더 띄운다. 색을 하나 더 쓰지 않고 위치만으로
    "이 칸은 종류가 다르다" 를 말한다.

    모든 수치는 결과 CSV에서 읽는다 (하드코딩 금지).
      (a)(b) baseline_cv_per_fold.csv — 두 데이터셋 동일 절차
          · random_cv / scaffold_cv : 5-fold 평균 ± fold 간 표준편차
          · temporal                : 단일 분할 (AUROC 는 95% 부트스트랩
                                      CI, ECE 는 prevalence_control.csv 의
                                      부트스트랩 CI)
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

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.95))
    W = 0.3
    splits = ["random_cv", "scaffold_cv", "temporal"]
    # 패널 폭이 좁다. 긴 라벨은 서로 붙어 읽히지 않으므로 한 단어로 줄이고
    # "5겹 교차검증" 은 캡션이 말한다.
    labels = ["Random", "Scaffold", "Temporal"]
    XP3 = np.array([0.0, 1.45, 3.05])
    XP4 = np.array([0.0, 1.0, 2.0, 3.35])
    SER = [("DILIrank", C["dili"]), ("DICTrank", C["dict"])]

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

    def ece_of(ds_key, prefix):
        s = prev[prev.dataset.str.startswith(ds_key) & prev.setting.str.startswith(prefix)]
        if not len(s):
            return np.nan, np.nan, np.nan
        r = s.iloc[0]
        if prefix == "T. ":
            return r.ECE, r.ECE - r.ECE_lo, r.ECE_hi - r.ECE
        sd = r.ECE_sd if pd.notna(r.ECE_sd) else 0.0
        return r.ECE, sd, sd

    def axis_of(ax, xp, labs):
        ax.set_xticks(xp)
        ax.set_xticklabels(labs)
        ax.set_xlim(xp[0] - 0.72, xp[-1] + 0.72)
        ax.tick_params(axis="x", length=0)
        bare(ax)

    def ebars(ax, x, v, lo, hi):
        ax.errorbar(
            x,
            v,
            yerr=[lo, hi],
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=1.6,
            capthick=0.6,
            zorder=4,
        )

    # ---------------------------------------------- (a) 분할 전략별 AUROC
    ax = axes[0]
    for k, (ds, col) in enumerate(SER):
        v, lo, hi = zip(*[stat_of(ds, sp, "AUROC") for sp in splits])
        ax.errorbar(
            XP3 + (k - 0.5) * W,
            v,
            yerr=[lo, hi],
            fmt=ds_marker(ds),
            ms=3.6,
            mfc=col,
            mec=col,
            ecolor=col,
            elinewidth=0.9,
            capsize=1.6,
            capthick=0.9,
            label=ds,
            clip_on=False,
            zorder=3,
        )
    ax.axhline(0.5, color=C["ref"], lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax.text(XP3[-1] + 0.7, 0.504, "chance", fontsize=6, color=C["ref"], ha="right", va="bottom")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.78)
    ax.set_yticks([0.5, 0.6, 0.7])
    axis_of(ax, XP3, labels)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.9,
        handlelength=0.9,
    )
    panel_label(ax, "a", x=-0.30, y=1.17)

    # ---------------------------------------------- (b) 분할 전략별 ECE
    ax = axes[1]
    for k, (ds, col) in enumerate(SER):
        v, lo, hi = zip(*[stat_of(ds, sp, "ECE") for sp in splits])
        ax.bar(XP3 + (k - 0.5) * W, v, W * 0.86, color=col, lw=0, zorder=2)
        ebars(ax, XP3 + (k - 0.5) * W, v, lo, hi)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 0.56)
    ax.set_yticks([0, 0.2, 0.4])
    axis_of(ax, XP3, labels)
    panel_label(ax, "b", x=-0.30, y=1.17)

    # ---------------------------------------------- (c) 유병률 통제별 ECE
    ax = axes[2]
    order = ["R1. ", "R2. ", "R3. ", "T. "]
    # R1/R2/R3 의 뜻은 캡션이 정의한다: R1 = 비율 맞춤 없음, R2 = 검증군만,
    # R3 = 학습·검증 모두.
    labels_c = ["R1", "R2", "R3", "Temporal"]
    for k, (ds, col) in enumerate(SER):
        v, lo, hi = zip(*[ece_of(ds, pf) for pf in order])
        ax.bar(XP4 + (k - 0.5) * W, v, W * 0.86, color=col, lw=0, zorder=2)
        ebars(ax, XP4 + (k - 0.5) * W, v, lo, hi)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, 0.56)
    ax.set_yticks([0, 0.2, 0.4])
    axis_of(ax, XP4, labels_c)
    panel_label(ax, "c", x=-0.30, y=1.17)

    fig.tight_layout(w_pad=1.9)
    save(fig, "Fig1_split_strategies")


# ============================================================ Fig 2
def fig2():
    """보정 곡선 (reliability diagram)."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.75))

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
            [0, 1],
            [0, 1],
            ls=(0, (1.5, 1.8)),
            color="#999999",
            lw=0.7,
            label="Perfect calibration",
            zorder=1,
        )

        for y, p, tag, col, mk, ls, fill in [
            (y_r, p_r, "Random split", C["ref"], "o", (0, (3.5, 2.5)), False),
            (y_t, p_t, "Temporal split", ds_color(nm), ds_marker(nm), "-", True),
        ]:
            xs, ys, ns = rel_curve(y, p)
            sz = 6 + 26 * ns / ns.max()
            ax.plot(xs, ys, ls=ls, color=col, lw=1.0, zorder=2)
            ax.scatter(
                xs,
                ys,
                s=sz,
                facecolor=col if fill else "white",
                marker=mk,
                edgecolor=col,
                linewidth=0.8,
                zorder=3,
                label=tag,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted probability")
        if lab == "a":
            ax.set_ylabel("Observed positive fraction")
        ax.set_title(nm, fontsize=7.5, color=ds_color(nm), pad=3)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        bare(ax)
        ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderaxespad=0.2)
        panel_label(ax, lab, x=-0.26, y=1.18)  # 제목/눈금과 겹치지 않게

    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig2_calibration_curves")


# ============================================================ Fig 7
def fig7():
    """견고성: 모델×표현 격자의 ECE 배율 분포(a)와 절단연도 민감도(b·c).

    2026-10-23: 전에는 (b) 하나에 ΔAUROC 와 ECE 배율을 **이중 축**으로
    겹쳐 그렸다(ax.twinx()). 두 축의 눈금을 어떻게 맞출지는 임의 선택이라
    없는 상관을 만들어 보여준다 — 데이터 시각화에서 가장 많이 지적되는
    오류다. 두 패널로 쪼갰다. 두 양이 같은 x(절단연도) 위에 나란히 있어
    비교는 그대로 되고, 축이 하나씩이라 오해가 생기지 않는다.
    """
    try:
        rob = pd.read_csv(RES / "robustness_models.csv")
        cut = pd.read_csv(RES / "robustness_cutoff.csv")
    except FileNotFoundError:
        print("  Fig7 건너뜀: 결과 파일 없음")
        return

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.95))

    # ---------------------------------------------- (a) 격자 전체의 ECE 배율
    ax = axes[0]
    names = list(rob["dataset"].unique())
    for i, ds in enumerate(names):
        s = rob[rob.dataset == ds]
        jit = np.random.default_rng(SEED).normal(0, 0.055, len(s))
        ax.scatter(
            np.full(len(s), i) + jit,
            s["ece_ratio"],
            s=14,
            color=ds_color(ds),
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )
        med = s["ece_ratio"].median()
        ax.plot([i - 0.24, i + 0.24], [med, med], color="black", lw=1.1, zorder=4)
        ax.text(i, s["ece_ratio"].max() + 0.3, f"{len(s)}/{len(s)}", ha="center", fontsize=6)

    ax.axhline(1.0, color=C["ref"], ls=(0, (4, 3)), lw=0.5, zorder=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("ECE ratio (temporal / random)")
    ax.set_ylim(0.8, 6.6)
    ax.text(
        0.98,
        0.97,
        "All 32 combinations > 1.0",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        ha="right",
        style="italic",
    )
    bare(ax)
    panel_label(ax, "a", x=-0.30, y=1.14)

    # ------------------------------------- (b)(c) 절단연도 민감도, 축 하나씩
    style = {
        "DILIrank": dict(marker="o", fill=True, dx=-0.12),
        "DICTrank": dict(marker="s", fill=False, dx=+0.12),
    }

    def year_panel(ax, col, ylab, ylim, hline=None):
        for ds in cut["dataset"].unique():
            s = cut[cut.dataset == ds].sort_values("cutoff")
            st = style.get(ds, dict(marker="o", fill=True, dx=0.0))
            c = ds_color(ds)
            ax.plot(
                s["cutoff"] + st["dx"],
                s[col],
                "-",
                marker=st["marker"],
                ms=3.4,
                lw=1.0,
                color=c,
                markerfacecolor=c if st["fill"] else "white",
                markeredgecolor=c,
                markeredgewidth=0.9,
                label=ds,
                zorder=3,
            )
        if hline is not None:
            ax.axhline(hline, color=C["ref"], ls=(0, (4, 3)), lw=0.5, zorder=1)
        ax.set_xticks([2005, 2010, 2015])
        ax.set_xlim(2002.5, 2017.5)
        ax.set_xlabel("Temporal split cutoff (year)")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        bare(ax)

    ax = axes[1]
    year_panel(ax, "d_auroc", "ΔAUROC (random − temporal)", (0, 0.13), hline=0)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.9,
        handlelength=1.3,
    )
    panel_label(ax, "b", x=-0.30, y=1.14)

    ax = axes[2]
    year_panel(ax, "ece_ratio", "ECE ratio (temporal / random)", (0, 4.0), hline=1.0)
    panel_label(ax, "c", x=-0.30, y=1.14)

    fig.tight_layout(w_pad=1.9)
    save(fig, "Fig7_robustness")


# ============================================================ Fig 3
def fig3():
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
        print("  Fig3 건너뜀: prevalence_sweep.csv 없음")
        return

    style = {
        "DILIrank": dict(c=C["dili"], marker="o", fill=True),
        "DICTrank": dict(c=C["dict"], marker="s", fill=False),
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

    ax.axvline(0, color=C["ref"], ls=(0, (4, 3)), lw=0.5, zorder=1)
    ax.set_xlabel("Training − test prevalence gap")
    ax.set_ylabel("ECE")
    bare(ax)
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

    ax.axhline(1.0, color=C["ref"], ls=(0, (4, 3)), lw=0.5, zorder=1)
    ax.axvline(0, color=C["ref"], ls=(0, (4, 3)), lw=0.5, zorder=1)
    ax.set_xlabel("Training − test prevalence gap")
    ax.set_ylabel("Ratio to value at zero gap")
    bare(ax)

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
    save(fig, "Fig3_prevalence_sweep")


# ============================================================ Fig S1
def figS1():
    """ATC 1단계 계열별 성능 — 데이터셋마다 한 패널.

    2026-10-23: 전에는 두 데이터셋 13개 계열을 한 축에 섞어 AUROC 순으로
    늘어놓았다. 어느 막대가 어느 데이터셋인지 표시가 없어서 ATC 글자가
    중복으로 보였고(C 두 번, N 두 번 ...), 읽을 수 없는 그림이었다.
    패널을 갈라 데이터셋마다 따로 그린다.

    계열 이름은 결과 CSV 가 한국어로 들고 있으므로 여기서 영문으로 옮긴다.
    """
    f = RES / "atc_stratified.csv"
    if not f.exists():
        print("  FigS1 건너뜀: atc_stratified.csv 없음")
        return
    d = pd.read_csv(f).dropna(subset=["auroc"])

    # ATC 1단계 분류의 공식 영문명 (WHO ATC index). 줄이 길어지면 y축이
    # 밀리므로 통용되는 짧은 형태를 쓰고, 글자 코드를 함께 남긴다.
    ATC_EN = {
        "A": "Alimentary, metabolism",
        "B": "Blood",
        "C": "Cardiovascular",
        "J": "Anti-infectives",
        "L": "Antineoplastic, immune",
        "N": "Nervous system",
        "V": "Various",
    }

    names = [x for x in ("DILIrank", "DICTrank") if (d.dataset == x).any()]
    fig, axes = plt.subplots(1, len(names), figsize=(COL2, 2.0), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, ds, lab in zip(axes, names, "ab"):
        s = d[d.dataset == ds].sort_values("auroc")
        yy = np.arange(len(s))
        ax.barh(yy, s["auroc"], height=0.72, color=ds_color(ds), lw=0)
        ax.axvline(0.5, color=C["ref"], ls=(0, (4, 3)), lw=0.5, zorder=3)
        ax.set_yticks(yy)
        ax.set_yticklabels(
            ["%s %s (n=%d)" % (r.atc, ATC_EN.get(r.atc, ""), int(r.n)) for _, r in s.iterrows()],
            fontsize=5.8,
        )
        ax.set_ylim(-0.7, len(s) - 0.3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("AUROC (temporal split)")
        ax.set_title(ds, fontsize=7.5, color=ds_color(ds), pad=3)
        ax.tick_params(axis="y", length=0)
        bare(ax)
        panel_label(ax, lab, x=-0.62, y=1.14)

    axes[-1].text(0.515, -0.62, "chance", fontsize=6, color=C["ref"])
    fig.tight_layout(w_pad=1.6)
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
    lab = {"A.": "None", "B.": "Platt", "C.": "Isotonic", "D.": "Prior\nshift", "E.": "Oracle"}
    style = {
        "DILIrank": dict(c=C["dili"], marker="o", mfc=C["dili"]),
        "DICTrank": dict(c=C["dict"], marker="s", mfc="white"),
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
    ax.set_xticks(x)
    ax.set_xticklabels([lab[p] for p in order])
    ax.set_xlim(-0.45, len(order) - 0.55)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, None)
    bare(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.2)
    panel_label(ax, "a", x=-0.22, y=1.10)

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
    cols = [ds_color(r[0]) for r in rows]
    ax.bar(xs - wd, [r[2] for r in rows], wd, color=C["ref"], lw=0)
    ax.bar(xs, [r[3] for r in rows], wd, color=cols, lw=0)
    ax.bar(
        xs + wd,
        [r[4] for r in rows],
        wd,
        facecolor="white",
        edgecolor=cols,
        lw=0.9,
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
    ax.tick_params(axis="x", length=0)
    bare(ax)
    _pair_fill = tuple(Patch(facecolor=c, lw=0) for c in (C["dili"], C["dict"]))
    _pair_open = tuple(
        Patch(facecolor="white", edgecolor=c, lw=0.9) for c in (C["dili"], C["dict"])
    )
    ax.legend(
        handles=[Patch(facecolor=C["ref"], lw=0), _pair_fill, _pair_open],
        labels=["Random", "Temporal", "Temporal, negatives"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        frameon=False,
        # 패널 안에 두면 막대를 덮는다. 패널 위 바깥에 한 줄로 둔다.
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=5.8,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.1,
    )
    panel_label(ax, "b", x=-0.22, y=1.14)

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
        "DILIrank": dict(c=C["dili"], marker="o", mfc=C["dili"]),
        "DICTrank": dict(c=C["dict"], marker="s", mfc="white"),
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
    # 델타는 수식(mathtext)으로 쓰지 않는다. mathtext 는 Arial 이 아니라
    # 기본 수식 폰트(DejaVu Sans)로 그려지고, 그러면 Fig6 PDF 에만 다른
    # 폰트가 하나 더 임베딩된다 (2026-10-23 확인: 8개 그림 중 Fig6 만
    # DejaVuSans 를 들고 있었다). 유니코드 글자로 쓰면 본문 폰트를 탄다.
    ax.set_ylabel("\u0394ECE vs fixed model")
    ax.set_xticks(sorted(q.delay.unique()))
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    panel_label(ax, "b")

    fig.tight_layout(w_pad=2.4)
    save(fig, "Fig6_retraining_delay")


def fig4():
    """이동은 실재하지만, 그 크기만으로는 붕괴를 설명하지 못한다 (§33).

    (a) alpha 를 0 에서 1 로 올리면 비시간 분할의 이동량(시대 분류 AUC)이
        무작위 수준에서 축 극단까지 연속으로 올라간다. 데이터셋마다 선이
        세 개(분자량·logP·무작위 방향)인데 거의 포개진다 — 어느 축을 쓰든
        alpha 가 이동량을 같은 식으로 올린다. 가로 점선이 시간분할의
        이동량이고, 속이 빈 표식이 거기에 맞춘 alpha* 다. 즉 이동량 자체는
        원하는 만큼 맞출 수 있다.
    (b) 그렇게 이동량을 맞춘 지점에서 ECE 를 비교한다. 가로선이 시간분할
        ECE 다. 같은 이동량인데도 여섯 설정 중 다섯에서 시간분할이 위에
        있다 — 이동의 크기가 같아도 시간분할만 더 무너진다.
    """
    try:
        sw = pd.read_csv(RES / "shift_matched_v2.csv")
        mt = pd.read_csv(RES / "shift_matched_v2_matched.csv")
    except FileNotFoundError as e:
        print(f"  Fig4 건너뜀: {e}")
        return

    # 축 이름은 결과 파일이 한국어다. 그림은 영문으로 쓴다.
    AX = {"A1. 분자량": "MolWt", "A2. logP": "logP", "A3. 무작위 방향": "Random direction"}
    DS = ["DILIrank", "DICTrank"]
    col = {"DILIrank": C["dili"], "DICTrank": C["dict"]}

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7))

    # -------------------------------------------- (a) alpha 를 훑는다
    ax = axes[0]
    for ds in DS:
        g = sw[sw.dataset == ds]
        for _axis, ga in g.groupby("axis"):
            ga = ga.sort_values("alpha_num")
            # 축 세 개를 같은 실선으로 그린다. 거의 포개지는 것 자체가
            # 결과다 — 어느 축을 쓰든 alpha 가 이동량을 같은 식으로 올린다.
            ax.plot(ga.alpha_num, ga.shift_auc, color=col[ds], lw=0.9, alpha=0.75, zorder=2)
        ax.axhline(
            float(g.shift_auc_temporal.iloc[0]),
            color=col[ds],
            lw=0.8,
            ls=":",
            zorder=1,
        )
        m = mt[mt.dataset == ds]
        ax.plot(
            m.alpha_star,
            m.shift_auc_matched,
            ls="none",
            marker="o",
            ms=4.5,
            mfc="white",
            mec=col[ds],
            mew=1.1,
            zorder=4,
        )
    ax.set_xlabel("alpha  (0 = random split, 1 = axis extreme)")
    ax.set_ylabel("Era-classification AUC of the split")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.45, 1.02)
    hs = [
        mpl.lines.Line2D([], [], color=C["dili"], lw=1.0),
        mpl.lines.Line2D([], [], color=C["dict"], lw=1.0),
        tuple(mpl.lines.Line2D([], [], color=c, lw=0.8, ls=":") for c in (C["dili"], C["dict"])),
        tuple(
            mpl.lines.Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=c, mew=1.0)
            for c in (C["dili"], C["dict"])
        ),
    ]
    ax.legend(
        hs,
        ["DILIrank", "DICTrank", "Time split", "Matched alpha*"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        frameon=False,
        loc="lower right",
        handlelength=1.8,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    bare(ax)
    panel_label(ax, "a", x=-0.22, y=1.10)

    # -------------------------------------------- (b) 맞춘 지점의 ECE
    ax = axes[1]
    xs, labs = {}, []
    pos = 0.0
    for ds in DS:
        m = mt[mt.dataset == ds].sort_values("axis")
        xs[ds] = np.arange(len(m)) + pos
        ax.hlines(
            float(m.ece_temporal.iloc[0]),
            xs[ds][0] - 0.42,
            xs[ds][-1] + 0.42,
            color=col[ds],
            lw=1.3,
            zorder=3,
        )
        ax.errorbar(
            xs[ds],
            m.ece_matched,
            yerr=m.ece_matched_sd,
            ls="none",
            marker="o",
            ms=4.5,
            mfc="white",
            mec=col[ds],
            mew=1.1,
            ecolor=col[ds],
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=4,
        )
        labs += [AX.get(a, a) for a in m.axis]
        pos += len(m) + 1.0
    allx = np.concatenate([xs[d] for d in DS])
    ax.set_xticks(allx)
    ax.set_xticklabels([t.replace(" ", "\n") for t in labs])
    ax.tick_params(axis="x", length=0)
    bare(ax)
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, None)
    for ds in DS:
        ax.text(
            xs[ds].mean(),
            1.02,
            ds,
            color=col[ds],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    hs = [
        tuple(mpl.lines.Line2D([], [], color=c, lw=1.3) for c in (C["dili"], C["dict"])),
        tuple(
            mpl.lines.Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=c, mew=1.0)
            for c in (C["dili"], C["dict"])
        ),
    ]
    ax.legend(
        hs,
        ["Time split", "Shift-matched split"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
        frameon=False,
        loc="lower left",
        handlelength=1.8,
        labelspacing=0.3,
        borderaxespad=0.2,
    )
    panel_label(ax, "b", x=-0.22, y=1.10)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    save(fig, "Fig4_shift_matched")


def main():
    print("그림 생성 중...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    figS1()
    print(f"\n완료. {FIG} 확인")
    print("PDF는 벡터(투고용), PNG는 600dpi(확인용)")


if __name__ == "__main__":
    main()
