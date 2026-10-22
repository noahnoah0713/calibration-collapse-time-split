# -*- coding: utf-8 -*-
"""'쓸 만한 개선' 기준의 임의성을 없앤다 — 한계 23번.

왜 필요한가
----------
§29 는 처방의 음성 결과를 세 판정으로 나눈다.

  (가) 구간이 전부 0 아래 — 개선이 유의하다
  (나) 구간이 허용하는 최대 개선조차 간격의 25% 에 못 미친다
  (다) 구간이 쓸 만한 개선을 아직 배제하지 못한다

여기서 25% 는 내가 정한 값이다. 한계 23번에 "사전에 정했으나 임의
선택" 이라고 적어 두었지만, 적어 두는 것만으로는 부족하다. 심사자가 묻는다
— "30% 로 하면 결론이 바뀌는가."

이 스크립트가 하는 것
--------------------
기준 U 를 0.05 에서 0.60 까지 훑어 각 팔의 판정이 어디서 뒤집히는지 센다.
핵심은 팔마다 뒤집히는 지점이 하나의 수로 정해진다는 것이다 —

    U* = (구간이 허용하는 최대 개선) / (메워야 할 간격) = (−Δ하한) / 간격

U > U* 면 (나), U < U* 면 (다) 다. 그래서 U* 를 팔마다 적어 두면 어떤
기준을 쓰더라도 독자가 판정을 직접 읽을 수 있다. 기준을 고를 필요가
사라진다.

모델을 다시 적합하지 않는다 — §29 가 저장한 Δ 구간과 간격만 쓴다.
그래서 몇 초면 끝난다.

사용법
------
    python src/useful_threshold_sweep.py

출력
----
    results/useful_threshold_sweep.csv        팔별 U*
    results/useful_threshold_grid.csv         기준별 판정 개수
    results/useful_threshold_sweep_summary.txt
"""

import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# 이 스크립트는 산수만 한다 — rdkit·sklearn·xgboost 를 쓰지 않는다.
# 그래서 _shared 를 끌어오지 않고 로그만 자체 구현한다. 모델 환경이 없는
# 곳에서도 돌아가야 하는 것이 요점이다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


class _Log(object):
    def __init__(self):
        self.lines = []

    def __call__(self, m=""):
        print(m, flush=True)
        self.lines.append(str(m))

    def save(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


log = _Log()

GRID = [round(0.05 * k, 2) for k in range(1, 13)]  # 0.05 ~ 0.60
REF = 0.25  # §29 가 쓴 기준


def verdict(lo, hi, gap, U):
    """§29 의 verdict() 와 같은 규칙. U 만 바꿔 끼운다."""
    if hi < 0:
        return "가"
    if lo > 0:
        return "가'"
    if gap != gap or gap <= 0:
        return "다"
    return "나" if (-lo) < U * gap else "다"


def main():
    log("'쓸 만한 개선' 기준의 임의성을 없앤다 — 한계 23번")
    log(
        "§29 의 기준 %.0f%% 를 %.0f%%–%.0f%% 로 훑어 판정이 어디서 뒤집히는지 센다."
        % (REF * 100, min(GRID) * 100, max(GRID) * 100)
    )
    log("모델을 다시 적합하지 않는다 — 저장된 Δ 구간과 간격만 쓴다.")
    log("")

    p = os.path.join(RES, "remedy_power.csv")
    assert os.path.exists(p), "remedy_power.csv 가 없다 — 먼저 §29 를 돌릴 것"
    d = pd.read_csv(p)
    pa = os.path.join(RES, "remedy_power_ad.csv")
    ad = pd.read_csv(pa) if os.path.exists(pa) else None

    # ---- 팔별 뒤집히는 지점 U* ----
    rows = []
    for _, r in d.iterrows():
        gap = float(r.gap_to_control)
        lo, hi = float(r.delta_lo), float(r.delta_hi)
        fixed = (hi < 0) or (lo > 0)  # U 와 무관하게 판정이 정해지는 팔
        ustar = (-lo) / gap if gap > 0 else float("nan")
        rows.append(
            {
                "dataset": r.dataset,
                "remedy": r.remedy,
                "deployable": r.deployable,
                "delta": float(r.delta),
                "delta_lo": lo,
                "delta_hi": hi,
                "gap_to_control": gap,
                "u_star": ustar,
                "u_independent": bool(fixed),
                "verdict_at_ref": verdict(lo, hi, gap, REF),
            }
        )
    U = pd.DataFrame(rows)
    U.to_csv(os.path.join(RES, "useful_threshold_sweep.csv"), index=False, encoding="utf-8")

    log(
        "%-10s %-36s %-14s %9s %-10s"
        % ("데이터셋", "처방", "배포", "U*", "기준 %.0f%% 판정" % (REF * 100))
    )
    log("-" * 96)
    for _, r in U.iterrows():
        log(
            "%-10s %-36s %-14s %9s %-10s"
            % (
                r.dataset,
                r.remedy,
                r.deployable,
                "—" if r.u_independent else "%.3f" % r.u_star,
                "(%s)" % r.verdict_at_ref,
            )
        )
    log("")

    # ---- 기준별 판정 개수 ----
    grid = []
    for u in GRID:
        v = [
            verdict(float(r.delta_lo), float(r.delta_hi), float(r.gap_to_control), u)
            for _, r in d.iterrows()
        ]
        grid.append(
            {
                "useful": u,
                "n_ga": v.count("가"),
                "n_ga_bad": v.count("가'"),
                "n_na": v.count("나"),
                "n_da": v.count("다"),
            }
        )
    G = pd.DataFrame(grid)
    G.to_csv(os.path.join(RES, "useful_threshold_grid.csv"), index=False, encoding="utf-8")
    log("%8s %8s %8s %8s %8s" % ("기준", "(가)", "(가')", "(나)", "(다)"))
    log("-" * 96)
    for _, r in G.iterrows():
        mark = "  ← §29" if abs(float(r.useful) - REF) < 1e-9 else ""
        log(
            "%7.0f%% %8d %8d %8d %8d%s" % (r.useful * 100, r.n_ga, r.n_ga_bad, r.n_na, r.n_da, mark)
        )
    log("")

    # ---- 판정 ----
    log("=" * 96)
    log("판정")
    log("=" * 96)
    _fixed = U[U.u_independent]
    log(
        "  **기준과 무관하게 판정이 정해지는 팔: %d/%d.** 구간이 0 을 포함하지 않으므로"
        % (len(_fixed), len(U))
    )
    log(
        "  어떤 기준을 써도 (가) 또는 (가') 다 — %s."
        % ", ".join(
            "%s %s" % (r.dataset, str(r.remedy).split(".")[0]) for _, r in _fixed.iterrows()
        )
    )
    log("")
    _var = U[~U.u_independent].sort_values("u_star")
    log(
        "  **기준에 걸리는 팔 %d개의 U\\*** (이 값보다 큰 기준에서는 (나), 작으면 (다)):"
        % len(_var)
    )
    for _, r in _var.iterrows():
        log(
            "      %-10s %-36s U* = %.3f  → 기준 %.0f%% 에서 (%s)"
            % (r.dataset, r.remedy, r.u_star, REF * 100, r.verdict_at_ref)
        )
    log("")
    if len(_var):
        lo_u, hi_u = float(_var.u_star.min()), float(_var.u_star.max())
        log(
            "  U\\* 가 %.3f–%.3f 에 흩어져 있다. 그래서 기준을 한 값으로 고를 필요가 없다 —"
            % (lo_u, hi_u)
        )
        log("  **팔마다 U\\* 를 적으면 독자가 자기 기준으로 판정을 읽을 수 있다.**")
        log("")
        # 기준을 바꿨을 때 (나)/(다) 가 바뀌는 팔
        for u in (0.10, 0.20, 0.25, 0.30, 0.50):
            chg = [
                (r.dataset, r.remedy)
                for _, r in _var.iterrows()
                if (verdict(r.delta_lo, r.delta_hi, r.gap_to_control, u) != r.verdict_at_ref)
            ]
            log(
                "    기준 %.0f%% → 기준 %.0f%% 대비 판정이 바뀌는 팔 %d개%s"
                % (
                    u * 100,
                    REF * 100,
                    len(chg),
                    (
                        (": " + ", ".join("%s %s" % (a, str(b).split(".")[0]) for a, b in chg))
                        if chg
                        else ""
                    ),
                )
            )
    log("")
    log(
        '  **본문에 쓸 문장** — "쓸 만한 개선" 의 기준을 %.0f%% 로 둔 것은 임의 '
        "선택이지만," % (REF * 100)
    )
    log("  각 팔의 U\\* 를 함께 보고하므로 **결론이 그 선택에 걸려 있지 않다.** 기준을")
    log(
        "  %.0f%% 로 낮추거나 %.0f%% 로 높여도 (가)·(가') 판정 %d건은 그대로다."
        % (min(GRID) * 100, max(GRID) * 100, len(_fixed))
    )
    log("")

    if ad is not None:
        # 필터 팔은 '간격' 을 그 데이터셋의 §29 간격으로 둔다 (같은 기준으로 읽기 위해)
        _gapd = {r.dataset: float(r.gap_to_control) for _, r in d.iterrows()}
        ad = ad.copy()
        ad["u_star"] = [abs(float(r.delta_lo)) / _gapd[r.dataset] for _, r in ad.iterrows()]
        ad["n_incl_zero"] = [(float(r.delta_lo) < 0 < float(r.delta_hi)) for _, r in ad.iterrows()]
        ad.to_csv(os.path.join(RES, "useful_threshold_sweep_ad.csv"), index=False, encoding="utf-8")
        log(
            "  적용범위 필터 %d개 설정도 같은 방식으로 환산했다 — 구간이 0 을 포함하는 "
            "설정 %d/%d, U* 는 %.3f–%.3f 다(§29.3). 버릴 비율이 커질수록 U* 가 올라간다 "
            "— 검정력이 사라지기 때문이다."
            % (len(ad), int(ad.n_incl_zero.sum()), len(ad), ad.u_star.min(), ad.u_star.max())
        )
        for _, r in ad.sort_values(["dataset", "drop_frac"]).iterrows():
            log("      %-10s 버림 %.1f  U* = %.3f" % (r.dataset, r.drop_frac, r.u_star))
        log("")
    log("저장: %s" % os.path.join(RES, "useful_threshold_sweep.csv"))
    log("      %s" % os.path.join(RES, "useful_threshold_grid.csv"))
    log.save(Path(RES) / "useful_threshold_sweep_summary.txt")


if __name__ == "__main__":
    main()
