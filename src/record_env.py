"""
실행 환경 기록 — 논문 Reproducibility 절과 실험결과 보고서 §3 의 근거 파일을 만든다.

반드시 분석에 쓴 환경(conda env `pharmml`)에서 실행할 것:

    conda activate pharmml
    python src/record_env.py

출력:
    results/environment.txt    사람이 읽는 표 + 전체 pip freeze
    results/environment.json   기계가 읽는 형태
    environment.yml            conda env export (conda 가 있으면)
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# 보고서 §3 에 적는 순서와 같게 유지한다
CORE = [
    ("RDKit", "rdkit", ("rdkitVersion", "__version__")),
    ("XGBoost", "xgboost", ("__version__",)),
    ("scikit-learn", "sklearn", ("__version__",)),
    ("NumPy", "numpy", ("__version__",)),
    ("pandas", "pandas", ("__version__",)),
    ("SciPy", "scipy", ("__version__",)),
    ("statsmodels", "statsmodels", ("__version__",)),
    ("LightGBM", "lightgbm", ("__version__",)),
]


def ver(mod, attrs):
    try:
        m = __import__(mod)
    except Exception as e:
        return f"IMPORT FAIL ({type(e).__name__})"
    for a in attrs:
        v = getattr(m, a, None)
        if v:
            return str(v)
    return "버전 속성 없음"


def run(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=180
        ).stdout.strip()
    except Exception as e:
        return f"(실행 실패: {e})"


def main():
    py = ".".join(map(str, sys.version_info[:3]))
    core = {label: ver(mod, attrs) for label, mod, attrs in CORE}

    meta = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": py,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_env": run("echo $CONDA_DEFAULT_ENV") or "(conda env 아님)",
        "core": core,
    }

    lines = []
    lines.append("실행 환경 기록")
    lines.append("=" * 70)
    lines.append(f"기록 시각 (UTC)  {meta['recorded_utc']}")
    lines.append(f"플랫폼           {meta['platform']} / {meta['machine']}")
    lines.append(f"conda env        {meta['conda_env']}")
    lines.append(f"python 실행파일  {meta['python_executable']}")
    lines.append("")
    lines.append("핵심 패키지 (보고서 §3 · 논문 Methods 와 대조할 값)")
    lines.append("-" * 70)
    lines.append(f"  {'Python':<16} {py}")
    for label in core:
        lines.append(f"  {label:<16} {core[label]}")

    fails = [k for k, v in core.items() if v.startswith("IMPORT FAIL")]
    lines.append("")
    if fails:
        lines.append(f"  [경고] import 실패 {len(fails)}건: {', '.join(fails)}")
        lines.append(
            "         분석에 쓴 환경이 맞는지 확인할 것 " "(conda activate pharmml 후 재실행)."
        )
    else:
        lines.append("  전부 import 성공 — 이 환경이 분석 환경이다.")

    lines.append("")
    lines.append("=" * 70)
    lines.append("pip freeze 전체")
    lines.append("=" * 70)
    lines.append(run(f'"{sys.executable}" -m pip freeze'))

    (RES / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES / "environment.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    yml = run("conda env export --no-builds")
    if yml and not yml.startswith("(실행 실패") and "name:" in yml:
        (ROOT / "environment.yml").write_text(yml + "\n", encoding="utf-8")
        wrote_yml = True
    else:
        wrote_yml = False

    print("\n".join(lines[: len(core) + 12]))
    print()
    print(f"저장: {RES/'environment.txt'}")
    print(f"      {RES/'environment.json'}")
    print(
        f"      {ROOT/'environment.yml'}"
        if wrote_yml
        else "      environment.yml — conda env export 실패, 건너뜀"
    )
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
