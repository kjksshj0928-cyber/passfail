#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
easy_qc_passfail_v1_0.py
=========================
경량 S-파라미터 QC 도구. 복잡한 골든/ML 파이프라인 없이도 직관적인 규칙 기반 검사를 수행한다.

핵심 아이디어
--------------
1. Y11(dB) 파형을 이동 평균으로 평활화한 뒤 잔차 RMS를 노이즈 지표로 사용한다.
2. 가장 깊은 노치(minimum)를 찾아 좌/우 봉우리 대비 깊이와 반치폭(FWHM)을 계산한다.
3. 노치 주변 기울기 균형, 다중 피크 개수, 전체 변동폭 등을 분석하여 일그러짐/컨택 불량을 검출한다.
4. 규칙을 위반하면 FAIL, 모든 조건을 만족하면 PASS로 분류하며 원인을 로그/CSV에 남긴다.

사용법
-------
python easy_qc_passfail_v1_0.py classify --root <s1p 폴더>
    [--noise_limit_db 0.35] [--min_notch_depth_db 8.0] [--max_fwhm_mhz 180.0]
    [--max_skew_db 6.0] [--min_variation_db 4.0] [--roi_low_mhz 0] [--roi_high_mhz 0]

- ROI(주파수 제한)를 지정하지 않으면 파일 내 전체 범위를 사용한다.
- 결과는 <root>/_easy_qc_results/qc_results.csv 에 저장되며 PASS/FAIL 비율과 위반 항목이 출력된다.
- --explain 옵션을 주면 각 파일별 계산된 피처를 표 형태로 요약한다.

ML/Shot-aware 시스템과의 차이
-----------------------------
- 복잡한 골든/보정 단계 없이 단일 파일만으로 즉시 판단한다.
- ML 학습 없이도 컨택 불량, 노이즈, 노치 붕괴와 같은 직관적 패턴을 규칙 기반으로 잡아낸다.
- 필요 시 기존 rev5/6 스크립트와 같은 폴더에서 함께 실행하여 교차 검증용 보조 도구로 사용할 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CurveFeatures:
    path: str
    label: str
    noise_rms_db: float
    notch_depth_db: float
    fwhm_mhz: float
    skew_db: float
    variation_db: float
    multi_peak_count: int
    reasons: List[str]


@dataclass
class Thresholds:
    noise_limit_db: float = 0.35
    min_notch_depth_db: float = 8.0
    max_fwhm_mhz: float = 180.0
    max_skew_db: float = 6.0
    min_variation_db: float = 4.0


# ---------------------------------------------------------------------------
# S1P 유틸리티
# ---------------------------------------------------------------------------

def _unit_scale(unit: str) -> float:
    unit = unit.upper()
    return {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}.get(unit, 1.0)


def parse_s1p(path: str) -> Tuple[np.ndarray, np.ndarray]:
    unit = "HZ"
    fmt = "RI"
    data: List[Tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                toks = line[1:].split()
                u = [t.upper() for t in toks]
                for candidate in ("HZ", "KHZ", "MHZ", "GHZ"):
                    if candidate in u:
                        unit = candidate
                        break
                if "DB" in u:
                    fmt = "DB"
                elif "MA" in u:
                    fmt = "MA"
                else:
                    fmt = "RI"
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not data:
        raise ValueError(f"빈 또는 파싱 불가 파일: {path}")
    arr = np.asarray(data, dtype=float)
    freq = arr[:, 0] * _unit_scale(unit)
    if fmt == "DB":
        mag_db = arr[:, 1]
        phase_deg = arr[:, 2]
        real = 10 ** (mag_db / 20) * np.cos(np.deg2rad(phase_deg))
        imag = 10 ** (mag_db / 20) * np.sin(np.deg2rad(phase_deg))
        s11 = real + 1j * imag
    elif fmt == "MA":
        mag = arr[:, 1]
        ang = np.deg2rad(arr[:, 2])
        s11 = mag * (np.cos(ang) + 1j * np.sin(ang))
    else:
        s11 = arr[:, 1] + 1j * arr[:, 2]
    y11_db = 20 * np.log10(np.abs(1 - s11))
    return freq, y11_db


# ---------------------------------------------------------------------------
# 특징 계산 함수
# ---------------------------------------------------------------------------

def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    window = min(window, arr.size)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def estimate_noise(y_db: np.ndarray, window: int = 51) -> float:
    window = max(5, min(window, y_db.size // 5 * 2 + 1))
    if window % 2 == 0:
        window += 1
    smooth = moving_average(y_db, window)
    residual = y_db - smooth
    return float(np.std(residual))


def restrict_roi(freq: np.ndarray, y_db: np.ndarray, low_hz: float, high_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    if low_hz <= 0 and high_hz <= 0:
        return freq, y_db
    lo = low_hz if low_hz > 0 else freq.min()
    hi = high_hz if high_hz > 0 else freq.max()
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        return freq, y_db
    return freq[mask], y_db[mask]


def notch_metrics(freq: np.ndarray, y_db: np.ndarray) -> Dict[str, float]:
    idx = int(np.argmin(y_db))
    min_db = float(y_db[idx])
    min_freq = float(freq[idx])
    span = max(5, y_db.size // 20)
    left = max(0, idx - span)
    right = min(y_db.size, idx + span)
    left_peak = float(np.max(y_db[left:idx])) if idx > left else min_db
    right_peak = float(np.max(y_db[idx:right])) if idx + 1 < right else min_db
    notch_depth = ((left_peak + right_peak) / 2.0) - min_db

    half_level = min_db + notch_depth / 2.0
    l_idx = idx
    while l_idx > 0 and y_db[l_idx] < half_level:
        l_idx -= 1
    r_idx = idx
    while r_idx < y_db.size - 1 and y_db[r_idx] < half_level:
        r_idx += 1
    fwhm = float((freq[r_idx] - freq[l_idx]) / 1e6) if r_idx > l_idx else 0.0

    left_slope = (left_peak - min_db) / max(1.0, (min_freq - float(freq[l_idx])) / 1e6)
    right_slope = (right_peak - min_db) / max(1.0, (float(freq[r_idx]) - min_freq) / 1e6)
    skew = abs(left_slope - right_slope)

    # multi-peak 탐지: 주변에서 threshold 이상의 보조 노치가 몇 개 있는지 계산
    prominence_threshold = notch_depth * 0.4
    diff = np.diff(np.sign(np.diff(y_db)))
    peak_indices = np.where(diff < 0)[0] + 1
    multi_count = 0
    for p in peak_indices:
        if p == idx:
            continue
        neighbor_left = max(0, p - span)
        neighbor_right = min(y_db.size, p + span)
        local_peak = y_db[p]
        local_env = max(np.max(y_db[neighbor_left:p]), np.max(y_db[p:neighbor_right]))
        if local_env - local_peak > prominence_threshold:
            multi_count += 1
    return {
        "depth_db": notch_depth,
        "fwhm_mhz": fwhm,
        "skew_db": skew,
        "multi_peak": float(multi_count),
        "variation_db": float(np.max(y_db) - np.min(y_db)),
    }


def classify_curve(path: str, freq: np.ndarray, y_db: np.ndarray, thresholds: Thresholds) -> CurveFeatures:
    noise = estimate_noise(y_db)
    notch = notch_metrics(freq, y_db)
    reasons: List[str] = []

    if noise > thresholds.noise_limit_db:
        reasons.append(f"NOISE>{thresholds.noise_limit_db:.2f}dB")
    if notch["depth_db"] < thresholds.min_notch_depth_db:
        reasons.append(f"SHALLOW_NOTCH<{thresholds.min_notch_depth_db:.1f}dB")
    if notch["fwhm_mhz"] > thresholds.max_fwhm_mhz:
        reasons.append(f"WIDE_FWHM>{thresholds.max_fwhm_mhz:.1f}MHz")
    if notch["skew_db"] > thresholds.max_skew_db:
        reasons.append(f"SKEW>{thresholds.max_skew_db:.1f}dB")
    if notch["variation_db"] < thresholds.min_variation_db:
        reasons.append(f"LOW_VARIATION<{thresholds.min_variation_db:.1f}dB")
    if notch["multi_peak"] >= 2:
        reasons.append("MULTI_NOTCH")

    label = "PASS" if not reasons else "FAIL"
    return CurveFeatures(
        path=path,
        label=label,
        noise_rms_db=float(noise),
        notch_depth_db=float(notch["depth_db"]),
        fwhm_mhz=float(notch["fwhm_mhz"]),
        skew_db=float(notch["skew_db"]),
        variation_db=float(notch["variation_db"]),
        multi_peak_count=int(notch["multi_peak"]),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_s1p(root: str) -> List[str]:
    paths: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".s1p"):
                paths.append(os.path.join(base, name))
    return sorted(paths)


def write_csv(root: str, rows: Sequence[CurveFeatures]) -> str:
    out_dir = os.path.join(root, "_easy_qc_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qc_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path",
            "label",
            "noise_rms_db",
            "notch_depth_db",
            "fwhm_mhz",
            "skew_db",
            "variation_db",
            "multi_peak_count",
            "reasons",
        ])
        for row in rows:
            writer.writerow([
                row.path,
                row.label,
                f"{row.noise_rms_db:.4f}",
                f"{row.notch_depth_db:.4f}",
                f"{row.fwhm_mhz:.4f}",
                f"{row.skew_db:.4f}",
                f"{row.variation_db:.4f}",
                row.multi_peak_count,
                "|".join(row.reasons),
            ])
    return out_path


def summarize(rows: Sequence[CurveFeatures]) -> None:
    total = len(rows)
    if total == 0:
        print("[경고] s1p 파일을 찾지 못했습니다.")
        return
    pass_cnt = sum(1 for r in rows if r.label == "PASS")
    fail_cnt = total - pass_cnt
    print(f"총 {total}개 → PASS {pass_cnt} ({pass_cnt/total*100:.1f}%), FAIL {fail_cnt} ({fail_cnt/total*100:.1f}%)")
    print("상위 FAIL 샘플:")
    for row in rows:
        if row.label == "FAIL":
            print(f" - {row.path}: {', '.join(row.reasons)}")
    if pass_cnt == total:
        print("모든 샘플이 기준을 만족했습니다.")


def explain(rows: Sequence[CurveFeatures], limit: int = 10) -> None:
    if not rows:
        return
    print("\n피처 요약 (상위 {limit}개):")
    header = (
        f"{'idx':>3}  {'label':<4}  {'noise':>6}  {'notch':>6}  {'fwhm':>6}  "
        f"{'skew':>6}  {'var':>6}  {'multi':>5}  path"
    )
    print(header)
    for idx, row in enumerate(rows[:limit]):
        print(
            f"{idx:>3}  {row.label:<4}  {row.noise_rms_db:6.3f}  {row.notch_depth_db:6.2f}  "
            f"{row.fwhm_mhz:6.1f}  {row.skew_db:6.2f}  {row.variation_db:6.2f}  "
            f"{row.multi_peak_count:5d}  {row.path}"
        )


def run_classify(args: argparse.Namespace) -> None:
    files = collect_s1p(args.root)
    if not files:
        print("[오류] 지정한 폴더에서 s1p 파일을 찾을 수 없습니다.")
        return

    thresholds = Thresholds(
        noise_limit_db=args.noise_limit_db,
        min_notch_depth_db=args.min_notch_depth_db,
        max_fwhm_mhz=args.max_fwhm_mhz,
        max_skew_db=args.max_skew_db,
        min_variation_db=args.min_variation_db,
    )

    rows: List[CurveFeatures] = []
    for path in files:
        try:
            freq, y_db = parse_s1p(path)
            freq, y_db = restrict_roi(freq, y_db, args.roi_low_mhz * 1e6, args.roi_high_mhz * 1e6)
            rows.append(classify_curve(path, freq, y_db, thresholds))
        except Exception as exc:
            rows.append(
                CurveFeatures(
                    path=path,
                    label="FAIL",
                    noise_rms_db=float("nan"),
                    notch_depth_db=float("nan"),
                    fwhm_mhz=float("nan"),
                    skew_db=float("nan"),
                    variation_db=float("nan"),
                    multi_peak_count=0,
                    reasons=[f"PARSE_ERROR:{exc}"],
                )
            )
    summarize(rows)
    out_path = write_csv(args.root, rows)
    print(f"결과 CSV: {out_path}")
    if args.explain:
        explain(rows, limit=args.explain_limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="간단한 규칙 기반 BAW QC 도구")
    sub = parser.add_subparsers(dest="command")

    classify = sub.add_parser("classify", help="폴더 단위 분류")
    classify.add_argument("--root", required=True, help="s1p 파일이 들어있는 폴더")
    classify.add_argument("--noise_limit_db", type=float, default=0.35)
    classify.add_argument("--min_notch_depth_db", type=float, default=8.0)
    classify.add_argument("--max_fwhm_mhz", type=float, default=180.0)
    classify.add_argument("--max_skew_db", type=float, default=6.0)
    classify.add_argument("--min_variation_db", type=float, default=4.0)
    classify.add_argument("--roi_low_mhz", type=float, default=0.0, help="ROI 하한 (MHz)")
    classify.add_argument("--roi_high_mhz", type=float, default=0.0, help="ROI 상한 (MHz)")
    classify.add_argument("--explain", action="store_true", help="피처 요약을 표로 출력")
    classify.add_argument("--explain_limit", type=int, default=10, help="요약에 포함할 최대 샘플 수")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "classify":
        parser.print_help()
        return
    run_classify(args)


if __name__ == "__main__":
    main()
