#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_ml_classifier.py
-----------------------
기존의 복잡한 QC 스크립트에서 머신러닝(ML) 기반 양품/불량 판정 로직만 추출한 간단한 버전입니다.

주요 기능:
1. S-parameter(.s1p) 파일 파싱
2. 데이터로부터 특징(Feature) 추출
3. LDA 모델 학습(fit) 및 JSON 파일로 저장
4. 저장된 모델로 새로운 데이터 분류(classify)

의존성: numpy
실행 방법:
- 모델 학습: python simple_ml_classifier.py fit --good_root <정상샘플폴더> --bad_root <불량샘플폴더> --out model.json
- 모델 분류: python simple_ml_classifier.py classify --root <분류대상폴더> --model model.json
"""

import os
import json
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

try:
    import numpy as np
except ImportError:
    print("[에러] numpy가 설치되지 않았습니다. 'pip install numpy'를 실행해주세요.")
    exit()


# --- 1. S1P 파일 파싱 및 데이터 처리 ---

@dataclass
class S1P:
    """S1P 파일의 데이터를 저장하기 위한 데이터 클래스"""
    freq_hz: np.ndarray
    s11: np.ndarray


def parse_s1p(path: str) -> S1P:
    """S1P 파일을 파싱하여 주파수와 S11 복소수 데이터를 반환합니다."""
    data = []
    unit = 'HZ'
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            if line.startswith('#'):
                # 단위(HZ, KHZ, MHZ, GHZ) 파싱
                toks = line[1:].split()
                tU = [t.upper() for t in toks]
                for cand in ('HZ', 'KHZ', 'MHZ', 'GHZ'):
                    if cand in tU:
                        unit = cand
                        break
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    # 주파수, 실수부, 허수부
                    data.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except (ValueError, IndexError):
                    continue

    if not data:
        raise ValueError(f"S1P 파일에서 유효한 데이터를 찾을 수 없습니다: {path}")

    arr = np.array(data, dtype=float)

    # 단위에 맞게 주파수 스케일 조정
    scale = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}.get(unit, 1.0)
    freq = arr[:, 0] * scale

    # S11을 복소수 형태로 변환 (실수부 + j*허수부)
    s11 = arr[:, 1] + 1j * arr[:, 2]

    # 주파수 순으로 정렬
    idx = np.argsort(freq)
    return S1P(freq_hz=freq[idx], s11=s11[idx])


def y11_db_norm_z50(s: np.ndarray) -> np.ndarray:
    """S11 파라미터를 Y11(Admittance) dB 스케일로 변환합니다."""
    eps = 1e-15
    num = (1.0 - s)
    den = (1.0 + s)
    den = np.where(np.abs(den) < eps, eps, den)
    y_norm = np.abs(num / den)
    y_norm = np.clip(y_norm, 1e-12, None)
    return 20.0 * np.log10(y_norm)


def resample_to(x_src: np.ndarray, y_src: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """소스 데이터를 레퍼런스 주파수 축에 맞게 리샘플링(보간)합니다."""
    return np.interp(x_ref, x_src, y_src)


# --- 2. 특징 추출 (Feature Extraction) ---


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """두 배열 간의 피어슨 상관계수(r)를 계산합니다."""
    if a.size != b.size or a.size < 2:
        return 0.0
    av = a - a.mean()
    bv = b - b.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 1e-9 else 0.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """두 배열 간의 RMSE(Root Mean Square Error)를 계산합니다."""
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size == b.size else float('inf')


def extract_features(sample_curve: np.ndarray, good_golden: np.ndarray, bad_golden: np.ndarray) -> Dict[str, float]:
    """
    샘플 커브로부터 Good/Bad 골든 커브와의 비교를 통해 주요 특징을 추출합니다.
    - r_good/rmse_good: Good 골든 커브와의 상관계수 및 RMSE
    - r_bad/rmse_bad: Bad 골든 커브와의 상관계수 및 RMSE
    """
    return {
        "r_good": pearson_r(sample_curve, good_golden),
        "rmse_good": rmse(sample_curve, good_golden),
        "r_bad": pearson_r(sample_curve, bad_golden),
        "rmse_bad": rmse(sample_curve, bad_golden),
    }


# --- 3. ML 모델 (LDA) ---


def fit_lda(X_good: np.ndarray, X_bad: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Good/Bad 특징 매트릭스로부터 LDA(선형 판별 분석) 모델의 파라미터를 학습합니다.
    """
    mu_g = X_good.mean(axis=0)
    mu_b = X_bad.mean(axis=0)

    # 각 클래스 내의 공분산 행렬(Within-class scatter matrix) 계산
    Sw = (X_good - mu_g).T @ (X_good - mu_g) + (X_bad - mu_b).T @ (X_bad - mu_b)

    # 정규화를 위한 작은 값 추가 후 역행렬 계산
    Sw_reg = Sw + 1e-6 * np.eye(Sw.shape[0])
    Sw_inv = np.linalg.pinv(Sw_reg)

    # 최적의 가중치(w) 계산
    w = Sw_inv @ (mu_g - mu_b)

    # 결정 경계(c) 계산 (두 클래스의 중앙값)
    c = 0.5 * (mu_g + mu_b) @ w

    return {"w": w, "c": c}


def predict_lda(x: np.ndarray, lda_params: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """학습된 LDA 모델로 새로운 데이터(x)의 클래스를 예측합니다."""
    w = np.asarray(lda_params["w"])
    c = float(lda_params["c"])

    # 판별 함수 z 계산
    z = float(x @ w - c)

    # z가 0보다 크거나 같으면 'GOOD', 작으면 'BAD'
    label = "GOOD" if z >= 0 else "BAD"

    # 점수를 확률(0~1)로 변환 (Sigmoid 함수)
    prob = 1.0 / (1.0 + math.exp(-np.clip(z, -25, 25)))

    return label, prob


# --- 4. 메인 실행 로직 (학습/분류) ---


def list_s1p_files(root: str) -> List[str]:
    """주어진 폴더와 하위 폴더에서 모든 .s1p 파일 목록을 찾습니다."""
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith('.s1p'):
                files.append(os.path.join(dp, fn))
    files.sort()
    return files


def build_golden_from_files(files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """여러 .s1p 파일로부터 평균적인 '골든 커브'를 생성합니다."""
    if not files:
        raise ValueError("골든 커브를 생성할 파일이 없습니다.")

    ref_s1p = parse_s1p(files[0])
    ref_freq = ref_s1p.freq_hz

    curves = []
    for path in files:
        sp = parse_s1p(path)
        y_curve = y11_db_norm_z50(sp.s11)
        resampled_curve = resample_to(sp.freq_hz, y_curve, ref_freq)
        curves.append(resampled_curve)

    # 모든 커브의 중간값을 사용하여 골든 커브 생성
    median_curve = np.median(np.vstack(curves), axis=0)
    return ref_freq, median_curve


def run_fit(args) -> None:
    """'fit' 커맨드 실행: 모델 학습"""
    print(f"[학습 시작] Good 샘플: '{args.good_root}', Bad 샘플: '{args.bad_root}'")

    good_files = list_s1p_files(args.good_root)
    bad_files = list_s1p_files(args.bad_root)

    if not good_files or not bad_files:
        raise RuntimeError("Good/Bad 샘플 폴더에 .s1p 파일이 충분하지 않습니다.")

    # 1. Good/Bad 샘플로부터 각각의 골든 커브 생성
    print("골든 커브 생성 중...")
    ref_freq, good_golden_curve = build_golden_from_files(good_files)
    _, bad_golden_curve = build_golden_from_files(bad_files)

    # 2. 각 샘플로부터 특징 추출
    print("특징 추출 중...")
    feature_names = ["r_good", "rmse_good", "r_bad", "rmse_bad"]
    X_good, X_bad = [], []

    for path in good_files:
        sp = parse_s1p(path)
        y_curve = y11_db_norm_z50(sp.s11)
        resampled_curve = resample_to(sp.freq_hz, y_curve, ref_freq)
        features = extract_features(resampled_curve, good_golden_curve, bad_golden_curve)
        X_good.append([features[name] for name in feature_names])

    for path in bad_files:
        sp = parse_s1p(path)
        y_curve = y11_db_norm_z50(sp.s11)
        resampled_curve = resample_to(sp.freq_hz, y_curve, ref_freq)
        features = extract_features(resampled_curve, good_golden_curve, bad_golden_curve)
        X_bad.append([features[name] for name in feature_names])

    # 3. LDA 모델 학습
    print("LDA 모델 학습 중...")
    lda_params = fit_lda(np.vstack(X_good), np.vstack(X_bad))

    # 4. 모델 파일(JSON) 저장
    model_data = {
        "feature_names": feature_names,
        "ref_freq": ref_freq.tolist(),
        "good_golden_curve": good_golden_curve.tolist(),
        "bad_golden_curve": bad_golden_curve.tolist(),
        "lda_params": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in lda_params.items()},
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)

    print(f"[학습 완료] 모델이 '{args.out}' 파일로 저장되었습니다.")


def run_classify(args) -> None:
    """'classify' 커맨드 실행: 모델로 분류"""
    print(f"[분류 시작] 대상 폴더: '{args.root}', 모델 파일: '{args.model}'")

    # 1. 학습된 모델 불러오기
    with open(args.model, 'r', encoding='utf-8') as f:
        model = json.load(f)

    feature_names = model["feature_names"]
    ref_freq = np.asarray(model["ref_freq"])
    good_golden = np.asarray(model["good_golden_curve"])
    bad_golden = np.asarray(model["bad_golden_curve"])
    lda_params = model["lda_params"]

    # 2. 분류 대상 파일 목록 가져오기
    target_files = list_s1p_files(args.root)
    if not target_files:
        print("분류할 .s1p 파일이 없습니다.")
        return

    # 3. 각 파일을 순회하며 분류 수행
    print("\n--- 분류 결과 ---")
    results = []
    for path in target_files:
        try:
            sp = parse_s1p(path)
            y_curve = y11_db_norm_z50(sp.s11)
            resampled_curve = resample_to(sp.freq_hz, y_curve, ref_freq)

            features = extract_features(resampled_curve, good_golden, bad_golden)
            feature_vector = np.array([features[name] for name in feature_names])

            label, probability = predict_lda(feature_vector, lda_params)
            results.append({"path": path, "label": label, "probability": probability})

            print(f"파일: {os.path.basename(path):<30} -> 판정: {label:<8} (확률: {probability:.2f})")

        except Exception as e:  # noqa: BLE001 - 간단한 CLI라 광범위 예외 허용
            print(f"파일 처리 중 에러: {os.path.basename(path)} ({e})")

    # 4. 결과 요약
    pass_count = sum(1 for r in results if r['label'] == 'GOOD')
    fail_count = len(results) - pass_count
    print("\n--- 요약 ---")
    print(f"총 파일 수: {len(results)}")
    print(f"GOOD (양품): {pass_count} 개")
    print(f"BAD (불량): {fail_count} 개")
    print("\n[분류 완료]")


def main() -> None:
    """CLI 인자를 파싱하고 해당 커맨드를 실행합니다."""
    parser = argparse.ArgumentParser(description="간단한 ML 기반 QC 분류기")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 'fit' (학습) 커맨드 설정
    p_fit = subparsers.add_parser('fit', help='Good/Bad 샘플로 ML 모델을 학습합니다.')
    p_fit.add_argument('--good_root', required=True, help='정상(.s1p) 샘플들이 있는 폴더 경로')
    p_fit.add_argument('--bad_root', required=True, help='불량(.s1p) 샘플들이 있는 폴더 경로')
    p_fit.add_argument('--out', default='model.json', help='학습된 모델을 저장할 파일 경로 (기본값: model.json)')
    p_fit.set_defaults(func=run_fit)

    # 'classify' (분류) 커맨드 설정
    p_classify = subparsers.add_parser('classify', help='학습된 모델로 새로운 샘플을 분류합니다.')
    p_classify.add_argument('--root', required=True, help='분류할 .s1p 파일들이 있는 폴더 경로')
    p_classify.add_argument('--model', required=True, help='학습된 모델 파일(.json) 경로')
    p_classify.set_defaults(func=run_classify)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
