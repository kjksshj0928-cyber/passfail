#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_newstart_ml_flat_stage1_v1_0.py
----------------------------------
Fresh start (from scratch) ML-based QC with Stage-1 "FlatnessGuard" as the primary decision:
- Stage-1 (rule): GOOD must be flat (평탄) across ROI; structured ripples/notches (노치/곡률/굴곡) → FAIL.
  (노이즈와는 다른, 지속적인 곡률/왜곡을 식별하도록 설계)
- Stage-2 (optional ML): LDA on engineered features; can be used as fallback or as additional vote.

Key features:
- Wide-cal sweep segments remembered in the model metadata (for resampling reference).
- Domain selectable: Y11(dB) [default] or S11(dB). Y11 computed from S11.
- Robust shape metrics: slope, max deviation, curvature RMS(2차차분), roughness RMS(잔차), peak/notch prominence.
- Lightweight peak detection without SciPy.
- JSON model with mu_good/mu_bad/Sigma_inv, thresholds, feature_names, calibration segments.

Usage
=====
1) Train ML (optional, but recommended):
   python qc_newstart_ml_flat_stage1_v1_0.py train --good <DIR_GOOD> --bad <DIR_BAD> --model model.json --domain Y11 --roi 1e8 8.5e9

2) Classify unlabeled data (Stage-1 FlatnessGuard + optional ML vote):
   python qc_newstart_ml_flat_stage1_v1_0.py classify --root <DIR> --model model.json --out out.csv --domain Y11 --roi 1e8 8.5e9 --use-ml yes

3) Evaluate on labeled (GOOD/BAD) dirs for confusion matrix:
   python qc_newstart_ml_flat_stage1_v1_0.py eval --good <DIR_GOOD> --bad <DIR_BAD> --model model.json --domain Y11 --roi 1e8 8.5e9 --use-ml yes

Notes
=====
- Stage-1 thresholds are tunable via CLI. Start conservative; tighten later.
- "Flatness" means: low global slope, small max deviation, low curvature (2nd diff), low roughness (noise-removed residual),
  and no significant peaks/notches (prominence).
- Y11(dB) is computed as 20*log10(|Y11|*Z0 + eps) where Y11=(1-S)/(1+S)/Z0 and S is complex S11; Z0 from file or 50.
- Files are resampled to a reference grid built from wide-cal segments (stored in model). If model absent, a default is used.
"""

import os
import re
import json
import csv
import math
import argparse
import sys
from datetime import datetime

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

# -----------------------------
# Wide-cal segments (remembered)
# -----------------------------
DEFAULT_WIDECAL = [
    # (f_start_Hz, f_stop_Hz, points)
    (100e6,    1e9,     91),
    (1.0005e9, 8.0e9,   14000),
    (8.01e9,   8.5e9,   50),
]

# -----------------------------
# Helpers
# -----------------------------

def _parse_s1p(path: Path):
    """
    Minimal s1p parser for 1-port Touchstone.
    Supports # line with: HZ|KHz|MHz|GHz, S, RI|MA|DB, R <Z0>
    Returns: (freq_Hz (N,), S11 complex (N,), meta: dict)
    """
    unit = "HZ"; fmt = "RI"; z0 = 50.0
    freqs = []; c1 = []; c2 = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            if s.startswith("#"):
                # Example: "# HZ S RI R 50"
                tokens = s.upper().split()
                if "HZ" in tokens: unit = "HZ"
                elif "KHZ" in tokens: unit = "KHZ"
                elif "MHZ" in tokens: unit = "MHZ"
                elif "GHZ" in tokens: unit = "GHZ"
                if "DB" in tokens: fmt = "DB"
                elif "MA" in tokens: fmt = "MA"
                elif "RI" in tokens: fmt = "RI"
                if "R" in tokens:
                    try:
                        z0 = float(tokens[tokens.index("R")+1])
                    except Exception:
                        z0 = 50.0
                continue
            # data line
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                fval = float(parts[0])
                a = float(parts[1]); b = float(parts[2])
            except Exception:
                continue
            # unit scale
            if unit == "KHZ": fval *= 1e3
            elif unit == "MHZ": fval *= 1e6
            elif unit == "GHZ": fval *= 1e9
            freqs.append(fval); c1.append(a); c2.append(b)
    if not freqs:
        return None, None, {"z0": z0, "unit": unit, "fmt": fmt}

    f = np.array(freqs, dtype=float)
    a = np.array(c1, dtype=float)
    b = np.array(c2, dtype=float)

    # to complex S11
    if fmt == "RI":
        S = a + 1j*b
    elif fmt == "MA":
        # magnitude linear, angle deg
        S = a * np.exp(1j*np.deg2rad(b))
    elif fmt == "DB":
        # magnitude dB, angle deg
        mag = 10**(a/20.0)
        S = mag * np.exp(1j*np.deg2rad(b))
    else:
        S = a + 1j*b
    return f, S, {"z0": z0, "unit": unit, "fmt": fmt}

def _s_to_y_db(S: np.ndarray, z0: float) -> np.ndarray:
    """Compute Y11 in dB-like scale: Y_norm = |((1-S)/(1+S))|, Y_db = 20*log10(Y_norm + eps)"""
    eps = 1e-15
    num = (1.0 - S)
    den = (1.0 + S)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
    mag = np.abs(ratio) + eps
    Y_db = 20.0*np.log10(mag)
    return Y_db

def _s_to_s_db(S: np.ndarray) -> np.ndarray:
    eps = 1e-15
    mag = np.abs(S) + eps
    return 20.0*np.log10(mag)

def build_ref_grid(segments: List[Tuple[float,float,int]]) -> np.ndarray:
    grids = []
    for f0,f1,n in segments:
        if n <= 1 or f1 <= f0:
            continue
        grids.append(np.linspace(f0, f1, int(n)))
    if not grids:
        return None
    g = np.unique(np.concatenate(grids))
    return g

def resample_to_ref(f: np.ndarray, y: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear resampling to reference grid within overlap range. Returns (f_ref, y_ref) where f_ref is subset of ref within [fmin,fmax]."""
    fmin, fmax = float(f.min()), float(f.max())
    sel = (ref >= fmin) & (ref <= fmax)
    if not np.any(sel):
        return None, None
    f_target = ref[sel]
    y_interp = np.interp(f_target, f, y)
    return f_target, y_interp

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win == 1:
        return x.copy()
    c = np.cumsum(np.insert(x,0,0.0))
    out = (c[win:] - c[:-win]) / float(win)
    # pad to length
    pad_left = win//2
    pad_right = len(x) - len(out) - pad_left
    return np.pad(out, (pad_left, pad_right), mode='edge')

def diff2(x: np.ndarray) -> np.ndarray:
    # second finite difference with edge padding
    d2 = np.zeros_like(x)
    d2[1:-1] = x[2:] - 2*x[1:-1] + x[:-2]
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2

def linear_fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    # returns absolute slope (y per Hz); convert to dB/GHz in features for interpretability
    xmean = x.mean(); ymean = y.mean()
    num = ((x - xmean)*(y - ymean)).sum()
    den = ((x - xmean)**2).sum() + 1e-30
    slope = num/den
    return abs(slope)

def find_peaks_prominence(y: np.ndarray, min_prom: float, min_dist: int=20) -> Tuple[int, float]:
    """
    Simple prominence-like detector for positive peaks on y.
    Steps:
      1) find local maxima y[i-1] < y[i] >= y[i+1]
      2) estimate prominence by difference from nearest higher "shoulders"
    Returns: (count, max_prominence)
    """
    N = len(y)
    if N < 3: return 0, 0.0
    peaks = []
    for i in range(1, N-1):
        if y[i] >= y[i-1] and y[i] >= y[i+1]:
            peaks.append(i)
    # enforce min_dist
    filtered = []
    last = -min_dist
    for idx in peaks:
        if idx - last >= min_dist:
            filtered.append(idx); last = idx
    # prominence estimate
    prom_vals = []
    for idx in filtered:
        # left min
        left_min = y[idx]
        j = idx
        while j > 0 and y[j] <= y[j-1]:
            j -= 1
        left_min = min(left_min, y[j])
        # right min
        right_min = y[idx]
        j = idx
        while j < N-1 and y[j] <= y[j+1]:
            j += 1
        right_min = min(right_min, y[j])
        baseline = max(left_min, right_min)
        prom = max(0.0, y[idx] - baseline)
        prom_vals.append(prom)
    prom_vals = np.array(prom_vals) if prom_vals else np.array([])
    count = int((prom_vals >= min_prom).sum()) if prom_vals.size else 0
    mx = float(prom_vals.max()) if prom_vals.size else 0.0
    return count, mx

def find_notches_prominence(y: np.ndarray, min_prom: float, min_dist: int=20) -> Tuple[int, float]:
    # Notches are peaks on (-y)
    c, m = find_peaks_prominence(-y, min_prom=min_prom, min_dist=min_dist)
    return c, m

# -----------------------------
# Feature engineering
# -----------------------------

def compute_flatness_features(f_Hz: np.ndarray, y_db: np.ndarray) -> Dict[str, float]:
    # normalize frequency to GHz for numerics
    f_GHz = f_Hz / 1e9
    # Trend vs residual: two-window smoothing
    trend = moving_average(y_db, win=max(5, len(y_db)//200))  # long-ish window
    residual = y_db - trend
    # Curvature on trend (structured bends, not noise)
    curv = diff2(trend)
    curvature_rms = float(np.sqrt(np.mean(curv**2)))
    # Roughness on residual (noise-like)
    roughness_rms = float(np.sqrt(np.mean(residual**2)))
    # Global slope (dB per GHz)
    slope_abs = linear_fit_slope(f_Hz, trend) * 1e9  # dB/GHz scale
    # Max absolute deviation from median (flatness)
    med = float(np.median(trend))
    max_dev = float(np.max(np.abs(trend - med)))
    # Peaks/Notches prominence (structured bumps or bites)
    pk_count, pk_prom = find_peaks_prominence(trend, min_prom=0.3, min_dist=max(5, len(trend)//500))
    nt_count, nt_prom = find_notches_prominence(trend, min_prom=0.3, min_dist=max(5, len(trend)//500))

    return {
        "curvature_rms": curvature_rms,
        "roughness_rms": roughness_rms,
        "slope_abs_db_per_GHz": slope_abs,
        "max_dev_db": max_dev,
        "peak_count": float(pk_count),
        "peak_prom_db": pk_prom,
        "notch_count": float(nt_count),
        "notch_prom_db": nt_prom,
        "median_db": med,
        "p2p_db": float(y_db.max() - y_db.min()),
    }

FEATURE_ORDER = [
    "curvature_rms",
    "roughness_rms",
    "slope_abs_db_per_GHz",
    "max_dev_db",
    "peak_count",
    "peak_prom_db",
    "notch_count",
    "notch_prom_db",
    "p2p_db",
]

# -----------------------------
# Stage-1 FlatnessGuard (rule)
# -----------------------------

def stage1_flatness_decision(feat: Dict[str,float], th: Dict[str, float]) -> Tuple[str, str]:
    """
    Returns (stage1_label, note)
    stage1_label: PASS if flat within thresholds, else FAIL
    th: dict with keys:
        curvature_rms_max, roughness_rms_max, slope_abs_max, max_dev_max,
        pk_prom_max, nt_prom_max, pk_count_max, nt_count_max, p2p_max
    """
    conds = []
    notes = []

    def ok(val, lim, name):
        c = val <= lim
        notes.append(f"{name}={val:.4g} <= {lim:.4g} {'OK' if c else 'NG'}")
        return c

    conds.append(ok(feat["curvature_rms"], th["curvature_rms_max"], "curv_rms"))
    conds.append(ok(feat["roughness_rms"], th["roughness_rms_max"], "rough_rms"))
    conds.append(ok(feat["slope_abs_db_per_GHz"], th["slope_abs_max"], "slope"))
    conds.append(ok(feat["max_dev_db"], th["max_dev_max"], "max_dev"))
    conds.append(ok(feat["peak_prom_db"], th["pk_prom_max"], "pk_prom"))
    conds.append(ok(feat["notch_prom_db"], th["nt_prom_max"], "nt_prom"))
    conds.append(ok(feat["peak_count"], th["pk_count_max"], "pk_cnt"))
    conds.append(ok(feat["notch_count"], th["nt_count_max"], "nt_cnt"))
    conds.append(ok(feat["p2p_db"], th["p2p_max"], "p2p"))

    label = "PASS" if all(conds) else "FAIL"
    return label, "; ".join(notes)

DEFAULT_STAGE1_TH = {
    "curvature_rms_max": 0.02,    # dB (2nd diff RMS on trend)
    "roughness_rms_max": 0.15,    # dB (noise-like residual)
    "slope_abs_max": 0.5,         # dB/GHz
    "max_dev_max": 0.8,           # dB
    "pk_prom_max": 0.6,           # dB
    "nt_prom_max": 0.6,           # dB
    "pk_count_max": 1.0,          # count
    "nt_count_max": 1.0,          # count
    "p2p_max": 1.2,               # dB
}

# -----------------------------
# Simple LDA (shared covariance)
# -----------------------------

class LDA:
    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)
        self.mu_good = None
        self.mu_bad = None
        self.Sigma_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # y: 1 for GOOD, 0 for BAD
        Xg = X[y == 1]
        Xb = X[y == 0]
        mu_g = Xg.mean(axis=0)
        mu_b = Xb.mean(axis=0)
        # pooled covariance
        Sg = np.cov(Xg, rowvar=False)
        Sb = np.cov(Xb, rowvar=False)
        # regularize a bit
        Sigma = 0.5*(Sg + Sb) + np.eye(X.shape[1])*1e-6
        Sigma_inv = np.linalg.pinv(Sigma)
        self.mu_good, self.mu_bad, self.Sigma_inv = mu_g, mu_b, Sigma_inv

    def score(self, x: np.ndarray) -> float:
        # Linear discriminant: s = w^T x + c; we convert to pseudo-prob via logistic
        w = self.Sigma_inv @ (self.mu_good - self.mu_bad)
        c = -0.5*(self.mu_good @ self.Sigma_inv @ self.mu_good - self.mu_bad @ self.Sigma_inv @ self.mu_bad)
        s = float(w @ x + c)
        # squashed
        p = 1.0/(1.0 + math.exp(-s))
        return p  # ~prob(GOOD)

    def to_json(self):
        return {
            "feature_names": self.feature_names,
            "mu_good": self.mu_good.tolist() if self.mu_good is not None else None,
            "mu_bad": self.mu_bad.tolist() if self.mu_bad is not None else None,
            "Sigma_inv": self.Sigma_inv.tolist() if self.Sigma_inv is not None else None,
        }

    @staticmethod
    def from_json(d):
        lda = LDA(d["feature_names"])
        lda.mu_good = np.array(d["mu_good"]) if d["mu_good"] is not None else None
        lda.mu_bad = np.array(d["mu_bad"]) if d["mu_bad"] is not None else None
        lda.Sigma_inv = np.array(d["Sigma_inv"]) if d["Sigma_inv"] is not None else None
        return lda

# -----------------------------
# Pipeline
# -----------------------------

def to_domain_db(f: np.ndarray, S: np.ndarray, meta: dict, domain: str) -> np.ndarray:
    domain = (domain or "Y11").upper()
    if domain == "Y11":
        return _s_to_y_db(S, meta.get("z0", 50.0))
    else:
        return _s_to_s_db(S)

def load_s1p_features(path: Path, domain: str, ref_grid: np.ndarray, roi: Tuple[float,float]) -> Tuple[Dict[str,float], int]:
    f, S, meta = _parse_s1p(path)
    if f is None or S is None or len(f) < 5:
        return None, 0
    y_db = to_domain_db(f, S, meta, domain=domain)
    # ROI crop
    if roi is not None:
        fmin, fmax = roi
        sel = (f >= fmin) & (f <= fmax)
        if not np.any(sel):
            return None, 0
        f = f[sel]; y_db = y_db[sel]
    # resample
    if ref_grid is not None:
        f2, y2 = resample_to_ref(f, y_db, ref_grid)
        if f2 is None or len(f2) < 50:
            return None, 0
        f, y_db = f2, y2
    # features
    feat = compute_flatness_features(f, y_db)
    return feat, len(f)

def scan_s1p_files(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*.s1p"):
        if p.is_file():
            paths.append(p)
    return sorted(paths)

def feats_to_matrix(feats: List[Dict[str,float]]) -> np.ndarray:
    X = np.array([[f[k] for k in FEATURE_ORDER] for f in feats], dtype=float)
    return X

def save_model_json(model_path: Path, lda: LDA, ref_segments, stage1_th, meta_extra=None):
    d = {
        "created": datetime.utcnow().isoformat(),
        "widecal_segments": ref_segments,
        "stage1_thresholds": stage1_th,
        "lda": lda.to_json() if lda is not None else None,
        "meta": meta_extra or {},
    }
    model_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

def load_model_json(model_path: Path):
    d = json.loads(model_path.read_text(encoding="utf-8"))
    segments = d.get("widecal_segments", DEFAULT_WIDECAL)
    ref_grid = build_ref_grid(segments)
    th = d.get("stage1_thresholds", DEFAULT_STAGE1_TH)
    lda = LDA.from_json(d["lda"]) if d.get("lda") else None
    meta = d.get("meta", {})
    return ref_grid, th, lda, segments, meta

# -----------------------------
# CLI Commands
# -----------------------------

def cmd_train(args):
    good_dir = Path(args.good); bad_dir = Path(args.bad)
    model_path = Path(args.model)
    domain = args.domain.upper()
    roi = (args.roi[0], args.roi[1]) if args.roi else None
    # reference grid from default wide-cal or user overrides
    segments = DEFAULT_WIDECAL
    ref_grid = build_ref_grid(segments)

    X_list = []; y_list = []

    for label, root in [(1, good_dir), (0, bad_dir)]:
        files = scan_s1p_files(root)
        for fp in files:
            feat, npts = load_s1p_features(fp, domain, ref_grid, roi)
            if feat is None:
                continue
            X_list.append([feat[k] for k in FEATURE_ORDER])
            y_list.append(label)

    if not X_list:
        print("[ERROR] No features extracted. Check paths/ROI.")
        return 1

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    lda = LDA(feature_names=FEATURE_ORDER)
    lda.fit(X, y)

    save_model_json(model_path, lda, segments, DEFAULT_STAGE1_TH, meta_extra={"domain": domain, "roi": roi})
    print(f"[OK] Model saved to: {model_path}")
    print(f"[INFO] Training samples: GOOD={int((y==1).sum())}, BAD={int((y==0).sum())}")
    return 0

def _stage1_and_ml_vote(feat: Dict[str,float], th: dict, lda: LDA, use_ml: bool, ml_thresh: float):
    s1_label, s1_note = stage1_flatness_decision(feat, th)
    ml_p = None; ml_label = None
    if use_ml and lda is not None:
        x = np.array([feat[k] for k in FEATURE_ORDER], dtype=float)
        ml_p = lda.score(x)  # prob GOOD
        ml_label = "PASS" if ml_p >= ml_thresh else "FAIL"
    # Decision policy (adjustable):
    # - Strict: Stage-1 must PASS AND (if ML used) ML must PASS → PASS, else FAIL.
    if use_ml and lda is not None:
        final = "PASS" if (s1_label == "PASS" and ml_label == "PASS") else "FAIL"
    else:
        final = s1_label
    return final, s1_label, s1_note, (ml_label, ml_p)

def cmd_classify(args):
    root = Path(args.root)
    model_path = Path(args.model) if args.model else None
    out_csv = Path(args.out) if args.out else None
    domain = args.domain.upper()
    roi = (args.roi[0], args.roi[1]) if args.roi else None
    use_ml = (args.use_ml.lower() in ["1","y","yes","true"]) if args.use_ml else False
    ml_thresh = float(args.ml_thresh)

    # Load model or fallback to default ref/thresholds
    if model_path and model_path.exists():
        ref_grid, th, lda, segments, meta = load_model_json(model_path)
        if roi is None:
            roi = meta.get("roi", None)
        if args.domain is None and "domain" in meta:
            domain = meta["domain"]
    else:
        ref_grid = build_ref_grid(DEFAULT_WIDECAL)
        th = DEFAULT_STAGE1_TH.copy()
        lda = None

    files = scan_s1p_files(root)
    rows = []
    for fp in files:
        feat, npts = load_s1p_features(fp, domain, ref_grid, roi)
        if feat is None:
            continue
        final, s1_label, s1_note, (ml_label, ml_p) = _stage1_and_ml_vote(feat, th, lda, use_ml, ml_thresh)
        row = {
            "file": str(fp),
            "final": final,
            "stage1": s1_label,
            "stage1_note": s1_note,
            "ml_label": ml_label if ml_label is not None else "",
            "ml_prob_good": f"{ml_p:.4f}" if ml_p is not None else "",
        }
        # append features for debugging
        for k in FEATURE_ORDER:
            row[k] = feat[k]
        rows.append(row)

    # Write CSV
    if out_csv:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file","final"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[OK] Wrote: {out_csv} ({len(rows)} rows)")
    else:
        # print preview
        for r in rows[:10]:
            print(r)
        print(f"[INFO] Total rows: {len(rows)}")
    return 0

def cmd_eval(args):
    good_dir = Path(args.good); bad_dir = Path(args.bad)
    model_path = Path(args.model) if args.model else None
    domain = args.domain.upper()
    roi = (args.roi[0], args.roi[1]) if args.roi else None
    use_ml = (args.use_ml.lower() in ["1","y","yes","true"]) if args.use_ml else False
    ml_thresh = float(args.ml_thresh)

    if model_path and model_path.exists():
        ref_grid, th, lda, segments, meta = load_model_json(model_path)
        if roi is None:
            roi = meta.get("roi", None)
        if args.domain is None and "domain" in meta:
            domain = meta["domain"]
    else:
        ref_grid = build_ref_grid(DEFAULT_WIDECAL)
        th = DEFAULT_STAGE1_TH.copy()
        lda = None

    y_true = []; y_pred = []
    details = []

    for label, root in [(1, good_dir), (0, bad_dir)]:
        files = scan_s1p_files(root)
        for fp in files:
            feat, npts = load_s1p_features(fp, domain, ref_grid, roi)
            if feat is None:
                continue
            final, s1_label, s1_note, (ml_label, ml_p) = _stage1_and_ml_vote(feat, th, lda, use_ml, ml_thresh)
            y_true.append(label)
            y_pred.append(1 if final=="PASS" else 0)
            details.append((str(fp), label, final, s1_label, ml_label, ml_p))

    # Confusion matrix
    import numpy as _np
    y_true = _np.array(y_true, dtype=int); y_pred = _np.array(y_pred, dtype=int)
    TP = int(((y_true==1)&(y_pred==1)).sum())
    TN = int(((y_true==0)&(y_pred==0)).sum())
    FP = int(((y_true==1)&(y_pred==0)).sum())
    FN = int(((y_true==0)&(y_pred==1)).sum())
    print(f"[EVAL] TP={TP} TN={TN} FP={FP} FN={FN}")
    total = max(1, len(y_true))
    acc = (TP+TN)/total
    print(f"[EVAL] Accuracy={acc:.4f}, FP_rate={FP/max(1,(TP+FP)):.4f}, FN_rate={FN/max(1,(TN+FN)):.4f}")
    return 0

def main():
    ap = argparse.ArgumentParser(description="Fresh ML QC with Stage-1 FlatnessGuard (Y11/S11).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common opts builder
    def add_common(p):
        p.add_argument("--domain", default="Y11", help="Y11 or S11 (default=Y11)")
        p.add_argument("--roi", nargs=2, type=float, default=None, metavar=("F_MIN_HZ","F_MAX_HZ"), help="ROI frequency range in Hz")
        return p

    # Train
    p_tr = sub.add_parser("train", help="Train LDA on GOOD/BAD folders")
    p_tr.add_argument("--good", required=True, help="GOOD dir")
    p_tr.add_argument("--bad", required=True, help="BAD dir")
    p_tr.add_argument("--model", required=True, help="Output model.json")
    add_common(p_tr)

    # Classify
    p_cl = sub.add_parser("classify", help="Classify unlabeled s1p files")
    p_cl.add_argument("--root", required=True, help="Root dir to scan")
    p_cl.add_argument("--model", required=False, help="Model.json (optional for Stage-1 only)")
    p_cl.add_argument("--out", required=False, help="Output CSV path")
    p_cl.add_argument("--use-ml", default="no", help="yes/no; if yes, Stage-1 AND ML must pass")
    p_cl.add_argument("--ml-thresh", default=0.5, type=float, help="Threshold on ML prob-good for PASS")
    add_common(p_cl)

    # Eval
    p_ev = sub.add_parser("eval", help="Evaluate on labeled GOOD/BAD dirs")
    p_ev.add_argument("--good", required=True, help="GOOD dir")
    p_ev.add_argument("--bad", required=True, help="BAD dir")
    p_ev.add_argument("--model", required=False, help="Model.json (optional for Stage-1 only)")
    p_ev.add_argument("--use-ml", default="no", help="yes/no; if yes, Stage-1 AND ML must pass")
    p_ev.add_argument("--ml-thresh", default=0.5, type=float, help="Threshold on ML prob-good for PASS")
    add_common(p_ev)

    # Stage-1 thresholds override
    ap.add_argument("--th-curv", type=float, default=DEFAULT_STAGE1_TH["curvature_rms_max"])
    ap.add_argument("--th-rough", type=float, default=DEFAULT_STAGE1_TH["roughness_rms_max"])
    ap.add_argument("--th-slope", type=float, default=DEFAULT_STAGE1_TH["slope_abs_max"])
    ap.add_argument("--th-maxdev", type=float, default=DEFAULT_STAGE1_TH["max_dev_max"])
    ap.add_argument("--th-pkprom", type=float, default=DEFAULT_STAGE1_TH["pk_prom_max"])
    ap.add_argument("--th-ntprom", type=float, default=DEFAULT_STAGE1_TH["nt_prom_max"])
    ap.add_argument("--th-pkcnt", type=float, default=DEFAULT_STAGE1_TH["pk_count_max"])
    ap.add_argument("--th-ntcnt", type=float, default=DEFAULT_STAGE1_TH["nt_count_max"])
    ap.add_argument("--th-p2p", type=float, default=DEFAULT_STAGE1_TH["p2p_max"])

    args = ap.parse_args()

    # apply overrides to defaults (for runtime)
    DEFAULT_STAGE1_TH["curvature_rms_max"] = args.th_curv
    DEFAULT_STAGE1_TH["roughness_rms_max"] = args.th_rough
    DEFAULT_STAGE1_TH["slope_abs_max"] = args.th_slope
    DEFAULT_STAGE1_TH["max_dev_max"] = args.th_maxdev
    DEFAULT_STAGE1_TH["pk_prom_max"] = args.th_pkprom
    DEFAULT_STAGE1_TH["nt_prom_max"] = args.th_ntprom
    DEFAULT_STAGE1_TH["pk_count_max"] = args.th_pkcnt
    DEFAULT_STAGE1_TH["nt_count_max"] = args.th_ntcnt
    DEFAULT_STAGE1_TH["p2p_max"] = args.th_p2p

    if args.cmd == "train":
        sys.exit(cmd_train(args))
    elif args.cmd == "classify":
        sys.exit(cmd_classify(args))
    elif args.cmd == "eval":
        sys.exit(cmd_eval(args))

if __name__ == "__main__":
    main()
