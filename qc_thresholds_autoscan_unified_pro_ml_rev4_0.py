#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_thresholds_autoscan_unified_pro_ml_rev4_0.py
-----------------------------------------------
STRICT superset of v3.8 (원코드 삭제/수정 없음, 기능만 추가).

NEW (rev4.0):
1) Labeled ML (Good/Bad) 학습 + 적용
   - GOOD/BAD 폴더에서 피처를 추출하여 LDA(선형 판별) 학습.
   - 모델(JSON)에는 mu_good, mu_bad, Sigma_inv, w, c, feature_names,
     ROI, domain, compare_mode, good/bad goldens, **S11 good golden + CC 임계치** 포함.
   - 피처: [r_good, rmse_good, r_bad, rmse_bad, noise_rms_db, peak_prom_db, fwhm_mhz]
2) 2-Stage 기준
   - 1차: 엄격(r_min, rmse_max)
   - 2차: flexible (완화배수, 기본 1.25) — ML이 GOOD이면 FLEX 가중.
3) S11 Cross-check (교차검증)
   - domain이 Y11일 때 S11에서도 r/rmse가 심하게 벗어나면 WARN/FAIL.
   - GOOD 학습셋으로 만든 **S11 golden**과 MAD 자동 임계치 사용(모델 JSON에 저장).
4) GUI 확장
   - 좌측 "ML / Cross-check" 패널: GOOD/BAD 폴더 선택, Fit&Save, Load, Use-ML(WARN/FAIL),
     Two-Stage 사용(완화배수), S11 Cross-check(WARN/FAIL, 완화배수) 등.
5) CSV 확장
   - stage(STRICT/FLEX/NONE), ml_label(GOOD/BAD/NA), ml_score, s11_cc(OK/WARN/FAIL) 추가.

의존성: numpy, tkinter, matplotlib  (scikit-learn 불필요; LDA는 내장구현)
"""

import os, sys, csv, json, argparse, platform, subprocess, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

try:
    import numpy as np
except Exception:
    print("[에러] numpy 필요. pip install numpy")
    raise

# ============================ S1P I/O ============================

@dataclass
class S1P:
    freq_hz: np.ndarray
    s11: np.ndarray
    z0: float
    unit: str
    fmt: str

def _unit_scale(u: str) -> float:
    u = u.upper()
    return {'HZ':1.0,'KHZ':1e3,'MHZ':1e6,'GHZ':1e9}.get(u,1.0)

def parse_s1p(path: str) -> 'S1P':
    unit, fmt, z0 = 'HZ', 'RI', 50.0
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('!'):
                continue
            if line.startswith('#'):
                toks = line[1:].split(); tU = [t.upper() for t in toks]
                for cand in ('HZ','KHZ','MHZ','GHZ'):
                    if cand in tU: unit=cand; break
                if 'DB' in tU: fmt='DB'
                elif 'MA' in tU: fmt='MA'
                else: fmt='RI'
                if 'R' in tU:
                    try: z0 = float(toks[tU.index('R')+1])
                    except Exception: z0 = 50.0
                continue
            parts = line.split()
            if len(parts) < 3: continue
            try:
                fval = float(parts[0]); a = float(parts[1]); b = float(parts[2])
                data.append((fval,a,b))
            except Exception:
                continue
    if not data: raise ValueError(f"데이터 없음: {path}")
    arr = np.array(data, dtype=float)
    freq = arr[:,0] * _unit_scale(unit)
    if fmt=='RI':
        s11 = arr[:,1] + 1j*arr[:,2]
    elif fmt=='MA':
        mag, ang = arr[:,1], np.deg2rad(arr[:,2])
        s11 = mag * (np.cos(ang)+1j*np.sin(ang))
    else:
        mag = 10.0**(arr[:,1]/20.0); ang = np.deg2rad(arr[:,2])
        s11 = mag * (np.cos(ang)+1j*np.sin(ang))
    idx = np.argsort(freq)
    return S1P(freq_hz=freq[idx], s11=s11[idx], z0=z0, unit=unit, fmt=fmt)

# ============================ Domain Curves ============================

def resample_to(x_src: np.ndarray, y_src: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    return np.interp(x_ref, x_src, y_src)

def s11_db(s: np.ndarray) -> np.ndarray:
    mag = np.clip(np.abs(s), 1e-12, None)
    return 20.0*np.log10(mag)

def y11_db_norm_z50(s: np.ndarray) -> np.ndarray:
    eps = 1e-15
    num = (1.0 - s); den = (1.0 + s)
    den = np.where(np.abs(den) < eps, eps, den)
    y_norm = np.abs(num/den)
    y_norm = np.clip(y_norm, 1e-12, None)
    return 20.0*np.log10(y_norm)

def curve_in_domain(sp: S1P, domain: str) -> np.ndarray:
    if domain == 'S11_DB': return s11_db(sp.s11)
    elif domain == 'Y11_DB': return y11_db_norm_z50(sp.s11)
    else: raise ValueError("unknown domain: " + domain)

# ============================ Core Metrics ============================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b=np.asarray(b)
    if a.size!=b.size or a.size<3: return 0.0
    av=a-a.mean(); bv=b-b.mean()
    denom = np.linalg.norm(av)*np.linalg.norm(bv)
    if denom<=1e-18: return 0.0
    return float(np.dot(av,bv)/denom)

@dataclass
class Metrics:
    r: float
    rmse_db: float

def _affine_fit_rmse(s: np.ndarray, g: np.ndarray) -> float:
    s = np.asarray(s); g = np.asarray(g)
    if s.size!=g.size or s.size<2: return float('inf')
    S = np.vstack([s, np.ones_like(s)]).T
    try:
        beta, *_ = np.linalg.lstsq(S, g, rcond=None)
        a = float(beta[0]); b = float(beta[1])
        pred = a*s + b
        rmse = float(np.sqrt(np.mean((pred-g)**2)))
        return rmse
    except Exception:
        return float('inf')

def compute_metrics(sample: S1P, golden_freq: np.ndarray, golden_curve: np.ndarray,
                    fmin: Optional[float], fmax: Optional[float], domain: str,
                    compare_mode: str='RAW') -> Metrics:
    s_curve = curve_in_domain(sample, domain)
    s_curve_res = resample_to(sample.freq_hz, s_curve, golden_freq)
    f = golden_freq
    mask = np.ones_like(f, dtype=bool)
    if fmin is not None: mask &= (f>=fmin)
    if fmax is not None: mask &= (f<=fmax)
    s_sel = s_curve_res[mask]; g_sel = golden_curve[mask]
    if s_sel.size==0: return Metrics(r=0.0, rmse_db=float('inf'))
    r = pearson_r(s_sel, g_sel)
    if compare_mode=='AFFINE': rmse = _affine_fit_rmse(s_sel, g_sel)
    else: rmse = float(np.sqrt(np.mean((s_sel-g_sel)**2)))
    return Metrics(r=r, rmse_db=rmse)

def auto_thresholds_mad(metrics: List[Metrics], k_sigma: float=3.0) -> Tuple[float,float]:
    if not metrics: return 0.98, 1.0
    r_arr = np.array([m.r for m in metrics], float)
    e_arr = np.array([m.rmse_db for m in metrics], float)
    r_med = np.median(r_arr); e_med = np.median(e_arr)
    r_mad = np.median(np.abs(r_arr-r_med)); e_mad = np.median(np.abs(e_arr-e_med))
    r_min = r_med - k_sigma*r_mad
    rmse_max = e_med + k_sigma*e_mad
    r_min = float(np.clip(r_min, 0.7, 0.9999))
    rmse_max = float(np.clip(rmse_max, 0.1, 15.0))
    return r_min, rmse_max

def auto_thresholds_quantile(metrics: List[Metrics], pass_rate: float=0.95) -> Tuple[float,float]:
    if not metrics: return 0.98, 1.0
    r_arr = np.array([m.r for m in metrics], float)
    e_arr = np.array([m.rmse_db for m in metrics], float)
    p = float(np.clip(pass_rate, 0.5, 0.999))
    r_min = float(np.quantile(r_arr, 1.0 - p))
    rmse_max = float(np.quantile(e_arr, p))
    r_min = float(np.clip(r_min, 0.7, 0.9999))
    rmse_max = float(np.clip(rmse_max, 0.1, 15.0))
    return r_min, rmse_max

# ============================ Noise / Shape helpers ============================

def _movavg(y: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1: return y.copy()
    w = np.ones(int(win), dtype=float)
    w /= w.sum()
    return np.convolve(y, w, mode='same')

def _mad_std(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)

def noise_rms_db(y_db: np.ndarray, smooth_win: int=31) -> float:
    base = _movavg(y_db, smooth_win)
    resid = y_db - base
    return _mad_std(resid)

def _peak_index(y: np.ndarray, domain: str) -> int:
    # Y11_DB → 최대값, S11_DB → 최소값(깊은 지점)
    return int(np.argmax(y)) if domain == 'Y11_DB' else int(np.argmin(y))

def peak_prominence_and_fwhm(freq_hz: np.ndarray, y_db: np.ndarray, domain: str,
                             smooth_win: int=31) -> Tuple[float, float]:
    """ prom: 중앙값 대비 피크 높이/깊이, FWHM: 절반 레벨 좌우 교차 거리 """
    y_s = _movavg(y_db, smooth_win)
    idx = _peak_index(y_s, domain)
    peak = y_s[idx]
    base = float(np.median(y_s))
    if domain == 'Y11_DB':
        prom = float(max(0.0, peak - base))
        half = peak - prom/2.0
        i = idx
        while i>0 and y_s[i] > half: i -= 1
        if i==0: left_f = freq_hz[0]
        else:
            x1,x2 = freq_hz[i], freq_hz[i+1]; y1,y2 = y_s[i], y_s[i+1]
            t = (half - y1)/max(1e-12, (y2-y1))
            left_f = x1 + np.clip(t,0,1)*(x2-x1)
        j = idx
        while j < len(y_s)-1 and y_s[j] > half: j += 1
        if j==len(y_s)-1: right_f = freq_hz[-1]
        else:
            x1,x2 = freq_hz[j-1], freq_hz[j]; y1,y2 = y_s[j-1], y_s[j]
            t = (half - y1)/max(1e-12, (y2-y1))
            right_f = x1 + np.clip(t,0,1)*(x2-x1)
        fwhm_hz = float(max(0.0, right_f - left_f))
    else:
        prom = float(max(0.0, base - peak))
        half = peak + prom/2.0
        i = idx
        while i>0 and y_s[i] < half: i -= 1
        if i==0: left_f = freq_hz[0]
        else:
            x1,x2 = freq_hz[i], freq_hz[i+1]; y1,y2 = y_s[i], y_s[i+1]
            t = (half - y1)/max(1e-12, (y2-y1))
            left_f = x1 + np.clip(t,0,1)*(x2-x1)
        j = idx
        while j < len(y_s)-1 and y_s[j] < half: j += 1
        if j==len(y_s)-1: right_f = freq_hz[-1]
        else:
            x1,x2 = freq_hz[j-1], freq_hz[j]; y1,y2 = y_s[j-1], y_s[j]
            t = (half - y1)/max(1e-12, (y2-y1))
            right_f = x1 + np.clip(t,0,1)*(x2-x1)
        fwhm_hz = float(max(0.0, right_f - left_f))
    return prom, fwhm_hz/1e6  # (dB, MHz)

# ============================ Golden Builders ============================

def build_golden_auto(files: List[str], domain: str) -> Tuple[np.ndarray, np.ndarray]:
    ref = parse_s1p(files[0])
    mats = []
    for p in files:
        sp = parse_s1p(p)
        curve = curve_in_domain(sp, domain)
        mats.append(resample_to(sp.freq_hz, curve, ref.freq_hz))
    med_curve = np.median(np.vstack(mats), axis=0)
    return ref.freq_hz, med_curve

def build_golden_single(path: str, domain: str) -> Tuple[np.ndarray, np.ndarray]:
    g = parse_s1p(path); return g.freq_hz, curve_in_domain(g, domain)

def build_golden_folder(folder: str, domain: str) -> Tuple[np.ndarray, np.ndarray]:
    files = []
    for dp,_,fns in os.walk(folder):
        for fn in fns:
            if fn.lower().endswith('.s1p'): files.append(os.path.join(dp, fn))
    files.sort()
    if not files: raise RuntimeError("Golden 폴더 내 s1p를 찾지 못했습니다.")
    return build_golden_auto(files, domain)

# ============================ ML: Features / LDA ============================

def _compute_basic_features(sp: S1P, g_freq_good, g_curve_good, g_freq_bad, g_curve_bad,
                            fmin, fmax, domain, compare_mode, smooth_win=31) -> np.ndarray:
    y = curve_in_domain(sp, domain)
    # good/bad metrics (각 골든 그리드에서 계산)
    m_g = compute_metrics(sp, g_freq_good, g_curve_good, fmin, fmax, domain, compare_mode)
    m_b = compute_metrics(sp, g_freq_bad,  g_curve_bad,  fmin, fmax, domain, compare_mode)
    # noise/shape on good grid with ROI
    y_good = resample_to(sp.freq_hz, y, g_freq_good)
    mask = np.ones_like(g_freq_good, dtype=bool)
    if fmin is not None: mask &= (g_freq_good>=fmin)
    if fmax is not None: mask &= (g_freq_good<=fmax)
    yf = y_good[mask]; ff = g_freq_good[mask]
    if yf.size < 5:
        nr, pr, fw = 0.0, 0.0, 0.0
    else:
        nr = noise_rms_db(yf, smooth_win)
        pr, fw = peak_prominence_and_fwhm(ff, yf, domain, smooth_win)
    return np.array([m_g.r, m_g.rmse_db, m_b.r, m_b.rmse_db, nr, pr, fw], float)

def _fit_lda(X_good: np.ndarray, X_bad: np.ndarray) -> Dict[str, np.ndarray]:
    mu_g = X_good.mean(axis=0); mu_b = X_bad.mean(axis=0)
    Xg_c = X_good - mu_g; Xb_c = X_bad - mu_b
    Sw = (Xg_c.T@Xg_c + Xb_c.T@Xb_c) / max(1, (len(X_good)+len(X_bad)-2))
    lam = 1e-6
    Sw_reg = Sw + lam*np.eye(Sw.shape[0])
    Sig_inv = np.linalg.pinv(Sw_reg)
    w = Sig_inv @ (mu_g - mu_b)
    c = 0.5 * (mu_g + mu_b) @ w  # equal priors
    return {"mu_good": mu_g, "mu_bad": mu_b, "Sigma_inv": Sig_inv, "w": w, "c": float(c)}

def _lda_score(x: np.ndarray, w: np.ndarray, c: float) -> float:
    return float(x @ w - c)

def _sigmoid(z: float) -> float:
    try: return 1.0/(1.0+math.exp(-np.clip(z, -50, 50)))
    except Exception: return 0.5

def list_s1p_files(root: str, regex: Optional[str]=None) -> List[str]:
    import re
    ret = []
    pat = re.compile(regex) if (regex and regex.strip()) else None
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith('.s1p'):
                full = os.path.join(dp, fn)
                if pat and not pat.search(full): continue
                ret.append(full)
    ret.sort()
    return ret

def fit_ml_model(good_root: str, bad_root: str, domain: str, compare_mode: str,
                 fmin: Optional[float], fmax: Optional[float], regex: Optional[str]=None,
                 smooth_win:int=31) -> Dict:
    good_files = list_s1p_files(good_root, regex)
    bad_files  = list_s1p_files(bad_root, regex)
    if not good_files or not bad_files:
        raise RuntimeError("GOOD/BAD s1p를 충분히 찾지 못했습니다.")
    # goldens (domain)
    g_freq_g, g_curve_g = build_golden_auto(good_files, domain)
    g_freq_b, g_curve_b = build_golden_auto(bad_files,  domain)
    # features
    Xg = []; Xb = []
    for p in good_files:
        sp = parse_s1p(p)
        Xg.append(_compute_basic_features(sp, g_freq_g, g_curve_g, g_freq_b, g_curve_b, fmin, fmax, domain, compare_mode, smooth_win))
    for p in bad_files:
        sp = parse_s1p(p)
        Xb.append(_compute_basic_features(sp, g_freq_g, g_curve_g, g_freq_b, g_curve_b, fmin, fmax, domain, compare_mode, smooth_win))
    Xg = np.vstack(Xg); Xb = np.vstack(Xb)
    lda = _fit_lda(Xg, Xb)

    # S11 cross-check: GOOD set에서 **S11 golden + MAD 임계치**
    s11_freq_g, s11_curve_g = build_golden_auto(good_files, 'S11_DB')
    mets_s11 = [compute_metrics(parse_s1p(p), s11_freq_g, s11_curve_g, fmin, fmax, 'S11_DB', compare_mode) for p in good_files]
    s11_rmin, s11_rmse = auto_thresholds_mad(mets_s11, k_sigma=3.0)

    model = dict(
        version="rev4.0",
        feature_names=["r_good","rmse_good","r_bad","rmse_bad","noise_rms_db","peak_prom_db","fwhm_mhz"],
        domain=domain, compare_mode=compare_mode, fmin=fmin, fmax=fmax, smooth_win=int(smooth_win),
        good_golden=dict(freq=g_freq_g.tolist(), curve=g_curve_g.tolist()),
        bad_golden=dict(freq=g_freq_b.tolist(), curve=g_curve_b.tolist()),
        lda=dict(mu_good=lda["mu_good"].tolist(), mu_bad=lda["mu_bad"].tolist(),
                 Sigma_inv=lda["Sigma_inv"].tolist(), w=lda["w"].tolist(), c=lda["c"]),
        s11_good_golden=dict(freq=s11_freq_g.tolist(), curve=s11_curve_g.tolist()),
        s11_xcheck=dict(r_min=float(s11_rmin), rmse_max=float(s11_rmse))
    )
    return model

def save_ml_model(model: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

def load_ml_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_with_model(sp: S1P, model: Dict) -> Tuple[str, float, np.ndarray]:
    dom = model["domain"]; cmp = model["compare_mode"]; fmin = model["fmin"]; fmax = model["fmax"]
    sw = int(model.get("smooth_win",31))
    gg = model["good_golden"]; gb = model["bad_golden"]
    g_freq_g = np.asarray(gg["freq"], float); g_curve_g = np.asarray(gg["curve"], float)
    g_freq_b = np.asarray(gb["freq"], float); g_curve_b = np.asarray(gb["curve"], float)
    x = _compute_basic_features(sp, g_freq_g, g_curve_g, g_freq_b, g_curve_b, fmin, fmax, dom, cmp, sw)
    lda = model["lda"]
    w = np.asarray(lda["w"], float); c = float(lda["c"])
    z = _lda_score(x, w, c)
    label = "GOOD" if z >= 0 else "BAD"
    prob = _sigmoid(z)
    return label, prob, x

# ============================ Classify (+v3.8 확장) ============================

@dataclass
class QCResult:
    path: str
    r: float
    rmse_db: float
    decision: str
    noise_rms_db: Optional[float] = None
    peak_prom_db: Optional[float] = None
    fwhm_mhz: Optional[float] = None
    gate_notes: str = ""
    # rev4.0 추가
    stage: str = "NONE"         # STRICT / FLEX / NONE
    ml_label: str = "NA"        # GOOD/BAD/NA
    ml_score: float = 0.5       # sigmoid(score)
    s11_cc: str = "NA"          # OK/WARN/FAIL

def _two_stage_pass(m: Metrics, r_min: float, rmse_max: float, use_two_stage: bool, relax: float) -> Tuple[bool, str]:
    if (m.r >= r_min) and (m.rmse_db <= rmse_max):
        return True, "STRICT"
    if use_two_stage:
        r_relax = max(0.0, r_min - 0.02*relax)  # r 완화
        e_relax = rmse_max * relax
        if (m.r >= r_relax) and (m.rmse_db <= e_relax):
            return True, "FLEX"
    return False, "NONE"

def _s11_cross_check(sp: S1P, model: Optional[Dict], fmin: Optional[float], fmax: Optional[float],
                     compare_mode: str, relax: float, mode: str) -> str:
    try:
        if model and "s11_good_golden" in model:
            gg = model["s11_good_golden"]
            freq = np.asarray(gg["freq"], float); curve = np.asarray(gg["curve"], float)
        else:
            # Fallback: S11 자체 곡선(비권장; 모델 사용 권장). 이 경우 CC는 항상 OK에 가깝습니다.
            freq = sp.freq_hz; curve = s11_db(sp.s11)
        met = compute_metrics(sp, freq, curve, fmin, fmax, 'S11_DB', compare_mode)
        if model and "s11_xcheck" in model:
            rmin = float(model["s11_xcheck"]["r_min"]); erm = float(model["s11_xcheck"]["rmse_max"])
        else:
            rmin, erm = 0.95, 5.0
        rmin = max(0.0, rmin - 0.02*relax); erm = erm * relax
        ok = (met.r >= rmin) and (met.rmse_db <= erm)
        if ok: return "OK"
        return "FAIL" if mode.upper()=="FAIL" else "WARN"
    except Exception:
        return "WARN"

def classify_folder(root: str, golden_mode: str, golden_path: Optional[str], fmin: Optional[float], fmax: Optional[float],
                    r_min: Optional[float], rmse_max: Optional[float], regex: Optional[str],
                    domain: str='Y11_DB', compare_mode: str='AFFINE',
                    # v3.8 gates:
                    use_noise_gate: bool=False, use_shape_gate: bool=False, gate_mode: str='WARN',
                    smoothing_window_pts: int=31, roughness_max_db: float=0.25,
                    min_prom_db: float=1.0, max_fwhm_mhz: float=180.0,
                    # rev4.0 add-ons:
                    use_two_stage: bool=True, relax_factor: float=1.25,
                    ml_model_path: Optional[str]=None, ml_mode: str='WARN',
                    s11_crosscheck: bool=True, s11_cc_mode: str='WARN', s11_cc_relax: float=1.25
                    ) -> Tuple[List[QCResult], Tuple[float,float], Tuple[np.ndarray,np.ndarray]]:
    import re
    s1p_files = []
    pat = re.compile(regex) if (regex and regex.strip()) else None
    for dp,_,fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith('.s1p'):
                full = os.path.join(dp, fn)
                if pat and not pat.search(full): continue
                s1p_files.append(full)
    s1p_files.sort()
    if not s1p_files: raise RuntimeError("대상 s1p 파일을 찾지 못했습니다.")

    if golden_mode == 'auto':
        g_freq, g_curve = build_golden_auto(s1p_files, domain)
    elif golden_mode == 'single':
        if not golden_path: raise RuntimeError("Golden 단일 파일 경로가 비었습니다.")
        g_freq, g_curve = build_golden_single(golden_path, domain)
    elif golden_mode == 'folder':
        if not golden_path: raise RuntimeError("Golden 폴더 경로가 비었습니다.")
        if not os.path.isdir(golden_path): raise RuntimeError("Golden 폴더가 존재하지 않습니다.")
        g_freq, g_curve = build_golden_folder(golden_path, domain)
    else:
        raise RuntimeError("알 수 없는 golden_mode")

    # base metrics vs golden
    mets = [compute_metrics(parse_s1p(p), g_freq, g_curve, fmin, fmax, domain, compare_mode) for p in s1p_files]
    auto_r, auto_e = auto_thresholds_mad(mets, k_sigma=3.0)
    use_r = r_min if r_min is not None else auto_r
    use_e = rmse_max if rmse_max is not None else auto_e

    # optional ML model
    model = None
    if ml_model_path and os.path.isfile(ml_model_path):
        try:
            model = load_ml_model(ml_model_path)
        except Exception as e:
            print("[경고] ML 모델 로드 실패:", e, file=sys.stderr)

    results: List[QCResult] = []

    for p, m in zip(s1p_files, mets):
        sp = parse_s1p(p)

        # ---- Gates (v3.8 동작 유지; base_decision 기본 PASS) ----
        notes = []
        base_decision = 'PASS'
        nr, pr, fw = None, None, None
        if use_noise_gate or use_shape_gate:
            y = curve_in_domain(sp, domain)
            y_res = resample_to(sp.freq_hz, y, g_freq)
            mask = np.ones_like(g_freq, dtype=bool)
            if fmin is not None: mask &= (g_freq>=fmin)
            if fmax is not None: mask &= (g_freq<=fmax)
            f_sel = g_freq[mask]; y_sel = y_res[mask]
            if y_sel.size>=5:
                if use_noise_gate:
                    nr = noise_rms_db(y_sel, smooth_win=int(smoothing_window_pts))
                    if nr > roughness_max_db:
                        notes.append(f"NOISE {nr:.3f}dB>{roughness_max_db:.3f}dB")
                        if gate_mode.upper()=='FAIL':
                            base_decision = 'FAIL'
                if use_shape_gate:
                    pr, fw = peak_prominence_and_fwhm(f_sel, y_sel, domain, smooth_win=int(smoothing_window_pts))
                    bad = False
                    if pr < min_prom_db:
                        notes.append(f"PROM {pr:.2f}dB<{min_prom_db:.2f}dB"); bad=True
                    if fw > max_fwhm_mhz:
                        notes.append(f"FWHM {fw:.1f}MHz>{max_fwhm_mhz:.1f}MHz"); bad=True
                    if bad and gate_mode.upper()=='FAIL':
                        base_decision = 'FAIL'

        # ---- Stage 1/2 (r/rmse) ----
        pass12, stage = _two_stage_pass(m, use_r, use_e, use_two_stage, relax_factor)

        # ---- ML (optional) ----
        ml_label, ml_prob = "NA", 0.5
        if model is not None:
            try:
                lab, prob, feats = predict_with_model(sp, model)
                ml_label, ml_prob = lab, float(prob)
                if ml_mode.upper()=="FAIL" and lab=="BAD":
                    pass12 = False  # ML에서 확실히 BAD면 FAIL
                elif lab=="GOOD" and (not pass12):
                    # FLEX 보정: ML이 GOOD이면 완화 상향
                    r_relax = max(0.0, use_r - 0.03*relax_factor)
                    e_relax = use_e * (relax_factor*1.05)
                    if (m.r >= r_relax) and (m.rmse_db <= e_relax):
                        pass12 = True; stage = "FLEX"
            except Exception:
                notes.append("ML_ERR")

        # ---- S11 cross-check ----
        s11cc = "NA"
        if s11_crosscheck and (domain == 'Y11_DB'):
            s11cc = _s11_cross_check(sp, model, fmin, fmax, compare_mode, s11_cc_relax, s11_cc_mode)
            if s11cc=="FAIL":
                pass12 = False

        decision = 'PASS' if (base_decision=='PASS' and pass12) else 'FAIL'

        results.append(QCResult(
            path=p, r=m.r, rmse_db=m.rmse_db, decision=decision,
            noise_rms_db=nr, peak_prom_db=pr, fwhm_mhz=fw,
            gate_notes=("; ".join(notes) if notes else ""),
            stage=stage, ml_label=ml_label, ml_score=ml_prob, s11_cc=s11cc
        ))

    return results, (use_r, use_e), (g_freq, g_curve)

# ============================ Save/Load (CSV 확장) ============================

def save_results_csv(root: str, results: List[QCResult], r_min: float, rmse_max: float, out_name="qc_results.csv") -> str:
    out_dir = os.path.join(root, "_qc_results"); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["path","r","rmse_db","decision","r_min","rmse_max",
                    "noise_rms_db","peak_prom_db","fwhm_mhz","gate_notes",
                    "stage","ml_label","ml_score","s11_cc"])
        for row in results:
            w.writerow([row.path, f"{row.r:.6f}", f"{row.rmse_db:.6f}", row.decision,
                        f"{r_min:.6f}", f"{rmse_max:.6f}",
                        ("" if row.noise_rms_db is None else f"{row.noise_rms_db:.6f}"),
                        ("" if row.peak_prom_db is None else f"{row.peak_prom_db:.6f}"),
                        ("" if row.fwhm_mhz is None else f"{row.fwhm_mhz:.6f}"),
                        row.gate_notes, row.stage, row.ml_label, f"{row.ml_score:.4f}", row.s11_cc])
    return out_path

def thresholds_json_path(root: str) -> str:
    out_dir = os.path.join(root, "_qc_results"); os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "thresholds.json")

def save_thresholds(root: str, r_min: float, rmse_max: float) -> str:
    path = thresholds_json_path(root)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"r_min": r_min, "rmse_max": rmse_max}, f, ensure_ascii=False, indent=2)
    return path

def load_thresholds(root: str) -> Tuple[float,float]:
    path = thresholds_json_path(root)
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return float(obj["r_min"]), float(obj["rmse_max"])

# ============================ GUI ============================

def _open_files(paths: List[str]):
    if not paths: return
    sysname = platform.system().lower()
    for p in paths:
        try:
            if sysname.startswith('win'): os.startfile(p)  # type: ignore[attr-defined]
            elif sysname.startswith('darwin'): subprocess.Popen(['open', p])
            else: subprocess.Popen(['xdg-open', p])
        except Exception:
            pass

def launch_gui():
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[에러] GUI 로딩 실패:", e); sys.exit(1)

    win = tk.Tk()
    win.title("QC Thresholds — Pro+++++ ML rev4.0 (ML + 2-Stage + S11-CC)")
    win.geometry("1820x1120")
    nb = ttk.Notebook(win); nb.pack(fill='both', expand=True)

    tab = ttk.Frame(nb); nb.add(tab, text="Classifier")

    def choose_root():
        d = filedialog.askdirectory(title="측정 데이터 루트 폴더 선택")
        if d: root_var.set(d)

    def choose_golden_file():
        p = filedialog.askopenfilename(title="Golden s1p 선택", filetypes=[('s1p','*.s1p'), ('all','*.*')])
        if p: golden_path_var.set(p)

    def choose_golden_folder():
        d = filedialog.askdirectory(title="Golden 폴더 선택")
        if d: golden_path_var.set(d)

    def _parse_freq(val: str, unit: str) -> Optional[float]:
        if not val or not val.strip(): return None
        try: x=float(val); return x*(1e9 if unit=='GHz' else 1.0)
        except Exception: return None

    def _gmode() -> str:
        m = golden_mode_var.get()
        return 'auto' if m=='Auto' else ('single' if m=='Single file' else 'folder')

    def _log(msg: str):
        log.configure(state='normal'); log.insert(tk.END, msg+"\n"); log.see(tk.END); log.configure(state='disabled')

    def _update_lists_and_summary(results, used_r, used_e):
        total = len(results)
        n_pass = sum(1 for r in results if r.decision=='PASS')
        n_fail = total - n_pass
        p_pass = (n_pass/total*100.0) if total else 0.0
        pass_header.configure(text=f"PASS 목록  ({n_pass} / {p_pass:.1f}%)")
        fail_header.configure(text=f"FAIL 목록  ({n_fail} / {100.0-p_pass:.1f}%)")
        thr_label_var.set(f"r_min={used_r:.5f} / rmse_max={used_e:.5f} dB")
        pass_list.delete(0, tk.END); fail_list.delete(0, tk.END)
        for r in results:
            (pass_list if r.decision=='PASS' else fail_list).insert(tk.END, r.path)
        pass_label.configure(text=f"PASS {n_pass} ({p_pass:.1f}%)", foreground="#0a7d00")
        fail_label.configure(text=f"FAIL {n_fail} ({100.0-p_pass:.1f}%)", foreground="#c20000")
        total_label.configure(text=f"총 {total}개")

    # plotting state
    state = {"g_freq":None, "g_curve":None, "domain":"Y11_DB", "fmin":None, "fmax":None, "compare":"RAW"}
    group_vars = {"Golden": True, "PASS": True, "FAIL": True}
    group_lines = {"Golden": [], "PASS": [], "FAIL": []}

    def overlay_from_selection():
        sel_pass = [pass_list.get(i) for i in pass_list.curselection()]
        sel_fail = [fail_list.get(i) for i in fail_list.curselection()]
        if not sel_pass and not sel_fail:
            messagebox.showinfo("안내","좌/우 목록에서 하나 이상 선택하세요."); return
        draw_overlay(sel_pass, sel_fail)

    def update_offplot_legend(pass_paths: List[str], fail_paths: List[str]):
        legend_text.configure(state='normal'); legend_text.delete('1.0', 'end')
        if pass_paths:
            legend_text.insert('end', "[PASS]\n")
            for p in pass_paths: legend_text.insert('end', f"{p}\n")
        if fail_paths:
            legend_text.insert('end', "[FAIL]\n")
            for p in fail_paths: legend_text.insert('end', f"{p}\n")
        legend_text.configure(state='disabled')

    def draw_overlay(pass_paths: List[str], fail_paths: List[str]):
        try:
            if state["g_freq"] is None:
                messagebox.showwarning("경고","먼저 분류를 실행하세요."); return
            g_freq = state["g_freq"]; g_curve = state["g_curve"]; domain = state["domain"]
            fig.clf(); ax = fig.add_subplot(111)
            mask = np.ones_like(g_freq, dtype=bool)
            if state["fmin"] is not None: mask &= (g_freq>=state["fmin"])
            if state["fmax"] is not None: mask &= (g_freq<=state["fmax"])

            for k in group_lines: group_lines[k].clear()

            ln_g, = ax.plot(g_freq[mask]*1e-9, g_curve[mask], linewidth=2.6)
            group_lines["Golden"].append(ln_g)

            for p in pass_paths[:30]:
                sp = parse_s1p(p); s_curve = curve_in_domain(sp, state["domain"])
                s_curve_res = resample_to(sp.freq_hz, s_curve, g_freq)
                ln, = ax.plot(g_freq[mask]*1e-9, s_curve_res[mask], alpha=0.9)
                group_lines["PASS"].append(ln)
            for p in fail_paths[:30]:
                sp = parse_s1p(p); s_curve = curve_in_domain(sp, state["domain"])
                s_curve_res = resample_to(sp.freq_hz, s_curve, g_freq)
                ln, = ax.plot(g_freq[mask]*1e-9, s_curve_res[mask], alpha=0.9)
                group_lines["FAIL"].append(ln)

            ax.set_xlabel("Frequency (GHz)")
            ylabel = "S11 (dB)" if state["domain"]=='S11_DB' else "|Y11| (dB)"
            ax.set_ylabel(ylabel); ax.grid(True, which='both', linestyle='--', alpha=0.35)
            canvas.draw(); apply_group_visibility(); update_offplot_legend(pass_paths, fail_paths)
        except Exception as e:
            messagebox.showerror("Plot 실패", str(e))

    def clear_plot():
        fig.clf(); ax = fig.add_subplot(111)
        ax.set_xlabel("Frequency (GHz)")
        ylabel = "S11 (dB)" if state["domain"]=='S11_DB' else "|Y11| (dB)"
        ax.set_ylabel(ylabel); ax.grid(True, which='both', linestyle='--', alpha=0.35)
        canvas.draw()
        legend_text.configure(state='normal'); legend_text.delete('1.0','end'); legend_text.configure(state='disabled')
        for k in group_lines: group_lines[k].clear()

    def apply_group_visibility():
        for grp, visible in group_vars.items():
            for ln in group_lines[grp]:
                ln.set_visible(visible)
        canvas.draw()

    def on_toggle_group():
        group_vars["Golden"] = var_show_golden.get()
        group_vars["PASS"]   = var_show_pass.get()
        group_vars["FAIL"]   = var_show_fail.get()
        apply_group_visibility()

    def on_oneclick(): _run_classify(auto=True)
    def on_classify(): _run_classify(auto=False)

    def on_auto_mad():
        try:
            root_dir = root_var.get().strip()
            if not root_dir: messagebox.showwarning("경고","--root 폴더를 선택하세요."); return
            unit = freq_unit_var.get()
            fmin = _parse_freq(freq_min_var.get(), unit); fmax = _parse_freq(freq_max_var.get(), unit)
            regex = regex_var.get().strip() or None
            domain = 'S11_DB' if domain_var.get()=="S11(dB)" else 'Y11_DB'
            compare_mode = 'AFFINE' if compare_mode_var.get()=="AFFINE (scale/offset 보정)" else 'RAW'
            files = list_s1p_files(root_dir, regex)
            if not files: raise RuntimeError("s1p 파일을 찾지 못했습니다.")
            g_freq, g_curve = build_golden_auto(files, domain)
            mets = [compute_metrics(parse_s1p(p), g_freq, g_curve, fmin, fmax, domain, compare_mode) for p in files]
            rmin, ermse = auto_thresholds_mad(mets, k_sigma=3.0)
            rmin_var.set(f"{rmin:.5f}"); rmse_var.set(f"{ermse:.5f}")
            messagebox.showinfo("추천 임계값(MAD)", f"r_min={rmin:.5f}\nrmse_max={ermse:.5f} dB\n(저장되지 않았습니다)")
        except Exception as e:
            messagebox.showerror("실패", str(e))

    def on_calibrate_save():
        try:
            root_dir = root_var.get().strip()
            if not root_dir: messagebox.showwarning("경고","--root 폴더를 선택하세요."); return
            unit = freq_unit_var.get()
            fmin = _parse_freq(freq_min_var.get(), unit); fmax = _parse_freq(freq_max_var.get(), unit)
            regex = regex_var.get().strip() or None
            domain = 'S11_DB' if domain_var.get()=="S11(dB)" else 'Y11_DB'
            compare_mode = 'AFFINE' if compare_mode_var.get()=="AFFINE (scale/offset 보정)" else 'RAW'
            files = list_s1p_files(root_dir, regex)
            if not files: raise RuntimeError("s1p 파일을 찾지 못했습니다.")
            g_freq, g_curve = build_golden_auto(files, domain)
            try: pass_rate = float(passrate_var.get())/100.0
            except Exception: pass_rate = 0.98
            mets = [compute_metrics(parse_s1p(p), g_freq, g_curve, fmin, fmax, domain, compare_mode) for p in files]
            rmin, ermse = auto_thresholds_quantile(mets, pass_rate=pass_rate)
            rmin_var.set(f"{rmin:.5f}"); rmse_var.set(f"{ermse:.5f}")
            save_path = save_thresholds(root_dir, rmin, ermse)
            messagebox.showinfo("Calibrate 저장", f"r_min={rmin:.5f}\nrmse_max={ermse:.5f} dB\n저장: {save_path}")
        except Exception as e:
            messagebox.showerror("실패", str(e))

    def on_load_thresholds():
        try:
            root_dir = root_var.get().strip()
            if not root_dir: messagebox.showwarning("경고","--root 폴더를 선택하세요."); return
            rmin, ermse = load_thresholds(root_dir)
            rmin_var.set(f"{rmin:.5f}"); rmse_var.set(f"{ermse:.5f}")
            messagebox.showinfo("불러오기 완료", f"r_min={rmin:.5f}\nrmse_max={ermse:.5f} dB")
        except Exception as e:
            messagebox.showerror("실패", str(e))

    # ---------- ML Trainer ----------

    def choose_good_folder():
        d = filedialog.askdirectory(title="GOOD 폴더 선택")
        if d: good_root_var.set(d)

    def choose_bad_folder():
        d = filedialog.askdirectory(title="BAD 폴더 선택")
        if d: bad_root_var.set(d)

    def save_model_dialog(model: dict):
        p = filedialog.asksaveasfilename(title="ML 모델 저장", defaultextension=".json",
                                         filetypes=[("JSON","*.json")], initialfile="qc_ml_model.json")
        if not p: return
        save_ml_model(model, p)
        ml_model_path_var.set(p)
        try: messagebox.showinfo("저장 완료", f"ML 모델 저장: {p}")
        except Exception: pass

    def on_fit_and_save():
        try:
            good_root = good_root_var.get().strip()
            bad_root  = bad_root_var.get().strip()
            if not good_root or not bad_root:
                messagebox.showwarning("경고","GOOD/BAD 폴더를 모두 지정하세요."); return
            unit = freq_unit_var.get()
            fmin = _parse_freq(freq_min_var.get(), unit); fmax = _parse_freq(freq_max_var.get(), unit)
            regex = regex_var.get().strip() or None
            domain = 'S11_DB' if domain_var.get()=="S11(dB)" else 'Y11_DB'
            compare_mode = 'AFFINE' if compare_mode_var.get()=="AFFINE (scale/offset 보정)" else 'RAW'
            model = fit_ml_model(good_root, bad_root, domain, compare_mode, fmin, fmax, regex, int(var_smooth_win.get() or "31"))
            save_model_dialog(model)
        except Exception as e:
            messagebox.showerror("학습 실패", str(e))

    def on_load_model():
        p = filedialog.askopenfilename(title="ML 모델 불러오기", filetypes=[("JSON","*.json"), ("All","*.*")])
        if p:
            ml_model_path_var.set(p)
            try:
                _ = load_ml_model(p)
                messagebox.showinfo("불러오기", f"모델 로드 성공: {p}")
            except Exception as e:
                messagebox.showerror("오류", str(e))

    def _run_classify(auto: bool):
        try:
            root_dir = root_var.get().strip()
            if not root_dir: messagebox.showwarning("경고","--root 폴더를 선택하세요."); return
            if not os.path.isdir(root_dir): messagebox.showerror("에러","폴더가 존재하지 않습니다: "+root_dir); return

            unit = freq_unit_var.get()
            fmin = _parse_freq(freq_min_var.get(), unit); fmax = _parse_freq(freq_max_var.get(), unit)
            regex = regex_var.get().strip() or None
            domain = 'S11_DB' if domain_var.get()=="S11(dB)" else 'Y11_DB'
            compare_mode = 'AFFINE' if compare_mode_var.get()=="AFFINE (scale/offset 보정)" else 'RAW'

            # gates (v3.8)
            g_use_noise  = bool(var_use_noise.get())
            g_use_shape  = bool(var_use_shape.get())
            g_gate_mode  = gate_mode_var.get()  # WARN / FAIL
            try: g_smooth = int(var_smooth_win.get() or "31")
            except Exception: g_smooth = 31
            try: g_rough  = float(var_rough_max.get() or "0.25")
            except Exception: g_rough  = 0.25
            try: g_minprom = float(var_min_prom.get() or "1.0")
            except Exception: g_minprom = 1.0
            try: g_maxfwhm = float(var_max_fwhm.get() or "180.0")
            except Exception: g_maxfwhm = 180.0

            # two-stage & ML & cross-check
            use_two = bool(var_use_two_stage.get())
            relax = float(var_relax.get() or "1.25")
            ml_use = bool(var_use_ml.get())
            ml_mode = ml_mode_var.get()
            ml_model_path = ml_model_path_var.get().strip() if ml_use else None
            cc_use = bool(var_s11_cc.get())
            cc_mode = s11_cc_mode_var.get()
            cc_relax = float(var_s11_relax.get() or "1.25")

            if auto: use_r, use_e = None, None; gmode = _gmode(); gpath = golden_path_var.get().strip() or None
            else:
                gmode = _gmode(); gpath = golden_path_var.get().strip() or None
                use_r = float(rmin_var.get()) if rmin_var.get().strip() else None
                use_e = float(rmse_var.get()) if rmse_var.get().strip() else None

            _log(f"[진행] classify: domain={domain}, compare={compare_mode}, golden={gmode}, path={gpath}, f=[{fmin},{fmax}], regex={regex}")
            _log(f"        gates: noise={g_use_noise}, shape={g_use_shape}, mode={g_gate_mode}, smooth={g_smooth}")
            _log(f"        2-stage={use_two} relax={relax} | ML={ml_use}({ml_mode}) model={ml_model_path} | S11-CC={cc_use}({cc_mode}, x{cc_relax})")

            results, (used_r, used_e), (g_freq, g_curve) = classify_folder(
                root=root_dir, golden_mode=gmode, golden_path=gpath, fmin=fmin, fmax=fmax,
                r_min=use_r, rmse_max=use_e, regex=regex, domain=domain, compare_mode=compare_mode,
                use_noise_gate=g_use_noise, use_shape_gate=g_use_shape, gate_mode=g_gate_mode,
                smoothing_window_pts=g_smooth, roughness_max_db=g_rough, min_prom_db=g_minprom, max_fwhm_mhz=g_maxfwhm,
                use_two_stage=use_two, relax_factor=relax,
                ml_model_path=ml_model_path, ml_mode=ml_mode,
                s11_crosscheck=cc_use, s11_cc_mode=cc_mode, s11_cc_relax=cc_relax
            )
            _log(f"[완료] r_min={used_r:.5f}, rmse_max={used_e:.5f} dB")

            out_csv = save_results_csv(root_dir, results, used_r, used_e)
            n_pass = sum(1 for r in results if r.decision=='PASS'); n_fail=len(results)-n_pass
            _log(f"CSV: {out_csv}")
            _log(f"요약: PASS={n_pass} ({(n_pass/len(results)*100.0 if results else 0):.1f}%), "
                 f"FAIL={n_fail} ({(n_fail/len(results)*100.0 if results else 0):.1f}%), 총 {len(results)}개")
            # 미리보기 일부
            for row in results[:40]:
                _log(f"{row.decision}\t{row.stage}\tml={row.ml_label}:{row.ml_score:.2f}\ts11cc={row.s11_cc}\t"
                     f"r={row.r:.4f}\trmse={row.rmse_db:.3f}\t{row.path}")

            _update_lists_and_summary(results, used_r, used_e)

            state["g_freq"]=g_freq; state["g_curve"]=g_curve; state["domain"]=domain; state["fmin"]=fmin; state["fmax"]=fmax; state["compare"]=compare_mode
            try:
                from tkinter import messagebox
                messagebox.showinfo("완료", f"PASS/FAIL 분류 완료\nCSV 저장: {out_csv}")
            except Exception:
                pass
        except Exception as e:
            try:
                from tkinter import messagebox
                messagebox.showerror("실패", str(e))
            except Exception:
                print("실패:", e, file=sys.stderr)
            _log(f"[에러] {e}")

    # ------------- Widgets -------------
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt

    frm = ttk.Frame(tab, padding=10); frm.pack(fill='both', expand=True)
    frm.columnconfigure(0, weight=0); frm.columnconfigure(1, weight=1)
    frm.rowconfigure(0, weight=1)

    # Left column
    left = ttk.Frame(frm); left.grid(row=0, column=0, sticky='nsw')
    ctl = ttk.Frame(left); ctl.pack(fill='x')

    ttk.Label(ctl, text="--root").grid(row=0, column=0, sticky='w')
    root_var = tk.StringVar(); ttk.Entry(ctl, textvariable=root_var, width=46).grid(row=0, column=1, sticky='w', padx=5)
    ttk.Button(ctl, text="찾기", command=choose_root).grid(row=0, column=2, padx=5)
    ttk.Button(ctl, text="HELP", command=lambda: messagebox.showinfo("HELP",
        "Flow 제안:\n"
        "1) ML 학습: GOOD/BAD 폴더 지정 → Fit&Save → 모델 저장(JSON)\n"
        "2) 분류: --root 지정 → (필요시 Load Thresholds) → 2-Stage/ML/S11-CC 옵션 조절 → 실행\n"
        "3) CSV의 stage/ML/s11_cc 열에서 사유 확인\n"
        "Gates는 기본 WARN으로 시작. r/rmse 기준 안정화 + ML GOOD/BAD로 보정 후 필요 시 FAIL로 전환.\n"
        "S11 Cross-check는 Y11 도메인에서 S11도 정상인지 검증합니다.") ).grid(row=0, column=3, padx=(2,0))

    ttk.Label(ctl, text="도메인").grid(row=1, column=0, sticky='w')
    domain_var = tk.StringVar(value="Y11(|Y| dB)")
    ttk.Combobox(ctl, textvariable=domain_var, values=("S11(dB)","Y11(|Y| dB)"), width=18, state="readonly").grid(row=1, column=1, sticky='w', padx=5)

    ttk.Label(ctl, text="비교 모드").grid(row=2, column=0, sticky='w')
    compare_mode_var = tk.StringVar(value="AFFINE (scale/offset 보정)")
    ttk.Combobox(ctl, textvariable=compare_mode_var, values=("RAW","AFFINE (scale/offset 보정)"), width=22, state="readonly").grid(row=2, column=1, sticky='w', padx=5)

    ttk.Label(ctl, text="Golden 소스").grid(row=3, column=0, sticky='w')
    golden_mode_var = tk.StringVar(value="Auto")
    ttk.Combobox(ctl, textvariable=golden_mode_var, values=("Auto","Single file","Folder"), width=18, state="readonly").grid(row=3, column=1, sticky='w', padx=5)

    ttk.Label(ctl, text="Golden 경로").grid(row=4, column=0, sticky='w')
    golden_path_var = tk.StringVar()
    ttk.Entry(ctl, textvariable=golden_path_var, width=46).grid(row=4, column=1, sticky='w', padx=5)
    btn_g = ttk.Frame(ctl); btn_g.grid(row=4, column=2, sticky='w')
    ttk.Button(btn_g, text="파일", command=choose_golden_file).pack(side='left', padx=(0,4))
    ttk.Button(btn_g, text="폴더", command=choose_golden_folder).pack(side='left')

    ttk.Label(ctl, text="--regex(선택)").grid(row=5, column=0, sticky='w')
    regex_var = tk.StringVar(); ttk.Entry(ctl, textvariable=regex_var, width=46).grid(row=5, column=1, sticky='w', padx=5)

    ttk.Label(ctl, text="주파수 범위").grid(row=6, column=0, sticky='w')
    freq_min_var = tk.StringVar(); freq_max_var = tk.StringVar()
    unit_opts = ("GHz","Hz"); freq_unit_var = tk.StringVar(value="GHz")
    freq_box = ttk.Frame(ctl); freq_box.grid(row=6, column=1, sticky='w', padx=5)
    ttk.Entry(freq_box, textvariable=freq_min_var, width=10).pack(side='left', padx=(0,4))
    ttk.Label(freq_box, text="~").pack(side='left')
    ttk.Entry(freq_box, textvariable=freq_max_var, width=10).pack(side='left', padx=(4,6))
    ttk.Combobox(freq_box, textvariable=freq_unit_var, values=unit_opts, width=5, state="readonly").pack(side='left')

    ttk.Label(ctl, text="임계값 r_min / rmse_max(dB)").grid(row=7, column=0, sticky='w')
    rmin_var = tk.StringVar(); rmse_var = tk.StringVar()
    thr_box = ttk.Frame(ctl); thr_box.grid(row=7, column=1, sticky='w', padx=5)
    ttk.Entry(thr_box, textvariable=rmin_var, width=10).pack(side='left', padx=(0,6))
    ttk.Entry(thr_box, textvariable=rmse_var, width=10).pack(side='left')

    ttk.Label(ctl, text="타깃 통과율(%)").grid(row=8, column=0, sticky='w')
    passrate_var = tk.StringVar(value="98")
    ttk.Entry(ctl, textvariable=passrate_var, width=6).grid(row=8, column=1, sticky='w', padx=(5,0))

    btns = ttk.Frame(ctl); btns.grid(row=9, column=0, columnspan=4, sticky='w', pady=8)
    ttk.Button(btns, text="Auto 추천(MAD)", command=on_auto_mad).pack(side='left', padx=3)
    ttk.Button(btns, text="Calibrate(양품→Save)", command=on_calibrate_save).pack(side='left', padx=3)
    ttk.Button(btns, text="Load Thresholds", command=on_load_thresholds).pack(side='left', padx=3)
    ttk.Button(btns, text="원클릭 분류(Easy)", command=on_oneclick).pack(side='left', padx=3)
    ttk.Button(btns, text="분류 실행", command=on_classify).pack(side='left', padx=3)

    # PASS/FAIL lists
    lists = ttk.Frame(left); lists.pack(fill='both', expand=True, pady=(10,6))
    pass_header = ttk.Label(lists, text="PASS 목록"); fail_header = ttk.Label(lists, text="FAIL 목록")
    pass_header.grid(row=0, column=0, sticky='w'); fail_header.grid(row=0, column=1, sticky='w')
    pass_xsb = ttk.Scrollbar(lists, orient='horizontal'); fail_xsb = ttk.Scrollbar(lists, orient='horizontal')
    pass_list = tk.Listbox(lists, height=22, selectmode=tk.EXTENDED, exportselection=False, xscrollcommand=pass_xsb.set)
    fail_list = tk.Listbox(lists, height=22, selectmode=tk.EXTENDED, exportselection=False, xscrollcommand=fail_xsb.set)
    pass_xsb.config(command=pass_list.xview); fail_xsb.config(command=fail_list.xview)
    pass_list.grid(row=1, column=0, sticky='nsew', padx=(0,8)); fail_list.grid(row=1, column=1, sticky='nsew')
    pass_xsb.grid(row=2, column=0, sticky='ew', padx=(0,8)); fail_xsb.grid(row=2, column=1, sticky='ew')
    lists.columnconfigure(0, weight=1); lists.columnconfigure(1, weight=1)

    # List buttons + context menu
    list_btns = ttk.Frame(left); list_btns.pack(fill='x', pady=(4,10))
    ttk.Button(list_btns, text="선택 오버레이", command=overlay_from_selection).pack(side='left', padx=3)
    ttk.Button(list_btns, text="PASS 전체 선택", command=lambda: (pass_list.select_set(0, 'end'))).pack(side='left', padx=3)
    ttk.Button(list_btns, text="FAIL 전체 선택", command=lambda: (fail_list.select_set(0, 'end'))).pack(side='left', padx=3)
    ttk.Button(list_btns, text="선택 해제", command=lambda: (pass_list.selection_clear(0,'end'), fail_list.selection_clear(0,'end'))).pack(side='left', padx=3)
    menu = tk.Menu(win, tearoff=0)
    def copy_paths():
        paths = [pass_list.get(i) for i in pass_list.curselection()] + [fail_list.get(i) for i in fail_list.curselection()]
        if not paths: return
        win.clipboard_clear(); win.clipboard_append("\n".join(paths))
        try: messagebox.showinfo("복사됨", "경로를 클립보드에 복사했습니다.")
        except Exception: pass
    def open_paths():
        paths = [pass_list.get(i) for i in pass_list.curselection()] + [fail_list.get(i) for i in fail_list.curselection()]
        if not paths: return
        _open_files(paths)
    menu.add_command(label="경로 복사", command=copy_paths)
    menu.add_command(label="파일 열기", command=open_paths)
    def show_menu(event, lb):
        try: lb.selection_set(lb.nearest(event.y))
        except Exception: pass
        menu.tk_popup(event.x_root, event.y_root)
    pass_list.bind("<Button-3>", lambda e: show_menu(e, pass_list))
    fail_list.bind("<Button-3>", lambda e: show_menu(e, fail_list))

    # Right column
    right = ttk.Frame(frm); right.grid(row=0, column=1, sticky='nsew', padx=8)
    right.rowconfigure(0, weight=6); right.rowconfigure(1, weight=0); right.rowconfigure(2, weight=1); right.rowconfigure(3, weight=0); right.rowconfigure(4, weight=3)
    right.columnconfigure(0, weight=1)

    fig = plt.Figure(figsize=(10.0,6.4))
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas_widget = canvas.get_tk_widget(); canvas_widget.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

    grpbar = ttk.Frame(right); grpbar.grid(row=1, column=0, sticky='w', padx=6, pady=(0,4))
    var_show_golden = tk.BooleanVar(value=True)
    var_show_pass = tk.BooleanVar(value=True)
    var_show_fail = tk.BooleanVar(value=True)
    ttk.Checkbutton(grpbar, text="Golden", variable=var_show_golden, command=on_toggle_group).pack(side='left', padx=(0,10))
    ttk.Checkbutton(grpbar, text="PASS 그룹", variable=var_show_pass, command=on_toggle_group).pack(side='left', padx=(0,10))
    ttk.Checkbutton(grpbar, text="FAIL 그룹", variable=var_show_fail, command=on_toggle_group).pack(side='left', padx=(0,10))
    ttk.Button(grpbar, text="그래프 Clear", command=clear_plot).pack(side='left', padx=(10,0))

    legend_frame = ttk.Frame(right); legend_frame.grid(row=2, column=0, sticky='nsew', padx=6, pady=(0,6))
    legend_xsb = ttk.Scrollbar(legend_frame, orient='horizontal')
    legend_ysb = ttk.Scrollbar(legend_frame, orient='vertical')
    legend_text = tk.Text(legend_frame, height=6, wrap='none', xscrollcommand=legend_xsb.set, yscrollcommand=legend_ysb.set, state='disabled')
    legend_xsb.config(command=legend_text.xview); legend_ysb.config(command=legend_text.yview)
    ttk.Label(legend_frame, text="파일 목록(오버레이) — on-plot 범주 대신 여기서 확인").grid(row=0, column=0, sticky='w')
    legend_text.grid(row=1, column=0, sticky='nsew'); legend_xsb.grid(row=2, column=0, sticky='ew'); legend_ysb.grid(row=1, column=1, sticky='ns')
    legend_frame.columnconfigure(0, weight=1); legend_frame.rowconfigure(1, weight=1)

    thr_label_var = tk.StringVar(value="임계값: -")
    bar = ttk.Frame(right); bar.grid(row=3, column=0, sticky='ew', padx=6, pady=(0,4))
    ttk.Label(bar, textvariable=thr_label_var).pack(side='left')
    total_label = ttk.Label(bar, text="총 -개"); total_label.pack(side='right', padx=(8,0))
    fail_label  = ttk.Label(bar, text="FAIL - (-%)", foreground="#c20000"); fail_label.pack(side='right', padx=(8,0))
    pass_label  = ttk.Label(bar, text="PASS - (-%)", foreground="#0a7d00"); pass_label.pack(side='right', padx=(8,0))

    ttk.Label(right, text="실행 로그").grid(row=4, column=0, sticky='w')
    log = tk.Text(right, height=16, width=100, state='disabled')
    log.grid(row=5, column=0, sticky='nsew', padx=5, pady=(0,5))

    # ----- Gates panel (v3.8) -----
    gatefrm = ttk.LabelFrame(left, text="Noise/Shape Gate (v3.8)")
    gatefrm.pack(fill='x', pady=(6,4))

    var_use_noise = tk.BooleanVar(value=False)
    var_use_shape = tk.BooleanVar(value=False)
    ttk.Checkbutton(gatefrm, text="Use Noise Gate", variable=var_use_noise).grid(row=0, column=0, sticky='w', padx=4, pady=2)
    ttk.Checkbutton(gatefrm, text="Use Shape Gate", variable=var_use_shape).grid(row=0, column=1, sticky='w', padx=4, pady=2)

    ttk.Label(gatefrm, text="Gate Mode").grid(row=0, column=2, sticky='e')
    gate_mode_var = tk.StringVar(value="WARN")
    ttk.Combobox(gatefrm, textvariable=gate_mode_var, values=("WARN","FAIL"), width=6, state="readonly").grid(row=0, column=3, sticky='w', padx=4)

    ttk.Label(gatefrm, text="Smoothing window (pts)").grid(row=1, column=0, sticky='w', padx=4)
    var_smooth_win = tk.StringVar(value="31")
    ttk.Entry(gatefrm, textvariable=var_smooth_win, width=8).grid(row=1, column=1, sticky='w', padx=4)

    ttk.Label(gatefrm, text="Max roughness (dB)").grid(row=2, column=0, sticky='w', padx=4)
    var_rough_max = tk.StringVar(value="0.25")
    ttk.Entry(gatefrm, textvariable=var_rough_max, width=8).grid(row=2, column=1, sticky='w', padx=4)

    ttk.Label(gatefrm, text="Min prominence (dB)").grid(row=3, column=0, sticky='w', padx=4)
    var_min_prom = tk.StringVar(value="1.0")
    ttk.Entry(gatefrm, textvariable=var_min_prom, width=8).grid(row=3, column=1, sticky='w', padx=4)

    ttk.Label(gatefrm, text="Max FWHM (MHz)").grid(row=4, column=0, sticky='w', padx=4)
    var_max_fwhm = tk.StringVar(value="180.0")
    ttk.Entry(gatefrm, textvariable=var_max_fwhm, width=8).grid(row=4, column=1, sticky='w', padx=4)

    # ----- ML / Cross-check panel -----
    mlfrm = ttk.LabelFrame(left, text="ML / Cross-check (rev4.0)")
    mlfrm.pack(fill='x', pady=(8,6))

    good_root_var = tk.StringVar(); bad_root_var = tk.StringVar()
    ttk.Label(mlfrm, text="GOOD 폴더").grid(row=0, column=0, sticky='w', padx=4)
    ttk.Entry(mlfrm, textvariable=good_root_var, width=40).grid(row=0, column=1, sticky='w', padx=4)
    ttk.Button(mlfrm, text="찾기", command=choose_good_folder).grid(row=0, column=2, padx=2)

    ttk.Label(mlfrm, text="BAD 폴더").grid(row=1, column=0, sticky='w', padx=4)
    ttk.Entry(mlfrm, textvariable=bad_root_var, width=40).grid(row=1, column=1, sticky='w', padx=4)
    ttk.Button(mlfrm, text="찾기", command=choose_bad_folder).grid(row=1, column=2, padx=2)

    ttk.Button(mlfrm, text="Fit & Save 모델", command=on_fit_and_save).grid(row=2, column=0, padx=4, pady=(4,2), sticky='w')

    ttk.Label(mlfrm, text="ML 모델").grid(row=3, column=0, sticky='w', padx=4)
    ml_model_path_var = tk.StringVar()
    ttk.Entry(mlfrm, textvariable=ml_model_path_var, width=40).grid(row=3, column=1, sticky='w', padx=4)
    ttk.Button(mlfrm, text="불러오기", command=on_load_model).grid(row=3, column=2, padx=2)

    var_use_ml = tk.BooleanVar(value=True)
    ttk.Checkbutton(mlfrm, text="Use ML", variable=var_use_ml).grid(row=4, column=0, sticky='w', padx=4)
    ttk.Label(mlfrm, text="ML Mode").grid(row=4, column=1, sticky='e')
    ml_mode_var = tk.StringVar(value="WARN")
    ttk.Combobox(mlfrm, textvariable=ml_mode_var, values=("WARN","FAIL"), width=6, state="readonly").grid(row=4, column=2, sticky='w', padx=4)

    var_use_two_stage = tk.BooleanVar(value=True)
    ttk.Checkbutton(mlfrm, text="Use 2-Stage", variable=var_use_two_stage).grid(row=5, column=0, sticky='w', padx=4)
    ttk.Label(mlfrm, text="Relax ×").grid(row=5, column=1, sticky='e')
    var_relax = tk.StringVar(value="1.25")
    ttk.Entry(mlfrm, textvariable=var_relax, width=6).grid(row=5, column=2, sticky='w', padx=4)

    var_s11_cc = tk.BooleanVar(value=True)
    ttk.Checkbutton(mlfrm, text="S11 Cross-check", variable=var_s11_cc).grid(row=6, column=0, sticky='w', padx=4)
    ttk.Label(mlfrm, text="CC Mode").grid(row=6, column=1, sticky='e')
    s11_cc_mode_var = tk.StringVar(value="WARN")
    ttk.Combobox(mlfrm, textvariable=s11_cc_mode_var, values=("WARN","FAIL"), width=6, state="readonly").grid(row=6, column=2, sticky='w', padx=4)
    ttk.Label(mlfrm, text="CC Relax ×").grid(row=7, column=1, sticky='e')
    var_s11_relax = tk.StringVar(value="1.25")
    ttk.Entry(mlfrm, textvariable=var_s11_relax, width=6).grid(row=7, column=2, sticky='w', padx=4)

    def on_domain_change(*_):
        state["domain"] = 'S11_DB' if domain_var.get()=="S11(dB)" else 'Y11_DB'
    domain_var.trace_add('write', on_domain_change)

    win.mainloop()

# ============================ CLI ============================

def main():
    p = argparse.ArgumentParser(description="QC Classifier Pro+++++ ML rev4.0 (ML + 2-Stage + S11-CC)")
    sub = p.add_subparsers(dest='cmd')
    p.add_argument('--no_gui', action='store_true')

    # classify
    pc = sub.add_parser('classify')
    pc.add_argument('--root', required=True); pc.add_argument('--freq_min', type=float); pc.add_argument('--freq_max', type=float)
    pc.add_argument('--regex'); pc.add_argument('--r_min', type=float); pc.add_argument('--rmse_max', type=float)
    pc.add_argument('--domain', choices=['S11_DB','Y11_DB'], default='Y11_DB')
    pc.add_argument('--golden_mode', choices=['auto','single','folder'], default='auto')
    pc.add_argument('--golden_path')
    pc.add_argument('--compare_mode', choices=['RAW','AFFINE'], default='AFFINE')
    # gate params (v3.8)
    pc.add_argument('--use_noise_gate', action='store_true')
    pc.add_argument('--use_shape_gate', action='store_true')
    pc.add_argument('--gate_mode', choices=['WARN','FAIL'], default='WARN')
    pc.add_argument('--smoothing_window_pts', type=int, default=31)
    pc.add_argument('--roughness_max_db', type=float, default=0.25)
    pc.add_argument('--min_prom_db', type=float, default=1.0)
    pc.add_argument('--max_fwhm_mhz', type=float, default=180.0)
    # rev4.0 options
    pc.add_argument('--use_two_stage', action='store_true')
    pc.add_argument('--relax_factor', type=float, default=1.25)
    pc.add_argument('--ml_model_path')
    pc.add_argument('--ml_mode', choices=['WARN','FAIL'], default='WARN')
    pc.add_argument('--s11_crosscheck', action='store_true')
    pc.add_argument('--s11_cc_mode', choices=['WARN','FAIL'], default='WARN')
    pc.add_argument('--s11_cc_relax', type=float, default=1.25)

    # fit-ml
    pf = sub.add_parser('fit')
    pf.add_argument('--good_root', required=True)
    pf.add_argument('--bad_root', required=True)
    pf.add_argument('--domain', choices=['S11_DB','Y11_DB'], default='Y11_DB')
    pf.add_argument('--compare_mode', choices=['RAW','AFFINE'], default='AFFINE')
    pf.add_argument('--freq_min', type=float); pf.add_argument('--freq_max', type=float)
    pf.add_argument('--regex'); pf.add_argument('--smooth_win', type=int, default=31)
    pf.add_argument('--out', default='qc_ml_model.json')

    if len(sys.argv)==1 and '--no_gui' not in sys.argv:
        launch_gui(); return
    args = p.parse_args()
    if args.cmd is None:
        if args.no_gui: p.error('no_gui 모드에서는 서브커맨드 필요')
        launch_gui(); return

    if args.cmd=='fit':
        model = fit_ml_model(args.good_root, args.bad_root, args.domain, args.compare_mode,
                             args.freq_min, args.freq_max, args.regex, args.smooth_win)
        save_ml_model(model, args.out)
        print("ML 모델 저장:", args.out)
        return

    if args.cmd=='classify':
        results, (rmin, ermse), _ = classify_folder(
            root=args.root, golden_mode=args.golden_mode, golden_path=args.golden_path,
            fmin=args.freq_min, fmax=args.freq_max, r_min=args.r_min, rmse_max=args.rmse_max,
            regex=args.regex, domain=args.domain, compare_mode=args.compare_mode,
            use_noise_gate=args.use_noise_gate, use_shape_gate=args.use_shape_gate, gate_mode=args.gate_mode,
            smoothing_window_pts=args.smoothing_window_pts, roughness_max_db=args.roughness_max_db,
            min_prom_db=args.min_prom_db, max_fwhm_mhz=args.max_fwhm_mhz,
            use_two_stage=args.use_two_stage, relax_factor=args.relax_factor,
            ml_model_path=args.ml_model_path, ml_mode=args.ml_mode,
            s11_crosscheck=args.s11_crosscheck, s11_cc_mode=args.s11_cc_mode, s11_cc_relax=args.s11_cc_relax
        )
        out_csv = save_results_csv(args.root, results, rmin, ermse)
        print(f"CSV: {out_csv}")
        n_pass = sum(1 for r in results if r.decision=='PASS'); n_fail=len(results)-n_pass
        print(f"요약: PASS={n_pass} ({(n_pass/len(results)*100.0 if results else 0):.1f}%), "
              f"FAIL={n_fail} ({(n_fail/len(results)*100.0 if results else 0):.1f}%), 총 {len(results)}개")
        for row in results[:80]:
            parts = []
            if row.noise_rms_db is not None: parts.append(f"noise={row.noise_rms_db:.3f}dB")
            if row.peak_prom_db is not None: parts.append(f"prom={row.peak_prom_db:.2f}dB")
            if row.fwhm_mhz is not None: parts.append(f"fwhm={row.fwhm_mhz:.1f}MHz")
            extra = ("  [" + " ".join(parts) + "]") if parts else ""
            print(f"{row.decision}\t{row.stage}\tml={row.ml_label}:{row.ml_score:.2f}\ts11cc={row.s11_cc}\t"
                  f"r={row.r:.4f}\trmse={row.rmse_db:.3f} dB\t {row.path}{extra}\t{row.gate_notes}")
        return

if __name__ == '__main__':
    main()
