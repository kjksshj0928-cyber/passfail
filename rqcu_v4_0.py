#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resonator QC Unified v4.0 (HJ)
- 기본 탭 리스트: file / verdict (reason 숨김)
- 고급(잠금) 탭 리스트: file / verdict / reason
- 표는 각 탭 내부에 표시, 플롯과 로그는 하단 공통
- 버튼 크기 확대, 마커 드래그/키 이동, CSV/TXT 내보내기 유지
"""

import os, re, sys, csv, traceback, subprocess, platform
import numpy as np
from pathlib import Path
from tkinter import (
    Tk, ttk, filedialog, StringVar, BooleanVar, DoubleVar, Entry, Label,
    Button, Checkbutton, messagebox, scrolledtext, Menu, Frame
)
import tkinter.font as tkfont

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Mini toolbar: Home + Zoom only ---
class MiniToolbar(NavigationToolbar2Tk):
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    )

Z0_DEFAULT = 50.0

# ---------- utils ----------
def moving_average(x, win:int):
    win = max(1, int(win))
    if win <= 1:
        return np.asarray(x, float).copy()
    return np.convolve(np.asarray(x, float), np.ones(win)/win, mode="same")

def median_abs_dev(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return np.median(np.abs(x - med))

def df_mhz_from(f_mhz):
    if len(f_mhz) < 2: return None
    return float(np.median(np.diff(f_mhz)))

def odd_round(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1

# ---------- touchstone ----------
_HDR_RE = re.compile(r"^\s*#\s+(\w+)\s+(\w)\s+(\w\w)\s+R\s+([\d\.Ee\+\-]+)", re.IGNORECASE)

def parse_s1p(path:Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    freq_unit, s_format, data_fmt, z0 = "HZ", "S", "RI", Z0_DEFAULT
    data=[]
    for line in lines:
        line=line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            m=_HDR_RE.match(line)
            if m:
                freq_unit, s_format, data_fmt, z0s = m.groups()
                freq_unit=freq_unit.upper(); data_fmt=data_fmt.upper()
                try: z0=float(z0s)
                except: z0=Z0_DEFAULT
            continue
        toks=line.split()
        if len(toks)<3: continue
        try:
            f=float(toks[0]); a=float(toks[1]); b=float(toks[2])
        except: continue
        data.append((f,a,b))
    if not data:
        raise ValueError(f"No numeric rows in {path.name}")
    arr=np.array(data,float)
    scale={"HZ":1.0,"KHZ":1e3,"MHZ":1e6,"GHZ":1e9}.get(freq_unit.upper(),1.0)
    f_hz=arr[:,0]*scale
    if data_fmt=="RI":
        s11=arr[:,1]+1j*arr[:,2]
    elif data_fmt=="MA":
        mag=arr[:,1]; ang=np.deg2rad(arr[:,2]); s11=mag*np.exp(1j*ang)
    elif data_fmt=="DB":
        mag=10**(arr[:,1]/20.0); ang=np.deg2rad(arr[:,2]); s11=mag*np.exp(1j*ang)
    else:
        raise ValueError(f"Unsupported data format {data_fmt}")
    return f_hz, s11, z0

def s_to_y_norm(s11: np.ndarray):
    eps=1e-12
    denom=(1.0+s11); denom[np.abs(denom)<eps]=eps
    return (1.0 - s11)/denom

# ---------- guards (LEGACY) ----------
def freq_to_index_window(f_hz, fmin_hz, fmax_hz):
    idx = np.where((f_hz>=fmin_hz) & (f_hz<=fmax_hz))[0]
    if idx.size==0: return np.array([],dtype=int)
    return idx

def contact_guard_s11_db(f_hz, s11, roi, excludes, smooth_mhz=3.0, rms_max=0.40, spike_ratio_max=0.02):
    idx_roi=freq_to_index_window(f_hz, roi[0], roi[1])
    if idx_roi.size==0:
        return {"pass":False,"reason":"ROI empty","metrics":{}}
    mask=np.zeros_like(f_hz,dtype=bool); mask[idx_roi]=True
    for ex in excludes:
        ie=freq_to_index_window(f_hz,ex[0],ex[1]); mask[ie]=False
    idx=np.where(mask)[0]
    if idx.size<5:
        return {"pass":False,"reason":"ROI after exclude too small","metrics":{}}
    f=f_hz[idx]; s11_db=20*np.log10(np.abs(s11[idx])+1e-15)
    df_mhz=np.median(np.diff(f))/1e6
    win=max(1,int(smooth_mhz/df_mhz))
    s_smooth=moving_average(s11_db,win)
    resid=s11_db - s_smooth
    rms=float(np.sqrt(np.mean(resid**2)))
    d=np.diff(s11_db)/np.diff(f/1e6)
    med=np.median(np.abs(d)); mad=median_abs_dev(np.abs(d))+1e-12
    thr=med+5.0*mad
    spike_ratio=float(np.mean(np.abs(d)>thr))
    ok_rms=rms<=rms_max
    ok_spike=spike_ratio<=spike_ratio_max
    passed=(ok_rms and ok_spike)
    reason=[]
    if not ok_rms: reason.append(f"resid_rms={rms:.3f}>{rms_max:.3f}")
    if not ok_spike: reason.append(f"spike_ratio={spike_ratio:.3f}>{spike_ratio_max:.3f}")
    if not reason: reason.append("OK")
    return {"pass":passed,"reason":"; ".join(reason),
            "metrics":{"rms":rms,"spike_ratio":spike_ratio,"df_mhz":float(df_mhz),"win_pts":win}}

def linearity_guard_y11_db(f_hz, s11, roi, excludes, smooth_mhz=3.0, tau_db_per_mhz=0.008, run_amp_min_db=6.0, L_min_mhz=200.0):
    y=s_to_y_norm(s11); y_db=20*np.log10(np.abs(y)+1e-15)
    idx_roi=freq_to_index_window(f_hz,roi[0],roi[1])
    if idx_roi.size==0:
        return {"pass":False,"reason":"ROI empty","metrics":{}}
    mask=np.zeros_like(f_hz,dtype=bool); mask[idx_roi]=True
    for ex in excludes:
        ie=freq_to_index_window(f_hz,ex[0],ex[1]); mask[ie]=False
    idx=np.where(mask)[0]
    if idx.size<5:
        return {"pass":True,"reason":"ROI too small → skip","metrics":{}}
    f=f_hz[idx]; df_mhz=np.median(np.diff(f))/1e6
    win=max(1,int(smooth_mhz/df_mhz))
    y_s=moving_average(y_db[idx],win)
    dy=np.diff(y_s); df=np.diff(f/1e6)
    slope=np.abs(dy/df)
    ok=(slope<=tau_db_per_mhz)
    spans=[]; cur=0
    for i,flag in enumerate(ok):
        if flag: cur+=1
        elif cur>0: spans.append((i-cur,i-1)); cur=0
    if cur>0: spans.append((len(ok)-cur,len(ok)-1))
    largest_run_mhz=0.0; offending=None
    for s,e in spans:
        f0=f[s]; f1=f[e+1] if (e+1)<len(f) else f[e]
        run_len=(f1-f0)/1e6
        amp=float(np.max(y_s[s:e+1]) - np.min(y_s[s:e+1]))
        if (run_len>=L_min_mhz) and (amp>=run_amp_min_db):
            if run_len>largest_run_mhz:
                largest_run_mhz=run_len; offending=(f0/1e6,f1/1e6,amp)
    if offending is not None:
        f0,f1,amp=offending
        return {"pass":False,
                "reason":f"largest_run={largest_run_mhz:.0f} MHz >= L_min={L_min_mhz:.0f} MHz, amp={amp:.2f} dB",
                "metrics":{"largest_run_mhz":largest_run_mhz}}
    return {"pass":True,"reason":"OK","metrics":{}}
# ---------- 레벨 교차 ----------
def _find_level_crossings(f_mhz, y_db, level_db, fs_mhz):
    try:
        f_mhz = np.asarray(f_mhz, float)
        y_db = np.asarray(y_db, float)
        n = f_mhz.size
        if n < 2 or n != y_db.size:
            return (None, None)
        y_src = y_db
        target = float(level_db)
        if fs_mhz is None or not np.isfinite(fs_mhz):
            i_split = n // 2
        else:
            i_split = int(np.argmin(np.abs(f_mhz - fs_mhz)))
        def _cross_on_segment(i0, i1, prefer_left=True):
            if i0 > i1: i0, i1 = i1, i0
            best = None
            for i in range(i0, i1):
                y0, y1 = y_src[i], y_src[i+1]
                d0, d1 = y0 - target, y1 - target
                if np.isnan(d0) or np.isnan(d1): continue
                if d0 == 0.0: best = (0.0, f_mhz[i]); break
                if d1 == 0.0: best = (0.0, f_mhz[i+1]); break
                if d0 * d1 < 0.0:
                    denom = (y1 - y0)
                    frac = 0.0 if abs(denom) < 1e-15 else (target - y0) / denom
                    frac = float(np.clip(frac, 0.0, 1.0))
                    f_cross = f_mhz[i] + frac * (f_mhz[i+1] - f_mhz[i])
                    cand = (abs(target - (y0 + frac*(y1-y0))), f_cross)
                    if best is None: best = cand
                    else:
                        if prefer_left:
                            if i > i0: best = cand
                        else:
                            if i < i1-1: best = cand
            if best is not None:
                return best[1]
            j = np.arange(i0, i1+1)
            yseg = y_src[j]
            if yseg.size == 0 or np.all(np.isnan(yseg)): return None
            k = int(np.nanargmin(np.abs(yseg - target)))
            idx = j[k]
            neigh = idx-1 if prefer_left else idx+1
            if neigh < 0 or neigh >= n or np.isnan(y_src[neigh]) or np.isnan(y_src[idx]) or f_mhz[neigh]==f_mhz[idx]:
                return float(f_mhz[idx])
            y0, y1 = y_src[idx], y_src[neigh]
            f0, f1 = f_mhz[idx], f_mhz[neigh]
            denom = (y1 - y0)
            frac = 0.0 if abs(denom) < 1e-15 else (target - y0) / denom
            frac = float(np.clip(frac, 0.0, 1.0))
            return float(f0 + frac * (f1 - f0))
        f_left = _cross_on_segment(0, i_split, prefer_left=True)
        f_right = _cross_on_segment(i_split, n-1, prefer_left=False)
        return (f_left, f_right)
    except Exception:
        return (None, None)

def _find_minus10db_crossings_mhz(f_mhz, y_db, fs_mhz, yfs_db, level_db):
    return _find_level_crossings(f_mhz, y_db, level_db, fs_mhz)

# ---------- marker search ----------
def _quad_refine_k(x, y, idx, k=2, mode='max'):
    n = len(x)
    i0 = max(0, idx - k)
    i1 = min(n - 1, idx + k)
    if i1 - i0 < 2:
        return float(x[idx]), float(y[idx])
    xs = np.asarray(x[i0:i1+1], float)
    ys = np.asarray(y[i0:i1+1], float)
    A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    try:
        a, b, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        if abs(a) < 1e-24:
            return float(x[idx]), float(y[idx])
        x_v = -b/(2*a)
        if x_v < xs[0] - 1e-9 or x_v > xs[-1] + 1e-9:
            return float(x[idx]), float(y[idx])
        y_v = a*x_v*x_v + b*x_v + c
        if (mode == 'max' and a < 0) or (mode == 'min' and a > 0):
            return float(x_v), float(y_v)
        return float(x[idx]), float(y[idx])
    except Exception:
        return float(x[idx]), float(y[idx])

def find_fs_fp_y11(f_hz, y11_db, roi_hz, smooth_mhz_for_coarse=8.0, fs_search_span_mhz=1200.0):
    idx = np.where((f_hz >= roi_hz[0]) & (f_hz <= roi_hz[1]))[0]
    if idx.size < 8:
        return None, None, None, None
    f_roi_hz = f_hz[idx].astype(float)
    y_raw_db = y11_db[idx].astype(float)
    df_mhz = float(np.median(np.diff(f_roi_hz)) / 1e6)
    if not np.isfinite(df_mhz) or df_mhz <= 0:
        return None, None, None, None
    win_pts = max(5, int(round(smooth_mhz_for_coarse / df_mhz)))
    if win_pts % 2 == 0: win_pts += 1
    win_pts = max(5, min(151, win_pts))
    y_s = moving_average(y_raw_db, win_pts)
    i_fp_c = int(np.nanargmin(y_s))
    fp_hz, yfp_db = _quad_refine_k(f_roi_hz, y_raw_db, i_fp_c, k=2, mode='min')
    span_hz = float(fs_search_span_mhz) * 1e6
    j = np.where((f_roi_hz >= fp_hz - span_hz) & (f_roi_hz <= fp_hz + span_hz))[0]
    if j.size >= 3:
        seg_raw = y_raw_db[j]
        mx = np.nanmax(seg_raw)
        cand = np.where(np.isfinite(seg_raw) & (seg_raw == mx))[0]
        if cand.size > 1:
            k_mid = int(np.round((cand[0] + cand[-1]) / 2.0))
            i_fs_c = int(j[0] + k_mid)
        else:
            i_fs_c = int(j[0] + int(np.nanargmax(seg_raw)))
    else:
        i_fs_c = int(np.nanargmax(y_raw_db))
    fs_hz, yfs_db = _quad_refine_k(f_roi_hz, y_raw_db, i_fs_c, k=2, mode='max')
    return fs_hz/1e6, fp_hz/1e6, yfs_db, yfp_db

# ============ Human-Shape & AutoBand ============
def _moving_rms(y, win_pts:int):
    if win_pts < 2: return np.zeros_like(y)
    c = np.convolve(y**2, np.ones(win_pts)/win_pts, 'same')
    c = np.maximum(c, 0)
    return np.sqrt(c)

def _shape_features(freq_mhz, y_db):
    n = len(y_db)
    if n < 7: return None
    df = float(np.mean(np.diff(freq_mhz)))
    sm_win = max(5, int(round(15.0/df)))
    y_s = moving_average(y_db, sm_win)
    dy = np.gradient(y_s, df)
    d2 = np.gradient(dy, df)
    fs_idx = int(np.argmax(y_s))
    fp_idx = int(np.argmin(y_s))
    rms_global = float(np.std(y_db))
    win_pts = max(5, int(round(30.0/df)))
    _ = _moving_rms(y_db, win_pts)
    mad_d2 = median_abs_dev(np.abs(d2)) + 1e-12
    spike_cnt = int(np.sum(np.abs(d2) > 10.0*mad_d2))
    kappa = np.abs(d2)/(1.0 + dy*dy)**1.5
    half = max(3, int(round(50.0/df)))
    L = max(0, fp_idx-half); R = min(n, fp_idx+half+1)
    kappa_mean = float(np.mean(kappa[L:R]))
    L2 = max(0, fp_idx - int(round(150.0/df))); R2 = min(n, fp_idx + int(round(150.0/df)))
    left = slice(L2, fp_idx); right = slice(fp_idx, R2)
    asym_grad = float(abs(np.mean(dy[left]) + np.mean(dy[right])))
    thr = 0.004  # dB/MHz
    def _run_len(side):
        g = dy[left] if side=="L" else dy[right]
        cnt = 0; best = 0
        if side == "L":
            for v in g:
                if v > +thr: cnt += 1; best = max(best, cnt)
                else: cnt = 0
        else:
            for v in g:
                if v < -thr: cnt += 1; best = max(best, cnt)
                else: cnt = 0
        return float(best * df)
    mono_L_mhz = _run_len("L"); mono_R_mhz = _run_len("R")
    BL0 = max(0, fs_idx - int(round(400.0/df)))
    BL1 = max(5, fs_idx - int(round(200.0/df)))
    base_tilt = float(np.mean(dy[BL0:BL1])) if (BL1-BL0)>5 else 0.0
    range_db = float(np.max(y_db) - np.min(y_db))
    return {
        "fs_idx": fs_idx, "fp_idx": fp_idx, "rms_global": rms_global,
        "spike_cnt": spike_cnt, "kappa_mean": kappa_mean, "asym_grad": asym_grad,
        "mono_L_mhz": mono_L_mhz, "mono_R_mhz": mono_R_mhz, "base_tilt": base_tilt,
        "range_db": range_db, "yfs_db": float(y_db[fs_idx]), "yfp_db": float(y_db[fp_idx]),
    }

LEVEL1 = {
    "rms_global_max": 6.0, "spike_cnt_max": 8, "kappa_min": 0.002, "asym_grad_max": 0.015,
    "mono_L_min_mhz": 80.0, "mono_R_min_mhz": 80.0, "base_tilt_max": 0.02, "range_db_min": 20.0,
}

def _human_like_verdict(feat:dict):
    reasons=[]
    if feat["rms_global"] > LEVEL1["rms_global_max"]: reasons.append(f"NoiseRMS>{LEVEL1['rms_global_max']}")
    if feat["spike_cnt"] > LEVEL1["spike_cnt_max"]: reasons.append("Spike>limit")
    if feat["kappa_mean"] < LEVEL1["kappa_min"]: reasons.append("Shallow_dip(curvature)")
    if abs(feat["asym_grad"]) > LEVEL1["asym_grad_max"]: reasons.append("Asymmetry_excess")
    if feat["mono_L_mhz"] < LEVEL1["mono_L_min_mhz"]: reasons.append("Left_not_monotonic")
    if feat["mono_R_mhz"] < LEVEL1["mono_R_min_mhz"]: reasons.append("Right_not_monotonic")
    if abs(feat["base_tilt"]) > LEVEL1["base_tilt_max"]: reasons.append("Baseline_tilt")
    if feat["range_db"] < LEVEL1["range_db_min"]: reasons.append("Too_flat_range")
    if reasons: return "FAIL", " | ".join(reasons)
    return "PASS", "Shape_OK"

def _trim_by_quantile(G:np.ndarray, q=0.1):
    if G.size == 0: return G
    low = np.quantile(G, q, axis=0)
    high = np.quantile(G, 1.0-q, axis=0)
    mask = np.all((G>=low) & (G<=high), axis=1)
    return G[mask]

def _mad_normalized(x): return 1.4826 * median_abs_dev(x)

def _quality_score(feat: dict) -> float:
    tgt = { "rms_global": 1.2, "spike_cnt": 3.0, "kappa_mean": 0.008, "asym_grad": 0.006,
            "mono_L_mhz": 80.0, "mono_R_mhz": 80.0, "range_db": 10.0 }
    p = []
    p.append(max(0.0, (feat["rms_global"] - tgt["rms_global"]) / max(0.3, tgt["rms_global"])) )
    p.append(max(0.0, (feat["spike_cnt"] - tgt["spike_cnt"]) / max(1.0, tgt["spike_cnt"])) )
    p.append(max(0.0, (tgt["kappa_mean"] - feat["kappa_mean"]) / max(1e-3, tgt["kappa_mean"])) )
    p.append(max(0.0, (abs(feat["asym_grad"]) - tgt["asym_grad"]) / max(1e-3, tgt["asym_grad"])) )
    p.append(max(0.0, (tgt["mono_L_mhz"] - feat["mono_L_mhz"]) / max(10.0, tgt["mono_L_mhz"])) )
    p.append(max(0.0, (tgt["mono_R_mhz"] - feat["mono_R_mhz"]) / max(10.0, tgt["mono_R_mhz"])) )
    p.append(max(0.0, (tgt["range_db"] - feat["range_db"]) / max(2.0, tgt["range_db"])) )
    return float(np.sum(np.square(p)))
# ---------- App ----------
class App:
    def __init__(self, m:Tk):
        self.m=m
        self.m.title("Resonator QC Unified v4.0 (HJ)")
        self.m.geometry("1460x980")

        # 공통 버튼 폰트 & 크기
        try:
            self.btn_font = tkfont.Font(family="Segoe UI", size=11)
        except Exception:
            self.btn_font = tkfont.nametofont("TkDefaultFont")
            self.btn_font.configure(size=11)

        # ---------- params ----------
        self.root_dir=StringVar(value="")
        self.roi_min=DoubleVar(value=100.0)
        self.roi_max=DoubleVar(value=8500.0)
        self.exclude_text=StringVar(value="")

        # 잠금 탭 파라미터들
        self.smooth_mhz_qc=DoubleVar(value=3.0)
        self.s11_rms_max=DoubleVar(value=0.40)
        self.s11_spike_ratio_max=DoubleVar(value=0.02)
        self.y11_tau=DoubleVar(value=0.008)
        self.y11_run_amp=DoubleVar(value=6.0)
        self.y11_L_min=DoubleVar(value=200.0)

        self.dual_relax=BooleanVar(value=True)
        self.show_s11=BooleanVar(value=False)
        self.show_y11=BooleanVar(value=True)
        self.zoom_roi=BooleanVar(value=True)
        self.ref_smooth_val=StringVar(value="14141")
        self.ref_smooth_mode=StringVar(value="auto")
        self.batch_adapt=BooleanVar(value=False)
        self.decision_mode = StringVar(value="Legacy (S11+Y11)")
        self.autoband_k = DoubleVar(value=2.5)
        self.autoband_auto_k = BooleanVar(value=True)
        self.cross_level_db = DoubleVar(value=-10.0)

        self._bands = None

        # ---------- Notebook (상단) ----------
        self.tabs = ttk.Notebook(m)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        m.grid_rowconfigure(0, weight=0)
        m.grid_columnconfigure(0, weight=1)

        self.tab_basic = Frame(self.tabs)
        self.tab_locked = Frame(self.tabs)
        self.tabs.add(self.tab_basic, text="기본")
        self.tabs.add(self.tab_locked, text="고급(잠금)")

        # ---------- BASIC TAB ----------
        self._build_basic(self.tab_basic)

        # ---------- LOCKED TAB ----------
        self._build_locked(self.tab_locked)

        # ---------- 하단 공통: 플롯 → 툴바 → 로그 ----------
        self.fig=plt.Figure(figsize=(8.6, 4.5))
        self.ax=self.fig.add_subplot(111)
        self.canvas=FigureCanvasTkAgg(self.fig,master=m)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=(0,4))
        m.grid_rowconfigure(1, weight=1)

        self.toolbar = MiniToolbar(self.canvas, m, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=2, column=0, sticky="w", padx=6, pady=(0,4))

        self.log=scrolledtext.ScrolledText(m, height=8)
        self.log.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0,6))
        try:
            self.log.configure(font=("Consolas", 10))
        except Exception:
            try: self.log.configure(font=("Courier New", 10))
            except Exception: pass
        self._write_log_header(found=None)

        # 컨텍스트 메뉴
        self.ctx = Menu(m, tearoff=0)
        self.ctx.add_command(label="Open file (default app)", command=self._ctx_open_default)
        self.ctx.add_command(label="Open in Explorer (select)", command=self._ctx_open_in_explorer)
        self.ctx.add_command(label="Copy full path", command=self._ctx_copy_path)
        self._ctx_tree = None

        # Marker 상태 & 이벤트
        self._markers = []
        self._active_idx = None
        self._last_plot_data = {}
        self._dragging_idx = None

        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('key_press_event', self._on_plot_key)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

        # 초기 프롬프트
        self.ax.set_axis_on()
        self.ax.text(0.5,0.5,"(분석 폴더 선택 후 '분석 시작')",ha="center",va="center",transform=self.ax.transAxes)
        self.fig.tight_layout(); self.canvas.draw_idle()

    # 공통 큰 버튼 생성기
    def _mkbtn(self, parent, **kwargs):
        b = Button(parent, **kwargs)
        try:
            b.configure(font=self.btn_font)
        except Exception:
            pass
        b.grid_configure(ipadx=8, ipady=4)
        return b

    # ---------------- UI BUILDERS ----------------
    def _build_basic(self, root:Frame):
        row = 0
        # 1행: 분석 폴더 + Clear
        self._mkbtn(root, text="분석 폴더", command=self.pick_folder).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        Entry(root, textvariable=self.root_dir, width=80).grid(row=row, column=1, columnspan=6, sticky="we", padx=4, pady=2)
        self._mkbtn(root, text="Clear List", command=self.clear_all).grid(row=row, column=7, sticky="w", padx=4, pady=2)
        root.grid_columnconfigure(6, weight=1)

        # 2행: ROI/Exclude + 분석 시작/보기
        row += 1
        Label(root,text="ROI (MHz)").grid(row=row,column=0,sticky="w",padx=4)
        Entry(root,textvariable=self.roi_min,width=8).grid(row=row,column=1,sticky="w")
        Entry(root,textvariable=self.roi_max,width=8).grid(row=row,column=2,sticky="w")

        Label(root,text="Exclude a-b, c-d").grid(row=row,column=3,sticky="e")
        Entry(root,textvariable=self.exclude_text,width=28).grid(row=row,column=4,sticky="we",padx=2)

        self._mkbtn(root, text="분석 시작", command=self.run).grid(row=row, column=5, sticky="w", padx=4, pady=2)

        optbox = Frame(root)
        optbox.grid(row=row, column=6, columnspan=3, sticky="w")
        Checkbutton(optbox, text="S11 보기", variable=self.show_s11, command=self.update_preview).pack(side="left", padx=(0,6))
        Checkbutton(optbox, text="Y11 보기", variable=self.show_y11, command=self.update_preview).pack(side="left", padx=(0,6))
        Checkbutton(optbox, text="ROI로 줌", variable=self.zoom_roi, command=self.update_preview).pack(side="left")

        # 3행: Decision/학습 + 통계
        row += 1
        Label(root, text="Decision mode").grid(row=row,column=0,sticky="e",padx=4)
        self.cb_dec = ttk.Combobox(root, values=[
            "Legacy (S11+Y11)",
            "Human-Shape (Rule-only)",
            "AutoBand (Zero-Label)"
        ], textvariable=self.decision_mode, width=22, state="readonly")
        self.cb_dec.grid(row=row,column=1,sticky="w")

        self._mkbtn(root, text="Train AutoBand", command=self._train_autoband_from_current_scan)\
            .grid(row=row, column=2, sticky="w", padx=4, pady=2)
        self._mkbtn(root, text="Train (Selected)", command=self._train_from_selected)\
            .grid(row=row, column=3, sticky="w", padx=4, pady=2)

        self.stat_total=StringVar(value="총 0")
        self.stat_pass=StringVar(value="PASS 0 (0.0%)")
        self.stat_fail=StringVar(value="FAIL 0 (0.0%)")

        stats = Frame(root)
        stats.grid(row=row, column=5, columnspan=4, sticky="w", padx=(10,0))
        Label(stats, textvariable=self.stat_total, foreground="#333").pack(side="left", padx=(0,8))
        Label(stats, textvariable=self.stat_pass,  foreground="#1e7e34").pack(side="left", padx=(0,8))
        Label(stats, textvariable=self.stat_fail,  foreground="#b02a37").pack(side="left")

        # 4행: 기본 탭 전용 리스트(Reason 숨김 = file, verdict)
        row += 1
        self.nb_list_basic = ttk.Notebook(root)
        self.nb_list_basic.grid(row=row, column=0, columnspan=9, sticky="nsew", padx=4, pady=(6,6))
        root.grid_rowconfigure(row, weight=1)

        self.tree_all_basic = self._make_tree(self.nb_list_basic, "All (Basic)", cols=("file","verdict"), height=14)
        self.tree_pass_basic= self._make_tree(self.nb_list_basic, "PASS (Basic)", cols=("file","verdict"), height=14)
        self.tree_fail_basic= self._make_tree(self.nb_list_basic, "FAIL (Basic)", cols=("file","verdict"), height=14)

        for t in (self.tree_all_basic,self.tree_pass_basic,self.tree_fail_basic):
            t.bind("<<TreeviewSelect>>", self.on_select)
            t.bind("<Configure>", lambda e, tree=t: self._autosize_columns(tree))
            t.bind("<Double-1>", lambda e, tree=t: self._open_selected_default(tree))
            t.bind("<Button-3>", lambda e, tree=t: self._on_tree_right_click(e, tree))

        self.nb_list_basic.bind("<<NotebookTabChanged>>", lambda e: self.update_preview())

        # column stretch
        for c in range(0,9):
            if c == 4:
                root.grid_columnconfigure(c, weight=1)

    def _build_locked(self, root:Frame):
        # 상단: Unlock 바
        bar = Frame(root)
        bar.grid(row=0, column=0, sticky="we", padx=6, pady=(4,0))
        Label(bar, text="고급 설정은 잠금 상태입니다. 비밀번호를 입력하세요.").grid(row=0, column=0, sticky="w")
        self._pwd = StringVar()
        Entry(bar, textvariable=self._pwd, width=28, show="*").grid(row=0, column=1, padx=6)
        self._mkbtn(bar, text="Unlock", command=self._do_unlock).grid(row=0, column=2, padx=(2,2))
        self._mkbtn(bar, text="Lock", command=self._do_lock).grid(row=0, column=3, padx=(2,2))
        bar.grid_columnconfigure(0, weight=1)

        # 내용 프레임(잠금 대상)
        self._locked_content = Frame(root, relief="groove", borderwidth=1)
        self._locked_content.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # 마스크 프레임
        self._mask = Frame(root, bg="#f0f0f0")
        Label(self._mask, text="🔒 잠금 상태입니다.\nUnlock 후 설정을 변경하세요.", bg="#f0f0f0",
              font=("Segoe UI", 12)).pack(expand=True, fill="both")
        self._locked=True
        self._show_lock_mask(True)

        # --- 고급 파라미터 ---
        row = 0
        r = self._locked_content
        Label(r,text="QC Smooth (MHz)").grid(row=row,column=0,sticky="w",padx=4)
        Entry(r,textvariable=self.smooth_mhz_qc,width=8,state="disabled").grid(row=row,column=1,sticky="w")
        Label(r,text="S11 RMS(dB) max").grid(row=row,column=2,sticky="w")
        Entry(r,textvariable=self.s11_rms_max,width=8,state="disabled").grid(row=row,column=3,sticky="w")
        Label(r,text="S11 Spike ratio max").grid(row=row,column=4,sticky="w")
        Entry(r,textvariable=self.s11_spike_ratio_max,width=8,state="disabled").grid(row=row,column=5,sticky="w")
        row+=1
        Label(r,text="Y11 tau(dB/MHz)").grid(row=row,column=0,sticky="w",padx=4)
        Entry(r,textvariable=self.y11_tau,width=8,state="disabled").grid(row=row,column=1,sticky="w")
        Label(r,text="run_amp_min(dB)").grid(row=row,column=2,sticky="w")
        Entry(r,textvariable=self.y11_run_amp,width=8,state="disabled").grid(row=row,column=3,sticky="w")
        Label(r,text="L_min(MHz)").grid(row=row,column=4,sticky="w")
        Entry(r,textvariable=self.y11_L_min,width=8,state="disabled").grid(row=row,column=5,sticky="w")
        row+=1

        Label(r, text="AutoBand k").grid(row=row,column=0,sticky="e")
        Entry(r, textvariable=self.autoband_k, width=6, state="disabled").grid(row=row,column=1,sticky="w")
        Checkbutton(r, text="Auto k", variable=self.autoband_auto_k, state="disabled")\
            .grid(row=row, column=2, sticky="w", padx=4)

        Label(r, text="Cross @ (dB)").grid(row=row, column=3, sticky="e")
        Entry(r, textvariable=self.cross_level_db, width=6, state="disabled").grid(row=row, column=4, sticky="w", padx=2)
        row+=1

        Label(r,text="Ref smooth").grid(row=row,column=0,sticky="e")
        Entry(r,textvariable=self.ref_smooth_val,width=8,state="disabled").grid(row=row,column=1,sticky="w")
        self.cb_mode=ttk.Combobox(r,values=["pts","MHz","auto"],textvariable=self.ref_smooth_mode,width=6,state="disabled")
        self.cb_mode.grid(row=row,column=2,sticky="w")
        Checkbutton(r,text="Batch-adapt",variable=self.batch_adapt,state="disabled").grid(row=row,column=3,sticky="w")
        self._mkbtn(r, text="Export TXT", command=self.export_txt).grid(row=row, column=4, sticky="w", padx=4)

        # 통계 (공유 변수 바인딩)
        statbox = Frame(r)
        statbox.grid(row=row, column=5, columnspan=3, sticky="w", padx=(10,0))
        Label(statbox, textvariable=self.stat_total, foreground="#333").pack(side="left", padx=(0,8))
        Label(statbox, textvariable=self.stat_pass,  foreground="#1e7e34").pack(side="left", padx=(0,8))
        Label(statbox, textvariable=self.stat_fail,  foreground="#b02a37").pack(side="left")

        # 2) 고급 탭 전용 리스트(Reason 포함 = file, verdict, reason)
        row += 1
        self.nb_list_adv = ttk.Notebook(r)
        self.nb_list_adv.grid(row=row, column=0, columnspan=8, sticky="nsew", padx=4, pady=(10,6))
        r.grid_rowconfigure(row, weight=1)

        self.tree_all_adv = self._make_tree(self.nb_list_adv, "All (Advanced)", cols=("file","verdict","reason"), height=14)
        self.tree_pass_adv= self._make_tree(self.nb_list_adv, "PASS (Advanced)", cols=("file","verdict","reason"), height=14)
        self.tree_fail_adv= self._make_tree(self.nb_list_adv, "FAIL (Advanced)", cols=("file","verdict","reason"), height=14)

        for t in (self.tree_all_adv,self.tree_pass_adv,self.tree_fail_adv):
            t.bind("<<TreeviewSelect>>", self.on_select)
            t.bind("<Configure>", lambda e, tree=t: self._autosize_columns(tree))
            t.bind("<Double-1>", lambda e, tree=t: self._open_selected_default(tree))
            t.bind("<Button-3>", lambda e, tree=t: self._on_tree_right_click(e, tree))

        self.nb_list_adv.bind("<<NotebookTabChanged>>", lambda e: self.update_preview())

        for c in range(0,8):
            r.grid_columnconfigure(c, weight=1 if c==6 else 0)

    def _show_lock_mask(self, show:bool):
        if show:
            self._mask.place(in_=self._locked_content, relx=0, rely=0, relwidth=1, relheight=1)
            for w in self._locked_content.winfo_children():
                try:
                    if isinstance(w, (Entry, ttk.Combobox, Checkbutton, Button)):
                        w.configure(state="disabled")
                except Exception:
                    pass
        else:
            self._mask.place_forget()
            for w in self._locked_content.winfo_children():
                try:
                    if isinstance(w, (Entry, ttk.Combobox, Checkbutton, Button)):
                        w.configure(state="normal")
                except Exception:
                    pass

    def _do_unlock(self):
        pwd = self._pwd.get().strip()
        if pwd == "hj12345":
            self._locked=False
            self._show_lock_mask(False)
            messagebox.showinfo("OK","고급 설정 Unlock")
        else:
            messagebox.showerror("Error","비밀번호가 올바르지 않습니다.")

    def _do_lock(self):
        self._locked=True
        self._show_lock_mask(True)
        messagebox.showinfo("Locked","고급 설정이 잠겼습니다.")

    # ------------ Tree helpers / context menu ------------
    def _make_tree(self, parent, label, cols=("file","verdict","reason"), height=10):
        frame=ttk.Frame(parent); parent.add(frame, text=label)
        tree=ttk.Treeview(frame,columns=cols,show="headings",selectmode="extended", height=height)
        for c in cols:
            tree.heading(c,text=c)
            tree.column(c, width=120)
        vsb=ttk.Scrollbar(frame,orient="vertical",command=tree.yview)
        hsb=ttk.Scrollbar(frame,orient="horizontal",command=tree.xview)
        tree.configure(yscrollcommand=vsb.set,xscrollcommand=hsb.set)
        tree.grid(row=0,column=0,sticky="nsew"); vsb.grid(row=0,column=1,sticky="ns")
        hsb.grid(row=1,column=0,sticky="ew")
        frame.grid_rowconfigure(0,weight=1); frame.grid_columnconfigure(0,weight=1)
        tree.tag_configure('PASS',background='#e9f7ef'); tree.tag_configure('FAIL',background='#fdecea')
        return tree

    def _autosize_columns(self, tree):
        try:
            fnt=tkfont.nametofont("TkDefaultFont"); pad=24
            cols = tree["columns"]
            widths={c: fnt.measure(c)+pad for c in cols}
            for iid in tree.get_children(""):
                vals=tree.item(iid,"values")
                for c, v in zip(cols, vals):
                    widths[c]=max(widths[c], fnt.measure(str(v))+pad)
            for c in cols:
                tree.column(c, width=widths[c])
        except Exception:
            pass

    def _records_from_names(self, names):
        if not names: return []
        name_set = set(names)
        return [r for r in self.results if Path(r["file"]).name in name_set]

    def _selected_names_in_tree(self, tree):
        return [tree.item(i, "values")[0] for i in tree.selection()]

    def _open_paths_default(self, paths):
        for p in paths:
            try:
                if platform.system() == "Windows":
                    os.startfile(p)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", p], check=False)
                else:
                    subprocess.run(["xdg-open", p], check=False)
            except Exception as e:
                self._log(f"[WARN] open failed: {p} ({e})\n")

    def _open_in_explorer_select(self, paths):
        for p in paths:
            try:
                if platform.system() == "Windows":
                    subprocess.run(["explorer", "/select,", p], check=False)
                else:
                    folder = str(Path(p).parent)
                    if platform.system() == "Darwin":
                        subprocess.run(["open", folder], check=False)
                    else:
                        subprocess.run(["xdg-open", folder], check=False)
            except Exception as e:
                self._log(f"[WARN] explorer open failed: {p} ({e})\n")

    def _copy_to_clipboard(self, text):
        try:
            self.m.clipboard_clear()
            self.m.clipboard_append(text)
            self._log("[OK] Copied to clipboard.\n")
        except Exception as e:
            self._log(f"[WARN] clipboard failed: {e}\n")

    def _on_tree_right_click(self, event, tree):
        iid = tree.identify_row(event.y)
        if iid and iid not in tree.selection():
            tree.selection_add(iid)
        self._ctx_tree = tree
        try:
            self.ctx.tk_popup(event.x_root, event.y_root)
        finally:
            self.ctx.grab_release()

    def _ctx_open_default(self):
        tree = getattr(self, "_ctx_tree", None)
        if not tree: return
        names = self._selected_names_in_tree(tree)
        recs = self._records_from_names(names)
        paths = [r["file"] for r in recs]
        if not paths: return
        self._open_paths_default(paths)

    def _ctx_open_in_explorer(self):
        tree = getattr(self, "_ctx_tree", None)
        if not tree: return
        names = self._selected_names_in_tree(tree)
        recs = self._records_from_names(names)
        paths = [r["file"] for r in recs]
        if not paths: return
        self._open_in_explorer_select(paths)

    def _ctx_copy_path(self):
        tree = getattr(self, "_ctx_tree", None)
        if not tree: return
        names = self._selected_names_in_tree(tree)
        recs = self._records_from_names(names)
        paths = [r["file"] for r in recs]
        if not paths: return
        self._copy_to_clipboard("\n".join(paths))
    # ------------ UI helpers / log ------------
    def _write_log_header(self, found:int|None):
        self.log.delete("1.0","end")
        self._log("Ready.\n")
        if isinstance(found,int):
            self._log(f"Found {found} files.\n")
        self._log(f"{'Fs(MHz)':>10} {'Y@Fs(dB)':>10} {'Fp(MHz)':>12} {'Y@Fp(dB)':>10} {'Kt2':>8} file\n")

    def _log_line(self, fs, yfs, fp, yfp, filename):
        if (fs is None) or (fp is None) or (yfs is None) or (yfp is None):
            self._log(f"{'Fs:':<4}{'NA':>8} {'NA':>10} {'Fp:':<4}{'NA':>8} {'NA':>10} {'Kt2:':<5}{'NA':>8} {filename}\n")
        else:
            kt2 = 1.0 - (fs/ fp)**2
            self._log(f"{'Fs:':<4}{fs:>8.3f} {yfs:>10.3f} {'Fp:':<4}{fp:>8.3f} {yfp:>10.3f} {'Kt2:':<5}{kt2:>8.4f} {filename}\n")

    def _log(self,s):
        self.log.insert("end",s); self.log.see("end")

    def pick_folder(self):
        d=filedialog.askdirectory()
        if d:
            self.root_dir.set(d)

    def clear_all(self):
        # 기본 리스트 초기화
        for t in (getattr(self, "tree_all_basic", None),
                  getattr(self, "tree_pass_basic", None),
                  getattr(self, "tree_fail_basic", None)):
            if t: t.delete(*t.get_children())
        # 고급 리스트 초기화
        for t in (getattr(self, "tree_all_adv", None),
                  getattr(self, "tree_pass_adv", None),
                  getattr(self, "tree_fail_adv", None)):
            if t: t.delete(*t.get_children())

        self.results=[]
        self.ax.clear()
        self.ax.set_axis_on()
        self.ax.text(0.5,0.5,"클리어됨\n(분석 폴더 선택 후 '분석 시작')",ha="center",va="center",transform=self.ax.transAxes)
        self.fig.tight_layout(); self.canvas.draw_idle()
        self.stat_total.set("총 0"); self.stat_pass.set("PASS 0 (0.0%)"); self.stat_fail.set("FAIL 0 (0.0%)")
        self._write_log_header(found=None)
        self._bands = None
        self._clear_markers()

    def parse_exclude(self):
        txt=self.exclude_text.get().strip(); out=[]
        if not txt: return out
        for seg in txt.split(","):
            seg=seg.strip()
            if "-" in seg:
                a,b=seg.split("-",1)
                try: out.append((float(a)*1e6,float(b)*1e6))
                except: pass
        return out

    # ---- per-record compute ----
    def _compute_markers_for_record(self, rec, roi_hz):
        f_hz = rec["f_hz"]; s11 = rec["s11"]
        f_mhz = f_hz / 1e6
        y_norm = s_to_y_norm(s11)
        z0 = float(rec.get("z0", Z0_DEFAULT))
        y_mho = y_norm * (1.0 / z0)
        y11_db = 20*np.log10(np.abs(y_mho) + 1e-15)
        s11_db = 20*np.log10(np.abs(s11) + 1e-15)

        fs, fp, yfs, yfp = find_fs_fp_y11(
            f_hz, y11_db, roi_hz, smooth_mhz_for_coarse=8.0, fs_search_span_mhz=1200.0
        )
        level_db = float(self.cross_level_db.get())
        f_left, f_right = _find_minus10db_crossings_mhz(f_mhz, y11_db, fs, yfs, level_db)

        rec.update({
            "f_mhz": f_mhz, "y11_db": y11_db, "s11_db": s11_db,
            "fs": fs, "fp": fp, "yfs": yfs, "yfp": yfp,
            "f_left": f_left, "f_right": f_right,
        })

    def _feature_vector(self, rec):
        f_mhz = rec["f_mhz"]; y11_db = rec["y11_db"]
        feat = _shape_features(f_mhz, y11_db)
        if feat is None: return None, None
        vec = np.array([
            feat["kappa_mean"], feat["mono_L_mhz"], feat["mono_R_mhz"], feat["asym_grad"],
            feat["rms_global"], float(feat["spike_cnt"]), feat["range_db"],
        ], float)
        cols = ["kappa","monoL","monoR","asym","rms","spike","range"]
        return feat, (vec, cols)

    # ---- run/update/export ----
    def run(self):
        root=Path(self.root_dir.get())
        if not root.exists():
            messagebox.showerror("에러","폴더가 존재하지 않습니다"); return
        files=sorted(root.rglob("*.s1p"))
        if not files:
            messagebox.showwarning("주의","s1p 파일이 없습니다."); return

        # 리스트 비우기
        self.clear_all()

        roi=(self.roi_min.get()*1e6, self.roi_max.get()*1e6)
        excludes=self.parse_exclude()
        rows_for_csv=[]
        tmp_records=[]

        for f in files:
            try:
                f_hz, s11, z0 = parse_s1p(f)
            except Exception as e:
                # 실패 항목도 기본/고급 모두에 표기
                bv=(f.name, "ERROR")
                av=(f.name, "ERROR", f"parse failed: {e}")
                self.tree_all_basic.insert("", "end", values=bv)
                self.tree_all_adv.insert("", "end", values=av)
                continue

            rec={"file":str(f),"f_hz":f_hz,"s11":s11,"z0":z0}
            try:
                self._compute_markers_for_record(rec, roi)
            except Exception as me:
                rec.update({"fs":None,"fp":None,"yfs":None,"yfp":None,"f_left":None,"f_right":None})
                self._log(f"[WARN] marker compute failed for {f.name}: {me}\n")

            s11_res=contact_guard_s11_db(
                f_hz,s11,roi,excludes,
                smooth_mhz=self.smooth_mhz_qc.get(),
                rms_max=self.s11_rms_max.get(),
                spike_ratio_max=self.s11_spike_ratio_max.get()
            )
            y11_res=linearity_guard_y11_db(
                f_hz,s11,roi,excludes,
                smooth_mhz=self.smooth_mhz_qc.get(),
                tau_db_per_mhz=self.y11_tau.get(),
                run_amp_min_db=self.y11_run_amp.get(),
                L_min_mhz=self.y11_L_min.get()
            )
            rec["legacy_s_ok"]=s11_res["pass"]; rec["legacy_s_reason"]=s11_res["reason"]
            rec["legacy_y_ok"]=y11_res["pass"]; rec["legacy_y_reason"]=y11_res["reason"]

            feat, fv = self._feature_vector(rec)
            rec["feat"]=feat; rec["fv"]=fv
            tmp_records.append(rec)

        if self.decision_mode.get() == "AutoBand (Zero-Label)":
            self._fit_autoband(tmp_records)

        n_pass=0; n_fail=0
        for rec in tmp_records:
            verdict, reason = self._decide(rec)
            rec["verdict"]=verdict; rec["reason"]=reason
            self.results.append(rec)

            name = Path(rec["file"]).name
            # 기본(2컬럼)
            vals_basic=(name, verdict)
            self.tree_all_basic.insert("", "end", values=vals_basic, tags=(verdict,))
            # 고급(3컬럼)
            vals_adv=(name, verdict, reason)
            self.tree_all_adv.insert("", "end", values=vals_adv, tags=(verdict,))

            if verdict=="PASS":
                self.tree_pass_basic.insert("", "end", values=vals_basic, tags=(verdict,))
                self.tree_pass_adv.insert("", "end", values=vals_adv, tags=(verdict,))
                n_pass+=1
            else:
                self.tree_fail_basic.insert("", "end", values=vals_basic, tags=(verdict,))
                self.tree_fail_adv.insert("", "end", values=vals_adv, tags=(verdict,))
                n_fail+=1

            rows_for_csv.append({"file": name, "verdict": verdict, "reason": reason})

        n=len(self.results)
        p_pass=(n_pass/n*100.0) if n else 0.0
        p_fail=(n_fail/n*100.0) if n else 0.0
        self.stat_total.set(f"총 {n}")
        self.stat_pass.set(f"PASS {n_pass} ({p_pass:.1f}%)")
        self.stat_fail.set(f"FAIL {n_fail} ({p_fail:.1f}%)")

        for rec in self.results:
            self._log_line(rec.get("fs"), rec.get("yfs"), rec.get("fp"), rec.get("yfp"), Path(rec["file"]).name)

        self.update_preview()

        outdir=Path(self.root_dir.get() or ".")
        csv_path=(outdir if outdir.is_dir() else outdir.parent)/"qc_unified_v4.0.csv"
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
                w=csv.writer(fcsv); w.writerow(["file","verdict","reason"])
                for r in rows_for_csv:
                    w.writerow([r["file"], r["verdict"], r["reason"]])
            self._log(f"[OK] Saved: {csv_path.name}\n")
        except Exception as e:
            self._log(f"[WARN] CSV save failed: {e}\n")
    # -------- Decision engines --------
    def _decide(self, rec):
        mode = self.decision_mode.get()
        if mode == "Legacy (S11+Y11)":
            s_ok = rec["legacy_s_ok"]; y_ok = rec["legacy_y_ok"]
            if not s_ok:
                return "FAIL", f"S11(Contact) FAIL: {rec['legacy_s_reason']}"
            elif s_ok and not y_ok and self.dual_relax.get():
                return "PASS", f"S11 PASS & (Y11 FAIL but relaxed): {rec['legacy_y_reason']}"
            else:
                vd = "PASS" if (s_ok and y_ok) else "FAIL"
                rs = "S11 OK; Y11 OK" if vd=="PASS" else f"Y11 FAIL: {rec['legacy_y_reason']}"
                return vd, rs

        feat = rec.get("feat")
        if feat is None:
            return "FAIL", "feature extraction failed"

        if mode == "Human-Shape (Rule-only)":
            return _human_like_verdict(feat)

        fv = rec.get("fv")
        if (self._bands is None) or (fv is None):
            return _human_like_verdict(feat)

        vec, _ = fv
        med = self._bands["med"]; mad = self._bands["mad"]
        denom = np.maximum(mad, 1e-6)
        z = np.abs(vec - med)/denom
        z_max = float(np.max(z))
        i_top = int(np.argmax(z))
        cols_b = self._bands.get("cols", ["kappa","monoL","monoR","asym","rms","spike","range"])
        top_name = cols_b[i_top] if i_top < len(cols_b) else f"f{i_top}"
        top_val = float(vec[i_top]); top_med = float(med[i_top]); top_mad = float(mad[i_top])
        dbg = f" | top={top_name} z={z_max:.2f} val={top_val:.3f} med={top_med:.3f} mad={top_mad:.3f}"
        k = float(self._bands.get("k_auto", self.autoband_k.get())) if self.autoband_auto_k.get() else float(self.autoband_k.get())

        if z_max <= 1.0:
            return "PASS", f"AutoBand z_max={z_max:.2f}<=1.0" + dbg
        elif z_max <= k:
            return "PASS", f"AutoBand borderline z_max={z_max:.2f}<={k:.1f}" + dbg
        else:
            return "FAIL", f"AutoBand FAIL z_max={z_max:.2f}>{k:.1f}" + dbg

    # -------- AutoBand helpers --------
    def _select_bootstrap_cohort(self, records):
        cand = [r for r in records if r.get("legacy_s_ok") and r.get("feat")]
        if not cand: return []
        clean = []
        for r in cand:
            f = r["feat"]
            if (f["spike_cnt"] > 20) or (f["range_db"] < 8.0) or (f["kappa_mean"] < 0.004) or (f["mono_L_mhz"] < 50.0) or (f["mono_R_mhz"] < 50.0):
                continue
            clean.append(r)
        if not clean: clean = cand[:]
        scored = [(r, _quality_score(r["feat"])) for r in clean]
        scored.sort(key=lambda x: x[1])
        N = max(12, min(40, int(len(scored) * 0.5)))
        cohort = [r for (r, s) in scored[:N]]
        return cohort

    def _fit_autoband(self, records):
        Vs=[]
        for r in records:
            fv = r.get("fv")
            if fv is None:
                f = r.get("feat")
                if f is None: continue
                vec = np.array([
                    f["kappa_mean"], f["mono_L_mhz"], f["mono_R_mhz"], f["asym_grad"],
                    f["rms_global"], float(f["spike_cnt"]), f["range_db"]
                ], float)
                Vs.append(vec)
            else:
                Vs.append(fv[0])
        if len(Vs) < 5:
            self._bands = None; return False
        G = np.vstack(Vs)
        Gt = _trim_by_quantile(G, q=0.10)
        if Gt.size == 0:
            self._bands = None; return False
        med = np.median(Gt, axis=0)
        mad = np.array([_mad_normalized(Gt[:,i]) for i in range(Gt.shape[1])], float)
        mad_floor = np.array([0.0010, 80.0, 80.0, 0.008, 1.20, 3.0, 10.0], float)
        mad = np.maximum(mad, mad_floor)
        rel = 0.12
        mad = np.maximum(mad, rel*np.abs(med))
        den = np.maximum(mad, 1e-6)
        z_all = np.max(np.abs((Gt - med) / den), axis=1)
        z_all = np.clip(z_all, 0.0, 12.0)
        q = 98.0
        k_auto = float(np.percentile(z_all, q))
        k_auto = max(1.6, min(k_auto, 4.5))
        self_pass_rate = float(np.mean(z_all <= 1.0))
        ok_rate = (self_pass_rate >= 0.85)
        ok_k = (1.6 <= k_auto <= 4.5)
        if not (ok_rate and ok_k):
            self._log(f"[AutoBand][ABORT] self_pass={self_pass_rate:.2f}, k_auto={k_auto:.2f} → fallback to Legacy\n")
            self._bands = None
            return False
        self._bands = {"med": med, "mad": mad, "cols":["kappa","monoL","monoR","asym","rms","spike","range"], "k_auto": k_auto}
        self._log(f"[AutoBand] med={np.round(med,3)}\n")
        self._log(f"[AutoBand] mad(norm)={np.round(mad,3)}\n")
        self._log(f"[AutoBand] self_pass={self_pass_rate:.2f}, k_auto(p{int(q)})={k_auto:.2f}\n")
        return True

    def _train_autoband_from_current_scan(self):
        if not self.results:
            messagebox.showwarning("AutoBand", "먼저 '분석 시작'으로 데이터를 로드하세요.")
            return
        cohort = self._select_bootstrap_cohort(self.results)
        if not cohort:
            messagebox.showwarning("AutoBand", "AutoBand 학습에 사용할 수 있는 PASS 데이터가 없습니다.")
            return
        if self._fit_autoband(cohort):
            msg = f"AutoBand 학습 완료 (n={len(cohort)}, k_auto={self._bands['k_auto']:.2f})"
            self._log(f"[AutoBand][TRAIN] {msg}\n")
            messagebox.showinfo("AutoBand", msg)
        else:
            messagebox.showerror("AutoBand", "AutoBand 학습에 실패했습니다. 로그를 확인하세요.")

    def _train_from_selected(self):
        if not self.results:
            messagebox.showwarning("AutoBand", "먼저 '분석 시작'으로 데이터를 로드하세요.")
            return
        targets = self._collect_selected()
        if not targets:
            messagebox.showwarning("AutoBand", "선택된 레코드가 없습니다.")
            return
        if self._fit_autoband(targets):
            msg = f"선택된 {len(targets)}건으로 AutoBand 학습 완료 (k_auto={self._bands['k_auto']:.2f})"
            self._log(f"[AutoBand][TRAIN] {msg}\n")
            messagebox.showinfo("AutoBand", msg)
        else:
            messagebox.showerror("AutoBand", "AutoBand 학습에 실패했습니다. 선택된 데이터를 확인하세요.")

    # ---- 선택/프리뷰 ----
    def _collect_selected(self):
        # 현재 상단 탭(basic/advanced)과 그 안의 리스트 탭(All/PASS/FAIL)을 기준으로 선택 항목 수집
        cur_top = self.tabs.index(self.tabs.select())
        if cur_top == 0:
            nb = self.nb_list_basic
            trees=[self.tree_all_basic,self.tree_pass_basic,self.tree_fail_basic]
        else:
            nb = self.nb_list_adv
            trees=[self.tree_all_adv,self.tree_pass_adv,self.tree_fail_adv]
        cur_idx = nb.index(nb.select())
        cur_tree=trees[cur_idx]
        names=[cur_tree.item(i,"values")[0] for i in cur_tree.selection()]
        if not names:
            # 다른 탭들에서도 찾아보기
            for i, t in enumerate(trees):
                if i == cur_idx: continue
                names += [t.item(iid,"values")[0] for iid in t.selection()]
        if not names:
            # 아무것도 없으면 상단 탭의 All에서 최대 5개
            base_tree = trees[0]
            names = [base_tree.item(i,"values")[0] for i in base_tree.get_children()[:5]]
        targets=[r for r in self.results if Path(r["file"]).name in names]
        return targets

    def on_select(self,_):
        self.update_preview()

    # ---- Marker helpers ----
    def _refresh_marker_styles(self):
        for i, m in enumerate(self._markers):
            ln, txt = m['ln'], m['txt']
            if i == self._active_idx:
                ln.set_markersize(8); ln.set_markerfacecolor('C3'); ln.set_alpha(1.0)
                txt.set_alpha(1.0)
            else:
                ln.set_markersize(6); ln.set_markerfacecolor('C0'); ln.set_alpha(0.75)
                txt.set_alpha(0.75)
        self.canvas.draw_idle()

    def _add_marker(self, f_mhz, y_db, label_prefix="Y11: ", idx_in_series=None):
        ln, = self.ax.plot([f_mhz], [y_db], marker='s', markersize=6, linestyle='None')
        f_ghz = f_mhz / 1000.0
        label = f"{label_prefix}f={f_ghz:.4f} GHz, {y_db:.3f} dB"
        txt = self.ax.annotate(label, (f_mhz, y_db), xytext=(6, 10),
                               textcoords="offset points", fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.75),
                               clip_on=False)
        if idx_in_series is None and self._last_plot_data.get('x') is not None:
            idx_in_series = int(np.argmin(np.abs(self._last_plot_data['x'] - f_mhz)))
        self._markers.append({'ln': ln, 'txt': txt, 'idx': int(idx_in_series or 0)})
        self._active_idx = len(self._markers) - 1
        self._refresh_marker_styles()

    def _update_marker_at(self, m, new_idx):
        xdata = self._last_plot_data.get('x'); ydata = self._last_plot_data.get('y')
        if xdata is None or ydata is None or len(xdata) == 0: return
        new_idx = int(np.clip(new_idx, 0, len(xdata)-1))
        f_mhz = float(xdata[new_idx]); y_db = float(ydata[new_idx])
        m['idx'] = new_idx
        m['ln'].set_data([f_mhz], [y_db])
        m['txt'].xy = (f_mhz, y_db)
        m['txt'].set_text(f"Y11: f={f_mhz/1000.0:.4f} GHz, {y_db:.3f} dB")
        self.canvas.draw_idle()

    def _clear_markers(self):
        for m in self._markers:
            try: m['ln'].remove()
            except Exception: pass
            try: m['txt'].remove()
            except Exception: pass
        self._markers.clear()
        self._active_idx = None
        self.canvas.draw_idle()

    def _nearest_marker_at(self, event, tol_px=10):
        if not self._markers or event.inaxes != self.ax:
            return None
        ex, ey = event.x, event.y
        best_i, best_d = None, float('inf')
        trans = self.ax.transData
        for i, m in enumerate(self._markers):
            x = m['ln'].get_xdata()[0]
            y = m['ln'].get_ydata()[0]
            px, py = trans.transform((x, y))
            d = ((px - ex)**2 + (py - ey)**2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        return best_i if best_d <= tol_px else None

    def _on_plot_key(self, event):
        if not self._markers or self._active_idx is None:
            return
        step = 1
        if event.key and isinstance(event.key, str) and 'shift' in event.key.lower():
            step = 10
        key = event.key.lower() if isinstance(event.key, str) else ''
        if key in ('left','right','delete','backspace'):
            m = self._markers[self._active_idx]
            if key == 'left':
                self._update_marker_at(m, m['idx'] - step)
            elif key == 'right':
                self._update_marker_at(m, m['idx'] + step)
            else:
                try: m['ln'].remove()
                except Exception: pass
                try: m['txt'].remove()
                except Exception: pass
                self._markers.pop(self._active_idx)
                self._active_idx = (len(self._markers)-1) if self._markers else None
            self._refresh_marker_styles()

    def _on_plot_click(self, event):
        if event.inaxes != self.ax:
            return
        try:
            self.canvas.get_tk_widget().focus_set()
        except Exception:
            pass
        if not self._last_plot_data or 'x' not in self._last_plot_data:
            return

        if event.button == 3:  # 우클릭: 전체 삭제
            self._clear_markers()
            self._dragging_idx = None
            return

        if event.button != 1:
            return

        hit = self._nearest_marker_at(event, tol_px=10)
        if hit is not None:
            self._dragging_idx = hit
            self._active_idx = hit
            self._refresh_marker_styles()
            return

        if event.key is not None and 'shift' in str(event.key).lower():
            if self._markers:
                m = self._markers.pop()
                try: m['ln'].remove()
                except Exception: pass
                try: m['txt'].remove()
                except Exception: pass
            self._active_idx = (len(self._markers)-1) if self._markers else None
            self._refresh_marker_styles()
            return

        xdata = self._last_plot_data.get('x')
        ydata = self._last_plot_data.get('y')
        if xdata is None or len(xdata) == 0:
            return
        x = float(event.xdata)
        idx = int(np.argmin(np.abs(xdata - x)))
        f_mhz = float(xdata[idx]); y_db = float(ydata[idx])
        self._add_marker(f_mhz, y_db, "Y11: ", idx_in_series=idx)

    def _on_mouse_move(self, event):
        if self._dragging_idx is None or event.inaxes != self.ax:
            return
        xdata = self._last_plot_data.get('x')
        if xdata is None or len(xdata) == 0:
            return
        x = float(event.xdata)
        idx = int(np.clip(np.argmin(np.abs(xdata - x)), 0, len(xdata)-1))
        m = self._markers[self._dragging_idx]
        self._update_marker_at(m, idx)
        self._active_idx = self._dragging_idx
        self._refresh_marker_styles()

    def _on_mouse_release(self, event):
        if event and event.button == 1:
            self._dragging_idx = None

    def update_preview(self):
        self.ax.clear()
        self.ax.set_axis_on()
        try:
            targets=self._collect_selected()
            if not targets:
                self.ax.text(0.5,0.5,"선택 없음\n(Shift/Ctrl로 다중 선택)",ha="center",va="center",transform=self.ax.transAxes)
                self.fig.tight_layout(); self.canvas.draw_idle(); return

            roi_mhz=(self.roi_min.get(), self.roi_max.get())
            xmins=[]; xmaxs=[]; ymins=[]; ymaxs=[]

            draw_s11 = self.show_s11.get()
            draw_y11 = self.show_y11.get()
            if not (draw_s11 or draw_y11): draw_y11 = True

            r0 = targets[0]
            snap_x = r0["f_mhz"]; snap_y = r0["y11_db"]
            self._last_plot_data = {"x": snap_x, "y": snap_y, "name": "Y11"}

            title1=""; title2=""
            for r in targets:
                if "f_mhz" not in r:
                    self._compute_markers_for_record(r, (self.roi_min.get()*1e6, self.roi_max.get()*1e6))
                f_mhz = r["f_mhz"]; y11_db = r["y11_db"]; s11_db = r["s11_db"]
                fs, fp = r.get("fs"), r.get("fp")

                if draw_s11:
                    self.ax.plot(f_mhz, s11_db, linewidth=1.0, alpha=0.85)
                    ymins.append(np.nanmin(s11_db)); ymaxs.append(np.nanmax(s11_db))
                if draw_y11:
                    self.ax.plot(f_mhz, y11_db, linewidth=1.25, alpha=0.95)
                    ymins.append(np.nanmin(y11_db)); ymaxs.append(np.nanmax(y11_db))

                xmins.append(f_mhz.min()); xmaxs.append(f_mhz.max())
                title1 = Path(r["file"]).name
                fs_txt = f"{fs:.3f} MHz" if fs else "NA"
                fp_txt = f"{fp:.3f} MHz" if fp else "NA"
                title2 = f"COMB:{r.get('verdict','NA')} mode:{self.decision_mode.get()} fs={fs_txt}, fp={fp_txt}"
                if fs: self.ax.axvline(fs, linestyle='--', alpha=0.5)
                if fp: self.ax.axvline(fp, linestyle='--', alpha=0.5)

            self.ax.axvspan(roi_mhz[0], roi_mhz[1], alpha=0.12)
            if self.zoom_roi.get():
                pad=0.02*(roi_mhz[1]-roi_mhz[0]); self.ax.set_xlim(roi_mhz[0]-pad, roi_mhz[1]+pad)
            else:
                self.ax.set_xlim(min(xmins), max(xmaxs))
            if ymins and ymaxs:
                ypad=0.05*(max(ymaxs)-min(ymins)+1e-9)
                self.ax.set_ylim(min(ymins)-ypad, max(ymaxs)+ypad)

            self.ax.set_xlabel("Frequency (MHz)")
            self.ax.set_ylabel("S11 (dB) / Y11 (dB, mho)")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title(f"{title1}\n{title2}", fontsize=9)
        except Exception as e:
            self.ax.text(0.5,0.5,f"Plot error: {e}",ha="center",va="center",transform=self.ax.transAxes)
        finally:
            self.fig.tight_layout(); self.canvas.draw_idle()

    def export_txt(self):
        if not self.results:
            messagebox.showwarning("Warn","결과가 없습니다. 먼저 '분석 시작'을 수행하세요.")
            return
        level_db = float(self.cross_level_db.get())
        header = f"file\tFs(MHz)\tY@Fs(dB)\tFp(MHz)\tY@Fp(dB)\tKt2\tL{int(level_db)}_f(MHz)\tR{int(level_db)}_f(MHz)"
        lines = [header]
        for rec in self.results:
            fn = Path(rec["file"]).name
            fs = rec.get("fs"); yfs = rec.get("yfs")
            fp = rec.get("fp"); yfp = rec.get("yfp")
            fL = rec.get("f_left"); fR = rec.get("f_right")
            if (fs is None) or (fp is None) or (yfs is None) or (yfp is None):
                line = f"{fn}\tNA\tNA\tNA\tNA\tNA\tNA\tNA"
            else:
                kt2 = 1.0 - (fs/fp)**2 if (fp not in (0,None)) else float("nan")
                fL_txt = f"{fL:.3f}" if isinstance(fL, (int, float)) else "NA"
                fR_txt = f"{fR:.3f}" if isinstance(fR, (int, float)) else "NA"
                line = f"{fn}\t{fs:.3f}\t{yfs:.3f}\t{fp:.3f}\t{yfp:.3f}\t{kt2:.6f}\t{fL_txt}\t{fR_txt}"
            lines.append(line)
        out=Path(self.root_dir.get() or ".")
        out=(out if out.is_dir() else out.parent)/"report_y11_v4.0.txt"
        try:
            out.write_text("\n".join(lines), encoding="utf-8")
            self._log(f"[OK] TXT saved: {out.name}\n")
            messagebox.showinfo("OK", f"Saved: {out}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------- main ----------
def main():
    root=Tk()
    App(root)
    root.mainloop()

if __name__=="__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
