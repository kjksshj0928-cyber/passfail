#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_newstart_ml_flat_stage1_gui_v1_1.py
--------------------------------------
Simple Tkinter GUI wrapper for:
 - Classify (Stage-1 FlatnessGuard + optional ML vote)
 - Train (LDA on GOOD/BAD)
 - Eval (confusion matrix)

Requires: qc_newstart_ml_flat_stage1_v1_0.py in the SAME folder.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import queue
from pathlib import Path

# Try to import core module
try:
    import qc_newstart_ml_flat_stage1_v1_0 as core
    _core_import_error = None
except Exception as _exc:
    core = None
    _core_import_error = _exc

def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC FlatnessGuard (Y11/S11) - GUI v1.1")
        self.geometry("980x700")
        self.resizable(True, True)

        if core is None:
            messagebox.showerror(
                "Import Error",
                "qc_newstart_ml_flat_stage1_v1_0.py 를 같은 폴더에 두고 실행하세요.\n\n에러:\n"
                + (str(_core_import_error) if _core_import_error else "Unknown import error"),
            )

        self._build_ui()

        # worker thread queue for logs
        self.log_q = queue.Queue()
        self._poll_log()

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.page_classify = ttk.Frame(nb)
        self.page_train = ttk.Frame(nb)
        self.page_eval = ttk.Frame(nb)

        nb.add(self.page_classify, text="Classify")
        nb.add(self.page_train, text="Train")
        nb.add(self.page_eval, text="Eval")

        self._build_classify(self.page_classify)
        self._build_train(self.page_train)
        self._build_eval(self.page_eval)
        self._build_log_area()

    def _build_log_area(self):
        frm = ttk.LabelFrame(self, text="Log")
        frm.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        self.txt = tk.Text(frm, height=12)
        self.txt.pack(fill="both", expand=True)
        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="Clear Log", command=lambda: self.txt.delete("1.0", "end")).pack(
            side="right", padx=4, pady=4
        )

    def log(self, msg):
        self.log_q.put(msg)

    def _poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # -------------------- Classify --------------------
    def _build_classify(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=8, pady=8)

        row = 0
        # root dir
        ttk.Label(frm, text="Root (unlabeled S1P dir)").grid(row=row, column=0, sticky="w")
        self.cl_root = tk.StringVar()
        ttk.Entry(frm, textvariable=self.cl_root, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm, text="Browse", command=lambda: self._pick_dir(self.cl_root)).grid(row=row, column=2, padx=2)
        row += 1

        # model optional
        ttk.Label(frm, text="Model (optional)").grid(row=row, column=0, sticky="w")
        self.cl_model = tk.StringVar()
        ttk.Entry(frm, textvariable=self.cl_model, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(
            frm, text="Browse", command=lambda: self._pick_file(self.cl_model, [("JSON", "*.json"), ("All", "*.*")])
        ).grid(row=row, column=2, padx=2)
        row += 1

        # out csv
        ttk.Label(frm, text="Out CSV").grid(row=row, column=0, sticky="w")
        self.cl_out = tk.StringVar(value="out.csv")
        ttk.Entry(frm, textvariable=self.cl_out, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(
            frm, text="Browse", command=lambda: self._save_file(self.cl_out, [("CSV", "*.csv"), ("All", "*.*")])
        ).grid(row=row, column=2, padx=2)
        row += 1

        # domain, roi
        ttk.Label(frm, text="Domain").grid(row=row, column=0, sticky="w")
        self.cl_domain = tk.StringVar(value="Y11")
        ttk.Combobox(frm, textvariable=self.cl_domain, values=["Y11", "S11"], width=10).grid(
            row=row, column=1, sticky="w", padx=5
        )

        ttk.Label(frm, text="ROI (Hz): min").grid(row=row, column=1, sticky="e")
        self.cl_roi_min = tk.StringVar(value="1e8")
        ttk.Entry(frm, textvariable=self.cl_roi_min, width=16).grid(row=row, column=2, sticky="w")

        ttk.Label(frm, text="max").grid(row=row, column=2, sticky="e", padx=(0, 50))
        self.cl_roi_max = tk.StringVar(value="8.5e9")
        ttk.Entry(frm, textvariable=self.cl_roi_max, width=16).grid(row=row, column=2, sticky="e", padx=(0, 0))
        row += 1

        # ML
        self.cl_use_ml = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Use ML (LDA)", variable=self.cl_use_ml).grid(row=row, column=0, sticky="w")
        ttk.Label(frm, text="ML thresh (prob_good)").grid(row=row, column=1, sticky="e")
        self.cl_ml_thresh = tk.StringVar(value="0.5")
        ttk.Entry(frm, textvariable=self.cl_ml_thresh, width=10).grid(row=row, column=2, sticky="w")
        row += 1

        # Advanced thresholds (collapsible)
        adv = ttk.LabelFrame(frm, text="Advanced thresholds (optional)")
        adv.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        self._build_thresholds(adv, prefix="cl_")
        row += 1

        # run
        ttk.Button(frm, text="RUN Classify", command=self._run_classify).grid(row=row, column=0, sticky="w", pady=8)

        # stretch
        frm.columnconfigure(1, weight=1)

    # -------------------- Train --------------------
    def _build_train(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=8, pady=8)
        row = 0
        ttk.Label(frm, text="GOOD dir").grid(row=row, column=0, sticky="w")
        self.tr_good = tk.StringVar()
        ttk.Entry(frm, textvariable=self.tr_good, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm, text="Browse", command=lambda: self._pick_dir(self.tr_good)).grid(row=row, column=2, padx=2)
        row += 1

        ttk.Label(frm, text="BAD dir").grid(row=row, column=0, sticky="w")
        self.tr_bad = tk.StringVar()
        ttk.Entry(frm, textvariable=self.tr_bad, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm, text="Browse", command=lambda: self._pick_dir(self.tr_bad)).grid(row=row, column=2, padx=2)
        row += 1

        ttk.Label(frm, text="Model out (json)").grid(row=row, column=0, sticky="w")
        self.tr_model = tk.StringVar(value="model.json")
        ttk.Entry(frm, textvariable=self.tr_model, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(
            frm, text="Browse", command=lambda: self._save_file(self.tr_model, [("JSON", "*.json"), ("All", "*.*")])
        ).grid(row=row, column=2, padx=2)
        row += 1

        ttk.Label(frm, text="Domain").grid(row=row, column=0, sticky="w")
        self.tr_domain = tk.StringVar(value="Y11")
        ttk.Combobox(frm, textvariable=self.tr_domain, values=["Y11", "S11"], width=10).grid(
            row=row, column=1, sticky="w", padx=5
        )

        ttk.Label(frm, text="ROI (Hz): min").grid(row=row, column=1, sticky="e")
        self.tr_roi_min = tk.StringVar(value="1e8")
        ttk.Entry(frm, textvariable=self.tr_roi_min, width=16).grid(row=row, column=2, sticky="w")
        ttk.Label(frm, text="max").grid(row=row, column=2, sticky="e", padx=(0, 50))
        self.tr_roi_max = tk.StringVar(value="8.5e9")
        ttk.Entry(frm, textvariable=self.tr_roi_max, width=16).grid(row=row, column=2, sticky="e")
        row += 1

        ttk.Button(frm, text="RUN Train", command=self._run_train).grid(row=row, column=0, sticky="w", pady=8)
        frm.columnconfigure(1, weight=1)

    # -------------------- Eval --------------------
    def _build_eval(self, parent):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, padx=8, pady=8)
        row = 0
        ttk.Label(frm, text="GOOD dir").grid(row=row, column=0, sticky="w")
        self.ev_good = tk.StringVar()
        ttk.Entry(frm, textvariable=self.ev_good, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm, text="Browse", command=lambda: self._pick_dir(self.ev_good)).grid(row=row, column=2, padx=2)
        row += 1

        ttk.Label(frm, text="BAD dir").grid(row=row, column=0, sticky="w")
        self.ev_bad = tk.StringVar()
        ttk.Entry(frm, textvariable=self.ev_bad, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(frm, text="Browse", command=lambda: self._pick_dir(self.ev_bad)).grid(row=row, column=2, padx=2)
        row += 1

        ttk.Label(frm, text="Model (optional)").grid(row=row, column=0, sticky="w")
        self.ev_model = tk.StringVar()
        ttk.Entry(frm, textvariable=self.ev_model, width=70).grid(row=row, column=1, sticky="we", padx=5)
        ttk.Button(
            frm, text="Browse", command=lambda: self._pick_file(self.ev_model, [("JSON", "*.json"), ("All", "*.*")])
        ).grid(row=row, column=2, padx=2)
        row += 1

        ttk.Label(frm, text="Domain").grid(row=row, column=0, sticky="w")
        self.ev_domain = tk.StringVar(value="Y11")
        ttk.Combobox(frm, textvariable=self.ev_domain, values=["Y11", "S11"], width=10).grid(
            row=row, column=1, sticky="w", padx=5
        )

        ttk.Label(frm, text="ROI (Hz): min").grid(row=row, column=1, sticky="e")
        self.ev_roi_min = tk.StringVar(value="1e8")
        ttk.Entry(frm, textvariable=self.ev_roi_min, width=16).grid(row=row, column=2, sticky="w")
        ttk.Label(frm, text="max").grid(row=row, column=2, sticky="e", padx=(0, 50))
        self.ev_roi_max = tk.StringVar(value="8.5e9")
        ttk.Entry(frm, textvariable=self.ev_roi_max, width=16).grid(row=row, column=2, sticky="e")
        row += 1

        # ML
        self.ev_use_ml = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Use ML (LDA)", variable=self.ev_use_ml).grid(row=row, column=0, sticky="w")
        ttk.Label(frm, text="ML thresh (prob_good)").grid(row=row, column=1, sticky="e")
        self.ev_ml_thresh = tk.StringVar(value="0.5")
        ttk.Entry(frm, textvariable=self.ev_ml_thresh, width=10).grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Button(frm, text="RUN Eval", command=self._run_eval).grid(row=row, column=0, sticky="w", pady=8)
        frm.columnconfigure(1, weight=1)

    # --------------- thresholds builder ---------------
    def _build_thresholds(self, parent, prefix="cl_"):
        grid = ttk.Frame(parent)
        grid.pack(fill="x", padx=4, pady=4)
        fields = [
            ("th_curv", "curvature_rms_max", "0.02"),
            ("th_rough", "roughness_rms_max", "0.15"),
            ("th_slope", "slope_abs_max", "0.5"),
            ("th_maxdev", "max_dev_max", "0.8"),
            ("th_pkprom", "pk_prom_max", "0.6"),
            ("th_ntprom", "nt_prom_max", "0.6"),
            ("th_pkcnt", "pk_count_max", "1.0"),
            ("th_ntcnt", "nt_count_max", "1.0"),
            ("th_p2p", "p2p_max", "1.2"),
        ]
        self.th_vars = getattr(self, "th_vars", {})
        r = 0
        c = 0
        for key, label, default in fields:
            tkvar = tk.StringVar(value="")
            self.th_vars[prefix + key] = tkvar
            ttk.Label(grid, text=label).grid(row=r, column=c, sticky="e", padx=2, pady=2)
            ttk.Entry(grid, textvariable=tkvar, width=10).grid(row=r, column=c + 1, sticky="w", padx=2, pady=2)
            c += 2
            if c >= 6:
                c = 0
                r += 1
        ttk.Label(parent, text="(빈칸이면 기본값을 사용합니다)").pack(anchor="w", padx=6)

    def _collect_thresholds(self, prefix="cl_"):
        # Start with defaults
        th = core.DEFAULT_STAGE1_TH.copy()
        mapping = {
            "th_curv": "curvature_rms_max",
            "th_rough": "roughness_rms_max",
            "th_slope": "slope_abs_max",
            "th_maxdev": "max_dev_max",
            "th_pkprom": "pk_prom_max",
            "th_ntprom": "nt_prom_max",
            "th_pkcnt": "pk_count_max",
            "th_ntcnt": "nt_count_max",
            "th_p2p": "p2p_max",
        }
        for short, full in mapping.items():
            key = prefix + short
            if key not in self.th_vars:
                continue
            v = self.th_vars[key].get().strip()
            if v:
                fv = safe_float(v, None)
                if fv is not None:
                    th[full] = fv
        return th

    # -------------------- Actions --------------------
    def _run_classify(self):
        if core is None:
            messagebox.showerror("Error", "core module import 실패")
            return
        root = self.cl_root.get().strip()
        out_csv = self.cl_out.get().strip()
        model = self.cl_model.get().strip()
        if not root:
            messagebox.showerror("Error", "Root 디렉토리를 선택하세요.")
            return
        use_ml = bool(self.cl_use_ml.get())
        ml_thresh = safe_float(self.cl_ml_thresh.get(), 0.5)
        domain = (self.cl_domain.get() or "Y11").upper()
        roi = (safe_float(self.cl_roi_min.get(), None), safe_float(self.cl_roi_max.get(), None))

        th = self._collect_thresholds(prefix="cl_")

        t = threading.Thread(
            target=self._do_classify,
            args=(root, out_csv, model, use_ml, ml_thresh, domain, roi, th),
            daemon=True,
        )
        t.start()

    def _do_classify(self, root, out_csv, model, use_ml, ml_thresh, domain, roi, th):
        try:
            if model and os.path.exists(model):
                ref_grid, th_m, lda, segments, meta = core.load_model_json(Path(model))
                # merge thresholds: GUI overrides take precedence if provided
                th = th or th_m
                if roi[0] is None or roi[1] is None:
                    roi = meta.get("roi", roi)
                if not domain and "domain" in meta:
                    domain = meta["domain"]
            else:
                ref_grid = core.build_ref_grid(core.DEFAULT_WIDECAL)
                lda = None

            files = core.scan_s1p_files(Path(root))
            rows = []
            for i, fp in enumerate(files, start=1):
                feat, npts = core.load_s1p_features(fp, domain, ref_grid, roi)
                if feat is None:
                    continue
                final, s1_label, s1_note, (ml_label, ml_p) = self._stage1_and_ml_vote(
                    feat, th, lda, use_ml, ml_thresh
                )
                row = {
                    "file": str(fp),
                    "final": final,
                    "stage1": s1_label,
                    "stage1_note": s1_note,
                    "ml_label": ml_label if ml_label is not None else "",
                    "ml_prob_good": f"{ml_p:.4f}" if ml_p is not None else "",
                }
                for k in core.FEATURE_ORDER:
                    row[k] = feat[k]
                rows.append(row)
                if i % 50 == 0:
                    self.log(f"[Classify] processed {i}/{len(files)}")

            # write CSV
            import csv

            if out_csv:
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=list(rows[0].keys()) if rows else ["file", "final"]
                    )
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                self.log(f"[OK] Wrote: {out_csv} ({len(rows)} rows)")
            else:
                self.log(f"[INFO] Total rows: {len(rows)}")
                for r in rows[:10]:
                    self.log(str(r))

        except Exception as ex:
            import traceback

            self.log("[ERROR] " + str(ex))
            self.log(traceback.format_exc())

    def _stage1_and_ml_vote(self, feat, th, lda, use_ml, ml_thresh):
        s1_label, s1_note = core.stage1_flatness_decision(feat, th)
        ml_p = None
        ml_label = None
        if use_ml and lda is not None:
            import numpy as np

            x = np.array([feat[k] for k in core.FEATURE_ORDER], dtype=float)
            ml_p = lda.score(x)  # prob GOOD
            ml_label = "PASS" if ml_p >= ml_thresh else "FAIL"
        if use_ml and lda is not None:
            final = "PASS" if (s1_label == "PASS" and ml_label == "PASS") else "FAIL"
        else:
            final = s1_label
        return final, s1_label, s1_note, (ml_label, ml_p)

    def _run_train(self):
        if core is None:
            messagebox.showerror("Error", "core module import 실패")
            return
        good = self.tr_good.get().strip()
        bad = self.tr_bad.get().strip()
        model = self.tr_model.get().strip()
        domain = (self.tr_domain.get() or "Y11").upper()
        roi = (safe_float(self.tr_roi_min.get(), None), safe_float(self.tr_roi_max.get(), None))

        if not (good and bad and model):
            messagebox.showerror("Error", "GOOD/BAD/Model 경로를 확인하세요.")
            return

        t = threading.Thread(target=self._do_train, args=(good, bad, model, domain, roi), daemon=True)
        t.start()

    def _do_train(self, good, bad, model, domain, roi):
        try:
            # Build ref grid from defaults
            segments = core.DEFAULT_WIDECAL
            ref_grid = core.build_ref_grid(segments)

            X_list = []
            y_list = []
            import numpy as np

            for label, root in [(1, Path(good)), (0, Path(bad))]:
                files = core.scan_s1p_files(root)
                for fp in files:
                    feat, npts = core.load_s1p_features(fp, domain, ref_grid, roi)
                    if feat is None:
                        continue
                    X_list.append([feat[k] for k in core.FEATURE_ORDER])
                    y_list.append(label)

            if not X_list:
                self.log("[ERROR] No features extracted. Check paths/ROI.")
                return

            X = np.array(X_list, dtype=float)
            y = np.array(y_list, dtype=int)

            lda = core.LDA(feature_names=core.FEATURE_ORDER)
            lda.fit(X, y)

            core.save_model_json(
                Path(model), lda, segments, core.DEFAULT_STAGE1_TH, meta_extra={"domain": domain, "roi": roi}
            )
            self.log(f"[OK] Model saved to: {model}")
            self.log(f"[INFO] Training samples: GOOD={(y == 1).sum()}, BAD={(y == 0).sum()}")
        except Exception as ex:
            import traceback

            self.log("[ERROR] " + str(ex))
            self.log(traceback.format_exc())

    def _run_eval(self):
        if core is None:
            messagebox.showerror("Error", "core module import 실패")
            return
        good = self.ev_good.get().strip()
        bad = self.ev_bad.get().strip()
        model = self.ev_model.get().strip()
        use_ml = bool(self.ev_use_ml.get())
        ml_thresh = safe_float(self.ev_ml_thresh.get(), 0.5)
        domain = (self.ev_domain.get() or "Y11").upper()
        roi = (safe_float(self.ev_roi_min.get(), None), safe_float(self.ev_roi_max.get(), None))

        if not (good and bad):
            messagebox.showerror("Error", "GOOD/BAD 디렉토리를 선택하세요.")
            return

        t = threading.Thread(
            target=self._do_eval, args=(good, bad, model, use_ml, ml_thresh, domain, roi), daemon=True
        )
        t.start()

    def _do_eval(self, good, bad, model, use_ml, ml_thresh, domain, roi):
        try:
            import numpy as np

            if model and os.path.exists(model):
                ref_grid, th, lda, segments, meta = core.load_model_json(Path(model))
                if roi[0] is None or roi[1] is None:
                    roi = meta.get("roi", roi)
                if not domain and "domain" in meta:
                    domain = meta["domain"]
            else:
                ref_grid = core.build_ref_grid(core.DEFAULT_WIDECAL)
                lda = None
                th = core.DEFAULT_STAGE1_TH.copy()

            y_true = []
            y_pred = []

            for label, root in [(1, Path(good)), (0, Path(bad))]:
                files = core.scan_s1p_files(root)
                for fp in files:
                    feat, npts = core.load_s1p_features(fp, domain, ref_grid, roi)
                    if feat is None:
                        continue
                    final, s1_label, s1_note, (ml_label, ml_p) = self._stage1_and_ml_vote(
                        feat, th, lda, use_ml, ml_thresh
                    )
                    y_true.append(label)
                    y_pred.append(1 if final == "PASS" else 0)

            y_true = np.array(y_true, dtype=int)
            y_pred = np.array(y_pred, dtype=int)
            TP = int(((y_true == 1) & (y_pred == 1)).sum())
            TN = int(((y_true == 0) & (y_pred == 0)).sum())
            FP = int(((y_true == 1) & (y_pred == 0)).sum())
            FN = int(((y_true == 0) & (y_pred == 1)).sum())
            total = max(1, len(y_true))
            acc = (TP + TN) / total
            fp_rate = FP / max(1, (TP + FP))
            fn_rate = FN / max(1, (TN + FN))

            self.log(f"[EVAL] TP={TP} TN={TN} FP={FP} FN={FN}")
            self.log(f"[EVAL] Accuracy={acc:.4f}, FP_rate={fp_rate:.4f}, FN_rate={fn_rate:.4f}")
        except Exception as ex:
            import traceback

            self.log("[ERROR] " + str(ex))
            self.log(traceback.format_exc())

    # -------------------- utils --------------------
    def _pick_dir(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _pick_file(self, var, types):
        f = filedialog.askopenfilename(filetypes=types)
        if f:
            var.set(f)

    def _save_file(self, var, types):
        f = filedialog.asksaveasfilename(defaultextension=types[0][1].replace("*", ""), filetypes=types)
        if f:
            var.set(f)


if __name__ == "__main__":
    # Tk on Windows sometimes needs this for DPI scaling
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
