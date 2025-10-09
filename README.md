# Resonator QC Unified v4.0

This repository contains the Python GUI application `rqcu_v4_0.py`. To run it:

1. Install Python 3.9+ with Tkinter and Matplotlib (`pip install matplotlib numpy`).
2. Download the **raw** `rqcu_v4_0.py` file (for example via `git clone` or "Save link as" from the raw view). Saving the diff/patch view will produce a file that begins with `diff --git` and Python will raise a `SyntaxError` like the one shown in the issue.
3. Launch the program with `python rqcu_v4_0.py`.

If you see `SyntaxError: invalid syntax` pointing at a `diff --git` line, delete the file and download it again using the raw file link so that the first line is `#!/usr/bin/env python3`.
