#!/usr/bin/env python3
"""Interactive heel-strike picker for the gait pipeline.

Opens two x-aligned panels of a trial's <stem>_PC_signals.csv - dynamic-
accel PC1/PC2 on top, sagittal angular-rate PC1 below - lets you mark
heel strikes with the mouse on either panel, and writes the result to
<trial>/heel_strikes.txt - the file rerun_gait.sh prefers over its
hardcoded (log-derived) strike times.

Usage:
    python pick_heel_strikes.py /path/to/<trial>/<stem>_PC_signals.csv

Mouse / keys:
    left click    add a heel strike (snapped to the nearest EMG sample on
                  the absolute 980 sps time base); works on either panel
    right click   remove the nearest mark (within 5% of the visible span)
    u             undo the last added strike
    c             clear all visible strikes
    s / Enter     save heel_strikes.txt and close
    q             quit without saving
    toolbar       magnifier = zoom (drag a rectangle), arrow cross = pan,
                  home = reset view. While pan or zoom is toggled on it
                  owns the mouse: marking clicks are ignored until you
                  toggle the tool off again (keeps a pan click from
                  dropping a strike).

Notes:
    * Use a *_PC_signals.csv (the PC recordings have fewer sample losses);
      the saved times are absolute (EMG sample / 980), so they apply to
      BOTH the PC and SD runs of the trial - same convention as before.
    * Strikes outside the PC recording's time coverage cannot be edited
      here; they are carried over from the previous source and reported
      when saving.
    * An existing heel_strikes.txt is re-loaded only if it was written by
      this script (its first line says so). The old hand-written files
      (incomplete / partly stale times) are ignored on load - but they
      also stay inert in rerun_gait.sh for the same reason.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

PICKER_SENTINEL = "pick_heel_strikes.py"
DUP_TOL_S = 0.05          # clicks closer than this to an existing mark = duplicate
DRAG_PX = 5.0             # press->release motion beyond this = toolbar drag, not a click
REMOVE_TOL_FRAC = 0.05    # right-click must land within 5% of the visible span


# ----------------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------------

def read_meta(path):
    """'# key: value' header comments of a trial CSV -> dict."""
    meta = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                k, v = body.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def parse_strikes(text):
    """'45.8, 46.9, ...' -> sorted list of floats ([] for empty input)."""
    if not text:
        return []
    return sorted(float(x) for x in text.split(",") if x.strip())


def is_picked_file(path):
    """True if heel_strikes.txt was written by this script (safe to load)."""
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        return PICKER_SENTINEL in f.readline()


def read_picked_file(path):
    """Data lines of a picked heel_strikes.txt -> sorted list of floats."""
    parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts.append(line)
    return parse_strikes(",".join(parts))


def load_existing(trial_dir, meta):
    """Previous strikes for this trial and where they came from.

    A heel_strikes.txt written by this script wins (it reflects the latest
    session); otherwise fall back to the 'heel_strikes_s' header of the
    signals.csv (what the pipeline currently uses). Old hand-written txt
    files are deliberately ignored - their times are incomplete/stale.
    """
    txt = os.path.join(trial_dir, "heel_strikes.txt")
    if is_picked_file(txt):
        return read_picked_file(txt), "heel_strikes.txt (previous session)"
    return parse_strikes(meta.get("heel_strikes_s", "")), "signals.csv header"


# ----------------------------------------------------------------------------
# Picker
# ----------------------------------------------------------------------------

class Picker:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.trial_dir = os.path.dirname(self.path)
        self.saved = False

        meta = read_meta(self.path)
        df = pd.read_csv(self.path, comment="#")
        for col in ("t_s", "acc_pc1_g", "acc_pc2_g", "rate_pc1_dps"):
            if col not in df.columns:
                raise SystemExit(f"{self.path}: missing column '{col}'")

        self.fs = float(meta.get("emg_fs_hz", 980.0))
        t0_imu = float(meta.get("imu_t0_s", 0.0))
        # Absolute EMG-sample time base - exactly what --heel-strikes expects
        self.t = df["t_s"].to_numpy(float) + t0_imu
        self.pc1 = df["acc_pc1_g"].to_numpy(float)
        self.pc2 = df["acc_pc2_g"].to_numpy(float)
        self.rate1 = df["rate_pc1_dps"].to_numpy(float)
        self.t0, self.t1 = float(self.t[0]), float(self.t[-1])

        existing, source = load_existing(self.trial_dir, meta)
        # In-coverage strikes are editable; out-of-coverage ones are kept
        # untouched so nothing is silently dropped on save.
        self.strikes = [s for s in existing if self.t0 <= s <= self.t1]
        self.outside = [s for s in existing if s < self.t0 or s > self.t1]
        print(f"Loaded {len(existing)} previous strike(s) from {source} "
              f"({len(self.strikes)} editable, {len(self.outside)} outside "
              f"PC coverage {self.t0:.3f}..{self.t1:.3f} s)")
        if not is_picked_file(os.path.join(self.trial_dir, "heel_strikes.txt")):
            print(f"Note: no picked heel_strikes.txt yet - saving creates one.")

        self._build_figure()
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._sync()

    # -- figure ------------------------------------------------------------

    def _build_figure(self):
        import matplotlib.pyplot as plt

        self.lines = {}          # strike time -> (axvline on accel, on rate)
        self.texts = []          # index labels (rebuilt on every change)
        self._press = None

        # Explicit margins instead of constrained_layout: a resized window
        # made constrained_layout warn "axes sizes collapsed to zero".
        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1, figsize=(20, 6.5),
                                                     sharex=True)
        self.fig.subplots_adjust(left=0.045, right=0.99, top=0.87,
                                 bottom=0.11, hspace=0.10)

        self.ax.plot(self.t, self.pc1, lw=1.0, color="k", label="accel PC1")
        self.ax.plot(self.t, self.pc2, lw=0.8, color="tab:red", alpha=0.6,
                     label="accel PC2")
        self.ax.set_ylabel("dyn accel (g)")
        self.ax.tick_params(labelbottom=False)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right", fontsize=8)

        self.ax2.plot(self.t, self.rate1, lw=1.0, color="k",
                      label="rate PC1 (sagittal)")
        self.ax2.set_ylabel("ang. rate (deg/s)")
        self.ax2.set_xlabel("time (s, absolute EMG-sample/980 base - "
                            "matches --heel-strikes)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc="upper right", fontsize=8)

        self.ax.set_xlim(self.t0, self.t1)

    def _sync(self):
        """Sort marks, redraw index labels and update the title."""
        order = sorted(self.strikes)
        for txt in self.texts:
            txt.remove()
        self.texts = [
            self.ax.text(s, 0.99, str(i + 1),
                         transform=self.ax.get_xaxis_transform(),
                         ha="center", va="top", fontsize=7, color="tab:blue")
            for i, s in enumerate(order)
        ]
        extra = (f" | {len(self.outside)} outside PC coverage (kept on save)"
                 if self.outside else "")
        self.ax.set_title(
            f"{os.path.basename(self.path)} - {len(self.strikes)} heel "
            f"strike(s){extra}\n"
            "left click = add (either panel)   right click = remove "
            "nearest   u = undo   c = clear   s = save   q = quit\n"
            "toolbar: zoom/pan first, then toggle the tool off again - "
            "marking clicks are ignored while it is active",
            fontsize=9.5)
        self.fig.canvas.draw_idle()

    # -- editing -----------------------------------------------------------

    def add(self, x):
        if not (self.t0 <= x <= self.t1):
            print(f"  ignored: {x:.4f} s is outside the PC recording "
                  f"({self.t0:.3f}..{self.t1:.3f} s)")
            return
        xs = round(x * self.fs) / self.fs        # snap to the nearest EMG sample
        if any(abs(xs - s) < DUP_TOL_S for s in self.strikes):
            print(f"  duplicate ignored ({xs:.4f} s already marked)")
            return
        self.lines[xs] = (self.ax.axvline(xs, color="tab:blue", ls="--",
                                          lw=1.2),
                          self.ax2.axvline(xs, color="tab:blue", ls="--",
                                           lw=1.2))
        self.strikes.append(xs)
        print(f"  + strike {xs:.4f} s ({len(self.strikes)} total)")
        self._sync()

    def remove_nearest(self, x):
        if not self.strikes:
            print("  nothing to remove")
            return
        nearest = min(self.strikes, key=lambda s: abs(s - x))
        tol = REMOVE_TOL_FRAC * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
        if abs(nearest - x) > tol:
            print("  right-click within 5% of the visible span of a mark")
            return
        self.remove(nearest)

    def remove(self, s):
        for line in self.lines.pop(s):
            line.remove()
        self.strikes.remove(s)
        print(f"  - strike {s:.4f} s ({len(self.strikes)} total)")
        self._sync()

    def undo(self):
        if not self.strikes:
            print("  nothing to undo")
            return
        self.remove(max(self.strikes, key=self.strikes.index))  # last added

    def clear(self):
        for lines in self.lines.values():
            for line in lines:
                line.remove()
        self.lines.clear()
        self.strikes.clear()
        print("  cleared all editable strikes")
        self._sync()

    # -- save --------------------------------------------------------------

    def save(self):
        all_strikes = sorted(self.strikes + self.outside)
        if not all_strikes:
            print("  nothing to save - add at least one strike")
            return
        out = os.path.join(self.trial_dir, "heel_strikes.txt")
        with open(out, "w") as f:
            f.write(f"# heel strikes written by {PICKER_SENTINEL} "
                    f"(s, absolute EMG-sample/980 base)\n")
            f.write(f"# source: {os.path.basename(self.path)}\n")
            f.write(", ".join(f"{s:.4f}" for s in all_strikes) + "\n")
        print(f"\nSaved {len(all_strikes)} strike(s) -> {out}"
              + (f" (incl. {len(self.outside)} outside PC coverage, "
                 f"carried over)" if self.outside else ""))
        print(", ".join(f"{s:.4f}" for s in all_strikes))
        print("rerun_gait.sh will pick this file up automatically.")
        self.saved = True
        import matplotlib.pyplot as plt
        plt.close(self.fig)

    # -- events ------------------------------------------------------------

    def _nav_active(self):
        """True while the toolbar's pan or zoom tool is toggled on.

        While a nav tool is active it owns the mouse; marking clicks are
        ignored so a click that just starts a pan cannot drop a strike.
        """
        mode = getattr(getattr(self.fig.canvas, "toolbar", None),
                       "mode", None)
        if mode is None:
            return False
        return str(mode) in ("pan/zoom", "zoom rect")

    def on_press(self, event):
        # Remember the press; the action happens on release so toolbar
        # zoom/pan drags (which always move) never add or remove marks.
        if event.inaxes in (self.ax, self.ax2) and not self._nav_active():
            self._press = (event.x, event.y, event.button)

    def on_release(self, event):
        press, self._press = self._press, None
        if press is None:
            return
        px, py, button = press
        if event.button != button or event.inaxes not in (self.ax, self.ax2):
            return
        if event.xdata is None:
            return
        if (event.x - px) ** 2 + (event.y - py) ** 2 > DRAG_PX ** 2:
            return                                  # drag = zoom/pan, not a click
        if event.button == 1:
            self.add(event.xdata)
        elif event.button == 3:
            self.remove_nearest(event.xdata)

    def on_key(self, event):
        if event.key == "u":
            self.undo()
        elif event.key == "c":
            self.clear()
        elif event.key in ("s", "enter", "return"):
            self.save()
        elif event.key == "q":
            import matplotlib.pyplot as plt
            plt.close(self.fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(__doc__)
    path = sys.argv[1]
    if not os.path.isfile(path):
        raise SystemExit(f"no such file: {path}")
    if "_PC_signals.csv" not in os.path.basename(path):
        print(f"WARNING: {os.path.basename(path)} is not a *_PC_signals.csv "
              f"- the picker uses the PC recordings (fewer sample losses)")

    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        raise SystemExit(f"matplotlib backend '{backend}' is not interactive "
                         f"- unset MPLBACKEND and run again")

    import matplotlib.pyplot as plt

    picker = Picker(path)
    plt.show()
    if not picker.saved:
        print("Closed WITHOUT saving - heel_strikes.txt unchanged.")


if __name__ == "__main__":
    main()
