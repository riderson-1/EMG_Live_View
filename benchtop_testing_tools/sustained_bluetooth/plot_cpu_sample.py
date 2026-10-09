import re, sys, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/session_0001.log"
# Map raw thread names (thread struct addresses in the log) to legend labels.
THREAD_LABELS = {
    "0x200025c0": "SD thread", "0x200026d8": "IMU thread", "0x200027f0": "Log thread",
    "0x20002908": "LED thread", "0x20002a20": "Transport thread", "0x20002b38": "EMG thread",
}
TS = r"\[(\d+):(\d+):(\d+)\.(\d+),(\d+)\]"
def t(g): h,m,s,ms,us = map(int,g[:5]); return h*3600+m*60+s+ms/1e3+us/1e6

emg, imu, cpu, last_t, tname = [], [], [], 0.0, None
for l in open(LOG, errors="ignore"):
    m = re.match(TS, l)
    if m: last_t = t(m.groups())
    if l.startswith("Thread analyze"): cpu.append((last_t, {}))
    m = re.match(r"\s*(\S.*?)\s*: STACK:", l)
    if m: tname = m.group(1)
    m = re.match(r"\s*: Total CPU cycles used: (\d+)", l)
    if m and cpu and tname: cpu[-1][1][tname] = int(m.group(1)); tname = None
    m = re.match(TS+r".*EMG/s: collected=(\d+) live_q_drop=(\d+) sd_q_drop=(\d+)", l)
    if m: emg.append((t(m.groups()), *map(int, m.groups()[5:8])))
    m = re.match(TS+r".*IMU/s: collected=(\d+) live_q_drop=(\d+) sd_q_drop=(\d+)", l)
    if m: imu.append((t(m.groups()), *map(int, m.groups()[5:8])))
emg, imu = np.array(emg), np.array(imu)

def rate(a):  # counters are per print interval, which jitters 1.0-1.23 s -> divide by real dt
    dt = np.diff(a[:,0]); r = a[1:,1]/dt
    r[dt > 1.6] = np.nan   # a report line was garbled/missing -> its count is unknown, not a loss
    return a[1:,0]-a[0,0], dt, r, a[1:,2], a[1:,3]

te, dte, re_, lqe, sde = rate(emg)
ti, dti, ri, lqi, sdi = rate(imu)
nom = np.nanmedian(re_); gap_e = dte > 1.6; print("EMG intervals with a missing report line:", int(gap_e.sum()))

fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
ax[0].plot(te, re_, lw=.8, label="EMG"); ax[0].axhline(nom, ls="--", c="gray", lw=.8, label=f"median {nom:.0f} Hz")
ax[0].set_ylabel("EMG samples/s\n(normalised by real dt)"); ax[0].legend(loc="lower right")
ax[1].bar(te, lqe, width=1, label="live queue drops"); ax[1].bar(te, sde, width=1, bottom=lqe, label="SD queue drops")
ax[1].set_ylabel("EMG samples dropped\nper interval"); ax[1].legend()
ax[2].plot(te, np.cumsum(np.where(gap_e, nom*dte, emg[1:,1])) - nom*te, lw=.8)
ax[2].set_ylabel("cumulative samples\nminus median-rate x t")
# CPU load per thread, stacked to 100 %. The analyzer's own "CPU: x %" is a running average
# since boot, so use deltas of the cumulative cycle counters. Drop incomplete (garbled) blocks.
full = max(len(d) for _, d in cpu)
ref = next(d for _, d in cpu if len(d) == full)           # reference thread set
blk = [(tt - emg[0,0], d) for tt, d in cpu if set(d) == set(ref)]  # skip garbled blocks
names = list(ref); tc = np.array([b[0] for b in blk])
M = np.array([[d[n] for n in names] for _, d in blk], float)
M = np.diff(M, axis=0); tc = tc[1:]; M = 100 * M / M.sum(1, keepdims=True)
act = [i for i, n in enumerate(names) if n != "idle" and M[:,i].mean() >= 0.3]
idle = M[:, names.index("idle")]
other = np.clip(100 - M[:, act].sum(1) - idle, 0, None)   # tiny threads + ISR + rounding
S = np.vstack([M[:, act].T, other, idle]); S *= 100 / np.maximum(S.sum(0), 100)
lab = [THREAD_LABELS.get(names[i], names[i]) for i in act] + ["other (small threads)", "idle"]
col = list(plt.cm.tab10.colors[:len(act)]) + ["#bbbbbb", "#eeeeee"]
ax[3].stackplot(tc, S, labels=lab, colors=col); ax[3].set_ylim(0, 100)
ax[3].set_ylabel("share of thread CPU time [%]\n(cycle deltas, ISR not incl.)"); ax[3].set_xlabel("time since boot [s]")
ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8)
print("CPU blocks used/dropped:", len(blk), len(cpu) - len(blk))
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"); os.makedirs(_out, exist_ok=True); plt.savefig(os.path.join(_out, f"samples_overview_{LOG.split('/')[-1].rsplit('.',1)[0]}.png"), dpi=130)
print("EMG drops live/sd:", lqe.sum(), sde.sum(), "| IMU:", lqi.sum(), sdi.sum())
print("intervals with any EMG drop:", int(((lqe+sde)>0).sum()), "of", len(te))