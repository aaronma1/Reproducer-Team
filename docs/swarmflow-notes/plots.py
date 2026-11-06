"""Render charts for the swarmflow notes from the agents' own run files (read-only)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.expanduser("~/Projects/reproducer-swarmflow/runs")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (light)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


def band(ax, x, ys, color, label):
    ys = np.asarray(ys)
    m = ys.mean(0)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=color, label=label)
    return m


# ---------------------------------------------------------------- reppo-retrying
R = f"{RUNS}/20261027-213301-reppo-retrying/work/repo"
arms = [
    ("RePPO m=1.2 (V2, live Q)", f"{R}/eval/reppo12_ref/v2"),
    ("RePPO m=1.2 (V1, frozen Q)", f"{R}/eval/reppo12_ref/v1"),
    ("PPO-V + Ent", f"{R}/arms/ppo-v-ent/results"),
    ("PPO-Q + Ent (auxiliary)", f"{R}/arms/ppo-q-ent/results"),
]
data = {}
for name, d in arms:
    runs = [json.load(open(f"{d}/run_seed{s}.json")) for s in range(10)]
    data[name] = runs

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for (name, _), c in zip(arms, S):
    runs = data[name]
    x = np.array([p["env_steps"] for p in runs[0]["eval_curve"]]) / 1e6
    ys = [[p["raw_return_mean"] for p in r["eval_curve"]] for r in runs]
    m = band(ax, x, ys, c, name)
    ax.annotate(f"{m[-1]:.0f}", (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", color=INK2, fontsize=9)
ax.set_xlabel("environment steps (millions)")
ax.set_ylabel("eval return (100 argmax episodes)")
ax.set_title("MinAtar Breakout: eval return during training, mean of 10 seeds (band = min-max)")
ax.legend(loc="upper left")
save(fig, "reppo_return_curves.png")

fig, ax = plt.subplots(figsize=(7.5, 4.0))
rng = np.random.default_rng(0)
for i, ((name, _), c) in enumerate(zip(arms, S)):
    v = np.array([r["final"]["raw_return_mean"] for r in data[name]])
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=40, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.hlines(v.mean(), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
    ax.annotate(f"mean {v.mean():.1f}", (i + 0.32, v.mean()), va="center", fontsize=9, color=INK2)
ax.set_xticks(range(len(arms)), [n.replace(" (", "\n(") for n, _ in arms])
ax.set_ylabel("final eval return (~10M steps)")
ax.set_title("Final return per seed (dots) and seed mean (bar)")
ax.set_xlim(-0.5, len(arms) - 0.1)
save(fig, "reppo_final_per_seed.png")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for (name, _), c in zip(arms, S):
    ys = [r["train_entropy_curve"] for r in data[name]]
    ys = [[(p["mean_policy_entropy"] if isinstance(p, dict) else p) for p in y] for y in ys]
    band(ax, np.arange(1, len(ys[0]) + 1), ys, c, name)
ax.set_xlabel("training iteration (131,072 env steps each)")
ax.set_ylabel("policy entropy (nats)")
ax.set_title("Policy entropy during training, mean of 10 seeds (band = min-max)")
ax.legend(loc="upper right")
save(fig, "reppo_entropy.png")

# ---------------------------------------------------------------- edge-of-stability
E = f"{RUNS}/20261104-124700-edge-of-stability/work/repo/arms/fc-tanh-edgestability/runs"
eos = [("eta = 0.02 (2/eta = 100)", "gd-lr-2over100", 100), ("eta = 0.01 (2/eta = 200)", "gd-lr-2over200", 200)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for (name, arm, thr), c in zip(eos, S):
    for s in range(3):
        r = json.load(open(f"{E}/{arm}/seed{s}.json"))
        sh = np.array(r["curves"]["sharpness"]); lo = np.array(r["curves"]["train_mse"])
        kw = dict(color=c, alpha=0.9 if s == 0 else 0.45, linewidth=2 if s == 0 else 1.2,
                  label=name if s == 0 else None)
        axes[0].plot(sh[:, 0], sh[:, 1], **kw)
        axes[1].plot(lo[:, 0], lo[:, 1], **kw)
    axes[0].axhline(thr, color=c, linestyle="--", linewidth=1.2)
    axes[0].annotate(f"2/eta = {thr}", (1.0, thr), xycoords=("axes fraction", "data"), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=9, color=INK2)
axes[0].set_title("Top Hessian eigenvalue (sharpness)")
axes[0].set_xlabel("GD iteration"); axes[0].set_ylabel("sharpness")
axes[1].set_title("Training loss (MSE)")
axes[1].set_xlabel("GD iteration"); axes[1].set_ylabel("train MSE")
axes[0].legend(loc="center right")
fig.suptitle("FC-tanh on 5k CIFAR-10, full-batch GD, 3 seeds per step size (bold = seed 0)", color=INK2, fontsize=10)
save(fig, "eos_sharpness_loss.png")

# ---------------------------------------------------------------- lottery-ticket-small
L = f"{RUNS}/20261104-124658-lottery-ticket-small/work/repo/arms/conv2-iterative-lottery/runs"
def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{L}/{pat}"))]
dense = load("dense/seed*.json"); ticket = load("ticket/seed*_level*.json"); reinit = load("reinit/seed*_level*_r*.json")

def by_pm(runs, key):
    g = {}
    for r in runs:
        g.setdefault(round(r["pm_at_level"], 2), []).append(r["metrics"][key])
    pm = sorted(g, reverse=True)
    return np.array(pm), [np.array(g[p]) for p in pm]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, key, title, ylab in [(axes[0], "test_acc_final", "Test accuracy at end of training (20k iterations)", "test accuracy (%)"),
                             (axes[1], "val_min_iter", "Early-stop iteration (minimum validation loss)", "iteration")]:
    for runs, name, c in [(ticket, "winning ticket (reset to theta_0)", S[0]), (reinit, "random reinit, same mask", S[1])]:
        pm, vals = by_pm(runs, key)
        m = np.array([v.mean() for v in vals])
        ax.fill_between(pm, [v.min() for v in vals], [v.max() for v in vals], color=c, alpha=0.15, linewidth=0)
        ax.plot(pm, m, color=c, marker="o", markersize=5, markeredgecolor=SURF, label=name)
    dv = np.array([r["metrics"][key] for r in dense])
    ax.axhline(dv.mean(), color=INK2, linestyle="--", linewidth=1.2, label="dense (unpruned)")
    ax.set_xscale("log"); ax.invert_xaxis()
    ticks = [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.9]
    ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("percent of weights remaining (Pm)")
    ax.set_ylabel(ylab); ax.set_title(title)
axes[0].legend(loc="lower left")
fig.suptitle("Conv-2 on CIFAR-10, iterative magnitude pruning; mean of 3 seeds (reinit: 3 seeds x 3 inits), band = min-max",
             color=INK2, fontsize=10)
save(fig, "lottery_acc_and_earlystop.png")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
show = [(100.0, dense, "dense, Pm 100"), (33.02, ticket, "ticket, Pm 33.0"), (3.68, ticket, "ticket, Pm 3.7"),
        (3.68, reinit, "reinit, Pm 3.7")]
for (pm, runs, name), c in zip(show, S):
    rs = [r for r in runs if abs(r["pm_at_level"] - pm) < 0.05]
    x = np.array([p[0] for p in rs[0]["curves"]["test_acc"]])
    band(ax, x, [[p[1] for p in r["curves"]["test_acc"]] for r in rs], c, f"{name} (n={len(rs)})")
ax.set_xscale("log")
ax.set_xlabel("training iteration"); ax.set_ylabel("test accuracy (%)")
ax.set_title("Test accuracy during training (mean, band = min-max)")
ax.legend(loc="lower right")
save(fig, "lottery_test_acc_curves.png")

# tuned vs untuned ticket accuracy (tuner probes, 3 seeds each)
T = f"{RUNS}/20261104-124658-lottery-ticket-small/work/repo/tuned/conv2-iterative-lottery/probes/final"
fig, ax = plt.subplots(figsize=(8, 4.4))
sets = [("Adam lr 2e-4 (paper setting)", ticket, dense, S[0]),
        ("Adam lr 5e-4 (tuner)", [json.load(open(f)) for f in glob.glob(f"{T}/final_lr5p0e4/ticket/*.json")],
         [json.load(open(f)) for f in glob.glob(f"{T}/final_lr5p0e4/dense/*.json")], S[2])]
for name, tr, dr, c in sets:
    pm, vals = by_pm(tr, "test_acc_final")
    m = np.array([v.mean() for v in vals])
    ax.fill_between(pm, [v.min() for v in vals], [v.max() for v in vals], color=c, alpha=0.15, linewidth=0)
    ax.plot(pm, m, color=c, marker="o", markersize=5, markeredgecolor=SURF, label=f"ticket, {name}")
    dv = np.mean([r["metrics"]["test_acc_final"] for r in dr])
    ax.axhline(dv, color=c, linestyle="--", linewidth=1.2)
    ax.annotate(f"dense {dv:.1f}", (1.0, dv), xycoords=("axes fraction", "data"), xytext=(-4, -12),
                textcoords="offset points", ha="right", fontsize=9, color=INK2)
ax.errorbar([3.7], [68.75], yerr=[[1.25], [1.25]], fmt="D", color=INK, markersize=7, capsize=3,
            label="paper, ticket at Pm 3.7 (Fig. 5, read by the figure agents)")
ax.set_xscale("log"); ax.invert_xaxis()
ticks = [100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.9]
ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
ax.set_xlabel("percent of weights remaining (Pm)"); ax.set_ylabel("test accuracy at 20k iterations (%)")
ax.set_title("Winning-ticket accuracy before and after the tuner, 3 seeds each (band = min-max)")
ax.legend(loc="lower left", fontsize=9)
save(fig, "lottery_tuned_vs_paper.png")
