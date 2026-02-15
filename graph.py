# # Save this as make_plots_from_preds.py and run with: python make_plots_from_preds.py
# # Or paste into a Colab cell and run.

# import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
# from pathlib import Path

# OUT = Path("plots")
# OUT.mkdir(exist_ok=True)

# # Paths (adjust if needed)
# PREDS = Path("outputs_plus/preds.pt")      # or /mnt/data/preds.pt
# MU = Path("outputs_plus/mu.json")          # or /mnt/data/mu.json
# STD = Path("outputs_plus/std.json")        # or /mnt/data/std.json
# CSV = Path("Renew.csv")                    # or /mnt/data/Renew.csv
# METRICS = Path("outputs_plus/metrics.json")  # optional

# # LOAD
# assert PREDS.exists(), f"{PREDS} not found. Put preds.pt at this path."
# data = torch.load(str(PREDS), map_location="cpu")
# # expected keys: "samples", "truth"
# S = data.get("samples", None)   # shape expected: (S_samples, B, N, T)
# Y = data.get("truth", None)     # shape expected: (B, N, T)
# if S is None or Y is None:
#     raise RuntimeError("preds.pt does not contain 'samples' or 'truth' keys.")

# S = S.cpu().numpy()
# Y = Y.cpu().numpy()

# mu = pd.read_json(str(MU), typ="series") if MU.exists() else None
# std = pd.read_json(str(STD), typ="series") if STD.exists() else None

# # Function to inverse transform (if mu/std available)
# def inv_transform(x_norm):
#     # x_norm shape: (..., N, T)
#     if mu is None or std is None:
#         return x_norm
#     # mu/std are series indexed by variable names or indices
#     # Convert to numpy arrays in same variable order as stored
#     mu_v = mu.values.reshape(1, -1, 1)    # (1,N,1)
#     std_v = std.values.reshape(1, -1, 1)
#     return x_norm * std_v + mu_v

# # 1) Time-series plot for a chosen batch item and a few variables
# batch_idx = 0
# vars_to_plot = list(range(min(3, S.shape[2])))  # first 3 variables
# horizon = S.shape[-1]
# x = np.arange(horizon)

# S_mean = S.mean(axis=0)   # (B,N,T)
# S_p5   = np.percentile(S, 5, axis=0)
# S_p95  = np.percentile(S, 95, axis=0)

# # inverse transform
# S_mean_it = inv_transform(S_mean)
# S_p5_it   = inv_transform(S_p5)
# S_p95_it  = inv_transform(S_p95)
# Y_it      = inv_transform(Y)

# for vi in vars_to_plot:
#     fig, ax = plt.subplots(figsize=(10,4))
#     ax.plot(x, Y_it[batch_idx, vi], label="Truth")
#     ax.plot(x, S_mean_it[batch_idx, vi], label="Mean pred")
#     ax.fill_between(x, S_p5_it[batch_idx, vi], S_p95_it[batch_idx, vi], alpha=0.25, label="5-95% band")
#     ax.set_title(f"Batch {batch_idx} — Variable {vi} — Truth vs mean prediction")
#     ax.set_xlabel("Horizon step")
#     ax.set_ylabel("Value")
#     ax.legend()
#     fig.savefig(OUT / f"ts_var_{vi}.png", bbox_inches="tight")
#     plt.close(fig)

# # 2) Heatmap of variable covariance (samples averaged over horizon)
# # Combine sample & batch axes: S -> (S_total, N, T)
# S_comb = S.reshape(-1, S.shape[2], S.shape[3])
# S_avg_time = S_comb.mean(axis=-1)   # (S_total, N)
# cov = np.cov(S_avg_time.T)          # (N,N)

# fig, ax = plt.subplots(figsize=(6,5))
# im = ax.imshow(cov, aspect="auto")
# ax.set_title("Covariance between variables (samples averaged over horizon)")
# ax.set_xlabel("Variable index"); ax.set_ylabel("Variable index")
# fig.colorbar(im, ax=ax)
# fig.savefig(OUT / "vars_cov_heatmap.png", bbox_inches="tight")
# plt.close(fig)

# # 3) mu/std bar plots (if available)
# if mu is not None and std is not None:
#     fig, ax = plt.subplots(figsize=(10,3.5))
#     ax.bar(np.arange(len(mu)), mu.values)
#     ax.set_title("Per-variable training mean (mu)")
#     ax.set_xlabel("Variable index"); ax.set_ylabel("mu")
#     fig.savefig(OUT / "mu_bar.png", bbox_inches="tight")
#     plt.close(fig)

#     fig, ax = plt.subplots(figsize=(10,3.5))
#     ax.bar(np.arange(len(std)), std.values)
#     ax.set_title("Per-variable training std (std)")
#     ax.set_xlabel("Variable index"); ax.set_ylabel("std")
#     fig.savefig(OUT / "std_bar.png", bbox_inches="tight")
#     plt.close(fig)

# # 4) If CSV present, plot last 48 rows of PV_production (or first numeric column)
# if CSV.exists():
#     df = pd.read_csv(str(CSV))
#     if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
#     if "Time" in df.columns:
#         df["Time"] = pd.to_datetime(df["Time"].str.replace("T"," "), errors="coerce")
#         df = df.sort_values("Time").set_index("Time")
#     # choose PV_production if present
#     col_candidates = ["PV_production","PV Production","PV_production(MW)"]
#     col = next((c for c in col_candidates if c in df.columns), None)
#     if col is None:
#         # choose first numeric column
#         col = df.select_dtypes(include=[float,int]).columns[0]
#     s = df[col].iloc[-48:]
#     fig, ax = plt.subplots(figsize=(10,3.5))
#     ax.plot(s.index, s.values)
#     ax.set_title(f"Last 48 rows of {col}")
#     ax.set_xlabel("Time"); ax.set_ylabel(col)
#     fig.savefig(OUT / "csv_last48.png", bbox_inches="tight")
#     plt.close(fig)

# print("Saved plots to:", OUT)
# print("Files created:", list(OUT.iterdir()))








# 







import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Paths ----------------
PREDS = Path("outputs_plus/preds.pt")
MU    = Path("outputs_plus/mu.json")
STD   = Path("outputs_plus/std.json")
OUT   = Path("analysis_plots")
OUT.mkdir(exist_ok=True)

# ---------------- Load ----------------
data = torch.load(str(PREDS), map_location="cpu")
S = data["samples"].cpu().numpy()    # (S_samples, B, N, T)
Y = data["truth"].cpu().numpy()      # (B_truth, N, T)

S_samples, B_pred, N, T = S.shape
B_truth = Y.shape[0]

mu  = pd.read_json(str(MU), typ="series") if MU.exists() else None
std = pd.read_json(str(STD), typ="series") if STD.exists() else None

# ---------------- Variable names ----------------
VARS = [
    "DHI","DNI","GHI","Wind_speed","Humidity",
    "Temperature","PV_production","Wind_production","Electric_demand"
]

# ---------------- Inverse transform ----------------
def inv_transform(x):
    if mu is None or std is None:
        return x
    mu_v  = mu.values.reshape(1, -1, 1)
    std_v = std.values.reshape(1, -1, 1)
    return x * std_v + mu_v

# ---------------- Fix mismatch: Trim predictions to truth count ----------------
S = S[:, :B_truth]                        # (S_samples, 32, N, T)
Y_i = inv_transform(Y)
S_mean = S.mean(axis=0)                   # (32, N, T)
S_med  = np.median(S, axis=0)
S_p05  = np.percentile(S, 5, axis=0)
S_p95  = np.percentile(S, 95, axis=0)
S_p25  = np.percentile(S, 25, axis=0)
S_p75  = np.percentile(S, 75, axis=0)

# inverse transform
S_mean_i = inv_transform(S_mean)
S_med_i  = inv_transform(S_med)
S_p05_i  = inv_transform(S_p05)
S_p95_i  = inv_transform(S_p95)
S_p25_i  = inv_transform(S_p25)
S_p75_i  = inv_transform(S_p75)

# ---------------- Errors ----------------
ERR = S_mean_i - Y_i      # now (32, 9, 24)

# ---------------- Helpers ----------------
def flatten(BNt, v):
    return BNt[:, v, :].reshape(-1)

def ensure_axes(fig, axes):
    return axes.ravel().tolist()

# ===============================================================
# 1) ERROR HISTOGRAMS (all variables)
# ===============================================================
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = ensure_axes(fig, axes)

for v in range(N):
    e = flatten(ERR, v)
    ax = axes[v]
    ax.hist(e, bins=30, alpha=0.8)
    ax.set_title(f"Error distribution: {VARS[v]}", fontsize=9)
    ax.set_xlabel("Prediction - Truth")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

fig.suptitle("Error Distributions for All Variables", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "errors_hist_all_variables.png", dpi=300)
plt.close(fig)

# ===============================================================
# 2) SCATTER: TRUE VS PRED
# ===============================================================
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = ensure_axes(fig, axes)

for v in range(N):
    true = flatten(Y_i, v)
    pred = flatten(S_mean_i, v)
    ax = axes[v]
    ax.scatter(true, pred, s=5, alpha=0.4)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, "k--")
    ax.set_title(f"True vs Pred: {VARS[v]}", fontsize=9)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    ax.grid(True, alpha=0.3)

fig.suptitle("True vs Predicted Scatter Plots", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "scatter_true_vs_pred_all_variables.png", dpi=300)
plt.close(fig)

# ===============================================================
# 3) HORIZON-WISE MAE & RMSE
# ===============================================================
mae_h = np.mean(np.abs(ERR), axis=(0,1))      # (T,)
rmse_h = np.sqrt(np.mean(ERR**2, axis=(0,1)))

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(mae_h, label="MAE")
ax.plot(rmse_h, label="RMSE")
ax.set_title("Horizon-wise MAE & RMSE (all variables)")
ax.set_xlabel("Horizon step")
ax.set_ylabel("Error")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "horizon_mae_rmse.png", dpi=300)
plt.close(fig)

# ===============================================================
# 4) CRPS PER VARIABLE
# ===============================================================
def crps_empirical(samples, truth):
    S_ = samples.shape[0]
    term1 = np.mean(np.abs(samples - truth))
    diff = np.abs(samples[:,None,:] - samples[None,:, :])
    term2 = 0.5 * np.mean(diff)
    return term1 - term2

crps = np.zeros(N)
for v in range(N):
    samples_v = S[:, :, v, :].reshape(S_samples, -1)
    truth_v   = Y_i[:, v, :].reshape(-1)
    crps[v] = crps_empirical(samples_v, truth_v)

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(np.arange(N), crps)
ax.set_xticks(np.arange(N))
ax.set_xticklabels(VARS, rotation=45, ha="right")
ax.set_title("CRPS per Variable")
ax.set_ylabel("CRPS (lower = better)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "crps_per_variable.png", dpi=300)
plt.close(fig)

# ===============================================================
# 5) SPAGHETTI PLOTS
# ===============================================================
batch_idx = 0     # show example 0
t = np.arange(T)

fig, axes = plt.subplots(3, 3, figsize=(13,10))
axes = ensure_axes(fig, axes)

for v in range(N):
    ax = axes[v]

    # all sample trajectories
    for s in range(S_samples):
        ax.plot(t, S[s, batch_idx, v], alpha=0.3, linewidth=0.6, color="gray")

    # truth + median
    ax.plot(t, Y_i[batch_idx, v], "k-", linewidth=2, label="Truth")
    ax.plot(t, S_med_i[batch_idx, v], "r--", linewidth=1.5, label="Median")

    ax.set_title(f"Spaghetti: {VARS[v]}", fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Spaghetti Plots (Batch 0)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "spaghetti_batch0.png", dpi=300)
plt.close(fig)

print("\nAll plots saved to:", OUT)
for p in sorted(OUT.iterdir()):
    print(" -", p.name)
