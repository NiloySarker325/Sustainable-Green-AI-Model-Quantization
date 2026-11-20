import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load energy profiling results
energy_df = pd.read_csv("results/energy_summary.csv")

# 2. Map all 5 precisions to their lm-eval output folders
eval_folders = {
    "FP16": "results/eval_fp16",
    "8-bit": "results/eval_8bit",
    "4-bit_NF4": "results/eval_4bit",
    "GGUF_Q8_0": "results/eval_gguf_q8",
    "GGUF_Q4_K_M": "results/eval_gguf_q4",
}

def extract_metric(task_dict, prefixes):
    for prefix in prefixes:
        for key, val in task_dict.items():
            if key.startswith(prefix) and "stderr" not in key and isinstance(val, (int, float)):
                return val * 100
    return None

accuracy_rows = []
for precision, folder in eval_folders.items():
    json_files = glob.glob(f"{folder}/**/*.json", recursive=True)
    if not json_files:
        print(f"Warning: No JSON results found in {folder}")
        continue

    latest_file = max(json_files)
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = data.get("results", data)
    arc_score = extract_metric(res.get("arc_challenge", {}), ["acc_norm", "acc"])
    gsm_score = extract_metric(res.get("gsm8k", {}), ["exact_match,flexible-extract", "exact_match,strict-match", "exact_match"])

    accuracy_rows.append({
        "Precision": precision,
        "ARC_Challenge_%": round(arc_score, 2) if arc_score is not None else None,
        "GSM8K_%": round(gsm_score, 2) if gsm_score is not None else None,
    })

acc_df = pd.DataFrame(accuracy_rows)
full_df = pd.merge(energy_df, acc_df, on="Precision", how="left")

# Calculate Energy Savings % relative to FP16
fp16_energy = full_df.loc[full_df["Precision"] == "FP16", "Joules_per_Token"].values[0]
full_df["Energy_vs_FP16_%"] = ((full_df["Joules_per_Token"] - fp16_energy) / fp16_energy * 100).round(1)

full_df.to_csv("results/full_benchmark.csv", index=False)

print("\n=== Complete 5-Way Energy-Accuracy Table ===")
cols = ["Precision", "VRAM_GB", "Throughput_tok_s", "Joules_per_Token", "Energy_vs_FP16_%", "GSM8K_%", "ARC_Challenge_%"]
print(full_df[cols].to_string(index=False))

# 3. Generate 2-Panel Publication Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: Energy vs. GSM8K Accuracy (Pareto Frontier)
ax1 = axes[0]
colors = {
    "FP16": "#2b5c8f",
    "8-bit": "#d95f02",
    "4-bit_NF4": "#7570b3",
    "GGUF_Q8_0": "#1b9e77",
    "GGUF_Q4_K_M": "#e7298a",
}

for _, row in full_df.iterrows():
    p = row["Precision"]
    ax1.scatter(
        row["Joules_per_Token"],
        row["GSM8K_%"],
        s=row["VRAM_GB"] * 85,
        color=colors.get(p, "#333333"),
        alpha=0.85,
        edgecolors="black",
        linewidth=1.2,
        zorder=3,
    )
    ax1.annotate(
        f"{p}\n({row['VRAM_GB']} GB)",
        (row["Joules_per_Token"], row["GSM8K_%"]),
        textcoords="offset points",
        xytext=(0, 13),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
    )

ax1.set_title("Energy–Accuracy Trade-off (Qwen2.5-3B, Bubble = VRAM)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Energy per Token (Joules/tok) — Lower is Better")
ax1.set_ylabel("GSM8K Math Accuracy (%) — Higher is Better")
ax1.set_ylim(55, 72)
ax1.set_xlim(0, 16.5)
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 2: Energy per Token Bar Chart (with % change vs FP16)
ax2 = axes[1]
bar_colors = [colors.get(p, "#333333") for p in full_df["Precision"]]
bars = ax2.bar(full_df["Precision"], full_df["Joules_per_Token"], color=bar_colors, edgecolor="black", alpha=0.85)

for bar, pct in zip(bars, full_df["Energy_vs_FP16_%"]):
    height = bar.get_height()
    label = "Baseline" if pct == 0 else f"{pct:+.1f}%"
    ax2.annotate(
        f"{height:.2f} J\n({label})",
        (bar.get_x() + bar.get_width() / 2, height),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
    )

ax2.set_title("Energy Consumption per Token by Runtime & Precision", fontsize=11, fontweight="bold")
ax2.set_ylabel("Energy per Token (Joules/tok)")
ax2.set_ylim(0, max(full_df["Joules_per_Token"]) * 1.22)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/final_5way_tradeoff.png", dpi=300)
print("\nSaved chart to results/final_5way_tradeoff.png")
plt.show()