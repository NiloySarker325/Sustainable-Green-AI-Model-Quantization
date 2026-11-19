import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your energy profiling results
energy_df = pd.read_csv("results/energy_summary.csv")

# 2. Map precision names to their lm-eval output folders
eval_folders = {
    "FP16": "results/eval_fp16",
    "8-bit": "results/eval_8bit",
    "4-bit_NF4": "results/eval_4bit",
}

def extract_metric(task_dict, preferred_keys):
    for key in preferred_keys:
        if key in task_dict:
            return task_dict[key] * 100  # Convert to percentage
    return None

accuracy_rows = []
for precision, folder in eval_folders.items():
    json_files = glob.glob(f"{folder}/**/*.json", recursive=True)
    if not json_files:
        print(f"Warning: No JSON results found in {folder}")
        continue

    # Grab the most recent results file in that folder
    latest_file = max(json_files)
    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = data.get("results", {})
    arc_score = extract_metric(res.get("arc_challenge", {}), ["acc_norm,none", "acc_norm", "acc,none", "acc"])
    gsm_score = extract_metric(res.get("gsm8k", {}), ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match"])

    accuracy_rows.append({
        "Precision": precision,
        "ARC_Challenge_%": round(arc_score, 2) if arc_score is not None else None,
        "GSM8K_%": round(gsm_score, 2) if gsm_score is not None else None,
    })

acc_df = pd.DataFrame(accuracy_rows)

# 3. Merge Energy + Accuracy and save
full_df = pd.merge(energy_df, acc_df, on="Precision", how="left")
full_df["Avg_Accuracy_%"] = full_df[["ARC_Challenge_%", "GSM8K_%"]].mean(axis=1).round(2)
full_df.to_csv("results/full_benchmark.csv", index=False)

print("\n=== Combined Energy-Accuracy Table ===")
print(full_df[["Precision", "VRAM_GB", "Throughput_tok_s", "Joules_per_Token", "ARC_Challenge_%", "GSM8K_%", "Avg_Accuracy_%"]].to_string(index=False))

# 4. Generate Pareto Frontier & Efficiency Chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Energy per Token vs. Average Accuracy (Pareto Frontier)
ax1 = axes[0]
colors = ["#2b5c8f", "#d95f02", "#1b9e77"]
for i, row in full_df.iterrows():
    ax1.scatter(
        row["Joules_per_Token"],
        row["Avg_Accuracy_%"],
        s=row["VRAM_GB"] * 80,  # Bubble size = VRAM usage
        color=colors[i % len(colors)],
        alpha=0.85,
        edgecolors="black",
        linewidth=1.2,
    )
    ax1.annotate(
        f"{row['Precision']}\n({row['VRAM_GB']} GB)",
        (row["Joules_per_Token"], row["Avg_Accuracy_%"]),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

ax1.set_title("Energy–Accuracy Trade-off (Bubble Size = VRAM)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Energy per Token (Joules/tok) — Lower is Better")
ax1.set_ylabel("Average Accuracy: ARC + GSM8K (%) — Higher is Better")
ax1.grid(True, linestyle="--", alpha=0.5)

# Plot 2: Side-by-Side Comparison of ARC vs GSM8K
ax2 = axes[1]
x = range(len(full_df))
width = 0.35
ax2.bar([p - width/2 for p in x], full_df["ARC_Challenge_%"], width, label="ARC-Challenge", color="#4c72b0")
ax2.bar([p + width/2 for p in x], full_df["GSM8K_%"], width, label="GSM8K (Math)", color="#dd8452")
ax2.set_xticks(list(x))
ax2.set_xticklabels(full_df["Precision"])
ax2.set_ylim(0, 100)
ax2.set_title("Benchmark Accuracy by Quantization", fontsize=11, fontweight="bold")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/energy_accuracy_tradeoff.png", dpi=300)
print("\nSaved chart to results/energy_accuracy_tradeoff.png")
plt.show()