import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from codecarbon import EmissionsTracker

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPTS = [
    "Explain the second law of thermodynamics in three concise sentences.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "A train travels 120 miles in 2 hours, then 180 miles in 3 hours. What is its average speed?",
    "Summarize the main differences between TCP and UDP.",
    "What causes inflation in a modern economy? Give three key factors.",
] * 4  # Exact same 20 prompts as profile_energy.py

GGUF_MODELS = {
    "GGUF_Q8_0": ("qwen25-3b-q8", r"D:\AI_Models\qwen2.5-3b-instruct-q8_0.gguf"),
    "GGUF_Q4_K_M": ("qwen25-3b-q4", r"D:\AI_Models\qwen2.5-3b-instruct-q4_k_m.gguf"),
}

results = []

for name, (ollama_tag, file_path) in GGUF_MODELS.items():
    print(f"\n{'='*50}\nLoading {name} ({ollama_tag})...\n{'='*50}")
    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"Model File Size: {file_size_gb:.2f} GB")

    # Warmup pass (loads model into GPU VRAM outside the energy tracker)
    requests.post(OLLAMA_URL, json={
        "model": ollama_tag,
        "prompt": "Hello",
        "stream": False,
        "options": {"num_predict": 10, "temperature": 0.0}
    }).raise_for_status()

    tracker = EmissionsTracker(
        project_name=f"Qwen2.5_3B_{name}",
        measure_power_secs=1,
        output_dir="./results",
        log_level="error",
    )

    total_tokens = 0
    tracker.start()
    start_time = time.time()

    pbar = tqdm(PROMPTS, desc=f"Profiling {name}", unit="prompt", colour="cyan")
    for prompt in pbar:
        resp = requests.post(OLLAMA_URL, json={
            "model": ollama_tag,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 128, "temperature": 0.0}
        }).json()

        total_tokens += resp.get("eval_count", 0)
        elapsed = time.time() - start_time
        pbar.set_postfix(tokens=total_tokens, speed=f"{total_tokens/elapsed:.1f} tok/s")

    end_time = time.time()
    emissions_kg = tracker.stop()

    elapsed_time = end_time - start_time
    energy_kwh = tracker._total_energy.kWh
    energy_joules = energy_kwh * 3_600_000

    results.append({
        "Model": MODEL_ID,
        "Precision": name,
        "VRAM_GB": round(file_size_gb, 2),
        "Total_Tokens": total_tokens,
        "Time_s": round(elapsed_time, 2),
        "Throughput_tok_s": round(total_tokens / elapsed_time, 2),
        "Energy_Joules": round(energy_joules, 2),
        "Joules_per_Token": round(energy_joules / total_tokens, 4),
        "CO2_g": round(emissions_kg * 1000, 4),
    })

    # Unload model from GPU VRAM before starting the next one
    requests.post(OLLAMA_URL, json={"model": ollama_tag, "keep_alive": 0})
    time.sleep(2)

# Merge with existing energy_summary.csv
new_df = pd.DataFrame(results)
csv_path = "results/energy_summary.csv"
if os.path.exists(csv_path):
    old_df = pd.read_csv(csv_path)
    old_df = old_df[~old_df["Precision"].isin(GGUF_MODELS.keys())]
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
else:
    combined_df = new_df

combined_df.to_csv(csv_path, index=False)
print("\nUpdated Energy Summary:")
print(combined_df.to_string(index=False))