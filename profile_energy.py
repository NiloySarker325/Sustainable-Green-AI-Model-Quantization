import os
import gc
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from codecarbon import EmissionsTracker

os.makedirs("results", exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Standardized test prompts (expand this list or load from GSM8K/MMLU later)
PROMPTS = [
    "Explain the second law of thermodynamics in three concise sentences.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "A train travels 120 miles in 2 hours, then 180 miles in 3 hours. What is its average speed?",
    "Summarize the main differences between TCP and UDP.",
    "What causes inflation in a modern economy? Give three key factors.",
] * 4  # 20 prompts for a solid initial test run

CONFIGS = {
    "FP16": None,
    "8-bit": BitsAndBytesConfig(load_in_8bit=True),
    "4-bit_NF4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ),
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
results = []

for name, quant_config in CONFIGS.items():
    print(f"\n{'='*50}\nLoading {MODEL_ID} [{name}]...\n{'='*50}")

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    vram_gb = model.get_memory_footprint() / (1024 ** 3)
    print(f"Model Memory Footprint: {vram_gb:.2f} GB")

    # Warmup pass (excluded from energy tracking)
    warmup_inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=10)
    torch.cuda.synchronize()

    # Start Energy Tracking
    tracker = EmissionsTracker(
        project_name=f"Qwen2.5_3B_{name}",
        measure_power_secs=1,
        output_dir="./results",
        log_level="error",
    )

    total_tokens = 0
    tracker.start()
    start_time = time.time()

    pbar = tqdm(PROMPTS, desc=f"Profiling {name}", unit="prompt", colour="green")
    for prompt in pbar:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Deterministic for reproducibility
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += new_tokens

        elapsed = time.time() - start_time
        pbar.set_postfix(tokens=total_tokens, speed=f"{total_tokens/elapsed:.1f} tok/s")

    torch.cuda.synchronize()
    end_time = time.time()
    emissions_kg = tracker.stop()

    elapsed_time = end_time - start_time
    energy_kwh = tracker._total_energy.kWh
    energy_joules = energy_kwh * 3_600_000

    results.append({
        "Model": MODEL_ID,
        "Precision": name,
        "VRAM_GB": round(vram_gb, 2),
        "Total_Tokens": total_tokens,
        "Time_s": round(elapsed_time, 2),
        "Throughput_tok_s": round(total_tokens / elapsed_time, 2),
        "Energy_Joules": round(energy_joules, 2),
        "Joules_per_Token": round(energy_joules / total_tokens, 4),
        "CO2_g": round(emissions_kg * 1000, 4),
    })

    # Free GPU memory before loading the next precision
    del model
    gc.collect()
    torch.cuda.empty_cache()

df = pd.DataFrame(results)
df.to_csv("results/energy_summary.csv", index=False)
print("\nFinal Energy Summary:")
print(df.to_string(index=False))