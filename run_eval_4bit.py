import os
import gc
import json
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

CONFIGS = {
    "eval_fp16": None,
    "eval_8bit": BitsAndBytesConfig(load_in_8bit=True),
}

for folder_name, quant_config in CONFIGS.items():
    print(f"\n{'='*50}\nEvaluating {MODEL_ID} [{folder_name}]...\n{'='*50}")

    # --- THIS IS THE UPDATED PART ---
    load_kwargs = {"device_map": "auto", "dtype": torch.float16}
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    # --------------------------------

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=["arc_challenge", "gsm8k"],
        limit=100,
    )

    out_dir = os.path.join("results", folder_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results["results"], f, indent=2)

    print(f"Saved results to {out_path}")
    print(json.dumps(results["results"], indent=2))

    del lm_obj, model
    gc.collect()
    torch.cuda.empty_cache()