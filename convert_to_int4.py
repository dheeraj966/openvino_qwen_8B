import os
import time
import shutil
import sys
from pathlib import Path

SOURCE_DIR = Path("models/qwen3_gpu")
TARGET_DIR = Path("models/qwen3_int4")
MODEL_XML = SOURCE_DIR / "openvino_model.xml"

# Hugging Face model ID for fresh export (used if nncf can't re-compress)
HF_MODEL_ID = "Qwen/Qwen3-8B"

def try_nncf_compress():
    """Attempt in-place nncf compression of existing IR weights."""
    import openvino as ov
    import nncf

    core = ov.Core()
    print("[*] Reading existing model weights...", flush=True)
    model = core.read_model(MODEL_XML)

    print("[*] Attempting nncf INT4 compression...", flush=True)
    compressed_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        group_size=64,
        ratio=1.0,
    )

    # Check if compression actually did anything
    ov.save_model(compressed_model, TARGET_DIR / "openvino_model.xml")
    new_size = (TARGET_DIR / "openvino_model.bin").stat().st_size / (1024**3)

    if new_size >= 6.0:
        # Compression didn't work (weights were already quantized in IR)
        print(f"[!] nncf output is {new_size:.2f} GB — no real compression.", flush=True)
        # Clean up the useless output
        shutil.rmtree(TARGET_DIR, ignore_errors=True)
        return False

    print(f"[*] nncf compressed to {new_size:.2f} GB", flush=True)
    return True


def export_from_hf():
    """Re-export from Hugging Face with optimum-intel INT4 quantization."""
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer

    print(f"[*] Exporting {HF_MODEL_ID} → INT4 via optimum-intel...", flush=True)
    print("[*] This downloads weights from HF and quantizes to INT4.", flush=True)
    print("[*] May take a while on first run (downloading ~16 GB)...", flush=True)

    model = OVModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        export=True,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    model.save_pretrained(str(TARGET_DIR))

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(str(TARGET_DIR))

    new_size = (TARGET_DIR / "openvino_model.bin").stat().st_size / (1024**3)
    print(f"[*] Exported model size: {new_size:.2f} GB", flush=True)
    return True


def copy_support_files():
    """Copy tokenizer/config files from source if missing in target."""
    for item in SOURCE_DIR.iterdir():
        dest = TARGET_DIR / item.name
        if item.is_file() and item.suffix not in ['.xml', '.bin', '.blob'] and not dest.exists():
            try:
                shutil.copy2(item, dest)
            except Exception:
                pass  # Skip locked/cache files


def convert():
    print(f"\n[*] STARTING INT4 CONVERSION...", flush=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    # Strategy 1: Try nncf on existing IR
    print("\n--- Strategy 1: nncf re-compression of existing IR ---", flush=True)
    try:
        if try_nncf_compress():
            copy_support_files()
            elapsed = time.perf_counter() - start_time
            final_size = (TARGET_DIR / "openvino_model.bin").stat().st_size / (1024**3)
            print(f"\n[SUCCESS] INT4 model: {final_size:.2f} GB (took {elapsed:.0f}s)")
            return
    except Exception as e:
        print(f"[!] nncf failed: {e}", flush=True)
        shutil.rmtree(TARGET_DIR, ignore_errors=True)
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy 2: Fresh export from Hugging Face
    print("\n--- Strategy 2: Fresh INT4 export from Hugging Face ---", flush=True)
    try:
        if export_from_hf():
            copy_support_files()
            elapsed = time.perf_counter() - start_time
            final_size = (TARGET_DIR / "openvino_model.bin").stat().st_size / (1024**3)
            print(f"\n[SUCCESS] INT4 model: {final_size:.2f} GB (took {elapsed:.0f}s)")
            return
    except Exception as e:
        print(f"\n[ERROR] HF export failed: {e}", flush=True)

    print("\n[FAILED] Could not produce INT4 model with either strategy.", flush=True)
    sys.exit(1)

if __name__ == "__main__":
    convert()
