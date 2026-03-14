import os
import time
import queue
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Generator
from threading import Thread, Lock, Event

from hw_diagnostics import HardwareDiagnostics


class QwenEngine:
    # GPU utilisation threshold (%) — above this, overflow goes to CPU
    GPU_OVERFLOW_THRESHOLD = 85
    # CPU utilisation threshold (%) — above this, try to push back to GPU
    CPU_OVERFLOW_THRESHOLD = 90

    def __init__(self, model_id_or_path: str, device: str = "GPU"):
        import openvino_genai as ov_genai
        self._genai = ov_genai

        self.model_path = Path(model_id_or_path)
        self.cache_dir = Path("model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Hardware diagnostics ──────────────────────────────────
        self.hw = HardwareDiagnostics()
        self.hw.snapshot_ram()                       # RAM baseline BEFORE model load
        self._hw_report = self.hw.run_full_diagnostics()
        self.hw.print_full_report()

        if not self._hw_report["ram"]["connected"]:
            print("[WARNING] RAM connectivity NOT confirmed — "
                  "intermediate weight storage may be impaired!", flush=True)

        # ── Device selection with fallback ────────────────────────
        requested_device = device
        self._primary_device = self._hw_report["recommended_device"]
        if requested_device == "GPU" and not self._hw_report["gpu"]["connected"]:
            print(f"[FALLBACK] GPU requested but not connected — using CPU", flush=True)
            self._primary_device = "CPU"
        else:
            self._primary_device = requested_device

        # ── Pipelines (primary + overflow) ────────────────────────
        ov_config = {
            "CACHE_DIR": str(self.cache_dir),
            "PERFORMANCE_HINT": "LATENCY",
        }

        self.pipe = None                 # primary pipeline
        self._overflow_pipe = None       # secondary (overflow) pipeline
        self._overflow_device = None
        self._active_device = self._primary_device
        self._device_lock = Lock()
        self._gpu_infer_count = 0
        self._cpu_infer_count = 0
        self._stop_flag = False
        self._gen_lock = Lock()
        self._gen_idle = Event()
        self._gen_idle.set()  # starts idle

        try:
            print(f"[*] Initializing PRIMARY pipeline on {self._primary_device}...", flush=True)
            t0 = time.perf_counter()
            self.pipe = ov_genai.LLMPipeline(str(self.model_path), self._primary_device, **ov_config)
            print(f"      Primary ready: {time.perf_counter() - t0:.2f}s")

            self.tokenizer = ov_genai.Tokenizer(str(self.model_path))

            # --- PRECISION DETECTION ---
            self.model_gb = sum(f.stat().st_size for f in self.model_path.glob('**/*') if f.is_file()) / (1024**3)
            self.precision = "INT4 (Compressed)" if self.model_gb < 5 else "INT8 (Standard)"

            print(f"[*] Model detected as {self.precision} (~{self.model_gb:.2f} GB)")

            # ── Build overflow pipeline (bidirectional) ───────────
            self._overflow_device = "CPU" if self._primary_device == "GPU" else "GPU"
            if self._overflow_device in [d for d in ("GPU", "CPU")
                                          if self._hw_report.get(d.lower(), {}).get("connected")]:
                print(f"[*] Initializing OVERFLOW pipeline on {self._overflow_device}...", flush=True)
                t1 = time.perf_counter()
                self._overflow_pipe = ov_genai.LLMPipeline(
                    str(self.model_path), self._overflow_device, **ov_config
                )
                print(f"      Overflow ready: {time.perf_counter() - t1:.2f}s")
            else:
                print(f"[*] No overflow device available ({self._overflow_device} not connected).")

            # ── RAM role verification (post-load) ─────────────────
            ram_role = self.hw.verify_ram_weight_storage()
            print(f"[*] RAM Role: {ram_role['role']}")
            print(f"[*] RAM weight delta: {ram_role['model_weight_delta_mb']:.0f} MB", flush=True)
            if ram_role["ram_is_storing_weights"]:
                print("[*] RAM confirmed storing model weights / KV-cache for inference.")
            else:
                print("[*] RAM holding minimal buffers — weights reside on device memory.")

            print(f"[*] Bandwidth Monitor: Calibrated.")
        except Exception as e:
            raise RuntimeError(f"Engine Load Failed: {e}")

    # ------------------------------------------------------------------ #
    #  QUEUE PRIORITY: GPU primary → overflow to CPU (bidirectional)
    # ------------------------------------------------------------------ #
    def _pick_pipeline(self):
        """
        Decide which pipeline to run inference on.
        - GPU is always preferred.
        - If system load suggests GPU is saturated, spill to CPU.
        - If CPU is also saturated but GPU freed up, swing back (bidirectional).
        Returns (pipeline, device_name).
        """
        with self._device_lock:
            load = self.hw.get_system_load()
            gpu_util = self.hw.get_gpu_utilization()  # may be None

            # If we have no overflow, always use primary
            if self._overflow_pipe is None:
                return self.pipe, self._primary_device

            primary_overloaded = False
            overflow_overloaded = False

            if self._primary_device == "GPU":
                # GPU utilisation counter available?
                if gpu_util is not None and gpu_util > self.GPU_OVERFLOW_THRESHOLD:
                    primary_overloaded = True
                # Also check RAM pressure as a proxy for iGPU shared memory
                elif load["ram_percent"] > 92:
                    primary_overloaded = True

                overflow_overloaded = load["cpu_percent"] > self.CPU_OVERFLOW_THRESHOLD
            else:
                # Primary is CPU
                primary_overloaded = load["cpu_percent"] > self.CPU_OVERFLOW_THRESHOLD
                if gpu_util is not None:
                    overflow_overloaded = gpu_util > self.GPU_OVERFLOW_THRESHOLD

            if primary_overloaded and not overflow_overloaded:
                print(f"[QUEUE] Primary ({self._primary_device}) overloaded → "
                      f"routing to {self._overflow_device}", flush=True)
                return self._overflow_pipe, self._overflow_device

            # Bidirectional: if overflow was being used but primary freed up, swing back
            return self.pipe, self._primary_device

    def cancel_generation(self):
        """Signal the running generation to stop and wait for it to finish."""
        self._stop_flag = True
        # Wait up to 10s for the generation thread to actually finish
        self._gen_idle.wait(timeout=10)
        self._stop_flag = False

    @property
    def is_generating(self):
        return not self._gen_idle.is_set()

    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[str, None, None]:
        # If a previous generation is still winding down, wait for it
        if not self._gen_idle.wait(timeout=15):
            yield "\n[Engine Error: previous generation did not finish in time]"
            return

        self._stop_flag = False
        self._gen_idle.clear()  # mark as busy

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        gen_config = self._genai.GenerationConfig()
        gen_config.max_new_tokens = kwargs.get('max_new_tokens', 1024)
        gen_config.temperature = kwargs.get('temperature', 0.7)
        gen_config.repetition_penalty = kwargs.get('repetition_penalty', 1.1)
        gen_config.do_sample = gen_config.temperature > 0.05

        # ── Pick device based on queue priority ───────────────────
        active_pipe, active_device = self._pick_pipeline()
        if active_device == "GPU":
            self._gpu_infer_count += 1
        else:
            self._cpu_infer_count += 1

        token_queue = queue.Queue()
        start_gen = time.perf_counter()
        token_count = 0
        stopped = False

        def _streamer_callback(token):
            """Called by OpenVINO for each token. Return True to abort."""
            if self._stop_flag:
                return True  # signal pipeline to stop
            token_queue.put(token)
            return False  # continue

        def run_gen():
            try:
                active_pipe.generate(prompt, gen_config, _streamer_callback)
            except Exception as e:
                err_msg = str(e)
                # Suppress the expected abort error
                if 'cancelled' not in err_msg.lower():
                    token_queue.put(f"\n[Engine Error: {e}]")
            finally:
                token_queue.put(None)
                self._gen_idle.set()  # mark as idle

        Thread(target=run_gen, daemon=True).start()

        while True:
            try:
                token = token_queue.get(timeout=0.05)
                if token is None:
                    elapsed = time.perf_counter() - start_gen
                    if token_count > 0 and not self._stop_flag:
                        tps = token_count / elapsed
                        bandwidth = self.model_gb * tps
                        print(f"\n[METRICS] Device: {active_device} | Mode: {self.precision}", flush=True)
                        print(f"[METRICS] Speed: {tps:.2f} tokens/sec", flush=True)
                        print(f"[METRICS] Bandwidth: {bandwidth:.2f} GB/s", flush=True)
                        print(f"[METRICS] GPU inferences: {self._gpu_infer_count} | "
                              f"CPU inferences: {self._cpu_infer_count}", flush=True)

                        # Post-inference RAM check
                        ram_snap = self.hw.verify_ram_weight_storage()
                        print(f"[METRICS] RAM delta: {ram_snap['model_weight_delta_mb']:.0f} MB "
                              f"({ram_snap['role']})", flush=True)
                    break

                if self._stop_flag:
                    # Drain remaining tokens without yielding
                    continue

                token_count += 1
                yield token
            except queue.Empty:
                if self._stop_flag:
                    # Still waiting for generation thread to put None
                    continue
                continue

    # ------------------------------------------------------------------ #
    #  HARDWARE STATUS (exposed via API)
    # ------------------------------------------------------------------ #
    def get_hardware_status(self) -> Dict:
        """Return a JSON-serialisable hardware status snapshot."""
        diag = self.hw.run_full_diagnostics()
        load = self.hw.get_system_load()
        ram_role = self.hw.verify_ram_weight_storage()
        return {
            "diagnostics": diag,
            "system_load": load,
            "ram_role": ram_role,
            "primary_device": self._primary_device,
            "overflow_device": self._overflow_device,
            "overflow_available": self._overflow_pipe is not None,
            "gpu_infer_count": self._gpu_infer_count,
            "cpu_infer_count": self._cpu_infer_count,
        }
