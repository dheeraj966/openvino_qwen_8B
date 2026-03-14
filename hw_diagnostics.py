"""
Hardware Diagnostics & Connectivity Checker
- Physical device detection (GPU, CPU, RAM)
- RAM connectivity and storage role verification
- GPU/CPU load monitoring for queue balancing decisions
"""

import os
import psutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class HardwareDiagnostics:
    """Detects and validates physical hardware connections for inference."""

    def __init__(self):
        self._ov_core = None
        self._gpu_available = False
        self._cpu_available = False
        self._ram_connected = False
        self._devices = []
        self._device_properties = {}
        self._ram_baseline_mb = 0.0

    def _get_ov_core(self):
        if self._ov_core is None:
            import openvino as ov
            self._ov_core = ov.Core()
        return self._ov_core

    # ------------------------------------------------------------------ #
    #  1. PHYSICAL HARDWARE CONNECTION CHECKS
    # ------------------------------------------------------------------ #
    def run_full_diagnostics(self) -> Dict:
        """Run all hardware connectivity checks. Returns a status dict."""
        results = {
            "gpu": self._check_gpu(),
            "cpu": self._check_cpu(),
            "ram": self._check_ram(),
            "summary": "",
            "recommended_device": "CPU",
        }

        failures = [k for k in ("gpu", "cpu", "ram") if not results[k]["connected"]]
        if failures:
            results["summary"] = f"CONNECTION ISSUES: {', '.join(f.upper() for f in failures)}"
        else:
            results["summary"] = "ALL HARDWARE CONNECTED"

        # Recommend device
        if results["gpu"]["connected"] and results["gpu"]["usable_for_inference"]:
            results["recommended_device"] = "GPU"
        else:
            results["recommended_device"] = "CPU"

        return results

    def _check_gpu(self) -> Dict:
        """Detect GPU via OpenVINO device enumeration."""
        info = {
            "connected": False,
            "usable_for_inference": False,
            "device_name": "N/A",
            "error": None,
        }
        try:
            core = self._get_ov_core()
            self._devices = core.available_devices
            if "GPU" in self._devices:
                info["connected"] = True
                try:
                    info["device_name"] = core.get_property("GPU", "FULL_DEVICE_NAME")
                except Exception:
                    info["device_name"] = "GPU (name query unsupported)"
                # Quick sanity — if we can read any property, the driver is alive
                info["usable_for_inference"] = True
                self._gpu_available = True
            else:
                info["error"] = "GPU not found in OpenVINO available_devices"
        except Exception as e:
            info["error"] = str(e)
        return info

    def _check_cpu(self) -> Dict:
        """Detect CPU via OpenVINO + psutil."""
        info = {
            "connected": False,
            "usable_for_inference": False,
            "device_name": "N/A",
            "logical_cores": 0,
            "error": None,
        }
        try:
            core = self._get_ov_core()
            if not self._devices:
                self._devices = core.available_devices
            if "CPU" in self._devices:
                info["connected"] = True
                info["usable_for_inference"] = True
                self._cpu_available = True
                try:
                    info["device_name"] = core.get_property("CPU", "FULL_DEVICE_NAME")
                except Exception:
                    info["device_name"] = "CPU (name query unsupported)"
            info["logical_cores"] = psutil.cpu_count(logical=True)
        except Exception as e:
            info["error"] = str(e)
        return info

    def _check_ram(self) -> Dict:
        """Check RAM connectivity — total, available, and whether OS can allocate."""
        info = {
            "connected": False,
            "total_gb": 0.0,
            "available_gb": 0.0,
            "percent_used": 0.0,
            "allocation_test": False,
            "error": None,
        }
        try:
            mem = psutil.virtual_memory()
            info["total_gb"] = round(mem.total / (1024 ** 3), 2)
            info["available_gb"] = round(mem.available / (1024 ** 3), 2)
            info["percent_used"] = mem.percent

            # Allocation sanity test — try to allocate & release 1 MB
            try:
                test_block = bytearray(1 * 1024 * 1024)  # 1 MB
                del test_block
                info["allocation_test"] = True
            except MemoryError:
                info["allocation_test"] = False
                info["error"] = "RAM allocation test failed — possible connectivity issue"

            if info["total_gb"] > 0 and info["allocation_test"]:
                info["connected"] = True
                self._ram_connected = True
            else:
                info["error"] = info.get("error") or "RAM reported 0 GB total"
        except Exception as e:
            info["error"] = str(e)
        return info

    # ------------------------------------------------------------------ #
    #  2. GPU / CPU LOAD MONITORING (for queue balancing)
    # ------------------------------------------------------------------ #
    def get_gpu_utilization(self) -> Optional[float]:
        """
        Attempt to read GPU utilisation %.
        Falls back to None if not readable (integrated GPUs often lack counters).
        """
        # Try reading from OpenVINO properties first
        try:
            core = self._get_ov_core()
            # Some drivers expose GPU_UTILIZATION or similar
            util = core.get_property("GPU", "GPU_UTILIZATION")
            return float(util)
        except Exception:
            pass

        # Fallback: try iGPU / dGPU Windows perf counter via psutil-style check
        try:
            import subprocess
            # Use a lightweight WMI call on Windows
            out = subprocess.check_output(
                'wmic path Win32_VideoController get AdapterRAM',
                shell=True, timeout=3, stderr=subprocess.DEVNULL
            ).decode()
            # If we got a response, GPU driver is alive — but no utilisation
            return None  # Driver alive, no utilisation counter
        except Exception:
            return None

    def get_system_load(self) -> Dict:
        """Snapshot of current CPU + RAM load for balancing decisions."""
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": round((mem.total - mem.available) / (1024 ** 3), 2),
            "ram_available_gb": round(mem.available / (1024 ** 3), 2),
            "ram_percent": mem.percent,
        }

    # ------------------------------------------------------------------ #
    #  3. RAM STORAGE ROLE VERIFICATION
    # ------------------------------------------------------------------ #
    def snapshot_ram(self) -> float:
        """Record current process RSS in MB. Call BEFORE loading weights."""
        proc = psutil.Process(os.getpid())
        self._ram_baseline_mb = proc.memory_info().rss / (1024 ** 2)
        return self._ram_baseline_mb

    def verify_ram_weight_storage(self) -> Dict:
        """
        Compare current process RSS against baseline to determine
        how much RAM is storing model weights / intermediate KV-cache.
        Call AFTER the model is loaded.
        """
        proc = psutil.Process(os.getpid())
        current_mb = proc.memory_info().rss / (1024 ** 2)
        delta_mb = current_mb - self._ram_baseline_mb

        system_mem = psutil.virtual_memory()

        return {
            "baseline_mb": round(self._ram_baseline_mb, 1),
            "current_mb": round(current_mb, 1),
            "model_weight_delta_mb": round(delta_mb, 1),
            "ram_is_storing_weights": delta_mb > 50,  # >50 MB delta = meaningful
            "system_ram_available_gb": round(system_mem.available / (1024 ** 3), 2),
            "system_ram_percent_used": system_mem.percent,
            "role": self._classify_ram_role(delta_mb),
        }

    @staticmethod
    def _classify_ram_role(delta_mb: float) -> str:
        if delta_mb > 2000:
            return "PRIMARY_WEIGHT_STORE — RAM holds full model weights, active in inference"
        elif delta_mb > 500:
            return "KV_CACHE_AND_BUFFERS — RAM stores KV-cache and intermediate tensors"
        elif delta_mb > 50:
            return "BUFFER_ONLY — RAM stores tokenizer + I/O buffers"
        else:
            return "MINIMAL — weights reside on device memory (GPU VRAM)"

    def print_full_report(self):
        """Pretty-print the full diagnostics report."""
        diag = self.run_full_diagnostics()
        load = self.get_system_load()
        ram_role = self.verify_ram_weight_storage()

        print("\n" + "=" * 55)
        print("  HARDWARE DIAGNOSTICS REPORT")
        print("=" * 55)

        # GPU
        g = diag["gpu"]
        status = "CONNECTED" if g["connected"] else "NOT DETECTED"
        print(f"\n  [GPU]  {status}")
        print(f"         Device   : {g['device_name']}")
        print(f"         Inference: {'YES' if g['usable_for_inference'] else 'NO'}")
        if g["error"]:
            print(f"         Issue    : {g['error']}")

        # CPU
        c = diag["cpu"]
        status = "CONNECTED" if c["connected"] else "NOT DETECTED"
        print(f"\n  [CPU]  {status}")
        print(f"         Device   : {c['device_name']}")
        print(f"         Cores    : {c['logical_cores']}")
        print(f"         Load     : {load['cpu_percent']:.1f}%")

        # RAM
        r = diag["ram"]
        status = "CONNECTED" if r["connected"] else "** NOT CONNECTED **"
        print(f"\n  [RAM]  {status}")
        print(f"         Total    : {r['total_gb']} GB")
        print(f"         Available: {r['available_gb']} GB")
        print(f"         Used     : {r['percent_used']}%")
        print(f"         Alloc OK : {'YES' if r['allocation_test'] else 'FAIL'}")
        if r["error"]:
            print(f"         Issue    : {r['error']}")

        # RAM weight storage role
        print(f"\n  [RAM ROLE]")
        print(f"         Baseline (pre-model) : {ram_role['baseline_mb']:.0f} MB")
        print(f"         Current (post-model) : {ram_role['current_mb']:.0f} MB")
        print(f"         Weight delta          : {ram_role['model_weight_delta_mb']:.0f} MB")
        print(f"         Storing weights?      : {'YES' if ram_role['ram_is_storing_weights'] else 'NO'}")
        print(f"         Classification        : {ram_role['role']}")

        # Recommendation
        print(f"\n  [RECOMMENDATION]")
        print(f"         Primary device: {diag['recommended_device']}")
        print("=" * 55 + "\n")


# Quick standalone test
if __name__ == "__main__":
    hd = HardwareDiagnostics()
    hd.snapshot_ram()
    hd.print_full_report()
