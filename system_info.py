import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


info = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "gpu_memory_gb": (
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        )
        if torch.cuda.is_available()
        else None
    ),
    "nvidia_smi": run_command("nvidia-smi"),
}


Path("results/system_info.json").write_text(
    json.dumps(info, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(info, indent=2, ensure_ascii=False))