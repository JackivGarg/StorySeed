"""Subphase 1.1: compute a derivative on CPU and the available GPU."""

import platform
import sys
from pathlib import Path

import torch

from storyseed.evidence import record


def calculate(device: str) -> dict:
    weight = torch.tensor(2.0, device=device, requires_grad=True)
    prediction = weight * 3.0
    loss = (prediction - 5.0).square()
    loss.backward()
    assert weight.grad is not None
    assert torch.allclose(weight.grad, torch.tensor(6.0, device=device))
    with torch.no_grad():
        updated = weight - 0.1 * weight.grad
        new_loss = (updated * 3.0 - 5.0).square()
    return {
        "device": device,
        "weight": weight.item(),
        "prediction": prediction.item(),
        "loss": loss.item(),
        "gradient": weight.grad.item(),
        "updated_weight": updated.item(),
        "loss_after_update": new_loss.item(),
    }


def main() -> None:
    torch.set_num_threads(1)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    config = Path(sys.prefix) / "pyvenv.cfg"
    record("01_environment", {
        "command": ".venv/Scripts/python lessons/phase01/01_environment.py",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "shares_installed_dependencies": config.exists() and "include-system-site-packages = true" in config.read_text().lower(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "worked_calculations": [calculate(device) for device in devices],
    })


if __name__ == "__main__":
    main()
