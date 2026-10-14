"""Subphase 1.8: save real weights and reproduce output in another process."""

import json
import subprocess
import sys

import torch

from storyseed.checkpoint import load_checkpoint, save_checkpoint
from storyseed.evidence import PROJECT_ROOT, record
from storyseed.generation import generate
from storyseed.training import train_bigram


def main() -> None:
    model, tokenizer, training = train_bigram()
    path = PROJECT_ROOT / "checkpoints" / "phase01" / "bigram.pt"
    artifact = save_checkpoint(path, model, tokenizer, training)
    restored, restored_tokenizer, _ = load_checkpoint(path)
    assert torch.equal(model.scores.detach().cpu(), restored.scores.detach())
    assert restored_tokenizer.tokens == tokenizer.tokens
    expected = generate(model, tokenizer, "mia ", seed=17)
    in_process = generate(restored, restored_tokenizer, "mia ", seed=17)
    assert expected == in_process
    child = subprocess.run([
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate.py"),
        "--checkpoint", str(path), "--prompt", "mia ", "--seed", "17", "--json",
    ], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=45)
    fresh = json.loads(child.stdout)
    assert fresh == expected
    record("08_checkpoint", {
        "command": ".venv/Scripts/python lessons/phase01/08_checkpoint.py",
        "checkpoint_relative_path": "checkpoints/phase01/bigram.pt",
        "artifact": artifact, "parameter_count": model.scores.numel(),
        "training_device": training["actual_device"], "load_device": str(restored.scores.device),
        "final_training_metrics": training["history"][-1],
        "weight_values_exactly_restored": True, "tokenizer_order_exactly_restored": True,
        "fresh_process_exit_code": child.returncode,
        "fresh_process_result_matches": True, "fresh_process_output": fresh,
        "optimizer_resume_supported": False,
    })


if __name__ == "__main__":
    main()
