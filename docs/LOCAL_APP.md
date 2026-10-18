# Run StorySeed locally

## On the current development laptop

Open PowerShell in `C:/Users/JACKIV/OneDrive/Desktop/StorySeed`:

```powershell
.venv/Scripts/python -m storyseed.app --gui
.venv/Scripts/python -m storyseed.app --prompt "Mia found a small red box." --seed 17 --json
.venv/Scripts/python -m storyseed.app --prompt "Once upon a time" --output runs/my-story.json
```

For the desktop window without a terminal, run `scripts/Start-StorySeed.ps1` in PowerShell. This uses the existing project Python; it does not install dependencies. Do not bypass system execution policy to run an unknown script. CLI `--help` lists controls. GUI controls are set inside the window; other generation flags are CLI settings.

The default loads the exact main128 inference artifact at checkpoints/phase07/main128-inference.pt and its Phase 2 tokenizer. The model was trained from random initialization. There is no remote inference API, automatic download, corpus access or training step in app generation. Missing weights are reported, not downloaded silently.

## Create the portable local bundle

```powershell
.venv/Scripts/python scripts/package_local_app.py
```

This creates `exports/StorySeed-main128` and `exports/StorySeed-main128.zip`. Existing output is preserved: choose `--destination exports/StorySeed-main128-v2` for another release. Check reports/phase09/03_package-*.json for the actual archive identity. The final handoff identifies the most recent complete bundle, which also includes model card/report/reconstruction material.

## Another folder or machine

Extract the ZIP and use Python 3.12 (the observed environment is 3.12.4) with Tcl/Tk for the GUI. Linux distributions may package Tk separately; the CLI does not require it. Create the environment inside the extracted folder:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-inference.txt
.venv/Scripts/python launch.py --json
.venv/Scripts/python launch.py --gui
```

These direct dependency pins are not a complete transitive hash lock. Dependency installation may require internet and substantial disk space; this bundle does not include Python or wheels. Once installed, local inference requires no network. The current laptop's working torch is 2.6.0+cu124; the plain torch==2.6.0 requirement accepts compatible builds and the app itself uses CPU. Alternate machines/clean dependency installations are not claimed as observed unless recorded.

`launch.py` locates files relative to itself, verifies the manifest, and imports the included runtime source. It does not rely on the original Desktop path or an editable project installation. The manifest catches accidental file changes; replacing both files and the manifest is not prevented by an unsigned manifest. Use the project-published archive digest when verifying a copied archive's identity.

## Interpret the controls

| Control | Default | Meaning |
|---|---:|---|
| Seed | 17 | Request-local CPU sampling sequence; reproducibility depends on runtime too |
| Temperature | 0.8 | Logit scaling; accepted 0.05–2 |
| Top-k | 40 | Retain k candidate logits after forbidding PAD/BOS; accepted 1–510 |
| New IDs | 48 | At most 1–256 sampled IDs, including possible EOS; not word count |

Prompts can be empty (BOS-only input) or up to 4,000 characters. Literal reserved control strings are rejected. The model supplies only the newest 128 input IDs on each step; the displayed full prompt can exceed that. The application reports initial cropping. EOS stops early; a length stop is not a completed narrative. Save result writes settings, IDs and hashes with the text; Copy story copies text only.

## Diagnose common problems

- Missing checkpoint: the Git repository intentionally omits weights. Use the local release ZIP or reproduce the declared training workflow; cloning alone does not supply the trained artifact.
- Changed tokenizer/model: keep the exact matched pair; vocabulary size alone cannot establish compatibility.
- Tk unavailable/no display: use the CLI, or install Python/Tk appropriate to the machine.
- Slow response: requests run CPU FP32, one at a time, with no KV cache. Try a smaller new-ID budget; this changes output length, not model knowledge.
- Strange grammar or forgotten opening: these are documented model limitations. Examine the actual context/sampling settings before assuming an app bug.

The app does not expose a web port or save prompts automatically. JSON saving is explicit. Human quality ratings and learner reconstruction remain pending. The final evaluation population has already been used.
