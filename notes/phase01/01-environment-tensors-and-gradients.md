# 1.1: our first calculation becomes executable

Our destination is a model that continues a story. Before it can learn language, we need to understand a smaller event: a parameter produces a prediction, that prediction receives an error, and an update changes the parameter. You have seen this pattern in ANN training. Today we make every number visible rather than beginning with a large network.

Read this chapter, then [the exercises](01-practice-and-solutions.md). Our code is [01_environment.py](../../lessons/phase01/01_environment.py), and its actual execution is recorded in [01_environment.md](../../reports/phase01/01_environment.md). No language model is trained by this first calculation.

## The environment is part of the experiment

A virtual environment gives this project its own Python entry point and installation location. On this machine we created it with:

```powershell
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\python lessons\phase01\01_environment.py
```

The explicit `--system-site-packages` option lets it see the already-installed PyTorch CUDA build. This is a project environment with shared dependencies, not complete dependency isolation. Its operation still depends on that existing installation. The report records this choice so it cannot be confused with a clean installation.

For a separate fully isolated setup, omit that option and install the pinned requirements before installing our package:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-phase01.txt
.venv\Scripts\python -m pip install --no-deps -e .
```

These are alternative setup paths; do not recreate the environment while a run is using it. The requirements record the actual 2.6.0+cu124 build. Future CUDA compatibility must be checked on the target machine. The `-e .` installation is editable: Python imports our source package directly, so editing a module changes what the next process executes. It neither trains nor downloads a model.

## Tensors and the meaning of a device

A tensor is an array of numbers with a shape. A scalar has shape `[]`; a vector of three values has shape `[3]`; a two-by-three matrix has shape `[2,3]`. Shape describes organization, not intelligence. Later a batch of token IDs and a batch of prediction scores will have different shapes and meanings.

An integer tensor can contain categorical token indices. A floating-point tensor can contain trainable weights. A tensor's device says where its storage and calculations reside. CPU and CUDA tensors cannot be arbitrarily combined in an operation. We calculate the same example on CPU and, when available, CUDA; agreement verifies this small supported operation, not that a future transformer will fit in GPU memory.

## Follow one parameter through a prediction

Let `w` be a trainable scalar weight, `x = 3` our input, and `y = 5` the observed answer. Our prediction is `p = wx`. Choose squared error for this introductory calculation:

```text
L = (p - y)^2 = (3w - 5)^2
```

At `w = 2`, the prediction is 6, the residual is 1, and the loss is 1. The derivative uses the chain rule:

```text
dL/dw = 2(3w - 5) × 3 = 6
```

The derivative describes the local change in loss per small change in weight. A positive derivative here means increasing the weight increases the loss locally. Gradient descent subtracts a scaled derivative. With learning rate `η = 0.1`:

```text
w_new = w - η(dL/dw) = 2 - 0.1 × 6 = 1.4
p_new = 3 × 1.4 = 4.2
L_new = (4.2 - 5)^2 = 0.64
```

The loss decreases even though the prediction crosses below the answer. This one update is not the exact optimum, `w = 5/3`. The learning rate controls the step size; a derivative is not a direct instruction to set the correct answer. Our later next-token model uses cross-entropy rather than squared error, but the prediction/loss/gradient/update pattern survives.

## Match each line to the calculation

`torch.tensor(2.0, requires_grad=True)` creates a floating-point scalar whose derivative we want. Multiplication and squaring build a computation graph: PyTorch records how the resulting loss depends on the weight. `loss.backward()` applies differentiation through that graph and accumulates the result in `weight.grad`.

The code then enters `torch.no_grad()` for the update illustration. We want to inspect the numerical result without building another training graph. It computes a new tensor rather than mutating the original weight; later an optimizer will update registered parameters. `.item()` converts a single tensor value to a Python number suitable for the report. It is for reporting, not for maintaining differentiable calculations.

The assertion compares the computed gradient with our hand result, 6. That is a meaningful independent calculation. Floating-point output can show `1.399999976` instead of exactly 1.4; finite-precision representation explains this, and an approximate numerical comparison is appropriate.

## What this establishes and what follows

The observed report tells us the Python/PyTorch versions, whether dependency sharing is enabled, and which devices actually completed the calculation. We now have executable evidence that a scalar parameter can participate in a gradient calculation. We have not yet supplied text, vocabulary, model weights or a language-learning objective.

Next we must translate a sentence into numbers without changing what its characters mean. That translation creates the questions our first language model will answer.

Reference: [PyTorch autograd introduction](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html). Our scalar example and derivation are original worked teaching material.
