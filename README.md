# 🕵️‍♂️ ML CUDA Detective

**A comprehensive GPU environment health-check and package inventory tool for NVIDIA hardware running Python-based ML/AI workloads on Linux and Windows.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg)]()
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Kepler%2B-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

ML CUDA Detective is a single-file Python script that audits your entire machine learning software stack in one shot. It tells you exactly what is installed, what version you have, whether it can actually reach the GPU, and whether the packages that require specific hardware features (Tensor Cores, FlashAttention, 2:4 structured sparsity, FP8) will work on your particular GPU generation.

The script requires **no external dependencies** beyond the Python standard library. Every package it checks is optional — a missing package shows as ❌ rather than crashing the script. It is safe to run in any environment.

---

## Why This Tool Exists

Setting up a working GPU ML environment is surprisingly difficult. Installing PyTorch successfully does not mean TensorFlow sees the GPU. Having flash-attn in your `pip list` does not mean it will load on a Pascal-era GTX 1080. RAPIDS requires CUDA Compute Capability 7.0 or higher and will silently fail to import on older hardware. Version mismatches between CUDA Toolkit, cuDNN, and framework wheels are one of the most common sources of mysterious failures in ML pipelines.

This tool was built to answer the question: *"Does my ML environment actually work, end to end, on this specific machine?"* — not just "is the package installed?"

---

## Features

### ⚡ GPU-Accelerated Package Detection
Checks every major CUDA-accelerated package and performs live GPU smoke tests:

- **PyTorch CUDA** — import check + live matmul on GPU with timing
- **TensorFlow GPU** — import check + GPU device enumeration + conv2d smoke test
- **JAX CUDA** — import check + GPU device detection + live computation
- **CuPy** — import check + GPU array operations
- **RAPIDS** — cuDF, cuML, cuGraph, cuSpatial, cuxfilter (requires CC 7.0+)
- **OpenAI Triton** — import check + CUDA backend verification
- **CUDA Python** — low-level CUDA Python bindings
- **cuDNN Python bindings** — cuDNN version and availability
- **ONNX Runtime** — CUDA EP and TensorRT EP detection
- **TensorRT / torch-tensorrt** — version and availability
- **llama-cpp-python** — detects CUDA build vs. CPU-only build
- **ctranslate2** — CUDA device detection
- **flash-attn** — version detection with CC gate reporting (requires CC 7.5+; FA-3 requires CC 9.0)
- **xformers** — memory-efficient attention library
- **bitsandbytes** — quantization library CUDA backend check
- **DeepSpeed** — distributed training library GPU detection

### 🖥️ CPU-Only Package Inventory
Organized by domain for readability:

| Category | Packages |
|---|---|
| Numeric / Data Science | numpy, scipy, pandas, polars, pyarrow, statsmodels, sympy |
| Data Wrangling | scikit-learn, imbalanced-learn, feature-engine, category-encoders |
| Computer Vision | Pillow, opencv-python, torchvision, albumentations, timm |
| NLP & Text | transformers, tokenizers, sentencepiece, spacy, nltk, gensim |
| Visualization | matplotlib, seaborn, plotly, bokeh, altair, dash |
| Audio / Speech | torchaudio, librosa, soundfile, speechbrain, pyaudio |
| TF Ecosystem | keras, tensorflow-datasets, tensorflow-hub, tensorflow-text |
| Datasets & Eval | datasets, evaluate, sacrebleu, rouge-score |
| Dev / Build | build, setuptools, wheel, twine, cython, pybind11 |

### 📓 Notebooks & Interactive Computing
- Jupyter core: JupyterLab, Notebook, IPython, nbconvert, nbformat
- Jupyter AI & LLM integration: jupyter-ai, jupyter-copilot
- Extensions & widgets: ipywidgets, jupyterlab-git, voila, rise
- Alternative notebooks: marimo, pluto (Julia bridge)
- Notebook testing: nbmake, testbook, nbval

### 🔬 MLOps — Experiment Tracking, Pipelines & Model Serving
- **Experiment tracking**: MLflow, Weights & Biases, DVC, ClearML, Comet ML, Neptune, Aim, TensorBoard
- **Pipeline orchestration**: ZenML, Metaflow, Kedro, Prefect, Airflow, Kubeflow, Flyte, Dagster
- **Model serving**: BentoML, Ray Serve, FastAPI, Gradio, Streamlit, ONNX Runtime, Triton Inference Server

### 🤖 Agentic AI — Frameworks, LLM Clients & Vector Stores
- **Orchestration frameworks**: LangChain, LangGraph, LlamaIndex, AutoGen, Semantic Kernel, CrewAI, Strands Agents, Pydantic AI, Smolagents, Llama Stack, Haystack
- **LLM provider clients**: OpenAI, Anthropic, Google Generative AI, Mistral, Cohere, Groq, Together AI, LiteLLM, Ollama
- **Memory, RAG & vector stores**: ChromaDB, Qdrant, Weaviate, Pinecone, FAISS-GPU, Milvus, LanceDB, Mem0

### 🏥 Environment Health Summary
- OS version and architecture
- Python version and distribution
- CUDA Toolkit version
- cuDNN version
- NVIDIA driver version
- GPU hardware inventory (name, VRAM, Compute Capability, SM count, Tensor Core count) via nvidia-smi and PyTorch
- pip dependency conflict check

---

## GPU Architecture Coverage

The script explicitly identifies your GPU's CUDA Compute Capability (CC) and uses it to explain why certain packages will or will not work:

| Architecture | CC | GPUs | Notes |
|---|---|---|---|
| Kepler | 3.x | GTX 700, K40, K80 | CUDA 12.x compiler support dropped |
| Maxwell | 5.x | GTX 900, Quadro M-series | Feature-frozen in CUDA 12.8 (Jan 2025) |
| Pascal | 6.x | GTX 1060/1070/1080, P100 | Feature-frozen in CUDA 12.8; no Tensor Cores; RAPIDS unsupported |
| Volta | 7.0 | V100, Titan V | First Tensor Cores; NGC dropped in release 25.01 (2025) |
| Turing | 7.5 | RTX 2080, T4 | FlashAttention-2 minimum; no BF16/FP8/sparsity |
| Ampere | 8.0/8.6 | A100, RTX 3090 | Full modern feature support; 2:4 sparsity gated at CC 8.0 |
| Ada Lovelace | 8.9 | RTX 4090, L40S | Ada-specific optimizations |
| Hopper | 9.0 | H100, H200 | FP8 / Transformer Engine; FlashAttention-3 |
| Blackwell | 10.0 | B200, GB200 | Latest generation; NGC 25.01+ |

---

## Benchmark / Smoke Test System

The script performs live GPU smoke tests — not just import checks. The `BENCHMARK_DEPTH` configuration variable controls thoroughness:

| Depth | Time | Description |
|---|---|---|
| `1` — fast | ~15–25s | 512×512 matrix, 1 pass, float32 only. Quick sanity check. |
| `2` — medium | ~45–75s | 1024×1024, 3 passes averaged, float32 + float16, 3-layer MLP autograd. |
| `3` — thorough | ~2–4 min | 2048×2048, 10 passes with mean + stddev, float32/float16/bfloat16, 5-layer MLP, 10 SGD steps, custom CuPy kernel. |
| `4` — memory | ~3–6 min | All depth-3 tests plus progressive memory pressure tests per framework, reporting peak usable GPU RAM in 256 MB steps. |

---

## Configuration

All settings are at the top of the script — one place, clearly documented:

```python
# ── Release date mode ──────────────────────────────────────────
FETCH_RELEASE_DATES: int = 0
# 0 — no PyPI calls; instant startup (default)
# 1 — fetch dates for installed packages; shows freshness indicator
# 2 — fetch dates for all packages including uninstalled (~150 requests)

# ── Report output ──────────────────────────────────────────────
SAVE_REPORT: bool = True          # mirror output to timestamped .txt file
REPORT_DIR:  str  = "~/ml-reports"  # destination directory (auto-created)
REPORT_WIDTH: int = 88            # column width for word-wrapped output

# ── Benchmark depth ────────────────────────────────────────────
BENCHMARK_DEPTH: int = 1          # 1 = fast | 2 = medium | 3 = thorough | 4 = memory

# ── Optional report sections ───────────────────────────────────
PRINT_NOTES:       int = 1        # 0 = suppress footnotes | 1 = print
PRINT_BIBLIOGRAPHY: int = 1       # 0 = suppress bibliography | 1 = print
```

### `PRINT_NOTES`
Prints explanatory footnotes at the end of the report covering package manager detection scope, pip vs. pipx vs. conda vs. system package visibility, hardware-gated feature requirements (CC thresholds), FlashAttention version gates, 2:4 sparsity hardware requirements, and multi-environment audit strategy.

### `PRINT_BIBLIOGRAPHY`
Prints a 65-entry IEEE-format annotated bibliography covering every major technical topic in the report. Each entry includes a verified URL and a 2–4 sentence annotation explaining its relevance to GPU ML infrastructure auditing. Categories include CUDA architecture, NVIDIA GPU whitepapers (Volta through Blackwell), Tensor Cores, FlashAttention, quantization, distributed training, LLM serving, and more.

Set both to `0` for a compact operational report. Set both to `1` when sharing with a wider audience or submitting as part of an infrastructure review.

---

## Requirements

- **Python**: 3.10 or higher
- **OS**: Linux or Windows (WSL2 supported)
- **GPU**: NVIDIA GPU recommended (Kepler CC 3.x through Blackwell CC 10.x)
  - The script will run without a GPU and report CPU-only status for GPU packages
- **Dependencies**: None — standard library only

---

## Installation

No installation required. Clone or download the single file and run it:

```bash
# Option 1 — clone the repository
git clone https://github.com/cms-001/ml_cuda_detective.git
cd ml_cuda_detective
python ml_cuda_detective.py

# Option 2 — download the script directly
wget https://raw.githubusercontent.com/cms-001/ml_cuda_detective/main/ml_cuda_detective.py
python ml_cuda_detective.py

# Option 3 — run directly without saving
curl -s https://raw.githubusercontent.com/cms-001/ml_cuda_detective/main/ml_cuda_detective.py | python3
```

---

## Usage

```bash
# Default run — fast benchmark depth, no PyPI date fetching
python ml_cuda_detective.py

# Suppress notes and bibliography for a compact report
# (edit PRINT_NOTES = 0 and PRINT_BIBLIOGRAPHY = 0 at the top of the file)

# Save report to a custom directory
# (edit REPORT_DIR = "/your/path" at the top of the file)

# Run with thorough benchmarks
# (edit BENCHMARK_DEPTH = 3 at the top of the file)
```

All configuration is done by editing the six variables at the top of the script — there are no command-line arguments.

---

## Sample Output

```
================================================================
🕵️‍♂️  ML CUDA Detective — Linux/Windows + NVIDIA GPU
================================================================
  Timestamp        2026-03-04 18:00:00 UTC
  Hostname         my-gpu-server
  OS               Ubuntu 22.04.3 LTS  (x86_64)
  Python           3.11.7  (CPython)
  Benchmark depth  1  (fast ~15-25s)
  Report width     88 columns
  Save report      ~/ml-reports/ml_cuda_detective_20260304_180000.txt
  Fetch dates      off

================================================================
🏥  Environment Health
================================================================
  CUDA Toolkit     12.3.0
  cuDNN            8.9.7
  NVIDIA Driver    545.29.06
  GPU 0            NVIDIA H100 SXM5 80GB
                   CC 9.0 · Hopper · 80 GB VRAM · 132 SMs
                   528 Tensor Core units (4th-gen)

================================================================
⚡  CUDA / GPU Accelerated
================================================================
  ✅ torch                   2.2.1+cu121    GPU: NVIDIA H100 SXM5 80GB
      matmul 512×512       0.42 ms  (float32)
  ✅ tensorflow              2.15.0         GPU: /device:GPU:0
      conv2d smoke test    1.83 ms
  ✅ jax                     0.4.23         GPU: TFRT_GPU_0
  ✅ cupy-cuda12x            12.3.0
  ✅ cudf                    23.12.0
  ✅ flash-attn              2.5.2          (CC 9.0 ✅ ≥ 7.5 required)
  ✅ bitsandbytes            0.42.0         CUDA available
  ...
```

*(Actual output varies by environment and installed packages.)*

---

## Who This Is For

### Cloud / Managed ML Environments
If you are running on a modern cloud GPU instance (AWS SageMaker, Google Vertex AI, Azure ML, CoreWeave, Lambda Labs, Crusoe, Nebius, RunPod, Paperspace, or similar), you will typically see very few ❌ results for packages you have actually installed. These platforms provision current-generation NVIDIA hardware (A100, H100, L40S, or newer) with pre-validated CUDA stacks from NVIDIA's NGC container catalog. Things generally just work.

### On-Premises Servers, Workstations, Desktops, and Laptops
This tool is most valuable for on-premises hardware, especially older GPU generations. Modern ML packages impose hard hardware gates via CUDA Compute Capability:

- **RAPIDS** requires CC 7.0+ — will not run on Pascal (GTX 1080, P100) or older
- **FlashAttention-2** requires CC 7.5+ — will not run on Volta V100 or older
- **FlashAttention-3** requires CC 9.0 — Hopper (H100) only
- **2:4 structured sparsity** requires CC 8.0+ — Ampere and newer only
- **FP8 / Transformer Engine** requires CC 9.0 — Hopper only
- **BF16 hardware support** requires CC 8.0+ — not available on Turing or older

The script identifies your GPU's CC and annotates each of these gates clearly so you know whether a ❌ means "not installed" or "hardware cannot support this."

---

## Understanding the Output Symbols

| Symbol | Meaning |
|---|---|
| ✅ | Installed and importable |
| ❌ | Not installed or failed to import |
| ⚠️ | Installed but outdated, or installed but with caveats |
| ℹ️ | Informational note |
| 🚫 | Hardware gate — your GPU's CC does not meet the minimum requirement |

---

## Frequently Asked Questions

**Q: Do I need to install anything before running this?**
No. The script uses only Python standard library modules. Run it as-is on any Python 3.10+ installation.

**Q: Will it crash if I don't have a GPU?**
No. Every GPU check gracefully degrades to a ❌ or an informational note. The CPU-only package inventory sections will still run completely.

**Q: Why does a package show ❌ even though I installed it?**
Several reasons are possible: the package was installed in a different virtual environment, it was installed by conda or pipx (not visible to pip-based detection), it was installed as a system package outside the active Python environment, or it installed successfully but cannot be imported (common for GPU packages on incompatible hardware). The Notes section (enabled by `PRINT_NOTES = 1`) covers all of these cases in detail.

**Q: My GPU is a GTX 1080. Why does RAPIDS show ❌?**
The GTX 1080 is Pascal architecture (CC 6.1). RAPIDS requires CC 7.0 or higher. This is a hard hardware gate — no software change will make RAPIDS work on Pascal. The script reports this explicitly.

**Q: Can I run this on macOS / Apple Silicon?**
The Linux/Windows version of this script targets NVIDIA CUDA hardware specifically. For Apple Silicon (Metal GPU) environments, see the companion script `ml_metal_detective.py`.

**Q: The report is very long. Can I make it shorter?**
Set `PRINT_NOTES = 0` and `PRINT_BIBLIOGRAPHY = 0` in the configuration section at the top of the file. This removes the footnotes and the annotated bibliography, leaving only the package inventory and environment summary.

**Q: What does `FETCH_RELEASE_DATES = 1` do?**
It makes ~150 parallel requests to PyPI to fetch the release dates of every installed package, then compares them to the latest available version. Each installed package line will show its installation date, the date of the latest release, and a freshness indicator (✅ up-to-date or ⚠️ behind). Mode `2` additionally fetches dates for packages that are not installed. Leave it at `0` for instant startup.

---

## Project Background

ML CUDA Detective began as a macOS / Apple Silicon environment auditor (`ml_metal_detective.py`) and was subsequently ported to Linux and Windows for NVIDIA CUDA environments. The CUDA version covers the complete NVIDIA ML software stack from bare CUDA/cuDNN through the major deep learning frameworks, the Agentic AI ecosystem, and MLOps tooling.

The script has been validated against NVIDIA GPU generations from Pascal through Hopper across Ubuntu 20.04, Ubuntu 22.04, and Windows 11 with WSL2.

---

## Contributing

Issues and pull requests are welcome. If you encounter a package that should be included in the inventory, or a detection logic bug, please open an issue with:

- Your OS and Python version
- Your NVIDIA driver and CUDA Toolkit version
- Your GPU model and Compute Capability
- The relevant section of the script output

---

## License

Copyright (c) 2026 Christopher Swenson. All rights reserved.

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Related Resources

- [NVIDIA NGC Container Catalog](https://catalog.ngc.nvidia.com) — pre-validated PyTorch/TensorFlow/JAX containers
- [PyTorch Get Started](https://pytorch.org/get-started/locally/) — official CUDA wheel selector
- [CUDA Compute Capability Table](https://developer.nvidia.com/cuda/gpus) — find your GPU's CC
- [RAPIDS Installation Guide](https://docs.rapids.ai/install) — CC 7.0+ requirement documentation
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) — hardware requirement details
- [NVIDIA Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) — official driver/CUDA/framework compatibility table
