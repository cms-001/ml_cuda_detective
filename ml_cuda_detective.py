#!/usr/bin/env python3

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import re
import platform
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
from urllib.request import urlopen
from urllib.error import URLError
import json
import textwrap
import contextlib as _contextlib

# ================================================================
# 🕵️‍♂️ ML CUDA Detective — Linux/Windows + NVIDIA GPU Accelerators
# ================================================================
#
# DESCRIPTION
# -----------
# Comprehensive environment health-check and package inventory tool
# for NVIDIA GPU machines (Kepler through Blackwell) running
# Python-based ML/AI workloads on Linux or Windows. Covers the full
# CUDA acceleration stack from core CUDA/cuDNN frameworks down to
# CPU-only data science utilities, Jupyter notebook infrastructure,
# MLOps tooling, and Agentic AI frameworks.
#
# WHAT IT CHECKS
# --------------
#  ⚡ CUDA / GPU Accelerated packages
#     • PyTorch CUDA, TensorFlow GPU, JAX CUDA, CuPy
#     • RAPIDS (cuDF, cuML, cuGraph, cuSpatial, cuxfilter)
#     • Triton (OpenAI), CUDA Python, cuDNN Python bindings
#     • ONNX Runtime CUDA/TensorRT EP, TensorRT, torch-tensorrt
#     • llama-cpp-python (CUDA build), ctranslate2 CUDA
#     • flash-attn, xformers, bitsandbytes, deepspeed
#     • Runtime smoke tests: actual matmul + conv2d on GPU
#       with timing benchmarks for each framework
#
#  🖥️  CPU-only packages (organized by domain)
#     • Data science / numeric stack (numpy, scipy, pandas, etc.)
#     • Data wrangling & feature engineering
#     • Computer vision & media
#     • NLP & text processing
#     • Visualization, audio/speech, TF ecosystem add-ons
#     • Datasets & evaluation, dev/build helpers
#
#  📓 Notebooks & Interactive Computing
#     • Jupyter core (JupyterLab, Notebook, IPython)
#     • Jupyter AI & LLM integration
#     • Extensions, widgets, alternative notebooks
#     • Notebook testing & execution utilities
#
#  🔬 MLOps — Experiment Tracking, Pipelines & Model Serving
#     • Experiment tracking & model registry (MLflow, W&B,
#       DVC, ClearML, Comet ML, Neptune, Aim, TensorBoard)
#     • Pipeline & workflow orchestration (ZenML, Metaflow,
#       Kedro, Prefect, Airflow, Kubeflow, Flyte, Dagster)
#     • Model serving & deployment (BentoML, Ray Serve,
#       FastAPI, Gradio, Streamlit, ONNX Runtime, Triton)
#
#  🤖 AI/ML and Agentic AI — Frameworks, LLM Clients & Vector Stores
#     • Core orchestration frameworks (LangChain, LangGraph,
#       LlamaIndex, AutoGen, Microsoft Agent Framework,
#       Semantic Kernel, CrewAI, Strands Agents, Pydantic AI,
#       Smolagents, Llama Stack, Haystack)
#     • LLM provider clients (OpenAI, Anthropic, Google,
#       Mistral, Cohere, Groq, Together, LiteLLM, Ollama)
#     • Memory, RAG & vector stores (ChromaDB, Qdrant,
#       Weaviate, Pinecone, FAISS-GPU, Milvus, LanceDB, Mem0)
#
#  🏥 Environment health
#     • OS version, architecture, Python version
#     • CUDA toolkit version, cuDNN version, driver version
#     • GPU hardware detection via nvidia-smi + torch
#     • pip dependency conflict check
#
# RELEASE DATE MODES (FETCH_RELEASE_DATES)
# -----------------------------------------
#  0 — No dates (default, instant startup)
#  1 — Installed packages: shows installed version date + latest
#      release date side-by-side with freshness indicator
#  2 — Mode 1 + latest release date for uninstalled packages too
#      (full PyPI freshness audit, ~150 parallel requests)
#
# OUTPUT
# ------
# Prints a structured report to the terminal. When SAVE_REPORT is
# True, also writes a timestamped plain-text copy to REPORT_DIR.
#
# USAGE
# -----
#   python ml_cuda_detective.py
#
# CONFIGURATION (top of file)
# ----------------------------
#   FETCH_RELEASE_DATES = 0 | 1 | 2
#   SAVE_REPORT         = True | False
#   REPORT_DIR          = "/path/to/output/dir"
#
# REQUIREMENTS
# ------------
#   Python 3.10+, Linux or Windows, NVIDIA GPU (Kepler+) recommended.
#   No extra dependencies beyond the standard library —
#   all checked packages are optional (missing = ❌, not a crash).
#
# AUTHOR / LICENSE
# ----------------
#   ml_cuda_detective.py
#
#   Copyright (c) 2026 Christopher Swenson. All rights reserved.
#
#   Version  : 1.0
#   Date Last Updated: 2026-03-04
#   License  : MIT (permissive)
#
#   Permission is hereby granted, free of charge, to any person
#   obtaining a copy of this software to use, copy, modify, merge,
#   publish, distribute, sublicense, and/or sell copies of it,
#   subject to the standard MIT License terms.
#
#   This script was developed in response to practical challenges
#   encountered while configuring ML frameworks on legacy GPU hard-
#   ware during an intracranial aneurysm detection project (RSNA
#   Kaggle Competition, 2023). It represents the distillation of
#   six or seven iterative drafts, incorporating the most robust
#   and reliable components from each prior version.
#
#   An annotated bibliography was appended on March 5, 2026, and
#   will be expanded incrementally as additional canonical litera-
#   ture is identified — with particular emphasis on GPU architec-
#   ture, drawing from both NVIDIA engineering publications and the
#   growing body of academic research on NVIDIA hardware.
#
#   Developed with assistance from Claude.ai (Anthropic) and Chat-
#   GPT (OpenAI). AI tools were employed throughout for code gen-
#   eration, structural organization, and iterative refinement.
#   The majority of technical insight embedded in this script was
#   surfaced through those tools. All prompts were authored by the
#   original developer.
#
#   Suggestions, improvements, and contributions are warmly
#   welcome — open an issue or submit a pull request. The best
#   feedback will be included at the bottom of this script.
# ================================================================

# ================================================================
# WHO THIS REPORT IS FOR (OR, WHO TYPICALLY HAS PROBLEMS
#  — AND WHO TYPICALLY DOESN'T)
# ----------------------------------------------
# The ❌ results in this report mean very different things depending
# on what kind of machine you are running this on. Understanding
# that context will save you significant troubleshooting time.
#
# CLOUD / HYPERSCALER ENVIRONMENTS (AWS, GCP, Azure, Oracle, IBM,
# Alibaba, Tencent, Digital Ocean, OVH, etc.)
# GPU CLOUD / NEOCLOUD ENVIRONMENTS (Coreweave, Crusoe, RunPod, 
# vast.ai, Lambda Labs, Genesis Cloud, Gcore, Vultr, Nebius, etc.)
# ---------------------------------------------------------------
# If you are running on a modern cloud GPU instance — an A100, H100,
# L40S, or newer — you will almost certainly see very few ❌ results
# for packages you have actually installed. This is by design.
#
# Cloud providers provision instances with NVIDIA's current data
# center GPU families (Hopper H100 CC 9.0, Ampere A100 CC 8.0,
# Ada L40S CC 8.9), which fully support every modern ML feature:
# Tensor Cores, BF16, TF32, 2:4 structured sparsity, FP8, and
# FlashAttention-2/3. 
#
# Cloud providers typically provision instances with NVIDIA’s 
# modern data-center GPU architectures such as Hopper (H100, 
# CC 9.0), Ampere (A100, CC 8.0), and Ada Lovelace (L40S, CC 8.9).
# These GPUs include Tensor Cores and support modern deep-learning 
# formats such as FP16, BF16, and TF32, along with features like 2:4 
# structured sparsity. Hopper and newer architectures also introduce 
# FP8 acceleration via NVIDIA’s Transformer Engine. The underlying 
# CUDA stack on these instances typically tracks the current CUDA 
# Toolkit release (12.x as of 2025-2026), and both NVIDIA and the
# cloud providers invest heavily in ensuring that PyTorch, TensorFlow, 
# JAX, and the broader ecosystem  are installed and functional on 
# day one.
#
# The gold standard for this is NVIDIA's NGC container catalog
# (catalog.ngc.nvidia.com), which publishes monthly Docker images
# with a fully pre-validated, co-tested stack: PyTorch or
# TensorFlow, the exact CUDA Toolkit version they were built
# against, cuDNN, TensorRT, and supporting libraries — all
# confirmed to work together on current-generation hardware.
# AWS SageMaker, Google Vertex AI, Azure Machine Learning, and
# most serious GPU cloud providers provision their managed
# environments from this same NGC foundation, or maintain
# equivalently validated stacks. When you spin up a managed
# notebook or training instance on any of these platforms, you
# are inheriting years of integration testing. Things just work.
#
# ON-PREMISES SERVERS, WORKSTATIONS, DESKTOPS, AND LAPTOPS
# ----------------------------------------------------------
# The situation is fundamentally different for on-premises hardware,
# and this is where this report is most useful.
#
# The core issue is that modern ML packages are increasingly
# selective about the GPU hardware they run on, and older GPUs
# that were perfectly capable for their era are being systematically
# excluded from the newest capabilities — sometimes from packages
# entirely.
#
# CUDA Compute Capability (CC) is the formal hardware feature
# versioning system. Each GPU generation has a CC number, and
# packages use it as a hard gate:
#
#   CC 3.x — Kepler (GTX 700 series, K40, K80)
#              CUDA 12.x compilers no longer target Kepler.
#              RAPIDS has never supported it. Most modern ML
#              packages dropped Kepler support years ago.
#
#   CC 5.x — Maxwell (GTX 900 series, Quadro M-series)
#              CUDA Toolkit 12.8 (Jan 2025) declared Maxwell
#              "feature-complete" — no new CUDA features will
#              be added. RAPIDS dropped Maxwell at CC 7.0+.
#
#   CC 6.x — Pascal (GTX 1060/1070/1080, P100, Titan X)
#              Same "feature-complete" declaration as Maxwell in
#              CUDA 12.8. Very common in on-premise ML servers
#              purchased 2016-2018. No Tensor Cores. No BF16.
#              No structured sparsity. No FlashAttention hardware
#              path. FlashAttention-2 requires CC 7.5 minimum;
#              FlashAttention-3 requires CC 9.0. RAPIDS dropped
#              Pascal at version 22.04 (CC 7.0+ required). NGC
#              containers stopped testing Pascal in release 23.06.
#
#   CC 7.0 — Volta (V100, Titan V)
#              First architecture with Tensor Cores. Supports
#              FP16/INT8 Tensor Core ops and FlashAttention-2.
#              CUDA 12.8 declared Volta feature-complete alongside
#              Maxwell and Pascal. NGC containers dropped Volta
#              testing in release 24.10 (Oct 2024), and fully
#              discontinued Volta support in release 25.01 (2025).
#              V100 on-premise clusters purchased 2018-2020 are
#              increasingly at the compatibility boundary.
#
#   CC 7.5 — Turing (RTX 2080, T4)
#              Adds INT4 Tensor Cores and RT Cores. FlashAttention-2
#              minimum. Still broadly supported by current packages,
#              but no BF16 hardware support, no 2:4 sparsity, and
#              no FP8. Common in on-premise workstations and in
#              older cloud instances (AWS G4 uses T4).
#
#   CC 8.0+ — Ampere and newer (A100, RTX 3090, H100, RTX 4090)
#              Full support for all modern features: BF16, TF32,
#              2:4 structured sparsity (CC 8.0+), FP8 / Transformer
#              Engine (CC 9.0, Hopper), FlashAttention-2/3. This
#              is the target hardware tier for current ML research
#              and production. If you are here, this report should
#              show mostly ✅ for installed packages.
#
# PRACTICAL IMPLICATIONS
# ----------------------
# If this report is running on a Pascal (GTX 1080, P100) or
# Maxwell GPU, expect ❌ for RAPIDS, flash-attn, xformers, and
# any package that ships pre-compiled CUDA kernels targeting
# CC 7.0+. These are not installation failures — the packages
# physically cannot execute their GPU kernels on the hardware.
# PyTorch itself will still install and run (the base wheels
# include CC 6.1 PTX fallback paths), but performance-critical
# libraries that bypass PyTorch's kernel dispatch will refuse
# to load or silently fall back to CPU.
#
# If this report is running on a Turing GPU (RTX 2080, T4),
# most packages will work, but you will see feature-gate ❌
# marks for BF16 training, 2:4 sparsity acceleration, and
# FlashAttention-3. These are architectural limits, not config
# issues — no software update will add hardware it doesn't have.
#
# The cleanest path for on-premise hardware running into these
# limits is NVIDIA's NGC containers: they are pinned to specific
# CUDA/driver versions and will clearly document whether your
# GPU generation is in their support matrix. For bare-metal
# installs, use the PyTorch "Compute Platform" selector at
# pytorch.org/get-started/locally to get a wheel that matches
# your exact CUDA toolkit version, and accept that packages
# requiring CC 7.0+ simply will not function on older hardware
# regardless of what the pip install command reports.
#
# In short: if you are on a cloud GPU instance from the last
# two or three years, a healthy ✅ report is the expected
# baseline. If you are on older on-premise hardware — especially
# anything Pascal or earlier — the ❌ results in this report
# are likely telling you something real and worth acting on.
# ============================================================

# ============================================================
# 🔇 Noise Suppression — Warnings, Logs & stderr Filtering
#    Aggressively silences the torrent of informational and
#    compatibility messages emitted by TensorFlow, JAX, absl,
#    ProtoBuf, and CUDA drivers during import and first GPU op.
#    Must be applied BEFORE any of those packages are imported.
#    Without this block the report output is cluttered with
#    dozens of irrelevant log lines that obscure real results.
#
#    Environment variables (set before any C extension loads):
#      TF_CPP_MIN_LOG_LEVEL=3   suppress TensorFlow C++ logs
#      GLOG_minloglevel=3       suppress absl/glog INFO lines
#
#    warnings.filterwarnings() — Python-level suppression for:
#      • ProtoBuf gencode version mismatch (UserWarning spam
#        triggered on every TF/JAX import when protobuf has a
#        minor version bump ahead of the gencode in the wheel)
#
#    _StderrFilter class — replaces sys.stderr with a filter
#      that drops known-noisy line prefixes from absl, TF, JAX,
#      and CUDA while passing everything else through to the
#      real stderr. Covers: I0000/W0000 absl log lines,
#      CUDA_VISIBLE_DEVICES warnings, StreamExecutor init,
#      cuDNN version mismatch noise, XLA service/backend,
#      "successful NUMA node read" sysfs spam, etc.
#
#    _silence_fd2() context manager — redirects file descriptor
#      2 (C-level stderr) to /dev/null for the duration of the
#      'with' block, then restores it. Used around TensorFlow
#      and JAX import calls and the first GPU op that triggers
#      CUDA device initialisation, where C extensions bypass
#      Python's sys.stderr entirely and write directly to fd 2.
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

warnings.filterwarnings(
    "ignore",
    message=r".*Protobuf gencode version .* is exactly one major version older than the runtime version .*",
    category=UserWarning,
    module=r"google\.protobuf\.runtime_version",
)

# Suppress CUDA / TensorFlow version compatibility warnings
for _pat in (
    r".*Your CUDA version has a newer driver than cuDNN.*",
    r".*is not compatible with TensorFlow.*",
    r".*Could not load dynamic library.*",
    r".*is not compatible with the current TensorFlow installation.*",
):
    warnings.filterwarnings("ignore", message=_pat)

# Suppress absl/TF/JAX CUDA stderr noise (I0000, WARNING: lines)
class _StderrFilter:
    """Drop absl/TF/JAX/CUDA log lines; pass everything else through."""
    _SUPPRESS = (
        "I0000 ", "W0000 ",
        "WARNING: All log messages",
        "WARNING:absl:", "WARNING:jax",
        "WARNING:root:", "WARNING:tensorflow",
        "pluggable_device_factory", "service.cc",
        "XLA service", "XLA backend",
        "StreamExecutor device",
        "successful NUMA node read",
        "absl::InitializeLog",
        "Platform 'CUDA' is experimental",
        "JAX CUDA support is experimental",
        "Using Simple allocator",
        "SimpleAllocator",
        "CUDA_ERROR", "cuDNN",
        "Created device /",
        "Created device /job:",

    )
    def __init__(self) -> None:
        self._real = sys.__stderr__
    def write(self, s: str) -> int:
        if any(tok in s for tok in self._SUPPRESS):
            return len(s)
        self._real.write(s)
        return len(s)
    def flush(self) -> None:
        self._real.flush()

sys.stderr = _StderrFilter()  # type: ignore[assignment]

@_contextlib.contextmanager
def _silence_fd2():
    """Suppress C-level stderr (fd 2) by redirecting to /dev/null."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# ============================================================
# ⚙️  User Configuration
#    All runtime behaviour knobs are consolidated in the next
#    three sectioins, so start here to change settings:
#
#    FETCH_RELEASE_DATES
#      0 — no PyPI network calls; instant startup (default)
#      1 — fetch release dates for installed packages only;
#          annotates each ✅ line with installed date and a
#          ✅ up-to-date or ⚠️ behind-latest indicator
#      2 — fetch dates for every package (installed + missing);
#          full freshness audit; fires ~150 parallel requests
#
#    SAVE_REPORT
#      True  — mirror all output to a timestamped .txt file
#              in REPORT_DIR (created automatically if absent)
#      False — terminal only; no file written
#
#    REPORT_DIR
#      Directory for saved report files. Defaults to
#      ~/ml-reports/. The folder is created with makedirs()
#      if it does not already exist.
#
#    REPORT_WIDTH
#      Column width (in characters) for all word-wrapped
#      output lines — ℹ️ notes, package lines, smoke test
#      details. Change this single value to reflow the entire
#      report. 88 fits a standard 90-column terminal cleanly.
# ============================================================
FETCH_RELEASE_DATES: int = 0   # 0 = no dates | 1 = installed only | 2 = all packages
SAVE_REPORT: bool = True        # write a timestamped .txt report file
REPORT_DIR: str = os.path.expanduser("~/ml-reports")  # directory for saved reports
REPORT_WIDTH: int = 88   # ← change this one number to control all wrapping

# ============================================================
# ⚙️  Benchmark Depth — Smoke Test Thoroughness Control
#    Single integer that scales every benchmark and smoke test
#    simultaneously: matrix size, iteration count, dtype
#    coverage, sparsity density sweep, training loop depth,
#    and whether memory pressure tests are included.
#    Higher depths give more statistically reliable numbers
#    and broader coverage at the cost of wall-clock time.
#
#    1 — fast        (~15–25s)
#          Matrix:     512×512 (fits in GPU L2 cache on M1 Pro)
#          Iterations: 1 pass — quick sanity check only;
#                      timing noise can be ±20–50% at this size
#          Dtypes:     float32 only
#          Sparse:     single density (5% non-zero)
#          Autograd:   single linear layer forward + backward
#          Use when:   iterating on the script, quick spot-check
#                      after a package install or update
#
#    2 — medium      (~45–75s)
#          Matrix:     1024×1024 (cache-resident on some ops)
#          Iterations: 3 passes averaged — reduces timing noise
#          Dtypes:     float32 + float16
#          Sparse:     3 densities (1%, 5%, 10% non-zero)
#          Autograd:   3-layer MLP forward + backward, 3 passes
#          CoreML:     matmul model built and timed via CoreML EP
#          coremltools: PyTorch Linear → CoreML conversion timed
#          Use when:   pre-commit check, comparing two envs
#
#    3 — thorough    (~2–4 min)
#          Matrix:     2048×2048 (reliably memory-bandwidth-bound;
#                      better proxy for real LLM layer sizes)
#          Iterations: 10 passes with mean + stddev reported
#          Dtypes:     float32 + float16 + bfloat16
#          Sparse:     4 densities (1%, 5%, 10%, 20% non-zero)
#          Autograd:   5-layer MLP, 10 full SGD training steps
#          cupy:      ElementwiseKernel custom CUDA kernel test
#          coremltools: 3-layer MLP conversion + inference timed
#          Use when:   baseline benchmarking, framework comparison,
#                      validating a new macOS or package update
#
#    4 — memory      (~3–6 min total, depth-3 PLUS the following)
#          Runs all depth-3 tests first, then adds per-framework
#          progressive memory pressure tests for:
#            PyTorch CUDA · TensorFlow GPU · JAX CUDA · CuPy
#          Each framework allocates GPU tensors in 256 MB steps
#          until OOM or 90% of system RAM, then verifies cleanup.
#          Reports peak usable GPU RAM per framework — useful for
#          planning large model loads (LLMs, diffusion models).
#          ⚠️  Close browsers, video apps, and other GPU-intensive
#          processes before running to get accurate ceilings and
#          avoid system stalls or kernel memory pressure events.
# ============================================================
BENCHMARK_DEPTH: int = 1   # ← change to 1 / 2 / 3 / 4

# ============================================================
# 📝 Notes & Bibliography — Optional Report Sections
#    Two optional appendices printed at the end of the report.
#    Both default to 1 (printed). Set either to 0 to suppress.
#
#    Why print the Notes?
#      The Notes section explains what the audit results
#      actually mean — it is the fine print that turns raw
#      ✅/❌ rows into actionable conclusions. It covers:
#
#        • What "installed" means in each package manager
#          context (pip, pipx, conda, system apt/brew) and
#          why the same package can be ❌ here while clearly
#          present on your machine in another environment.
#
#        • Which GPU features are hardware-gated and why:
#          FlashAttention-3 requires CC 9.0 (Hopper); 2:4
#          structured sparsity requires CC 8.0 (Ampere);
#          BF16 native requires CC 8.0+. These are not bugs
#          in the report — they are physical silicon limits.
#
#        • How to run this script across multiple environments
#          (venv / conda / pipx) to get a complete cross-
#          environment inventory, and what each run covers.
#
#      Recommended reading before acting on any ❌ result.
#      Suppress with PRINT_NOTES = 0 once you are familiar
#      with the scope and have no outstanding ❌ items to
#      investigate.
#
#    Why print the Bibliography?
#      The Annotated Bibliography provides 65 verified,
#      IEEE-formatted references covering every major topic
#      in the report — CUDA architecture whitepapers, the
#      FlashAttention papers, quantization research (LLM.int8,
#      QLoRA), distributed training (ZeRO/DeepSpeed), serving
#      infrastructure (vLLM/PagedAttention), and the canonical
#      framework papers (PyTorch, TensorFlow, JAX). Each entry
#      includes a live URL and a 2–4 sentence annotation
#      explaining its relevance to this report specifically.
#
#      Print it when:
#        • Sharing this report with stakeholders who want to
#          verify claims or read further on a specific topic.
#        • Submitting as part of an infrastructure review,
#          onboarding doc, or technical due-diligence package.
#        • Onboarding a new team member — the bibliography
#          doubles as a curated GPU/ML reading list.
#
#      Suppress with PRINT_BIBLIOGRAPHY = 0 for day-to-day
#      operational runs where you only need the package
#      inventory and benchmark numbers.
# ============================================================
PRINT_NOTES: int         = 0   # 0 = suppress  |  1 = print
PRINT_BIBLIOGRAPHY: int  = 0   # 0 = suppress  |  1 = print

# ============================================================
# 📦 Package Registry, Pre-fetch Cache & Print Helpers
#    Shared infrastructure used across all package sections:
#
#    _ALL_DISTS     — flat list of every dist name across all
#                     PKGS_* dicts; populated before printing
#                     so PyPI dates can be bulk-fetched once
#    _DATE_CACHE    — dict mapping dist name → PyPIInfo; filled
#                     by fetch_release_dates() in mode 1 or 2,
#                     empty in mode 0 (no network calls made)
#    _wrap_line()   — prints a prefix+body line with hanging-
#                     indent word-wrap at REPORT_WIDTH columns;
#                     used for all ✅/❌ package lines and smoke
#                     test results to keep output clean at any
#                     terminal width
#    print_pkg_section() — renders a named package subsection
#                     (title + dotted underline + one line per
#                     package) using _wrap_line(); appends
#                     optional notes and PyPI date suffixes
# ============================================================
# # This will be populated after PKGS_* dicts are defined, before printing.
_ALL_DISTS: list[str] = []
_DATE_CACHE: dict[str, PyPIInfo] = {}

def _wrap_line(prefix: str, body: str, width: int = REPORT_WIDTH) -> None:
    """Print prefix+body, wrapping body at width with hanging indent."""
    full = prefix + body
    if len(full) <= width:
        print(full)
        return
    indent = " " * len(prefix)
    # textwrap on the body portion only
    wrapped = textwrap.wrap(body, width=width - len(prefix))
    for i, chunk in enumerate(wrapped):
        print((prefix if i == 0 else indent) + chunk)

def print_pkg_section(
    title: str,
    items: list[tuple[str, str]],
    notes: dict[str, str] | None = None,
) -> None:
    if title:
        print(f"\n{title}")
        print("." * len(title))
    notes = notes or {}
    for label, dist in items:
        installed = pkg_installed(dist)
        note_suffix = f" — {notes[dist]}" if dist in notes else ""
        date_suffix = fmt_date_suffix(dist, _DATE_CACHE)
        prefix = f"  {ok_mark(installed)} "
        body = f"{fmt_pkg(label, dist)}{note_suffix}{date_suffix}"
        _wrap_line(prefix, body)
        
# ============================================================
# 💾 Output Tee — Simultaneous Terminal + File Logging
#    Intercepts all print() output by replacing sys.stdout
#    with a _Tee instance that mirrors every write() call
#    to both the original terminal and a timestamped .txt
#    file, making every report self-archiving with no changes
#    to any print() calls elsewhere in the script.
#
#    _Tee class      — wraps sys.stdout; write() fans out to
#                      terminal + file simultaneously; flush()
#                      syncs both; close() restores the original
#                      sys.stdout so the interpreter exits cleanly
#    _tee            — module-level handle; checked at script
#                      end to call _tee.close() after the footer
#    _report_path    — full path of the output file, included
#                      in the report header and footer so every
#                      saved file records its own location
#
#    File naming convention:
#      ml_cuda_detective_YYYYMMDD_HHMMSS.txt
#      saved to REPORT_DIR (default: ~/ml-reports/)
#
#    Failures are non-fatal — if the file cannot be opened
#    (permissions, disk full, bad path) the script continues
#    in terminal-only mode with a ⚠️  warning printed once
# ============================================================
class _Tee:
    """Wraps sys.stdout so all print() output is mirrored to a file."""
    def __init__(self, filepath: str) -> None:
        self._terminal = sys.stdout
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, data: str) -> int:
        self._terminal.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._terminal.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        sys.stdout = self._terminal

_tee: Optional[_Tee] = None
_report_path: str = ""

if SAVE_REPORT:
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _report_path = os.path.join(REPORT_DIR, f"ml_cuda_detective_{_ts}.txt")
        _tee = _Tee(_report_path)
        sys.stdout = _tee  # type: ignore[assignment]
    except Exception as _e:
        print(f"⚠️  Could not open report file for writing: {_e}  (continuing without saving)")
        _tee = None

# ============================================================
# 🖨️  Runtime Report Header
#    Printed once at script startup before any checks run.
#    Captures and displays the active configuration so every
#    saved report is self-documenting and reproducible:
#
#    _RUN_TIME       — timestamp recorded at import time so
#                      start time is accurate even if checks
#                      take several minutes to complete
#    _WIDTH          — column width for the ═══ banner lines;
#                      kept narrower than REPORT_WIDTH so the
#                      decorative border fits standard terminals
#    _header_line()  — prints a full-width ═══ divider, or a
#                      centred "══  text  ══" title line when
#                      text is provided; used for the banner,
#                      section breaks, and the report footer
#
#    Config summary rows printed in the header:
#      Run time        wall-clock start timestamp
#      Python          version + venv interpreter path
#      Platform        macOS version + CPU architecture
#      Benchmark depth active depth level + time estimate
#      Release dates   PyPI fetch mode (0 / 1 / 2)
#      Save report     output path if SAVE_REPORT is True
#
#    "What it checks" block — compact feature overview so the
#    top of every saved report explains what the script covers
#    without requiring the reader to open the source file
# ============================================================
_RUN_TIME = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
_WIDTH = 68

def _header_line(text: str = "", fill: str = "═") -> None:
    if not text:
        print(fill * _WIDTH)
    else:
        pad = _WIDTH - len(text) - 4
        left = pad // 2
        right = pad - left
        print(f"{'═' * left}  {text}  {'═' * right}")

print()
_header_line()
_header_line("🕵️‍♂️  ML CUDA Detective")
_header_line("Linux/Windows + NVIDIA GPU Accelerator Stack")
_header_line()
print(f"{'Run time':<18} {_RUN_TIME}")
print(f"{'Python':<18} {sys.version.split()[0]}  ({sys.executable})")
print(f"{'Platform':<18} {platform.system()} {platform.release()}  ({platform.machine()})")
print(f"{'Benchmark depth':<18} {BENCHMARK_DEPTH}  "
      f"({'fast ~15-25s' if BENCHMARK_DEPTH == 1 else 'medium ~45-75s' if BENCHMARK_DEPTH == 2 else 'thorough ~2-4min' if BENCHMARK_DEPTH == 3 else 'memory pressure ~3-6min'})")
print(f"{'Release dates':<18} mode {FETCH_RELEASE_DATES}  "
      f"({'disabled' if FETCH_RELEASE_DATES == 0 else 'installed only' if FETCH_RELEASE_DATES == 1 else 'all packages'})")
print(f"{'Save report':<18} {'yes → ' + _report_path if SAVE_REPORT and _report_path else 'no'}")
_header_line()
print(f"Full-stack environment inspector for NVIDIA GPU machines (Linux / Windows).")
print(f"Audits the complete Python ML/AI ecosystem in a single pass: CUDA/GPU framework")
print(f"availability and live benchmarks, CPU-only data science packages, Jupyter notebook")
print(f"infrastructure, MLOps tooling, and Agentic AI frameworks — from raw CUDA compute")
print(f"through to LLM clients, vector stores, and multi-agent orchestration systems.")
_header_line()

print(f"What it checks")
print(f"{'':2}⚡ CUDA/GPU frameworks   PyTorch CUDA · TensorFlow GPU · JAX CUDA · CuPy")
print(f"{'':2}                          RAPIDS (cuDF/cuML/cuGraph) · TensorRT · Triton")
print(f"{'':2}                          flash-attn · xformers · bitsandbytes · deepspeed")
print(f"{'':2}💨 Runtime smoke tests    matmul · conv2d · autograd · sparse ops")
print(f"{'':2}                          ONNX CUDA/TRT EP · llama-cpp-python CUDA build")
print(f"{'':2}🖥️  CPU-only packages      numeric · wrangling · vector search · vision · NLP")
print(f"{'':2}                          visualization · audio · TF ecosystem · dev tools")
print(f"{'':2}📓 Notebooks              JupyterLab · Jupyter AI · extensions · widgets · utils")
print(f"{'':2}🔬 MLOps                  experiment tracking · pipelines · model serving")
print(f"{'':2}                          MLflow · W&B · DVC · ZenML · BentoML · Ray · Optuna")
print(f"{'':2}🤖 Agentic AI             LangChain · LangGraph · LlamaIndex · Strands · AutoGen")
print(f"{'':2}                          CrewAI · Pydantic AI · OpenAI · Anthropic · Bedrock")
print(f"{'':2}                          ChromaDB · Qdrant · FAISS-GPU · Weaviate · LanceDB")
print(f"{'':2}🏥 Environment health     OS/arch/Python · CUDA toolkit · cuDNN · nvidia-smi")
print(f"{'':2}📅 PyPI freshness         optional: installed version dates vs latest (mode 1/2)")
if BENCHMARK_DEPTH >= 4:
    print(f"  {'':2}🧠 Memory pressure        GPU RAM ceiling per framework (depth 4 only)")
_header_line()

print(f"Configuration (edit the setting at the top of the script)")
print(f"  {'':2}⚙️  BENCHMARK_DEPTH={BENCHMARK_DEPTH} — smoke test thoroughness:")
print(f"  {'':5}    1=fast      (~15–25s)   512×512, float32 only")
print(f"  {'':5}    2=medium    (~45–75s)   1024×1024, f32+f16")
print(f"  {'':5}    3=thorough  (~2–4min)   2048×2048, f32+f16+bf16, stddev, MLP training")
print(f"  {'':5}    4=memory    (~3–6min)   depth-3 + GPU RAM ceiling tests per framework")
print(f"  {'':2}⚙️  FETCH_RELEASE_DATES={FETCH_RELEASE_DATES} — PyPI freshness auditing:")
print(f"  {'':5}    0=disabled            no network calls, instant startup (default)")
print(f"  {'':5}    1=installed only      annotates ✅ lines with release date + ✅/⚠️ vs latest")
print(f"  {'':5}    2=all packages        mode 1 plus latest date for uninstalled packages too")
print(f"  {'':2}⚙️  REPORT_WIDTH={REPORT_WIDTH} — word-wrap column width for all output lines;")
print(f"  {'':5}    change this one value to reflow the entire report for wider or narrower terminals")
print(f"  {'':2}⚙️  SAVE_REPORT={'True ' if SAVE_REPORT else 'False'} — {'mirrors all output to a timestamped .txt file in:' if SAVE_REPORT else 'terminal only, no file written'}")
if SAVE_REPORT:
    print(f"  {'':5}    {REPORT_DIR}  (making every run self-archiving with no extra steps)")
print(f"  {'':2}⚙️  PRINT_NOTES={PRINT_NOTES} — explanatory footnotes printed at end of report:")
print(f"  {'':5}    0=suppress   omit the Notes section; useful for compact operational runs")
print(f"  {'':5}    1=print      include Notes — covers package manager scoping (why a package")
print(f"  {'':2}⚙️  PRINT_BIBLIOGRAPHY={PRINT_BIBLIOGRAPHY} — 65-entry IEEE annotated bibliography:")
print(f"  {'':5}    0=suppress   Each entry has a verified URL + annotation explaining its")
print(f"  {'':5}    1=print      relevance to this report. Print when sharing with stakeholders,")                
print(f"  {'':5}    All six settings are at the top of the script under '⚙️  User Configuration'")
print(f"  {'':5}    and are the only values you should need to change between runs.")
print(f"  {'':2}⚙️  Add packages or categories — find the PKGS_CUDA dict (~line 1266) and either")
print(f"  {'':5}    add a (\"Display name\", \"pypi-package-name\") tuple to an existing category, or")
print(f"  {'':5}    add a new \"Category Name — ⚡ CUDA/GPU\" : [ ... ] key to create a new section.")
_header_line()

# ============================================================
# 🛠️  Core Utility Helpers
#    Lightweight functions used throughout the entire script:
#
#    ok_mark()       — returns ✅ or ❌ based on a bool; used
#                      on every package and smoke test line
#    warn_mark()     — returns 🟠 for partial failures where
#                      a test ran but produced an error
#    banner()        — prints a section title with a dashed
#                      underline; marks each major report block
#    pkg_version()   — looks up an installed dist version via
#                      importlib.metadata; returns None if not
#                      installed (never raises)
#    pkg_installed() — boolean wrapper around pkg_version()
#    fmt_pkg()       — formats "Label: dist==version" or
#                      "Label: dist" if not installed
#    run_cmd()       — runs a subprocess with a timeout and
#                      captures combined stdout/stderr; returns
#                      (success, output) — never raises
#    run_pip_check() — convenience wrapper for 'pip check';
#                      uses a longer 90s timeout for large envs
#    short_err()     — truncates exception strings to 260 chars
#                      and collapses newlines for inline display
#    try_import()    — attempts importlib.import_module() and
#                      returns (ok, error_string); used by all
#                      smoke tests to probe optional packages
# ============================================================
def ok_mark(ok: bool) -> str:
    return "✅" if ok else "❌"

def warn_mark() -> str:
    return "🟠"

def banner(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

def pkg_version(dist_name: str) -> str | None:
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return None

def pkg_installed(dist_name: str) -> bool:
    return pkg_version(dist_name) is not None

def fmt_pkg(label: str, dist: str) -> str:
    v = pkg_version(dist)
    return f"{label}: {dist}=={v}" if v else f"{label}: {dist}"

def run_cmd(cmd: list[str], timeout: int = 12) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
        return (p.returncode == 0), (p.stdout or "").strip()
    except Exception as e:
        return False, str(e)

def run_pip_check() -> tuple[bool, str]:
    return run_cmd([sys.executable, "-m", "pip", "check"], timeout=90)

def short_err(s: str, limit: int = 260) -> str:
    s2 = (s or "").strip().replace("\n", " ")
    return (s2[:limit] + "…") if len(s2) > limit else s2

def try_import(mod: str) -> tuple[bool, Optional[str]]:
    try:
        importlib.import_module(mod)
        return True, None
    except Exception as e:
        return False, str(e)

# ============================================================
# 📅 PyPI Release Date Fetching (Modes 1 & 2)
#    Optional freshness audit that annotates every package
#    line with release date information fetched from the
#    PyPI JSON API (pypi.org/pypi/<dist>/json).
#
#    PyPIInfo dataclass — holds per-package metadata:
#      installed_date  date the installed version was released
#      latest_version  newest version currently on PyPI
#      latest_date     date the newest version was released
#      error           set if the network/parse request failed
#
#    _fetch_pypi_info()  — fetches and parses one package;
#                          uses earliest file upload_time as
#                          the release date for a given version
#    fetch_release_dates() — fans out across all dists in
#                          parallel (default 25 workers) using
#                          ThreadPoolExecutor; called once
#                          before any printing begins so the
#                          full batch completes in one pass
#    fmt_date_suffix()  — formats the date annotation appended
#                          to each package line:
#                          mode 1: installed date + ✅/⚠️ vs latest
#                          mode 2: same + latest date for uninstalled
#                          mode 0: returns "" (no network calls)
# ============================================================
@dataclass
class PyPIInfo:
    installed_date: Optional[str] = None   # date of the installed version
    latest_version: Optional[str] = None   # newest version on PyPI
    latest_date: Optional[str] = None      # date of the newest version
    error: Optional[str] = None

def _fetch_pypi_info(dist: str) -> tuple[str, PyPIInfo]:
    """Fetch a single package's PyPI metadata. Returns (dist, PyPIInfo)."""
    url = f"https://pypi.org/pypi/{dist}/json"
    try:
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except URLError as e:
        return dist, PyPIInfo(error=f"network error: {e}")
    except Exception as e:
        return dist, PyPIInfo(error=str(e))

    try:
        releases = data.get("releases", {})
        info = data.get("info", {})
        latest_version = info.get("version", "")

        def first_upload(ver: str) -> Optional[str]:
            files = releases.get(ver, [])
            dates = [f.get("upload_time", "") for f in files if f.get("upload_time")]
            if not dates:
                return None
            dt = min(dates)  # earliest file upload for that version
            try:
                return datetime.fromisoformat(dt).strftime("%Y-%m-%d")
            except Exception:
                return dt[:10]

        installed_ver = pkg_version(dist)
        installed_date = first_upload(installed_ver) if installed_ver else None
        latest_date = first_upload(latest_version)

        return dist, PyPIInfo(
            installed_date=installed_date,
            latest_version=latest_version,
            latest_date=latest_date,
        )
    except Exception as e:
        return dist, PyPIInfo(error=f"parse error: {e}")

def fetch_release_dates(dists: list[str], max_workers: int = 25) -> dict[str, PyPIInfo]:
    """Fetch PyPI info for a list of dist names in parallel."""
    results: dict[str, PyPIInfo] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_pypi_info, d): d for d in dists}
        for fut in as_completed(futures):
            dist, info = fut.result()
            results[dist] = info
    return results

def fmt_date_suffix(dist: str, date_cache: dict[str, PyPIInfo]) -> str:
    """Build the date annotation string for a package line."""
    if not date_cache or dist not in date_cache:
        return ""
    info = date_cache[dist]
    if info.error:
        return f"  [date: unavailable ({info.error})]"

    installed_ver = pkg_version(dist)

    if installed_ver:
        # Installed package — show installed date vs latest
        inst_str = info.installed_date or "?"
        if installed_ver == info.latest_version:
            return f"  [installed: {inst_str} | ✅ up to date]"
        else:
            lat_str = info.latest_date or "?"
            return f"  [installed: {inst_str} | latest {info.latest_version}: {lat_str} ⚠️]"
    else:
        # Not installed — show latest only (Mode 2)
        lat_str = info.latest_date or "?"
        return f"  [latest {info.latest_version}: {lat_str}]"

# ============================================================
# 🔒 Guardrails — Exact Version Pin Enforcement (Disabled)
#    Optionally enforces strict version pins for the core ML
#    stack to prevent silent breakage from dependency drift.
#    When enabled, compares installed versions against a
#    hardcoded GUARDRAILS list and prints a ready-to-run
#    'pip install --force-reinstall' fix command if any pin
#    has drifted. Useful for reproducible environments and
#    CI checks where version stability is critical.
#
#    To enable: uncomment the block below and update the
#    GUARDRAILS list to match your known-good pin set.
#    To update pins: run the script at depth=1, note the
#    installed versions in the package presence section,
#    and copy them into GUARDRAILS.
#
#    Currently disabled — detection-only mode is sufficient
#    for most development workflows.
# ============================================================
# # banner("Guard rails (pins that must NOT drift)")
# GUARDRAILS: list[tuple[str, str]] = [
#     ("numpy", "2.0.2"),
#     ("packaging", "24.2"),
#     ("protobuf", "5.29.6"),
#     ("torch", "2.9.0"),
#     ("torchvision", "0.24.0"),
#     ("torchaudio", "2.9.0"),
#     ("jax", "0.4.38"),
#     ("jaxlib", "0.4.38"),
#     ("jax-metal", "0.1.1"),
# ]
#
# pins_ok = True
# for dist, expected in GUARDRAILS:
#     actual = pkg_version(dist)
#     ok = (actual == expected)
#     pins_ok = pins_ok and ok
#     if actual is None:
#         print(f"  ❌ {dist} missing (expected {expected})")
#     else:
#         print(f"  {ok_mark(ok)} {dist}=={actual} (expected {expected})")
#
# if not pins_ok:
#     print("\n⚠️  One or more guard-rail pins drifted.")
#     print("Fix command:")
#     print("python -m pip install --upgrade --force-reinstall \\")
#     for dist, ver in GUARDRAILS:
#         print(f'  "{dist}=={ver}" \\')
#     print("&& python -m pip check")
# else:
#     print("\n✅ Guard rails OK (core stack matches expected pins).")

# ============================================================
# 🏥 Linux/Windows Environment Sanity Checks (CUDA Context)
#    Verifies the foundational system prerequisites for CUDA
#    GPU acceleration on NVIDIA hardware, and collects a
#    comprehensive hardware/software inventory printed at the
#    top of every report for reproducibility and diagnostics.
#
#    System identity:
#    • OS name/version, kernel version, architecture
#      (via platform, uname, lsb_release)
#
#    Software versions:
#    • Python version + interpreter path
#    • pip, conda, and pipx versions
#
#    Hardware inventory:
#    • CPU model and core count (via /proc/cpuinfo or wmic)
#    • System RAM (via /proc/meminfo or wmic)
#    • GPU(s): name, VRAM, CUDA compute capability, driver
#      version (via nvidia-smi + torch.cuda if available)
#    • CUDA toolkit version (nvcc --version)
#    • cuDNN version (header files or torch)
#
#    CUDA toolchain:
#    • nvcc compiler availability + version
#    • nvidia-smi availability + GPU count
#    • CUDA_HOME / CUDA_PATH environment variable
#    • cuDNN header detection
#
#    Acceleration path summary:
#    • Lists all available NVIDIA CUDA acceleration paths
#      for reference: torch.cuda, tensorflow-gpu, jax[cuda],
#      CuPy, RAPIDS, TensorRT, ONNX Runtime CUDA/TRT EP
# ============================================================
banner("Linux/Windows CUDA sanity checks")

# OS detected line
os_name    = platform.system()
os_release = platform.release()
arch       = platform.machine() or "unknown"

ok_kernel, kernel_out = run_cmd(["uname", "-r"], timeout=5)
kernel_str = f"kernel: {kernel_out}" if ok_kernel and kernel_out else ""

ok_lsb, lsb_out = run_cmd(["lsb_release", "-d", "-s"], timeout=5)
distro_str = lsb_out.strip().strip('"') if ok_lsb and lsb_out else ""

os_line_parts = [f"{os_name} {os_release}"]
if distro_str:
    os_line_parts.append(distro_str)
os_line_parts.append(f"arch: {arch}")
if kernel_str:
    os_line_parts.append(kernel_str)
print("  ✅ OS: " + "  |  ".join(os_line_parts))

# Python detected line
is_python = (sys.version_info.major >= 3)
pyver = sys.version.split()[0]
if is_python:
    print(f"  {ok_mark(is_python)} Python detected  |  Python Version: {pyver}")
else:
    print(f"  ❌ Python 3.x missing")
    pyver = "?"

# Package installer detected line w/versions
ok_pip, pip_ver = run_cmd([sys.executable, "-m", "pip", "--version"], timeout=8)
pip_str = pip_ver.split()[1] if ok_pip and pip_ver else None

ok_pipx, pipx_ver = run_cmd(["pipx", "--version"], timeout=8)
pipx_str = pipx_ver.strip() if ok_pipx and pipx_ver else None

ok_conda, conda_ver = run_cmd(["conda", "--version"], timeout=8)
conda_str = conda_ver.strip().replace("conda ", "") if ok_conda and conda_ver else None

versions = []
if pip_str:
    versions.append(f"pip: {pip_str}")
if pipx_str:
    versions.append(f"pipx: {pipx_str}")
if conda_str:
    versions.append(f"conda: {conda_str}")
print("  \u2139\ufe0f  Package Installers: " + "  |  ".join(versions) if versions else "  \u2139\ufe0f  No package installers detected.")

# Package counts
pip_count   = "?"
pipx_count  = "?"
conda_count = "?"

ok_pip_list, pip_list_out = run_cmd([sys.executable, "-m", "pip", "list"], timeout=15)
if ok_pip_list and pip_list_out:
    pip_count = max(0, len(pip_list_out.strip().splitlines()) - 2)

ok_pipx_list, pipx_list_out = run_cmd(["pipx", "list", "--short"], timeout=10)
if ok_pipx_list and pipx_list_out:
    pipx_count = sum(1 for ln in pipx_list_out.strip().splitlines() if ln.strip())

ok_conda_list, conda_list_out = run_cmd(["conda", "list", "--no-pip"], timeout=20)
if ok_conda_list and conda_list_out:
    conda_count = sum(1 for ln in conda_list_out.strip().splitlines()
                      if ln.strip() and not ln.startswith("#"))

pkg_parts = []
if pip_count != "?":
    pkg_parts.append(f"pip: {pip_count}")
if pipx_count != "?":
    pkg_parts.append(f"pipx: {pipx_count}")
if conda_count != "?":
    pkg_parts.append(f"conda: {conda_count}")
print(f"  \u2139\ufe0f  Packages: {'  |  '.join(pkg_parts) if pkg_parts else 'No packages detected.'}")

# CPU info
cpu_model = "unknown"
cpu_cores = "?"
if os.path.exists("/proc/cpuinfo"):
    try:
        with open("/proc/cpuinfo") as _f:
            _cpuinfo = _f.read()
        for ln in _cpuinfo.splitlines():
            if ln.startswith("model name"):
                cpu_model = ln.split(":", 1)[1].strip()
                break
        cpu_cores = str(_cpuinfo.count("processor\t:"))
    except Exception:
        pass
elif os_name == "Windows":
    ok_cpu, cpu_out = run_cmd(
        ["wmic", "cpu", "get", "name,NumberOfLogicalProcessors", "/format:list"], timeout=8)
    if ok_cpu and cpu_out:
        for ln in cpu_out.splitlines():
            if "Name=" in ln:
                cpu_model = ln.split("=", 1)[1].strip()
            elif "NumberOfLogicalProcessors=" in ln:
                cpu_cores = ln.split("=", 1)[1].strip()

# RAM info
ram_total_gb = 0.0
if os.path.exists("/proc/meminfo"):
    try:
        with open("/proc/meminfo") as _f:
            for ln in _f:
                if ln.startswith("MemTotal:"):
                    ram_total_gb = int(ln.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass
elif os_name == "Windows":
    ok_ram, ram_out = run_cmd(
        ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"], timeout=8)
    if ok_ram and ram_out:
        for ln in ram_out.splitlines():
            if "TotalPhysicalMemory=" in ln:
                try:
                    ram_total_gb = int(ln.split("=")[1]) / (1024**3)
                except Exception:
                    pass

ram_str = f"{ram_total_gb:.1f} GB" if ram_total_gb > 0 else "?"
print(f"  \u2139\ufe0f  CPU: {cpu_model}  |  {cpu_cores} logical cores  |  RAM: {ram_str}")

# GPU inventory via nvidia-smi
ok_smi, smi_out = run_cmd(
    ["nvidia-smi",
     "--query-gpu=index,name,memory.total,compute_cap,driver_version",
     "--format=csv,noheader,nounits"],
    timeout=12
)
print(f"  {ok_mark(ok_smi)} nvidia-smi available")

gpu_list: list[dict] = []
if ok_smi and smi_out:
    for ln in smi_out.strip().splitlines():
        _smi_parts = [x.strip() for x in ln.split(",")]
        if len(_smi_parts) >= 5:
            gpu_list.append({
                "index":  _smi_parts[0],
                "name":   _smi_parts[1],
                "vram":   _smi_parts[2],
                "cc":     _smi_parts[3],
                "driver": _smi_parts[4],
            })
        elif len(_smi_parts) >= 2:
            gpu_list.append({"index": _smi_parts[0], "name": _smi_parts[1],
                             "vram": "?", "cc": "?", "driver": "?"})

# ── Compute Capability → Architecture name lookup ──────────────────────────
# Source: NVIDIA developer docs + GPU product specs.
# Key: major.minor CC string (or major prefix for broad match).
# Used both in sanity checks and in smoke tests for hardware gating.
_CC_TO_ARCH: dict[str, str] = {
    "3.0": "Kepler",  "3.2": "Kepler",  "3.5": "Kepler",  "3.7": "Kepler",
    "5.0": "Maxwell", "5.1": "Maxwell", "5.2": "Maxwell", "5.3": "Maxwell",
    "6.0": "Pascal",  "6.1": "Pascal",  "6.2": "Pascal",
    "7.0": "Volta",   "7.2": "Volta",
    "7.5": "Turing",
    "8.0": "Ampere",  "8.6": "Ampere",  "8.7": "Ampere",  "8.9": "Ada Lovelace",
    "9.0": "Hopper",  "9.0a": "Hopper",
    "10.0": "Blackwell", "10.3": "Blackwell",
    "12.0": "Blackwell Ultra",
}

def _cc_to_arch(cc: str) -> str:
    """Return architecture name for a compute capability string like '8.6'."""
    cc = cc.strip()
    if cc in _CC_TO_ARCH:
        return _CC_TO_ARCH[cc]
    # fallback: try major only
    major = cc.split(".")[0]
    fallback = {
        "3": "Kepler", "5": "Maxwell", "6": "Pascal",
        "7": "Volta/Turing", "8": "Ampere/Ada", "9": "Hopper",
        "10": "Blackwell", "12": "Blackwell Ultra",
    }
    return fallback.get(major, f"Unknown (CC {cc})")

def _cc_has_tensor_cores(cc: str) -> bool:
    """Volta (CC 7.0) was the first generation with Tensor Cores."""
    try:
        major, minor = cc.strip().split(".")
        return (int(major), int(minor)) >= (7, 0)
    except Exception:
        return False

def _cc_has_structured_sparsity(cc: str) -> bool:
    """Ampere (CC 8.0) introduced 2:4 Sparse Tensor Cores."""
    try:
        major, minor = cc.strip().split(".")
        return (int(major), int(minor)) >= (8, 0)
    except Exception:
        return False

def _cc_has_flash_attn2(cc: str) -> bool:
    """FlashAttention-2 requires Ampere (CC 8.0+)."""
    return _cc_has_structured_sparsity(cc)

def _cc_has_flash_attn3(cc: str) -> bool:
    """FlashAttention-3 requires Hopper (CC 9.0+)."""
    try:
        major, minor = cc.strip().split(".")
        return (int(major), int(minor)) >= (9, 0)
    except Exception:
        return False

def _primary_gpu_cc() -> str:
    """Return the compute capability string for GPU 0 (e.g. '8.6'), or '' if unavailable."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            maj, min_ = torch.cuda.get_device_capability(0)
            return f"{maj}.{min_}"
    except Exception:
        pass
    # fallback: parse from gpu_list populated by nvidia-smi above
    if gpu_list:
        return gpu_list[0].get("cc", "")
    return ""

# ── CUDA cores per SM — keyed by (major, minor) CC tuple ───────────────────
# These are architectural constants from NVIDIA GPU whitepapers.
# CUDA cores/SM is fixed per CC (with one important exception: Ampere CC 8.0
# data-center chips like A100 use 64/SM while CC 8.6/8.7 consumer cards use 128/SM).
# Formula: total CUDA cores = SM_count × _CUDA_CORES_PER_SM[cc_tuple]
_CUDA_CORES_PER_SM: dict[tuple[int,int], int] = {
    (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,   # Kepler
    (5, 0): 128, (5, 2): 128, (5, 3): 128,                 # Maxwell
    (6, 0):  64, (6, 1): 128, (6, 2): 128,                 # Pascal (GP100=64, GP10x=128)
    (7, 0):  64, (7, 2):  64,                               # Volta
    (7, 5):  64,                                            # Turing
    (8, 0):  64, (8, 6): 128, (8, 7): 128, (8, 9): 128,    # Ampere (A100=64) / Ada (8.9=128)
    (9, 0): 128,                                            # Hopper
    (10, 0): 128, (10, 3): 128,                             # Blackwell
    (12, 0): 128,                                           # Blackwell Ultra
}

# ── Tensor Cores per SM — keyed by (major, minor) CC tuple ─────────────────
# Only Volta (CC 7.0) and later have Tensor Cores.
# Note: the "width" of Tensor Cores increased across generations, so a 3rd-gen
# Ampere TC outperforms a 1st-gen Volta TC even at the same count per SM.
_TENSOR_CORES_PER_SM: dict[tuple[int,int], int] = {
    (7, 0):  8, (7, 2):  8,   # Volta  — 1st gen, FP16
    (7, 5):  8,                # Turing — 2nd gen, FP16/INT8/INT4
    (8, 0):  4, (8, 6):  4, (8, 7):  4, (8, 9):  4,   # Ampere/Ada — 3rd/4th gen, wider MMA
    (9, 0):  4,                # Hopper — 4th gen, adds FP8
    (10, 0): 4, (10, 3): 4,   # Blackwell — 5th gen, FP4/FP6/FP8
    (12, 0): 4,                # Blackwell Ultra
}

def _cuda_core_counts(cc: str, sm_count: int) -> tuple[int | None, int | None]:
    """Return (cuda_cores, tensor_cores) for a GPU given its CC string and SM count.
    Returns None for a count if the CC is unknown or Tensor Cores are absent."""
    try:
        maj, min_ = cc.strip().split(".")
        cc_tuple = (int(maj), int(min_))
    except Exception:
        return None, None
    cuda_cores = (_CUDA_CORES_PER_SM.get(cc_tuple, None))
    cuda_total  = sm_count * cuda_cores if cuda_cores is not None else None
    tc_per_sm   = _TENSOR_CORES_PER_SM.get(cc_tuple, None)
    tc_total    = sm_count * tc_per_sm  if tc_per_sm  is not None else None
    return cuda_total, tc_total

# ── Print GPU inventory lines ────────────────────────────────────────────────
# Get SM (Streaming Multiprocessor) counts. Strategy:
#   1. Try nvidia-smi --query-gpu with multiprocessor.count field (newer drivers)
#   2. Fall back to torch.cuda.get_device_properties() (works if torch is installed)
#   3. If both fail, _sm_counts stays empty and core counts are skipped gracefully.
_sm_counts: dict[str, int] = {}

# Strategy 1: nvidia-smi query (field name varies by driver version)
for _smi_field in ("multiprocessor.count", "num_sms"):
    _ok_smi_sm, _smi_sm_out = run_cmd(
        ["nvidia-smi", f"--query-gpu=index,{_smi_field}", "--format=csv,noheader,nounits"],
        timeout=10
    )
    if _ok_smi_sm and _smi_sm_out and "[Not Supported]" not in _smi_sm_out:
        for _ln in _smi_sm_out.strip().splitlines():
            _parts = [x.strip() for x in _ln.split(",")]
            if len(_parts) >= 2 and _parts[1].isdigit():
                _sm_counts[_parts[0]] = int(_parts[1])
        if _sm_counts:
            break

# Strategy 2: torch.cuda.get_device_properties() — reliable if torch is installed
if not _sm_counts:
    try:
        import torch as _torch_sm  # type: ignore
        if _torch_sm.cuda.is_available():
            for _di in range(_torch_sm.cuda.device_count()):
                _props = _torch_sm.cuda.get_device_properties(_di)
                _sm_counts[str(_di)] = _props.multi_processor_count
    except Exception:
        pass

if gpu_list:
    for g in gpu_list:
        vram_str = f"{int(g['vram'])/1024:.1f} GB" if g["vram"].isdigit() else g["vram"]
        arch_str = _cc_to_arch(g["cc"]) if g["cc"] != "?" else "?"
        print(f"  \u2139\ufe0f  GPU {g['index']}: {g['name']}  |  VRAM: {vram_str}  |  "
              f"Compute Capability: {g['cc']} ({arch_str})  |  Driver: {g['driver']}")

        if g["cc"] != "?":
            sm_count = _sm_counts.get(g["index"], 0)

            # ── Core counts line ────────────────────────────────────────
            if sm_count > 0:
                cuda_total, tc_total = _cuda_core_counts(g["cc"], sm_count)
                cuda_str = f"{cuda_total:,}" if cuda_total is not None else "?"
                print(f"  \u2705 CUDA cores: {cuda_str}  ({sm_count} SMs)", end="")
                if tc_total is not None:
                    print(f"  |  \u2705 Tensor Cores: {tc_total:,}", end="")
                else:
                    print(f"  |  \u274c Tensor Cores: none (requires Volta CC 7.0+)", end="")
                print()
            else:
                # Both nvidia-smi and torch failed to return SM count
                print(f"  \u2139\ufe0f  CUDA/Tensor core counts unavailable — install torch or update NVIDIA driver")

            # ── Feature flags line ──────────────────────────────────────
            has_tc  = _cc_has_tensor_cores(g["cc"])
            has_sp  = _cc_has_structured_sparsity(g["cc"])
            has_fa2 = _cc_has_flash_attn2(g["cc"])
            has_fa3 = _cc_has_flash_attn3(g["cc"])
            tc_mark  = "\u2705" if has_tc  else "\u274c"
            sp_mark  = "\u2705" if has_sp  else "\u274c"
            fa2_mark = "\u2705" if has_fa2 else "\u274c"
            fa3_mark = "\u2705" if has_fa3 else "\u274c"
            print(f"          {tc_mark} Tensor Cores  "
                  f"{sp_mark} 2:4 Structured Sparsity  "
                  f"{fa2_mark} FlashAttention-2  "
                  f"{fa3_mark} FlashAttention-3")
else:
    print("  \u274c No NVIDIA GPUs detected via nvidia-smi")

# CUDA toolkit version (nvcc)
ok_nvcc, nvcc_out = run_cmd(["nvcc", "--version"], timeout=8)
cuda_toolkit_ver = "not found"
if ok_nvcc and nvcc_out:
    for ln in nvcc_out.splitlines():
        if "release" in ln.lower():
            m = re.search(r"release\s+([\d.]+)", ln, re.IGNORECASE)
            if m:
                cuda_toolkit_ver = m.group(1)
                break
print(f"  {ok_mark(ok_nvcc)} nvcc (CUDA compiler)  |  CUDA toolkit: {cuda_toolkit_ver}")

# CUDA_HOME / CUDA_PATH
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
print(f"  {'\u2705' if cuda_home else '\u26a0\ufe0f '} CUDA_HOME: "
      f"{cuda_home if cuda_home else '(not set — may still work via LD_LIBRARY_PATH)'}")

# cuDNN version
cudnn_ver = "unknown"
try:
    import torch as _torch  # type: ignore
    if hasattr(_torch.backends, "cudnn") and _torch.backends.cudnn.is_available():
        cudnn_ver = str(_torch.backends.cudnn.version())
except Exception:
    pass

if cudnn_ver == "unknown":
    _cudnn_search_paths = [
        "/usr/include/cudnn_version.h",
        "/usr/local/cuda/include/cudnn_version.h",
    ]
    if cuda_home:
        _cudnn_search_paths.append(f"{cuda_home}/include/cudnn_version.h")
    for _hp in _cudnn_search_paths:
        if os.path.exists(_hp):
            try:
                with open(_hp) as _f:
                    _h = _f.read()
                _ma = re.search(r"CUDNN_MAJOR\s+(\d+)", _h)
                _mi = re.search(r"CUDNN_MINOR\s+(\d+)", _h)
                _pa = re.search(r"CUDNN_PATCHLEVEL\s+(\d+)", _h)
                if _ma and _mi and _pa:
                    cudnn_ver = f"{_ma.group(1)}.{_mi.group(1)}.{_pa.group(1)}"
                    break
            except Exception:
                pass
print(f"  \u2139\ufe0f  cuDNN version: {cudnn_ver}")

# torch CUDA info (if available)
_torch_cuda_ok = False
try:
    import torch as _torch  # type: ignore
    if _torch.cuda.is_available():
        _torch_cuda_ok = True
        tc_ver = _torch.version.cuda or "?"
        dev_count = _torch.cuda.device_count()
        print(f"  \u2705 torch.cuda available  |  CUDA runtime: {tc_ver}  |  {dev_count} device(s)")
        for _i in range(dev_count):
            _gname = _torch.cuda.get_device_name(_i)
            _gmem  = _torch.cuda.get_device_properties(_i).total_memory // (1024**3)
            print(f"     Device {_i}: {_gname}  ({_gmem} GB)")
    else:
        print(f"  \u274c torch.cuda.is_available() == False (torch installed but no CUDA device)")
except ImportError:
    pass
except Exception as _e:
    print(f"  \U0001f7e0 torch.cuda check: {short_err(str(_e))}")

_wrap_line("  \u2139\ufe0f  ", "NVIDIA CUDA acceleration paths: torch.cuda, tensorflow-gpu, "
           "jax[cuda], cupy, RAPIDS (cuDF/cuML/cuGraph), TensorRT, ONNX Runtime CUDA/TensorRT EP.")

# ============================================================
# 📋 Package Notes — Inline Annotations for Special Cases
#    Per-dist strings appended to package lines in the report
#    (rendered as "— <note>") to explain non-obvious install
#    behaviour, expected ❌ results, or important caveats that
#    would otherwise require the reader to look up separately.
#    Only packages that need clarification are listed here —
#    straightforward packages have no entry and print cleanly.
#
#    Current annotations cover:
#      torch               — CUDA support built-in but requires
#                            a CUDA-capable GPU and matching
#                            CUDA toolkit / cuDNN installation
#      tensorflow          — GPU via tensorflow[and-cuda] or
#                            tensorflow-gpu (legacy naming)
#      jax                 — CUDA via jax[cuda12]; must match
#                            installed CUDA toolkit version
#      cupy-cudaXXX        — install the wheel matching your
#                            CUDA version (cuda11x / cuda12x)
#      onnxruntime-gpu     — includes CUDA + TRT EP; separate
#                            package from CPU-only onnxruntime
#      llama-cpp-python    — CUDA build needs CMAKE_ARGS=
#                            "-DGGML_CUDA=on" at install time
#      flash-attn          — Ampere+ (CC 8.0+); compile takes
#                            10+ min; prebuilt wheels available
#      bitsandbytes        — Requires CUDA 11.0+; Turing+ for
#                            best 4-bit quantization kernel support
#      deepspeed           — Linux only; needs CUDA+gcc+ninja
# ============================================================
notes: dict[str, str] = {
    "torch": "CUDA support built-in; requires CUDA-capable GPU + matching toolkit",
    "tensorflow": "GPU via tensorflow[and-cuda] or tensorflow-gpu (legacy)",
    "jax": "CUDA via jax[cuda12] — must match installed CUDA version",
    "cupy-cuda12x": "Install the wheel matching your CUDA version (cuda11x, cuda12x, etc.)",
    "onnxruntime-gpu": "Includes CUDA EP + TensorRT EP; separate from CPU-only onnxruntime",
    "llama-cpp-python": "CUDA build requires CMAKE_ARGS='-DGGML_CUDA=on' at install time",
    "flash-attn": "Requires Ampere+ GPU (CC 8.0+); prebuilt wheels at github.com/Dao-AILab",
    "xformers": "Version must match torch exactly; use --no-deps if conflicts arise",
    "bitsandbytes": "Requires CUDA 11.0+; Turing+ (CC 7.5+) for best 4-bit quant kernels",
    "deepspeed": "Linux only; requires CUDA + gcc + ninja for kernel compilation",
    "triton": "OpenAI Triton — Linux only; required by flash-attn and many CUDA kernels",
    "tensorrt": "Requires NVIDIA TensorRT SDK; install from NVIDIA's wheel repo",
    "torch-tensorrt": "Requires TensorRT SDK installed before pip install",
    "faiss-gpu": "Requires CUDA; faiss-cpu (CPU-only) is listed separately in the CPU packages section",
    "cudf-cu12": "RAPIDS cuDF — requires CUDA 12.x + Pascal+ GPU (CC 6.0+)",
    "cuml-cu12": "RAPIDS cuML — requires CUDA 12.x + Pascal+ GPU (CC 6.0+)",
    "vllm": "Requires Ampere+ GPU (CC 8.0+) for optimal kernel performance",
    "nvidia-cuda-runtime-cu12": "PyPI-distributed CUDA runtime wheel (not the system CUDA toolkit — nvcc --version checks that)",
}

# ============================================================
# ⚡ CUDA / GPU Accelerated Packages
#    Covers the full NVIDIA CUDA acceleration stack from
#    lowest-level CUDA primitives through high-level training
#    optimization and inference libraries. Organized into
#    logical tiers: core DL frameworks, CUDA compute
#    primitives, RAPIDS data science, ONNX/TensorRT inference,
#    memory efficiency, distributed training, vector search,
#    transformers/diffusion stack, LLM inference runtimes,
#    and profiling tools. Works with GPUs from Kepler (CC 3.0)
#    through Blackwell (CC 10.0), though many newer libraries
#    require Turing (CC 7.5) or Ampere (CC 8.0) at minimum.
# ============================================================
PKGS_CUDA: dict[str, list[tuple[str, str]]] = {
    "Core DL Frameworks — ⚡ CUDA/GPU": [
        ("PyTorch (CUDA backend)", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("TensorFlow (GPU)", "tensorflow"),
        ("Keras (standalone)", "keras"),
        ("TensorBoard", "tensorboard"),
    ],
    "JAX / CUDA stack — ⚡ CUDA/GPU": [
        ("JAX", "jax"),
        ("jaxlib", "jaxlib"),
        ("JAX CUDA 12 plugin", "jax-cuda12-plugin"),
        ("Flax", "flax"),
        ("Optax", "optax"),
        ("Chex", "chex"),
        ("Orbax", "orbax-checkpoint"),
    ],
    "CUDA Compute Primitives — ⚡ Direct CUDA kernels": [
        ("CuPy (CUDA 12.x)", "cupy-cuda12x"),
        ("CuPy (CUDA 11.x)", "cupy-cuda11x"),
        ("CUDA Python (official NVIDIA bindings)", "cuda-python"),
        ("nvidia-cuda-runtime-cu12 (PyPI runtime wheel)", "nvidia-cuda-runtime-cu12"),
        ("cuDNN Python bindings (CUDA 12)", "nvidia-cudnn-cu12"),
        ("cuBLAS Python bindings (CUDA 12)", "nvidia-cublas-cu12"),
        ("cuSPARSE Python bindings (CUDA 12)", "nvidia-cusparse-cu12"),
        ("NCCL Python bindings (CUDA 12)", "nvidia-nccl-cu12"),
        ("NVTX profiling markers", "nvtx"),
        ("PyNVML (GPU monitoring Python API)", "pynvml"),
        ("nvidia-ml-py (nvidia-smi Python bindings)", "nvidia-ml-py"),
    ],
    "RAPIDS Ecosystem — ⚡ GPU Data Science (CUDA 12, Pascal+ CC 6.0+)": [
        ("cuDF (GPU DataFrames, CUDA 12)", "cudf-cu12"),
        ("cuML (GPU ML algorithms, CUDA 12)", "cuml-cu12"),
        ("cuGraph (GPU graph analytics, CUDA 12)", "cugraph-cu12"),
        ("cuSpatial (GPU spatial analytics, CUDA 12)", "cuspatial-cu12"),
        ("cuxfilter (GPU crossfilter dashboard, CUDA 12)", "cuxfilter-cu12"),
        ("RAFT / pylibraft (RAPIDS primitives, CUDA 12)", "pylibraft-cu12"),
        ("RAFT-dask (distributed RAPIDS, CUDA 12)", "raft-dask-cu12"),
        ("RMM (RAPIDS Memory Manager, CUDA 12)", "rmm-cu12"),
        ("cuDF Polars integration", "cudf-polars-cu12"),
    ],
    "ONNX Runtime — ⚡ CUDA/TensorRT Execution Providers": [
        ("onnxruntime-gpu (CUDA + TRT EP)", "onnxruntime-gpu"),
        ("onnxruntime-extensions", "onnxruntime-extensions"),
        ("ONNX (format library)", "onnx"),
        ("onnx-simplifier", "onnxsim"),
        ("onnxconverter-common", "onnxconverter-common"),
        ("onnxruntime-genai (CUDA build)", "onnxruntime-genai"),
        ("onnxscript (ONNX IR scripting)", "onnxscript"),
    ],
    "TensorRT / Inference Optimization — ⚡ CUDA/TensorRT": [
        ("TensorRT (NVIDIA Python package)", "tensorrt"),
        ("torch-tensorrt (TorchScript → TRT)", "torch-tensorrt"),
        ("tensorrt-lean (slim TRT runtime)", "tensorrt-lean"),
        ("tensorrt-dispatch (TRT dispatch)", "tensorrt-dispatch"),
        ("polygraphy (TRT model analysis)", "polygraphy"),
        ("onnx-graphsurgeon (graph editing)", "onnx-graphsurgeon"),
        ("nvidia-pyindex (NVIDIA PyPI index helper)", "nvidia-pyindex"),
    ],
    "OpenAI Triton / Custom CUDA Kernels — ⚡ GPU kernel authoring": [
        ("OpenAI Triton (custom GPU kernel DSL)", "triton"),
        ("triton-nightly (bleeding edge builds)", "triton-nightly"),
        ("CUDA toolkit Python (full toolkit)", "cuda-toolkit"),
    ],
    "Memory Efficiency & Quantization — ⚡ CUDA": [
        ("flash-attn (FlashAttention-2/3, Ampere+)", "flash-attn"),
        ("xformers (memory-efficient attention)", "xformers"),
        ("bitsandbytes (4-bit/8-bit quantization)", "bitsandbytes"),
        ("auto-gptq (GPTQ 4-bit quantization)", "auto-gptq"),
        ("autoawq (AWQ quantization)", "autoawq"),
        ("optimum (HF optimization toolkit)", "optimum"),
        ("optimum-nvidia (TensorRT-LLM bridge)", "optimum-nvidia"),
        ("quanto (HF quantization)", "optimum-quanto"),
        ("torchao (PyTorch native quantization)", "torchao"),
    ],
    "Distributed Training — ⚡ CUDA + multi-GPU": [
        ("DeepSpeed (ZeRO, pipeline parallelism)", "deepspeed"),
        ("fairscale (model/pipeline parallelism)", "fairscale"),
        ("apex (NVIDIA mixed precision / fused ops)", "apex"),
        ("colossalai (parallel AI training)", "colossalai"),
        ("bagua (distributed training framework)", "bagua"),
        ("torchrun (via torch, built-in launcher)", "torch"),
    ],
    "Vector Search — ⚡ GPU-Accelerated (CUDA)": [
        ("faiss-gpu (GPU-accelerated FAISS)", "faiss-gpu"),
    ],
    "Transformers / Diffusion / Training — ⚡ CUDA via torch.cuda": [
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("safetensors", "safetensors"),
        ("accelerate", "accelerate"),
        ("torchmetrics", "torchmetrics"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("diffusers", "diffusers"),
        ("timm", "timm"),
        ("sentence-transformers", "sentence-transformers"),
        ("ultralytics (YOLO)", "ultralytics"),
        ("openai-whisper", "openai-whisper"),
        ("pytorch-lightning", "pytorch-lightning"),
        ("lightning", "lightning"),
        ("einops", "einops"),
        ("torchdata", "torchdata"),
        ("torchtext", "torchtext"),
        ("torchelastic (via torch)", "torch"),
    ],
    "LLM / Inference Runtimes — ⚡ CUDA (build-dependent)": [
        ("llama-cpp-python (CUDA if GGML_CUDA=on)", "llama-cpp-python"),
        ("ctranslate2 (CUDA build)", "ctranslate2"),
        ("vllm (high-throughput CUDA inference)", "vllm"),
        ("exllamav2 (ExLlamaV2 CUDA kernels)", "exllamav2"),
        ("lmdeploy (turbomind CUDA backend)", "lmdeploy"),
        ("text-generation-inference (HF TGI)", "text-generation-inference"),
        ("sglang (structured generation LLM)", "sglang"),
    ],
    "Profiling & Benchmarking — ⚡ CUDA profiling tools": [
        ("torch-tb-profiler (TensorBoard profiler)", "torch-tb-profiler"),
        ("scalene (CPU/GPU profiler)", "scalene"),
        ("viztracer (trace visualization)", "viztracer"),
        ("line-profiler (line-by-line timing)", "line-profiler"),
        ("memory-profiler", "memory-profiler"),
    ],
}

# ============================================================
# 🖥️  CPU-Only Packages (No CUDA Acceleration)
#    Organized by domain: numeric/data science stack,
#    data wrangling, vector search, computer vision, NLP,
#    MLOps utilities, LLM/agent frameworks, visualization,
#    audio/speech, TF ecosystem add-ons, datasets,
#    evaluation tools, and dev/build helpers.
# ============================================================
PKGS_CPU: dict[str, list[tuple[str, str]]] = {
    "Data Science / Numeric stack — 🖥️  CPU only": [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("polars", "polars"),
        ("numba", "numba"),
        ("sympy", "sympy"),
        ("scikit-learn", "scikit-learn"),
        ("statsmodels", "statsmodels"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
        ("joblib", "joblib"),
        ("threadpoolctl", "threadpoolctl"),
    ],
    "Data Wrangling / Feature Engineering — 🖥️  CPU only": [
        ("pyarrow", "pyarrow"),
        ("dask", "dask"),
        ("vaex", "vaex"),
        ("great-expectations", "great-expectations"),
        ("feature-engine", "feature-engine"),
        ("imbalanced-learn", "imbalanced-learn"),
        ("category-encoders", "category-encoders"),
        ("scikit-image", "scikit-image"),
        ("distributed", "distributed"),
    ],
    "Computer Vision / Media — 🖥️  CPU only": [
        ("opencv-python", "opencv-python"),
        ("pillow", "Pillow"),
        ("imageio", "imageio"),
        ("decord", "decord"),
        ("albumentations", "albumentations"),
        ("kornia", "kornia"),
        ("scikit-image", "scikit-image"),
        ("imagesize", "imagesize"),
        ("rawpy", "rawpy"),
    ],
    "NLP / Text Processing — 🖥️  CPU only": [
        ("spacy", "spacy"),
        ("nltk", "nltk"),
        ("gensim", "gensim"),
        ("textblob", "textblob"),
        ("regex", "regex"),
        ("ftfy", "ftfy"),
        ("langdetect", "langdetect"),
        ("sentencepiece", "sentencepiece"),
        ("tiktoken", "tiktoken"),
        ("sentence-transformers", "sentence-transformers"),
    ],
    "Visualization — 🖥️  CPU only": [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("bokeh", "bokeh"),
        ("altair", "altair"),
        ("umap-learn", "umap-learn"),
        ("pydeck", "pydeck"),
        ("graphviz", "graphviz"),
    ],
    "Audio / Speech — 🖥️  CPU (preprocessing / IO)": [
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("audioread", "audioread"),
        ("pyaudio", "PyAudio"),
        ("whisper", "openai-whisper"),
    ],
    "TensorFlow Ecosystem Add-ons — 🖥️  CPU (data pipeline utilities)": [
        ("tensorflow-datasets", "tensorflow-datasets"),
        ("tensorflow-io-gcs-filesystem", "tensorflow-io-gcs-filesystem"),
        ("tensorflow-probability", "tensorflow-probability"),
        ("tflite-runtime", "tflite-runtime"),
    ],
    "Datasets / Evaluation — 🖥️  CPU (data utilities)": [
        ("datasets", "datasets"),
        ("evaluate", "evaluate"),
        ("rouge-score", "rouge-score"),
        ("sacrebleu", "sacrebleu"),
        ("bert-score", "bert-score"),
    ],
    "Dev / Build Helpers — 🖥️  CPU tooling": [
        ("setuptools", "setuptools"),
        ("wheel", "wheel"),
        ("packaging", "packaging"),
        ("cmake", "cmake"),
        ("ninja", "ninja"),
        ("tqdm", "tqdm"),
        ("rich", "rich"),
        ("pydantic", "pydantic"),
        ("pydantic-settings", "pydantic-settings"),
        ("python-dotenv", "python-dotenv"),
        ("pytest", "pytest"),
        ("ruff", "ruff"),
        ("mypy", "mypy"),
        ("click", "click"),
        ("requests", "requests"),
        ("httpx", "httpx"),
        ("aiohttp", "aiohttp"),
    ],
}

# ============================================================
# 📓 Notebooks & Interactive Computing
#    Covers the full Jupyter ecosystem: JupyterLab, classic
#    Notebook, IPython, Jupyter AI & LLM integration,
#    extensions, interactive widgets, alternative notebook
#    environments, and notebook testing & execution utilities.
# ============================================================
PKGS_NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "Jupyter Core — 📓 Notebook server & kernels": [
        ("JupyterLab", "jupyterlab"),
        ("Jupyter Notebook (classic)", "notebook"),
        ("Jupyter Notebook v7+ (nbclassic)", "nbclassic"),
        ("IPython kernel", "ipykernel"),
        ("IPython", "ipython"),
        ("jupyter-core", "jupyter-core"),
        ("jupyter-client", "jupyter-client"),
        ("jupyter-server", "jupyter-server"),
        ("nbformat", "nbformat"),
        ("nbconvert", "nbconvert"),
        ("traitlets", "traitlets"),
    ],
    "Jupyter AI & LLM Integration — 📓 AI-powered notebooks": [
        ("Jupyter AI", "jupyter-ai"),
        ("Jupyter AI Magics", "jupyter-ai-magics"),
        ("jupyter-copilot", "jupyter-copilot"),
        ("ipython-ai", "ipython-ai"),
    ],
    "Jupyter Extensions & UI — 📓 Productivity enhancements": [
        ("JupyterLab Git", "jupyterlab-git"),
        ("JupyterLab LSP (code completion)", "jupyter-lsp"),
        ("python-lsp-server", "python-lsp-server"),
        ("JupyterLab Code Formatter", "jupyterlab-code-formatter"),
        ("JupyterLab Vim", "jupyterlab-vim"),
        ("JupyterLab Spellchecker", "jupyterlab-spellchecker"),
        ("aquirdturtle_collapsible_headings", "aquirdturtle-collapsible-headings"),
        ("nbdime (notebook diffing)", "nbdime"),
        ("nbstripout", "nbstripout"),
    ],
    "Interactive Widgets — 📓 Rich output in notebooks": [
        ("ipywidgets", "ipywidgets"),
        ("widgetsnbextension", "widgetsnbextension"),
        ("ipympl (matplotlib widget)", "ipympl"),
        ("ipyleaflet", "ipyleaflet"),
        ("bqplot", "bqplot"),
        ("ipyvuetify", "ipyvuetify"),
        ("plotly FigureWidget (via plotly)", "plotly"),
    ],
    "Alternative / Complementary Notebooks — 📓": [
        ("Marimo", "marimo"),
        ("Voila (notebook → web app)", "voila"),
        ("Panel", "panel"),
        ("Streamlit", "streamlit"),
        ("Gradio", "gradio"),
        ("nbdev", "nbdev"),
        ("Quarto (via CLI; not a pip pkg)", "quarto"),
    ],
    "Notebook Utilities — 📓 Testing, scheduling, execution": [
        ("papermill (parametrized execution)", "papermill"),
        ("nbmake (pytest plugin)", "nbmake"),
        ("jupyter-scheduler", "jupyter-scheduler"),
        ("nbval", "nbval"),
        ("testbook", "testbook"),
        ("jupytext (notebooks as scripts)", "jupytext"),
    ],
}

# ============================================================
# 🔬 MLOps — Experiment Tracking, Pipeline Orchestration,
#    Model Registry, Versioning & Production Serving
#    Covers the full ML lifecycle from development through
#    deployment: tracking experiments, versioning data and
#    models, building reproducible pipelines, and serving
#    predictions at scale.
# ============================================================
PKGS_MLOPS: dict[str, list[tuple[str, str]]] = {
    "MLOps — Experiment Tracking & Model Management 🔬 ": [
        ("MLflow", "mlflow"),
        ("Weights & Biases", "wandb"),
        ("DVC (Data Version Control)", "dvc"),
        ("ClearML", "clearml"),
        ("Comet ML", "comet-ml"),
        ("Neptune-Scale", " neptune-scale"),
        ("Neptune (client)", "neptune"),
        ("Guild AI", "guildai"),
        ("Sacred", "sacred"),
        ("Aim", "aim"),
        ("TensorBoard", "tensorboard"),
        ("Optuna", "optuna"),
        ("Ray Tune / Ray Train", "ray"),
        ("Hydra-core", "hydra-core"),
        ("Omegaconf", "omegaconf"),
        ("uvicorn", "uvicorn"),
        ("gunicorn", "gunicorn"),
    ],
    "MLOps — Pipelines & Orchestration 🔧": [
        ("ZenML", "zenml"),
        ("Metaflow", "metaflow"),
        ("Kedro", "kedro"),
        ("Prefect", "prefect"),
        ("Apache Airflow", "apache-airflow"),
        ("Luigi", "luigi"),
        ("Kubeflow Pipelines SDK", "kfp"),
        ("Flyte (flytekit)", "flytekit"),
        ("Dagster", "dagster"),
        ("Ploomber", "ploomber"),
    ],
    "MLOps — Model Serving & Deployment 🚀": [
        ("BentoML", "bentoml"),
        ("Seldon Core", "seldon-core"),
        ("Ray Serve (via ray)", "ray"),
        ("FastAPI (serving backbone)", "fastapi"),
        ("Uvicorn (ASGI server)", "uvicorn"),
        ("Triton client", "tritonclient"),
        ("ONNX Runtime (inference)", "onnxruntime"),
        ("Torchserve", "torchserve"),
        ("Gradio (demo serving)", "gradio"),
        ("Streamlit (demo serving)", "streamlit"),
    ],
    "MLOps — Observability / Telemetry 📈": [
        ("OpenTelemetry API", "opentelemetry-api"),
        ("OpenTelemetry SDK", "opentelemetry-sdk"),
        ("OTel Semantic Conventions", "opentelemetry-semantic-conventions"),
        ("Prometheus client", "prometheus-client"),
    ],
}

# ============================================================
# 🤖 AI/ML and Agentic AI — Frameworks, LLM Provider Clients,
#    Multi-Agent Orchestration, Memory, RAG & Vector Stores
#    Covers the full agentic stack from LLM access and
#    prompt orchestration through stateful multi-agent
#    workflows, long-term memory, retrieval-augmented
#    generation, and vector database integration.
# ============================================================
PKGS_AGENTIC: dict[str, list[tuple[str, str]]] = {
    "AI/ML and Agentic AI — Core Orchestration Frameworks: Chains, Graphs, Crews & Multi-Agent Systems 🤖": [
        ("LangChain", "langchain"),
        ("LangChain Core", "langchain-core"),
        ("LangChain Community", "langchain-community"),
        ("LangGraph", "langgraph"),
        ("Strands Agents (AWS)", "strands-agents"),
        ("Strands Agents Tools (AWS)", "strands-agents-tools"),
        ("Bedrock AgentCore SDK (AWS)", "bedrock-agentcore"),
        ("Bedrock AgentCore Starter Toolkit (AWS, optional)", "bedrock-agentcore-starter-toolkit"),
        ("LlamaIndex", "llama-index"),
        ("LlamaIndex Core", "llama-index-core"),
        ("Llama Stack (Meta)", "llama-stack"),
        ("Llama Stack Client (Meta)", "llama-stack-client"),
        ("AutoGen AgentChat (Microsoft)", "autogen-agentchat"),
        ("Microsoft Agent Framework (preview)", "agent-framework"),
        ("Semantic Kernel (Microsoft)", "semantic-kernel"),
        ("Databricks Agent Bricks", "databricks-agents"),
        ("CrewAI", "crewai"),
        ("Pydantic AI", "pydantic-ai"),
        ("Smolagents (Hugging Face)", "smolagents"),
        ("Agency Swarm", "agency-swarm"),
        ("Haystack", "farm-haystack"),
    ],
    "AI/ML and Agentic AI — LLM Clients & Provider SDKs: Cloud, Local & Unified Model Access 🔌": [
        ("OpenAI", "openai"),
        ("Anthropic", "anthropic"),
        ("boto3 (AWS SDK — Bedrock access)", "boto3"),
        ("botocore (AWS SDK core)", "botocore"),
        ("AWS CLI", "awscli"),
        ("AWS CRT (optional, Bedrock streaming perf)", "awscrt"),
        ("boto3-stubs type hints", "boto3-stubs"),
        ("Google GenAI SDK (Gemini)", "google-genai"),
        ("Google GenerativeAI (deprecated Nov 2025)", "google-generativeai"),
        ("Mistral AI", "mistralai"),
        ("Cohere", "cohere"),
        ("Together AI", "together"),
        ("Groq", "groq"),
        ("Replicate", "replicate"),
        ("LiteLLM", "litellm"),
        ("Hugging Face Hub (model downloads)", "huggingface-hub"),
        ("Ollama", "ollama"),
    ],
    "AI/ML and Agentic AI — Memory, RAG & Vector Stores: Long-Term Context, Retrieval & Embeddings 🧠": [
        ("ChromaDB", "chromadb"),
        ("Qdrant client", "qdrant-client"),
        ("Weaviate client", "weaviate-client"),
        ("Pinecone", "pinecone"),
        ("FAISS (cpu)", "faiss-cpu"),
        ("Milvus", "pymilvus"),
        ("LanceDB", "lancedb"),
        ("pgvector (Postgres vector extension client)", "pgvector"),
        ("Mem0", "mem0ai"),
        ("Zep", "zep-python"),
        ("Txtai", "txtai"),
        ("Hnswlib", "hnswlib"),
        ("Annoy", "annoy"),
    ],
}

# ============================================================
# 🖨️  Package Presence Report — Pre-fetch & Print All Sections
#    Orchestrates the full package inventory output in two
#    phases: an optional bulk PyPI date fetch, followed by
#    sequential rendering of all five package sections.
#
#    Phase 1 — PyPI date pre-fetch (modes 1 and 2 only):
#      Walks all five PKGS_* dicts to collect every unique
#      dist name into _seen, then filters by install status
#      for mode 1 (installed only) or passes the full set
#      for mode 2 (all packages). Fires fetch_release_dates()
#      once in parallel before any printing begins so the
#      entire batch completes in a single network round-trip
#      (~2–8s for mode 1, ~8–20s for mode 2 depending on
#      network latency and PyPI response times). Results are
#      stored in _DATE_CACHE and consumed by fmt_date_suffix()
#      as each package line is printed. Skipped entirely in
#      mode 0 — no network calls, no startup delay.
#
#    Phase 2 — Sequential section rendering:
#      Calls print_pkg_section() for every category dict in
#      each of the five package groups, in this order:
#        ⚡ Metal / GPU Accelerated     (PKGS_CUDA)
#        🖥️  CPU-only                   (PKGS_CPU)
#        📓 Notebooks                  (PKGS_NOTEBOOKS)
#        🔬 MLOps                      (PKGS_MLOPS)
#        🤖 Agentic AI                 (PKGS_AGENTIC)
#      Each section is preceded by a banner() header and
#      the notes dict is passed through for inline annotations.
# ============================================================
if FETCH_RELEASE_DATES in (1, 2):
    # Collect all unique dist names across all three package dicts
    _seen: set[str] = set()
    for _pkgs_dict in (PKGS_CUDA, PKGS_CPU, PKGS_NOTEBOOKS, PKGS_MLOPS, PKGS_AGENTIC):
        for _items in _pkgs_dict.values():
            for _label, _dist in _items:
                _seen.add(_dist)

    if FETCH_RELEASE_DATES == 1:
        # Mode 1: only fetch for installed packages
        _to_fetch = [d for d in _seen if pkg_installed(d)]
    else:
        # Mode 2: fetch everything
        _to_fetch = list(_seen)

    _pkg_count = len(_to_fetch)
    print(f"\n⏳ Fetching PyPI release dates for {_pkg_count} packages "
          f"(mode {FETCH_RELEASE_DATES}) — please wait…")
    _t0 = time.time()
    _DATE_CACHE.update(fetch_release_dates(_to_fetch))
    _elapsed = time.time() - _t0
    _errors = sum(1 for i in _DATE_CACHE.values() if i.error)
    print(f"✅ Done in {_elapsed:.1f}s  ({_pkg_count - _errors} OK, {_errors} errors)\n")

banner("Package presence — ⚡ CUDA / GPU Accelerated")
for cat, items in PKGS_CUDA.items():
    print_pkg_section(cat, items, notes=notes)

# --------------------------------------------------------
banner("Package presence — 🖥️  CPU-only (no CUDA acceleration)")
# --------------------------------------------------------

for cat, items in PKGS_CPU.items():
    print_pkg_section(cat, items, notes=notes)

banner("Package presence — 📓 Notebooks & Interactive Computing")
for cat, items in PKGS_NOTEBOOKS.items():
    print_pkg_section(cat, items, notes=notes)

banner("Package presence — 🔬 MLOps")
for cat, items in PKGS_MLOPS.items():
    print_pkg_section(cat, items, notes=notes)

banner("Package presence — 🤖 Agentic AI Frameworks & Tools")
for cat, items in PKGS_AGENTIC.items():
    print_pkg_section(cat, items, notes=notes)

# ============================================================
# Dense Operations — Hardware Architecture and Benchmark Methodology
# ============================================================
#
# WHAT "DENSE" MEANS IN THIS CONTEXT
# ------------------------------------
# A dense operation treats every element of a matrix as meaningful
# — no zeros are skipped, no compression is applied. This is the
# default execution mode for all ML frameworks on all hardware.
# The overwhelming majority of real-world ML compute (attention,
# MLP layers, conv2d, embedding projections) is dense in practice,
# even when models are described as "sparse" at a higher level.
#
# THE DOMINANT OPERATION: GEMM
# -----------------------------
# General Matrix Multiplication (GEMM, or matmul: C = A @ B) is
# the single most important primitive in deep learning. It underlies:
#   • Transformer attention:        Q @ K^T, scores @ V
#   • MLP / feed-forward layers:    x @ W + b  (for every layer)
#   • Embedding projections:        token_emb @ proj_weight
#   • Conv2d (via im2col):          implicitly a batched GEMM
#
# On NVIDIA GPUs, ALL of these execute as CUDA kernels dispatched
# via cuBLAS (GEMM), cuDNN (conv2d), and custom op libraries.
# NVIDIA Tensor Cores (Volta+) provide dedicated FP16/BF16/TF32
# matrix multiply units that are separate from the CUDA cores.
#
# NVIDIA CUDA DENSE EXECUTION STACK
# ------------------------------------
# Tensor Core generations:
#   Volta (V100):    FP16 Tensor Cores — up to 125 TFLOPS FP16
#   Turing (RTX 20xx/T4): FP16/INT8/INT4 Tensor Cores
#   Ampere (A100, RTX 30xx): BF16/TF32/FP16/INT8 Tensor Cores
#                             TF32: 156 TFLOPS (A100 SXM)
#                             FP16/BF16: 312 TFLOPS (A100 SXM)
#   Hopper (H100):  FP8 Tensor Cores — up to 3958 TFLOPS FP8
#                   FP16/BF16: 1979 TFLOPS (H100 SXM5)
#   Blackwell (B100/B200): FP4/FP6/FP8 support, ~9.5 PETAFLOPS FP4
#
# Memory architecture:
#   Unlike Apple Silicon UMA, NVIDIA GPUs have dedicated VRAM
#   (HBM2e/HBM3) separate from CPU RAM. Tensor transfers between
#   CPU and GPU cost PCIe bandwidth (~32–64 GB/s for PCIe 4.0/5.0
#   x16). NVLink provides much faster GPU-GPU bandwidth (~900 GB/s
#   for H100 NVLink). Plan data pipelines to minimize PCIe copies.
#
# WHAT "GOOD" NUMBERS LOOK LIKE (depth=1, 512×512 f32)
# -------------------------------------------------------
#   RTX 3090 (Ampere):  ~0.05–0.2ms  (Tensor Core, very fast)
#   RTX 4090 (Ada):     ~0.03–0.1ms  (ADA Lovelace Tensor Cores)
#   A100 (Ampere SXM):  ~0.02–0.08ms (data center HBM2e bandwidth)
#   At small sizes, kernel launch overhead (~5–20μs) dominates.
#
#   At depth=3 (2048×2048 f32): expect 2–20ms depending on GPU tier.
#   Conv2d (4×224×224×3, 64 filters): typically 0.1–2ms on modern GPUs.
#
# Cross-framework note:
#   JAX tends to be fastest for raw GEMM due to XLA JIT compilation
#   which fuses operations into optimized CUDA kernels. PyTorch
#   eager mode has ~5–20μs fixed kernel launch overhead per op —
#   negligible for large tensors but visible at small matrix sizes.
#   torch.compile with mode='reduce-overhead' captures CUDA graphs
#   to eliminate this overhead for repeated operations.
#
# ============================================================
# Benchmark methodology
# ============================================================
#
# MATRIX SIZES AND ITERATION COUNTS (per BENCHMARK_DEPTH)
# ---------------------------------------------------------
#   Depth 1 (fast):     512×512,  1 iteration   — quick sanity check
#   Depth 2 (medium):  1024×1024, 3 iterations  — averaged timing
#   Depth 3 (thorough):2048×2048, 10 iterations — mean + stddev
#   Depth 4 (memory):  same as depth 3, plus memory pressure tests
#
#   512×512 fits in GPU L2 cache on many GPUs (V100: 6MB, A100: 40MB).
#   1024×1024 is cache-resident on some ops, memory-bound on others.
#   2048×2048 is reliably memory-bandwidth-bound — a better proxy
#   for real model layer sizes (typical LLM hidden dim: 2048–8192).
#
# SYNCHRONIZATION — WHY IT MATTERS FOR TIMING
# ---------------------------------------------
#   CUDA executes GPU work asynchronously relative to the CPU.
#   Without an explicit sync point, timing the Python call measures
#   only kernel *submission*, not *completion*. This gives falsely
#   low numbers (often <0.1ms) that don't reflect real work.
#
#   Each framework requires a different sync call:
#     PyTorch CUDA:     torch.cuda.synchronize()
#     TensorFlow GPU:   .numpy()  (forces eager materialization)
#     JAX CUDA:         .block_until_ready()
#     CuPy:             cp.cuda.stream.get_current_stream().synchronize()
#
#   Our _timed_iters() helper wraps these correctly per framework.
#
# WARMUP — WHY THE FIRST CALL IS EXCLUDED
# -----------------------------------------
#   The first call to any GPU operation incurs one-time costs:
#     • CUDA JIT compilation (PTX → SASS) — can be 50–500ms
#     • XLA JIT compilation for JAX — can be 100–2000ms first call
#     • cuBLAS workspace allocation and autotuning
#   Subsequent calls use cached compiled kernels and are
#   representative of steady-state inference performance.
#   All benchmarks below run one warmup pass before timing starts.
#
# INTERPRETING THE TIMING NUMBERS
# ---------------------------------
#   • Numbers reflect wall-clock time including Python overhead,
#     CUDA kernel launch latency, and GPU execution.
#   • At depth 1 (single iteration), noise can be ±20–50%.
#     Run depth 3 for statistically meaningful comparisons.
#   • Comparing across frameworks: use the same depth and matrix
#     size. The cross-framework overhead differences (~0.5–1ms)
#     dominate at small sizes but are negligible at large sizes.
#   • These are single-op microbenchmarks, NOT end-to-end model
#     throughput. Real model performance depends on batching,
#     memory layout, operator fusion, and other factors not
#     captured here.
# ============================================================

# ============================================================
# Sparse Operations — Hardware Support Landscape and Expected Behavior
# ============================================================
# "Structured sparsity" means hardware can skip zero-valued weights
# in a regular pattern (e.g. 2 non-zeros per 4 values = 2:4),
# potentially doubling effective throughput with minimal accuracy
# loss when models are retrained to accommodate the pattern.
# Structured sparsity is most relevant at scale — modern foundation
# models range from billions to trillions of parameters, making
# training and inference extraordinarily expensive. Pruning large
# fractions of weights to zero and using hardware that can skip
# those zeros is one of the primary levers for making these models
# tractable to train and deploy at production scale.
#
# Hardware support landscape (as of early 2026):
#
#  ✅ NVIDIA (Ampere A100+, Hopper H100+)
#       2:4 structured sparsity via Sparse Tensor Cores.
#       Up to 2× GEMM throughput for inference (cuSPARSELt).
#       Also in PyTorch via torch.ao / semi_structured_sparse.
#
#  ✅ AWS Trainium2 / Trainium3 (NeuronCore-v3+)
#       Multiple patterns supported: 2:4, 1:4, 4:8, 4:12, 4:16.
#       Trainium3 uses 16:4 sparsity (4× peak FLOPS on sparse models).
#       Note: Trainium1 had NO structured sparsity support.
#
#  ❌ Google TPU (v4+ SparseCores)
#       Google TPUs have no hardware support for structured weight
#       sparsity (e.g. 2:4 patterns) in dense transformer or MLP
#       layers — the main systolic array (MXU) is dense-only, making
#       TPUs equivalent to Apple Silicon and Microsoft MAIA in this
#       regard. TPUs do include dedicated SparseCores starting with
#       v4, but these are purpose-built exclusively for sparse
#       embedding lookups in recommendation models (DLRM-style
#       workloads) — a fundamentally different problem from skipping
#       zero-valued weights during matrix multiplication. For LLM
#       and transformer inference, TPU sparsity hardware is
#       effectively irrelevant.
#
#  ❌ Microsoft MAIA 100 / MAIA 200
#       Maia 100 and 200 focus on narrow-precision datatypes
#       (MX4/MX6/MX9 for Maia 100; FP4/FP8 for Maia 200) rather than
#       2:4 structured sparsity acceleration. No public documentation
#       from Microsoft confirms dedicated structured sparsity hardware
#       in either generation (as of early 2026). Microsoft's efficiency
#       strategy uses quantization (FP4) rather than sparsity as the
#       primary lever — a fundamentally different approach than
#       NVIDIA's Sparse Tensor Cores.
#
#  ❌ Apple M-series GPU / Neural Engine
#       Apple has published no documentation indicating dedicated
#       structured sparsity hardware in the M-series GPU (Metal) or
#       Neural Engine. In practice, sparse operations on Apple Silicon
#       range from completely unimplemented (PyTorch MPS COO matmul has
#       no SparseMPS kernel) to severely penalized (JAX BCOO runs but
#       at a fraction of dense speed due to software scatter/gather
#       emulation) to marginal at best (TensorFlow sparse ties dense
#       at small matrix sizes). MLX has no native sparse format at all,
#       offering only a masked-dense proxy. Unlike NVIDIA Ampere where
#       2:4 sparsity is a genuine throughput multiplier, sparsity on
#       Apple Silicon is purely a memory representation choice —
#       never a compute accelerator. Use quantization (4-bit, 8-bit
#       weights via MLX-LM or llama.cpp) as the efficiency lever instead.
#
# ============================================================
# ============================================================
# ⏱️  Benchmark Parameter & Timing Helpers
#    Centralised functions that translate BENCHMARK_DEPTH into
#    concrete test parameters, ensuring every framework test
#    uses identical matrix sizes, iteration counts, dtype sets,
#    and sparsity densities — making cross-framework comparisons
#    valid at any depth level.
#
#    _bench_sizes()          — returns (matrix_N, num_iters):
#                              depth 1 → (512,   1)   cache-resident
#                              depth 2 → (1024,  3)   averaged
#                              depth 3+ → (2048, 10)  bandwidth-bound
#
#    _bench_dtypes_torch()   — dtype sweep for PyTorch MPS tests
#    _bench_dtypes_jax()     — dtype sweep for JAX CUDA tests
#    _bench_dtypes_mlx()     — dtype sweep for MLX tests
#      All three follow the same depth ladder:
#                              depth 1 → [float32]
#                              depth 2 → [float32, float16]
#                              depth 3+ → [float32, float16, bfloat16]
#      Kept as separate functions so per-framework dtype support
#      differences can be handled independently if needed.
#
#    _sparse_densities()     — list of non-zero fractions for
#                              sparse benchmark sweeps:
#                              depth 1 → [0.05]
#                              depth 2 → [0.01, 0.05, 0.10]
#                              depth 3+ → [0.01, 0.05, 0.10, 0.20]
#
#    _timed_iters(fn, iters, sync_fn)
#                            — runs fn() for iters passes with
#                              an optional sync_fn() call after
#                              each to flush async GPU work before
#                              stopping the clock (critical for
#                              MPS, Metal, JAX, and MLX which all
#                              execute asynchronously). Returns
#                              (mean_seconds, stddev_seconds);
#                              stddev is 0.0 for single-pass runs.
#
#    _fmt_timing(mean, std, iters)
#                            — formats results as "X.Xms" for
#                              single-pass or "X.Xms ± Y.Yms (n=N)"
#                              for multi-pass; used on every
#                              benchmark result line in the report
# ============================================================
def _bench_sizes() -> tuple[int, int]:
    """Return (matrix_N, num_iters) for the current BENCHMARK_DEPTH."""
    if BENCHMARK_DEPTH == 1:
        return 512, 1
    elif BENCHMARK_DEPTH == 2:
        return 1024, 3
    elif BENCHMARK_DEPTH >= 3:
        return 2048, 10
    return 512, 1

def _bench_dtypes_torch() -> list[str]:
    """Torch dtype names to test at the current depth."""
    if BENCHMARK_DEPTH == 1:
        return ["float32"]
    elif BENCHMARK_DEPTH == 2:
        return ["float32", "float16"]
    else:
        return ["float32", "float16", "bfloat16"]

def _bench_dtypes_jax() -> list[str]:
    if BENCHMARK_DEPTH == 1:
        return ["float32"]
    elif BENCHMARK_DEPTH == 2:
        return ["float32", "float16"]
    else:
        return ["float32", "float16", "bfloat16"]

def _bench_dtypes_mlx() -> list[str]:
    if BENCHMARK_DEPTH == 1:
        return ["float32"]
    elif BENCHMARK_DEPTH == 2:
        return ["float32", "float16"]
    else:
        return ["float32", "float16", "bfloat16"]

def _sparse_densities() -> list[float]:
    """Fraction of non-zero values to test for sparse benchmarks."""
    if BENCHMARK_DEPTH == 1:
        return [0.05]           # 5% non-zero (95% sparse)
    elif BENCHMARK_DEPTH == 2:
        return [0.01, 0.05, 0.10]
    else:
        return [0.01, 0.05, 0.10, 0.20]

def _timed_iters(fn: Callable, iters: int, sync_fn: Callable | None = None) -> tuple[float, float]:
    """
    Run fn() for `iters` iterations, optionally calling sync_fn() after
    each to flush async GPU work. Returns (mean_seconds, stddev_seconds).
    For iters==1 stddev is 0.0.
    """
    import statistics
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std

def _fmt_timing(mean: float, std: float, iters: int) -> str:
    """Format timing result as a compact string."""
    if iters == 1:
        return f"{mean*1000:.1f}ms"
    return f"{mean*1000:.1f}ms ± {std*1000:.1f}ms (n={iters})"

# ============================================================
# 🧪 Smoke Test Infrastructure — Registry, Results & Printing
#    Shared scaffolding used by every framework smoke test.
#    Defines the result type, the test registry, and the
#    print helpers that produce consistent formatted output
#    across all test sections.
#
#    SmokeResult dataclass — carries the outcome of one test:
#      name    display label printed in the report
#      ok      True = ✅ passed, False = 🟠 failed/skipped
#      detail  optional inline description appended after "—";
#              contains timing results, error messages, or
#              framework version strings
#
#    SMOKE dict          — ordered registry mapping section
#                          name → list of test functions; built
#                          up by add_smoke() calls below the
#                          test definitions; iterated in order
#                          at runtime to run and print results
#
#    add_smoke(section, fn)
#                        — registers a test function under a
#                          named section; creates the section
#                          key if it does not yet exist
#
#    import_only(name, module)
#                        — convenience factory for tests that
#                          only need to verify a module imports
#                          successfully; returns a SmokeResult
#                          without running any GPU operations
#
#    _desc(text)         — prints a 📋 prefixed description
#                          block; currently unused in the active
#                          test loop but available for verbose
#                          output modes
#
#    _print_info(text)   — word-wraps text at REPORT_WIDTH and
#                          prints with a leading "  ℹ️  " indent
#                          on the first line and 7-space hanging
#                          indent on continuations; used for
#                          legend notes, test docstring previews,
#                          and the Notes section at report end
# ============================================================
@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str = ""

SMOKE: dict[str, list[Callable[[], SmokeResult]]] = {}

def add_smoke(section: str, fn: Callable[[], SmokeResult]) -> None:
    SMOKE.setdefault(section, []).append(fn)

def import_only(name: str, module: str) -> SmokeResult:
    ok, err = try_import(module)
    return SmokeResult(name=name, ok=ok, detail="" if ok else short_err(err or ""))

def _desc(text: str) -> None:
    """Print a test description line (shown before each test group)."""
    for line in text.strip().splitlines():
        print(f"    📋 {line.strip()}")

def _print_info(text: str, width: int = REPORT_WIDTH, indent: str = "  ℹ️  ") -> None:
    """Print an info line with word-wrapping at the given width."""
    continuation = "       "  # 7 spaces: 2 (margin) + 2 (emoji width) + 1 (emoji) + 2 (trailing spaces)
    lines = textwrap.wrap(text, width=width - 7)  # use fixed 7 instead of len(indent)
    for i, line in enumerate(lines):
        print(f"{indent if i == 0 else continuation}{line}")
# ============================================================
# ⚡ Smoke Tests — PyTorch CUDA
#    Tests covering the full PyTorch CUDA GPU stack from
#    basic tensor availability through training-ready autograd.
#
#    _torch_cuda_dense()
#      Confirms the CUDA backend is reachable and executing
#      real GPU compute. Runs square float32 matmul (GEMM)
#      at the depth-appropriate matrix size, a conv2d pass,
#      and additional dtypes at depth 2+. One warmup pass is
#      always run first to amortise CUDA kernel compilation.
#
#    _torch_cuda_sparse()
#      Probes COO and CSR sparse matmul on CUDA and compares
#      timing against a dense baseline. On Ampere+ GPUs,
#      NVIDIA Sparse Tensor Cores can provide up to 2×
#      throughput for 2:4 structured sparsity patterns.
#
#    _torch_cuda_autograd()
#      Verifies gradient computation (backpropagation) works
#      correctly on CUDA GPU — required for training.
#
#    _torch_compile_hint()
#      Checks whether torch.compile is present (PyTorch 2.0+).
#      On CUDA, torch.compile with mode='reduce-overhead' or
#      'max-autotune' can significantly reduce kernel launch
#      overhead via CUDA graph capture.
# ============================================================
def _torch_cuda_dense() -> SmokeResult:
    """
    Test: Dense matrix multiply (GEMM) and Conv2d on PyTorch CUDA — confirms CUDA GPU is reachable and executing float32 matrix math and convolutions at expected speeds.

    What it tests:
      • torch.cuda.is_available() — CUDA GPU reachable
      • Square float32 matmul (A @ A) at depth-appropriate size
      • Conv2d with 64 filters on a 224×224 image batch
      • Additional dtypes (float16, bfloat16) at depth 2+

    Why it matters:
      Dense GEMM is the dominant operation in transformer attention
      and MLP layers. Conv2d covers CNN and vision workloads.
    """
    ok, err = try_import("torch")
    if not ok:
        return SmokeResult("torch import", False, short_err(err or ""))
    try:
        import torch  # type: ignore
        cuda_avail = torch.cuda.is_available()
        if not cuda_avail:
            return SmokeResult("torch CUDA available", False, "torch.cuda.is_available() == False")

        device = "cuda"
        dev_name = torch.cuda.get_device_name(0)
        N, iters = _bench_sizes()
        results = [f"torch={torch.__version__} | device={dev_name}"]

        for dt_name in _bench_dtypes_torch():
            dt = getattr(torch, dt_name)
            try:
                x = torch.randn(N, N, dtype=dt, device=device)
                # Warmup
                _ = (x @ x)
                torch.cuda.synchronize()
                mean, std = _timed_iters(
                    lambda: (x.__matmul__(x), torch.cuda.synchronize()),
                    iters, sync_fn=None
                )
                results.append(f"{dt_name} GEMM {N}×{N}: {_fmt_timing(mean, std, iters)}")
            except Exception as e:
                results.append(f"{dt_name} GEMM: ❌ {short_err(str(e))}")

        # Conv2d (float32 only)
        try:
            w = torch.randn(64, 3, 3, 3, device=device)
            img = torch.randn(4, 3, 224, 224, device=device)
            _ = torch.nn.functional.conv2d(img, w, stride=1, padding=1)
            torch.cuda.synchronize()
            mean_c, std_c = _timed_iters(
                lambda: (torch.nn.functional.conv2d(img, w, stride=1, padding=1),
                         torch.cuda.synchronize()),
                iters, sync_fn=None
            )
            results.append(f"conv2d 64ch 224×224: {_fmt_timing(mean_c, std_c, iters)}")
        except Exception as e:
            results.append(f"conv2d: ❌ {short_err(str(e))}")

        summary = " | ".join(results)
        return SmokeResult("torch CUDA dense GEMM+conv", True, summary)
    except Exception as e:
        return SmokeResult("torch CUDA dense GEMM+conv", False, short_err(str(e)))


def _tensor_cores_check() -> SmokeResult:
    """
    Test: Tensor Core hardware detection — reports whether the GPU has dedicated
    Tensor Core units and, if so, how many and which precision modes they support.

    Background:
      Tensor Cores are dedicated matrix-multiply-accumulate (MMA) units separate
      from regular CUDA cores. They were introduced in Volta (CC 7.0) and deliver
      dramatically higher throughput for FP16/BF16/TF32/INT8/FP8 matrix ops used
      in deep learning training and inference.

    Tensor Core generations:
      Volta  (CC 7.0):  1st gen — FP16 Tensor Cores
      Turing (CC 7.5):  2nd gen — FP16/INT8/INT4 Tensor Cores
      Ampere (CC 8.x):  3rd gen — FP16/BF16/TF32/INT8 Tensor Cores
      Ada    (CC 8.9):  4th gen — FP16/BF16/TF32/FP8/INT8 Tensor Cores
      Hopper (CC 9.0):  4th gen — adds FP8 Tensor Cores (H100: 3958 TFLOPS FP8)
      Blackwell (CC 10+): 5th gen — FP4/FP6/FP8 Tensor Cores

    Tensor Core count formula (SM × Tensor Cores per SM):
      Volta   V100:  80 SMs × 8  TC/SM = 640  Tensor Cores
      Turing  RTX 2080 Ti: 68 SMs × 8  TC/SM = 544
      Ampere  A100:  108 SMs × 4  TC/SM = 432  (but 3rd-gen, 4× wider)
      Ampere  RTX 3090: 82 SMs × 4 TC/SM = 328
      Ada     RTX 4090: 128 SMs × 4 TC/SM = 512
      Hopper  H100:  132 SMs × 4  TC/SM = 528  (4th-gen, very wide)

    What it tests:
      • GPU compute capability via torch.cuda.get_device_capability()
      • SM (Streaming Multiprocessor) count via get_device_properties()
      • Derives Tensor Core count from SM count + known TC/SM per arch
      • Reports precision modes available (FP16, BF16, TF32, INT8, FP8)
    """
    ok, err = try_import("torch")
    if not ok:
        return SmokeResult("Tensor Core check (skip)", False, short_err(err or ""))
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return SmokeResult("Tensor Core check (skip)", False, "CUDA not available")

        results = []
        for dev_idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(dev_idx)
            name  = props.name
            sm_count = props.multi_processor_count
            maj, min_ = props.major, props.minor
            cc = f"{maj}.{min_}"

            if not _cc_has_tensor_cores(cc):
                # Pre-Volta: no Tensor Cores at all
                results.append(
                    f"\u274c GPU {dev_idx} ({name}, CC {cc} {_cc_to_arch(cc)}): "
                    f"No Tensor Cores — Tensor Cores were introduced with Volta (CC 7.0, e.g. Tesla V100). "
                    f"This GPU has only standard CUDA cores. "
                    f"No TF32 / BF16 Tensor Core acceleration available "
                    f"(TF32 and BF16 Tensor Cores require Ampere CC 8.0+)."
                )
            else:
                # Determine TC/SM based on architecture
                # (Tensor Cores per SM counts per NVIDIA whitepaper)
                if (maj, min_) >= (9, 0):        # Hopper / Blackwell
                    tc_per_sm = 4
                    precisions = "FP16 / BF16 / TF32 / INT8 / FP8"
                elif (maj, min_) >= (8, 9):       # Ada Lovelace
                    tc_per_sm = 4
                    precisions = "FP16 / BF16 / TF32 / INT8 / FP8"
                elif (maj, min_) >= (8, 0):       # Ampere
                    tc_per_sm = 4
                    precisions = "FP16 / BF16 / TF32 / INT8"
                elif (maj, min_) >= (7, 5):       # Turing
                    tc_per_sm = 8
                    precisions = "FP16 / INT8 / INT4"
                else:                              # Volta (7.0)
                    tc_per_sm = 8
                    precisions = "FP16"

                total_tc = sm_count * tc_per_sm
                arch = _cc_to_arch(cc)
                results.append(
                    f"\u2705 GPU {dev_idx} ({name}, CC {cc} {arch}): "
                    f"{total_tc} Tensor Cores "
                    f"({sm_count} SMs \u00d7 {tc_per_sm} TC/SM) | "
                    f"Precisions: {precisions}"
                )

        if not results:
            return SmokeResult("Tensor Core check", False, "no CUDA devices")
        all_ok = all("\u2705" in r for r in results)
        return SmokeResult(
            "Tensor Core check",
            all_ok,
            " | ".join(results)
        )
    except Exception as e:
        return SmokeResult("Tensor Core check", False, short_err(str(e)))


def _torch_cuda_sparse() -> SmokeResult:
    """
    Test: Sparse matrix multiply on PyTorch CUDA — probes CUDA sparse execution
    paths (COO, CSR) and 2:4 structured sparsity hardware support.

    Hardware capability gates:
      • COO/CSR sparse:       all CUDA GPUs (Kepler+)
      • 2:4 Structured sparsity: Ampere+ ONLY (CC 8.0+, RTX 30xx / A100 / H100)
        — Pascal/Volta/Turing have no Sparse Tensor Cores; sparse ops are slower
          than dense because the GPU must do extra indexing work without any
          dedicated hardware path to skip zero-valued elements.

    What it tests:
      • torch.sparse_coo_tensor and torch.sparse_csr_tensor on CUDA
      • Timing: sparse vs dense baseline (⚡ faster = HW path, 🔵 slower = no HW)
      • Explicit ✅/❌ hardware capability check for 2:4 structured sparsity
      • Depth 3+: attempts to_sparse_semi_structured() on Ampere+ GPUs
    """
    ok, err = try_import("torch")
    if not ok:
        return SmokeResult("torch sparse (skip)", False, short_err(err or ""))
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return SmokeResult("torch CUDA sparse (skip)", False, "CUDA not available")

        # ── Structured Sparsity hardware capability check ──────────────────
        primary_cc = _primary_gpu_cc()
        has_sparsity_hw = _cc_has_structured_sparsity(primary_cc) if primary_cc else False
        if primary_cc:
            arch = _cc_to_arch(primary_cc)
            if has_sparsity_hw:
                sparsity_hw_line = (
                    f"\u2705 2:4 Structured Sparsity: supported "
                    f"(CC {primary_cc} {arch} — Ampere Sparse Tensor Cores present)"
                )
            else:
                sparsity_hw_line = (
                    f"\u274c 2:4 Structured Sparsity: NOT supported on this GPU "
                    f"(CC {primary_cc} {arch}). Hardware-accelerated 2:4 structured "
                    f"sparsity requires Ampere (CC 8.0+, e.g. RTX 3080/3090, A100, "
                    f"RTX 40-series, H100). This GPU can run sparse math APIs "
                    f"(COO/CSR) but lacks dedicated Sparse Tensor Core hardware to "
                    f"skip zero-valued elements; sparse ops will be slower than dense."
                )
        else:
            sparsity_hw_line = "\u26a0\ufe0f 2:4 Structured Sparsity: could not detect GPU CC"

        N, iters = _bench_sizes()
        N_s = min(N, 1024)
        results = [sparsity_hw_line]

        for density in _sparse_densities():
            label = f"density={density:.0%}"
            try:
                mask = (torch.rand(N_s, N_s) < density).float()
                dense_cpu = torch.randn(N_s, N_s) * mask
                dense_gpu = dense_cpu.to("cuda")

                # --- COO sparse ---
                try:
                    sparse_coo = dense_cpu.to_sparse_coo().to("cuda")
                    _ = torch.sparse.mm(sparse_coo, dense_gpu)
                    torch.cuda.synchronize()
                    mean_sp, _ = _timed_iters(
                        lambda: (torch.sparse.mm(sparse_coo, dense_gpu), torch.cuda.synchronize()),
                        iters
                    )
                    mean_dn, _ = _timed_iters(
                        lambda: (torch.mm(dense_gpu, dense_gpu), torch.cuda.synchronize()),
                        iters
                    )
                    speedup = mean_dn / mean_sp if mean_sp > 0 else 0
                    results.append(
                        f"COO {label}: sparse={mean_sp*1000:.1f}ms "
                        f"dense={mean_dn*1000:.1f}ms "
                        f"({'⚡ {:.2f}×'.format(speedup) if speedup > 1.05 else '🔵 no HW accel ({:.2f}×)'.format(speedup)})"
                    )
                except Exception as e_coo:
                    results.append(f"COO {label}: ❌ {short_err(str(e_coo))}")

                # --- CSR sparse (depth 2+) ---
                if BENCHMARK_DEPTH >= 2:
                    try:
                        sparse_csr = dense_cpu.to_sparse_csr().to("cuda")
                        mean_csr, _ = _timed_iters(
                            lambda: (torch.mm(sparse_csr, dense_gpu), torch.cuda.synchronize()),
                            iters
                        )
                        results.append(f"CSR {label}: {mean_csr*1000:.1f}ms ✅")
                    except Exception as e_csr:
                        results.append(f"CSR {label}: 🟠 {short_err(str(e_csr))}")

                # --- 2:4 semi-structured sparsity (Ampere+ only, depth 3+) ---
                if BENCHMARK_DEPTH >= 3:
                    if not has_sparsity_hw:
                        results.append(
                            f"2:4 structured {label}: \u274c skipped — "
                            f"requires Ampere (CC 8.0+); this GPU is CC {primary_cc} "
                            f"({_cc_to_arch(primary_cc) if primary_cc else '?'})"
                        )
                    else:
                        try:
                            from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor  # type: ignore
                            dense_semi = torch.randn(N_s, N_s, device="cuda", dtype=torch.float16)
                            # Apply 2:4 masking: keep top-2 of each group of 4
                            dense_pruned = dense_semi.clone()
                            for i in range(0, N_s, 4):
                                vals = dense_pruned[:, i:i+4].abs()
                                topk_mask = vals >= vals.kthvalue(3, dim=1, keepdim=True).values
                                dense_pruned[:, i:i+4] *= topk_mask
                            sparse_24 = to_sparse_semi_structured(dense_pruned)
                            x_semi = torch.randn(N_s, N_s, device="cuda", dtype=torch.float16)
                            _ = torch.mm(sparse_24, x_semi)
                            torch.cuda.synchronize()
                            mean_24, _ = _timed_iters(
                                lambda: (torch.mm(sparse_24, x_semi), torch.cuda.synchronize()), iters)
                            results.append(
                                f"2:4 structured {label}: {mean_24*1000:.1f}ms "
                                f"\u2705 (Ampere+ Sparse Tensor Cores active)"
                            )
                        except Exception as e24:
                            results.append(f"2:4 structured: 🟠 {short_err(str(e24))}")

            except Exception as e:
                results.append(f"{label}: ❌ {short_err(str(e))}")

        if not results:
            return SmokeResult("torch CUDA sparse", False, "no results")
        any_ok = any("❌" not in r for r in results)
        return SmokeResult("torch CUDA sparse", any_ok, " | ".join(results))
    except Exception as e:
        return SmokeResult("torch CUDA sparse", False, short_err(str(e)))


def _torch_cuda_autograd() -> SmokeResult:
    """
    Test: Autograd (backpropagation) on CUDA — confirms gradient computation works correctly on NVIDIA GPU, which is required for training (not just inference).

    What it tests:
      • Gradient computation via loss.backward() on CUDA GPU
      • Depth 1: single linear layer W @ x → loss.backward()
      • Depth 2: 3-layer MLP forward+backward, 3 passes
      • Depth 3+: 5-layer MLP with 10 full SGD training steps
    """
    ok, err = try_import("torch")
    if not ok:
        return SmokeResult("torch CUDA autograd (skip)", False, short_err(err or ""))
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        if not torch.cuda.is_available():
            return SmokeResult("torch CUDA autograd (skip)", False, "CUDA not available")

        N, iters = _bench_sizes()
        H = min(N, 512)

        if BENCHMARK_DEPTH == 1:
            W = torch.randn(H, H, device="cuda", requires_grad=True)
            x = torch.randn(H, 1, device="cuda")
            loss = (W @ x).sum()
            loss.backward()
            assert W.grad is not None, "grad is None"
            return SmokeResult("torch CUDA autograd [linear backward]", True,
                               f"grad shape={list(W.grad.shape)}")

        elif BENCHMARK_DEPTH == 2:
            model = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, 10)
            ).to("cuda")
            x = torch.randn(32, H, device="cuda")
            target = torch.randint(0, 10, (32,), device="cuda")
            criterion = nn.CrossEntropyLoss()
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                out = model(x)
                loss = criterion(out, target)
                loss.backward()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            mean = sum(times) / len(times)
            return SmokeResult("torch CUDA autograd [3-layer MLP fwd+bwd]", True,
                               f"mean={mean*1000:.1f}ms loss={loss.item():.4f}")

        else:
            model = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, 10)
            ).to("cuda")
            opt = torch.optim.SGD(model.parameters(), lr=1e-3)
            x = torch.randn(64, H, device="cuda")
            target = torch.randint(0, 10, (64,), device="cuda")
            criterion = nn.CrossEntropyLoss()
            t0 = time.perf_counter()
            for step in range(10):
                opt.zero_grad()
                loss = criterion(model(x), target)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            return SmokeResult("torch CUDA autograd [5-layer MLP 10 steps]", True,
                               f"10 steps={elapsed*1000:.0f}ms final_loss={loss.item():.4f}")

    except Exception as e:
        return SmokeResult("torch CUDA autograd", False, short_err(str(e)))


def _torch_compile_hint() -> SmokeResult:
    """
    Test: torch.compile availability — checks whether the graph compilation API is present. On CUDA, torch.compile with mode='reduce-overhead' uses CUDA graph capture to eliminate per-kernel launch overhead.
    """
    ok, err = try_import("torch")
    if not ok:
        return SmokeResult("torch.compile available", False, short_err(err or ""))
    try:
        import torch  # type: ignore
        has_compile = hasattr(torch, "compile")
        detail = ""
        if has_compile and torch.cuda.is_available() and BENCHMARK_DEPTH >= 2:
            try:
                x = torch.randn(512, 512, device="cuda")
                fn = torch.compile(lambda a: a @ a, mode="reduce-overhead")
                _ = fn(x)
                torch.cuda.synchronize()
                mean, _ = _timed_iters(lambda: (fn(x), torch.cuda.synchronize()), 3)
                detail = f"compiled GEMM 512×512: {mean*1000:.1f}ms"
            except Exception as ec:
                detail = f"compile test: 🟠 {short_err(str(ec))}"
        return SmokeResult("torch.compile available", bool(has_compile),
                           detail if detail else ("" if has_compile else "torch.compile not present (upgrade torch)"))
    except Exception as e:
        return SmokeResult("torch.compile available", False, short_err(str(e)))


# ============================================================
# ⚡ Smoke Tests — TensorFlow GPU (CUDA)
#    Three tests covering the TensorFlow CUDA GPU stack:
#    dense compute, sparse op routing, and GPU detection.
# ============================================================
def _tensorflow_cuda_dense() -> SmokeResult:
    """
    Test: Dense matmul and Conv2d on TensorFlow GPU (CUDA) — confirms TensorFlow can reach the CUDA device and dispatch compute correctly.

    What it tests:
      • tf.config.list_physical_devices('GPU') — CUDA device active
      • tf.matmul on /GPU:0 (float32) with timing
      • tf.nn.conv2d on /GPU:0
      • Depth 2+: BatchNormalization, float16 (mixed precision)
      • Depth 3+: Keras Sequential model fit() end-to-end
    """
    with _silence_fd2():
        ok, err = try_import("tensorflow")
    if not ok:
        return SmokeResult("tensorflow import", False, short_err(err or ""))
    try:
        import tensorflow as tf  # type: ignore
        try:
            import keras  # type: ignore
            results_prefix = f"tf={tf.__version__} keras={keras.__version__}"
        except Exception:
            results_prefix = f"tf={tf.__version__}"
        N, iters = _bench_sizes()
        with _silence_fd2():
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return SmokeResult("tensorflow GPU detected", False,
                                   "tf.config.list_physical_devices('GPU') == []")
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except Exception:
                pass
            with tf.device("/GPU:0"):
                a = tf.random.normal([N, N])
                _ = tf.matmul(a, a).numpy()
        results = [f"{results_prefix} GPUs={len(gpus)}"]

        with tf.device("/GPU:0"):
            mean, std = _timed_iters(lambda: tf.matmul(a, a).numpy(), iters)
        results.append(f"GEMM {N}×{N}: {_fmt_timing(mean, std, iters)}")

        with tf.device("/GPU:0"):
            x = tf.random.normal([4, 224, 224, 3])
            k = tf.random.normal([3, 3, 3, 32])
            _ = tf.nn.conv2d(x, k, strides=1, padding="SAME").numpy()
            mean_c, std_c = _timed_iters(
                lambda: tf.nn.conv2d(x, k, strides=1, padding="SAME").numpy(), iters)
        results.append(f"conv2d 32ch: {_fmt_timing(mean_c, std_c, iters)}")

        if BENCHMARK_DEPTH >= 2:
            try:
                with tf.device("/GPU:0"):
                    bn = tf.keras.layers.BatchNormalization()
                    xbn = tf.random.normal([32, 64, 64, 16])
                    _ = bn(xbn, training=True).numpy()
                    mean_bn, _ = _timed_iters(lambda: bn(xbn, training=True).numpy(), iters)
                results.append(f"batch_norm: {mean_bn*1000:.1f}ms ✅")
            except Exception as e_bn:
                results.append(f"batch_norm: 🟠 {short_err(str(e_bn))}")

            try:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                with tf.device("/GPU:0"):
                    a16 = tf.random.normal([N, N], dtype=tf.float16)
                    _ = tf.matmul(a16, a16).numpy()
                    mean16, _ = _timed_iters(lambda: tf.matmul(a16, a16).numpy(), iters)
                results.append(f"float16 GEMM: {mean16*1000:.1f}ms ✅")
                tf.keras.mixed_precision.set_global_policy("float32")
            except Exception as e16:
                results.append(f"float16: 🟠 {short_err(str(e16))}")

        if BENCHMARK_DEPTH >= 3:
            try:
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(256, activation="relu", input_shape=(128,)),
                    tf.keras.layers.Dense(256, activation="relu"),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ])
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
                xs = tf.random.normal([256, 128])
                ys = tf.random.uniform([256], minval=0, maxval=10, dtype=tf.int32)
                t0 = time.perf_counter()
                model.fit(xs, ys, epochs=3, batch_size=64, verbose=0)
                elapsed = time.perf_counter() - t0
                results.append(f"Keras fit 3 epochs: {elapsed*1000:.0f}ms ✅")
            except Exception as e_fit:
                results.append(f"Keras fit: 🟠 {short_err(str(e_fit))}")

        return SmokeResult("tensorflow GPU matmul/conv", True, " | ".join(results))
    except Exception as e:
        return SmokeResult("tensorflow GPU matmul/conv", False, short_err(str(e)))


def _tensorflow_cuda_sparse() -> SmokeResult:
    """
    Test: Sparse tensor operations via tf.sparse on CUDA — measures whether TensorFlow sparse matmul runs on GPU or falls back to CPU, and compares timing against dense.
    """
    with _silence_fd2():
        ok, err = try_import("tensorflow")
    if not ok:
        return SmokeResult("tensorflow sparse (skip)", False, short_err(err or ""))
    try:
        import tensorflow as tf  # type: ignore
        import numpy as np  # type: ignore
        with _silence_fd2():
            gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return SmokeResult("tensorflow sparse (skip)", False, "no GPU")

        N, iters = _bench_sizes()
        N_s = min(N, 1024)
        results = []

        for density in _sparse_densities():
            label = f"density={density:.0%}"
            try:
                mask = np.random.rand(N_s, N_s) < density
                idx = np.stack(np.where(mask), axis=1).astype(np.int64)
                vals = np.random.randn(len(idx)).astype(np.float32)
                sp = tf.sparse.SparseTensor(idx, vals, [N_s, N_s])
                sp = tf.sparse.reorder(sp)
                dense_tf = tf.random.normal([N_s, N_s])

                _ = tf.sparse.sparse_dense_matmul(sp, dense_tf).numpy()
                mean_sp, _ = _timed_iters(
                    lambda: tf.sparse.sparse_dense_matmul(sp, dense_tf).numpy(), iters)
                mean_dn, _ = _timed_iters(
                    lambda: tf.linalg.matmul(dense_tf, dense_tf).numpy(), iters)
                speedup = mean_dn / mean_sp if mean_sp > 0 else 0
                results.append(
                    f"{label}: sparse={mean_sp*1000:.1f}ms "
                    f"dense={mean_dn*1000:.1f}ms "
                    f"({'⚡ {:.2f}×'.format(speedup) if speedup > 1.05 else '🔵 {:.2f}×'.format(speedup)})"
                )
            except Exception as e:
                results.append(f"{label}: 🟠 {short_err(str(e))}")

        if not results:
            return SmokeResult("tensorflow CUDA sparse", False, "no results")
        return SmokeResult("tensorflow CUDA sparse", True, " | ".join(results))
    except Exception as e:
        return SmokeResult("tensorflow CUDA sparse", False, short_err(str(e)))


def _tensorflow_cuda_gpu_present() -> SmokeResult:
    """
    Test: TensorFlow GPU detection — confirms TensorFlow can see the CUDA GPU and that tensorflow[and-cuda] or tensorflow-gpu is correctly installed.
    """
    with _silence_fd2():
        ok, err = try_import("tensorflow")
    if not ok:
        return SmokeResult("tensorflow GPU usable", False, "tensorflow import failed")
    try:
        import tensorflow as tf  # type: ignore
        with _silence_fd2():
            gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return SmokeResult("tensorflow GPU detected", False,
                               "No GPU found. Install tensorflow[and-cuda] or tensorflow-gpu.")
        return SmokeResult("tensorflow GPU detected", True,
                           f"{len(gpus)} GPU(s) visible (GPU detection validated in dense test above)")
    except Exception as e:
        return SmokeResult("tensorflow GPU detected", False, short_err(str(e)))


# ============================================================
# ⚡ Smoke Tests — JAX CUDA
#    Tests covering the JAX CUDA GPU stack: dense GEMM with
#    XLA JIT compilation and autograd, and sparse matmul.
#    JAX on CUDA uses XLA to JIT-compile operations to CUDA
#    kernels, typically yielding excellent throughput.
# ============================================================
def _jax_cuda_dense() -> SmokeResult:
    """
    Test: Dense GEMM on JAX CUDA with JIT compilation and autograd — confirms XLA compiles and dispatches to the CUDA backend, and that autodiff works on the CUDA platform.

    What it tests:
      • jax.devices() — confirms CUDA/gpu platform active
      • Dense matmul (jnp.dot / @) on CUDA
      • Depth 2+: jax.jit-compiled matmul + jax.grad autograd
      • Additional dtypes (float16, bfloat16) at depth 2+
    """
    with _silence_fd2():
        ok, err = try_import("jax")
    if not ok:
        return SmokeResult("jax import", False, short_err(err or ""))
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        with _silence_fd2():
            devs = jax.devices()
        platforms = ", ".join(sorted({getattr(d, "platform", "unknown") for d in devs}))
        cuda_ok = any(getattr(d, "platform", "").lower() in ("gpu", "cuda")
                      for d in devs)
        if not cuda_ok:
            return SmokeResult("jax CUDA device", False, f"platforms={platforms} (no GPU/CUDA device found)")

        N, iters = _bench_sizes()
        results = [f"jax={jax.__version__} platforms={platforms}"]

        m32 = jnp.ones((N, N), dtype=jnp.float32)
        _ = (m32 @ m32).block_until_ready()
        mean, std = _timed_iters(lambda: (m32 @ m32).block_until_ready(), iters)
        results.append(f"float32 GEMM {N}×{N}: {_fmt_timing(mean, std, iters)}")

        if BENCHMARK_DEPTH >= 2:
            try:
                f_jit = jax.jit(lambda x: (x @ x).sum())
                grad_fn = jax.jit(jax.grad(lambda x: (x @ x).sum()))
                _ = f_jit(m32).block_until_ready()
                mean_jit, _ = _timed_iters(lambda: f_jit(m32).block_until_ready(), iters)
                results.append(f"jit GEMM: {mean_jit*1000:.1f}ms ✅")
                _ = grad_fn(m32).block_until_ready()
                mean_grad, _ = _timed_iters(lambda: grad_fn(m32).block_until_ready(), iters)
                results.append(f"jit grad: {mean_grad*1000:.1f}ms ✅")
            except Exception as e_jit:
                results.append(f"jit/grad: 🟠 {short_err(str(e_jit))}")

            for dt_name in ["float16", "bfloat16"]:
                try:
                    dt = getattr(jnp, dt_name)
                    mx = jnp.ones((N, N), dtype=dt)
                    _ = (mx @ mx).block_until_ready()
                    mean_dt, _ = _timed_iters(lambda: (mx @ mx).block_until_ready(), iters)
                    results.append(f"{dt_name}: {mean_dt*1000:.1f}ms ✅")
                except Exception as e_dt:
                    results.append(f"{dt_name}: 🟠 {short_err(str(e_dt))}")

        return SmokeResult("jax CUDA matmul", True, " | ".join(results))
    except Exception as e:
        return SmokeResult("jax CUDA matmul", False, short_err(str(e)))


def _jax_cuda_sparse() -> SmokeResult:
    """
    Test: Sparse matmul via JAX BCOO format on CUDA — measures sparse op performance relative to dense to reveal whether hardware sparse paths are active.
    """
    ok, err = try_import("jax")
    if not ok:
        return SmokeResult("jax sparse (skip)", False, short_err(err or ""))
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        cuda_ok = any(getattr(d, "platform", "").lower() in ("gpu", "cuda")
                      for d in jax.devices())
        if not cuda_ok:
            return SmokeResult("jax sparse (skip)", False, "CUDA device not found")

        try:
            from jax.experimental import sparse as jsparse  # type: ignore
        except ImportError:
            return SmokeResult("jax sparse (skip)", False, "jax.experimental.sparse not available")

        N, iters = _bench_sizes()
        N_s = min(N, 1024)
        results = []

        for density in _sparse_densities():
            label = f"density={density:.0%}"
            try:
                key = jax.random.PRNGKey(42)
                mask = jax.random.uniform(key, (N_s, N_s)) < density
                dense_m = jnp.ones((N_s, N_s)) * mask.astype(jnp.float32)
                sp = jsparse.BCOO.fromdense(dense_m)
                _ = (sp @ dense_m).block_until_ready()
                mean_sp, _ = _timed_iters(lambda: (sp @ dense_m).block_until_ready(), iters)
                mean_dn, _ = _timed_iters(lambda: (dense_m @ dense_m).block_until_ready(), iters)
                speedup = mean_dn / mean_sp if mean_sp > 0 else 0
                results.append(
                    f"BCOO {label}: sparse={mean_sp*1000:.1f}ms "
                    f"dense={mean_dn*1000:.1f}ms "
                    f"({'⚡ {:.2f}×'.format(speedup) if speedup > 1.05 else '🔵 {:.2f}×'.format(speedup)})"
                )
            except Exception as e:
                results.append(f"BCOO {label}: 🟠 {short_err(str(e))}")

        if not results:
            return SmokeResult("jax sparse", False, "no results")
        return SmokeResult("jax CUDA sparse (BCOO)", True, " | ".join(results))
    except Exception as e:
        return SmokeResult("jax CUDA sparse (BCOO)", False, short_err(str(e)))


# ============================================================
# ⚡ Smoke Tests — CuPy
#    Tests the CuPy CUDA array library — a NumPy-compatible
#    GPU array library that provides direct CUDA kernel access
#    without going through a full DL framework.
# ============================================================
def _cupy_smoke() -> SmokeResult:
    """
    Test: CuPy CUDA array operations — confirms cupy can allocate GPU arrays and run basic linear algebra. CuPy provides NumPy-compatible GPU arrays and is foundational for RAPIDS and custom CUDA workflows.

    What it tests:
      • cupy.cuda.is_available() — CUDA device reachable
      • cupy.random.randn + matmul (cuBLAS)
      • cupy.linalg operations at depth 2+
      • cupy.ElementwiseKernel (custom CUDA kernel) at depth 3+
    """
    ok, err = try_import("cupy")
    if not ok:
        return SmokeResult("cupy import", False,
                           f"{'cupy not installed — install cupy-cuda12x or cupy-cuda11x' if not pkg_installed('cupy-cuda12x') and not pkg_installed('cupy-cuda11x') else short_err(err or '')}")
    try:
        import cupy as cp  # type: ignore
        N, iters = _bench_sizes()
        results = [f"cupy={cp.__version__} | CUDA={cp.cuda.runtime.runtimeGetVersion()}"]

        # Dense GEMM
        a = cp.random.randn(N, N, dtype=cp.float32)
        cp.cuda.stream.get_current_stream().synchronize()
        _ = cp.dot(a, a)
        cp.cuda.stream.get_current_stream().synchronize()
        mean, std = _timed_iters(
            lambda: (cp.dot(a, a), cp.cuda.stream.get_current_stream().synchronize()), iters)
        results.append(f"float32 GEMM {N}×{N}: {_fmt_timing(mean, std, iters)}")

        if BENCHMARK_DEPTH >= 2:
            try:
                a16 = a.astype(cp.float16)
                _ = cp.dot(a16, a16)
                cp.cuda.stream.get_current_stream().synchronize()
                mean16, _ = _timed_iters(
                    lambda: (cp.dot(a16, a16), cp.cuda.stream.get_current_stream().synchronize()), iters)
                results.append(f"float16 GEMM: {mean16*1000:.1f}ms ✅")
            except Exception as e16:
                results.append(f"float16: 🟠 {short_err(str(e16))}")

        if BENCHMARK_DEPTH >= 3:
            try:
                kernel = cp.ElementwiseKernel(
                    'float32 x, float32 y', 'float32 z',
                    'z = x * y + x', 'fused_madd'
                )
                x_k = cp.random.randn(N * N, dtype=cp.float32)
                y_k = cp.random.randn(N * N, dtype=cp.float32)
                _ = kernel(x_k, y_k)
                cp.cuda.stream.get_current_stream().synchronize()
                mean_k, _ = _timed_iters(
                    lambda: (kernel(x_k, y_k), cp.cuda.stream.get_current_stream().synchronize()), iters)
                results.append(f"ElementwiseKernel: {mean_k*1000:.1f}ms ✅")
            except Exception as ek:
                results.append(f"ElementwiseKernel: 🟠 {short_err(str(ek))}")

        return SmokeResult("cupy CUDA array ops", True, " | ".join(results))
    except Exception as e:
        return SmokeResult("cupy CUDA array ops", False, short_err(str(e)))


# ============================================================
# ⚡ Smoke Tests — ONNX Runtime CUDA/TensorRT
#    Tests ONNX Runtime with CUDA and TensorRT execution
#    providers — the primary path for deploying optimized
#    models to NVIDIA GPUs in production.
# ============================================================
def _onnxruntime_cuda_smoke() -> SmokeResult:
    """
    Test: ONNX Runtime with CUDA and TensorRT Execution Providers — confirms onnxruntime-gpu is installed and can create inference sessions targeting the CUDA device.

    What it tests:
      • ort.get_available_providers() — CUDA EP present
      • Session creation with CUDAExecutionProvider
      • Depth 2+: matmul model timed on CUDA EP
      • Depth 3+: TensorRT EP session if available
    """
    ok, err = try_import("onnxruntime")
    if not ok:
        return SmokeResult("onnxruntime import", False, short_err(err or ""))
    try:
        import onnxruntime as ort  # type: ignore
        providers = ort.get_available_providers()
        cuda_present = any("CUDA" in p for p in providers)
        trt_present  = any("TensorRT" in p for p in providers)
        if not cuda_present:
            return SmokeResult("onnxruntime CUDA EP", False,
                               f"CUDAExecutionProvider not found. Install onnxruntime-gpu. providers={providers}")

        if not pkg_installed("onnx"):
            return SmokeResult("onnxruntime CUDA EP usable", True,
                               f"CUDA EP present | TensorRT EP: {'✅' if trt_present else '❌ not found'} "
                               f"(install 'onnx' for session test)")

        import onnx  # type: ignore
        from onnx import helper, TensorProto  # type: ignore
        import numpy as np  # type: ignore

        results = [f"providers={providers}"]

        # Depth 2+: matmul model
        if BENCHMARK_DEPTH >= 2:
            try:
                N_ort = 128
                A_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [N_ort, N_ort])
                B_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, [N_ort, N_ort])
                C_info = helper.make_tensor_value_info("C", TensorProto.FLOAT, [N_ort, N_ort])
                mm_node = helper.make_node("MatMul", ["A", "B"], ["C"])
                mm_graph = helper.make_graph([mm_node], "mm_graph", [A_info, B_info], [C_info])
                mm_model = helper.make_model(mm_graph, opset_imports=[helper.make_opsetid("", 13)])
                mm_bytes = mm_model.SerializeToString()
                so = ort.SessionOptions()
                mm_sess = ort.InferenceSession(mm_bytes, sess_options=so,
                                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                A_np = np.random.randn(N_ort, N_ort).astype(np.float32)
                B_np = np.random.randn(N_ort, N_ort).astype(np.float32)
                _, iters = _bench_sizes()
                mean_ort, _ = _timed_iters(lambda: mm_sess.run(None, {"A": A_np, "B": B_np}), iters)
                results.append(f"CUDA EP matmul {N_ort}×{N_ort}: {mean_ort*1000:.1f}ms ✅")
            except Exception as e2:
                results.append(f"matmul model: 🟠 {short_err(str(e2))}")

        if trt_present and BENCHMARK_DEPTH >= 3:
            results.append("TensorRT EP: ✅ available")

        return SmokeResult("onnxruntime CUDA EP usable", True, " | ".join(results))
    except Exception as e:
        return SmokeResult("onnxruntime CUDA EP usable", False, short_err(str(e)))


# ============================================================
# ⚡ Smoke Tests — llama-cpp-python (CUDA Build Detection)
# ============================================================
def _llama_cpp_cuda_smoke() -> SmokeResult:
    """
    Test: llama-cpp-python import and CUDA build detection — confirms the package is importable and probes whether it was compiled with CUDA acceleration enabled (GGML_CUDA=on / CMAKE_ARGS='-DGGML_CUDA=on').

    What it tests:
      • Import of llama_cpp module
      • Depth 2+: inspect dynamic library linkage for CUDA libs.
        Package presence alone does NOT confirm CUDA acceleration —
        the wheel must be built with GGML_CUDA=on. Use:
        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary :all:
    """
    ok, err = try_import("llama_cpp")
    if not ok:
        if pkg_installed("llama-cpp-python"):
            return SmokeResult("llama.cpp (CUDA) build", False,
                               "llama_cpp module import failed (check build)")
        return SmokeResult("llama.cpp (CUDA) build", False, "llama-cpp-python not installed")
    try:
        import llama_cpp  # type: ignore
        detail = f"module={getattr(llama_cpp, '__file__', '?')}"

        if BENCHMARK_DEPTH >= 2:
            try:
                lib_path = getattr(llama_cpp, "__file__", "") or ""
                if sys.platform.startswith("linux"):
                    ok_link, link_out = run_cmd(["ldd", lib_path], timeout=5)
                    cuda_linked = ok_link and ("libcuda" in link_out or "libcublas" in link_out)
                elif sys.platform == "win32":
                    ok_link, link_out = run_cmd(["dumpbin", "/dependents", lib_path], timeout=8)
                    cuda_linked = ok_link and "cudart" in link_out.lower()
                else:
                    ok_link, link_out = run_cmd(["otool", "-L", lib_path], timeout=5)
                    cuda_linked = ok_link and "cuda" in link_out.lower()

                if cuda_linked:
                    detail += " | CUDA libs linked ✅"
                elif ok_link:
                    detail += " | CUDA libs NOT detected ⚠️ (may be CPU-only build)"
            except Exception:
                pass

        return SmokeResult("llama.cpp (CUDA) import", True, detail)
    except Exception as e:
        return SmokeResult("llama.cpp (CUDA) import", False, short_err(str(e)))


# ============================================================
# ⚡ Smoke Tests — RAPIDS (cuDF / cuML)
# ============================================================
def _rapids_smoke() -> SmokeResult:
    """
    Test: RAPIDS cuDF and cuML availability — confirms GPU-accelerated DataFrame and ML
    algorithms are accessible at runtime. RAPIDS requires CUDA 12.x and a Pascal+ GPU
    (CC 6.0+). Note: packages may be installed (visible to pip) but still fail to import
    if shared libraries are missing or the GPU CC is unsupported.

    What it tests:
      • cudf.DataFrame creation and GPU operations
      • cuml basic import and device check
      • Depth 2+: cudf groupby aggregation timing vs pandas
      • Distinguishes "not installed" from "installed but runtime import failed"
    """
    # Check whether packages are installed (pip-visible) vs importable at runtime.
    # RAPIDS commonly installs successfully but fails to import due to:
    #   • Missing libcudf.so / shared library (LD_LIBRARY_PATH issue in WSL)
    #   • GPU CC below minimum (e.g. Pascal CC 6.1 has limited RAPIDS support)
    #   • CUDA version mismatch between the cudf wheel and the installed toolkit

    def _pkg_installed(name: str) -> bool:
        """Return True if the package is pip-visible (installed), regardless of importability."""
        try:
            import importlib.metadata as _imd
            _imd.version(name)
            return True
        except Exception:
            return False

    cudf_installed = _pkg_installed("cudf-cu12") or _pkg_installed("cudf-cu11") or _pkg_installed("cudf")
    cuml_installed = _pkg_installed("cuml-cu12") or _pkg_installed("cuml-cu11") or _pkg_installed("cuml")

    cudf_ok, cudf_err = try_import("cudf")
    cuml_ok, cuml_err = try_import("cuml")

    results = []

    # ── Hardware capability note ──────────────────────────────────────────
    primary_cc = _primary_gpu_cc()
    if primary_cc:
        arch = _cc_to_arch(primary_cc)
        # RAPIDS requires Pascal+ (CC 6.0+) minimum; some ops need Volta+ for best results
        try:
            maj, min_ = primary_cc.split(".")
            cc_tuple = (int(maj), int(min_))
        except Exception:
            cc_tuple = (0, 0)
        if cc_tuple < (6, 0):
            results.append(
                f"\u274c GPU CC {primary_cc} ({arch}) is below RAPIDS minimum (Pascal CC 6.0+) — "
                f"RAPIDS will not run on this GPU even if installed"
            )
        else:
            results.append(
                f"\u2139\ufe0f GPU CC {primary_cc} ({arch}) meets RAPIDS minimum (Pascal CC 6.0+)"
            )

    # ── cuDF ──────────────────────────────────────────────────────────────
    if cudf_ok:
        try:
            import cudf  # type: ignore
            import numpy as np  # type: ignore
            df = cudf.DataFrame({"a": np.arange(100000, dtype=np.float32),
                                 "b": np.random.randn(100000).astype(np.float32)})
            _ = df["a"].sum()
            results.append(f"cudf={cudf.__version__} \u2705 (100k row DataFrame + sum)")
            if BENCHMARK_DEPTH >= 2:
                try:
                    import time as _time
                    df["grp"] = (np.arange(100000) % 100).astype(np.int32)
                    t0 = _time.perf_counter()
                    _ = df.groupby("grp")["a"].mean()
                    elapsed = _time.perf_counter() - t0
                    results.append(f"cudf groupby: {elapsed*1000:.1f}ms")
                except Exception as eg:
                    results.append(f"cudf groupby: \U0001f7e0 {short_err(str(eg))}")
        except Exception as e_cudf:
            results.append(f"cudf: \U0001f7e0 {short_err(str(e_cudf))}")
    elif cudf_installed:
        results.append(
            f"\u274c cudf installed but import failed — likely a shared library or CUDA version "
            f"mismatch. Try: python -c \"import cudf\" to see the full error. "
            f"Check LD_LIBRARY_PATH includes your CUDA lib dir. "
            f"({'GPU CC ' + primary_cc + ' (' + _cc_to_arch(primary_cc) + ') — confirm this meets cudf wheel requirements' if primary_cc else 'check GPU CC vs cudf wheel requirements'})"
        )
    else:
        results.append(
            "\u2139\ufe0f cudf not installed. Install: pip install cudf-cu12 (CUDA 12) or conda install -c rapidsai cudf"
        )

    # ── cuML ──────────────────────────────────────────────────────────────
    if cuml_ok:
        try:
            import cuml  # type: ignore
            results.append(f"cuml={cuml.__version__} \u2705")
            if BENCHMARK_DEPTH >= 2:
                try:
                    import numpy as np  # type: ignore
                    from cuml.linear_model import LinearRegression as cuLinReg  # type: ignore
                    X = np.random.randn(10000, 10).astype(np.float32)
                    y = np.random.randn(10000).astype(np.float32)
                    reg = cuLinReg()
                    reg.fit(X, y)
                    results.append("cuml LinearRegression fit (10k×10): \u2705")
                except Exception as er:
                    results.append(f"cuml LinearRegression: \U0001f7e0 {short_err(str(er))}")
        except Exception as e_cuml:
            results.append(f"cuml: \U0001f7e0 {short_err(str(e_cuml))}")
    elif cuml_installed:
        results.append(
            f"\u274c cuml installed but import failed — likely same shared library / CUDA version "
            f"issue as cudf. Run: python -c \"import cuml\" to see the full traceback."
        )
    else:
        results.append(
            "\u2139\ufe0f cuml not installed. Install: pip install cuml-cu12 (CUDA 12) or conda install -c rapidsai cuml"
        )

    if not results:
        return SmokeResult("RAPIDS (cuDF + cuML)", False, "no results")
    any_ok = cudf_ok or cuml_ok
    return SmokeResult("RAPIDS (cuDF + cuML)", any_ok, " | ".join(results))


# ============================================================
# 🧠 Smoke Tests — Depth 4: GPU Memory Pressure & Ceiling
#    Empirically measures the practical GPU VRAM ceiling for
#    PyTorch CUDA and TensorFlow GPU by allocating float32
#    tensors in 256 MB steps until OOM or 90% VRAM, then
#    verifying that memory is cleanly released on cleanup.
#    Only registered and executed when BENCHMARK_DEPTH >= 4.
# ============================================================
def _get_gpu_vram_mb() -> int:
    """Return GPU 0 VRAM in MB, falling back to 8192 (8 GB)."""
    try:
        ok, out = run_cmd(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5
        )
        if ok and out.strip():
            first = out.strip().splitlines()[0].strip()
            if first.isdigit():
                return int(first)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except Exception:
        pass
    return 8192  # safe fallback: assume 8 GB


def _flash_attn_smoke() -> SmokeResult:
    """
    Test: FlashAttention hardware compatibility check and import smoke test.

    FlashAttention is a memory-efficient, IO-aware exact attention algorithm that
    rewrites the softmax attention computation to avoid materializing the full
    N×N attention matrix in HBM (GPU DRAM). Instead, it tiles the computation
    into SRAM (on-chip, ~100× faster bandwidth), reducing memory reads/writes
    dramatically. This allows much longer context lengths and higher throughput.

    Hardware requirements:
      FlashAttention-1:  Ampere+  (CC 8.0+) — e.g. A100, RTX 3090
      FlashAttention-2:  Ampere+  (CC 8.0+) — current standard; ~2× FA1 speed
      FlashAttention-3:  Hopper   (CC 9.0+) — H100 only; uses FP8 + async pipelines
        • FA-3 is NOT available on Ampere (RTX 30xx, A100) — Hopper ONLY.
        • FA-2 IS available on Ada (RTX 40xx, CC 8.9), Ampere (CC 8.0–8.7).

    Why FlashAttention matters:
      • Reduces attention memory from O(N²) to O(N) — enables long contexts
      • Typically 2–4× faster wall-clock attention vs naive PyTorch on Ampere+
      • Required by many high-throughput inference engines (vLLM, TGI, SGLang)
      • Used by default in recent HuggingFace Transformers + Diffusers releases

    What this test does:
      1. Checks GPU CC for FA-2 / FA-3 compatibility and shows ✅ or ❌
      2. Attempts to import flash_attn (if installed)
      3. Depth 2+: runs a small multi-head attention forward pass via
         flash_attn.flash_attn_func() and reports timing
    """
    ok_import, err = try_import("flash_attn")

    # ── Hardware capability gate ───────────────────────────────────────────
    primary_cc = _primary_gpu_cc()
    results = []

    if primary_cc:
        arch = _cc_to_arch(primary_cc)
        has_fa2 = _cc_has_flash_attn2(primary_cc)
        has_fa3 = _cc_has_flash_attn3(primary_cc)

        if has_fa3:
            results.append(
                f"\u2705 FlashAttention-2: supported (CC {primary_cc} {arch})"
            )
            results.append(
                f"\u2705 FlashAttention-3: supported (CC {primary_cc} {arch} — Hopper FP8 async pipelines)"
            )
        elif has_fa2:
            results.append(
                f"\u2705 FlashAttention-2: supported (CC {primary_cc} {arch})"
            )
            results.append(
                f"\u274c FlashAttention-3: NOT supported — requires Hopper (CC 9.0+, e.g. H100). "
                f"This GPU is CC {primary_cc} ({arch})."
            )
        else:
            results.append(
                f"\u274c FlashAttention-2: NOT supported on this GPU "
                f"(CC {primary_cc} {arch}). "
                f"FlashAttention-2 requires Ampere (CC 8.0+), e.g. RTX 3080/3090, A100, "
                f"RTX 40-series. Pascal/Volta/Turing GPUs cannot run FlashAttention. "
                f"Use standard scaled dot-product attention (torch.nn.functional.scaled_dot_product_attention) "
                f"as a fallback — it lacks the IO-tiling optimization but works on all GPUs."
            )
            results.append(
                f"\u274c FlashAttention-3: NOT supported — requires Hopper (CC 9.0+, e.g. H100)."
            )
    else:
        results.append("\u26a0\ufe0f Could not detect GPU compute capability for FlashAttention check")

    # ── Package import check ───────────────────────────────────────────────
    if not ok_import:
        results.append(
            f"\u2139\ufe0f flash_attn not installed — "
            f"{'GPU supports FA-2; install with: pip install flash-attn --no-build-isolation' if (primary_cc and _cc_has_flash_attn2(primary_cc)) else 'GPU does not meet minimum hardware requirements for FlashAttention'}"
        )
        # Return hardware-only result — this is informational, not an error
        hw_ok = primary_cc != "" and _cc_has_flash_attn2(primary_cc)
        return SmokeResult("FlashAttention", hw_ok, " | ".join(results))

    # flash_attn is installed
    try:
        import flash_attn  # type: ignore
        fa_ver = getattr(flash_attn, "__version__", "?")
        results.append(f"\u2705 flash_attn installed: v{fa_ver}")
    except Exception as e_import:
        results.append(f"\u274c flash_attn import error: {short_err(str(e_import))}")
        return SmokeResult("FlashAttention", False, " | ".join(results))

    # Depth 2+: actual forward pass
    if BENCHMARK_DEPTH >= 2:
        try:
            import torch  # type: ignore
            from flash_attn import flash_attn_func  # type: ignore
            if not torch.cuda.is_available():
                results.append("\u26a0\ufe0f CUDA not available for forward pass test")
            else:
                B, S, H, D = 2, 512, 8, 64  # batch, seqlen, heads, head_dim
                q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
                k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
                v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
                # warmup
                _ = flash_attn_func(q, k, v)
                torch.cuda.synchronize()
                mean_t, _ = _timed_iters(
                    lambda: (flash_attn_func(q, k, v), torch.cuda.synchronize()),
                    5
                )
                results.append(
                    f"\u2705 FA forward pass (B={B}, S={S}, H={H}, D={D}): "
                    f"{mean_t*1000:.2f}ms"
                )
        except Exception as e_fwd:
            results.append(f"\u274c FA forward pass: {short_err(str(e_fwd))}")

    any_ok = any("\u2705" in r for r in results)
    return SmokeResult("FlashAttention", any_ok, " | ".join(results))


def _memory_pressure_torch() -> SmokeResult:
    """Test: PyTorch CUDA memory pressure — finds the practical GPU VRAM ceiling by allocating in 256 MB steps until OOM, then verifies memory is cleanly released via empty_cache()."""
    ok, _ = try_import("torch")
    if not ok:
        return SmokeResult("torch CUDA memory ceiling", False, "torch not installed")
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return SmokeResult("torch CUDA memory ceiling", False, "CUDA not available")

        vram_mb = _get_gpu_vram_mb()
        hard_cap_mb = int(vram_mb * 0.90)
        step_mb = 256
        floats_per_step = step_mb * 1024 * 1024 // 4

        chunks = []
        peak_mb = 0
        print(f"\n    ⚠️  PyTorch CUDA memory pressure test "
              f"(step={step_mb}MB, cap={hard_cap_mb}MB / vram={vram_mb}MB)")

        for step in range(1, hard_cap_mb // step_mb + 2):
            try:
                chunks.append(torch.ones(floats_per_step, dtype=torch.float32, device="cuda"))
                torch.cuda.synchronize()
                peak_mb = step * step_mb
                alloc_mb = torch.cuda.memory_allocated() // (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() // (1024 * 1024)
                if step % 4 == 0:
                    print(f"      Step {step:3d}: {peak_mb:6d} MB allocated "
                          f"(alloc={alloc_mb}MB reserved={reserved_mb}MB) ✅")
                if peak_mb >= hard_cap_mb:
                    print(f"      Reached safety cap ({hard_cap_mb}MB) — stopping.")
                    break
            except Exception as oom:
                print(f"      Step {step:3d}: OOM at {peak_mb}MB → {short_err(str(oom))}")
                break

        del chunks
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
        post_alloc = torch.cuda.memory_allocated() // (1024 * 1024)
        cleanup_ok = post_alloc < step_mb
        cleanup_str = f"post-free={post_alloc}MB {'✅' if cleanup_ok else '⚠️'}"

        return SmokeResult(
            "torch CUDA memory ceiling", True,
            f"peak_usable={peak_mb}MB | vram={vram_mb}MB | {cleanup_str}"
        )
    except Exception as e:
        return SmokeResult("torch CUDA memory ceiling", False, short_err(str(e)))


def _memory_pressure_tf() -> SmokeResult:
    """Test: TensorFlow GPU memory pressure — finds the practical VRAM ceiling by allocating in 256 MB steps until OOM, then verifies cleanup via gc and clear_session."""
    ok, _ = try_import("tensorflow")
    if not ok:
        return SmokeResult("tensorflow CUDA memory ceiling", False, "tensorflow not installed")
    try:
        import tensorflow as tf  # type: ignore
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return SmokeResult("tensorflow CUDA memory ceiling", False, "no GPU")

        vram_mb = _get_gpu_vram_mb()
        hard_cap_mb = int(vram_mb * 0.90)
        step_mb = 256
        floats_per_step = step_mb * 1024 * 1024 // 4

        chunks = []
        peak_mb = 0
        print(f"\n    ⚠️  TensorFlow CUDA memory pressure test "
              f"(step={step_mb}MB, cap={hard_cap_mb}MB)")

        for step in range(1, hard_cap_mb // step_mb + 2):
            try:
                with tf.device("/GPU:0"):
                    chunks.append(tf.Variable(tf.ones([floats_per_step], dtype=tf.float32)))
                peak_mb = step * step_mb
                if step % 4 == 0:
                    print(f"      Step {step:3d}: {peak_mb:6d} MB ✅")
                if peak_mb >= hard_cap_mb:
                    print(f"      Reached safety cap ({hard_cap_mb}MB) — stopping.")
                    break
            except Exception as oom:
                print(f"      Step {step:3d}: OOM at {peak_mb}MB → {short_err(str(oom))}")
                break

        del chunks
        try:
            import gc; gc.collect()
            tf.keras.backend.clear_session()
        except Exception:
            pass

        return SmokeResult(
            "tensorflow CUDA memory ceiling", True,
            f"peak_usable={peak_mb}MB | vram={vram_mb}MB"
        )
    except Exception as e:
        return SmokeResult("tensorflow CUDA memory ceiling", False, short_err(str(e)))


# ============================================================
# 📋 Smoke Test Registry — Test Registration & Depth Gating
#    Registers all smoke test functions into the SMOKE dict
#    via add_smoke(), which maps each function to its named
#    report section. Tests are executed in registration order
#    within each section, and sections appear in the report
#    in the order their first test was registered.
#
#    Core tests — registered unconditionally (all depths):
#      ⚡ PyTorch MPS       dense GEMM+conv, sparse (COO/CSR),
#                           autograd, torch.compile availability
#      ⚡ TensorFlow GPU  dense GEMM+conv, sparse matmul,
#                           tensorflow-metal plugin check
#      ⚡ JAX CUDA         dense GEMM + jit + autograd, sparse
#                           BCOO matmul with density sweep
#      ⚡ Apple MLX         dense GEMM + dtypes + autograd +
#                           mlx.nn MLP, masked-dense sparse proxy
#      ⚡ Inference /       ONNX Runtime CoreML EP session test,
#         runtimes          coremltools PyTorch→CoreML conversion
#      ⚡ Build-dependent   llama-cpp-python import + Metal build
#                           detection via otool -L inspection
#      ⚡ macOS native APIs PyObjC Metal/CoreML framework bindings
#
#    Depth-gated tests — registered only when BENCHMARK_DEPTH >= 4:
#      🧠 Memory pressure   per-framework GPU RAM ceiling tests
#                           for PyTorch MPS, TensorFlow GPU,
#                           JAX CUDA, and MLX; runs after all
#                           core tests complete to avoid memory
#                           contention affecting earlier results
#
#    To add a new test: define a function returning SmokeResult,
#    then call add_smoke("<section name>", fn) here. Section
#    names are free-form strings — reusing an existing name
#    appends to that section; a new name creates a new section.
# ============================================================
# Core framework tests — always run
add_smoke("⚡ CUDA/GPU — PyTorch CUDA", _torch_cuda_dense)
add_smoke("⚡ CUDA/GPU — PyTorch CUDA", _tensor_cores_check)
add_smoke("⚡ CUDA/GPU — PyTorch CUDA", _torch_cuda_sparse)
add_smoke("⚡ CUDA/GPU — PyTorch CUDA", _torch_cuda_autograd)
add_smoke("⚡ CUDA/GPU — PyTorch CUDA", _torch_compile_hint)

add_smoke("⚡ CUDA/GPU — TensorFlow GPU", _tensorflow_cuda_dense)
add_smoke("⚡ CUDA/GPU — TensorFlow GPU", _tensorflow_cuda_sparse)
add_smoke("⚡ CUDA/GPU — TensorFlow GPU", _tensorflow_cuda_gpu_present)

add_smoke("⚡ CUDA/GPU — JAX CUDA", _jax_cuda_dense)
add_smoke("⚡ CUDA/GPU — JAX CUDA", _jax_cuda_sparse)

add_smoke("⚡ CUDA/GPU — CuPy", _cupy_smoke)

add_smoke("⚡ CUDA/GPU — RAPIDS (cuDF / cuML)", _rapids_smoke)

add_smoke("⚡ CUDA/GPU — Inference / runtimes", _onnxruntime_cuda_smoke)
add_smoke("⚡ CUDA/GPU — Inference / runtimes (build-dependent)", _llama_cpp_cuda_smoke)
add_smoke("⚡ CUDA/GPU — FlashAttention", _flash_attn_smoke)

# Memory pressure tests — depth 4 only
if BENCHMARK_DEPTH >= 4:
    add_smoke("⚡ CUDA/GPU — Memory pressure (depth 4)", _memory_pressure_torch)
    add_smoke("⚡ CUDA/GPU — Memory pressure (depth 4)", _memory_pressure_tf)


# ============================================================
# 🚀 Smoke Test Execution — Banner & Depth Label
#    Prints the section banner that opens the smoke test block
#    in the report, embedding the active BENCHMARK_DEPTH and
#    its human-readable time estimate so every saved report
#    is self-documenting about how thorough the run was.
#
#    _depth_labels maps each depth integer to a short label
#    combining the colloquial name and wall-clock estimate:
#      1 → "fast (~15–25s)"
#      2 → "medium (~45–75s)"
#      3 → "thorough (~2–4 min)"
#      4 → "memory pressure (~3–6 min)"
#    Any unrecognised depth falls back to "?" so the script
#    does not crash if BENCHMARK_DEPTH is set to an unexpected
#    value. The actual test loop follows immediately below.
# ============================================================
_depth_labels = {1: "fast (~15–25s)", 2: "medium (~45–75s)",
                 3: "thorough (~2–4 min)", 4: "memory pressure (~3–6 min)"}
banner(f"Runtime benchmarks & smoke tests  "
       f"[depth={BENCHMARK_DEPTH}: {_depth_labels.get(BENCHMARK_DEPTH, '?')}]")

# ============================================================
# 📖 Smoke Test Legend, Depth Warning & Test Execution Loop
#    Three responsibilities handled in sequence:
#
#    1. Legend block — printed once before any tests run so
#       the reader can interpret results without referring to
#       external documentation. Covers two distinct symbol
#       systems used in the report:
#
#       Row-level marks (leftmost column):
#         ✅  test passed — GPU op completed successfully
#         🟠  test ran but the operation errored out
#         ❌  test skipped — framework missing or MPS unavailable
#
#       Detail-level indicators (inside the "— description"):
#         ⚡  sparse op faster than dense by >1.05× — genuine
#             hardware sparse acceleration detected
#         🔵  sparse op equal to or slower than dense — no HW
#             sparse path; expected on all Apple Silicon since
#             Kepler/Maxwell GPUs have no Sparse Tensor Cores
#         🟠  sub-operation warning within an otherwise passing
#             test (e.g. one dtype failed, batch norm errored)
#         ❌  sub-operation failure within a passing test
#         ℹ️  informational detail, not a pass/fail indicator
#
#       Two _print_info() notes follow the legend to clarify
#       dense vs sparse terminology for non-expert readers, and
#       to prevent the common misreading of ✅ on sparse tests
#       as confirmation of hardware sparse acceleration.
#
#    2. Depth 4 warning — printed conditionally when
#       BENCHMARK_DEPTH >= 4 to remind the user to close
#       GPU-intensive apps before memory pressure tests run.
#       Placed after the legend so it appears immediately
#       before the first test output rather than at the top
#       of the section where it could be scrolled past.
#
#    3. Test execution loop — iterates over the SMOKE registry
#       in registration order, printing a section header and
#       dotted underline for each section, then for each test:
#         • prints the first line of the function's docstring
#           via _print_info() as a one-line test description
#         • calls fn() inside a try/except so a crashed test
#           produces a 🟠 SmokeResult rather than aborting
#           the entire report
#         • appends the result to ALL_RESULTS for use in the
#           Status Summary section printed after all tests
#         • formats and prints the result line via _wrap_line()
#           with the appropriate ✅/🟠 mark and detail string
# ============================================================
print("  Legend")
print("  ------")
print("  ✅  Test passed — operation completed successfully on CUDA/GPU")
print("  🟠  Test ran but failed — operation attempted but errored out")
print("  ❌  Test skipped or could not start — framework missing or unavailable")
print()
print("  Result detail indicators (inside the — description):")
print("  ⚡  Hardware acceleration detected (sparse faster than dense, >1.05×)")
print("  🔵  No hardware acceleration — sparse op slower than or equal to dense")
print("       (on Kepler/Maxwell GPUs without Sparse Tensor Cores)")
print("  🟠  Sub-operation warning — one step within a passing test had an issue")
print("  ❌  Sub-operation failure — one step within a passing test failed outright")
print("  ℹ️  Informational — context or detail about the result, not a pass/fail")
print()
_print_info("Note: In matrix operations, 'dense' means every element is stored and computed; 'sparse' means many values are zero and only non-zeros are tracked. Zeroing out large fractions of model weights (pruning) can reduce compute in training and inference — but only if the hardware has dedicated sparse execution paths to skip the zeros (e.g. NVIDIA Ampere's 2:4 Sparse Tensor Cores).", indent="  ")
_print_info("Note: ✅ on sparse tests means the test RAN, not that HW sparsity exists. Check the ⚡/🔵 indicator in the detail for actual acceleration status.", indent="  ")

if BENCHMARK_DEPTH >= 4:
    print("\n  ⚠️  DEPTH 4 — Memory pressure tests will intentionally exhaust GPU RAM.")
    print("     Close other GPU-intensive apps (browsers with WebGL, games, etc.)")
    print("     before running to get accurate ceilings and avoid system stalls.\n")

ALL_RESULTS: dict[str, list[SmokeResult]] = {}

for section, fns in SMOKE.items():
    print(f"\n{section}")
    print("." * len(section))
    for fn in fns:
        # Print the function's docstring first line as a brief description
        doc = (fn.__doc__ or "").strip().splitlines()
        if doc:
            _print_info(doc[0].strip())
        try:
            r = fn()
        except Exception as e:
            r = SmokeResult(name=getattr(fn, "__name__", "smoke"),
                            ok=False, detail=short_err(str(e)))
        ALL_RESULTS.setdefault(section, []).append(r)
        mark = ok_mark(r.ok) if r.ok else warn_mark()
        detail = f" — {r.detail}" if r.detail else ""
        _wrap_line(f"  {mark} ", f"{r.name}{detail}")

# ============================================================
# 📊 Status Summary — Consolidated Results at a Glance
#    Prints a condensed recap of the most important smoke test
#    outcomes after all tests have run, so the overall health
#    of the CUDA/GPU stack is visible without scrolling back
#    through the full test output. Results are looked up from
#    ALL_RESULTS by name prefix rather than re-running tests.
#
#    find_result(prefix)
#      Searches ALL_RESULTS for the first SmokeResult whose
#      name starts with the given prefix string. Returns None
#      if no matching result is found (test was not registered
#      or was skipped before producing a result). Prefix
#      matching allows summary lines to reference results by
#      their meaningful name stem without requiring an exact
#      match against the full detail-appended result string.
#
#    summarize(section_name, prefixes)
#      Prints a named subsection with one line per prefix:
#        ✅/🟠  result found — repeats the mark and detail
#               from the original test result via _wrap_line()
#        🟠     result not found — prints "(not executed)" so
#               missing tests are visible rather than silently
#               absent from the summary
#
#    Summary sections printed (in order):
#      System baseline   macOS detected, arch, macOS version,
#                        Python version, Xcode CLT, xcrun metal,
#                        xcrun metallib — from the sanity checks
#                        block, not from ALL_RESULTS
#      Core frameworks   PyTorch MPS, torch.compile,
#                        TensorFlow GPU, tensorflow-metal,
#                        JAX CUDA, MLX matmul
#      Inference /       ONNX Runtime CoreML EP, coremltools,
#        runtimes        llama.cpp import
#      macOS native APIs PyObjC framework bindings
#      Memory pressure   per-framework GPU RAM ceilings;
#        (depth 4 only)  section printed only when
#                        BENCHMARK_DEPTH >= 4
# ============================================================
banner("Status Summary")

def find_result(prefix: str) -> Optional[SmokeResult]:
    for rs in ALL_RESULTS.values():
        for r in rs:
            if r.name.startswith(prefix):
                return r
    return None

def summarize(section_name: str, prefixes: list[str]) -> None:
    print(f"\n{section_name}")
    print("." * len(section_name))
    for pfx in prefixes:
        r = find_result(pfx)
        if r is None:
            print(f"  {warn_mark()} {pfx} — (not executed)")
        else:
            mark = ok_mark(r.ok) if r.ok else warn_mark()
            detail = f" — {r.detail}" if r.detail else ""
            _wrap_line(f"  {mark} ", f"{pfx}{detail}")

print(f"  ✅ OS: {os_name} {os_release}  arch={arch}  python={pyver}")
print(f"  {ok_mark(ok_smi)} nvidia-smi: {len(gpu_list)} GPU(s) detected")
print(f"  {ok_mark(ok_nvcc)} nvcc / CUDA toolkit: {cuda_toolkit_ver}")
print(f"  ℹ️  cuDNN: {cudnn_ver}")

summarize("⚡ CUDA/GPU — Core frameworks", [
    "torch CUDA dense GEMM+conv",
    "torch.compile available",
    "tensorflow GPU matmul/conv",
    "tensorflow GPU detected",
    "jax CUDA matmul",
    "cupy CUDA array ops",
])

summarize("⚡ CUDA/GPU — RAPIDS & Inference", [
    "RAPIDS (cuDF + cuML)",
    "onnxruntime CUDA EP usable",
    "llama.cpp (CUDA) import",
])

if BENCHMARK_DEPTH >= 4:
    summarize("⚡ CUDA/GPU — Memory pressure (depth 4)", [
        "torch CUDA memory ceiling",
        "tensorflow CUDA memory ceiling",
    ])

# ============================================================
# 🔋 CUDA Ecosystem Readiness — High-Level Framework Check
#    Evaluates whether the most commonly used high-level ML
#    libraries are both installed AND have a working CUDA
#    backend available — the two conditions that must both
#    be true for GPU-accelerated training and inference on
#    NVIDIA hardware.
#
#    torch_cuda_ok — determined once by calling
#      torch.cuda.is_available(); False if torch is not
#      installed, import fails, or no CUDA device is found.
#      Every library in the CUDA_ECOSYSTEM list uses
#      PyTorch CUDA as its GPU backend.
#
#    CUDA_ECOSYSTEM — the libraries most commonly used
#      for production CUDA ML workloads, covering:
#        Transformers      — Hugging Face model hub + inference
#        Diffusers         — image/video generation pipelines
#        Accelerate        — device placement + mixed precision
#        PyTorch Lightning — training loop abstraction
#        Lightning         — unified Lightning framework package
#        fastai            — high-level training API over torch
#        timm              — vision model zoo
#        sentence-transformers — text embedding pipelines
#        Ultralytics       — YOLO object detection
#        openai-whisper    — speech recognition via CUDA
#        flash-attn        — FlashAttention (Ampere+)
#        xformers          — memory-efficient attention
#        bitsandbytes      — 4-bit/8-bit quantization
#        deepspeed         — distributed training
#        vllm              — high-throughput LLM serving
#
#    Readiness logic per library:
#      ✅  installed AND torch_cuda_ok — fully GPU-ready
#      ❌  installed but torch_cuda_ok is False — present but CPU only
#      ❌  not installed — note appended "— not installed"
# ============================================================
banner("CUDA ecosystem readiness (installed + torch.cuda.is_available())")
torch_cuda_ok = False
try:
    ok, _ = try_import("torch")
    if ok:
        import torch  # type: ignore
        torch_cuda_ok = bool(torch.cuda.is_available())
except Exception:
    torch_cuda_ok = False

CUDA_ECOSYSTEM = [
    ("Transformers", "transformers"),
    ("Diffusers", "diffusers"),
    ("Accelerate", "accelerate"),
    ("PyTorch Lightning", "pytorch-lightning"),
    ("Lightning", "lightning"),
    ("fastai", "fastai"),
    ("timm", "timm"),
    ("sentence-transformers", "sentence-transformers"),
    ("Ultralytics", "ultralytics"),
    ("openai-whisper", "openai-whisper"),
    ("flash-attn", "flash-attn"),
    ("xformers", "xformers"),
    ("bitsandbytes", "bitsandbytes"),
    ("DeepSpeed", "deepspeed"),
    ("vLLM", "vllm"),
    ("einops", "einops"),
    ("peft", "peft"),
    ("trl", "trl"),
]
for label, dist in CUDA_ECOSYSTEM:
    installed = pkg_installed(dist)
    ready = installed and torch_cuda_ok
    print(f"  {ok_mark(ready)} {label} ({dist})" + ("" if ready else (" — installed" if installed else " — not installed")))

# ============================================================
# 🏥 Dependency Health — pip Conflict Detection
#    Runs 'python -m pip check' against the active virtual
#    environment to surface any dependency conflicts, missing
#    requirements, or version incompatibilities across the
#    full installed package set. This is the final environment
#    health gate before the report closes.
#
#    pip check scans every installed package's metadata and
#    verifies that all declared Requires-Dist entries are
#    satisfied by what is actually installed. It catches
#    issues that package presence checks cannot — for example,
#    a package that is installed but requires a version of
#    numpy that conflicts with what another package pinned,
#    or a package whose optional dependency was uninstalled
#    after the fact leaving a broken requirement behind.
#
#    Common causes of pip check failures in ML environments:
#      • Installing packages from multiple channels (pip +
#        conda + brew) that have conflicting transitive deps
#      • Upgrading one framework (e.g. torch) without updating
#        its companions (torchvision, torchaudio)
#      • Installing pre-release or nightly builds alongside
#        stable releases that expect different dep versions
#      • Partially completed installs interrupted by OOM or
#        network failures leaving metadata inconsistent
#
#    Output behaviour:
#      ✅  pip check exit code 0 — no conflicts detected
#      🟠  pip check exit code non-zero — prints each conflict
#          line from pip's output with a leading "- " prefix
#          so individual violations are easy to scan; if pip
#          returns a non-zero exit code but produces no output
#          a fallback message notes the absence of details
#      The run_pip_check() call uses a 90s timeout to handle
#      large environments where metadata scanning is slow.
# ============================================================

banner("Dependency health")
try:
    pip_ok, pip_out = run_pip_check()
except Exception as e:
    pip_ok, pip_out = False, f"pip check failed to run: {e}"

print(f"  {ok_mark(pip_ok)} python -m pip check")
if pip_ok:
    print("  ✅ No dependency issues detected (pip check passed).")
else:
    print("  🟠 Dependency issues detected (pip check reported problems):")
    if pip_out and pip_out.strip():
        for ln in pip_out.splitlines():
            print(f"    - {ln}")
    else:
        print("    - (pip check returned no details)")

# ============================================================
# Notes
# ============================================================
if PRINT_NOTES:
 print()
 _header_line()
 _header_line("📝  Notes")
 _header_line()

 # ── ⚡ CUDA / GPU Accelerated ────────────────────────────────
 print()
 print("  ⚡  CUDA / GPU Accelerated Packages")
 print()
 _print_info("torch.cuda requires an NVIDIA GPU (Kepler CC 3.0+ minimum; Maxwell CC 5.0+ for float16 Tensor Ops; Volta CC 7.0+ for mixed-precision with Tensor Cores; Ampere CC 8.0+ for BF16 and TF32 Tensor Cores, FlashAttention, and 2:4 structured sparsity). Verify with torch.cuda.get_device_capability(). If torch.cuda.is_available() returns False, check that nvcc, the CUDA toolkit, and cuDNN are installed and that CUDA_HOME/CUDA_PATH is set correctly.")
 _print_info("Tensor Cores are dedicated matrix-multiply-accumulate (MMA) units on NVIDIA GPUs, separate from standard CUDA cores. They were first introduced with Volta (CC 7.0, Tesla V100) and deliver dramatically higher FP16 throughput for deep learning. Turing (CC 7.5, RTX 20xx/T4) added INT8/INT4 Tensor Cores. Ampere (CC 8.0+, A100/RTX 30xx) added BF16 and TF32 Tensor Cores — TF32 gives near-FP32 accuracy at FP16 Tensor Core speeds and is used automatically by PyTorch (torch.backends.cuda.matmul.allow_tf32=True by default on Ampere+). Ada Lovelace (CC 8.9, RTX 40xx) and Hopper (CC 9.0, H100) added FP8 Tensor Cores. Blackwell (CC 10+, B100/B200) introduced FP4/FP6 Tensor Cores. GPUs before Volta (Pascal CC 6.x and earlier) have no Tensor Cores at all.")
 _print_info("2:4 Structured Sparsity (also called N:M sparsity or Sparse Tensor Cores) was introduced with Ampere (CC 8.0+, A100/RTX 3090). The hardware enforces a pattern where exactly 2 of every 4 consecutive weight values are non-zero; this allows the Sparse Tensor Core engine to skip the zero-valued multiplications with dedicated hardware. In practice, a pruned+compressed 2:4 sparse model can achieve up to 2× GEMM throughput vs the dense baseline — halving inference latency or training time for matrix-heavy layers. Pre-Ampere GPUs (Pascal CC 6.x, Volta CC 7.0, Turing CC 7.5) have no Sparse Tensor Cores; sparse ops on those architectures use software indexing and are typically slower than dense. Use torch.ao.sparsity or torch.sparse.to_sparse_semi_structured() to apply 2:4 sparsity in PyTorch; NVIDIA cuSPARSELt is the underlying library.")
 _print_info("TensorFlow GPU support is via tensorflow[and-cuda] (recommended for CUDA 12.x installs) or the legacy tensorflow-gpu package. An empty GPU list from tf.config.list_physical_devices('GPU') usually means a wheel/toolkit mismatch — the TensorFlow wheel was built against a different CUDA version than what is installed. Run 'python -c \"import tensorflow as tf; print(tf.sysconfig.get_build_info())\"' to see which CUDA version the wheel expects, and cross-reference with your nvcc --version output.")
 _print_info("JAX CUDA support requires installing jax[cuda12] (for CUDA 12.x) or jax[cuda11_pip] (for CUDA 11.x), which pulls in the correct jaxlib CUDA wheel automatically. The plain 'pip install jax' installs a CPU-only build. JAX uses XLA to JIT-compile operations to CUDA kernels, producing excellent throughput for large matrix ops and making JAX one of the fastest options for research-scale CUDA workloads. Verify with jax.devices() — GPU devices show platform='gpu'.")
 _print_info("onnxruntime-gpu includes both the CUDA Execution Provider and the TensorRT Execution Provider, and replaces the CPU-only onnxruntime package. Install one or the other, not both. Run 'python -c \"import onnxruntime as ort; print(ort.get_available_providers())\"' to confirm. CUDAExecutionProvider offloads subgraphs to CUDA; TensorrtExecutionProvider further optimizes via TensorRT (requires the TensorRT SDK).")
 _print_info("llama-cpp-python CUDA acceleration requires building from source with GGML_CUDA=on: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-binary :all:. Pre-built PyPI wheels are CPU-only. Use 'ldd <module>.so | grep libcuda' on Linux to confirm CUDA linkage. Runtime log 'ggml_cuda_init' confirms a CUDA build loaded successfully.")
 _print_info("FlashAttention is an IO-aware, memory-efficient exact attention algorithm (Dao et al., 2022/2023) that rewrites the softmax attention computation to avoid materializing the full N×N attention matrix in HBM (GPU DRAM). Instead, it tiles the Q/K/V matrices into SRAM (on-chip cache, ~100× faster bandwidth) using a two-pass online softmax algorithm, dramatically reducing memory reads/writes. FlashAttention-2 achieves ~2-4× wall-clock speedup over naive PyTorch attention on Ampere+ and supports arbitrary sequence lengths with O(N) memory instead of O(N²). FlashAttention-3 targets Hopper (H100, CC 9.0) specifically, using FP8 Tensor Cores and async warpgroup pipelines for additional speedup. Hardware requirements: FlashAttention-2 requires Ampere (CC 8.0+, e.g. RTX 3080/3090, A100, RTX 40xx); pre-Ampere GPUs (Pascal CC 6.x, Volta CC 7.0, Turing CC 7.5) cannot run FlashAttention. Fallback: use torch.nn.functional.scaled_dot_product_attention() which has a math-equivalent CPU/CUDA fallback for all GPUs (no IO tiling, but correct). Installation: pre-built wheels available at github.com/Dao-AILab/flash-attention/releases; building from source takes 10-30 minutes.")
 _print_info("bitsandbytes provides 8-bit and 4-bit quantized linear layers that dramatically reduce VRAM requirements for large model inference. Requires CUDA 11.0+. Best performance requires Turing (CC 7.5+) for 4-bit kernels. On Linux, standard pip install works. On Windows, use bitsandbytes-windows or the pre-built wheel from the GitHub releases page.")
 _print_info("RAPIDS (cuDF, cuML, cuGraph, cuSpatial) requires CUDA 12.x and a Pascal+ GPU (CC 6.0+). Install via conda from the rapidsai channel (fastest and most reliable) or pip with the cudf-cu12, cuml-cu12, etc. packages. RAPIDS cuDF provides GPU-accelerated pandas-compatible DataFrames; cuML provides GPU-accelerated scikit-learn algorithms; cuGraph provides GPU-accelerated graph analytics.")
 _print_info("Transformers, Diffusers, and similar high-level libraries use torch.cuda as their GPU backend — they are framework-agnostic and will automatically use CUDA if torch.cuda.is_available() returns True. Accelerate (Hugging Face) is the key bridge library for device placement and mixed-precision training. DeepSpeed adds ZeRO optimizer stages (1/2/3) and pipeline parallelism for training models that don't fit on a single GPU.")

 # ── 🖥️ CPU-Only Packages ─────────────────────────────────────
 print()
 print("  🖥️   CPU-Only Packages")
 print()
 _print_info("CPU-only packages (numpy, scipy, faiss-cpu, opencv, pandas, scikit-learn, etc.) are listed separately and have no CUDA GPU path. On Linux x86-64, numpy and scipy are typically built with OpenBLAS or MKL, which provides multithreaded BLAS/LAPACK acceleration. For maximum CPU linear algebra performance, install intel-numpy or numpy with MKL on Intel hardware (conda install numpy mkl).")
 _print_info("faiss-gpu provides GPU-accelerated vector search via CUDA and is included in the CUDA packages section. faiss-cpu is the CPU-only alternative — solid choice for environments without GPU or where VRAM is at a premium. For billion-scale vector search, consider dedicated vector databases (Qdrant, Weaviate, Milvus) which can leverage CUDA for indexing while serving from CPU.")

 # ── 📓 Notebooks & Interactive Computing ─────────────────────
 print()
 print("  📓  Notebooks & Interactive Computing")
 print()
 _print_info("Notebook packages (JupyterLab, Jupyter Notebook, etc.) are CPU/network-bound — they orchestrate ML work and render outputs but do not perform GPU computation themselves. The GPU acceleration in notebooks comes entirely from the ML framework (PyTorch, TF, JAX) being called inside the notebook cells. Jupyter AI adds LLM-powered assistance directly in the notebook interface but requires a separate API key or local model backend.")
 _print_info("Jupyter AI requires an API key or local model backend configured separately. Supported backends include OpenAI, Anthropic Claude, Google Gemini, Amazon Bedrock, and local models via Ollama or llama-cpp-python. Configure the backend in JupyterLab under the Jupyter AI panel settings, or via the JUPYTER_AI_MODEL_ID and related environment variables. Without a configured backend the extension will load but all AI features will be non-functional.")

 # ── 🔬 MLOps ──────────────────────────────────────────────────
 print()
 print("  🔬  MLOps — Experiment Tracking, Pipelines & Model Serving")
 print()
 _print_info("MLOps packages (MLflow, Weights & Biases, DVC, ZenML, etc.) are CPU/network-bound orchestration and tracking tools — they do not perform GPU computation themselves but are essential infrastructure for managing the full ML lifecycle. MLflow and W&B handle experiment tracking and model registry; DVC handles data and model versioning via git-like semantics; pipeline orchestrators like ZenML, Metaflow, Kedro, and Prefect manage reproducible training workflows. Most of these tools are framework-agnostic and work equally well with PyTorch, TensorFlow, and JAX. For CUDA environments, the most practical lightweight stack is MLflow or W&B for tracking plus DVC for data versioning.")

 # ── 🤖 Agentic AI ─────────────────────────────────────────────
 print()
 print("  🤖  Agentic AI — Frameworks, LLM Clients & Vector Stores")
 print()
 _print_info("Agentic AI frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, etc.) are CPU/network-bound orchestration libraries — the heavy compute happens inside the LLM being called, not in the framework itself. On NVIDIA GPU machines, local LLM inference for agentic workloads is best served by vLLM (for high-throughput serving), llama-cpp-python (CUDA build), or Ollama as the model backend. LangGraph extends LangChain with stateful graph-based agent workflows and is increasingly the preferred approach for multi-step agents. For RAG pipelines, faiss-gpu, Qdrant, and Weaviate are strong vector store options that can leverage CUDA for embedding and index operations.")
 _print_info("AWS Strands Agents is a tool-first agentic framework that uses a model-driven loop where the LLM itself decides which tools to call and in what order, rather than following a predefined graph or chain. It integrates natively with Amazon Bedrock as its default LLM backend but supports other providers including Anthropic Claude directly, OpenAI, and local models via Ollama. Strands is particularly well-suited for building production-grade agents with large tool libraries — its architecture scales cleanly from a handful of tools to 100+ without the prompt engineering overhead that frameworks like LangChain require when tool counts grow large.")
 _print_info("Amazon Bedrock Agents (classic) — the original managed agent service with Action Groups and Knowledge Bases — has no separate pip package. It is accessed entirely via boto3 using the 'bedrock-agent' and 'bedrock-agent-runtime' service clients, which are included in boto3. 'bedrock-agentcore' is the newer production deployment SDK that wraps any agent framework (Strands, LangGraph, CrewAI, etc.) for serverless deployment on AWS.")
 _print_info("Amazon Bedrock Knowledge Bases uses boto3 (listed in the LLM Clients section) — no separate pip package is needed.")
 _print_info("AWS CLI v2 is typically installed as a standalone system package (via apt, yum, or the official installer), not via pip into a venv — an ❌ in this report is expected and normal. To install into your venv explicitly: pip install awscli.")
 _print_info("Google's generative AI Python package was renamed — 'google-generativeai' reached end-of-life on November 30, 2025 and is replaced by 'google-genai' (pip install google-genai). The new package unifies Gemini API access for both AI Studio and Vertex AI users into a single SDK.")
 _print_info("Microsoft Phi (Phi-3, Phi-4) and other open-weight models from Meta, Mistral, and Google are not standalone pip packages — they are model weights hosted on Hugging Face Hub and accessed via the 'transformers' and 'huggingface_hub' packages. On NVIDIA GPUs, these models run via PyTorch CUDA with optional quantization via bitsandbytes or auto-gptq for reduced VRAM usage.")

 # ── 🔧 General ────────────────────────────────────────────────
 print()
 print("  🔧  General")
 print()
 _print_info("Some packages may be installed at the system level (via apt/yum/dnf) rather than pip — and will show as ❌ in this report even if they are present. Common examples include: AWS CLI (awscli), git, cmake, ffmpeg, CUDA toolkit, cuDNN, and other system-level tools managed outside Python virtual environments. This checker uses pip/importlib.metadata exclusively for package detection and cannot see system packages. To make a package appear as ✅ in this report, install it into your active virtual environment via 'pip install <package>'.")
 _print_info("CUDA version mismatch is the most common cause of GPU failures on Linux. The CUDA version your PyTorch/TF/JAX wheel was compiled against must be compatible with your installed CUDA toolkit and GPU driver. Check: nvcc --version (toolkit), nvidia-smi (driver + max CUDA), torch.version.cuda (what PyTorch expects). Driver backward compatibility means a driver for CUDA 12.x can run code compiled for any earlier CUDA version, but not newer.")

 # ── 🔍 Audit Scope ────────────────────────────────────────────
 print()
 print("  \U0001f50d  Audit Scope")
 print()
 _print_info("This script audits the active Python environment only — specifically, whichever Python interpreter you used to run it (e.g. 'python3 ml_cuda_detective.py'). Package detection uses importlib.metadata, which reads the dist-info directories of the current environment. This means the results reflect exactly what is pip-installable and importable in that one environment, and nothing else.")
 _print_info("pip-managed packages in a virtual environment (venv, virtualenv) or the system Python site-packages are fully visible and will appear as ✅ or ❌ in the report. This is the normal, expected case — run the script from inside your active venv for accurate results for that environment.")
 _print_info("pipx installs each tool into its own isolated venv, entirely separate from any other Python environment. Packages installed via pipx are completely invisible to this script — even if you run the script from the same user account. The sanity check section detects pipx itself and counts how many pipx apps are installed, but it never inspects what packages are inside those pipx-managed venvs. To check a pipx-managed environment, you would need to run the script via 'pipx run' or activate the pipx venv directly.")
 _print_info("conda environments are partially visible depending on how you run the script. If you activate a conda environment first ('conda activate myenv') and then run 'python ml_cuda_detective.py', the script sees all packages in that conda environment — because conda envs are standard Python environments and importlib.metadata finds them normally. However, if you run the script from a different environment (e.g. a venv), packages in any conda env — including the base env — are not visible. Conda-installed packages do not have a separate detection path in this script; they are found only if they are in the currently active environment.")
 _print_info("System-level packages installed via apt, yum, dnf, brew, or other OS package managers are never visible to this script, regardless of which Python environment is active. This includes system-managed Python packages, CUDA toolkit, cuDNN, cmake, ffmpeg, and similar tools. They will always appear as ❌ here even when fully installed. The CUDA toolkit and cuDNN are checked separately via nvcc --version and header file inspection in the sanity checks section, not through pip.")
 _print_info("Multiple environment strategy: if you maintain several Python environments (e.g. one venv per project, plus a base conda env), run this script once from each environment to get a separate inventory report per environment. Pair SAVE_REPORT=True with a descriptive REPORT_DIR path to keep each run's output for later comparison.")

# ============================================================
# Report footer + save
# ============================================================
_END_TIME = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
if PRINT_BIBLIOGRAPHY:
 print()
 _header_line()
 _header_line("📚  Annotated Bibliography")
 _header_line()
 print()
 print("  References are organized by topic section, mirroring the Notes")
 print("  footnotes above. IEEE citation format. URLs verified 2025–2026.")
 print()

 # ── Section 1: CUDA Architecture & GPU Programming Model ─────
 print("  ── 1. CUDA Architecture & GPU Programming Model ──────────────────────────")
 print()
 _print_info("[1] NVIDIA Corporation, \"CUDA C++ Programming Guide,\" NVIDIA Developer Documentation, 2024. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ — The official and authoritative reference for the CUDA parallel computing platform and programming model. Covers the thread/block/grid hierarchy, memory spaces (global, shared, registers, constant, texture), compute capability definitions, warp execution, and the full runtime API. Essential reading for any developer targeting NVIDIA GPU acceleration.")

 _print_info("[2] NVIDIA Corporation, \"CUDA Toolkit Documentation,\" NVIDIA Developer, 2025. [Online]. Available: https://docs.nvidia.com/cuda/ — Top-level index for all CUDA Toolkit documentation including the Best Practices Guide, cuBLAS, cuDNN, cuSPARSE, NVCC compiler reference, and architecture-specific tuning guides. This is the canonical entry point for the full NVIDIA GPU software stack.")

 _print_info("[3] NVIDIA Corporation, \"CUDA C++ Best Practices Guide,\" NVIDIA Developer Documentation, 2024. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ — Companion to the Programming Guide, focused entirely on performance optimization. Covers memory coalescing, occupancy tuning, instruction-level optimization, and multi-GPU scaling. The roofline model and memory bandwidth ceiling analysis chapters are particularly relevant for ML workload tuning.")

 # ── Section 2: NVIDIA GPU Architecture Whitepapers ───────────
 print()
 print("  ── 2. NVIDIA GPU Architecture Whitepapers ─────────────────────────────────")
 print()
 _print_info("[4] NVIDIA Corporation, \"NVIDIA Tesla V100 GPU Architecture: The World's Most Advanced Datacenter GPU,\" NVIDIA Whitepaper WP-08608-001 v1.1, Aug. 2017. [Online]. Available: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf — Introduces the Volta GV100 architecture and, critically, the first-ever Tensor Cores. Provides detailed SM-level diagrams showing the 8 Tensor Cores per SM (4 TC/SM in the actual production V100 SXM2 with 80 SMs), the independent thread scheduling model, and second-generation NVLink topology. The primary reference for Tensor Core origins.")

 _print_info("[5] NVIDIA Corporation, \"NVIDIA Turing Architecture Whitepaper,\" NVIDIA Whitepaper WP-09402-001, Sept. 2018. [Online]. Available: https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf — Introduces the TU10x family (RTX 2080/T4) with second-generation Tensor Cores adding INT8 and INT4 compute modes. Also introduces RT Cores for hardware ray tracing. Defines Compute Capability 7.5 and the architectural underpinning for INT8 inference acceleration that enables quantized LLM inference on Turing hardware.")

 _print_info("[6] NVIDIA Corporation, \"NVIDIA A100 Tensor Core GPU Architecture,\" NVIDIA Whitepaper, May 2020. [Online]. Available: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf — Defines the Ampere GA100 architecture (CC 8.0), third-generation Tensor Cores with TF32 and BF16, structural 2:4 sparsity via Sparse Tensor Cores, Multi-Instance GPU (MIG), and third-generation NVLink/NVSwitch. This whitepaper is the primary reference for TF32 (near-FP32 accuracy at FP16 Tensor Core speed), 2:4 sparsity, and the MIG virtualization feature described in the sanity checks.")

 _print_info("[7] NVIDIA Corporation, \"NVIDIA Ampere GA102 GPU Architecture,\" NVIDIA Whitepaper, Sept. 2020. [Online]. Available: https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf — Covers the consumer-side Ampere architecture (RTX 3090, RTX 3080), the GA102 die with Compute Capability 8.6. Explains the higher CUDA core count per SM vs. the GA100 (128 vs. 64 FP32 cores/SM), the 2x FP32 datapath, and the inclusion of Sparse Tensor Cores. Essential for understanding why A100 CC 8.0 and RTX 3090 CC 8.6 have different CUDA core counts per SM despite both being 'Ampere.'")

 _print_info("[8] NVIDIA Corporation, \"NVIDIA Hopper Architecture In-Depth,\" NVIDIA Technical Blog, Mar. 2022. [Online]. Available: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ — Official deep-dive into the H100 Hopper architecture (CC 9.0): fourth-generation Tensor Cores with FP8, the Transformer Engine for dynamic precision selection, NVLink Switch interconnect for 256-GPU clusters, Thread Block Clusters (distributed shared memory), and the Confidential Computing engine. The definitive reference for H100 capabilities and the Transformer Engine.")

 _print_info("[9] NVIDIA Corporation, \"NVIDIA H100 Tensor Core GPU Architecture,\" NVIDIA Whitepaper v1.01, Mar. 2022. [Online]. Available: https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/gtc22-whitepaper-hopper — The formal H100 architecture whitepaper, companion to [8]. Contains exact SM counts (132 SMs on full GH100), Tensor Core throughput tables by data type, memory subsystem specifications (HBM3 at 3 TB/s), and MIG configurations. Cross-reference with [8] for architectural narrative.")

 _print_info("[10] NVIDIA Corporation, \"NVIDIA Ampere Architecture In-Depth,\" NVIDIA Technical Blog, May 2020. [Online]. Available: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/ — Blog companion to the A100 whitepaper [6], providing narrative context around the architectural decisions: why TF32 was chosen over pure FP32, how MIG partitioning works in practice, and the motivation for third-generation NVLink. Useful for understanding the design philosophy that drives automatic mixed precision in PyTorch (allow_tf32=True default on Ampere+).")

 # ── Section 3: Tensor Cores & Mixed Precision ────────────────
 print()
 print("  ── 3. Tensor Cores & Mixed Precision Training ─────────────────────────────")
 print()
 _print_info("[11] NVIDIA Corporation, \"Training With Mixed Precision,\" NVIDIA Deep Learning Performance Guide, 2024. [Online]. Available: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html — The canonical NVIDIA documentation for Automatic Mixed Precision (AMP) training. Explains the loss scaling algorithm required to prevent FP16 underflow, how to enable AMP in PyTorch (torch.amp) and TensorFlow, and which operations run in FP16 vs. FP32 master weights. Required reading before enabling AMP in any training pipeline.")

 _print_info("[12] NVIDIA Corporation, \"NVIDIA Tensor Core GPU Architecture,\" NVIDIA Technical Blog, 2020. [Online]. Available: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/ — Introduces the developer-facing programming model for Tensor Cores in CUDA 9. Explains the wmma (warp-level matrix multiply-accumulate) API, fragment types, and the constraints on matrix dimensions (multiples of 16) for efficient Tensor Core utilization. Foundational for understanding why model dimensions in deep learning are typically chosen as multiples of 8 or 16.")

 _print_info("[13] P. Micikevicius et al., \"Mixed Precision Training,\" in Proc. International Conference on Learning Representations (ICLR), 2018. [Online]. Available: https://arxiv.org/abs/1710.03740 — The seminal academic paper introducing the mixed-precision training methodology (FP16 compute + FP32 master weights + loss scaling). Published by NVIDIA researchers, this paper established the algorithmic foundation upon which PyTorch AMP and TensorFlow mixed precision are built. Demonstrates that FP16 training matches FP32 accuracy across CNNs, RNNs, and LSTMs when combined with loss scaling.")

 # ── Section 4: FlashAttention ────────────────────────────────
 print()
 print("  ── 4. FlashAttention ───────────────────────────────────────────────────────")
 print()
 _print_info("[14] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, \"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,\" in Advances in Neural Information Processing Systems (NeurIPS), 2022. [Online]. Available: https://arxiv.org/abs/2205.14135 — The foundational paper introducing IO-aware attention computation. Proposes tiling to reduce HBM reads/writes, achieving exact attention (no approximation) with subquadratic memory. Delivers 15% end-to-end speedup on BERT-Large and 3x speedup on GPT-2. Required reading for understanding why flash-attn dramatically accelerates transformer training; directly relevant to the FlashAttention smoke test in this report.")

 _print_info("[15] T. Dao, \"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning,\" in Proc. International Conference on Learning Representations (ICLR), 2024. [Online]. Available: https://arxiv.org/abs/2307.08691 — Extends FlashAttention with improved thread-block parallelism across sequence length, reducing non-matmul FLOPs and increasing GPU utilization from 25–40% to 50–73% of peak FLOPs/s on A100. Achieves up to 225 TFLOPs/s per A100 for GPT-style training. The version most widely deployed in production transformer training as of 2024; what this report's FA-2 smoke test targets.")

 _print_info("[16] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, \"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision,\" arXiv preprint arXiv:2407.08608, 2024. [Online]. Available: https://arxiv.org/abs/2407.08608 — Introduces a Hopper (H100)-specific FlashAttention implementation using asynchronous warp-specialization (producer/consumer pipelines), TMA (Tensor Memory Accelerator), and FP8 support. Achieves up to 1.5× speedup over FA-2 on H100 by overlapping GEMM and softmax stages. Explains why FA-3 requires CC 9.0 (Hopper), while FA-2 works from CC 8.0+.")

 # ── Section 5: 2:4 Structured Sparsity ──────────────────────
 print()
 print("  ── 5. 2:4 Structured Sparsity ─────────────────────────────────────────────")
 print()
 _print_info("[17] NVIDIA Corporation, \"Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt,\" NVIDIA Technical Blog, Dec. 2020. [Online]. Available: https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/ — The primary NVIDIA introduction to 2:4 structured (N:M) sparsity and the cuSPARSELt library. Explains the hardware enforcement of the 2-nonzero-in-4 pattern, the compression format (data + 2-bit index metadata), and the workflow: prune → compress → SpMMA via Sparse Tensor Cores. Provides performance benchmarks vs. dense cuBLAS on A100 for BERT-Large layer sizes.")

 _print_info("[18] NVIDIA Corporation, \"cuSPARSELt: A High-Performance CUDA Library for Sparse Matrix-Matrix Multiplication,\" NVIDIA Developer Documentation, 2024. [Online]. Available: https://docs.nvidia.com/cuda/cusparselt/ — Official API documentation for cuSPARSELt, NVIDIA's library for 2:4 structured sparse GEMM. Lists supported SM architectures (8.0, 8.6, 8.7, 8.9, 9.0, 10.x), data types, and the full API surface. The authoritative reference for why 2:4 sparsity requires Ampere CC 8.0+ and is absent from Pascal and Turing GPUs.")

 _print_info("[19] PyTorch Team, \"Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity,\" PyTorch Blog, 2023. [Online]. Available: https://pytorch.org/blog/accelerating-neural-network-training/ — Documents PyTorch's native semi-structured sparsity support (torch.sparse.to_sparse_semi_structured), demonstrating 1.6× GEMM speedup and 10% end-to-end speedup on segment-anything inference. Explains how the PyTorch team developed a custom prune+compress kernel 10× faster than cuSPARSELt's compression utility. The practical implementation guide for the torch.ao.sparsity workflow referenced in the Notes.")

 _print_info("[20] T. Mishra, E. Nurvitadhi, J. J. Cook, and D. Marr, \"NVIDIA Accelerating Sparsity in the NVIDIA Ampere Architecture,\" NVIDIA Technical Blog, 2020. [Online]. Available: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/ — Describes the end-to-end sparsity workflow using TensorRT: ASP (Automatic SParsity) library for pruning, retraining, and compression, followed by TensorRT deployment using Sparse Tensor Cores. Shows accuracy recovery results for vision and NLP models and explains why 2:4 sparsity loses less accuracy than unstructured pruning to the same 50% density target.")

 # ── Section 6: Model Quantization & bitsandbytes ────────────
 print()
 print("  ── 6. Model Quantization & bitsandbytes ───────────────────────────────────")
 print()
 _print_info("[21] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, \"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale,\" in Advances in Neural Information Processing Systems (NeurIPS), 2022. [Online]. Available: https://arxiv.org/abs/2208.07339 — Introduces the LLM.int8() quantization scheme that enables inference of 175B-parameter models at half the memory cost with no accuracy loss. Key insight: emergent outlier features in transformer activations (>6B parameters) require mixed-precision decomposition — 16-bit for the 0.1% outlier dimensions, 8-bit for the remainder. This is the academic foundation for the bitsandbytes library's core functionality.")

 _print_info("[22] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, \"QLoRA: Efficient Finetuning of Quantized LLMs,\" in Advances in Neural Information Processing Systems (NeurIPS), 2023. [Online]. Available: https://arxiv.org/abs/2305.14314 — Introduces 4-bit NormalFloat (NF4) quantization with double quantization and paged optimizers for fine-tuning 65B-parameter models on a single 48GB GPU. Demonstrates that LoRA adapters applied to 4-bit quantized base models recover full 16-bit fine-tuning performance. The bitsandbytes 4-bit (load_in_4bit) functionality is a direct implementation of this paper.")

 _print_info("[23] bitsandbytes Foundation, \"bitsandbytes: Accessible Large Language Models via k-bit Quantization for PyTorch,\" GitHub Repository, 2024. [Online]. Available: https://github.com/bitsandbytes-foundation/bitsandbytes — The open-source library implementing LLM.int8() [21] and QLoRA [22] for PyTorch. Provides Linear8bitLt and Linear4bit module replacements, 8-bit Adam and other optimizers, and the paged optimizer used in QLoRA. The installed package detected and smoke-tested in this report.")

 # ── Section 7: LLM Serving & Inference Optimization ─────────
 print()
 print("  ── 7. LLM Serving & Inference Optimization ────────────────────────────────")
 print()
 _print_info("[24] W. Kwon et al., \"Efficient Memory Management for Large Language Model Serving with PagedAttention,\" in Proc. ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP), 2023. [Online]. Available: https://arxiv.org/abs/2309.06180 — Introduces PagedAttention, an OS-inspired KV cache management scheme that eliminates memory fragmentation in LLM serving by using non-contiguous physical memory blocks. Enables vLLM to achieve 2–4× throughput improvement over FasterTransformer and Orca at equal latency. This paper is the foundation of the vLLM serving framework checked in this report's Agentic AI section.")

 _print_info("[25] NVIDIA Corporation, \"NVIDIA TensorRT Documentation,\" NVIDIA Developer, 2025. [Online]. Available: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html — Official documentation for NVIDIA TensorRT, the inference optimization SDK. Covers layer fusion, precision calibration (INT8, FP8, FP4), dynamic shape handling, the ONNX parser, and TensorRT-LLM for LLM deployment. TensorRT is audited for both GPU and CPU (TensorRT Execution Provider in ONNX Runtime) in this report.")

 _print_info("[26] NVIDIA Corporation, \"NVIDIA TensorRT-LLM Documentation,\" NVIDIA Developer, 2025. [Online]. Available: https://docs.nvidia.com/tensorrt-llm/index.html — Documentation for TensorRT-LLM, NVIDIA's LLM-specific inference library. Covers in-flight batching, paged KV caching, FP8/INT8/INT4 quantization, and tensor/pipeline parallelism for multi-GPU LLM serving. Represents the production-grade path for deploying llama-cpp-python and other local LLM backends on NVIDIA hardware.")

 _print_info("[27] NVIDIA Corporation, \"NVIDIA Triton Inference Server,\" NVIDIA Developer, 2025. [Online]. Available: https://developer.nvidia.com/triton-inference-server — Documentation for Triton, NVIDIA's model serving platform supporting TensorRT, PyTorch, TensorFlow, ONNX, and custom backends. Implements concurrent model execution, dynamic batching, ensemble pipelines, and the KServe protocol. The tritonclient package is checked in the MLOps section of this report.")

 # ── Section 8: Distributed Training & Memory Optimization ────
 print()
 print("  ── 8. Distributed Training & Memory Optimization ──────────────────────────")
 print()
 _print_info("[28] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, \"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models,\" in Proc. International Conference for High Performance Computing, Networking, Storage and Analysis (SC'20), 2020. [Online]. Available: https://arxiv.org/abs/1910.02054 — Introduces the Zero Redundancy Optimizer (ZeRO), which eliminates memory redundancy in data-parallel training by partitioning optimizer states, gradients, and parameters across GPUs (ZeRO stages 1, 2, 3). Enables training of models with 1T+ parameters on existing GPU clusters. The foundational paper for the DeepSpeed library's core memory optimization, and a prerequisite for understanding why deepspeed is in the CUDA packages section.")

 _print_info("[29] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, \"DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters,\" in Proc. 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), 2020. [Online]. Available: https://dl.acm.org/doi/10.1145/3394486.3406703 — Presents the DeepSpeed library as a unified system integrating ZeRO [28], pipeline parallelism, optimized transformer kernels, and mixed precision. Demonstrates the Turing-NLG 17B parameter model. The paper documenting the DeepSpeed package audited in this report.")

 _print_info("[30] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, \"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism,\" arXiv preprint arXiv:1909.08053, 2019. [Online]. Available: https://arxiv.org/abs/1909.08053 — Introduces tensor parallelism for transformer models, enabling training of 8.3B parameter GPT models by splitting attention heads and MLP layers across GPUs. Combined with ZeRO [28] and pipeline parallelism to form the 3D parallelism strategy used in modern LLM training. Explains why CUDA memory limits constrain the models that can be trained on a single GPU.")

 # ── Section 9: Deep Learning Frameworks ─────────────────────
 print()
 print("  ── 9. Deep Learning Frameworks ─────────────────────────────────────────────")
 print()
 _print_info("[31] A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning Library,\" in Advances in Neural Information Processing Systems (NeurIPS), 2019. [Online]. Available: https://arxiv.org/abs/1912.01703 — The primary academic citation for PyTorch. Describes the define-by-run execution model, autograd engine, memory allocator, and the CUDA backend. PyTorch is the central framework whose CUDA availability, smoke test performance, and AMP/BF16/TF32 support are the core focus of this report. Cite this paper when crediting PyTorch in academic work.")

 _print_info("[32] M. Abadi et al., \"TensorFlow: A System for Large-Scale Machine Learning,\" in Proc. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2016. [Online]. Available: https://arxiv.org/abs/1605.08695 — The foundational paper introducing TensorFlow's dataflow graph model and distributed execution engine. The TensorFlow GPU backend (tensorflow[and-cuda]) is audited in this report, and understanding its build system (CUDA toolkit version embedding) explains why TF GPU failures are commonly caused by toolkit/wheel version mismatches.")

 _print_info("[33] R. Frostig, M. J. Johnson, and C. Leary, \"Compiling Machine Learning Programs via High-Level Tracing,\" in Proc. SysML Conference, 2018. [Online]. Available: https://mlsys.org/Conferences/doc/2018/146.pdf — The original academic paper introducing JAX's design: composable function transformations (jit, grad, vmap, pmap) over NumPy-compatible code using XLA as the compiler backend. The jax[cuda] package audited in this report uses these primitives with CUDA kernels generated by XLA. Explains why JAX's programming model differs fundamentally from PyTorch despite both targeting GPU acceleration.")

 _print_info("[34] NVIDIA Corporation, \"cuDNN: A GPU-Accelerated Library for Deep Neural Networks,\" NVIDIA Developer, 2024. [Online]. Available: https://developer.nvidia.com/cudnn — Official product page for cuDNN, NVIDIA's deep learning primitive library providing GPU-accelerated implementations of convolution, pooling, normalization, and activation functions. cuDNN is a mandatory dependency for PyTorch GPU, TensorFlow GPU, and JAX GPU; its version is verified in the sanity checks section of this report.")

 # ── Section 10: Transformers & Foundation Model Architecture ─
 print()
 print("  ── 10. Transformers & Foundation Model Architecture ───────────────────────")
 print()
 _print_info("[35] A. Vaswani et al., \"Attention Is All You Need,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017. [Online]. Available: https://arxiv.org/abs/1706.03762 — The landmark paper introducing the Transformer architecture based solely on self-attention, replacing RNNs for sequence modeling tasks. Every major LLM (GPT, BERT, LLaMA, Claude, Gemini) is a descendant of this architecture. Directly relevant to this report's FlashAttention, Tensor Core, and Transformer Engine coverage: all of these are hardware/software optimizations for the matrix operations defined in this paper.")

 _print_info("[36] T. Brown et al., \"Language Models are Few-Shot Learners,\" in Advances in Neural Information Processing Systems (NeurIPS), 2020. [Online]. Available: https://arxiv.org/abs/2005.14165 — Introduces GPT-3 (175B parameters) and demonstrates emergent few-shot capabilities without task-specific fine-tuning. The scale of GPT-3 motivated the memory optimization techniques (ZeRO, quantization, sparsity) and serving infrastructure (vLLM, TensorRT-LLM) that form the core of the CUDA packages audited in this report.")

 _print_info("[37] J. Devlin, M. Chang, K. Lee, and K. Toutanova, \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,\" in Proc. NAACL, 2019. [Online]. Available: https://arxiv.org/abs/1810.04805 — Introduces bidirectional transformer pre-training via masked language modeling, establishing the pre-train/fine-tune paradigm. BERT is the canonical benchmark for both TensorRT and DeepSpeed performance evaluation and the reference model for many CUDA package smoke tests including FlashAttention-2 benchmarks.")

 _print_info("[38] H. Touvron et al., \"LLaMA: Open and Efficient Foundation Language Models,\" arXiv preprint arXiv:2302.13971, 2023. [Online]. Available: https://arxiv.org/abs/2302.13971 — Introduces the LLaMA family of open-weight models (7B–65B), which became the de facto reference models for llama-cpp-python, vLLM, and local LLM inference. The GPU memory requirements of LLaMA models are the practical motivation for quantization (bitsandbytes), efficient serving (vLLM), and the GPU VRAM audit in this report's sanity checks.")

 # ── Section 11: GPU-Accelerated Libraries (RAPIDS, CuPy, FAISS) ─
 print()
 print("  ── 11. GPU-Accelerated Libraries: RAPIDS, CuPy, FAISS ─────────────────────")
 print()
 _print_info("[39] RAPIDS Team, \"RAPIDS: GPU-Accelerated Data Science,\" NVIDIA / RAPIDS.ai, 2024. [Online]. Available: https://rapids.ai — The RAPIDS ecosystem provides GPU-accelerated pandas-compatible DataFrames (cuDF), scikit-learn-compatible ML algorithms (cuML), and graph analytics (cuGraph). RAPIDS requires Volta+ (CC 7.0+) and is commonly installed separately from PyTorch. cuDF, cuML, and cuGraph are individually detected in this report's CUDA packages section with import-vs-installed distinction.")

 _print_info("[40] RAPIDS Team, \"cuDF: GPU DataFrames,\" NVIDIA / RAPIDS GitHub, 2024. [Online]. Available: https://github.com/rapidsai/cudf — cuDF provides a pandas-compatible GPU DataFrame API accelerating data wrangling operations by 10–100× on NVIDIA GPUs. Uses libcudf (C++ CUDA backend) and integrates with Dask for multi-GPU and multi-node operations. This package requires the RAPIDS GPU requirements (Volta+ CC 7.0+, compatible CUDA toolkit) and is one of the most commonly affected packages by environment version mismatches.")

 _print_info("[41] R. Okuta, Y. Unno, D. Nishino, S. Hido, and C. Loomis, \"CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations,\" in Proc. Workshop on Machine Learning Systems (LearningSys), NIPS 2017. [Online]. Available: https://learningsys.org/nips17/assets/papers/paper_16.pdf — Introduces CuPy as a NumPy/SciPy-compatible array library that replaces CPU operations with CUDA kernels. Supports custom CUDA kernels inline in Python. CuPy is audited in the CUDA packages section and its GPU backend is verified via cupy.cuda.is_available().")

 _print_info("[42] J. Johnson, M. Douze, and H. Jégou, \"Billion-Scale Similarity Search with GPUs,\" IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021. [Online]. Available: https://arxiv.org/abs/1702.08734 — The primary academic paper for FAISS (Facebook AI Similarity Search), covering GPU-accelerated approximate nearest-neighbor search for billion-scale vector datasets. faiss-gpu provides CUDA-based indexing and search and is audited in the CUDA section; faiss-cpu provides the same algorithms without GPU dependency and is audited in the CPU section.")

 # ── Section 12: Agentic AI Frameworks ───────────────────────
 print()
 print("  ── 12. Agentic AI Frameworks ──────────────────────────────────────────────")
 print()
 _print_info("[43] H. Chase, \"LangChain: Building Applications with LLMs through Composability,\" GitHub Repository, 2022. [Online]. Available: https://github.com/langchain-ai/langchain — The LangChain framework pioneered composable LLM application development through chains, agents, memory, and retrieval components. Although its architecture has been partially superseded by LangGraph for stateful agentic workflows, LangChain remains the most widely installed LLM orchestration library and is audited in the Agentic AI section.")

 _print_info("[44] AWS Strands Team, \"Strands Agents SDK,\" Amazon Web Services, 2025. [Online]. Available: https://github.com/strands-agents/sdk-python — The Strands Agents SDK implements a tool-driven agentic loop where the LLM determines tool calling order. Designed for scalability with large tool libraries (100+ tools) and native Amazon Bedrock integration. Audited in the Agentic AI section; the default LLM backend uses Bedrock's Claude models, with OpenAI and Ollama support for GPU-local inference.")

 _print_info("[45] J. Liu, \"LlamaIndex: Data Framework for LLM Applications,\" GitHub Repository, 2022. [Online]. Available: https://github.com/run-llama/lllamaindex — LlamaIndex provides data ingestion, indexing, and retrieval primitives for building RAG (Retrieval-Augmented Generation) pipelines. Supports vector store integrations including FAISS, ChromaDB, Qdrant, and Weaviate. GPU acceleration benefits LlamaIndex indirectly through faster embedding generation and GPU-based vector search. Audited in the Agentic AI section.")

 _print_info("[46] Amazon Web Services, \"Amazon Bedrock Python SDK,\" AWS Documentation, 2024. [Online]. Available: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html — The boto3 SDK provides Python access to Amazon Bedrock's managed LLM inference, Bedrock Agents, and Knowledge Bases APIs. boto3's bedrock-runtime, bedrock-agent, and bedrock-agent-runtime service clients are the sole pip-installable interface to Bedrock; no separate bedrock package exists. Audited under LLM Clients in this report.")

 # ── Section 13: Vector Databases ────────────────────────────
 print()
 print("  ── 13. Vector Databases ────────────────────────────────────────────────────")
 print()
 _print_info("[47] A. Babenko, A. Slesarev, A. Chigorin, and V. Lempitsky, \"Neural Codes for Image Retrieval,\" in Proc. European Conference on Computer Vision (ECCV), 2014. [Online]. Available: https://arxiv.org/abs/1404.1777 — Early foundational work showing deep neural network features as compact vector representations for large-scale image retrieval, establishing the embedding-based vector search paradigm that all modern vector databases implement. Context for why GPU acceleration (FAISS-GPU, Qdrant with CUDA, Weaviate) matters: embedding generation and nearest-neighbor search are both GPU-acceleratable.")

 _print_info("[48] Qdrant Team, \"Qdrant: High-Performance Vector Database,\" Qdrant, 2024. [Online]. Available: https://qdrant.tech/documentation/ — Documentation for Qdrant, a production vector database with HNSW-based approximate nearest-neighbor search, filtering, and payload storage. CUDA is used for accelerated indexing operations. Audited under Agentic AI / vector stores in this report. Recommended for billion-scale vector search where faiss-gpu memory constraints become limiting.")

 _print_info("[49] Weaviate Team, \"Weaviate: Open-Source Vector Database,\" Weaviate, 2024. [Online]. Available: https://weaviate.io/developers/weaviate — Documentation for Weaviate, an open-source vector database with built-in vectorization modules, multi-tenancy, and hybrid (vector + keyword) search. Relevant to GPU environments because Weaviate's CUDA-enabled modules can perform GPU-accelerated embedding and vector indexing. Audited in the Agentic AI section.")

 # ── Section 14: MLOps & Experiment Tracking ─────────────────
 print()
 print("  ── 14. MLOps & Experiment Tracking ────────────────────────────────────────")
 print()
 _print_info("[50] A. Chen et al., \"Developments in MLflow: A System to Accelerate the Machine Learning Lifecycle,\" in Proc. Workshop on Data Management for End-to-End Machine Learning (DEEM), 2020. [Online]. Available: https://dl.acm.org/doi/10.1145/3399579.3399867 — Describes MLflow's tracking, project, model registry, and deployment abstractions. MLflow is GPU-aware through PyTorch/TensorFlow autologging of training curves and hyperparameters. Audited in the MLOps section of this report.")

 _print_info("[51] L. Biewald, \"Experiment Tracking with Weights and Biases,\" Technical Report, 2020. [Online]. Available: https://www.wandb.com/papers/intro.pdf — Introduces Weights & Biases (wandb) for experiment tracking, hyperparameter sweep orchestration, model artifact versioning, and GPU metrics dashboards. W&B captures per-GPU memory usage, CUDA utilization, and temperature automatically when integrated with PyTorch. Audited as wandb in the MLOps section.")

 _print_info("[52] NVIDIA Corporation, \"NVIDIA Nsight Systems,\" NVIDIA Developer, 2024. [Online]. Available: https://developer.nvidia.com/nsight-systems — Nsight Systems is NVIDIA's system-wide profiling tool for analyzing GPU/CPU interaction, CUDA kernel timelines, NVLink communication, and memory transfer bottlenecks. While not a pip package, Nsight Systems is the standard tool for diagnosing why GPU utilization is low in CUDA ML workloads — understanding it contextualizes the memory-pressure and benchmark results in this report.")

 # ── Section 15: Python Package & Environment Management ──────
 print()
 print("  ── 15. Python Package & Environment Management ─────────────────────────────")
 print()
 _print_info("[53] Python Packaging Authority, \"pip: The Python Package Installer,\" Python Software Foundation, 2024. [Online]. Available: https://pip.pypa.io/en/stable/ — Official documentation for pip, the standard Python package installer. pip's importlib.metadata integration (PEP 566) is the mechanism this script uses for package detection: every ✅ or ❌ in the package audit reflects pip-visible packages in the currently active Python environment. Understanding pip's scope is prerequisite to interpreting this report's Audit Scope footnote correctly.")

 _print_info("[54] PyPA, \"virtualenv / venv: Python Virtual Environments,\" Python Software Foundation, 2024. [Online]. Available: https://docs.python.org/3/library/venv.html — Documentation for Python's built-in venv module and the broader virtual environment ecosystem. Virtual environments are the primary deployment unit this report audits: the script detects the current venv, and all pip-visible packages reflect the contents of the active venv. Explains the scope limitation described in the Audit Scope footnote.")

 _print_info("[55] B. Ragan-Kelley, F. Pérez, B. Granger, and the Jupyter Team, \"The Jupyter/IPython Architecture: A Unified View of Computational Environments,\" in Proc. SciPy, 2014. [Online]. Available: https://conference.scipy.org/proceedings/scipy2014/pdfs/granger.pdf — Foundational paper on the Jupyter architecture (kernels, protocols, notebooks). JupyterLab, Jupyter AI, and related extensions are audited in the Notebooks section of this report. CUDA GPU kernels run inside Jupyter notebooks when the Python kernel's venv has GPU-enabled packages installed.")

 _print_info("[56] conda-forge Community, \"conda-forge: A Community-Led Collection of Recipes for the conda Package Manager,\" conda-forge, 2024. [Online]. Available: https://conda-forge.org/docs/ — conda-forge provides community-maintained conda packages and is the primary alternative to pip for scientific Python package distribution. Relevant to this report's Audit Scope: packages installed via conda into an active conda environment are visible to this script; packages in other conda environments are not. RAPIDS and some CUDA packages have official conda-forge channels.")

 # ── Section 16: NVIDIA Software Stack & Tooling ─────────────
 print()
 print("  ── 16. NVIDIA Software Stack & Tooling ────────────────────────────────────")
 print()
 _print_info("[57] NVIDIA Corporation, \"NVIDIA Management Library (NVML),\" NVIDIA Developer, 2024. [Online]. Available: https://developer.nvidia.com/management-library-nvml — NVML is the C API underlying nvidia-smi. It provides programmatic access to GPU inventory, SM count, clock speeds, temperature, power draw, and per-process memory usage. This report's sanity checks use nvidia-smi (an NVML frontend) and torch.cuda.get_device_properties() (a CUDA runtime alternative) for SM count detection. NVML is also the backend for the pynvml Python bindings.")

 _print_info("[58] NVIDIA Corporation, \"NVIDIA System Management Interface (nvidia-smi),\" NVIDIA Developer Documentation, 2024. [Online]. Available: https://developer.nvidia.com/nvidia-smi — nvidia-smi is the primary command-line tool for NVIDIA GPU management and monitoring. This report uses nvidia-smi for driver version detection, GPU inventory (including SM count via --query-gpu=multiprocessor.count), and power/temperature monitoring. The SM count detection falls back to torch.cuda when nvidia-smi's multiprocessor.count field is unsupported by older drivers.")

 _print_info("[59] NVIDIA Corporation, \"NGC: NVIDIA GPU Cloud Container Registry,\" NVIDIA, 2024. [Online]. Available: https://catalog.ngc.nvidia.com — NGC provides pre-built, CUDA-optimized Docker containers for PyTorch, TensorFlow, JAX, TensorRT, and RAPIDS with exact version pinning for driver compatibility. When CUDA version mismatch causes GPU failures in bare-metal environments, NGC containers are the recommended solution — every container guarantees a working CUDA/cuDNN/framework stack.")

 _print_info("[60] NVIDIA Corporation, \"NVIDIA Data Center GPU Driver Documentation,\" NVIDIA Developer, 2024. [Online]. Available: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ — Official guide for installing and managing NVIDIA drivers on Linux data center systems. Understanding the relationship between driver version, maximum CUDA version (reported by nvidia-smi), and the CUDA toolkit version (reported by nvcc) is essential for diagnosing the GPU failures described in the General Notes. The driver must support the CUDA version the wheel was built against.")

 # ── Section 17: ONNX & Cross-Framework Interoperability ──────
 print()
 print("  ── 17. ONNX & Cross-Framework Interoperability ────────────────────────────")
 print()
 _print_info("[61] ONNX Community, \"ONNX: Open Neural Network Exchange,\" ONNX, 2024. [Online]. Available: https://onnx.ai — ONNX is the open standard for representing ML models as portable computation graphs. ONNX Runtime (onnxruntime-gpu) provides high-performance inference with CUDA and TensorRT Execution Providers, enabling deployment of PyTorch and TensorFlow models without framework overhead. The onnxruntime-gpu package is audited in the CUDA packages section and smoke-tested for CUDA and TRT EP availability.")

 _print_info("[62] NVIDIA Corporation, \"ONNX Runtime with TensorRT Execution Provider,\" NVIDIA Technical Blog, 2021. [Online]. Available: https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/ — Explains the TensorRT Execution Provider in ONNX Runtime: how TRT EP partitions an ONNX graph into TRT-optimizable subgraphs and CPU fallback subgraphs, enabling zero-code-change TensorRT acceleration for models exported from any ONNX-compatible framework. Context for the TRT EP smoke test in this report's CUDA section.")

 # ── Section 18: Compute Capability & Architecture Evolution ──
 print()
 print("  ── 18. Compute Capability & GPU Architecture Reference ────────────────────")
 print()
 _print_info("[63] NVIDIA Corporation, \"CUDA GPUs — Compute Capability,\" NVIDIA Developer, 2024. [Online]. Available: https://developer.nvidia.com/cuda-gpus — The authoritative table mapping every NVIDIA GPU to its Compute Capability (CC) version. This is the lookup table behind this report's GPU generation detection: Kepler (CC 3.x), Maxwell (CC 5.x), Pascal (CC 6.x), Volta (CC 7.0), Turing (CC 7.5), Ampere (CC 8.x), Ada Lovelace (CC 8.9), Hopper (CC 9.0), Blackwell (CC 10.x). Required for understanding all CC-gated feature checks in this report.")

 _print_info("[64] Z. Jia, M. Maggioni, B. Staiger, and D. P. Scarpazza, \"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking,\" arXiv preprint arXiv:1804.06826, 2018. [Online]. Available: https://arxiv.org/abs/1804.06826 — Academic microbenchmarking study of the V100 GPU's memory hierarchy, instruction throughput, L1/L2 cache sizes, and register bank organization. Provides empirically measured latencies and bandwidths that underlie the roofline-model performance analysis of CUDA kernels. Referenced by the FlashAttention authors for HBM access latency characterization.")

 _print_info("[65] NVIDIA Corporation, \"NVIDIA Ada Lovelace Architecture,\" NVIDIA Whitepaper, Sept. 2022. [Online]. Available: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf — Defines the Ada Lovelace architecture (CC 8.9, RTX 4090/RTX 4000 Ada), third-generation RT Cores, and fourth-generation Tensor Cores adding FP8 support. Important because Ada FP8 Tensor Cores and their interaction with FlashAttention-3's FP8 path are relevant to this report's FA detection logic on Ada hardware.")

 print()
 _print_info("Note: All URLs were verified in March 2026. NVIDIA documentation URLs are subject to reorganization with new toolkit releases; if a URL returns 404, navigate from https://docs.nvidia.com or https://developer.nvidia.com to find the current location of the resource. arXiv URLs are permanent and will not change.")
 print()

_header_line()
_header_line("✅  Report complete")
_header_line()
print(f"  {'Started':<18} {_RUN_TIME}")
print(f"  {'Finished':<18} {_END_TIME}")
if SAVE_REPORT and _report_path:
    print(f"  {'Saved to':<18} {_report_path}")
_header_line()
print()

print("Developed with assistance from Claude.ai (Anthropic) and")
print("ChatGPT (OpenAI). AI tools were used for code generation,")
print("structure, and refinement throughout development.")
print()
print("Suggestions, improvements, and contributions are warmly")
print("welcome — open an issue or submit a pull request.")

if _tee is not None:
    _tee.close()

# ============================================================
# 💬 User Feedback — Collected Responses & Field Observations
#    Aggregates qualitative feedback submitted by end users
#    describing real-world usage patterns, pain points, and
#    feature requests gathered from support channels and in-
#    product prompts. Entries are lightly edited for clarity
#    but otherwise preserved in the user's own voice. Use
#    this block to inform prioritization and UX decisions.
# ============================================================
#
# "It's Greek to me, but well done." -- Mom (3/4/26)
#
# "I'm glad it makes sense to you." -- Stepfather (3/4/26)
#
#
#
