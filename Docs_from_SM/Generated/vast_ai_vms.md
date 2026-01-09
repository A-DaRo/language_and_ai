# Setting Up Vast.ai VMs for Neuro-Stylometry Pipeline

This guide provides step-by-step instructions for deploying and running the neuro-stylometry pipeline (Phase A and Phase D) on Vast.ai GPU instances. The pipeline supports dual-mode execution with automatic hardware detection for optimal performance on HPC environments.

## Table of Contents
1. [Logging into Vast.ai](#1-logging-into-vastai)
2. [Generating and Adding SSH Keys](#2-generating-and-adding-ssh-keys)
3. [Choosing the Right VM Configuration](#3-choosing-the-right-vm-configuration)
4. [Connecting via SSH](#4-connecting-via-ssh)
5. [Cloning the Repository](#5-cloning-the-repository)
6. [Installing the Development Environment](#6-installing-the-development-environment)
7. [Optimizing Worker Configuration](#7-optimizing-worker-configuration)
8. [Transferring Data Files](#8-transferring-data-files)
9. [Running the Pipeline](#9-running-the-pipeline)

---

## 1. Logging into Vast.ai

### Initial Account Setup

1. Navigate to [https://vast.ai](https://vast.ai) in your web browser
2. Click **"Sign Up"** if you don't have an account, or **"Log In"** if you do (My credentials are in the pdf I sent you)
3. Enter your credentials:
   - **Email**: `your.email@example.com`
   - **Password**: `your_secure_password`

### Understanding Vast.ai Pricing

- Pricing is per GPU per hour (typically $0.10-$2.00/hour depending on GPU)
- You're charged for the entire instance lifetime (storage + compute)
- Billing continues until you explicitly **destroy** the instance
- Monitor your balance regularly to avoid unexpected charges

---

## 2. Generating and Adding SSH Keys

SSH keys are required for secure access to Vast.ai instances. This section covers key generation and configuration.

### Generating an Ed25519 SSH Key Pair

**On Windows (PowerShell or Git Bash):**
```powershell
# Open PowerShell or Git Bash
ssh-keygen -t ed25519 -C "vastai-neuro-stylometry" -f ~/.ssh/vastai_ed25519
```

**On Linux/Mac (Terminal):**
```bash
ssh-keygen -t ed25519 -C "vastai-neuro-stylometry" -f ~/.ssh/vastai_ed25519
```

When prompted:
- **Enter file to save the key**: Press Enter (uses default `~/.ssh/vastai_ed25519`)
- **Enter passphrase**: Press Enter for no passphrase, or enter a secure passphrase
- **Enter same passphrase again**: Confirm your choice

This creates two files:
- `~/.ssh/vastai_ed25519` (private key - **NEVER share this**)
- `~/.ssh/vastai_ed25519.pub` (public key - safe to share)

### Adding the Public Key to Vast.ai

1. Display your public key:
   ```bash
   # Windows (PowerShell)
   Get-Content ~/.ssh/vastai_ed25519.pub
   
   # Linux/Mac
   cat ~/.ssh/vastai_ed25519.pub
   ```

2. Copy the entire output (starts with `ssh-ed25519 AAAA...`)

3. In Vast.ai web interface:
   - Click **Account** (top-right corner)
   - Navigate to **SSH Keys** section
   - Click **"+ Add SSH Key"**
   - Paste your public key into the text field
   - Give it a descriptive name: `"Neuro-Stylometry Workstation"`
   - Click **"Add Key"**

4. Verify the key appears in your SSH Keys list

---

## 3. Choosing the Right VM Configuration

The neuro-stylometry pipeline has different hardware requirements for Phase A (pollution guard) and Phase D (neural stylometry training). This section helps you select optimal configurations.

### Phase A: Pollution Guard Pipeline

**Primary Requirements:**
- **High GPU memory bandwidth** (throughput > raw VRAM)
- **Moderate VRAM**: 12-24 GB sufficient for GLiNER + LEACE
- **CPU cores**: 16+ physical cores for parallel chunking (no virtual cores be aware)
- **RAM**: 32-64 GB (depends on dataset size)
- **Storage**: 50 GB minimum (for datasets + artifacts)

**Recommended GPU Models (prioritized by throughput):**

| GPU Model | VRAM | Memory Bandwidth  | Best For |
|-----------|------|------------------|--------------|----------|
| **RTX 4090** | 24 GB | 1008 GB/s | Best overall (Phase A + D) |
| **RTX 3090** | 24 GB | 936 GB/s | Excellent value |
| **A5000** | 24 GB | 768 GB/s | Professional stability |
| **RTX 3080 Ti** | 12 GB | 912 GB/s | Budget option (sufficient) |
| **V100** | 16 GB | 900 GB/s |  Cloud-optimized |

**Minimum specs for Phase A:**
```
GPU: RTX 3060 (12 GB VRAM, 360 GB/s bandwidth)
CPU: 6+ cores
RAM: 24 GB
Storage: 40 GB
```

### Phase D: Neural Stylometry Training

**Primary Requirements:**
- **High VRAM**: 24-48 GB (model + batch size)
- **GPU compute**: High CUDA cores for training
- **Memory bandwidth**: Important but secondary to VRAM
- **CPU/RAM**: Similar to Phase A
- **Storage**: 100+ GB (for checkpoints + datasets)

**Recommended GPU Models (prioritized by VRAM + compute):**

| GPU Model | VRAM | FP32 TFLOPS |Best For |
|-----------|------|-------------|--------------|----------|
| **RTX 4090** | 24 GB | 82.6 | Best single-GPU option |
| **A6000** | 48 GB | 38.7 | $Large batch sizes |
| **A100** | 40/80 GB | 19.5 (FP32) | Professional training |
| **RTX 3090** | 24 GB | 35.6 | Good value for most models |

### Selecting an Instance on Vast.ai

1. Click **"Search"** in the top navigation bar
2. Set filters in the left sidebar:
   ```
   GPU Type: Select from recommendations above
   GPU RAM: ≥12 GB (Phase A) or ≥24 GB (Phase D)
   Total VRAM: Match your requirement
   CPU Cores: ≥8 (these are virtual once again be aware)
   System RAM: ≥32 GB
   Disk Space: ≥50 GB
   ```

3. Enable advanced filters:
   - **DLPerf**: ≥0.7 (reliability score)
   - **Inet Down**: ≥100 Mbps (for data transfer)
   - **CUDA Version**: ≥11.8 (check with `nvcc --version` after connection)
   - **Docker Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel` or `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`

4. Sort results by:
   - **"DPH"** (dollars per hour) - for budget
   - **"DLP"** (DLPerf score) - for reliability
   - **"BW"** (bandwidth) - for Phase A throughput

5. Click on an instance to see details:
   - Check **"Host Details"** for uptime/reliability
   - Review **"Machine Specifications"** carefully
   - Verify **"Network Ports"** shows available SSH port

6. Click **"Rent"** and select:
   - **Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel` (recommended)
   - **Disk Space**: 50-100 GB
   - **On-start script**: Leave empty for now
   - Click **"Rent"** to confirm

7. Wait for instance status to change from **"Loading"** → **"Running"**
   - This typically takes 1-3 minutes
   - You'll receive connection details (IP address, SSH port)

### IMPORTANT!

Differences between physical and virtual cores are subtle and depend on device. Cores are also shared between users, always pay attention to availability. Best practice, always check online cpu device specs
Another important factor is network latency. European providers usually have the lowest latency, network is also shared between users. (you don't wanna spend 20minutes+ to transfer a file)

---

## 4. Connecting via SSH

Once your instance is running, you'll need to establish an SSH connection with port forwarding for potential Jupyter/web interfaces.

### Getting Connection Details

1. In Vast.ai dashboard, find your running instance
2. Click **"Connect"** or the terminal icon
3. Note the connection details:
   ```
   IP Address: 123.456.789.012
   SSH Port: 12345
   ```

### Basic SSH Connection

**Standard command:**
```bash
ssh -p 12345 root@123.456.789.012
```

**With port forwarding (recommended):**
```bash
ssh -p 12345 root@123.456.789.012 -L 8080:localhost:8080
```

This forwards local port 8080 to remote port 8080, allowing you to:
- Access Jupyter notebooks running on the VM
- Use TensorBoard or other web interfaces
- Run local web-based monitoring tools

### Advanced Connection Options

**With multiple port forwards:**
```bash
ssh -p 12345 root@123.456.789.012 \
    -L 8080:localhost:8080 \
    -L 6006:localhost:6006 \
    -L 8888:localhost:8888
```
- `8080`: General web services
- `6006`: TensorBoard
- `8888`: Jupyter Lab (alternative port)

**With verbose output (debugging):**
```bash
ssh -v -p 12345 root@123.456.789.012 -L 8080:localhost:8080
```

### First Connection Checklist

After connecting, verify the environment:

```bash
# 1. Check GPU availability
nvidia-smi

# Expected output: GPU table with utilization, temperature, etc.

# 2. Verify CUDA installation
nvcc --version

# Expected: CUDA Version 11.8 or higher

# 3. Check Python version
python --version

# Expected: Python 3.9 or higher

# 4. Check system resources
lscpu | grep "CPU(s):"
free -h
df -h

# 5. Verify internet connectivity
ping -c 3 google.com

# 6. Check PyTorch GPU access (if PyTorch image)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# PyTorch: 2.1.0+cu118
# CUDA available: True
# GPU: NVIDIA GeForce RTX 3090 (or your GPU model)
```

### Maintaining Connection Stability

Long-running pipelines require stable connections. Use `tmux` or `screen`:

**Using tmux (recommended):**
```bash
# Install tmux if not present
apt-get update && apt-get install -y tmux

# Start new tmux session
tmux new -s neuro_pipeline

# Inside tmux, run your pipeline
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t neuro_pipeline
# List sessions: tmux ls
```

**Using screen (alternative):**
```bash
# Start new screen session
screen -S neuro_pipeline

# Detach: Ctrl+A, then D
# Reattach: screen -r neuro_pipeline
# List sessions: screen -ls
```

Benefits:
- Pipeline continues running if SSH disconnects
- Safe from network interruptions
- Can close laptop without stopping work
- Multiple terminal windows in one SSH session

**To detach from vast ai ssh session press Ctrl+D**

**To detach from vast ai ssh session but keep process running press Ctrl+A+D**

---

## 5. Cloning the Repository

The neuro-stylometry repository is public, so no Git authentication is required.

### Installing Git (if needed)

Most Docker images include Git, but if not:
```bash
apt-get update && apt-get install -y git
```

### Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/A-DaRo/language_and_ai.git

# Navigate into the repository
cd language_and_ai

# Verify clone was successful
ls -la

# Expected output: LICENSE, README.md, pyproject.toml, src/, conf/, etc.
```

### Checking Out Specific Branch

The current working branch is `interim_py`:
```bash
# Verify current branch
git branch

# If not on interim_py, switch to it
git checkout interim_py

# Pull latest changes
git pull origin interim_py

# Verify you're on the correct branch
git log --oneline -n 5
```

### Optional: Git Configuration (for development)

If you plan to commit changes from the VM:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

**Note:** For most HPC pipeline runs, you'll only need read access (no Git config required). Avoid putting sensitive data e.g. passwords. Hardware owner 'could' have access (according to vast ai provider rules this is not possible, but a lot of CVE's exist on the topic)


---

## 6. Installing the Development Environment

The neuro-stylometry package uses modern Python packaging with editable installation and optional dependency groups.

### Prerequisites Check

```bash
# Ensure Python 3.9+ is available
python --version

# Should output: Python 3.9.x or higher
# If not, install Python 3.10:
# apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip
```

### Editable Installation with Development Dependencies

```bash
# Navigate to repository (if not already there)
cd /language_and_ai

# Upgrade pip to latest version (important for pyproject.toml parsing)
pip install --upgrade pip setuptools wheel

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# This installs:
# - Core package: neuro-stylometry
# - Required dependencies: torch, transformers, pyarrow, gliner, etc.
# - Development tools: pytest, ruff, black, mypy, etc.
# - Extras: notebook tools (jupyter, plotly) if included in [dev]
```

### Verifying Installation

```bash
# 1. Check package is installed
pip show neuro-stylometry

# Expected output:
# Name: neuro-stylometry
# Version: 0.1.0 (or current version)
# Location: /root/language_and_ai/src

# 2. Verify CLI entry point
neuro-stylometry --help

# Expected output: CLI usage information with available commands

# 3. Test core imports
python -c "
from neuro_stylometry.data_engine.dataset import SOBRDataset
from neuro_stylometry.pollution_guard.gliner_detector import GLiNERDetector
from neuro_stylometry.pollution_guard.leace_computer import LEACEComputer
from neuro_stylometry.hardware_ops.detection import detect_hardware_profile, ProfileType
print('✓ Core imports successful')
print(f'Hardware profile: {detect_hardware_profile()}')
"

# Expected output:
# ✓ Core imports successful
# Hardware profile: ProfileType.HPC

# 4. Verify GPU access in package context
python -c "
import torch
from neuro_stylometry.hardware_ops.detection import detect_hardware_profile
profile = detect_hardware_profile()
print(f'Profile: {profile}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# Expected output (on HPC with GPU):
# Profile: ProfileType.HPC
# CUDA available: True
# GPU device: NVIDIA GeForce RTX 3090
# GPU memory: 24.00 GB
```

### Understanding Dependency Groups

The `pyproject.toml` defines optional dependency groups:

- **`.[dev]`**: Development tools (pytest, linting, formatting)
- **`.[notebook]`**: Jupyter and visualization tools (if defined)

For HPC pipeline runs, `[dev]` is recommended as it includes testing tools.

---

## 7. Optimizing Worker Configuration

The pipeline uses parallel chunking for data processing. Optimal worker count depends on CPU cores, memory, and I/O characteristics. This step benchmarks different configurations to find the best setting.

### Understanding Worker Scaling

The neuro-stylometry pipeline processes data in parallel using Python's `multiprocessing`. Key factors:

- **CPU cores**: More cores allow more parallel workers
- **Memory per worker**: Each worker loads data chunks (typically 2-4 GB per worker)
- **I/O bottlenecks**: Too many workers can saturate disk/network bandwidth
- **Overhead**: Process creation and IPC have costs
- **Optimal range**: Usually 60-80% of available CPU cores

### Running the Worker Scaling Benchmark

```bash
# Navigate to repository root
cd /root/language_and_ai

# Run the worker scaling test suite
pytest tests/benchmarks/test_parallel_chunking.py::TestWorkerScaling -v -s

# This test will:
# 1. Detect available CPU cores
# 2. Test worker counts: [1, 2, 4, 8, 12, 16, 24, 32]
# 3. Process sample data chunks with each configuration
# 4. Measure throughput (chunks/second) and total time
# 5. Display results table and recommendation
```

**Expected output:**
```
================================ test session starts ================================
platform linux -- Python 3.10.11, pytest-7.4.0, pluggy-1.0.0
cachedir: .pytest_cache
rootdir: /root/language_and_ai
configfile: pyproject.toml
collected 1 item

tests/benchmarks/test_parallel_chunking.py::TestWorkerScaling::test_worker_scaling_performance

Worker Scaling Benchmark Results:
┌─────────┬────────────┬────────────────┬──────────────┐
│ Workers │ Total Time │ Throughput     │ Efficiency   │
├─────────┼────────────┼────────────────┼──────────────┤
│ 1       │ 45.2s      │ 22.1 chunks/s  │ 100.0% (ref) │
│ 2       │ 23.8s      │ 42.0 chunks/s  │ 95.0%        │
│ 4       │ 12.5s      │ 80.0 chunks/s  │ 90.2%        │
│ 8       │ 6.8s       │ 147.1 chunks/s │ 83.1%        │
│ 12      │ 5.1s       │ 196.1 chunks/s │ 73.7%        │ ← OPTIMAL
│ 16      │ 4.9s       │ 204.1 chunks/s │ 57.8%        │
│ 24      │ 5.2s       │ 192.3 chunks/s │ 36.2%        │
│ 32      │ 5.7s       │ 175.4 chunks/s │ 24.8%        │
└─────────┴────────────┴────────────────┴──────────────┘

RECOMMENDATION: Use 12 workers for optimal throughput/efficiency balance
System: 16 CPU cores, 48 GB RAM

PASSED [100%]

================================ 1 passed in 182.45s ================================
```

### Interpreting Results

- **Total Time**: End-to-end processing time for the test dataset
- **Throughput**: Chunks processed per second (higher is better)
- **Efficiency**: Utilization relative to linear scaling (100% = perfect parallelism)
- **Optimal**: Highest throughput before efficiency drop-off

**Decision criteria:**
1. Choose the worker count with highest throughput
2. If multiple values are close (within 5%), prefer lower count (less overhead)
3. Ensure sufficient RAM (check `free -h` during benchmark)
4. For Phase A GLiNER detection, optimal typically falls at 75-85% of CPU cores

### Applying Optimal Configuration

Once you identify the optimal worker count (e.g., 12), configure it for pipeline runs:

**Method 1: Environment Variable**
```bash
export NEURO_STYLOMETRY_WORKERS=12
```

**Method 2: YAML Configuration**
Edit `conf/hpc/pipeline.yaml`:
```yaml
# Find the chunking/parallel section and update:
parallel:
  num_workers: 12  # Based on benchmark result
```

---

## 8. Transferring Data Files

You'll need to transfer the processed Arrow dataset (`sobr.arrow`) from your local machine to the VM, and retrieve artifacts (projection matrices, logs) afterwards.

### Understanding SCP Syntax

SCP (Secure Copy Protocol) uses SSH for file transfer. Basic syntax:
```bash
scp -P PORT [OPTIONS] SOURCE DESTINATION
```

**Key points:**
- `-P PORT`: SSH port (UPPERCASE -P, not lowercase -p)
- Source/destination format: `user@host:/path` for remote, `/path` for local
- Always use `root` as user for Vast.ai VMs
- Paths are absolute on remote (start with `/`)

### Transferring Files TO the VM (Local → Remote)

**Basic upload (single file):**
```bash
# From your LOCAL machine (Windows PowerShell, Git Bash, or WSL)
scp -P 12345 C:\Users\username\Desktop\Year3\language_and_ai\artifacts\data\sobr.arrow root@123.456.789.012:/root/language_and_ai/artifacts/data/

# Linux/Mac syntax (forward slashes):
scp -P 12345 /path/to/local/sobr.arrow root@123.456.789.012:/root/language_and_ai/artifacts/data/
```

**Upload with progress and compression:**
```bash
scp -P 12345 -C -v C:\Users\username\Desktop\Year3\language_and_ai\artifacts\data\sobr.arrow root@123.456.789.012:/root/language_and_ai/artifacts/data/sobr.arrow

# -C: Enable compression (faster for text/Arrow files)
# -v: Verbose output (shows progress)
```

**Upload entire directory:**
```bash
# Upload all data files at once
scp -P 12345 -r C:\Users\username\Desktop\Year3\language_and_ai\artifacts\data\ root@123.456.789.012:/root/language_and_ai/artifacts/

# -r: Recursive (copies entire directory tree)
```

**Upload with bandwidth limit (if on metered connection):**
```bash
scp -P 12345 -l 8000 C:\Users\username\Desktop\Year3\language_and_ai\artifacts\data\sobr.arrow root@123.456.789.012:/root/language_and_ai/artifacts/data/

# -l 8000: Limit to 8000 Kbit/s (1 MB/s)
```

### Transferring Files FROM the VM (Remote → Local)

**Basic download (single file):**
```bash
# From your LOCAL machine
scp -P 12345 root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/projection_matrix.pt C:\Users\username\Desktop\Year3\language_and_ai\artifacts\phase_a\

# Linux/Mac:
scp -P 12345 root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/projection_matrix.pt /local/path/
```

**Download entire Phase A output:**
```bash
scp -P 12345 -r root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/ C:\Users\username\Desktop\Year3\language_and_ai\artifacts\
```

**Download specific artifacts:**
```bash
# Get projection matrix
scp -P 12345 root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/projection_matrix.pt ./

# Get clean dataset
scp -P 12345 root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/clean_dataset.arrow ./

# Get pollution logs
scp -P 12345 root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/pollution_logs.arrow ./
```

**Download logs:**
```bash
scp -P 12345 root@123.456.789.012:/root/language_and_ai/phase_a_output.log ./
```

### Windows-Specific Considerations

**Using PowerShell:**
- Use backslashes `\` for local Windows paths
- Use forward slashes `/` for remote Linux paths
- Enclose paths with spaces in quotes: `"C:\Program Files\..."`

**Using Git Bash/WSL:**
- Use Unix-style paths: `/c/Users/...` (Git Bash) or `/mnt/c/Users/...` (WSL)
- Standard forward slashes throughout

**Using WinSCP (GUI alternative):**
1. Download WinSCP from [https://winscp.net](https://winscp.net)
2. Configure connection:
   - File protocol: SCP
   - Host: `123.456.789.012`
   - Port: `12345`
   - User: `root`
   - Private key: `C:\Users\username\.ssh\vastai_ed25519`
3. Drag-and-drop files between local and remote panels

### Estimating Transfer Times (Prefer European providers if high-latency is observed)

Typical file sizes and transfer times (assuming 100 Mbps connection):

| File | Size | Transfer Time |
|------|------|---------------|
| `sobr.arrow` (full dataset) | ~2-4 GB | 3-6 minutes |
| `sobr_laptop.arrow` (subset) | ~200-500 MB | 20-50 seconds |
| `projection_matrix.pt` | ~50-200 MB | 5-20 seconds |
| `pollution_logs.arrow` | ~100-500 MB | 10-50 seconds |
| `clean_dataset.arrow` | ~2-4 GB | 3-6 minutes |

**Optimization tips:**
- Use `-C` for compression (reduces transfer size by 30-60% for Arrow files)
- Transfer during off-peak hours for better network performance
- Consider using `rsync` for resumable transfers of very large files

### Alternative: Using rsync (for resumable transfers)

```bash
# Upload with resume support
rsync -avz --progress -e "ssh -p 12345" C:\Users\username\Desktop\Year3\language_and_ai\artifacts\data\sobr.arrow root@123.456.789.012:/root/language_and_ai/artifacts/data/

# Download with resume support
rsync -avz --progress -e "ssh -p 12345" root@123.456.789.012:/root/language_and_ai/artifacts/phase_a/ ./phase_a_results/

# Benefits:
# - Resumes interrupted transfers
# - Only transfers changed files
# - More efficient for large datasets
```

### Verifying Transfer Integrity

After transferring files, verify they arrived intact:

**On remote VM:**
```bash
# Check file exists and size
ls -lh /root/language_and_ai/artifacts/data/sobr.arrow

# Compute checksum
sha256sum /root/language_and_ai/artifacts/data/sobr.arrow
```

**On local machine (before transfer):**
```bash
# Windows (PowerShell)
Get-FileHash C:\Users\username\Desktop\Year3\language_and_ai\artifacts\data\sobr.arrow -Algorithm SHA256

# Linux/Mac
sha256sum /path/to/sobr.arrow
```

Compare the checksums - they should match exactly.

---

## 9. Running the Pipeline

With the environment set up and data transferred, you're ready to run the neuro-stylometry pipeline. This section covers running Phase A with proper logging and monitoring.

### Pipeline Architecture Overview

**Phase A (Pollution Guard Pipeline):**
1. **Ingest**: Load Arrow dataset (`sobr.arrow`)
2. **Detect**: GLiNER entity detection for PII/sensitive spans
3. **Mask**: Apply typed masking to detected spans
4. **Project**: Compute LEACE projection matrix
5. **Output**: Write clean dataset, projection matrix, and logs

**Key Scripts:**
- `scripts/run_phase_a.py`: Main Phase A orchestrator
- `scripts/validate_phase_a_handover.py`: Verify output artifacts
- `scripts/run_phase_d.py`: Phase D neural training (separate guide)

### Pre-Flight Checklist

Before running the pipeline, verify:

```bash
# 1. Dataset is present
ls -lh /root/language_and_ai/artifacts/data/sobr.arrow

# 2. Output directory exists
mkdir -p /root/language_and_ai/artifacts/phase_a

# 3. GPU is available (for HPC mode)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Hardware profile detection
python -c "from neuro_stylometry.hardware_ops.detection import detect_hardware_profile; print(detect_hardware_profile())"

# Expected: ProfileType.HPC

# 5. Configuration is correct
cat /root/language_and_ai/conf/hpc/pipeline.yaml

# 6. Start tmux session (for long-running processes)
tmux new -s phase_a_run
```

### Running Phase A Pipeline

**Basic execution:**
```bash
python neuro_stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output_dir artifacts/phase_a \
    --config_path conf/hpc/pipeline.yaml \
    --mode hpc
```

**With timestamped log file:**
```bash
LOGFILE="phase_a_$(date +%Y%m%d_%H%M%S).log"
python neuro_stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output_dir artifacts/phase_a \
    --config_path conf/hpc/pipeline.yaml \
    --mode hpc \
    2>&1 | tee "$LOGFILE"

echo "Log saved to: $LOGFILE"
```

**With resource monitoring:**
```bash
# In one tmux pane, run the pipeline
python neuro_stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output_dir artifacts/phase_a \
    --config_path conf/hpc/pipeline.yaml \
    --mode hpc \
    2>&1 | tee phase_a_output.log

# In another pane (Ctrl+B, then C to create new pane), monitor resources
watch -n 2 nvidia-smi

# Or monitor everything:
watch -n 2 'echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader; echo "=== CPU/RAM ==="; top -bn1 | head -n 5'
```

**Note:** As of current repository state, CLI may have partially implemented commands. Verify with:
```bash
neuro-stylometry --help
```

### Running Other Scripts

**Generate reports: (not implemented yet)**
```bash
python scripts/generate_report.py \
    --phase-a artifacts/phase_a \
    --output artifacts/reports/phase_a_report.html \
    2>&1 | tee report_generation.log
```

### Monitoring Long-Running Pipelines

**Set up continuous logging:**
```bash
# Log to file with automatic rotation
python neuro_stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output_dir artifacts/phase_a \
    --config_path conf/hpc/pipeline.yaml \
    --mode hpc \
    2>&1 | tee -a phase_a_run.log

# Monitor log in real-time from another terminal/tmux pane
tail -f phase_a_run.log

# Or with grep filtering
tail -f phase_a_run.log | grep -E "INFO|ERROR|WARNING"
```

**Track GPU utilization over time:**
```bash
# Log GPU stats every 10 seconds
while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv >> gpu_usage.log
    sleep 10
done
```

**Alert on completion:**
```bash
# Run pipeline and beep/notify when done
python neuro_stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output_dir artifacts/phase_a \
    --config_path conf/hpc/pipeline.yaml \
    --mode hpc \
    2>&1 | tee phase_a_output.log \
    && echo "Phase A complete!" \
    || echo "Phase A failed!"

# Or send email (if mail configured):
python scripts/run_phase_a.py [...] && echo "Success" | mail -s "Phase A Done" your@email.com
```

### Handling Errors

**Common issues and solutions:**

**1. Out of GPU memory:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```
**Solution:** Reduce batch size in `conf/hpc/pipeline.yaml`:
```yaml
gliner:
  batch_size: 32  # Reduce from 64
  max_length: 384  # Reduce from 512
```

**2. Dataset not found:**
```
FileNotFoundError: artifacts/data/sobr.arrow
```
**Solution:** Verify file was transferred correctly:
```bash
ls -lh artifacts/data/sobr.arrow
# If missing, re-run SCP transfer
```

**3. CUDA not available:**
```
Hardware profile: ProfileType.LAPTOP (expected HPC)
```
**Solution:** Verify GPU is accessible:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
# If False, reinstall PyTorch with CUDA support
```

**4. Checkpoint/resume (if pipeline crashes mid-run):**
The current implementation may not have built-in checkpointing. To add basic resume:
```bash
# Check if partial artifacts exist
ls -lh artifacts/phase_a/

# If clean_dataset.arrow exists but no projection_matrix.pt:
# You may need to re-run the full pipeline
# Future enhancement: Add --resume flag
```

---

## Appendix: Quick Reference

### Connection Command Template
```bash
ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080
```

### SCP Upload Template
```bash
scp -P <PORT> -C <LOCAL_FILE> root@<IP_ADDRESS>:<REMOTE_PATH>
```

### SCP Download Template
```bash
scp -P <PORT> -C root@<IP_ADDRESS>:<REMOTE_FILE> <LOCAL_PATH>
```

### Phase A Execution Template
```bash
cd /root/language_and_ai
tmux new -s phase_a
python neuro_stylometry run-phase-a \
    --dataset artifacts/data/sobr.arrow \
    --output_dir artifacts/phase_a \
    --config_path conf/hpc/pipeline.yaml \
    --mode hpc \
    2>&1 | tee phase_a_$(date +%Y%m%d_%H%M%S).log
```

### Tmux Quick Reference
```
Start session:   tmux new -s <name>
Detach:          Ctrl+B, then D
Reattach:        tmux attach -t <name>
List sessions:   tmux ls
Kill session:    tmux kill-session -t <name>
```

---