#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=rome
#SBATCH --job-name=notion-scraper
#SBATCH --output=notion-scraper-%j.out
#SBATCH --error=notion-scraper-%j.err

# ============================================================================
# OPTIMIZED SLURM JOB SCRIPT FOR NOTION SCRAPER
# ============================================================================
# This script is optimized for the puppeteer-cluster based Node.js application
# which dynamically spawns concurrent browser instances based on available RAM
#
# Key optimizations:
# - Single task with multiple CPUs (--cpus-per-task=16)
# - Explicit memory allocation (32GB) for predictable concurrency
# - Environment variables to maximize Node.js and Puppeteer performance
# - Proper output logging for debugging
# ============================================================================

echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Load required modules
echo "Loading modules..."
module load 2025
echo "Modules loaded successfully"
echo ""

# Set Node.js environment variables for optimal performance
export NODE_OPTIONS="--max-old-space-size=28672"  # Allow Node.js to use ~28GB RAM
export UV_THREADPOOL_SIZE=$SLURM_CPUS_PER_TASK     # Match UV thread pool to CPU count

# Puppeteer/Chrome environment variables
export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=false
export CHROME_PATH=/usr/bin/chromium-browser       # Adjust if needed for your system

# Set number of concurrent workers (optional override)
# The application auto-calculates based on free RAM, but you can force it:
# export MAX_CONCURRENCY=12

echo "========================================="
echo "Environment Configuration"
echo "========================================="
echo "NODE_OPTIONS: $NODE_OPTIONS"
echo "UV_THREADPOOL_SIZE: $UV_THREADPOOL_SIZE"
echo "Working directory: $(pwd)"
echo ""

# Display system resources
echo "========================================="
echo "Available System Resources"
echo "========================================="
free -h
echo ""
lscpu | grep -E "^CPU\(s\)|^Model name|^Thread\(s\) per core"
echo ""

# Run the application
echo "========================================="
echo "Starting Notion Scraper"
echo "========================================="
echo ""

# For production run (actual scraping):
# node main.js --yes

# For dry-run (planning phase only):
node main.js --dry-run

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job Completion Summary"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================="

# Exit with the application's exit code
exit $EXIT_CODE
