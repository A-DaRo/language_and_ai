#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=rome
#SBATCH --job-name=notion-scraper-full
#SBATCH --output=notion-scraper-full-%j.out
#SBATCH --error=notion-scraper-full-%j.err

# ============================================================================
# PRODUCTION SLURM JOB SCRIPT FOR NOTION SCRAPER (CLUSTER MODE - FULL RUN)
# ============================================================================
# This script is optimized for full production scraping with maximum resources
# using the distributed cluster mode architecture
#
# Resource allocation:
# - 32 CPUs for maximum parallel worker processes
# - 64GB RAM to support ~30 concurrent worker instances
# - 2 hours runtime for large site scraping
# - Each worker requires ~1GB RAM and runs isolated Puppeteer instance
# ============================================================================

echo "========================================="
echo "SLURM Job Information (PRODUCTION)"
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
export NODE_OPTIONS="--max-old-space-size=57344"  # Allow Node.js to use ~56GB RAM
export UV_THREADPOOL_SIZE=$SLURM_CPUS_PER_TASK     # Match UV thread pool to CPU count

# Puppeteer/Chrome environment variables
export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=false

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

# Run the application in cluster mode
echo "========================================="
echo "Starting Notion Scraper (CLUSTER MODE - FULL RUN)"
echo "========================================="
echo ""

node main-cluster.js

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Job Completion Summary"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Display output statistics if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Scraping completed successfully!"
    echo "Output directory contents:"
    ls -lh course_material/ 2>/dev/null || echo "Output directory not found"
fi

echo "========================================="

exit $EXIT_CODE
