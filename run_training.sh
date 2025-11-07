#!/bin/bash

# Training launcher script with monitoring and logging
# Usage: ./run_training.sh [config_name]

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/training.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        print_error "Training already running with PID $OLD_PID"
        echo "To stop: kill $OLD_PID"
        echo "Or wait for completion"
        exit 1
    else
        print_warn "Stale PID file found, removing..."
        rm "$PID_FILE"
    fi
fi

# Check GPU availability
print_info "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. GPU not available?"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
print_info "GPU detected: $GPU_INFO"

# Check available memory
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
if [ "$FREE_MEM" -lt 8000 ]; then
    print_warn "Low GPU memory: ${FREE_MEM}MB free. Training may fail."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python environment
print_info "Checking Python environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    print_info "Virtual environment activated"
else
    print_warn "No virtual environment found. Using system Python."
fi

# Verify dependencies
print_info "Verifying dependencies..."
python3 -c "import torch; import timm; import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
fi

# Check data directories
print_info "Checking data directories..."
if ! python3 -c "from main import Config; import os; c=Config(); assert os.path.exists(c.BRCA_WSI_DIR), 'BRCA dir missing'; assert os.path.exists(c.RNA_CSV_PATH), 'RNA CSV missing'" 2>/dev/null; then
    print_error "Data directories not configured or missing."
    print_error "Update paths in main.py (Config class)"
    exit 1
fi

print_info "Data directories found ✓"

# Create output directory
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

# Backup previous run if exists
if [ -f "${OUTPUT_DIR}/results_summary.json" ]; then
    BACKUP_DIR="${OUTPUT_DIR}/backup_${TIMESTAMP}"
    print_info "Backing up previous results to ${BACKUP_DIR}"
    mkdir -p "$BACKUP_DIR"
    cp -r "${OUTPUT_DIR}"/*.pth "${OUTPUT_DIR}"/*.json "${OUTPUT_DIR}"/*.png "$BACKUP_DIR/" 2>/dev/null || true
fi

# Print training configuration
print_info "Training Configuration:"
python3 << EOF
from main import Config
c = Config()
print(f"  Batch size: {c.BATCH_SIZE}")
print(f"  Learning rate: {c.LEARNING_RATE}")
print(f"  Max epochs: {c.NUM_EPOCHS}")
print(f"  Top genes: {c.NUM_TOP_GENES}")
print(f"  Max patches: {c.MAX_PATCHES_PER_PATIENT}")
print(f"  Device: {c.DEVICE}")
EOF

# Confirm start
echo ""
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Training cancelled"
    exit 0
fi

# Start training
print_info "Starting training at $(date)"
print_info "Logging to: $LOG_FILE"
print_info "Monitor with: tail -f $LOG_FILE"
echo ""
print_info "To run in background:"
print_info "  1. Press Ctrl+Z to suspend"
print_info "  2. Type: bg"
print_info "  3. Type: disown"
echo ""

# Run training with logging
(
    echo "=========================================="
    echo "Training started at $(date)"
    echo "=========================================="
    echo ""
    
    python3 -u main.py 2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    echo "=========================================="
    echo "Training ended at $(date)"
    echo "Exit code: $EXIT_CODE"
    echo "=========================================="
    
    # Cleanup PID file
    rm -f "$PID_FILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_info "Training completed successfully!"
        print_info "Results saved to: $OUTPUT_DIR"
        
        # Generate summary
        if [ -f "${OUTPUT_DIR}/results_summary.json" ]; then
            print_info "Performance Summary:"
            python3 << EOF
import json
with open('${OUTPUT_DIR}/results_summary.json') as f:
    results = json.load(f)
    for model, metrics in results.items():
        if 'accuracy' in metrics:
            print(f"  {model}: {metrics['accuracy']:.4f}")
EOF
        fi
    else
        print_error "Training failed with exit code $EXIT_CODE"
        print_error "Check log file: $LOG_FILE"
    fi
    
    exit $EXIT_CODE
) &

# Save PID
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

print_info "Training running with PID: $TRAIN_PID"
print_info "To check status: ps -p $TRAIN_PID"
print_info "To stop: kill $TRAIN_PID"

# Wait for completion if not backgrounded
wait $TRAIN_PID
EXIT_CODE=$?

exit $EXIT_CODE