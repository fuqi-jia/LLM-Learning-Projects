#!/bin/bash

# Transformer Training Script for Math Addition
# This script trains a Transformer model on math addition dataset

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_DATA_DIR="./dataset"
DEFAULT_MAX_SEQ_LEN=30
DEFAULT_D_MODEL=256
DEFAULT_N_HEADS=8
DEFAULT_N_LAYERS=4
DEFAULT_D_FF=1024
DEFAULT_DROPOUT=0.1
DEFAULT_BATCH_SIZE=32
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_EPOCHS=50
DEFAULT_MAX_GRAD_NORM=1.0
DEFAULT_LOG_INTERVAL=100
DEFAULT_SEED=42
DEFAULT_SAVE_DIR="./models"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

print_config() {
    echo -e "${CYAN}[CONFIG]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train Transformer model for math addition"
    echo ""
    echo "Data Options:"
    echo "  --data-dir DIR           Directory containing dataset (default: $DEFAULT_DATA_DIR)"
    echo "  --max-seq-len NUM        Maximum sequence length (default: $DEFAULT_MAX_SEQ_LEN)"
    echo ""
    echo "Model Options:"
    echo "  --d-model NUM            Model dimension (default: $DEFAULT_D_MODEL)"
    echo "  --n-heads NUM            Number of attention heads (default: $DEFAULT_N_HEADS)"
    echo "  --n-layers NUM           Number of transformer layers (default: $DEFAULT_N_LAYERS)"
    echo "  --d-ff NUM               Feed-forward dimension (default: $DEFAULT_D_FF)"
    echo "  --dropout FLOAT          Dropout rate (default: $DEFAULT_DROPOUT)"
    echo ""
    echo "Training Options:"
    echo "  --batch-size NUM         Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --learning-rate FLOAT    Learning rate (default: $DEFAULT_LEARNING_RATE)"
    echo "  --epochs NUM             Number of epochs (default: $DEFAULT_EPOCHS)"
    echo "  --max-grad-norm FLOAT    Maximum gradient norm (default: $DEFAULT_MAX_GRAD_NORM)"
    echo "  --log-interval NUM       Log interval (default: $DEFAULT_LOG_INTERVAL)"
    echo "  --early-stopping         Enable early stopping"
    echo ""
    echo "Other Options:"
    echo "  --seed NUM               Random seed (default: $DEFAULT_SEED)"
    echo "  --save-dir DIR           Model save directory (default: $DEFAULT_SAVE_DIR)"
    echo "  --test-only              Run test only (requires trained model)"
    echo "  --resume PATH            Resume training from checkpoint"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Preset configurations:"
    echo "  --small                  Small model (d_model=128, n_layers=2, fast training)"
    echo "  --medium                 Medium model (d_model=256, n_layers=4, balanced)"
    echo "  --large                  Large model (d_model=512, n_layers=6, best quality)"
    echo "  --quick                  Quick training (10 epochs, for testing)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Train with default settings"
    echo "  $0 --small --quick       # Quick training with small model"
    echo "  $0 --large --epochs 100  # Train large model for 100 epochs"
    echo "  $0 --test-only           # Run evaluation only"
    echo "  $0 --resume models/best_model.pth  # Resume from checkpoint"
}

# Initialize variables with defaults
DATA_DIR=$DEFAULT_DATA_DIR
MAX_SEQ_LEN=$DEFAULT_MAX_SEQ_LEN
D_MODEL=$DEFAULT_D_MODEL
N_HEADS=$DEFAULT_N_HEADS
N_LAYERS=$DEFAULT_N_LAYERS
D_FF=$DEFAULT_D_FF
DROPOUT=$DEFAULT_DROPOUT
BATCH_SIZE=$DEFAULT_BATCH_SIZE
LEARNING_RATE=$DEFAULT_LEARNING_RATE
EPOCHS=$DEFAULT_EPOCHS
MAX_GRAD_NORM=$DEFAULT_MAX_GRAD_NORM
LOG_INTERVAL=$DEFAULT_LOG_INTERVAL
SEED=$DEFAULT_SEED
SAVE_DIR=$DEFAULT_SAVE_DIR
EARLY_STOPPING=""
TEST_ONLY=""
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --max-seq-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --d-model)
            D_MODEL="$2"
            shift 2
            ;;
        --n-heads)
            N_HEADS="$2"
            shift 2
            ;;
        --n-layers)
            N_LAYERS="$2"
            shift 2
            ;;
        --d-ff)
            D_FF="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max-grad-norm)
            MAX_GRAD_NORM="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --early-stopping)
            EARLY_STOPPING="--early_stopping"
            shift
            ;;
        --test-only)
            TEST_ONLY="--test_only"
            shift
            ;;
        --small)
            D_MODEL=128
            N_LAYERS=2
            D_FF=512
            BATCH_SIZE=64
            shift
            ;;
        --medium)
            D_MODEL=256
            N_LAYERS=4
            D_FF=1024
            BATCH_SIZE=32
            shift
            ;;
        --large)
            D_MODEL=512
            N_LAYERS=6
            D_FF=2048
            BATCH_SIZE=16
            shift
            ;;
        --quick)
            EPOCHS=10
            LOG_INTERVAL=50
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validation functions
validate_positive_int() {
    if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
        return 1
    fi
    return 0
}

validate_positive_float() {
    # Support regular decimals and scientific notation (e.g., 1e-4, 2.5e-3)
    if [[ "$1" =~ ^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]] || [[ "$1" =~ ^[0-9]+[eE][+-]?[0-9]+$ ]]; then
        # Use python to check if the number is positive (more reliable than bc)
        if python -c "import sys; sys.exit(0 if float('$1') > 0 else 1)" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Validate inputs
print_step "Validating configuration..."

if ! validate_positive_int "$MAX_SEQ_LEN"; then
    print_error "Maximum sequence length must be a positive integer"
    exit 1
fi

if ! validate_positive_int "$D_MODEL"; then
    print_error "Model dimension must be a positive integer"
    exit 1
fi

if ! validate_positive_int "$N_HEADS"; then
    print_error "Number of heads must be a positive integer"
    exit 1
fi

if [ $((D_MODEL % N_HEADS)) -ne 0 ]; then
    print_error "Model dimension ($D_MODEL) must be divisible by number of heads ($N_HEADS)"
    exit 1
fi

if ! validate_positive_int "$N_LAYERS"; then
    print_error "Number of layers must be a positive integer"
    exit 1
fi

if ! validate_positive_int "$D_FF"; then
    print_error "Feed-forward dimension must be a positive integer"
    exit 1
fi

if ! validate_positive_float "$DROPOUT" || ! python -c "import sys; sys.exit(0 if 0 < float('$DROPOUT') < 1 else 1)" 2>/dev/null; then
    print_error "Dropout rate must be between 0 and 1"
    exit 1
fi

if ! validate_positive_int "$BATCH_SIZE"; then
    print_error "Batch size must be a positive integer"
    exit 1
fi

if ! validate_positive_float "$LEARNING_RATE"; then
    print_error "Learning rate must be a positive number"
    exit 1
fi

if ! validate_positive_int "$EPOCHS"; then
    print_error "Number of epochs must be a positive integer"
    exit 1
fi

# Check dependencies
print_step "Checking dependencies..."

if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "transformer/train_transformer.py" ]; then
    print_error "transformer/train_transformer.py not found. Please run this script from the project root directory."
    exit 1
fi

if [ -z "$TEST_ONLY" ] && [ ! -d "$DATA_DIR" ]; then
    print_error "Dataset directory $DATA_DIR not found. Please generate dataset first using: scripts/generate.sh"
    exit 1
fi

if [ -z "$TEST_ONLY" ]; then
    for file in "train.json" "val.json" "test.json" "vocab.json"; do
        if [ ! -f "$DATA_DIR/$file" ]; then
            print_error "Required file $DATA_DIR/$file not found"
            exit 1
        fi
    done
fi

# Check Python packages
print_info "Checking Python packages..."
python -c "import torch, json, matplotlib.pyplot" 2>/dev/null || {
    print_error "Required Python packages not found. Please install: torch matplotlib"
    print_info "You can install them using: pip install torch matplotlib"
    exit 1
}

print_success "All dependencies satisfied"

# Print configuration
print_step "Training Configuration"
echo ""
print_config "Data Configuration:"
echo "  📁 Dataset directory: $DATA_DIR"
echo "  📏 Max sequence length: $MAX_SEQ_LEN"
echo ""
print_config "Model Configuration:"
echo "  🧠 Model dimension: $D_MODEL"
echo "  👁️  Attention heads: $N_HEADS"
echo "  🏗️  Transformer layers: $N_LAYERS"
echo "  🔄 Feed-forward dimension: $D_FF"
echo "  💧 Dropout rate: $DROPOUT"
echo ""
print_config "Training Configuration:"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📈 Learning rate: $LEARNING_RATE"
echo "  🔄 Epochs: $EPOCHS"
echo "  ✂️  Max gradient norm: $MAX_GRAD_NORM"
echo "  📝 Log interval: $LOG_INTERVAL"
echo "  🎲 Random seed: $SEED"
echo "  💾 Save directory: $SAVE_DIR"
echo ""

# Estimate model size and training time
ESTIMATED_PARAMS=$((D_MODEL * D_MODEL * 4 * N_LAYERS + D_MODEL * D_FF * 2 * N_LAYERS + D_MODEL * 15))
ESTIMATED_SIZE_MB=$((ESTIMATED_PARAMS * 4 / 1024 / 1024))

print_info "Estimated model parameters: $(printf "%d" $ESTIMATED_PARAMS)"
print_info "Estimated model size: ${ESTIMATED_SIZE_MB}MB"

if [ "$EPOCHS" -gt 20 ] && [ "$D_MODEL" -gt 256 ]; then
    print_warning "Large model with many epochs detected. Training may take several hours."
fi

# Ask for confirmation for long training
if [ "$EPOCHS" -gt 50 ] && [ -z "$TEST_ONLY" ]; then
    read -p "$((echo -e ${YELLOW}[WARNING]${NC} Long training detected ($EPOCHS epochs). Continue? [y/N]: ))" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Training cancelled by user"
        exit 0
    fi
fi

# Create save directory
if [ ! -d "$SAVE_DIR" ]; then
    print_info "Creating save directory: $SAVE_DIR"
    mkdir -p "$SAVE_DIR"
fi

# Build Python command
PYTHON_CMD="python transformer/train_transformer.py"
PYTHON_CMD="$PYTHON_CMD --data_dir $DATA_DIR"
PYTHON_CMD="$PYTHON_CMD --max_seq_len $MAX_SEQ_LEN"
PYTHON_CMD="$PYTHON_CMD --d_model $D_MODEL"
PYTHON_CMD="$PYTHON_CMD --n_heads $N_HEADS"
PYTHON_CMD="$PYTHON_CMD --n_layers $N_LAYERS"
PYTHON_CMD="$PYTHON_CMD --d_ff $D_FF"
PYTHON_CMD="$PYTHON_CMD --dropout $DROPOUT"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --learning_rate $LEARNING_RATE"
PYTHON_CMD="$PYTHON_CMD --epochs $EPOCHS"
PYTHON_CMD="$PYTHON_CMD --max_grad_norm $MAX_GRAD_NORM"
PYTHON_CMD="$PYTHON_CMD --log_interval $LOG_INTERVAL"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"
PYTHON_CMD="$PYTHON_CMD --save_dir $SAVE_DIR"

if [ -n "$EARLY_STOPPING" ]; then
    PYTHON_CMD="$PYTHON_CMD $EARLY_STOPPING"
fi

if [ -n "$RESUME" ]; then
    PYTHON_CMD="$PYTHON_CMD --resume $RESUME"
fi

# Execute training
if [ -n "$TEST_ONLY" ]; then
    print_step "Running evaluation only..."
else
    print_step "Starting training..."
fi

echo ""
print_info "Command: $PYTHON_CMD"
echo ""

START_TIME=$(date +%s)

if $PYTHON_CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))
    
    print_success "Training completed successfully!"
    echo ""
    print_info "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    
    # Show saved files
    if [ -d "$SAVE_DIR" ]; then
        print_info "Files in save directory:"
        ls -lh "$SAVE_DIR"/ | grep -E '\.(pth|png)$' | while read -r line; do
            echo "  📄 $line"
        done
    fi
    
    echo ""
    print_success "Training completed! Check the results above."
    print_info "Next steps:"
    echo "  1. Review training curves in training_curves.png"
    echo "  2. Check model performance in the output above"
    echo "  3. Use the trained model for inference"
    
    if [ -f "$SAVE_DIR/best_model.pth" ]; then
        print_info "Best model saved at: $SAVE_DIR/best_model.pth"
    fi
    
else
    print_error "Training failed!"
    print_info "Common issues:"
    echo "  - Check if dataset is properly generated"
    echo "  - Verify Python dependencies are installed"
    echo "  - Check GPU/CPU memory availability"
    echo "  - Review error messages above"
    exit 1
fi
