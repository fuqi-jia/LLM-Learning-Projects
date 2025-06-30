#!/bin/bash

# Math Addition Dataset Generator Script
# This script calls data/generate.py to generate training datasets for Transformer models

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_NUM_SAMPLES=10000
DEFAULT_MAX_NUMBERS=5
DEFAULT_MAX_VALUE=100
DEFAULT_MIN_NUMBERS=2
DEFAULT_OUTPUT_DIR="./dataset"
DEFAULT_TRAIN_RATIO=0.8
DEFAULT_VAL_RATIO=0.1
DEFAULT_TEST_RATIO=0.1
DEFAULT_SEED=42

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate math addition dataset for Transformer training"
    echo ""
    echo "Options:"
    echo "  -n, --num-samples NUM     Number of samples to generate (default: $DEFAULT_NUM_SAMPLES)"
    echo "  -x, --max-numbers NUM     Maximum number of addends (default: $DEFAULT_MAX_NUMBERS)"
    echo "  -v, --max-value NUM       Maximum value of numbers (default: $DEFAULT_MAX_VALUE)"
    echo "  -m, --min-numbers NUM     Minimum number of addends (default: $DEFAULT_MIN_NUMBERS)"
    echo "  -o, --output-dir DIR      Output directory (default: $DEFAULT_OUTPUT_DIR)"
    echo "  -t, --train-ratio RATIO   Training set ratio (default: $DEFAULT_TRAIN_RATIO)"
    echo "  -a, --val-ratio RATIO     Validation set ratio (default: $DEFAULT_VAL_RATIO)"
    echo "  -e, --test-ratio RATIO    Test set ratio (default: $DEFAULT_TEST_RATIO)"
    echo "  -s, --seed NUM            Random seed (default: $DEFAULT_SEED)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Preset configurations:"
    echo "  --small                  Small dataset (1,000 samples, max 3 numbers, max value 50)"
    echo "  --medium                 Medium dataset (50,000 samples, max 4 numbers, max value 100)"
    echo "  --large                  Large dataset (500,000 samples, max 6 numbers, max value 999)"
    echo "  --debug                  Debug dataset (100 samples, max 2 numbers, max value 10)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Generate default dataset"
    echo "  $0 --small               # Generate small dataset"
    echo "  $0 -n 20000 -x 4         # Generate 20k samples with max 4 addends"
    echo "  $0 --large -o ./big_data # Generate large dataset in ./big_data directory"
}

# Initialize variables with defaults
NUM_SAMPLES=$DEFAULT_NUM_SAMPLES
MAX_NUMBERS=$DEFAULT_MAX_NUMBERS
MAX_VALUE=$DEFAULT_MAX_VALUE
MIN_NUMBERS=$DEFAULT_MIN_NUMBERS
OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
TRAIN_RATIO=$DEFAULT_TRAIN_RATIO
VAL_RATIO=$DEFAULT_VAL_RATIO
TEST_RATIO=$DEFAULT_TEST_RATIO
SEED=$DEFAULT_SEED

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -x|--max-numbers)
            MAX_NUMBERS="$2"
            shift 2
            ;;
        -v|--max-value)
            MAX_VALUE="$2"
            shift 2
            ;;
        -m|--min-numbers)
            MIN_NUMBERS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        -a|--val-ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        -e|--test-ratio)
            TEST_RATIO="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --small)
            NUM_SAMPLES=1000
            MAX_NUMBERS=3
            MAX_VALUE=50
            MIN_NUMBERS=2
            shift
            ;;
        --medium)
            NUM_SAMPLES=50000
            MAX_NUMBERS=4
            MAX_VALUE=100
            MIN_NUMBERS=2
            shift
            ;;
        --large)
            NUM_SAMPLES=500000
            MAX_NUMBERS=6
            MAX_VALUE=999
            MIN_NUMBERS=2
            shift
            ;;
        --debug)
            NUM_SAMPLES=100
            MAX_NUMBERS=2
            MAX_VALUE=10
            MIN_NUMBERS=2
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

# Validate inputs
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [ "$NUM_SAMPLES" -le 0 ]; then
    print_error "Number of samples must be a positive integer"
    exit 1
fi

if ! [[ "$MAX_NUMBERS" =~ ^[0-9]+$ ]] || [ "$MAX_NUMBERS" -le 0 ]; then
    print_error "Maximum numbers must be a positive integer"
    exit 1
fi

if ! [[ "$MIN_NUMBERS" =~ ^[0-9]+$ ]] || [ "$MIN_NUMBERS" -le 0 ]; then
    print_error "Minimum numbers must be a positive integer"
    exit 1
fi

if [ "$MIN_NUMBERS" -gt "$MAX_NUMBERS" ]; then
    print_error "Minimum numbers cannot be greater than maximum numbers"
    exit 1
fi

if ! [[ "$MAX_VALUE" =~ ^[0-9]+$ ]] || [ "$MAX_VALUE" -le 0 ]; then
    print_error "Maximum value must be a positive integer"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if data/generate.py exists
if [ ! -f "data/generate.py" ]; then
    print_error "data/generate.py not found. Please run this script from the project root directory."
    exit 1
fi

# Print configuration
print_info "Dataset Generation Configuration:"
echo "  📊 Number of samples: $NUM_SAMPLES"
echo "  🔢 Number range: $MIN_NUMBERS-$MAX_NUMBERS addends"
echo "  📈 Value range: 1-$MAX_VALUE"
echo "  📁 Output directory: $OUTPUT_DIR"
echo "  📋 Dataset split: train($TRAIN_RATIO) / val($VAL_RATIO) / test($TEST_RATIO)"
echo "  🎲 Random seed: $SEED"
echo ""

# Estimate generation time and dataset size
if [ "$NUM_SAMPLES" -lt 1000 ]; then
    TIME_EST="< 1 minute"
elif [ "$NUM_SAMPLES" -lt 10000 ]; then
    TIME_EST="1-2 minutes"
elif [ "$NUM_SAMPLES" -lt 100000 ]; then
    TIME_EST="2-5 minutes"
else
    TIME_EST="5+ minutes"
fi

SIZE_EST=$(( NUM_SAMPLES * 200 / 1024 ))  # Rough estimate in KB
if [ "$SIZE_EST" -lt 1024 ]; then
    SIZE_EST="${SIZE_EST}KB"
else
    SIZE_EST="$(( SIZE_EST / 1024 ))MB"
fi

print_info "Estimated generation time: $TIME_EST"
print_info "Estimated dataset size: ~$SIZE_EST"
echo ""

# Ask for confirmation for large datasets
if [ "$NUM_SAMPLES" -gt 100000 ]; then
    read -p "$(echo -e ${YELLOW}[WARNING]${NC} Large dataset detected. Continue? [y/N]: )" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Generation cancelled by user"
        exit 0
    fi
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Build Python command
PYTHON_CMD="python data/generate.py"
PYTHON_CMD="$PYTHON_CMD --num_samples $NUM_SAMPLES"
PYTHON_CMD="$PYTHON_CMD --max_numbers $MAX_NUMBERS"
PYTHON_CMD="$PYTHON_CMD --max_value $MAX_VALUE"
PYTHON_CMD="$PYTHON_CMD --min_numbers $MIN_NUMBERS"
PYTHON_CMD="$PYTHON_CMD --output_dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --train_ratio $TRAIN_RATIO"
PYTHON_CMD="$PYTHON_CMD --val_ratio $VAL_RATIO"
PYTHON_CMD="$PYTHON_CMD --test_ratio $TEST_RATIO"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"

# Execute the Python script
print_info "Starting dataset generation..."
echo "Command: $PYTHON_CMD"
echo ""

START_TIME=$(date +%s)

if $PYTHON_CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_success "Dataset generation completed successfully!"
    echo ""
    print_info "Generation took: ${DURATION} seconds"
    print_info "Files created:"
    
    # List generated files with sizes
    if [ -f "$OUTPUT_DIR/train.json" ]; then
        TRAIN_SIZE=$(ls -lh "$OUTPUT_DIR/train.json" | awk '{print $5}')
        echo "  📈 $OUTPUT_DIR/train.json ($TRAIN_SIZE)"
    fi
    
    if [ -f "$OUTPUT_DIR/val.json" ]; then
        VAL_SIZE=$(ls -lh "$OUTPUT_DIR/val.json" | awk '{print $5}')
        echo "  🔍 $OUTPUT_DIR/val.json ($VAL_SIZE)"
    fi
    
    if [ -f "$OUTPUT_DIR/test.json" ]; then
        TEST_SIZE=$(ls -lh "$OUTPUT_DIR/test.json" | awk '{print $5}')
        echo "  🧪 $OUTPUT_DIR/test.json ($TEST_SIZE)"
    fi
    
    if [ -f "$OUTPUT_DIR/vocab.json" ]; then
        VOCAB_SIZE=$(ls -lh "$OUTPUT_DIR/vocab.json" | awk '{print $5}')
        echo "  📚 $OUTPUT_DIR/vocab.json ($VOCAB_SIZE)"
    fi
    
    echo ""
    print_success "You can now use these datasets to train your Transformer model!"
    print_info "Next steps:"
    echo "  1. Check the generated files in $OUTPUT_DIR"
    echo "  2. Use the training script: python train.py --data_dir $OUTPUT_DIR"
    echo "  3. Monitor training progress and adjust hyperparameters as needed"
    
else
    print_error "Dataset generation failed!"
    exit 1
fi 