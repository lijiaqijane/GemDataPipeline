#!/bin/bash

# Entity Extractor Runner Script
# This script demonstrates how to run the entity extractor on crawled repositories

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Entity Extractor Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default values
CONFIG_PATH="examples/config/code_agent.yaml"
REPO_DIR="taskdb/code_agent/crawled_repos"
OUTPUT_DIR="taskdb/code_agent/extracted_entities"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="--repo-dir $2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="--output-dir $2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config CONFIG_PATH       Path to configuration file"
            echo "  --repo-dir REPO_DIR        Directory containing crawled repositories"
            echo "  --output-dir OUTPUT_DIR    Directory to save extracted entities"
            echo "  --verbose, -v              Enable verbose logging"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --verbose"
            echo "  $0 --config my_config.yaml --output-dir /tmp/entities"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration file: $CONFIG_PATH${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"

# Run the entity extractor
echo -e "${BLUE}Starting entity extraction...${NC}"
echo ""

python3 examples/code_agent/run_entity_extractor.py --config "$CONFIG_PATH" --repo-dir $REPO_DIR --output-dir $OUTPUT_DIR --verbose

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Entity extraction completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Entity extraction failed with exit code $exit_code${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $exit_code
