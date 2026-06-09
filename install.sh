#!/bin/bash

# --- Color Definitions for Premium Look ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}          Lingua Franca Auto-Installer          ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH.${NC}"
    echo -e "Please install Python 3 and try again."
    exit 1
fi

# Detect uv package manager (extremely fast, preferred if available)
USE_UV=false
if command -v uv &> /dev/null; then
    USE_UV=true
    UV_PATH=$(command -v uv)
elif [ -f "$HOME/.local/bin/uv" ]; then
    USE_UV=true
    UV_PATH="$HOME/.local/bin/uv"
fi

if [ "$USE_UV" = true ]; then
    echo -e "${GREEN}[✔] Found uv package manager at: ${UV_PATH}${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo -e "${BLUE}Creating virtual environment using uv...${NC}"
        "$UV_PATH" venv
    else
        echo -e "${YELLOW}Virtual environment .venv already exists. Skipping creation.${NC}"
    fi
    
    # Install dependencies
    echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
    "$UV_PATH" pip install -r requirements.txt
else
    echo -e "${YELLOW}[!] uv not found. Falling back to standard venv and pip...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo -e "${BLUE}Creating virtual environment using venv...${NC}"
        python3 -m venv .venv
    else
        echo -e "${YELLOW}Virtual environment .venv already exists. Skipping creation.${NC}"
    fi
    
    # Install dependencies
    echo -e "${BLUE}Installing dependencies...${NC}"
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}          Setup Completed Successfully!         ${NC}"
echo -e "${GREEN}===============================================${NC}"
echo -e "To start the application, run:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo -e "  ${YELLOW}python request.py${NC}"
echo -e "==============================================="
