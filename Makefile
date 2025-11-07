# Makefile for BRCA Classification Project
# Usage: make <target>

.PHONY: help setup test validate train monitor analyze inference clean

# Default target
help:
	@echo "BRCA Classification - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup & Testing:"
	@echo "  make setup       - Install dependencies and setup environment"
	@echo "  make test        - Run installation tests"
	@echo "  make validate    - Validate data structure and quality"
	@echo ""
	@echo "Training:"
	@echo "  make train       - Start model training"
	@echo "  make monitor     - Monitor training progress"
	@echo "