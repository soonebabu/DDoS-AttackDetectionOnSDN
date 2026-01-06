#!/bin/bash

# Setup script for SDN DDoS Detection System

echo "Setting up SDN DDoS Detection System..."
echo "========================================"

# Create directories
echo "Creating directories..."
mkdir -p data/{raw,processed}
mkdir -p models/trained
mkdir -p logs
mkdir -p results

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install -r ryu_controller/requirements.txt
pip install -r web_dashboard/requirements.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x mininet_scripts/start_mininet.sh
chmod +x scripts/*.sh 2>/dev/null

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download CIC DDoS 2019 dataset to data/raw/"
echo "2. Run preprocessing: python src/preprocess.py"
echo "3. Train models: python src/train_models.py"
echo "4. Start system: ./scripts/start_all.sh"