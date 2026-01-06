# SDN DDoS Detection System

A complete system for detecting DDoS attacks in Software-Defined Networks using deep learning models.

## Features
- Real-time DDoS detection in SDN environments
- Four deep learning models: LSTM, GRU, BiLSTM, and LSTM-GRU
- LSTM-GRU selected as best model for deployment
- Integration with Ryu SDN controller
- Web dashboard for real-time monitoring
- Automatic attack mitigation

## Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository-url>
cd sdn-ddos-detection
./scripts/setup.sh