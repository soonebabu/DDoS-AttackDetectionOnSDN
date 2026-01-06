"""
Utility functions for DDoS detection system
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name.upper()} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{model_name.upper()} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f'results/{model_name}_training_history.png'
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Training plot saved to {plot_filename}")

def ensure_directory(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)