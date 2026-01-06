"""
Web dashboard for monitoring DDoS detection
"""

from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    stats_file = 'logs/stats.json'
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {
            'total_packets': 0,
            'attack_packets': 0,
            'normal_packets': 0,
            'blocked_flows': 0,
            'active_flows': 0
        }
    
    return jsonify(stats)

@app.route('/api/attacks')
def get_attacks():
    """Get recent attacks"""
    attacks_file = 'logs/attacks.json'
    
    if os.path.exists(attacks_file):
        with open(attacks_file, 'r') as f:
            attacks = json.load(f)
        recent = attacks[-10:]  # Last 10 attacks
    else:
        recent = []
    
    return jsonify(recent)

@app.route('/api/models')
def get_models():
    """Get model information"""
    models = {}
    
    model_files = ['lstm_model.h5', 'gru_model.h5', 'bilstm_model.h5', 'lstm_gru_model.h5']
    
    for model_file in model_files:
        model_path = f'models/trained/{model_file}'
        model_name = model_file.replace('_model.h5', '').upper()
        models[model_name] = os.path.exists(model_path)
    
    return jsonify(models)

if __name__ == '__main__':
    print("Starting DDoS Detection Dashboard...")
    print("Dashboard URL: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)