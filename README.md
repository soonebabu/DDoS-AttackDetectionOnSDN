# SDN DDoS Detection System

A complete system for detecting DDoS attacks in Software-Defined Networks using deep learning models.

## Features
- Real-time DDoS detection in SDN environments
- Four deep learning models: LSTM, GRU, BiLSTM, and LSTM-GRU
- LSTM-GRU selected as best model for deployment
- Integration with Ryu SDN controller
- Web dashboard for real-time monitoring
- Automatic attack mitigation

# DDoS Attack Detection on Software-Defined Networks (SDN) using Deep Learning

This project implements a **Deep Learning–based DDoS attack detection system for Software-Defined Networks (SDN)**.  
It integrates **SDN controllers (Ryu)**, **network emulation (Mininet)**, and **machine learning models** to identify and mitigate Distributed Denial-of-Service (DDoS) attacks in real time.

The system collects network flow statistics from the SDN controller, preprocesses the data, trains deep learning models, and detects malicious traffic patterns indicative of DDoS attacks. A lightweight **web dashboard** is also provided for monitoring and visualization.

---

## 🛠️ Technology Stack

- Python 3
- Software-Defined Networking (SDN)
- Ryu Controller
- Mininet
- Deep Learning (TensorFlow / PyTorch – configurable)
- YAML for configuration
- Flask (Web Dashboard)
- Ubuntu / Linux (recommended)

---

## 📂 Project Structure

```text
sdn-ddos-detection/
│
├── main.py                         # Main entry point
├── README.md                       # Project documentation
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/                        # Configuration files
│   ├── model_config.yaml
│   └── network_config.yaml
│
├── data/
│   └── processed/
│       └── README.md
│
├── logs/                           # Logs generated during execution
│   └── README.md
│
├── models/
│   └── trained/                   # Saved trained models
│       └── README.md
│
├── mininet_scripts/               # Mininet topology and startup scripts
│   ├── network.py
│   └── start_mininet.sh
│
├── ryu_controller/                # Ryu SDN controller logic
│   ├── __init__.py
│   ├── ddos_detector.py
│   └── requirements.txt
│
├── scripts/                       # Helper scripts
│   ├── setup.sh
│   └── start_all.sh
│
├── src/                           # Core ML pipeline
│   ├── __init__.py
│   ├── models.py
│   ├── preprocess.py
│   ├── train_models.py
│   └── utils.py
│
└── web_dashboard/                 # Monitoring dashboard
    ├── app.py
    ├── requirements.txt
    ├── static/
    │   ├── css/style.css
    │   └── js/dashboard.js
    └── templates/
        └── index.html


Research from the paper
Distributed Denial of Service Attack Detection on Software De􀁼ned Networking Using Deep Learning
http://conference.ioe.edu.np/ioegc10/papers/ioegc-10-093-10127.pdf

System Requirements

Ubuntu 20.04 / 22.04 (recommended)

Python 3.8 or higher

Mininet

Ryu SDN Controller

pip

sudo privileges (for Mininet)

🚀 Installation & Setup (Ubuntu)
1️⃣ Update System and Install Dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git mininet


Verify installation:

python3 --version
pip3 --version
mn --version

2️⃣ Clone the Repository
git clone git@github.com:soonebabu/sdn-ddos-detection.git
cd sdn-ddos-detection

3️⃣ Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate

4️⃣ Install Python Dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ryu_controller/requirements.txt
pip install -r web_dashboard/requirements.txt

🧠 Model Training

To preprocess data and train the deep learning model:

python src/train_models.py


Trained models will be saved in:

models/trained/

🌐 Running the SDN Environment
1️⃣ Start Ryu Controller
ryu-manager ryu_controller/ddos_detector.py

2️⃣ Start Mininet Topology (New Terminal)
sudo bash mininet_scripts/start_mininet.sh

3️⃣ Run the Main Detection System
python main.py

📊 Web Dashboard

Start the dashboard:

python web_dashboard/app.py


Access it in your browser:

http://127.0.0.1:5000/


The dashboard displays:

Traffic statistics

Detected attacks

Model predictions

🔍 Detection Workflow

Mininet generates network traffic

Ryu controller collects flow statistics

Data is preprocessed and normalized

Deep learning model classifies traffic

DDoS attacks are detected in real time

Results are logged and visualized

📈 Expected Results

Accurate detection of DDoS traffic patterns

Improved network resilience using SDN control

Real-time monitoring and analytics

Scalable and modular design

Mininet Permission Error
sudo mn -c

Missing Python Modules
pip install -r requirements.txt


This project is inspired by academic research on DDoS detection in SDN using deep learning, combining centralized SDN control with data-driven security intelligence.


