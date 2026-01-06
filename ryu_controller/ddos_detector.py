"""
RYU Controller with LSTM-GRU DDoS Detection
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
from ryu.lib import hub

import tensorflow as tf
import numpy as np
import pickle
import json
import time
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DDoSDetector(app_manager.RyuApp):
    """
    RYU controller that detects DDoS attacks using LSTM-GRU model
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(DDoSDetector, self).__init__(*args, **kwargs)
        
        print("\n" + "="*60)
        print("SDN DDoS DETECTOR - RYU CONTROLLER")
        print("Using LSTM-GRU Model")
        print("="*60)
        
        self.flow_stats = {}
        self.attack_log = []
        self.blocked_flows = set()
        
        self.model = None
        self.scaler = None
        self.load_model()
        
        self.detection_threshold = 0.7
        self.flow_timeout = 60
        self.sampling_interval = 5
        
        self.stats = {
            'total_packets': 0,
            'attack_packets': 0,
            'normal_packets': 0,
            'blocked_flows': 0,
            'active_flows': 0
        }
        
        self.monitor_thread = hub.spawn(self.monitor_flows)
        
        print("\nController initialized successfully!")
        print("Waiting for switch connections...")
    
    def load_model(self):
        """Load only LSTM-GRU model"""
        print("\nLoading LSTM-GRU model...")
        
        models_dir = 'models/trained'
        
        try:
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Scaler loaded")
            else:
                print("✗ Scaler file not found")
                return
            
            model_path = os.path.join(models_dir, 'lstm_gru_model.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("✓ LSTM-GRU model loaded")
            else:
                print("✗ LSTM-GRU model not found")
                print("Please train models first: python src/train_models.py")
        
        except Exception as e:
            print(f"Error loading model: {e}")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                         ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        print(f"\n✓ Switch {datapath.id} connected")
    
    def add_flow(self, datapath, priority, match, actions):
        """Add flow entry to switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                            actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                               match=match, instructions=inst)
        
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets"""
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        if not eth:
            return
        
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt:
            return
        
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        
        src_ip = ip_pkt.src
        dst_ip = ip_pkt.dst
        protocol = ip_pkt.proto
        
        src_port = 0
        dst_port = 0
        if tcp_pkt:
            src_port = tcp_pkt.src_port
            dst_port = tcp_pkt.dst_port
        elif udp_pkt:
            src_port = udp_pkt.src_port
            dst_port = udp_pkt.dst_port
        
        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        
        self.update_flow_stats(flow_key, len(msg.data), in_port)
        
        self.stats['total_packets'] += 1
        
        features = self.extract_flow_features(flow_key)
        if features is not None and self.model is not None:
            self.detect_ddos(features, flow_key, datapath)
    
    def update_flow_stats(self, flow_key, packet_size, in_port):
        """Update statistics for a flow"""
        current_time = time.time()
        
        if flow_key not in self.flow_stats:
            self.flow_stats[flow_key] = {
                'start_time': current_time,
                'last_seen': current_time,
                'packet_count': 1,
                'byte_count': packet_size,
                'packet_sizes': [packet_size],
                'timestamps': [current_time],
                'in_port': in_port
            }
        else:
            flow = self.flow_stats[flow_key]
            flow['packet_count'] += 1
            flow['byte_count'] += packet_size
            flow['packet_sizes'].append(packet_size)
            flow['timestamps'].append(current_time)
            flow['last_seen'] = current_time
    
    def extract_flow_features(self, flow_key):
        """Extract features from flow"""
        if flow_key not in self.flow_stats:
            return None
        
        flow = self.flow_stats[flow_key]
        current_time = time.time()
        
        recent_indices = []
        for i, timestamp in enumerate(flow['timestamps']):
            if current_time - timestamp <= 10:
                recent_indices.append(i)
        
        if len(recent_indices) < 2:
            return None
        
        recent_packets = [flow['packet_sizes'][i] for i in recent_indices]
        
        flow_duration = current_time - flow['start_time']
        total_packets = len(recent_packets)
        total_bytes = sum(recent_packets)
        
        features = np.array([
            flow_duration,
            total_packets,
            0,
            total_bytes / max(flow_duration, 0.001),
            total_packets / max(flow_duration, 0.001),
            np.mean(recent_packets),
            0,
            np.mean(recent_packets),
            np.std(recent_packets) if len(recent_packets) > 1 else 0,
            np.mean(recent_packets),
            total_bytes,
            0,
            1460,
            0
        ])
        
        return features.reshape(1, -1)
    
    def detect_ddos(self, features, flow_key, datapath):
        """Use LSTM-GRU model to detect DDoS attacks"""
        try:
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            features_reshaped = features_scaled.reshape(1, 1, -1)
            
            prediction = self.model.predict(features_reshaped, verbose=0)[0][0]
            
            if prediction > self.detection_threshold:
                self.handle_attack(flow_key, prediction, datapath)
        
        except Exception as e:
            print(f"Error in DDoS detection: {e}")
    
    def handle_attack(self, flow_key, probability, datapath):
        """Handle detected DDoS attack"""
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key
        flow = self.flow_stats.get(flow_key, {})
        
        attack_record = {
            'timestamp': datetime.now().isoformat(),
            'source_ip': src_ip,
            'destination_ip': dst_ip,
            'source_port': src_port,
            'destination_port': dst_port,
            'protocol': 'TCP' if protocol == 6 else 'UDP' if protocol == 17 else 'Other',
            'probability': float(probability),
            'model': 'LSTM-GRU',
            'packet_count': flow.get('packet_count', 0),
            'byte_count': flow.get('byte_count', 0)
        }
        
        self.attack_log.append(attack_record)
        self.stats['attack_packets'] += flow.get('packet_count', 0)
        
        if flow_key not in self.blocked_flows:
            self.block_flow(flow_key, datapath)
            self.blocked_flows.add(flow_key)
            self.stats['blocked_flows'] += 1
        
        self.log_attack(attack_record)
        
        print(f"\n{'!'*60}")
        print("🚨 DDoS ATTACK DETECTED!")
        print(f"{'!'*60}")
        print(f"Time: {attack_record['timestamp']}")
        print(f"Source: {src_ip}:{src_port}")
        print(f"Target: {dst_ip}:{dst_port}")
        print(f"Protocol: {attack_record['protocol']}")
        print(f"Model: LSTM-GRU")
        print(f"Probability: {probability:.2%}")
        print(f"Flow blocked: ✓")
        print(f"{'!'*60}\n")
    
    def block_flow(self, flow_key, datapath):
        """Block malicious flow"""
        src_ip, dst_ip, src_port, dst_port, protocol = flow_key
        
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        if protocol == 6:
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip,
                ip_proto=protocol,
                tcp_src=src_port,
                tcp_dst=dst_port
            )
        elif protocol == 17:
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip,
                ip_proto=protocol,
                udp_src=src_port,
                udp_dst=dst_port
            )
        else:
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip,
                ip_proto=protocol
            )
        
        actions = []
        self.add_flow(datapath, 65535, match, actions)
        
        print(f"Blocked flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port}")
    
    def log_attack(self, attack_record):
        """Save attack information to log file"""
        log_file = 'logs/attacks.json'
        
        try:
            os.makedirs('logs', exist_ok=True)
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(attack_record)
            
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        
        except Exception as e:
            print(f"Error logging attack: {e}")
    
    def monitor_flows(self):
        """Background thread to monitor flows"""
        while True:
            hub.sleep(self.sampling_interval)
            
            self.stats['normal_packets'] = self.stats['total_packets'] - self.stats['attack_packets']
            
            current_time = time.time()
            active_flows = 0
            flows_to_remove = []
            
            for flow_key, flow_data in self.flow_stats.items():
                if current_time - flow_data['last_seen'] <= self.flow_timeout:
                    active_flows += 1
                else:
                    flows_to_remove.append(flow_key)
            
            self.stats['active_flows'] = active_flows
            
            for flow_key in flows_to_remove:
                del self.flow_stats[flow_key]
            
            if self.stats['total_packets'] % 100 == 0:
                print(f"[STATS] Total: {self.stats['total_packets']:,} | "
                      f"Normal: {self.stats['normal_packets']:,} | "
                      f"Attack: {self.stats['attack_packets']:,} | "
                      f"Blocked: {self.stats['blocked_flows']:,} | "
                      f"Active: {self.stats['active_flows']:,}")