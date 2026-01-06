import os
import sys

def main():
    """Main function - prints setup instructions"""
    
    print("\n" + "="*60)
    print("SDN DDoS Detection System")
    print("="*60)
    print("\nSETUP INSTRUCTIONS:")
    print("="*60)
    
    print("\n1. FIRST, Download CIC DDoS 2019 dataset:")
    print("   - Place it in: data/raw/CIC_DDoS_2019.csv")
    
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("   pip install -r ryu_controller/requirements.txt")
    
    print("\n3. Run preprocessing:")
    print("   python src/preprocess.py")
    
    print("\n4. Train models:")
    print("   python src/train_models.py")
    
    print("\n5. Start the complete system:")
    print("   ./scripts/start_all.sh")
    
    print("\n6. Access dashboard:")
    print("   Open: http://localhost:5000")
    
    print("\n" + "="*60)
    print("MANUAL START (if needed):")
    print("="*60)
    
    print("\nTerminal 1 - Ryu controller:")
    print("   ryu-manager ryu_controller/ddos_detector.py --verbose")
    
    print("\nTerminal 2 - Mininet:")
    print("   sudo python mininet_scripts/network.py")
    
    print("\nTerminal 3 - Dashboard:")
    print("   python web_dashboard/app.py")
    
    print("\n" + "="*60)
    
    # Create directories if they don't exist
    print("\nCreating necessary directories...")
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/trained", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("\nDirectories created successfully!")
    print("\nFollow the instructions above to set up and run the system.")

if __name__ == "__main__":
    main()