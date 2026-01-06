"""
Mininet SDN testbed setup
Creates network with 3 switches and 18 hosts
"""

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import threading

class SDNNetwork:
    """Create SDN testbed network"""
    
    def __init__(self):
        self.net = None
        
    def create_network(self):
        """Create the network topology"""
        info('*** Creating SDN testbed network\n')
        
        self.net = Mininet(
            controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True
        )
        
        info('*** Adding Ryu controller\n')
        self.net.addController('c0')
        
        s1 = self.net.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.net.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.net.addSwitch('s3', protocols='OpenFlow13')
        
        info('*** Creating 18 hosts\n')
        hosts = []
        
        for i in range(1, 19):
            ip = f'10.0.0.{i}'
            mac = f'00:00:00:00:00:{i:02d}'
            host = self.net.addHost(f'h{i}', ip=ip, mac=mac)
            hosts.append(host)
        
        info('*** Connecting hosts to switches\n')
        for i in range(6):
            self.net.addLink(s1, hosts[i])
        
        for i in range(6, 12):
            self.net.addLink(s2, hosts[i])
        
        for i in range(12, 18):
            self.net.addLink(s3, hosts[i])
        
        info('*** Connecting switches\n')
        self.net.addLink(s1, s2)
        self.net.addLink(s2, s3)
        
        info('*** Network created successfully\n')
        return self.net
    
    def start_network(self):
        """Start the network"""
        if self.net is None:
            self.create_network()
        
        info('*** Starting network\n')
        self.net.start()
        
        info('*** Testing connectivity\n')
        self.net.pingAll()
        
        self.configure_hosts()
        
        info('*** Network ready\n')
    
    def configure_hosts(self):
        """Configure network services on hosts"""
        info('*** Configuring hosts\n')
        
        victim = self.net.get('h10')
        info(f'*** Starting web server on h10 ({victim.IP()})\n')
        victim.cmd('python3 -m http.server 80 &')
        info(f'  Web server running on {victim.IP()}:80\n')
        
        self.start_normal_traffic()
        
        info('*** Host configuration complete\n')
    
    def start_normal_traffic(self):
        """Start normal background traffic"""
        info('*** Starting normal background traffic\n')
        
        victim_ip = self.net.get('h10').IP()
        normal_hosts = ['h1', 'h2', 'h3', 'h4', 'h5']
        
        for host_name in normal_hosts:
            host = self.net.get(host_name)
            
            cmd = f'''
            while true; do
                curl -s http://{victim_ip}:80 > /dev/null
                sleep $((RANDOM % 10 + 5))
            done &
            '''
            
            host.cmd(cmd)
            info(f'  Started background traffic on {host_name}\n')
    
    def start_attacks(self, delay=30):
        """Start DDoS attacks after delay"""
        info(f'*** Starting DDoS attacks in {delay} seconds\n')
        
        attack_thread = threading.Thread(target=self.execute_attacks, args=(delay,))
        attack_thread.daemon = True
        attack_thread.start()
    
    def execute_attacks(self, delay):
        """Execute DDoS attacks"""
        time.sleep(delay)
        
        victim = self.net.get('h10')
        victim_ip = victim.IP()
        
        info(f'\n*** Starting DDoS attacks on {victim_ip}\n')
        
        attackers = ['h15', 'h16', 'h17', 'h18']
        
        for attacker_name in attackers:
            attacker = self.net.get(attacker_name)
            
            syn_cmd = f'hping3 -S -p 80 --flood {victim_ip} > /dev/null 2>&1 &'
            attacker.cmd(syn_cmd)
            info(f'  {attacker_name}: Started TCP SYN Flood\n')
            time.sleep(1)
            
            udp_cmd = f'hping3 -2 --flood --rand-source {victim_ip} > /dev/null 2>&1 &'
            attacker.cmd(udp_cmd)
            info(f'  {attacker_name}: Started UDP Flood\n')
            time.sleep(1)
        
        info('*** All attacks started\n')
    
    def run_cli(self):
        """Start Mininet CLI"""
        info('*** Starting Mininet CLI\n')
        info('Type "exit" to stop\n')
        CLI(self.net)
    
    def stop_network(self):
        """Stop the network"""
        if self.net is not None:
            info('*** Stopping network\n')
            self.net.stop()
            info('*** Network stopped\n')

def main():
    """Main function to run network"""
    setLogLevel('info')
    
    network = SDNNetwork()
    
    try:
        network.start_network()
        network.start_attacks(delay=30)
        network.run_cli()
    
    except KeyboardInterrupt:
        info('\n*** Stopping by user request\n')
    
    finally:
        network.stop_network()

if __name__ == '__main__':
    main()