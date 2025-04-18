#!/usr/bin/env python3
"""
Demo Website Server for Credit Card Fraud Detection System

This script runs a simple HTTP server to serve the demo website and
launches the Streamlit dashboards in separate processes.
"""

import os
import sys
import time
import argparse
import subprocess
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run the Credit Card Fraud Detection Demo')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port for the demo website (default: 8000)')
    parser.add_argument('--dashboard1-port', type=int, default=8501,
                        help='Port for the Transaction Monitoring Dashboard (default: 8501)')
    parser.add_argument('--dashboard2-port', type=int, default=8502,
                        help='Port for the Fraud Investigation Dashboard (default: 8502)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')
    return parser.parse_args()

def start_http_server(port, directory):
    """Start HTTP server for the demo website"""
    os.chdir(directory)
    
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    
    httpd = HTTPServer(('', port), Handler)
    print(f"[+] Demo website server started at http://localhost:{port}")
    return httpd

def start_streamlit_dashboard(script_path, port):
    """Start a Streamlit dashboard in a separate process"""
    cmd = [
        'streamlit', 'run', script_path,
        '--server.port', str(port),
        '--server.headless', 'true',
        '--browser.serverAddress', 'localhost',
        '--browser.gatherUsageStats', 'false'
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment to ensure the process starts
    time.sleep(2)
    
    # Check if process is still running
    if process.poll() is not None:
        print(f"[!] Failed to start dashboard: {script_path}")
        stdout, stderr = process.communicate()
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return None
    
    print(f"[+] Dashboard started at http://localhost:{port}")
    return process

def main():
    """Main function"""
    args = parse_args()
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Start the Streamlit dashboards
    dashboard1_path = os.path.join(base_dir, 'src', 'dashboard', 'transaction_monitoring.py')
    dashboard2_path = os.path.join(base_dir, 'src', 'dashboard', 'fraud_investigation.py')
    
    print("[*] Starting Transaction Monitoring Dashboard...")
    dashboard1_process = start_streamlit_dashboard(dashboard1_path, args.dashboard1_port)
    
    print("[*] Starting Fraud Investigation Dashboard...")
    dashboard2_process = start_streamlit_dashboard(dashboard2_path, args.dashboard2_port)
    
    # Start the demo website server
    print("[*] Starting demo website server...")
    website_dir = os.path.join(base_dir, 'demo_website')
    httpd = start_http_server(args.port, website_dir)
    
    # Open browser if requested
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    
    try:
        # Run the server until interrupted
        print("[*] Press Ctrl+C to stop the servers")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down servers...")
        
        # Stop the HTTP server
        httpd.shutdown()
        
        # Terminate the Streamlit processes
        if dashboard1_process:
            dashboard1_process.terminate()
        
        if dashboard2_process:
            dashboard2_process.terminate()
        
        print("[+] All servers stopped")

if __name__ == "__main__":
    main()
