#!/usr/bin/env python3
"""
Run controller for the Latent Semantic Search Engine.
This script launches both the Flask backend and React frontend using a simple approach.
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path

# Global variables for process tracking
processes = []

def cleanup_processes():
    """Terminate all running processes"""
    for process in processes:
        try:
            if process.poll() is None:  # If process is still running
                process.terminate()
                process.wait(timeout=5)
        except:
            # Force kill if termination fails
            try:
                process.kill()
            except:
                pass

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\nShutting down services...")
    cleanup_processes()
    sys.exit(0)

def run_app(debug=False):
    """Run both frontend and backend"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Path setup
    project_dir = Path(__file__).parent.absolute()
    frontend_dir = project_dir / "frontend"
    backend_file = project_dir / "app.py"
    
    # Start Flask backend
    print("Starting Flask backend server...")
    backend_cmd = [sys.executable, str(backend_file), "--port", "5001"]
    if debug:
        backend_cmd.append("--debug")
    
    backend_process = subprocess.Popen(backend_cmd)
    processes.append(backend_process)
    
    # Give backend time to start
    time.sleep(2)
    
    # Start frontend
    print("Starting React frontend...")
    # Check if npm is installed
    try:
        npm_process = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if npm_process.returncode != 0:
            raise Exception("npm not found")
    except:
        print("Error: npm is not installed or not in PATH. Please install Node.js and npm.")
        cleanup_processes()
        sys.exit(1)
    
    # Run npm dev server
    os.chdir(frontend_dir)
    frontend_process = subprocess.Popen(["npm", "run", "dev"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    processes.append(frontend_process)
    
    # Wait a moment for the frontend to start and capture its output to get the URL
    time.sleep(3)
    
    # Default frontend URL
    frontend_url = "http://localhost:5173"
    
    # Try to read the actual URL from the Vite output
    if frontend_process.stdout:
        for i in range(20):  # Try reading a few lines
            line = frontend_process.stdout.readline().strip()
            if "Local:" in line and "http" in line:
                # Extract URL from something like "Local:   http://localhost:5173/"
                parts = line.split("Local:")
                if len(parts) > 1:
                    url_part = parts[1].strip()
                    if url_part.startswith("http"):
                        frontend_url = url_part
                        break
            if not line and i > 5:
                break
            time.sleep(0.1)
    
    print("\n")
    print("="*80)
    print("="*80)
    print("                   APPLICATION IS NOW RUNNING!")
    print("="*80)
    print("="*80)
    print("\n")
    print("  IMPORTANT: USE THIS URL TO ACCESS THE SEARCH ENGINE:")
    print("\n")
    print("  >>>>>>>  " + frontend_url + "  <<<<<<<")
    print("\n")
    print("  COPY AND PASTE THIS URL INTO YOUR BROWSER")
    print("\n")
    print("  DO NOT use the backend URLs below that show 'Running on http://...'")
    print("  Those are only for API access and won't display the search interface")
    print("\n")
    print("="*80)
    print("Press Ctrl+C to stop all services")
    print("="*80)
    
    # Keep the script running and monitor child processes
    try:
        while True:
            # Exit if either process has terminated
            if backend_process.poll() is not None:
                print("Backend process has stopped. Shutting down...")
                break
            if frontend_process.poll() is not None:
                print("Frontend process has stopped. Shutting down...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_processes()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run the Latent Semantic Search Engine")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    run_app(debug=args.debug)

if __name__ == "__main__":
    main() 