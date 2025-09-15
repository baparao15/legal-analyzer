#!/usr/bin/env python3
"""
Robust startup script for the Legal Document Analyzer
Handles PyTorch/Streamlit compatibility issues
"""

import os
import sys
import subprocess
import warnings

# Suppress all warnings that might cause issues
warnings.filterwarnings("ignore")

# Set environment variables to prevent PyTorch issues
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def main():
    """Start the Streamlit application with proper configuration"""
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Set up the command
    cmd = [
        sys.executable, 
        "-m", "streamlit", 
        "run", 
        "app.py",
        "--server.fileWatcherType=poll",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    print("Starting Legal Document Analyzer...")
    print("This may take a moment to load all dependencies...")
    
    try:
        # Start the application
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Trying alternative startup method...")
        
        # Alternative method
        try:
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", "app.py", 
                       "--server.fileWatcherType=poll",
                       "--server.headless=true"]
            stcli.main()
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("Please try running: streamlit run app.py --server.fileWatcherType=poll")

if __name__ == "__main__":
    main()
