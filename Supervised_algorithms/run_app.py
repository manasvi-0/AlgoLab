#!/usr/bin/env python3
"""
Supervised Learning Visualizer - Startup Script
Choose your preferred frontend interface
"""

import os
import sys
import subprocess
import webbrowser
import time

def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("🤖 Supervised Learning Visualizer")
    print("=" * 60)
    print("Choose your preferred frontend interface:")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import flask
        import plotly
        import sklearn
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit app...")
    print("📱 Opening browser to http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    import threading
    threading.Thread(target=open_browser).start()
    
    # Run streamlit
    subprocess.run(["streamlit", "run", "app.py"])

def run_flask():
    """Run the Flask application"""
    print("🌐 Starting Flask web app...")
    print("📱 Opening browser to http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:5000")
    
    import threading
    threading.Thread(target=open_browser).start()
    
    # Run flask
    subprocess.run([sys.executable, "web_app.py"])

def run_demo():
    """Run the Python demo"""
    print("🐍 Running Python demo...")
    print()
    subprocess.run([sys.executable, "demo.py"])

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    while True:
        print("Available options:")
        print("1. 🚀 Streamlit App (Recommended) - Interactive web interface")
        print("2. 🌐 Flask Web App - Pure HTML/CSS/JavaScript frontend")
        print("3. 🐍 Python Demo - Command line interface")
        print("4. ❌ Exit")
        print()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            run_streamlit()
            break
        elif choice == "2":
            run_flask()
            break
        elif choice == "3":
            run_demo()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your installation and try again.")
