#!/usr/bin/env python3
"""
Multi-Model application starter for GIS & Remote Sensing Assistant.
Supports multiple AI providers and models.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_keys():
    """Return the list of providers whose API key is configured."""
    provider_keys = {
        "xai": "XAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
    }
    return [p for p, var in provider_keys.items() if os.getenv(var)]

def start_streamlit():
    """Start the Streamlit application."""
    print("Starting Multi-Model GIS & Remote Sensing Assistant...")
    print("=" * 60)

    available_providers = check_api_keys()

    if available_providers:
        print(f"[OK] Available AI Providers: {', '.join(available_providers).title()}")
        print(f"Total providers configured: {len(available_providers)}")
    else:
        print("[ERROR] No API keys found. Please add API keys to .env file:")
        print("   - XAI_API_KEY=your_key_here")
        print("   - GOOGLE_API_KEY=your_key_here")
        print("   - ANTHROPIC_API_KEY=your_key_here")
        print("   - OPENAI_API_KEY=your_key_here")
        print("   - ZHIPU_API_KEY=your_key_here")
        return

    print("\nStarting Streamlit app...")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install streamlit openai python-dotenv pandas anthropic google-generativeai")

if __name__ == "__main__":
    start_streamlit()