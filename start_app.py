#!/usr/bin/env python3
"""
Multi-Model application starter for GIS & Remote Sensing Assistant.
Supports multiple AI providers and models.
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from app.main_comprehensive import UniversalAIProvider

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check which API keys are available."""
    api_keys = {
        "xai": os.getenv("XAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "zhipu": os.getenv("ZHIPU_API_KEY")
    }

    available_providers = []
    for provider, key in api_keys.items():
        if key:
            available_providers.append(provider)

    return available_providers, api_keys

def start_streamlit():
    """Start the Streamlit application."""
    print("Starting Multi-Model GIS & Remote Sensing Assistant...")
    print("=" * 60)

    available_providers, api_keys = check_api_keys()

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

    print("\n[INFO] COMPREHENSIVE FEATURES:")
    print("[OK] Multiple AI providers (XAI, Google, Anthropic, OpenAI, Zhipu)")
    print("[OK] 20+ models to choose from")
    print("[OK] Real-time model switching")
    print("[OK] Dark, high-contrast chat interface")
    print("[OK] Enhanced document retrieval")
    print("[OK] Sample questions to try")
    print("[OK] Provider badges and model information")

    # Count total models available
    total_models = 0
    provider = UniversalAIProvider("dummy", "dummy", "xai")
    all_models = provider.get_all_available_models()
    for provider_models in all_models.values():
        total_models += len(provider_models)

    print(f"\n[STATS] Total Available Models: {total_models}")

    print("\n[PROVIDERS] MODEL PROVIDERS:")
    for provider_name, models in all_models.items():
        if provider_name in available_providers:
            print(f"   {provider_name.title()}: {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                print(f"      • {model['display_name']}")

    print("\n[USAGE] TO USE:")
    print("1. Choose your preferred AI provider in the sidebar")
    print("2. Select a model from the dropdown")
    print("   • Adjust settings (temperature, max tokens)")
    print("4. Start asking GIS questions!")

    print("\nStarting Comprehensive Multi-Model Application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main_comprehensive.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install streamlit openai python-dotenv pandas anthropic google-generativeai")

if __name__ == "__main__":
    start_streamlit()