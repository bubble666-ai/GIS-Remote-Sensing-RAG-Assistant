#!/usr/bin/env python3
"""
Installation script for GIS RAG Assistant.
Handles core dependencies and optional model providers.
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def install_requirements():
    """Install core requirements from requirements.txt."""
    print("🚀 Installing core dependencies...")

    core_packages = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "langchain-openai>=0.0.2",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "openai>=1.3.0",
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "PyPDF2>=3.0.1",
        "python-docx>=0.8.11",
        "unstructured>=0.10.0",
        "tiktoken>=0.5.1",
        "pandas>=2.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "numpy>=1.24.0",
        "nltk>=3.8.1",
        "chromadb>=0.4.15",
        "watchdog>=3.0.0",
        "google-generativeai>=0.3.0",
        "anthropic>=0.8.0",
        "requests>=2.31.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "google-api-python-client>=2.108.0",
        "langchain-google-genai>=0.0.6",
        "langchain-anthropic>=0.0.4"
    ]

    success_count = 0
    for package in core_packages:
        if install_package(package):
            success_count += 1

    print(f"\n📊 Core installation summary: {success_count}/{len(core_packages)} packages installed successfully")
    return success_count == len(core_packages)


def install_optional_providers():
    """Install optional model providers."""
    print("\n🔧 Installing optional model providers...")

    optional_packages = [
        ("xai", "XAI (Grok) models"),
        ("zhipuai>=2.0.0", "Zhipu AI (GLM) models")
    ]

    installed_optional = []
    for package, description in optional_packages:
        print(f"\nInstalling {description} ({package})...")
        if install_package(package):
            installed_optional.append(description)

    if installed_optional:
        print(f"\n✅ Optional providers installed: {', '.join(installed_optional)}")
    else:
        print("\n⚠️ No optional providers were installed")

    return installed_optional


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
        return True


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = ".env"
    env_example = ".env.example"

    if not os.path.exists(env_file) and os.path.exists(env_example):
        print("\n📝 Creating .env file from template...")
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ .env file created successfully")
            print("📝 Please edit .env file with your API keys")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")


def main():
    """Main installation function."""
    print("🌍 GIS & Remote Sensing RAG Assistant - Installation Script")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install core requirements
    if not install_requirements():
        print("\n❌ Core installation failed. Please check the errors above.")
        sys.exit(1)

    # Install optional providers
    optional_providers = install_optional_providers()

    # Create .env file
    create_env_file()

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Installation completed!")
    print("\n📝 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Add your GIS documents to the 'data/' folder")
    print("3. Run: streamlit run app/main.py")

    if optional_providers:
        print(f"\n✅ Optional model providers available: {', '.join(optional_providers)}")
    else:
        print("\n⚠️ Only basic model providers (OpenAI, Anthropic, Google) are available")
        print("   To install additional providers, run:")
        print("   pip install xai zhipuai>=2.0.0")

    print("\n🚀 Ready to start!")


if __name__ == "__main__":
    main()