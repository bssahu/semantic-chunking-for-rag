#!/usr/bin/env python3
"""
Dependency checker for the Semantic Document Chunking and RAG application.
This script checks if all required dependencies are properly installed.
"""

import importlib
import subprocess
import sys
import os
from pathlib import Path

def check_python_package(package_name, min_version=None):
    """Check if a Python package is installed and meets the minimum version requirement."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        else:
            version = "Unknown"
        
        if min_version and version != "Unknown":
            from packaging import version as packaging_version
            if packaging_version.parse(version) < packaging_version.parse(min_version):
                print(f"❌ {package_name} version {version} is installed, but version {min_version} or higher is required")
                return False
        
        print(f"✅ {package_name} (version {version}) is installed")
        return True
    
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_system_dependency(command, name=None):
    """Check if a system dependency is installed."""
    if name is None:
        name = command
    
    try:
        result = subprocess.run([command, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            print(f"✅ {name} is installed ({version})")
            return True
        else:
            print(f"❌ {name} is not installed or not working properly")
            return False
    except FileNotFoundError:
        print(f"❌ {name} is not installed or not in PATH")
        return False

def check_aws_credentials():
    """Check if AWS credentials are properly configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    
    if not aws_access_key or aws_access_key == "your_access_key":
        print("❌ AWS_ACCESS_KEY_ID is not set or is using the default placeholder value")
        return False
    
    if not aws_secret_key or aws_secret_key == "your_secret_key":
        print("❌ AWS_SECRET_ACCESS_KEY is not set or is using the default placeholder value")
        return False
    
    if not aws_region:
        print("❌ AWS_REGION is not set")
        return False
    
    print(f"✅ AWS credentials are configured (region: {aws_region})")
    return True

def check_docker():
    """Check if Docker is running and Qdrant container is available."""
    try:
        # Check if Docker is running
        result = subprocess.run(['docker', 'ps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("❌ Docker is not running or not installed")
            return False
        
        # Check if Qdrant container is running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=qdrant', '--format', '{{.Names}}'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if 'qdrant' not in result.stdout:
            print("❌ Qdrant container is not running")
            print("   Start it with: docker-compose up -d")
            return False
        
        print("✅ Docker is running and Qdrant container is available")
        return True
    
    except FileNotFoundError:
        print("❌ Docker is not installed or not in PATH")
        return False

def check_pdf_file():
    """Check if the sample PDF file exists."""
    pdf_path = Path("enhanced_financial_report.pdf")
    if pdf_path.exists():
        print(f"✅ Sample PDF file exists: {pdf_path}")
        return True
    else:
        print(f"❌ Sample PDF file not found: {pdf_path}")
        return False

def main():
    """Main function to check all dependencies."""
    print("Checking dependencies for Semantic Document Chunking and RAG application...\n")
    
    # Check Python packages
    packages = [
        ("langchain", "0.1.0"),
        ("langchain_community", "0.0.13"),
        ("langchain_core", "0.1.10"),
        ("langchain_aws", "0.1.1"),
        ("langchain_text_splitters", "0.0.1"),
        ("fastapi", "0.104.1"),
        ("uvicorn", "0.24.0"),
        ("python-dotenv", "1.0.0"),
        ("pydantic", "2.4.2"),
        ("pypdf", "3.17.1"),
        ("qdrant_client", "1.6.4"),
        ("boto3", "1.28.64"),
        ("unstructured", "0.10.30"),
        ("sentence_transformers", "2.2.2"),
        ("pi_heif", "0.13.0")
    ]
    
    print("Checking Python packages:")
    python_packages_ok = all(check_python_package(package, version) for package, version in packages)
    print()
    
    # Check system dependencies
    print("Checking system dependencies:")
    system_deps_ok = True
    if sys.platform == "darwin":  # macOS
        system_deps_ok = all([
            check_system_dependency("tesseract"),
            check_system_dependency("pdfinfo", "poppler"),
            # libmagic is usually available on macOS
        ])
    elif sys.platform.startswith("linux"):  # Linux
        system_deps_ok = all([
            check_system_dependency("tesseract"),
            check_system_dependency("pdfinfo", "poppler"),
            # Check if libmagic is installed
            check_python_package("magic")
        ])
    else:  # Windows or other
        print("⚠️ System dependency check not implemented for this platform")
        system_deps_ok = True
    print()
    
    # Check AWS credentials
    print("Checking AWS credentials:")
    aws_ok = check_aws_credentials()
    print()
    
    # Check Docker and Qdrant
    print("Checking Docker and Qdrant:")
    docker_ok = check_docker()
    print()
    
    # Check PDF file
    print("Checking sample PDF file:")
    pdf_ok = check_pdf_file()
    print()
    
    # Summary
    print("Summary:")
    if python_packages_ok:
        print("✅ All required Python packages are installed")
    else:
        print("❌ Some Python packages are missing or have incorrect versions")
        print("   Run: pip install -r requirements.txt")
    
    if system_deps_ok:
        print("✅ All required system dependencies are installed")
    else:
        print("❌ Some system dependencies are missing")
        if sys.platform == "darwin":
            print("   Run: brew install tesseract poppler libmagic")
        elif sys.platform.startswith("linux"):
            print("   Run: sudo apt-get install -y tesseract-ocr poppler-utils libmagic1")
    
    if aws_ok:
        print("✅ AWS credentials are properly configured")
    else:
        print("❌ AWS credentials are not properly configured")
        print("   Update your .env file with valid AWS credentials")
    
    if docker_ok:
        print("✅ Docker and Qdrant are running")
    else:
        print("❌ Docker or Qdrant is not running")
        print("   Start Qdrant with: docker-compose up -d")
    
    if pdf_ok:
        print("✅ Sample PDF file is available")
    else:
        print("❌ Sample PDF file is missing")
    
    # Final verdict
    if all([python_packages_ok, system_deps_ok, aws_ok, docker_ok, pdf_ok]):
        print("\n✅ All dependencies are properly installed and configured!")
        print("   You can now run the application with: python app.py")
        return 0
    else:
        print("\n❌ Some dependencies are missing or not properly configured.")
        print("   Please fix the issues above before running the application.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 