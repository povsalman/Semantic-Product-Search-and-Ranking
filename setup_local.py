"""
Local Setup Script for Task 2: Semantic Product Search
=======================================================

This script automates the setup process for running the project locally.

Run this after:
1. Installing requirements.txt
2. Downloading model files from Kaggle

Author: [Your Name]
Course: Generative AI - Fall 2025
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "src",
        "kaggle_training"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}/")
    
    print("✅ All directories created!")

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'sentence-transformers',
        'streamlit',
        'nltk',
        'sklearn'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("   Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    import nltk
    
    datasets = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"   ✓ {dataset}")
        except Exception as e:
            print(f"   ❌ {dataset}: {str(e)}")
    
    print("✅ NLTK data downloaded!")

def check_model_files():
    """Check if model files are present"""
    print("\n🤖 Checking model files...")
    
    required_files = {
        "models/best_model.pth": "Trained model weights",
        "models/product_embeddings.npy": "Product embeddings",
        "models/product_metadata.parquet": "Product metadata"
    }
    
    missing = []
    
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✓ {filepath} ({size:.2f} MB) - {description}")
        else:
            print(f"   ❌ {filepath} - {description}")
            missing.append(filepath)
    
    if missing:
        print("\n⚠️ Missing model files!")
        print("\n📥 Please download these files from Kaggle:")
        for filepath in missing:
            print(f"   - {filepath}")
        print("\nSee README.md for instructions.")
        return False
    
    print("✅ All model files present!")
    return True

def check_data_files():
    """Check if dataset files are present"""
    print("\n📊 Checking dataset files...")
    
    data_files = [
        "data/raw/shopping_queries_dataset_examples.parquet",
        "data/raw/shopping_queries_dataset_products.parquet"
    ]
    
    missing = []
    
    for filepath in data_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✓ {filepath} ({size:.2f} MB)")
        else:
            print(f"   ❌ {filepath}")
            missing.append(filepath)
    
    if missing:
        print("\n⚠️ Missing dataset files!")
        print("\nℹ️ Dataset files are optional for running the web app.")
        print("   They are only needed if you want to retrain the model.")
        print("\n📥 Download from:")
        print("   https://github.com/amazon-science/esci-data")
        return False
    
    print("✅ All dataset files present!")
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    modules = [
        ('src.data_loader', 'DataLoader'),
        ('src.preprocessor', 'TextPreprocessor'),
        ('src.model', 'SemanticRankingModel'),
        ('src.evaluate', 'RankingMetrics'),
        ('src.utils', 'get_device')
    ]
    
    all_ok = True
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {str(e)}")
            all_ok = False
    
    if all_ok:
        print("✅ All modules can be imported!")
    else:
        print("⚠️ Some modules failed to import. Check error messages above.")
    
    return all_ok

def show_next_steps():
    """Show next steps to user"""
    print_header("🎉 SETUP COMPLETE!")
    
    print("📋 Next Steps:")
    print("\n1. 📥 Download Model Files (if not already done):")
    print("   - Train model on Kaggle using task2_kaggle_training.ipynb")
    print("   - Download best_model.pth, product_embeddings.npy, product_metadata.parquet")
    print("   - Place files in models/ directory")
    
    print("\n2. 🌐 Launch Web Application:")
    print("   streamlit run app.py")
    
    print("\n3. 🧪 Test the Application:")
    print("   - Open browser at http://localhost:8501")
    print("   - Enter search queries")
    print("   - View ranked results")
    
    print("\n📚 Additional Resources:")
    print("   - README.md: Complete documentation")
    print("   - prompts.txt: All prompts used")
    
    print("\n" + "=" * 70)

def main():
    """Main setup function"""
    print_header("🚀 Task 2: Semantic Product Search - Local Setup")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️ Please install missing dependencies first")
        print("   Run: pip install -r requirements.txt")
        return
    
    # Download NLTK data
    try:
        download_nltk_data()
    except Exception as e:
        print(f"⚠️ Failed to download NLTK data: {str(e)}")
        print("   You can download manually in Python:")
        print("   >>> import nltk")
        print("   >>> nltk.download('all')")
    
    # Check model files
    models_ok = check_model_files()
    
    # Check data files (optional)
    data_ok = check_data_files()
    
    # Test imports
    imports_ok = test_imports()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SETUP SUMMARY")
    print("=" * 70)
    print(f"   Python Version: {'✅' if True else '❌'}")
    print(f"   Dependencies: {'✅' if True else '❌'}")
    print(f"   Model Files: {'✅' if models_ok else '❌'}")
    print(f"   Dataset Files: {'✅' if data_ok else '⚠️ (optional)'}")
    print(f"   Module Imports: {'✅' if imports_ok else '❌'}")
    
    if models_ok and imports_ok:
        show_next_steps()
    else:
        print("\n⚠️ Setup incomplete. Please address issues above.")
        print("   See README.md for detailed instructions.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()