"""
Installation Test Suite
Verifies that all components are correctly installed and configured
"""

import sys
import os
from pathlib import Path


class InstallationTester:
    """Test suite for installation verification"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def test(self, name, func):
        """Run a single test"""
        try:
            result = func()
            if result is True:
                print(f"✓ {name}")
                self.passed += 1
            elif result is None:
                print(f"⚠ {name}")
                self.warnings += 1
            else:
                print(f"✗ {name}: {result}")
                self.failed += 1
        except Exception as e:
            print(f"✗ {name}: {str(e)}")
            self.failed += 1
    
    def test_python_version(self):
        """Check Python version"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            return True
        return f"Python 3.8+ required, found {version.major}.{version.minor}"
    
    def test_pytorch(self):
        """Check PyTorch installation"""
        try:
            import torch
            version = torch.__version__
            return True
        except ImportError:
            return "PyTorch not installed"
    
    def test_cuda(self):
        """Check CUDA availability"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                return True
            return "CUDA not available"
        except:
            return "Cannot check CUDA"
    
    def test_timm(self):
        """Check timm library"""
        try:
            import timm
            return True
        except ImportError:
            return "timm not installed"
    
    def test_sklearn(self):
        """Check scikit-learn"""
        try:
            import sklearn
            return True
        except ImportError:
            return "scikit-learn not installed"
    
    def test_pandas(self):
        """Check pandas"""
        try:
            import pandas
            return True
        except ImportError:
            return "pandas not installed"
    
    def test_matplotlib(self):
        """Check matplotlib"""
        try:
            import matplotlib
            return True
        except ImportError:
            return "matplotlib not installed"
    
    def test_seaborn(self):
        """Check seaborn"""
        try:
            import seaborn
            return True
        except ImportError:
            return "seaborn not installed"
    
    def test_pil(self):
        """Check PIL/Pillow"""
        try:
            from PIL import Image
            return True
        except ImportError:
            return "Pillow not installed"
    
    def test_cv2(self):
        """Check OpenCV"""
        try:
            import cv2
            return True
        except ImportError:
            return "opencv-python not installed"
    
    def test_tqdm(self):
        """Check tqdm"""
        try:
            import tqdm
            return True
        except ImportError:
            return "tqdm not installed"
    
    def test_main_script(self):
        """Check main.py exists and imports"""
        if not os.path.exists('main.py'):
            return "main.py not found"
        
        try:
            from main import Config, RNAOnlyClassifier
            return True
        except Exception as e:
            return f"Cannot import from main.py: {str(e)}"
    
    def test_inference_script(self):
        """Check inference.py exists"""
        if not os.path.exists('inference.py'):
            return None  # Warning, not required
        return True
    
    def test_attention_rollout(self):
        """Check attention_rollout.py exists"""
        if not os.path.exists('attention_rollout.py'):
            return None  # Warning, not required
        return True
    
    def test_data_dir_config(self):
        """Check if data directories are configured"""
        try:
            from main import Config
            c = Config()
            
            if c.BRCA_WSI_DIR == "/path/to/brca/folder":
                return "BRCA_WSI_DIR not configured in main.py"
            
            if c.RNA_CSV_PATH == "/path/to/rna_seq.csv":
                return "RNA_CSV_PATH not configured in main.py"
            
            return True
        except:
            return "Cannot check config"
    
    def test_data_dir_exists(self):
        """Check if data directories exist"""
        try:
            from main import Config
            c = Config()
            
            if not os.path.exists(c.BRCA_WSI_DIR):
                return f"BRCA directory not found: {c.BRCA_WSI_DIR}"
            
            if not os.path.exists(c.RNA_CSV_PATH):
                return f"RNA CSV not found: {c.RNA_CSV_PATH}"
            
            return True
        except:
            return None  # Warning
    
    def test_output_dir(self):
        """Check output directory"""
        try:
            from main import Config
            c = Config()
            os.makedirs(c.OUTPUT_DIR, exist_ok=True)
            return True
        except:
            return "Cannot create output directory"
    
    def test_gpu_memory(self):
        """Check GPU memory"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            
            # Get free memory
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            free_mem = int(result.stdout.strip())
            
            if free_mem < 8000:
                return f"Low GPU memory: {free_mem}MB (recommend 8GB+)"
            
            return True
        except:
            return None
    
    def test_model_creation(self):
        """Test model instantiation"""
        try:
            from main import Config, RNAOnlyClassifier, WSIOnlyClassifier
            import torch
            
            c = Config()
            
            # Test RNA model
            rna_model = RNAOnlyClassifier(
                input_dim=c.NUM_TOP_GENES,
                hidden_dim=c.RNA_HIDDEN_DIM,
                num_classes=c.NUM_CLASSES
            )
            
            # Test WSI model
            wsi_model = WSIOnlyClassifier(
                feature_dim=c.WSI_FEATURE_DIM,
                num_classes=c.NUM_CLASSES
            )
            
            return True
        except Exception as e:
            return f"Model creation failed: {str(e)}"
    
    def test_forward_pass(self):
        """Test forward pass"""
        try:
            from main import Config, RNAOnlyClassifier
            import torch
            
            c = Config()
            
            model = RNAOnlyClassifier(
                input_dim=c.NUM_TOP_GENES,
                hidden_dim=c.RNA_HIDDEN_DIM,
                num_classes=c.NUM_CLASSES
            )
            
            # Test forward pass
            dummy_input = torch.randn(2, c.NUM_TOP_GENES)
            logits, features = model(dummy_input)
            
            if logits.shape[1] != c.NUM_CLASSES:
                return f"Wrong output shape: {logits.shape}"
            
            return True
        except Exception as e:
            return f"Forward pass failed: {str(e)}"
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 70)
        print(" " * 20 + "INSTALLATION TEST SUITE")
        print("=" * 70)
        print()
        
        print("System Requirements:")
        print("-" * 70)
        self.test("Python 3.8+", self.test_python_version)
        print()
        
        print("Core Dependencies:")
        print("-" * 70)
        self.test("PyTorch", self.test_pytorch)
        self.test("CUDA", self.test_cuda)
        self.test("timm", self.test_timm)
        self.test("scikit-learn", self.test_sklearn)
        self.test("pandas", self.test_pandas)
        self.test("matplotlib", self.test_matplotlib)
        self.test("seaborn", self.test_seaborn)
        self.test("Pillow", self.test_pil)
        self.test("OpenCV", self.test_cv2)
        self.test("tqdm", self.test_tqdm)
        print()
        
        print("Project Files:")
        print("-" * 70)
        self.test("main.py", self.test_main_script)
        self.test("inference.py", self.test_inference_script)
        self.test("attention_rollout.py", self.test_attention_rollout)
        print()
        
        print("Configuration:")
        print("-" * 70)
        self.test("Data paths configured", self.test_data_dir_config)
        self.test("Data directories exist", self.test_data_dir_exists)
        self.test("Output directory", self.test_output_dir)
        print()
        
        print("Hardware:")
        print("-" * 70)
        self.test("GPU memory", self.test_gpu_memory)
        print()
        
        print("Model Tests:")
        print("-" * 70)
        self.test("Model instantiation", self.test_model_creation)
        self.test("Forward pass", self.test_forward_pass)
        print()
        
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"✓ Passed:   {self.passed}")
        print(f"✗ Failed:   {self.failed}")
        print(f"⚠ Warnings: {self.warnings}")
        print("=" * 70)
        print()
        
        if self.failed == 0:
            print("✓ All critical tests passed! Ready to train.")
            print()
            print("Next steps:")
            print("  1. Verify data paths in main.py")
            print("  2. Run: python data_preparation.py --validate_only")
            print("  3. Start training: ./run_training.sh")
            return 0
        else:
            print("✗ Some tests failed. Please fix the issues above.")
            print()
            print("Common fixes:")
            print("  - Install missing packages: pip install -r requirements.txt")
            print("  - Configure data paths in main.py (Config class)")
            print("  - Check CUDA installation: nvidia-smi")
            return 1


def print_system_info():
    """Print detailed system information"""
    print("\nDETAILED SYSTEM INFO")
    print("=" * 70)
    
    # Python
    print(f"Python: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            
            # Memory
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            mem_info = result.stdout.strip().split(',')
            print(f"GPU Memory: {mem_info[0]}MB total, {mem_info[2]}MB free")
    except:
        print("PyTorch: Not installed")
    
    # Disk space
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"Disk Space: {free // (2**30)}GB free of {total // (2**30)}GB")
    
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test installation')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed system information'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print_system_info()
        print()
    
    # Run tests
    tester = InstallationTester()
    exit_code = tester.run_all_tests()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()