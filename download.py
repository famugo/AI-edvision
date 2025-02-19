

import sys
import subprocess

with open('requirements.txt') as f:
    packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])