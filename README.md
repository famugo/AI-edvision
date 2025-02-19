# AI-EdVision - 智能教育视频分析与个性化辅导系统

## 1. 项目介绍
AI-EdVision 是一个基于人工智能的教育视频分析与个性化辅导系统。

## 2. 环境要求
- Python = 3.10（推荐使用 Python 3.10）
- CUDA = 11.8（如果使用 GPU）
- cuDNN 对应版本
- 显存建议 >= 8GB（使用 GPU 时）
- 内存建议 >= 16GB

## 3. 环境配置

### 3.1 基础环境要求
- Python 3.10
- CUDA 11.8 及以上（如果使用 GPU）
- cuDNN（与 CUDA 版本匹配）

### 3.2 创建、激活虚拟环境
```bash
# 创建名为 ai_vision 的虚拟环境
conda create -n ai_vision python=3.10
# 激活环境
conda activate ai_vision
```

### 3.3 安装 PyTorch 和 FAISS
```bash
# 1. 安装 PyTorch（使用 CUDA 11.8）
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 FAISS-GPU
conda install -c conda-forge faiss-gpu=1.8.0 -y
```

### 3.4 运行'..\AI-EdVision\download.py'安装其他依赖
```bash
# 切换到download.py所在目录
# 运行下载脚本
python download.py

conda install spacy-pkuseg -y
```

# 7. 下载 spacy 中文语言模型
```bash
python -m spacy download zh_core_web_sm 
```

### 3.5 验证安装(在虚拟环境中运行)
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# 验证 transformers 安装
from transformers import AutoTokenizer
print("Transformers 测试成功")

# 验证 spacy 安装
import spacy
nlp = spacy.load("zh_core_web_sm")
print("Spacy 测试成功")
```

## 4. 运行程序
```bash
# 切换到 IVS 目录
cd IVS

# 启动应用
streamlit run main.py
```

## 5. 常见问题

### 5.1 依赖安装失败
如果在安装依赖时遇到问题，可以尝试以下方法：
1. 确保已经正确安装 CUDA 和 cuDNN
2. 使用国内镜像源：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
3. 单独安装失败的包，查看具体错误信息

### 5.2 运行时 GPU 内存不足
1. 可以尝试减小批处理大小
2. 关闭其他占用 GPU 的程序
3. 如果仍然不足，考虑使用更大显存的 GPU

### 5.3 CUDA 相关错误
确保 CUDA 版本与 PyTorch 版本匹配，当前配置使用的是：
- CUDA 11.8
- PyTorch 2.0.0
- cuDNN 对应版本

## 6. 注意事项
1. 所有依赖版本都是经过测试的，请不要随意更改版本号
2. 建议使用 GPU 运行，CPU 运行可能会很慢
3. 首次运行时会下载一些模型文件，请确保网络通畅
4. 如遇到问题，请查看错误信息并对照常见问题解决
