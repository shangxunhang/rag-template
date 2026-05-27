import os

# 临时绕过 OpenMP 重复加载问题，只用于环境验证
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

print("=" * 80)
print("[Python]")
print("python executable:", sys.executable)
print("python version:", sys.version)

print("=" * 80)
print("[PyTorch]")
import torch

print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
    print("gpu count:", torch.cuda.device_count())

print("=" * 80)
print("[Sentence Transformers]")
import sentence_transformers
print("sentence-transformers version:", sentence_transformers.__version__)

print("=" * 80)
print("[FAISS]")
import faiss
print("faiss imported successfully")
print("faiss version:", getattr(faiss, "__version__", "unknown"))

print("=" * 80)
print("[Done]")