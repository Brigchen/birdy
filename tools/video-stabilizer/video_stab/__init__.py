# -*- coding: utf-8 -*-
"""Birdy 衍生：观鸟录像稳定工具。"""

import os

# Anaconda 的 MKL 与 PyTorch 各带一份 libiomp5md.dll。
# 跟踪（ByteTrack）一加载第二份，Windows 会直接 Abort，界面表现为闪退。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "0.1.0"
