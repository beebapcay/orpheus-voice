#!/usr/bin/env python3
"""Kiểm TF32 có thật sự tắt không, bằng số (chạy trong môi trường antideepfake, trên GPU).

SpeechBrain 1.0.2 tự bật quirk `allow_tf32` khi import (torch.backends.cuda.matmul.allow_tf32 và
torch.backends.cudnn.allow_tf32 = True), nên trên GPU Ampere/Ada, matmul và convolution fp32 chạy bằng TF32
(10 bit phần định trị) nếu không ghi đè. Bộ server đặt NVIDIA_TF32_OVERRIDE=0 để cuBLAS/cuDNN bỏ qua TF32 (tính fp32
như T4). Script so kết quả fp32 trên GPU với fp64: sai số tương đối ~1e-6 hoặc nhỏ hơn là fp32, ~1e-3 là TF32.

  NVIDIA_TF32_OVERRIDE=0 python tf32_check.py      (bootstrap.sh chạy cả hai trường hợp: có và không đặt biến)
"""
import os

import torch
import speechbrain  # noqa: F401  (áp quirk allow_tf32 như khi main.py import)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def main():
    torch.manual_seed(0)
    dev = "cuda"
    a, b = torch.randn(2048, 2048, device=dev), torch.randn(2048, 2048, device=dev)
    m = rel((a @ b).double(), a.double() @ b.double())
    x, w = torch.randn(4, 1, 32000, device=dev), torch.randn(512, 1, 10, device=dev)
    F = torch.nn.functional
    c = rel(F.conv1d(x, w, stride=5).double(), F.conv1d(x.double(), w.double(), stride=5))
    verdict = "TF32" if max(m, c) > 1e-4 else "fp32"
    print(f"TF32_CHECK gpu={torch.cuda.get_device_name(0)} NVIDIA_TF32_OVERRIDE={os.environ.get('NVIDIA_TF32_OVERRIDE')} "
          f"allow_tf32 matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32} "
          f"matmul_rel_err={m:.1e} conv1d_rel_err={c:.1e} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
