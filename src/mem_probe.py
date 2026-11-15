#!/usr/bin/env python3
"""Chạy main.py gốc của AntiDeepfake nguyên vẹn (runpy) và đo bộ nhớ GPU thật của PyTorch.

nvidia-smi chỉ thấy bộ nhớ PyTorch đã giữ (caching allocator), nên gần như luôn báo gần đầy
(GĐ4 pilot: 14,6 GB cả khi overfit 100 câu). Script này ghi thêm, khi tiến trình thoát (kể cả
khi lỗi hết bộ nhớ):
  MEM_PROBE max_allocated_mib=... max_reserved_mib=... total_mib=...
và mỗi `MEM_PROBE_EVERY` giây (mặc định 60) một dòng MEM_PROBE_T với đỉnh tính đến lúc đó.
Không đổi phép tính: chỉ đọc bộ đếm của torch.cuda.

Cách dùng (cwd = thư mục repo AntiDeepfake, như `python main.py ...`):
  python mem_probe.py main.py <yaml> --base_path ... [tham số khác của main.py]
"""
import atexit
import os
import runpy
import sys
import threading
import time

import torch


def _line(tag):
    if not torch.cuda.is_available():
        return f"{tag} không có GPU"
    mib = 2 ** 20
    total = torch.cuda.get_device_properties(0).total_memory / mib
    return (f"{tag} max_allocated_mib={torch.cuda.max_memory_allocated() / mib:.0f} "
            f"max_reserved_mib={torch.cuda.max_memory_reserved() / mib:.0f} total_mib={total:.0f} "
            f"t={time.time() - T0:.0f}")


def _report():
    print(_line("MEM_PROBE"), flush=True)


def _periodic(every):
    while True:
        time.sleep(every)
        try:
            print(_line("MEM_PROBE_T"), flush=True)
        except Exception:  # noqa: BLE001
            return


T0 = time.time()
atexit.register(_report)
threading.Thread(target=_periodic, args=(float(os.environ.get("MEM_PROBE_EVERY", "60")),),
                 daemon=True).start()

script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
# Như `python main.py`: sys.path[0] là thư mục của main.py (repo), không phải thư mục của
# script này (src/ của đồ án), để `import utils`, `models`, `dataio`... lấy đúng bản của repo.
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name="__main__")
