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


def _batch_hash(n):
    """Tuỳ chọn (BATCH_HASH_N=n): in dấu vân tay của n batch đầu mà mỗi tiến trình đọc dữ liệu tạo ra:
      BATCH_HASH pid=... ids=<md5 danh sách ID> n=<số câu> wav=<md5 sóng âm đã qua RawBoost, chuẩn hoá>
    Dùng ở lần thử đo để kiểm: cùng seed và cùng số worker thì SFT và GRPO có nhận đúng cùng các batch, cùng
    phép tăng cường không (seed được đặt sau khi dựng mô hình; sampler dùng seed cố định). Không đổi phép
    tính: chỉ đọc batch sau khi tạo. Vá phương thức của chính lớp PaddedBatch (không tạo lớp con) để batch
    gửi từ worker về tiến trình chính vẫn pickle được."""
    import hashlib
    from speechbrain.dataio import batch as sbb
    orig, left = sbb.PaddedBatch.__init__, {"n": n}

    def init(self, examples, *a, **k):
        orig(self, examples, *a, **k)
        if left["n"] > 0 and hasattr(self, "wav") and hasattr(self, "id"):
            left["n"] -= 1
            ids = hashlib.md5("|".join(map(str, self.id)).encode()).hexdigest()[:16]
            wav = hashlib.md5(self.wav.data.detach().cpu().numpy().tobytes()).hexdigest()[:16]
            print(f"BATCH_HASH pid={os.getpid()} ids={ids} n={len(self.id)} wav={wav}", flush=True)

    sbb.PaddedBatch.__init__ = init


T0 = time.time()
atexit.register(_report)
if int(os.environ.get("BATCH_HASH_N", "0") or 0) > 0:
    _batch_hash(int(os.environ["BATCH_HASH_N"]))
threading.Thread(target=_periodic, args=(float(os.environ.get("MEM_PROBE_EVERY", "60")),),
                 daemon=True).start()

script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
# Như `python main.py`: sys.path[0] là thư mục của main.py (repo), không phải thư mục của
# script này (src/ của đồ án), để `import utils`, `models`, `dataio`... lấy đúng bản của repo.
sys.path[0] = os.path.dirname(script)
runpy.run_path(script, run_name="__main__")
