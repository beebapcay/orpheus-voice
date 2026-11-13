# Kernel Kaggle "orpheus-zeroshot" (GPU T4 x2, có internet).
# Dựng môi trường AntiDeepfake (setup_antideepfake.sh), rồi chấm các job trong JOBS.
# Kết quả ở /kaggle/working/results (giữ lại sau phiên, tải về bằng push.sh output).
# Đẩy lên bằng: bash kaggle/push.sh push zeroshot
#
# Mỗi lần chạy chỉ đặt số job vừa một phiên (giới hạn 12 giờ). Job đã có kết quả
# trong RESUME (kết quả lần trước, nếu gắn vào) thì bỏ qua.
import glob
import os
import shutil
import subprocess
import sys
import time

# ---- Sửa phần này cho mỗi lần chạy -------------------------------------------
# <tập>:<mô hình>,<mô hình>. Thứ tự = thứ tự ưu tiên; hai GPU lấy job lần lượt.
# Lần 1 (13/11, xong 2,42 giờ): toy:mms_300m; radar_vi và ood_forget cho
#   mms_300m, w2v_small, w2v_large, w2v2_aasist; itw_full:mms_300m,w2v2_aasist.
# Lần 2: bản NDA của 3 backbone nhỏ + MMS-1B, XLS-R-1B (cả hai bản).
#   mms_300m_nda đứng đầu: cần cho phát hiện 1 (RawBoost có lợi trên tiếng Việt không).
# Lần 3: hubert_xl, hubert_xl_nda, xlsr_2b, xlsr_2b_nda.
JOBS = [
    "radar_vi:mms_300m_nda,mms_1b,xlsr_1b,mms_1b_nda,xlsr_1b_nda,w2v_large_nda,w2v_small_nda",
]
SHARD_SIZE = 2000
# -------------------------------------------------------------------------------

T0 = time.time()


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        sys.exit(f"Không thấy {name} trong /kaggle/input")
    return os.path.dirname(hits[0])


src = find("score_zeroshot.py")
data = find("orpheus_data_manifest.json")
print("Code:", src, "| Dữ liệu:", data, flush=True)
for f in ("VERSION", ):
    p = os.path.join(src, f)
    if os.path.exists(p):
        print(open(p).read(), flush=True)
print(open(os.path.join(data, "orpheus_data_manifest.json")).read(), flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; df -h /kaggle/working /tmp")

work = "/tmp/orpheus"
base, cache, out = f"{work}/work", f"{work}/cache", "/kaggle/working/results"
os.makedirs(cache, exist_ok=True)
os.makedirs(out, exist_ok=True)

# Kết quả của lần chạy trước (nếu đã gắn output cũ làm input) -> chép vào out để bỏ qua
for prev in glob.glob("/kaggle/input/**/results/summary.csv", recursive=True):
    prev_dir = os.path.dirname(prev)
    if os.path.realpath(prev_dir) != os.path.realpath(out):
        print("Dùng lại kết quả cũ từ", prev_dir, flush=True)
        shutil.copytree(prev_dir, out, dirs_exist_ok=True)

# Script cần chạy từ thư mục có quyền ghi (dataset là chỉ đọc)
code = f"{work}/src"
shutil.copytree(src, code, dirs_exist_ok=True)

try:
    import safetensors  # noqa: F401
except ImportError:
    sh([sys.executable, "-m", "pip", "install", "-q", "safetensors"], check=True)

env = dict(os.environ, CACHE_DIR=cache)
r = sh(["bash", f"{code}/setup_antideepfake.sh", base, "env"], env=env)
for f in ("setup.log", "versions.txt", "pip-freeze.txt"):
    if os.path.exists(f"{base}/{f}"):
        shutil.copy(f"{base}/{f}", out)
if r.returncode != 0:
    sys.exit("Dựng môi trường thất bại, xem results/setup.log")
print(f"Dựng môi trường xong sau {(time.time() - T0) / 60:.1f} phút", flush=True)

import torch  # noqa: E402
gpus = ",".join(str(i) for i in range(max(1, torch.cuda.device_count())))
r = sh([sys.executable, f"{code}/score_zeroshot.py", "--base", base, "--data-root", data,
        "--out", out, "--cache", cache, "--gpus", gpus, "--shard-size", str(SHARD_SIZE),
        "--jobs", *JOBS])
sh(f"cat {out}/summary.csv")
print(f"Tổng thời gian {(time.time() - T0) / 3600:.2f} giờ", flush=True)
sys.exit(r.returncode)
