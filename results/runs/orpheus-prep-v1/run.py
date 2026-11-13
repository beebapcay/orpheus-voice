# Kernel Kaggle "orpheus-prep" (CPU, có internet, không tốn quota GPU).
# Tải âm thanh RADAR tiếng Việt, RADAR tiếng Anh (tập con), In-the-Wild (đủ bộ và tập con)
# và tạo CSV protocol. Kết quả ở /kaggle/working/data, được kernel orpheus-zeroshot
# gắn vào làm dữ liệu đầu vào (kernel_sources).
# Đẩy lên bằng: bash kaggle/push.sh push prep
import glob
import os
import shutil
import subprocess
import sys

SETS = "radar_vi,radar_en_sub,itw_sub,itw_full"


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        sys.exit(f"Không thấy {name} trong /kaggle/input. Dataset orpheus-src đã gắn chưa?")
    return os.path.dirname(hits[0])


src = find("prep_data.py")
print("Code:", src)
print(open(os.path.join(src, "VERSION")).read() if os.path.exists(os.path.join(src, "VERSION")) else "")
subprocess.run(["df", "-h", "/kaggle/working", "/tmp"])
tmp = "/tmp/orpheus_prep"
os.makedirs(tmp, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=False)
subprocess.run([sys.executable, os.path.join(src, "prep_data.py"), "fetch",
                "--ids-dir", src, "--out", "/kaggle/working/data", "--tmp", tmp,
                "--sets", SETS], check=True)
shutil.rmtree(tmp, ignore_errors=True)
subprocess.run(["du", "-sh", "/kaggle/working/data"])
