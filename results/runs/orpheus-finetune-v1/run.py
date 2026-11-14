# Kernel Kaggle "orpheus-finetune" (GPU T4 x2, có internet).
# Dựng môi trường AntiDeepfake, rồi mỗi GPU chạy lần lượt các lần huấn luyện trong RUNS
# (src/finetune.py: huấn luyện bằng main.py gốc, lấy checkpoint tốt nhất, chấm bằng score_zeroshot.py).
# Kết quả ở /kaggle/working/results: train_summary.csv, train/<run>/ (log, monitor.csv,
# exp_log.txt), scores/ (điểm, độ đo), ckpt/<run>.ckpt (không tải về máy theo mặc định).
# Đẩy lên bằng: bash kaggle/push.sh push finetune
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

# ---- Sửa phần này cho mỗi lần chạy -------------------------------------------
# GĐ4 ngày 2, lần 1: pilot đo đạc (không phải thí nghiệm GĐ5).
#   GPU 0: R giới hạn 300 bước, có RawBoost (cấu hình gốc) -> tốc độ, bộ nhớ, CPU.
#   GPU 1: overfit 100 câu (n50, không RawBoost, lr 1e-5) -> pipeline học được không;
#          rồi A0 giới hạn 300 bước (không RawBoost) -> chi phí RawBoost; chấm lại dev
#          qua score_zeroshot để đối chiếu EER dev lúc huấn luyện (cùng trọng số, dev sạch).
#   Hai GPU chạy song song: đo luôn 4 nhân CPU có đủ cho hai lần huấn luyện không.
PILOT = ["number_of_epochs=1", "optimizer_step_limit=300", "valid_step=100000", "num_workers=2"]
RUNS = {
    0: [
        {"run": "pilot_R", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": PILOT, "score": ["seaspoof_vi_dev"]},
    ],
    1: [
        {"run": "overfit_n50", "model": "mms_300m", "train": "seaspoof_vi_train_n50",
         "valid": "seaspoof_vi_train_n50",
         "set": ["number_of_epochs=30", "valid_step=25", "use_da=False", "lr=1e-5", "warmup=1",
                 "num_workers=2"], "score": []},
        {"run": "pilot_A0", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": PILOT + ["use_da=False"],
         "score": ["seaspoof_vi_dev", "ood_forget"]},
    ],
}
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


src = find("finetune.py")
sea = find("seaspoof_vi_train.csv")
prep = find("orpheus_data_manifest.json")
print("Code:", src, "| SEA-Spoof:", sea, "| RADAR/ItW:", prep, flush=True)
p = os.path.join(src, "VERSION")
if os.path.exists(p):
    print(open(p).read(), flush=True)
sh(f"ls {sea} | head; ls {sea}/seaspoof_vi/flac | wc -l")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; df -h /kaggle/working /tmp; "
   "echo nproc=$(nproc); lscpu | grep -E 'Model name|^CPU\\(s\\)'")

work = "/tmp/orpheus"
base, cache, out = f"{work}/work", f"{work}/cache", "/kaggle/working/results"
os.makedirs(cache, exist_ok=True)
os.makedirs(out, exist_ok=True)
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

common = ["--base", base, "--data-root", sea, prep, "--out", out, "--cache", cache]
# Tải trước checkpoint khởi tạo (một lần cho mỗi mô hình), tránh hai GPU cùng tải một file
for m in sorted({r["model"] for runs in RUNS.values() for r in runs}):
    sh([sys.executable, f"{code}/finetune.py", *common, "--run", f"prefetch_{m}", "--model", m,
        "--prefetch"], check=True)

failures = []


def worker(gpu, runs):
    for r in runs:
        cmd = [sys.executable, f"{code}/finetune.py", *common, "--gpu", str(gpu),
               "--run", r["run"], "--model", r["model"], "--method", r.get("method", "sft"),
               "--train", r["train"], "--valid", r["valid"], "--set", *r["set"]]
        if r["score"]:
            cmd += ["--score", *r["score"]]
        logf = f"{out}/train/{r['run']}.driver.log"
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        t0 = time.time()
        print(f"[GPU {gpu}] bắt đầu {r['run']}", flush=True)
        with open(logf, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"[GPU {gpu}] xong {r['run']} (mã {rc}) sau {(time.time() - t0) / 60:.1f} phút", flush=True)
        sh(f"grep -E 'KẾT QUẢ|THẤT BẠI|lỗi|Lỗi' {logf} | tail -5")
        if rc != 0:
            failures.append(r["run"])


threads = [threading.Thread(target=worker, args=(g, runs)) for g, runs in RUNS.items()]
for t in threads:
    t.start()
    time.sleep(20)
for t in threads:
    t.join()
for f in ("train_summary.csv", "scores/summary.csv"):
    sh(f"cat {out}/{f}")
sh(f"du -sh {out}/*")
print(f"Thất bại: {failures}" if failures else "Mọi lần chạy xong", flush=True)
print(f"Tổng thời gian {(time.time() - T0) / 3600:.2f} giờ", flush=True)
sys.exit(1 if failures else 0)
