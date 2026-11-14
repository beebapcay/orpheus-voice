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
# GĐ5, lần 1 (chore/GD5-Ablation.md mục 2): R seed 1 và A0 seed 1, hai GPU song song.
#   Cấu hình SFT đã chốt ở GĐ4 (chore/GD4-Du-Lieu-Va-Pipeline.md mục 4.2, E1–E3): 3 epoch trên
#   toàn bộ train (17,88 giờ), batch 50 giây, lr đỉnh 1e-6, warmup 10% (460 / 4.600 bước), chỉ
#   kiểm tra dev một lần sau epoch cuối nên checkpoint lưu là mô hình cuối.
#   Sau khi cả hai GPU xong: chấm S0 (dùng thẳng, không fine-tune) trên SEA-Spoof dev/test cho
#   3 backbone sẽ fine-tune ở GĐ5 và MMS-300M bản NDA (mốc của N0).
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget"]
RUNS = {
    0: [
        {"run": "R_s1", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1"], "score": SCORE},
    ],
    1: [
        {"run": "A0_s1", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1", "use_da=False"], "score": SCORE},
    ],
}
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = ["seaspoof_vi_dev:mms_300m,w2v_large,w2v_small,mms_300m_nda",
            "seaspoof_vi_test:mms_300m,w2v_large,w2v_small,mms_300m_nda"]
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
print(f"Huấn luyện và chấm sau fine-tune xong sau {(time.time() - T0) / 3600:.2f} giờ", flush=True)

if ZEROSHOT:
    cmd = [sys.executable, f"{code}/score_zeroshot.py", "--base", base, "--data-root", sea, prep,
           "--out", f"{out}/zeroshot", "--cache", cache, "--gpus", "0,1", "--jobs", *ZEROSHOT]
    logf = f"{out}/zeroshot.driver.log"
    t0 = time.time()
    print(f"Chấm zero-shot: {' '.join(ZEROSHOT)}", flush=True)
    with open(logf, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"Chấm zero-shot xong (mã {rc}) sau {(time.time() - t0) / 60:.1f} phút", flush=True)
    sh(f"grep -E 'KẾT QUẢ|THẤT BẠI|lỗi|Lỗi' {logf} | tail -12")
    if rc != 0:
        failures.append("zeroshot")

for f in ("train_summary.csv", "scores/summary.csv", "zeroshot/summary.csv"):
    sh(f"cat {out}/{f}")
sh(f"du -sh {out}/*")
print(f"Thất bại: {failures}" if failures else "Mọi lần chạy xong", flush=True)
print(f"Tổng thời gian {(time.time() - T0) / 3600:.2f} giờ", flush=True)
sys.exit(1 if failures else 0)
