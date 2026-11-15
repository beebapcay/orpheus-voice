# Kernel Kaggle "orpheus-finetune" (GPU T4 x2, có internet).
# Dựng môi trường AntiDeepfake, rồi mỗi GPU chạy lần lượt các lần huấn luyện trong RUNS
# (src/finetune.py: huấn luyện bằng main.py gốc, lấy checkpoint tốt nhất, chấm bằng score_zeroshot.py).
# Kết quả ở /kaggle/working/results: train_summary.csv, train/<run>/ (log, monitor.csv,
# exp_log.txt), scores/ (điểm, độ đo), ckpt/<run>.ckpt (không tải về máy theo mặc định).
# Đẩy lên bằng: bash kaggle/push.sh push finetune
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time

# ---- Sửa phần này cho mỗi lần chạy -------------------------------------------
# GĐ6 (chore/GD6-GRPO.md mục 6). Hai kernel dùng chung file này: đặt JOB trước mỗi lần đẩy;
#   bản đã đẩy của từng kernel lưu ở results/runs/<kernel>-v<N>/run.py.
#   "grpo1" -> orpheus-finetune v6: G1 seed 1 (GPU 0), G2 seed 1 (GPU 1).
#   "r45"   -> orpheus-finetune-b v2: R seed 4, 5 (P1: thêm seed cho thước đo độ lệch của R).
#   Mọi lần dùng NGUYÊN bộ ghi đè SFT của GĐ5 (lịch learning rate E1', GĐ5 mục 4.1): GRPO chỉ đổi
#   file YAML (hàm loss). GRPO thêm (đã chốt 15/11 sau lần thử đo v5, mục 3):
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (batch 50 hết bộ nhớ do phân mảnh nếu không
#   bật; không đổi phép tính), --oom-retry 0 (không tự giảm batch), --probe-mem (ghi đỉnh bộ nhớ
#   thật). Lần R giữ đúng như R seed 1–3 (không thêm gì).
JOB = "r45"
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test"]
GARGS = ["--oom-retry", "0", "--probe-mem", "--env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
_r = {"model": "mms_300m", "train": "seaspoof_vi_train", "valid": "seaspoof_vi_dev", "score": SCORE}
JOBS = {
    "grpo1": {
        0: [dict(_r, run="G1_s1", method="grpo_v1", set=SFT + ["seed=1"], args=GARGS)],
        1: [dict(_r, run="G2_s1", method="grpo_v2", set=SFT + ["seed=1"], args=GARGS)],
    },
    "r45": {
        0: [dict(_r, run="R_s4", set=SFT + ["seed=4"])],
        1: [dict(_r, run="R_s5", set=SFT + ["seed=5"])],
    },
    # Tuần quota từ 21/11 (mục 6, lần 3–5). G3 = G1 với beta = 0 (ghi đè lồng rl_config, đã kiểm
    # HyperPyYAML gộp đúng: mục 2; trên Kaggle: v5 probe_G1_b25x2 với inner_step_size)
    "grpo2": {
        0: [dict(_r, run="G1_s2", method="grpo_v1", set=SFT + ["seed=2"], args=GARGS)],
        1: [dict(_r, run="G1_s3", method="grpo_v1", set=SFT + ["seed=3"], args=GARGS)],
    },
    "grpo3": {
        0: [dict(_r, run="G3_s1", method="grpo_v1", set=SFT + ["seed=1", "rl_config={beta: 0.0}"], args=GARGS)],
        1: [dict(_r, run="G1_s4", method="grpo_v1", set=SFT + ["seed=4"], args=GARGS)],
    },
    "grpo4": {
        0: [dict(_r, run="G1_s5", method="grpo_v1", set=SFT + ["seed=5"], args=GARGS)],
        1: [],   # trống: ngày 2 (ít mẫu) hoặc chốt sau
    },
}
RUNS = JOBS[JOB]
print(f"JOB = {JOB}", flush=True)
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = []
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
# Tập MLAAD fake/vi (GĐ5 phát hiện 2), không bắt buộc: dataset orpheus-mlaad-vi nếu được gắn
mlaad_hits = sorted(glob.glob("/kaggle/input/**/mlaad_vi_test.csv", recursive=True))
roots = [sea, prep] + ([os.path.dirname(mlaad_hits[0])] if mlaad_hits else [])
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

common = ["--base", base, "--data-root", *roots, "--out", out, "--cache", cache]
# Tải trước checkpoint khởi tạo (một lần cho mỗi mô hình và cách khởi tạo), tránh hai GPU cùng
# tải một file. init: "post" (checkpoint post-trained, mặc định) hoặc "fairseq" (P0)
for m, ini in sorted({(r["model"], r.get("init", "post")) for runs in RUNS.values() for r in runs}):
    sh([sys.executable, f"{code}/finetune.py", *common, "--run", f"prefetch_{m}_{ini}", "--model", m,
        "--init", ini, "--prefetch"], check=True)

failures = []
probe_failures = []   # lần chạy đo ("probe"): hết bộ nhớ là kết quả, không phải lỗi của kernel


def worker(gpu, runs):
    for r in runs:
        dep = r.get("only_if_failed")
        if dep and not any(d in probe_failures + failures for d in dep):
            print(f"[GPU {gpu}] bỏ qua {r['run']} (chỉ chạy khi {dep} lỗi)", flush=True)
            continue
        cmd = [sys.executable, f"{code}/finetune.py", *common, "--gpu", str(gpu),
               "--run", r["run"], "--model", r["model"], "--method", r.get("method", "sft"),
               "--init", r.get("init", "post"),
               "--train", r["train"], "--valid", r["valid"], *r.get("args", []), "--set", *r["set"]]
        if r["score"]:
            cmd += ["--score", *r["score"]]
        logf = f"{out}/train/{r['run']}.driver.log"
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        t0 = time.time()
        print(f"[GPU {gpu}] bắt đầu {r['run']}", flush=True)
        with open(logf, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"[GPU {gpu}] xong {r['run']} (mã {rc}) sau {(time.time() - t0) / 60:.1f} phút", flush=True)
        sh(f"grep -E 'KẾT QUẢ|THẤT BẠI|lỗi|Lỗi|hết bộ nhớ' {logf} | tail -5")
        if r.get("probe"):
            sh(f"grep -h 'MEM_PROBE ' {out}/train/{r['run']}/main.log | tail -1; "
               f"grep -h -E 'OutOfMemoryError|CUDA out of memory' {out}/train/{r['run']}/main.log | tail -2")
        if rc != 0:
            (probe_failures if r.get("probe") else failures).append(r["run"])


HEARTBEAT_SEC = 600   # in tiến độ ra log chính mỗi 10 phút (log huấn luyện nằm ở file riêng)
BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s+([\d.]+)(s/it|it/s)(?:, train_loss=([\d.]+))?")


def tail_text(path, n=20000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def heartbeat(stop):
    """Mỗi HEARTBEAT_SEC: với mỗi lần chạy, bước huấn luyện mới nhất (thanh tqdm trong main.log),
    loss, bộ nhớ và mức dùng GPU (monitor.csv), hoặc dòng chấm điểm mới nhất (driver.log)."""
    while not stop.wait(HEARTBEAT_SEC):
        lines = [f"--- tiến độ sau {(time.time() - T0) / 60:.0f} phút ---"]
        for gpu, runs in RUNS.items():
            for r in runs:
                d = f"{out}/train/{r['run']}"
                drv = tail_text(f"{d}.driver.log", 4000).strip().splitlines()
                if not drv:
                    continue
                bars = BAR.findall(tail_text(f"{d}/main.log").replace("\r", "\n"))
                mon = tail_text(f"{d}/monitor.csv", 200).strip().splitlines()
                mon = mon[-1].split(",") if mon and not mon[-1].startswith("t,") else []
                if bars and not any("KẾT QUẢ HUẤN LUYỆN" in l for l in drv):
                    n, tot, el, rem, rate, unit, loss = bars[-1]
                    lines.append(f"[GPU {gpu}] {r['run']}: batch {n} (epoch {int(n) // max(int(tot), 1) + 1}, "
                                 f"{tot} batch/epoch), {rate} {unit}, loss {loss or '?'}"
                                 + (f", GPU {mon[1]} MiB {mon[2]}%, CPU {mon[3]}%" if len(mon) == 4 else ""))
                else:
                    lines.append(f"[GPU {gpu}] {r['run']}: {drv[-1][:200]}")
        zs = tail_text(f"{out}/zeroshot.driver.log", 4000).strip().splitlines()
        if zs:
            lines.append(f"[zero-shot] {zs[-1][:200]}")
        print("\n".join(lines), flush=True)


hb_stop = threading.Event()
threading.Thread(target=heartbeat, args=(hb_stop,), daemon=True).start()
threads = [threading.Thread(target=worker, args=(g, runs)) for g, runs in RUNS.items()]
for t in threads:
    t.start()
    time.sleep(20)
for t in threads:
    t.join()
print(f"Huấn luyện và chấm sau fine-tune xong sau {(time.time() - T0) / 3600:.2f} giờ", flush=True)

if ZEROSHOT:
    cmd = [sys.executable, f"{code}/score_zeroshot.py", "--base", base, "--data-root", *roots,
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

hb_stop.set()
for f in ("train_summary.csv", "scores/summary.csv", "zeroshot/summary.csv"):
    sh(f"cat {out}/{f}")
sh(f"du -sh {out}/*")
if probe_failures:
    print(f"Lần đo không chạy hết (xem oom trong train_summary.csv): {probe_failures}", flush=True)
print(f"Thất bại: {failures}" if failures else "Mọi lần chạy xong", flush=True)
print(f"Tổng thời gian {(time.time() - T0) / 3600:.2f} giờ", flush=True)
sys.exit(1 if failures else 0)
