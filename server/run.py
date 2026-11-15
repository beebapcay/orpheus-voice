#!/usr/bin/env python3
"""Điều phối bộ thí nghiệm trên server (server/experiments.py). Chạy trên server, bằng Python của môi
trường orpheus-driver (server/bootstrap.sh tạo).

  python server/run.py list                       danh sách lần chạy theo tầng, trạng thái
  python server/run.py prefetch                   tải trước checkpoint khởi tạo (một lần, tuần tự)
  python server/run.py run --tiers 0              thử đo (tầng 0)
  python server/run.py run [--workers N] [--gpus 0,1,2,3]   mọi tầng ≥ 1 (hoặc --tiers 1 2 ...)
  python server/run.py run --only R_s1 G1_s1      chỉ một số lần
  python server/run.py run ... --dry-run          in lệnh, không chạy
  python server/run.py status                     in results/status.txt

Hàng đợi động: mỗi lần một GPU nhận việc, run.py ĐỌC LẠI experiments.py, nên thêm, bớt lần chạy ở điểm kiểm (sửa
file rồi `vast.sh code`) có hiệu lực ngay mà không dừng lần đang chạy. GPU lấy lần chưa xong có tầng nhỏ nhất (thứ tự
trong experiments.py) mà mọi lần nó cần (`needs`, ví dụ WiSE-FT cần checkpoint của lần nguồn) đã xong; chưa xong thì
lấy lần sau, hết việc làm được thì chờ. Lần cần đã lỗi hoặc bị bỏ thì lần phụ thuộc thành "skipped".

Điều khiển khi đang chạy (từ máy: `vast.sh skip`, `vast.sh hold`, `vast.sh release`):
  results/skip.txt   tên lần chạy cần bỏ qua (mỗi dòng một tên)
  results/hold       có file này thì GPU rảnh không nhận việc mới (lần đang chạy vẫn chạy tiếp)
  chạy lại một lần đã lỗi/bỏ: xoá results/jobs/<tên>.json (và tên trong skip.txt); không cần khởi động lại

Trạng thái từng lần: results/jobs/<tên>.json (pending, running, done, failed, skipped) và log results/jobs/<tên>.log.
Khởi động lại `run` thì bỏ qua lần đã xong; lần "running" của một lần điều phối đã chết được đặt lại thành pending
(finetune.py xoá thư mục exp cũ và chạy lại từ đầu, vì không chạy tiếp giữa chừng được: GĐ5 L6). Lần lỗi không tự
chạy lại: xem log, sửa, rồi xoá file trạng thái hoặc `run --retry-failed`.

Mọi tiến trình con nhận NVIDIA_TF32_OVERRIDE=0 (tính fp32 như T4: SpeechBrain 1.0.2 tự bật TF32 khi import, GPU
Ampere/Ada dùng TF32 cho matmul và convolution nếu không ghi đè).
"""
import argparse
import glob
import importlib
import shutil
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import experiments as E  # noqa: E402

ROOT = os.environ.get("ORPHEUS_ROOT", "/workspace/orpheus")
CODE, BASE, CACHE = (os.path.join(ROOT, d) for d in ("code", "work", "cache"))
DATA = [os.path.join(ROOT, "data", d) for d in ("seaspoof", "prep", "mlaad")]
OUT = os.path.join(ROOT, "results")
JOBS = os.path.join(OUT, "jobs")
HOLD = os.path.join(OUT, "hold")
ENV = dict(os.environ, NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1")
BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s+([\d.]+)(s/it|it/s)(?:, train_loss=([\d.]+))?")
print_lock = threading.Lock()     # chỉ để in log
pick_lock = threading.Lock()      # chọn việc, đọc lại experiments.py (không gọi log() khi đang giữ)
running_here = set()              # lần đang chạy do điều phối này khởi động
T0 = time.time()
WAIT = 60
MIN_FREE_GB = 25      # ổ còn ít hơn thì không nhận việc mới (một lần huấn luyện cần ~4 GB tạm + 1,3 GB checkpoint)


def log(msg):
    with print_lock:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def state_path(name):
    return os.path.join(JOBS, f"{name}.json")


def read_state(name):
    p = state_path(name)
    try:
        return json.load(open(p)) if os.path.exists(p) else {"state": "pending"}
    except (OSError, ValueError):
        return {"state": "pending"}


def write_state(name, **kw):
    st = read_state(name)
    st.update(kw)
    tmp = state_path(name) + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, state_path(name))
    return st


def command(j, gpu):
    py = sys.executable
    if j["kind"] == "zeroshot":
        # --keep-ckpt: cache dùng chung với các lần fine-tune khởi tạo từ cùng checkpoint (chấm S0 xong mà xoá
        # cache/mms_300m.ckpt thì lần R/G1/A0 đang khởi động ở GPU khác báo "Not find" hoặc tải lại trùng nhau)
        return [py, os.path.join(CODE, "score_zeroshot.py"), "--base", BASE, "--data-root", *DATA,
                "--out", os.path.join(OUT, "zeroshot"), "--cache", CACHE, "--gpus", str(gpu), "--keep-ckpt",
                "--jobs", *j["jobs"]]
    if j["kind"] == "interp":
        return [py, os.path.join(CODE, "interp_score.py"), "--base", BASE, "--data-root", *DATA, "--out", OUT,
                "--cache", CACHE, "--gpu", str(gpu), "--name", j["name"], "--src", j["src"],
                "--alpha", str(j["alpha"]), "--model", j["model"], "--score", *j["score"]]
    cmd = [py, os.path.join(CODE, "finetune.py"), "--base", BASE, "--data-root", *DATA, "--out", OUT,
           "--cache", CACHE, "--gpu", str(gpu), "--run", j["name"], "--model", j["model"], "--method", j["method"],
           "--init", j["init"], "--train", j["train"], "--valid", j["valid"], *j["args"], "--set", *j["set"]]
    if j["score"]:
        cmd += ["--score", *j["score"]]
    return cmd


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:  # noqa: BLE001
        return 0


def tail(path, n=20000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def progress(name):
    """Bước huấn luyện mới nhất (thanh tqdm của main.log) hoặc dòng cuối của log lần chạy."""
    bars = BAR.findall(tail(os.path.join(OUT, "train", name, "main.log")).replace("\r", "\n"))
    drv = tail(os.path.join(JOBS, f"{name}.log"), 3000).strip().splitlines()
    scoring = any("KẾT QUẢ HUẤN LUYỆN" in l for l in drv) or not bars
    if bars and not scoring:
        n, tot, _, _, rate, unit, loss = bars[-1]
        return f"batch {n} ({tot}/epoch), {rate} {unit}, loss {loss or '?'}"
    return (drv[-1][:160] if drv else "khởi động")


def free_gb():
    try:
        return shutil.disk_usage(OUT).free / 1e9
    except OSError:
        return float("inf")


def cleanup_shards(j):
    """Sau khi chấm xong: xoá embedding theo shard (*_emb.pkl), bản sao của emb.npy đã gộp, chỉ cần để chạy tiếp
    giữa chừng trên Kaggle. Giữ emb.npy, điểm, độ đo. Tiết kiệm ~160 MB mỗi lần chấm 5 tập (~20 GB cả bộ)."""
    if j["kind"] == "zeroshot":
        dirs = [os.path.join(OUT, "zeroshot", spec.split(":", 1)[1].split(",")[0], spec.split(":", 1)[0])
                for spec in j["jobs"]]
    else:
        dirs = glob.glob(os.path.join(OUT, "scores", j["name"], "*"))
    n = 0
    for d in dirs:
        if os.path.exists(os.path.join(d, "emb.npy")):
            for f in glob.glob(os.path.join(d, "shards", "*_emb.pkl")):
                os.remove(f)
                n += 1
    return n


def skip_list():
    """results/skip.txt: tên lần chạy cần bỏ qua (mỗi dòng một tên), đọc lại mỗi lần GPU nhận việc."""
    p = os.path.join(OUT, "skip.txt")
    return {l.strip() for l in open(p)} - {""} if os.path.exists(p) else set()


def select(args, mod=None):
    mod = mod or E
    # Sắp ổn định theo tầng: lần thêm vào cuối build() ở điểm kiểm vẫn đứng đúng chỗ theo tầng của nó
    jobs = sorted(mod.build(args.workers), key=lambda j: j["tier"])
    if args.only:
        sel = [j for j in jobs if j["name"] in args.only]
        missing = set(args.only) - {j["name"] for j in sel}
        if missing:
            raise SystemExit(f"không có lần chạy: {sorted(missing)}")
        return sel
    tiers = set(args.tiers) if args.tiers else {j["tier"] for j in jobs if j["tier"] >= 1}
    return [j for j in jobs if j["tier"] in tiers]


def load_jobs(args, msgs):
    """Đọc lại experiments.py rồi chọn lần chạy (gọi khi đang giữ pick_lock; lỗi thì dùng bản đã nạp trước)."""
    global E
    try:
        E = importlib.reload(E)
    except Exception as e:  # noqa: BLE001
        msgs.append(f"CẢNH BÁO: đọc lại experiments.py lỗi ({e}); dùng bản đã nạp trước")
    try:
        return select(args)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        msgs.append(f"CẢNH BÁO: experiments.build() lỗi ({e}); không nhận việc mới lần này")
        return []


def pick(jobs, gpu, msgs):
    """Chọn lần tiếp theo cho GPU (đang giữ pick_lock). Trả về (lần chạy hoặc None, có lần đang chờ phụ thuộc)."""
    skip, names, waiting = skip_list(), {j["name"] for j in jobs}, False
    for j in jobs:
        n = j["name"]
        if n in running_here or read_state(n)["state"] != "pending":
            continue
        if n in skip:
            write_state(n, state="skipped", reason="results/skip.txt", tier=j["tier"])
            msgs.append(f"bỏ qua {n} (results/skip.txt)")
            continue
        needs = j.get("needs", [])
        st_needs = {d: read_state(d)["state"] for d in needs}
        bad = [d for d, s in st_needs.items() if s in ("failed", "skipped") or (s != "done" and d not in names)]
        if bad:
            write_state(n, state="skipped", reason=f"lần cần lỗi, bị bỏ hoặc không trong hàng đợi: {bad}", tier=j["tier"])
            msgs.append(f"bỏ qua {n}: lần cần {bad} lỗi, bị bỏ hoặc không trong hàng đợi")
            continue
        if any(s != "done" for s in st_needs.values()):
            waiting = True
            continue
        running_here.add(n)
        write_state(n, state="running", gpu=gpu, start=time.time(), tier=j["tier"])
        return j, waiting
    return None, waiting


def worker(gpu, args):
    held = low_disk = False
    while True:
        if os.path.exists(HOLD):
            if not held:
                log(f"[GPU {gpu}] tạm dừng nhận việc (có {HOLD})")
                held = True
            time.sleep(WAIT)
            continue
        if held:
            log(f"[GPU {gpu}] nhận việc lại")
            held = False
        if free_gb() < MIN_FREE_GB:
            if not low_disk:
                log(f"[GPU {gpu}] CẢNH BÁO: ổ chỉ còn {free_gb():.0f} GB (< {MIN_FREE_GB}); tạm dừng nhận việc tới khi đủ")
                low_disk = True
            time.sleep(WAIT)
            continue
        low_disk = False
        msgs = []
        with pick_lock:
            jobs = load_jobs(args, msgs)
            j, waiting = pick(jobs, gpu, msgs)
        for m in msgs:
            log(m)
        if j is None:
            # Chờ (không thoát) khi còn lần đang chờ phụ thuộc, hoặc GPU khác còn đang chạy: lần thêm vào
            # experiments.py ở điểm kiểm vẫn có GPU nhận. Thoát khi hết việc và không còn lần nào đang chạy.
            with pick_lock:
                busy = bool(running_here)
            if waiting or busy:
                time.sleep(WAIT)
                continue
            return
        cmd = command(j, gpu)
        log(f"[GPU {gpu}] bắt đầu {j['name']} (tầng {j['tier']})")
        t0 = time.time()
        with open(os.path.join(JOBS, f"{j['name']}.log"), "w") as lf:
            lf.write(" ".join(cmd) + "\n")
            lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV).returncode
        st = write_state(j["name"], state="done" if rc == 0 else "failed", rc=rc, end=time.time(),
                         seconds=round(time.time() - t0))
        with pick_lock:
            running_here.discard(j["name"])
        log(f"[GPU {gpu}] {'xong' if rc == 0 else 'LỖI'} {j['name']} (mã {rc}) sau {st['seconds'] / 60:.1f} phút")
        if rc == 0:
            try:
                cleanup_shards(j)
            except OSError as e:
                log(f"[GPU {gpu}] dọn embedding theo shard của {j['name']} lỗi: {e}")
        with pick_lock:
            jobs = select(args)
        write_status(jobs)


def kind_key(j):
    return j["kind"] + ("_grpo" if j.get("method", "").startswith("grpo") else "")


def write_status(jobs):
    rows, cnt, dur, failed = [], {}, {}, []
    for j in jobs:
        st = read_state(j["name"])
        s = st["state"]
        if s == "pending" and any(read_state(d)["state"] != "done" for d in j.get("needs", [])):
            s = "chờ phụ thuộc"
        cnt[s] = cnt.get(s, 0) + 1
        if st["state"] == "done" and "seconds" in st:
            dur.setdefault(kind_key(j), []).append(st["seconds"])
        if st["state"] == "running":
            rows.append(f"  GPU {st.get('gpu')}: {j['name']} (tầng {j['tier']}, "
                        f"{(time.time() - st.get('start', time.time())) / 60:.0f} phút) {progress(j['name'])}")
        if st["state"] == "failed":
            failed.append(j["name"])
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        smi = "(không đọc được nvidia-smi)"
    mean = {k: sum(v) / len(v) / 3600 for k, v in dur.items()}
    by_tier = {}
    for j in jobs:
        if read_state(j["name"])["state"] not in ("done", "skipped"):
            by_tier[j["tier"]] = by_tier.get(j["tier"], 0) + 1
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  điều phối chạy {(time.time() - T0) / 3600:.2f} giờ"
             + ("  [TẠM DỪNG NHẬN VIỆC: results/hold]" if os.path.exists(HOLD) else ""),
             "Trạng thái: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())),
             f"Ổ: còn {free_gb():.0f} GB" + (f" (DƯỚI {MIN_FREE_GB} GB: không nhận việc mới)" if free_gb() < MIN_FREE_GB else ""),
             "Còn lại theo tầng: " + (", ".join(f"tầng {t}: {n}" for t, n in sorted(by_tier.items())) or "-"),
             "Thời gian TB mỗi lần đã xong (giờ): " + (", ".join(f"{k} {v:.2f}" for k, v in sorted(mean.items())) or "-"),
             "Đang chạy:", *(rows or ["  (không)"]), "GPU (index, %, MiB dùng, MiB tổng):", smi]
    if failed:
        lines.append("LỖI: " + " ".join(failed))
    tmp = os.path.join(OUT, f"status.txt.tmp{threading.get_ident()}")   # mỗi luồng một file tạm
    open(tmp, "w").write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(OUT, "status.txt"))


def heartbeat(args, stop, every=60):
    while not stop.wait(every):
        try:
            with pick_lock:
                jobs = select(args)
            write_status(jobs)
        except Exception as e:  # noqa: BLE001
            log(f"heartbeat lỗi: {e}")


def cmd_run(args):
    os.makedirs(JOBS, exist_ok=True)
    jobs = select(args)
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(gpu_count()))
    if args.dry_run:                        # chạy khô: không ghi trạng thái, không xét phụ thuộc
        if not gpus:
            gpus = [0]
        queue = [j for j in jobs if read_state(j["name"])["state"] != "done"]
        log(f"(dry-run) GPU {gpus}; {len(queue)} lần chờ / {len(jobs)} lần đã chọn; num_workers {args.workers}")
        for i, j in enumerate(queue):
            g = gpus[i % len(gpus)]
            needs = f" (cần {', '.join(j['needs'])})" if j.get("needs") else ""
            log(f"[GPU {g}] (dry-run) tầng {j['tier']} {j['name']}{needs}: {' '.join(command(j, g))}")
        return
    for j in jobs:                          # lần "running" của lần điều phối trước (đã chết) -> pending
        st = read_state(j["name"])
        if st["state"] == "running" or (args.retry_failed and st["state"] == "failed"):
            write_state(j["name"], state="pending")
    pending = [j for j in jobs if read_state(j["name"])["state"] == "pending"]
    if not gpus:
        sys.exit("không thấy GPU (nvidia-smi -L)")
    log(f"ROOT {ROOT}; GPU {gpus}; {len(pending)} lần chờ / {len(jobs)} lần đã chọn; "
        f"num_workers {args.workers}; NVIDIA_TF32_OVERRIDE={ENV['NVIDIA_TF32_OVERRIDE']}; hàng đợi động (đọc lại "
        f"experiments.py mỗi lần nhận việc)")
    prefetch(pending)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(args, stop), daemon=True).start()
    threads = [threading.Thread(target=worker, args=(g, args)) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(15)                      # lệch giờ khởi động: giảm tranh CPU lúc nạp mô hình
    for t in threads:
        t.join()
    stop.set()
    jobs = select(args)
    write_status(jobs)
    done = sum(read_state(j["name"])["state"] == "done" for j in jobs)
    log(f"Hết hàng đợi: {done}/{len(jobs)} lần xong; điều phối chạy {(time.time() - T0) / 3600:.2f} giờ")


def prefetch(jobs):
    for m, ini in E.prefetch_list(jobs):
        cmd = [sys.executable, os.path.join(CODE, "finetune.py"), "--base", BASE, "--data-root", *DATA, "--out", OUT,
               "--cache", CACHE, "--run", f"prefetch_{m}_{ini}", "--model", m, "--init", ini, "--prefetch"]
        log(f"tải trước {m} ({ini})")
        rc = subprocess.run(cmd, env=ENV).returncode
        if rc != 0:
            sys.exit(f"tải trước {m} ({ini}) lỗi (mã {rc})")


def cmd_list(args):
    for j in select(args) if (args.tiers or args.only) else E.build(args.workers):
        st = read_state(j["name"]) if os.path.isdir(JOBS) else {"state": "-"}
        extra = f"{st.get('seconds', 0) / 3600:.2f} giờ" if st.get("seconds") else ""
        needs = f"cần {','.join(j['needs'])}" if j.get("needs") else ""
        print(f"tầng {j['tier']}  {j['name']:<16} {j['kind']:<8} {st['state']:<8} {extra} {needs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "list", "prefetch", "status"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=E.NUM_WORKERS)
        p.add_argument("--tiers", type=int, nargs="*", default=None, help="mặc định: mọi tầng ≥ 1")
        p.add_argument("--only", nargs="*", default=None)
        if name == "run":
            p.add_argument("--gpus", default=None, help="ví dụ 0,1,2,3; mặc định mọi GPU")
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "list":
        cmd_list(a)
    elif a.cmd == "prefetch":
        prefetch(E.build(a.workers))
    else:
        p = os.path.join(OUT, "status.txt")
        print(open(p).read() if os.path.exists(p) else "chưa có status.txt")


if __name__ == "__main__":
    main()
