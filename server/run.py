#!/usr/bin/env python3
"""Điều phối bộ thí nghiệm trên server (server/experiments.py). Chạy trên server, bằng Python của môi
trường orpheus-driver (server/bootstrap.sh tạo).

  python server/run.py list                       danh sách lần chạy theo tầng, trạng thái
  python server/run.py prefetch                   tải trước checkpoint khởi tạo (một lần, tuần tự)
  python server/run.py run --tiers 0              thử đo (tầng 0)
  python server/run.py run --tiers 1 2 3 4 5 6 [--workers N] [--gpus 0,1,2,3]
  (bỏ qua một lần đang chờ: thêm tên vào results/skip.txt; xoá results/jobs/<tên>.json để chạy lại)
  python server/run.py run --only R_s1 G1_s1      chỉ một số lần
  python server/run.py run ... --dry-run          in lệnh, không chạy
  python server/run.py status                     in results/status.txt

Mỗi GPU chạy một lần huấn luyện tại một thời điểm, lấy lần chưa xong có tầng nhỏ nhất (thứ tự trong
experiments.py). Trạng thái từng lần: results/jobs/<tên>.json (pending, running, done, failed) và log
results/jobs/<tên>.log. Chạy lại lệnh `run` thì bỏ qua lần đã xong; lần đang "running" của một lần điều
phối đã chết được đặt lại thành pending (finetune.py xoá thư mục exp cũ và chạy lại từ đầu, vì không
chạy tiếp giữa chừng được: GĐ5 L6). Lần lỗi không tự chạy lại: xem log, sửa, rồi `run --retry-failed`.

Mọi tiến trình con nhận NVIDIA_TF32_OVERRIDE=0 (tính fp32 như T4; GPU Ampere/Ada mặc định dùng TF32
cho convolution).
"""
import argparse
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
ENV = dict(os.environ, NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1")
BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s+([\d.]+)(s/it|it/s)(?:, train_loss=([\d.]+))?")
lock = threading.Lock()
T0 = time.time()


def log(msg):
    with lock:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def state_path(name):
    return os.path.join(JOBS, f"{name}.json")


def read_state(name):
    p = state_path(name)
    return json.load(open(p)) if os.path.exists(p) else {"state": "pending"}


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


def heartbeat(jobs, stop, every=60):
    while not stop.wait(every):
        write_status(jobs)


def write_status(jobs):
    rows, cnt, dur = [], {}, {}
    for j in jobs:
        st = read_state(j["name"])
        cnt[st["state"]] = cnt.get(st["state"], 0) + 1
        if st["state"] == "done" and "seconds" in st:
            dur.setdefault(j["kind"] + ("_grpo" if j.get("method", "").startswith("grpo") else ""), []).append(st["seconds"])
        if st["state"] == "running":
            rows.append(f"  GPU {st.get('gpu')}: {j['name']} (tầng {j['tier']}, "
                        f"{(time.time() - st.get('start', time.time())) / 60:.0f} phút) {progress(j['name'])}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        smi = "(không đọc được nvidia-smi)"
    mean = {k: sum(v) / len(v) / 3600 for k, v in dur.items()}
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  điều phối chạy {(time.time() - T0) / 3600:.2f} giờ",
             "Trạng thái: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())),
             "Thời gian TB mỗi lần đã xong (giờ): " + (", ".join(f"{k} {v:.2f}" for k, v in sorted(mean.items())) or "-"),
             "Đang chạy:", *(rows or ["  (không)"]), "GPU (index, %, MiB dùng, MiB tổng):", smi]
    failed = [j["name"] for j in jobs if read_state(j["name"])["state"] == "failed"]
    if failed:
        lines.append("LỖI: " + " ".join(failed))
    tmp = os.path.join(OUT, "status.txt.tmp")
    open(tmp, "w").write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(OUT, "status.txt"))


claimed = set()   # lần đã có GPU nhận trong lần điều phối này (nhận dưới khoá: hai GPU không lấy trùng)


def skip_list():
    """results/skip.txt: tên lần chạy cần bỏ qua (mỗi dòng một tên), đọc lại mỗi lần GPU nhận việc, nên
    thêm tên vào được khi điều phối đang chạy (`vast.sh skip <tên>...`)."""
    p = os.path.join(OUT, "skip.txt")
    return {l.strip() for l in open(p)} - {""} if os.path.exists(p) else set()


def worker(gpu, queue, jobs, dry):
    skipped = []
    while True:
        with lock:
            skip = skip_list()
            for x in queue:
                if x["name"] in skip and x["name"] not in claimed:
                    claimed.add(x["name"])
                    skipped.append(x["name"])
                    if not dry:
                        write_state(x["name"], state="skipped")
            j = next((x for x in queue if x["name"] not in claimed), None)
            if j is not None:
                claimed.add(j["name"])
                if not dry:
                    write_state(j["name"], state="running", gpu=gpu, start=time.time(), tier=j["tier"])
        # log() cũng lấy `lock` (không vào lại được), nên chỉ ghi log sau khi đã nhả khoá
        while skipped:
            log(f"bỏ qua {skipped.pop()} (results/skip.txt)")
        if j is None:
            return
        cmd = command(j, gpu)
        log(f"[GPU {gpu}] bắt đầu {j['name']} (tầng {j['tier']})")
        if dry:
            log(f"[GPU {gpu}] (dry-run) {' '.join(cmd)}")
            continue
        t0 = time.time()
        with open(os.path.join(JOBS, f"{j['name']}.log"), "w") as lf:
            lf.write(" ".join(cmd) + "\n")
            lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=ENV).returncode
        st = write_state(j["name"], state="done" if rc == 0 else "failed", rc=rc, end=time.time(),
                         seconds=round(time.time() - t0))
        log(f"[GPU {gpu}] {'xong' if rc == 0 else 'LỖI'} {j['name']} (mã {rc}) sau {st['seconds'] / 60:.1f} phút")
        write_status(jobs)


def select(args):
    jobs = E.build(args.workers)
    if args.only:
        sel = [j for j in jobs if j["name"] in args.only]
        missing = set(args.only) - {j["name"] for j in sel}
        if missing:
            sys.exit(f"không có lần chạy: {sorted(missing)}")
        return sel
    return [j for j in jobs if j["tier"] in args.tiers]


def cmd_run(args):
    os.makedirs(JOBS, exist_ok=True)
    jobs = select(args)
    if args.dry_run:                        # chạy khô: không ghi trạng thái
        queue = [j for j in jobs if read_state(j["name"])["state"] != "done"]
    else:
        for j in jobs:                      # lần "running" của lần điều phối trước (đã chết) -> pending
            st = read_state(j["name"])
            if st["state"] == "running" or (args.retry_failed and st["state"] == "failed"):
                write_state(j["name"], state="pending")
            elif not os.path.exists(state_path(j["name"])):
                write_state(j["name"], state="pending", tier=j["tier"])
        queue = [j for j in jobs if read_state(j["name"])["state"] == "pending"]
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(gpu_count()))
    if not gpus:
        sys.exit("không thấy GPU (nvidia-smi -L)")
    log(f"ROOT {ROOT}; GPU {gpus}; {len(queue)} lần chờ / {len(jobs)} lần đã chọn; "
        f"num_workers {args.workers}; NVIDIA_TF32_OVERRIDE={ENV['NVIDIA_TF32_OVERRIDE']}")
    if not args.dry_run:
        prefetch(queue)
    stop = threading.Event()
    if not args.dry_run:
        threading.Thread(target=heartbeat, args=(jobs, stop), daemon=True).start()
    threads = [threading.Thread(target=worker, args=(g, queue, jobs, args.dry_run)) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(0 if args.dry_run else 15)     # lệch giờ khởi động: giảm tranh CPU lúc nạp mô hình
    for t in threads:
        t.join()
    stop.set()
    if not args.dry_run:
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
    for j in E.build(args.workers):
        st = read_state(j["name"]) if os.path.isdir(JOBS) else {"state": "-"}
        extra = f"{st.get('seconds', 0) / 3600:.2f} giờ" if st.get("seconds") else ""
        print(f"tầng {j['tier']}  {j['name']:<16} {j['kind']:<8} {st['state']:<8} {extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "list", "prefetch", "status"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=E.NUM_WORKERS)
        if name == "run":
            p.add_argument("--tiers", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
            p.add_argument("--only", nargs="*", default=None)
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
