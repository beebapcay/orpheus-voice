#!/usr/bin/env python3
"""Gộp kết quả server (đã `vast.sh fetch` về results/server/raw/) vào cây kết quả riêng của bộ server,
và lưu bằng chứng lần chạy. Chỉ dùng thư viện chuẩn (chạy bằng python3 của máy).

  python3 server/collect.py --raw results/server/raw --dst results/server --runs results/runs/server-<nhãn> --tag <nhãn>

Cây kết quả của bộ server tách hẳn khỏi bộ Kaggle (results/finetune, results/zeroshot), để phân tích
không trộn hai phần cứng:
  results/server/finetune/<lần>/<tập>/   scores.csv, metrics*.csv|txt (chấm sau fine-tune)
  results/server/zeroshot/<mô hình>/<tập>/ (S0, mốc XLS-R-2B)
  results/server/train_summary.csv, finetune/summary.csv, zeroshot/summary.csv
Phân tích: python src/analyze_ablation.py --finetune results/server/finetune --zeroshot results/server/zeroshot
           --out results/server/analysis (tương tự calibrate.py, analyze_mechanism.py).

Bằng chứng (chỉ lưu trên máy chủ đồ án; results/runs/ không đưa lên repo chung): results/runs/server-<nhãn>/ gồm code đã chạy (VERSION, server/*.py|sh), môi
trường (env/: máy, phiên bản, pip freeze, log dựng, toy example), driver.log, status.txt, trạng thái
và log từng lần (jobs/), log huấn luyện (train_logs/<lần>/), train_summary.csv, summary chấm, RUN.md.
Không lưu: checkpoint, embedding, âm thanh.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import time

KEEP_SCORE = ("scores.csv", "metrics.csv", "metrics.txt", "metrics_subsets.csv", "metrics_subsets.txt", "timing.json")
KEEP_TRAIN = ("main.log", "monitor.csv", "exp_log.txt", "hyperparams.yaml", "valid_history.json", "preflight.log",
              "best_CKPT.yaml", "env.log", "log.txt", "interp.json")
WARN = re.compile(r"Traceback|Error|is not in the model|Wrong para\. length|OutOfMemory|LỖI|THẤT BẠI")


def copy_tree(src_root, dst_root, keep):
    n = 0
    for p in glob.glob(os.path.join(src_root, "*", "*", "*")):
        if os.path.basename(p) in keep:
            d = os.path.join(dst_root, os.path.relpath(p, src_root))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(p, d)
            n += 1
    return n


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def md_table(rows, keys):
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    raw, dst, run = a.raw, a.dst, a.runs

    # 1. Kết quả cho phân tích
    n1 = copy_tree(os.path.join(raw, "scores"), os.path.join(dst, "finetune"), KEEP_SCORE)
    n2 = copy_tree(os.path.join(raw, "zeroshot"), os.path.join(dst, "zeroshot"), KEEP_SCORE)
    for s, d in (("train_summary.csv", "train_summary.csv"), ("scores/summary.csv", "finetune/summary.csv"),
                 ("zeroshot/summary.csv", "zeroshot/summary.csv")):
        if os.path.exists(os.path.join(raw, s)):
            os.makedirs(os.path.dirname(os.path.join(dst, d)) or dst, exist_ok=True)
            shutil.copy2(os.path.join(raw, s), os.path.join(dst, d))
    print(f"Đã chép {n1} file điểm/độ đo sau fine-tune, {n2} file zero-shot vào {dst}")

    # 2. Bằng chứng
    os.makedirs(run, exist_ok=True)
    for sub in ("env", "jobs", "_code"):
        if os.path.isdir(os.path.join(raw, sub)):
            shutil.copytree(os.path.join(raw, sub), os.path.join(run, "code" if sub == "_code" else sub),
                            dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "host.env"))
    for f in ("driver.log", "status.txt", "train_summary.csv"):
        if os.path.exists(os.path.join(raw, f)):
            shutil.copy2(os.path.join(raw, f), run)
    for s, d in (("scores/summary.csv", "scores_summary.csv"), ("zeroshot/summary.csv", "zeroshot_summary.csv")):
        if os.path.exists(os.path.join(raw, s)):
            shutil.copy2(os.path.join(raw, s), os.path.join(run, d))
    nt = 0
    for d in sorted(glob.glob(os.path.join(raw, "train", "*"))):
        if not os.path.isdir(d):
            continue
        for f in KEEP_TRAIN:
            if os.path.exists(os.path.join(d, f)):
                os.makedirs(os.path.join(run, "train_logs", os.path.basename(d)), exist_ok=True)
                shutil.copy2(os.path.join(d, f), os.path.join(run, "train_logs", os.path.basename(d), f))
                nt += 1

    # 3. RUN.md (phần sinh tự động; giữ mục Ghi chú viết tay nếu đã có)
    jobs = []
    for p in sorted(glob.glob(os.path.join(raw, "jobs", "*.json"))):
        st = json.load(open(p))
        st["name"] = os.path.basename(p)[:-5]
        st["hours"] = f"{st['seconds'] / 3600:.2f}" if "seconds" in st else ""
        st["start_local"] = time.strftime("%m-%d %H:%M", time.localtime(st["start"])) if "start" in st else ""
        jobs.append(st)
    jobs.sort(key=lambda s: (s.get("tier", 9), s.get("start", 0)))
    train = read_csv(os.path.join(raw, "train_summary.csv"))
    scores = read_csv(os.path.join(raw, "scores", "summary.csv")) + read_csv(os.path.join(raw, "zeroshot", "summary.csv"))
    machine = open(os.path.join(raw, "env", "machine.txt"), errors="replace").read() if os.path.exists(
        os.path.join(raw, "env", "machine.txt")) else ""
    gpus = sorted(set(re.findall(r"^\d+, (NVIDIA [^,]+|[A-Z][^,]*GeForce[^,]*|[^,]+), (\d+ MiB)", machine, re.M)))
    version = open(os.path.join(raw, "_code", "VERSION")).read().strip() if os.path.exists(
        os.path.join(raw, "_code", "VERSION")) else "?"
    warns = []
    for p in sorted(glob.glob(os.path.join(raw, "jobs", "*.log"))):
        for line in open(p, errors="replace"):
            if WARN.search(line) and "FutureWarning" not in line:
                warns.append(f"{os.path.basename(p)}: {line.strip()[:220]}")
    old = open(os.path.join(run, "RUN.md")).read() if os.path.exists(os.path.join(run, "RUN.md")) else ""
    notes = old[old.index("## Ghi chú"):] if "## Ghi chú" in old else "## Ghi chú\n\n(viết tay)\n"

    L = [f"# Server {a.tag}", "", "File này do `server/collect.py` sinh ra. Mục **Ghi chú** cuối file viết tay.", "",
         "## Thông tin chạy", "", "| | |", "|---|---|",
         f"| Máy | {'; '.join(f'{g} ({m})' for g, m in gpus) or 'xem env/machine.txt'} |",
         f"| Code | `{version.splitlines()[0] if version else '?'}` (`code/VERSION`, `code/server/`) |",
         f"| Số lần chạy | {len(jobs)}: " + ", ".join(f"{k} {sum(j['state'] == k for j in jobs)}"
                                            for k in ("done", "failed", "running", "pending")) + " |",
         f"| Gộp lúc | {time.strftime('%Y-%m-%d %H:%M:%S')} |",
         "| Môi trường | `env/machine.txt`, `env/versions.txt`, `env/pip-freeze.txt`, `env/driver-pip-freeze.txt`, `env/toy_check.txt` |",
         "", "## Các lần chạy (jobs/)", ""]
    L += md_table(jobs, ["tier", "name", "state", "rc", "gpu", "start_local", "hours"])
    if train:
        L += ["", "## Huấn luyện (train_summary.csv)", ""]
        L += md_table(train, ["run", "status", "method", "train", "train_seconds", "batches_per_epoch", "sec_per_it",
                              "last_steps", "last_lr", "peak_alloc_mib", "gpu_util_mean", "cpu_util_mean", "oom"])
    if scores:
        L += ["", "## Chấm điểm (summary.csv; EER %, thấp hơn là tốt hơn)", ""]
        L += md_table(scores, ["dataset", "model", "n", "audio_h", "gpu_seconds", "x_realtime", "eer_%", "minDCF", "actDCF"])
    L += ["", "## Dòng log đáng chú ý", "", "```", *(warns[:60] or ["Không có."]), "```", "",
          f"Log huấn luyện từng lần: `train_logs/` ({nt} file). Checkpoint ở server `results/ckpt/` (không tải về).", "",
          notes]
    open(os.path.join(run, "RUN.md"), "w").write("\n".join(L))
    print(f"Đã lưu bằng chứng vào {run} ({len(jobs)} lần, {nt} file log huấn luyện)")


if __name__ == "__main__":
    main()
