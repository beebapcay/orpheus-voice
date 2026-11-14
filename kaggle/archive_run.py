"""Lưu bằng chứng của một lần chạy kernel Kaggle vào results/runs/<kernel>-v<N>/ (commit lên git).

Dùng cho báo cáo và để tái hiện: code đã chạy, log, môi trường, thời gian, kết quả.
push.sh gọi script này; chỉ dùng thư viện chuẩn.

  python3 kaggle/archive_run.py pushed  --run-dir D --build-dir B --kernel K --version N
      lúc đẩy kernel: chép run.py, kernel-metadata.json, ghi pushed.json (giờ đẩy, commit git)
  python3 kaggle/archive_run.py collect --run-dir D --raw R --kernel K --version N [--log L] [--quota Q]
      sau khi kernel xong: chép log, môi trường, log từng shard, summary.csv; viết RUN.md

Không chép: embedding (emb.npy, *_emb.pkl), điểm từng shard (đã gộp trong scores.csv),
dữ liệu âm thanh.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# File nhỏ của mỗi lần huấn luyện (src/finetune.py) được lưu làm bằng chứng
TRAIN_FILES = (".log", "monitor.csv", "exp_log.txt", "valid_history.json", "hyperparams.yaml",
               "best_CKPT.yaml", ".txt")
WARN_RE = re.compile(r"Traceback|Error|is not in the model|Wrong para\. length|failures\.json")
IGNORE_RE = re.compile(r"SyntaxWarning|mistune|nbconvert|filter_links|Ignoring unknown extended header")


def git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def now():
    t = dt.datetime.now().astimezone()
    return {"local": t.isoformat(timespec="seconds"),
            "utc": t.astimezone(dt.timezone.utc).isoformat(timespec="seconds")}


def cmd_pushed(a):
    os.makedirs(a.run_dir, exist_ok=True)
    for f in ("run.py", "kernel-metadata.json"):
        shutil.copy(os.path.join(a.build_dir, f), a.run_dir)
    info = {
        "kernel": a.kernel, "version": a.version, "pushed_at": now(),
        "repo_commit": git("rev-parse", "--short", "HEAD"),
        "repo_dirty_kaggle_src": bool(git("status", "--porcelain", "--", "kaggle", "src", "protocols")),
        "note": "run.py và kernel-metadata.json trong thư mục này là bản đã đẩy lên Kaggle "
                "(KAGGLE_USER đã được thay bằng tên tài khoản)."
    }
    with open(os.path.join(a.run_dir, "pushed.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("Đã lưu bản đẩy vào", a.run_dir)


def read_log(path):
    """Log của Kaggle là mảng JSON [{stream_name, time, data}, ...]."""
    if not path or not os.path.exists(path):
        return []
    raw = open(path, encoding="utf-8").read().strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Đôi khi thiếu dấu ] cuối khi kernel bị dừng
        try:
            return json.loads(raw.rstrip(",") + "]")
        except json.JSONDecodeError:
            return [{"stream_name": "stdout", "time": 0, "data": raw}]


def cmd_collect(a):
    os.makedirs(a.run_dir, exist_ok=True)
    res = os.path.join(a.raw, "results")
    copied = []

    # 1. Log kernel: bản gốc (JSON) và bản đọc được
    log_path = a.log or next(iter(glob.glob(os.path.join(a.raw, "*.log"))), None)
    entries = read_log(log_path)
    if log_path:
        dst = os.path.join(a.run_dir, "kernel.log.json")
        if not os.path.exists(dst) or not os.path.samefile(log_path, dst):
            shutil.copy(log_path, dst)
        with open(os.path.join(a.run_dir, "kernel.log.txt"), "w", encoding="utf-8") as f:
            for e in entries:
                tag = "" if e.get("stream_name") == "stdout" else "[stderr] "
                for line in str(e.get("data", "")).splitlines():
                    f.write(f"{float(e.get('time', 0)):9.1f}s {tag}{line}\n")
        copied += ["kernel.log.json", "kernel.log.txt"]

    # 2. Môi trường và tóm tắt của riêng lần chạy này. Kernel fine-tune có hai bảng: chấm sau
    # fine-tune (results/scores/) và chấm zero-shot (results/zeroshot/); chép tường minh để
    # không lẫn (glob đệ quy bên dưới lấy file đầu tiên tìm thấy).
    for rel, name in (("scores/summary.csv", "summary.csv"),
                      ("zeroshot/summary.csv", "zeroshot_summary.csv"),
                      ("zeroshot.driver.log", "zeroshot.driver.log")):
        p = os.path.join(res, rel)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(a.run_dir, name))
            copied.append(name)
    # Độ đo theo tập con SEA-Spoof (src/eval_seaspoof.py), mỗi (mô hình, tập) một file
    for p in sorted(glob.glob(os.path.join(res, "*", "*", "seaspoof_vi_*", "metrics_subsets.csv"))):
        model, dataset = p.split(os.sep)[-3:-1]
        dst = os.path.join(a.run_dir, "subsets", f"{model}__{dataset}.csv")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)
    for f in ("summary.csv", "train_summary.csv", "setup.log", "versions.txt", "pip-freeze.txt",
              "failures.json", "orpheus_data_manifest.json"):
        for p in glob.glob(os.path.join(res, f)) + glob.glob(os.path.join(a.raw, "**", f), recursive=True)[:1]:
            if os.path.exists(p) and not os.path.exists(os.path.join(a.run_dir, f)):
                shutil.copy(p, a.run_dir)
                copied.append(f)

    # 3a. Kernel fine-tune: log huấn luyện của từng lần (không có checkpoint)
    for p in sorted(glob.glob(os.path.join(res, "train", "*.driver.log"))
                    + [q for q in glob.glob(os.path.join(res, "train", "*", "*"))
                       if os.path.isfile(q) and q.endswith(TRAIN_FILES)]):
        dst = os.path.join(a.run_dir, "train_logs", os.path.relpath(p, os.path.join(res, "train")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)

    # 3b. Log và thời gian của từng shard (main.py), log của baseline
    n_shard = 0
    for p in sorted(glob.glob(os.path.join(res, "**", "shards", "*"), recursive=True)):
        if p.endswith((".log", ".done")):
            rel = os.path.relpath(p, res).replace(os.sep + "shards" + os.sep, os.sep)
            dst = os.path.join(a.run_dir, "shard_logs", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(p, dst)
            n_shard += 1
    for p in sorted(glob.glob(os.path.join(res, "*", "*", "run.log"))
                    + glob.glob(os.path.join(res, "*", "*", "timing.json"))):
        dst = os.path.join(a.run_dir, "shard_logs", os.path.relpath(p, res))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)
        n_shard += 1

    n_shard = sum(len(fs) for _, _, fs in os.walk(os.path.join(a.run_dir, "shard_logs")))

    if a.quota:
        shutil.copy(a.quota, os.path.join(a.run_dir, "quota_after.txt"))
        copied.append("quota_after.txt")

    write_run_md(a, entries, copied, n_shard)
    print("Đã lưu bằng chứng vào", a.run_dir)


def write_run_md(a, entries, copied, n_shard):
    text = [str(e.get("data", "")) for e in entries]
    stdout = "".join(t for e, t in zip(entries, text) if e.get("stream_name") == "stdout")
    pushed = {}
    pj = os.path.join(a.run_dir, "pushed.json")
    if os.path.exists(pj):
        pushed = json.load(open(pj, encoding="utf-8"))

    def grab(pattern):
        m = re.search(pattern, stdout, re.M)
        return m.group(1).strip() if m else "?"

    stamps = re.findall(r"^\[(\d\d:\d\d:\d\d)\]", stdout, re.M)
    last_t = max((float(e.get("time", 0)) for e in entries), default=0)
    warn = sorted({l.strip() for t in text for l in t.splitlines()
                   if WARN_RE.search(l) and not IGNORE_RE.search(l)})

    jobs = "?"
    rp = os.path.join(a.run_dir, "run.py")
    if os.path.exists(rp):
        src = open(rp, encoding="utf-8").read()
        m = re.search(r"^JOBS = \[(.*?)^\]", src, re.S | re.M)
        if m:
            jobs = "\n".join("- `" + j + "`" for j in re.findall(r'"([^"]+)"', m.group(1)))
        m = re.search(r"^(PILOT = .*?\n)?^RUNS = \{(.*?)^\}", src, re.S | re.M)
        if m:
            jobs = "```python\n" + m.group(0).strip() + "\n```"
        # Kernel fine-tune: lấy cả khối cấu hình giữa hai dòng đánh dấu (SFT, SCORE, RUNS, ZEROSHOT)
        m = re.search(r"^# ---- Sửa phần này[^\n]*\n(.*?)^# -{20,}", src, re.S | re.M)
        if m and "RUNS = {" in m.group(1):
            jobs = "```python\n" + m.group(1).strip() + "\n```"

    rows = []
    for name in ("summary.csv", "zeroshot_summary.csv"):
        sp = os.path.join(a.run_dir, name)
        if os.path.exists(sp):
            rows += list(csv.DictReader(open(sp, encoding="utf-8")))
    subsets = []
    for p in sorted(glob.glob(os.path.join(a.run_dir, "subsets", "*.csv"))):
        model, dataset = os.path.basename(p)[:-4].split("__", 1)
        eer = {r["subset"]: r["eer_%"] for r in csv.DictReader(open(p, encoding="utf-8"))}
        subsets.append((model, dataset, eer))
    trows = []
    tp = os.path.join(a.run_dir, "train_summary.csv")
    if os.path.exists(tp):
        trows = list(csv.DictReader(open(tp, encoding="utf-8")))

    L = [f"# {a.kernel} version {a.version}", "",
         "File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.", "",
         "## Thông tin chạy", "",
         "| | |", "|---|---|",
         f"| Kernel | `{a.kernel}`, version {a.version} (riêng tư trên Kaggle) |",
         f"| Đẩy lúc | {pushed.get('pushed_at', {}).get('local', '? (xem Ghi chú)')} |",
         f"| Commit repo lúc đẩy | `{pushed.get('repo_commit', '?')}` |",
         f"| Code trong dataset `orpheus-src` | `{grab(r'^(git [0-9a-f]+.*)$')}`, {grab(r'^(staged .*)$')} |",
         f"| Bắt đầu (UTC, dòng log đầu có giờ) | {stamps[0] if stamps else '?'} |",
         f"| Kết thúc (UTC, dòng log cuối có giờ) | {stamps[-1] if stamps else '?'} |",
         f"| Thời gian chạy | {grab(r'^Tổng thời gian (.*)$')} (log ghi đến giây {last_t:.0f}) |",
         f"| Dựng môi trường | {grab(r'^Dựng môi trường xong sau (.*)$')} |",
         f"| GPU | {grab(r'^(Tesla [^,]+, \d+ MiB)')} (xem `kernel.log.txt` cho đủ) |",
         f"| Lưu trữ lúc | {now()['local']} |", "",
         "## Job", "", jobs, ""]
    if trows:
        keys = ["run", "status", "train", "train_seconds", "batches_per_epoch", "sec_per_it",
                "last_steps", "max_batch_length", "peak_mem_mib", "gpu_util_mean", "cpu_util_mean",
                "n_valid", "best_valid_eer", "last_lr"]
        L += ["## Huấn luyện (train_summary.csv)", "",
              "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in trows:
            L.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
        L += ["", "Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.", ""]
    if rows:
        L += ["## Kết quả của lần chạy này", "",
              "| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['dataset']} | {r['model']} | {r['n']} | {r['audio_h']} | {r['gpu_seconds']} | "
                     f"{r['x_realtime']} | {float(r['eer_%']):.4f} | {float(r['minDCF']):.4f} | "
                     f"{float(r['actDCF']):.4f} |")
        L.append("")
    if subsets:
        cols = ["pooled", "test_seen", "test_f5", "test_minimax", "test_cvreal"]
        L += ["## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)", "",
              "| Mô hình | Tập | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for model, dataset, eer in subsets:
            L.append(f"| {model} | {dataset} | "
                     + " | ".join(f"{float(eer[c]):.2f}" if c in eer else "" for c in cols) + " |")
        L.append("")
    L += ["## Dòng log đáng chú ý", "",
          "Các dòng chứa Traceback, Error hoặc cảnh báo nạp trọng số (đã bỏ SyntaxWarning của nbconvert):", ""]
    L += (["```"] + warn[:50] + ["```"]) if warn else ["Không có."]
    L += ["", "## File trong thư mục này", "",
          "| File | Nội dung |", "|---|---|",
          "| `run.py`, `kernel-metadata.json` | Code kernel và cấu hình máy đúng như đã đẩy |",
          "| `pushed.json` | Giờ đẩy, commit repo |",
          "| `kernel.log.json` | Log gốc tải từ Kaggle |",
          "| `kernel.log.txt` | Cùng log, dạng đọc được (giây tính từ lúc kernel bắt đầu) |",
          "| `summary.csv` | Kết quả riêng của lần chạy này (bảng gộp ở `results/zeroshot/summary.csv`) |",
          "| `setup.log`, `versions.txt`, `pip-freeze.txt` | Môi trường thực tế |",
          f"| `shard_logs/` | Log `main.py` và thời gian của từng shard ({n_shard} file) |",
          ] + (["| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |"]
               if os.path.exists(os.path.join(a.run_dir, "quota_after.txt")) else []) + [
          "",
          "Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu "
          "(ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).", "",
          "## Tái hiện", "",
          "1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).",
          "2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.",
          "3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.",
          "4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định "
          "(`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.", "",
          "## Ghi chú", "", "(viết tay)", ""]
    path = os.path.join(a.run_dir, "RUN.md")
    if os.path.exists(path):
        # Giữ phần Ghi chú đã viết tay
        old = open(path, encoding="utf-8").read()
        if "## Ghi chú" in old:
            L = L[:-4] + ["## Ghi chú", "", old.split("## Ghi chú", 1)[1].strip(), ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


def merge_csv(src, dst, key, tag):
    """Gộp src vào dst: dòng mới thay dòng cũ cùng khoá. Cột tag=(tên, giá trị) chỉ gán cho dòng
    mới (dòng cũ giữ nguồn của nó, để trống nếu chưa có cột này). Hợp cột của hai file."""
    if not os.path.exists(src):
        return
    new = list(csv.DictReader(open(src, encoding="utf-8")))
    old = list(csv.DictReader(open(dst, encoding="utf-8"))) if os.path.exists(dst) else []
    k = lambda r: tuple(r.get(x) for x in key)  # noqa: E731
    seen = {k(r) for r in new}
    for r in new:
        r[tag[0]] = tag[1]
    rows = new + [r for r in old if k(r) not in seen]
    fields = []
    for r in rows:
        fields += [f for f in r if f not in fields]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cmd_merge_csv(a):
    """push.sh output zeroshot: gộp summary.csv của một lần chạy vào results/zeroshot/summary.csv."""
    merge_csv(a.src, a.dst, a.key.split(","), tuple(a.tag.split("=", 1)))


def cmd_merge_finetune(a):
    """Chép kết quả kernel fine-tune vào results/finetune/ (commit): train_summary.csv và
    summary.csv gộp (dòng mới thay dòng cũ cùng khoá), điểm và độ đo từng (lần chạy, tập).
    Không chép embedding, shard, checkpoint."""
    res = os.path.join(a.raw, "results")

    merge_csv(os.path.join(res, "train_summary.csv"), os.path.join(a.dst, "train_summary.csv"), ["run"],
              ("kaggle_version", a.version))
    keep = ("scores.csv", "metrics.csv", "metrics.txt", "metrics_subsets.csv", "metrics_subsets.txt")
    # Chấm sau fine-tune -> results/finetune/; chấm zero-shot trong cùng kernel (mốc S0) ->
    # results/zeroshot/ (cạnh kết quả zero-shot GĐ3, cùng cách gộp theo (tập, mô hình))
    for sub, dst_dir in (("scores", a.dst), ("zeroshot", os.path.join(os.path.dirname(a.dst), "zeroshot"))):
        if not os.path.isdir(os.path.join(res, sub)):
            continue
        tag = ("kaggle_version", a.version) if sub == "scores" else ("source", f"orpheus-finetune-v{a.version}")
        merge_csv(os.path.join(res, sub, "summary.csv"), os.path.join(dst_dir, "summary.csv"),
              ["dataset", "model"], tag)
        n = 0
        for p in glob.glob(os.path.join(res, sub, "*", "*", "*")):
            if os.path.basename(p) in keep:
                dst = os.path.join(dst_dir, os.path.relpath(p, os.path.join(res, sub)))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(p, dst)
                n += 1
        print(f"Đã gộp {sub}/summary.csv và {n} file điểm/độ đo vào {dst_dir}")
    print(f"Đã gộp train_summary.csv vào {a.dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pushed", "collect", "merge-finetune", "merge-csv"])
    p.add_argument("--src")
    p.add_argument("--key", default="dataset,model")
    p.add_argument("--tag", default="source=?")
    p.add_argument("--dst")
    p.add_argument("--run-dir")
    p.add_argument("--kernel")
    p.add_argument("--version", default="?")
    p.add_argument("--build-dir")
    p.add_argument("--raw")
    p.add_argument("--log")
    p.add_argument("--quota")
    a = p.parse_args()
    {"pushed": cmd_pushed, "collect": cmd_collect, "merge-finetune": cmd_merge_finetune,
     "merge-csv": cmd_merge_csv}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
