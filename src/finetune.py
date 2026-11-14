#!/usr/bin/env python3
"""Fine-tune một checkpoint AntiDeepfake trên SEA-Spoof tiếng Việt, rồi chấm sau fine-tune.

Dùng code gốc AntiDeepfake (`main.py`, `hparams/ft_grpo/<mô hình>_pre-post-<cách>.yaml`),
không sửa code; chỉ thêm phần điều phối:
  - ghi đè siêu tham số bằng tham số dòng lệnh của SpeechBrain (--set khoá=giá trị);
  - theo dõi GPU (bộ nhớ, mức dùng) và CPU mỗi 5 giây -> monitor.csv;
  - hết bộ nhớ GPU thì giảm một nửa max_batch_length rồi chạy lại (tối đa 2 lần);
  - đọc log: số batch mỗi epoch, giây mỗi bước, EER dev từng lần kiểm tra;
  - lấy checkpoint tốt nhất (EER dev thấp nhất, SpeechBrain giữ 1 bản) -> <run>.ckpt
    (state_dict cùng tên khoá với checkpoint post-trained), chấm bằng score_zeroshot.py --local;
  - ghi một dòng vào train_summary.csv.

Lưu ý về code gốc (chore/GD4-Du-Lieu-Va-Pipeline.md mục 2.2):
  - dev chỉ được chấm mỗi `valid_step` bước (cấu hình 20.000) và sau epoch cuối;
  - learning rate tăng tuyến tính tới `lr` trong `warmup` bước (cấu hình 80.000);
  - khi use_da=True, RawBoost áp lên cả dev (audio_pipeline dùng chung).

Ví dụ:
  python src/finetune.py --base /tmp/w --data-root D1 D2 --out O --cache C --gpu 0 \
      --run R_s1 --model mms_300m --method sft \
      --train seaspoof_vi_train --valid seaspoof_vi_dev \
      --set number_of_epochs=10 valid_step=800 warmup=800 optimizer_step_limit=8000 seed=1 \
      --score seaspoof_vi_test radar_vi ood_forget
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_zeroshot as sz  # noqa: E402

OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Monitor(threading.Thread):
    """Ghi bộ nhớ, mức dùng GPU và mức dùng CPU (toàn máy) mỗi `every` giây."""

    def __init__(self, gpu, path, every=5):
        super().__init__(daemon=True)
        self.gpu, self.path, self.every = gpu, path, every
        self.stop_ev = threading.Event()

    @staticmethod
    def _cpu():
        a = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(a), a[3] + a[4]          # tổng, idle + iowait

    def run(self):
        last = self._cpu()
        with open(self.path, "w") as f:
            f.write("t,mem_mib,gpu_util,cpu_util\n")
            t0 = time.time()
            while not self.stop_ev.wait(self.every):
                try:
                    q = subprocess.run(["nvidia-smi", "-i", str(self.gpu), "--query-gpu=memory.used,"
                                        "utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.strip()
                    mem, util = [x.strip() for x in q.split(",")]
                except Exception:  # noqa: BLE001
                    mem, util = "", ""
                cur = self._cpu()
                dt, di = cur[0] - last[0], cur[1] - last[1]
                last = cur
                cpu = round(100 * (1 - di / dt), 1) if dt else ""
                f.write(f"{time.time() - t0:.0f},{mem},{util},{cpu}\n")
                f.flush()

    def summary(self):
        rows = list(csv.DictReader(open(self.path)))

        def col(k):
            return [float(r[k]) for r in rows if r[k] not in ("", None)]
        mem, gu, cu = col("mem_mib"), col("gpu_util"), col("cpu_util")
        avg = lambda v: round(sum(v) / len(v), 1) if v else None  # noqa: E731
        return {"peak_mem_mib": max(mem) if mem else None, "gpu_util_mean": avg(gu),
                "cpu_util_mean": avg(cu), "monitor_samples": len(rows)}


def parse_train_log(text):
    """Số batch mỗi epoch và tốc độ từ thanh tqdm của main.py; EER dev từ dòng kiểm tra."""
    info = {}
    # chỉ thanh của bước huấn luyện (có train_loss); thanh kiểm tra dev cuối cùng thì không
    bars = re.findall(r"(\d+)/(\d+) \[([\d:]+)<[\d:?]+,\s+([\d.]+)(s/it|it/s), train_loss", text)
    if bars:
        n, total, _, rate, unit = bars[-1]
        info["batches_per_epoch"] = int(total)
        info["sec_per_it"] = round(float(rate) if unit == "s/it" else 1 / float(rate), 3)
    return info


def parse_exp_log(path):
    """exp_log.txt (FileTrainLogger): 'epoch: 1, steps: 300, lr: ... - valid loss: ..., valid eer: ...'"""
    rows = []
    if os.path.exists(path):
        for line in open(path):
            m = re.search(r"epoch: (\d+), steps: (\d+), lr: ([\d.e+-]+).*valid loss: (?:tensor\()?"
                          r"([\d.e+-]+).*valid eer: (?:tensor\()?([\d.e+-]+)", line)
            if m:
                rows.append({"epoch": int(m.group(1)), "steps": int(m.group(2)),
                             "lr": float(m.group(3)), "valid_loss": float(m.group(4)),
                             "valid_eer": float(m.group(5))})
    return rows


def run_train(args, ckpt_init, overrides):
    repo = os.path.join(args.base, "AntiDeepfake")
    yaml = f"hparams/ft_grpo/{args.model.replace('_nda', '')}_pre-post-{args.method}.yaml"
    run_dir = os.path.join(args.out, "train", args.run)
    exp = os.path.join(run_dir, "exp")
    os.makedirs(run_dir, exist_ok=True)
    train_csv, valid_csv = (sz.find_csv(args, n) for n in (args.train, args.valid))
    if not train_csv or not valid_csv:
        sys.exit(f"Không thấy {args.train}.csv hoặc {args.valid}.csv trong {args.data_root}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")   # main.py bật use_deterministic_algorithms
    ov = dict(overrides)
    for attempt in range(3):
        if os.path.exists(exp) and not args.resume:
            shutil.rmtree(exp)
        cmd = [args.mm, "run", "-n", "antideepfake", "python", "main.py", yaml,
               "--base_path", args.base, "--output_folder", exp,
               "--train_csv", train_csv, "--valid_csv", valid_csv, "--ckpt_path", ckpt_init]
        for k, v in ov.items():
            cmd += [f"--{k}", str(v)]
        log(f"[{args.run}] huấn luyện, GPU {args.gpu}: {' '.join(cmd[4:])}")
        log_path = os.path.join(run_dir, "main.log")
        mon = Monitor(args.gpu, os.path.join(run_dir, "monitor.csv"))
        mon.start()
        t0 = time.time()
        rc = sz.run_cmd(cmd, repo, env, log_path)
        secs = time.time() - t0
        mon.stop_ev.set()
        mon.join()
        text = open(log_path, errors="replace").read()
        warns = [l for l in text.splitlines() if any(p in l for p in sz.WARN_PATTERNS)]
        if warns:
            raise RuntimeError(f"[{args.run}] cảnh báo nạp trọng số: {warns[:3]}")
        if rc != 0 and any(p in text for p in OOM_PATTERNS):
            mbl = float(ov.get("max_batch_length", 100)) / 2
            log(f"[{args.run}] hết bộ nhớ GPU với max_batch_length={ov.get('max_batch_length', 100)}; "
                f"thử lại với {mbl:g}")
            shutil.copy(log_path, os.path.join(run_dir, f"main.oom{attempt}.log"))
            ov["max_batch_length"] = mbl
            continue
        break
    info = {"run": args.run, "model": args.model, "method": args.method, "train": args.train,
            "valid": args.valid, "returncode": rc, "train_seconds": round(secs),
            "overrides": json.dumps(ov), "max_batch_length": ov.get("max_batch_length", 100)}
    info.update(parse_train_log(text))
    info.update(mon.summary())
    vals = parse_exp_log(os.path.join(exp, "exp_log.txt"))
    json.dump(vals, open(os.path.join(run_dir, "valid_history.json"), "w"), indent=1)
    if vals:
        best = min(vals, key=lambda r: r["valid_eer"])
        info.update({"n_valid": len(vals), "last_steps": vals[-1]["steps"], "last_lr": vals[-1]["lr"],
                     "best_valid_eer": best["valid_eer"], "best_steps": best["steps"]})
    for f in ("exp_log.txt", "hyperparams.yaml", "env.log", "log.txt"):
        if os.path.exists(os.path.join(exp, f)):
            shutil.copy(os.path.join(exp, f), run_dir)
    if rc != 0:
        info["status"] = "error"
        log(f"[{args.run}] main.py lỗi (mã {rc}). Cuối log:\n{sz.tail(log_path)}")
        return info, None
    found = sorted(glob.glob(os.path.join(exp, "save", "CKPT+*", "ssl.ckpt")))
    if not found:
        info["status"] = "no_ckpt"
        return info, None
    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"{args.run}.ckpt")
    shutil.copy(found[-1], ckpt)
    meta = os.path.join(os.path.dirname(found[-1]), "CKPT.yaml")
    if os.path.exists(meta):
        shutil.copy(meta, os.path.join(run_dir, "best_CKPT.yaml"))
    info["ckpt_mb"] = round(os.path.getsize(ckpt) / 1e6)
    if not args.keep_exp:
        shutil.rmtree(os.path.join(exp, "save"), ignore_errors=True)   # optimizer ~2x mô hình
    info["status"] = "ok"
    return info, ckpt


def append_summary(path, row):
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    rows = [r for r in rows if r.get("run") != row["run"]] + [row]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-root", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--run", required=True, help="tên lần chạy, ví dụ R_s1")
    ap.add_argument("--model", default="mms_300m", help="checkpoint post-trained khởi tạo (khoá MODELS)")
    ap.add_argument("--method", default="sft", help="sft | grpo_v1 | grpo_v2 (tên file YAML ft_grpo)")
    ap.add_argument("--train", default="seaspoof_vi_train")
    ap.add_argument("--valid", default="seaspoof_vi_dev")
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè YAML: khoá=giá trị")
    ap.add_argument("--score", nargs="*", default=[], help="các tập chấm sau fine-tune")
    ap.add_argument("--resume", action="store_true", help="không xoá thư mục exp cũ")
    ap.add_argument("--keep-exp", action="store_true", help="giữ save/ (có trạng thái optimizer)")
    ap.add_argument("--prefetch", action="store_true",
                    help="chỉ tải checkpoint khởi tạo vào cache rồi thoát (gọi trước khi chạy song song)")
    args = ap.parse_args()
    args.mm = os.path.join(args.base, "bin", "micromamba")
    os.environ["MAMBA_ROOT_PREFIX"] = os.path.join(args.base, "mamba")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    sz.prepare_data_dir(args)
    overrides = dict(kv.split("=", 1) for kv in args.set)

    ckpts = sz.Checkpoints(args.cache, {args.model: 1})
    ckpt_init = ckpts.get(args.model)
    if args.prefetch:
        log(f"Đã có checkpoint khởi tạo {ckpt_init}")
        return
    info, ckpt = run_train(args, ckpt_init, overrides)
    append_summary(os.path.join(args.out, "train_summary.csv"), info)
    log(f"[{args.run}] KẾT QUẢ HUẤN LUYỆN: {json.dumps(info, ensure_ascii=False)}")
    if ckpt is None:
        sys.exit(1)
    if args.score:
        base_model = args.model
        cmd = [sys.executable, os.path.join(HERE, "score_zeroshot.py"), "--base", args.base,
               "--data-root", *args.data_root, "--out", os.path.join(args.out, "scores"),
               "--cache", args.cache, "--gpus", str(args.gpu),
               "--local", f"{args.run}={ckpt},{base_model}",
               "--jobs", *[f"{d}:{args.run}" for d in args.score]]
        log(f"[{args.run}] chấm: {' '.join(cmd[2:])}")
        rc = subprocess.run(cmd).returncode
        sys.exit(rc)


if __name__ == "__main__":
    main()
