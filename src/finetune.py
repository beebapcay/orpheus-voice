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

# Trọng số fairseq gốc (chỉ pre-train, chưa post-train) cho cấu hình P0 (--init fairseq).
# README AntiDeepfake mục "Download checkpoints" trỏ tới trang pretrained models của fairseq;
# MMS-300M: examples/mms (base_300m.pt, 3,81 GB, kiểm HEAD 14/11/2026).
FAIRSEQ_URLS = {"mms_300m": "https://dl.fbaipublicfiles.com/mms/pretraining/base_300m.pt"}
# Khi khởi tạo từ fairseq, các khoá này của checkpoint không có trong mô hình thì vẫn chấp nhận
# (chỉ dùng cho học tự giám sát, không đi vào nhánh phân loại); khoá khác thiếu là lỗi.
FAIRSEQ_OK_MISSING = ("quantizer", "project_q", "final_proj", "label_embs", "_float_tensor")


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


def preflight_load(args, ckpt_init):
    """Nạp checkpoint khởi tạo vào mô hình bằng đúng hàm của code gốc (utils.load_weights, cùng
    name_mapper với SSLBrain._init_model) trên CPU, trước khi huấn luyện; trả về các dòng cảnh báo.
    main.py chỉ in cảnh báo rồi chạy tiếp, nên nếu không kiểm trước thì phải chờ hết lần huấn
    luyện mới biết trọng số nạp sai."""
    backbone = args.model.replace("_nda", "")
    code = ("import sys; sys.path.insert(0, '.'); import torch; from models.W2V import Model; "
            "from utils import load_weights; m = Model(" + repr(backbone) + "); "
            "s = m.state_dict(); n = sum(v.numel() for v in s.values()); "
            "load_weights(s, " + repr(ckpt_init) + ", func_name_change=lambda x: f'm_ssl.model.{x}'); "
            "print(f'PREFLIGHT_OK {len(s)} tensor, {n} tham số')")
    log_path = os.path.join(args.out, "train", args.run, "preflight.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    rc = sz.run_cmd([args.mm, "run", "-n", "antideepfake", "python", "-c", code],
                    os.path.join(args.base, "AntiDeepfake"), env, log_path)
    text = open(log_path, errors="replace").read()
    if rc != 0 or "PREFLIGHT_OK" not in text:
        raise RuntimeError(f"[{args.run}] kiểm nạp trọng số lỗi (mã {rc}):\n{sz.tail(log_path)}")
    warns = [l for l in text.splitlines() if any(p in l for p in sz.WARN_PATTERNS)]
    log(f"[{args.run}] kiểm nạp trọng số: {len(warns)} cảnh báo; "
        f"{[l for l in text.splitlines() if 'PREFLIGHT_OK' in l][0]}")
    return warns


def longest_subset(train_csv, n, path):
    """CSV gồm n câu dài nhất (theo cột Duration, cột mà DynamicBatchSampler dùng để chia batch)
    của train_csv: bài thử áp lực bộ nhớ, vì batch toàn câu dài tốn bộ nhớ attention nhiều nhất
    (bậc hai theo số khung) trong khi tổng thời lượng batch vẫn bị giới hạn như nhau."""
    rows = list(csv.DictReader(open(train_csv)))
    rows.sort(key=lambda r: float(r["Duration"]), reverse=True)
    rows = rows[:n]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    d = [float(r["Duration"]) for r in rows]
    log(f"tập con {n} câu dài nhất: {min(d):.2f}–{max(d):.2f} s, "
        f"{sum(r['Label'] == 'real' for r in rows)} thật -> {path}")
    return path


def parse_mem_probe(text):
    """Dòng MEM_PROBE cuối cùng (src/mem_probe.py): đỉnh bộ nhớ PyTorch thật sự cấp phát."""
    hits = re.findall(r"MEM_PROBE max_allocated_mib=(\d+) max_reserved_mib=(\d+) total_mib=(\d+)", text)
    if not hits:
        return {}
    a, r, t = hits[-1]
    return {"peak_alloc_mib": int(a), "peak_reserved_mib": int(r), "gpu_total_mib": int(t)}


def run_train(args, ckpt_init, overrides, allowed_warns=()):
    repo = os.path.join(args.base, "AntiDeepfake")
    yaml = f"hparams/ft_grpo/{args.model.replace('_nda', '')}_pre-post-{args.method}.yaml"
    run_dir = os.path.join(args.out, "train", args.run)
    exp = os.path.join(run_dir, "exp")
    os.makedirs(run_dir, exist_ok=True)
    train_csv, valid_csv = (sz.find_csv(args, n) for n in (args.train, args.valid))
    if not train_csv or not valid_csv:
        sys.exit(f"Không thấy {args.train}.csv hoặc {args.valid}.csv trong {args.data_root}")
    if args.train_longest:
        train_csv = longest_subset(train_csv, args.train_longest,
                                   os.path.join(run_dir, f"{args.train}_longest{args.train_longest}.csv"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")   # main.py bật use_deterministic_algorithms
    env.update(dict(kv.split("=", 1) for kv in args.env))
    ov = dict(overrides)
    # main.py gốc, hoặc main.py chạy qua mem_probe.py (cùng phép tính, ghi thêm đỉnh bộ nhớ thật)
    entry = [os.path.join(HERE, "mem_probe.py"), "main.py"] if args.probe_mem else ["main.py"]
    if args.loss_variant:
        # Hàm loss thay thế qua runpy (src/loss_variants.py), không sửa code gốc; ví dụ sft_kl = SFT + KL của GRPO
        entry = [os.path.join(HERE, "loss_variants.py"), args.loss_variant] + entry
    for attempt in range(args.oom_retry + 1):
        if os.path.exists(exp) and not args.resume:
            shutil.rmtree(exp)
        cmd = [args.mm, "run", "-n", "antideepfake", "python", *entry, yaml,
               "--base_path", args.base, "--output_folder", exp,
               "--train_csv", train_csv, "--valid_csv", valid_csv, "--ckpt_path", ckpt_init]
        for k, v in ov.items():
            cmd += [f"--{k}", str(v)]
        log(f"[{args.run}] huấn luyện, GPU {args.gpu}"
            + (f", env {' '.join(args.env)}" if args.env else "") + f": {' '.join(cmd[4:])}")
        log_path = os.path.join(run_dir, "main.log")
        mon = Monitor(args.gpu, os.path.join(run_dir, "monitor.csv"))
        mon.start()
        t0 = time.time()
        rc = sz.run_cmd(cmd, repo, env, log_path)
        secs = time.time() - t0
        mon.stop_ev.set()
        mon.join()
        text = open(log_path, errors="replace").read()
        warns = [l for l in text.splitlines() if any(p in l for p in sz.WARN_PATTERNS)
                 and l.strip() not in allowed_warns]
        if warns:
            raise RuntimeError(f"[{args.run}] cảnh báo nạp trọng số: {warns[:3]}")
        oom = rc != 0 and any(p in text for p in OOM_PATTERNS)
        if oom and attempt == args.oom_retry:
            # --oom-retry 0 (GRPO): không tự giảm batch, vì đổi batch là đổi số bước và lịch
            # learning rate (không còn so được với R, A0; GĐ5 mục 9, L1)
            log(f"[{args.run}] hết bộ nhớ GPU với max_batch_length={ov.get('max_batch_length', 100)}; "
                f"không thử lại (--oom-retry {args.oom_retry})")
            break
        if oom:
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
    if args.train_longest:
        info["train_longest"] = args.train_longest
    if args.env:
        info["env"] = " ".join(args.env)
    if args.loss_variant:
        info["loss_variant"] = args.loss_variant
    info["oom"] = any(p in text for p in OOM_PATTERNS)
    info.update(parse_train_log(text))
    # Tích luỹ gradient: thanh tqdm đếm batch; mỗi bước optimizer (và lịch learning rate) gồm
    # `grad_accumulation_factor` batch
    gaf = int(ov.get("grad_accumulation_factor", 1))
    info["grad_accumulation_factor"] = gaf
    if "sec_per_it" in info:
        info["sec_per_opt_step"] = round(info["sec_per_it"] * gaf, 3)
    info.update(parse_mem_probe(text))
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
    if args.discard_ckpt:
        # Lần chạy đo (bộ nhớ, tốc độ): không giữ mô hình, không chấm
        shutil.rmtree(os.path.join(exp, "save"), ignore_errors=True)
        info["status"] = "ok"
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
    # Khoá file: nhiều tiến trình finetune.py (mỗi GPU một) cùng đọc-ghi một train_summary.csv
    import fcntl
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
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
    ap.add_argument("--init", default="post", choices=["post", "fairseq"],
                    help="post: checkpoint post-trained AntiDeepfake (mặc định); fairseq: trọng số "
                         "pre-train gốc, chưa post-train (cấu hình P0)")
    ap.add_argument("--train", default="seaspoof_vi_train")
    ap.add_argument("--valid", default="seaspoof_vi_dev")
    ap.add_argument("--set", nargs="*", default=[], help="ghi đè YAML: khoá=giá trị")
    ap.add_argument("--score", nargs="*", default=[], help="các tập chấm sau fine-tune")
    ap.add_argument("--resume", action="store_true", help="không xoá thư mục exp cũ")
    ap.add_argument("--keep-exp", action="store_true", help="giữ save/ (có trạng thái optimizer)")
    ap.add_argument("--prefetch", action="store_true",
                    help="chỉ tải checkpoint khởi tạo vào cache rồi thoát (gọi trước khi chạy song song)")
    ap.add_argument("--oom-retry", type=int, default=2,
                    help="số lần giảm một nửa max_batch_length rồi chạy lại khi hết bộ nhớ GPU "
                         "(mặc định 2, như GĐ4–5); GRPO dùng 0 để không lặng lẽ đổi batch")
    ap.add_argument("--probe-mem", action="store_true",
                    help="chạy main.py qua mem_probe.py: ghi đỉnh bộ nhớ PyTorch thật (MEM_PROBE)")
    ap.add_argument("--train-longest", type=int, default=0,
                    help="chỉ huấn luyện trên N câu dài nhất của --train (thử áp lực bộ nhớ)")
    ap.add_argument("--discard-ckpt", action="store_true",
                    help="lần chạy đo: không giữ checkpoint, không chấm")
    ap.add_argument("--loss-variant", default=None, choices=["sft_kl"],
                    help="hàm loss thay thế (src/loss_variants.py): sft_kl = cross-entropy như SFT + beta*KL k3 như "
                         "GRPO; dùng với --method grpo_v2 (có mô hình tham chiếu)")
    ap.add_argument("--env", nargs="*", default=[],
                    help="biến môi trường thêm cho main.py, KHOÁ=giá trị "
                         "(ví dụ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)")
    args = ap.parse_args()
    args.mm = os.path.join(args.base, "bin", "micromamba")
    os.environ["MAMBA_ROOT_PREFIX"] = os.path.join(args.base, "mamba")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    sz.prepare_data_dir(args)
    overrides = dict(kv.split("=", 1) for kv in args.set)

    if args.init == "fairseq":
        # P0: khởi tạo từ trọng số fairseq gốc (chưa post-train); lớp FC phân loại khởi tạo ngẫu
        # nhiên theo seed (main.py: set_random_seed trước khi tạo mô hình)
        if args.model not in FAIRSEQ_URLS:
            sys.exit(f"Chưa có URL fairseq cho {args.model}")
        url = FAIRSEQ_URLS[args.model]
        ckpt_init = sz.download(url, os.path.join(args.cache, "fairseq_" + os.path.basename(url)))
    else:
        ckpts = sz.Checkpoints(args.cache, {args.model: 1})
        ckpt_init = ckpts.get(args.model)
    if args.prefetch:
        log(f"Đã có checkpoint khởi tạo {ckpt_init}")
        return
    warns = preflight_load(args, ckpt_init)
    allowed = ()
    if warns:
        bad = [w for w in warns if args.init != "fairseq" or " is not in the model" not in w
               or not w.strip().startswith(FAIRSEQ_OK_MISSING)]
        if bad:
            info = {"run": args.run, "model": args.model, "init": args.init, "status": "load_warning",
                    "load_warnings": len(warns)}
            append_summary(os.path.join(args.out, "train_summary.csv"), info)
            sys.exit(f"[{args.run}] nạp trọng số sai, dừng trước khi huấn luyện: {bad[:5]}")
        allowed = tuple(w.strip() for w in warns)
        log(f"[{args.run}] bỏ qua {len(warns)} khoá chỉ dùng cho học tự giám sát: {warns[:5]}")
    info, ckpt = run_train(args, ckpt_init, overrides, allowed)
    info["init"] = args.init
    info["load_warnings_allowed"] = len(allowed)
    append_summary(os.path.join(args.out, "train_summary.csv"), info)
    log(f"[{args.run}] KẾT QUẢ HUẤN LUYỆN: {json.dumps(info, ensure_ascii=False)}")
    if ckpt is None:
        sys.exit(0 if args.discard_ckpt and info.get("status") == "ok" else 1)
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
