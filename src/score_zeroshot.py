#!/usr/bin/env python3
"""Chấm zero-shot các checkpoint AntiDeepfake (và baseline W2V2-AASIST) trên nhiều tập.

Dùng code gốc của AntiDeepfake (`main.py analysis`), chỉ thêm phần điều phối:
  - chia mỗi tập thành shard; shard nào có kết quả thì bỏ qua, nên chạy tiếp được
    sau khi mất phiên;
  - mỗi shard một exp_name mới (tránh bẫy: thư mục exp đã có checkpoint thì
    inference nạp checkpoint đó thay vì --ckpt_path);
  - dò cảnh báo nạp trọng số trong log, có cảnh báo thì dừng job;
  - chạy song song trên nhiều GPU (Kaggle T4 x2), giới hạn RAM khi nạp mô hình lớn;
  - gộp shard, tính EER/minDCF/actDCF bằng src/evaluate.py, ghi thời gian chạy.

Môi trường: chạy bằng Python hệ thống (cần pandas, numpy, torch, safetensors để
chuyển checkpoint). AntiDeepfake chạy trong env micromamba do
setup_antideepfake.sh dựng.

Ví dụ:
  python src/score_zeroshot.py --base /tmp/work --data-root /kaggle/input/.../data \
      --out /kaggle/working/results --cache /tmp/cache --gpus 0,1 \
      --jobs toy:mms_300m radar_vi:mms_300m,w2v_small itw_full:mms_300m

Mỗi phần tử của --jobs có dạng <tập>:<mô hình>[,<mô hình>...].
Tập: toy, radar_vi, radar_en_sub, itw_sub, itw_full, ood_forget (CSV <tập>.csv trong --data-root).
Mô hình: xem MODELS bên dưới; thêm hậu tố _nda cho bản không RawBoost; w2v2_aasist là baseline.
"""
import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# khoá -> (file YAML, repo Hugging Face, tên file trên Zenodo, RAM cần khi nạp, GB)
MODELS = {
    "w2v_small": ("w2v_small.yaml", "wav2vec-small-anti-deepfake", "w2v_small.ckpt", 2),
    "w2v_large": ("w2v_large.yaml", "wav2vec-large-anti-deepfake", "w2v_large.ckpt", 4),
    "mms_300m": ("mms_300m.yaml", "mms-300m-anti-deepfake", "mms_300m.ckpt", 4),
    "mms_1b": ("mms_1b.yaml", "mms-1b-anti-deepfake", "mms_1b.ckpt", 9),
    "xlsr_1b": ("xlsr_1b.yaml", "xls-r-1b-anti-deepfake", "xlsr_1b.ckpt", 9),
    "hubert_xl": ("hubert_xtralarge.yaml", "hubert-xlarge-anti-deepfake", "hubert_xl.ckpt", 9),
    "xlsr_2b": ("xlsr_2b.yaml", "xls-r-2b-anti-deepfake", "xlsr_2b.ckpt", 19),
}
for _k in list(MODELS):
    y, hf, zf, ram = MODELS[_k]
    MODELS[_k + "_nda"] = (y, hf + "-nda", zf.replace(".ckpt", "_nda.ckpt"), ram)

HF_URL = "https://huggingface.co/nii-yamagishilab/{repo}/resolve/main/model.safetensors"
ZENODO_URL = "https://zenodo.org/records/15580543/files/{name}"
AASIST_ONNX = "https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST/resolve/main/w2v2-aasist.onnx"
TOY_URL = "https://zenodo.org/records/7497769/files/project-04-toy_example.tar.gz"
WARN_PATTERNS = ("is not in the model", "Wrong para. length")

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = time.time()
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            os.replace(dest + ".part", dest)
            size = os.path.getsize(dest) / 1e9
            dt = time.time() - t0
            log(f"  tải {os.path.basename(dest)}: {size:.2f} GB, {dt:.0f} s ({size * 1e3 / max(dt, 1):.0f} MB/s)")
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"  tải lỗi lần {attempt}: {e}")
            time.sleep(15 * attempt)
    raise RuntimeError(f"không tải được {url}")


class Checkpoints:
    """Tải và chuyển checkpoint một lần cho mỗi mô hình; xoá khi job cuối của mô hình xong."""

    def __init__(self, cache, refcount):
        self.cache = cache
        self.refcount = dict(refcount)
        self.locks = {m: threading.Lock() for m in refcount}
        self.guard = threading.Lock()

    def get(self, model):
        yaml, repo, zname, _ = MODELS[model]
        if repo is None:                  # checkpoint cục bộ (--local), ví dụ sau fine-tune
            return zname
        ckpt = os.path.join(self.cache, zname)
        with self.locks[model]:
            if os.path.exists(ckpt):
                return ckpt
            log(f"[{model}] tải checkpoint từ Hugging Face ({repo})")
            st = os.path.join(self.cache, zname.replace(".ckpt", ".safetensors"))
            try:
                download(HF_URL.format(repo=repo), st)
                # model.safetensors trên HF là state_dict của cùng lớp mô hình
                # (PyTorchModelHubMixin), cùng tên khoá với file .ckpt trên Zenodo.
                from safetensors.torch import load_file
                import torch
                state = load_file(st)
                torch.save(state, ckpt + ".part")
                os.replace(ckpt + ".part", ckpt)
                os.remove(st)
                log(f"[{model}] đã chuyển safetensors -> {os.path.basename(ckpt)} ({len(state)} tensor)")
            except Exception as e:  # noqa: BLE001
                log(f"[{model}] HF lỗi ({e}); tải bản .ckpt từ Zenodo (chậm hơn)")
                download(ZENODO_URL.format(name=zname), ckpt)
            return ckpt

    def release(self, model):
        with self.guard:
            self.refcount[model] -= 1
            last = self.refcount[model] == 0
        if last and MODELS[model][1] is not None:     # không xoá checkpoint cục bộ
            for f in glob.glob(os.path.join(self.cache, MODELS[model][2].replace(".ckpt", "") + ".*")):
                os.remove(f)
            log(f"[{model}] xong mọi tập, xoá checkpoint khỏi cache")


class RamBudget:
    """Không cho hai mô hình lớn nạp cùng lúc vượt RAM máy."""

    def __init__(self, gb):
        self.free = gb
        self.cv = threading.Condition()

    def acquire(self, gb):
        with self.cv:
            while self.free < gb:
                self.cv.wait()
            self.free -= gb

    def release(self, gb):
        with self.cv:
            self.free += gb
            self.cv.notify_all()


def available_ram_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 16.0


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


def make_shards(df, size):
    return [df.iloc[i:i + size] for i in range(0, len(df), size)]


def run_cmd(cmd, cwd, env, log_path):
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode


def tail(path, n=40):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


def job_antideepfake(args, ckpts, ram, gpu, dataset, model, csv_path, outdir):
    import numpy as np
    import pandas as pd
    import pickle

    df = read_csv(csv_path)
    shards = make_shards(df, args.shard_size)
    os.makedirs(os.path.join(outdir, "shards"), exist_ok=True)
    todo = [k for k in range(len(shards))
            if not os.path.exists(os.path.join(outdir, "shards", f"{k:03d}.done"))]
    log(f"[{model}/{dataset}] GPU {gpu}: {len(df)} câu, {len(shards)} shard, còn {len(todo)}")
    if todo:
        ckpt = ckpts.get(model)
        yaml = MODELS[model][0]
        repo_dir = os.path.join(args.base, "AntiDeepfake")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        for k in todo:
            shard = shards[k]
            sdir = os.path.join(outdir, "shards")
            shard_csv = os.path.join(sdir, f"{k:03d}_protocol.csv")
            shard.to_csv(shard_csv, index=False)
            exp = f"zs_{model}_{dataset}_{k:03d}"
            for old in glob.glob(os.path.join(args.base, "Log", "exps", f"exp_*_{exp}")):
                shutil.rmtree(old)
            log_path = os.path.join(sdir, f"{k:03d}.log")
            cmd = [args.mm, "run", "-n", "antideepfake", "python", "main.py", "analysis",
                   f"hparams/{yaml}", "--base_path", args.base, "--exp_name", exp,
                   "--test_csv", shard_csv, "--ckpt_path", ckpt]
            gb = MODELS[model][3]
            ram.acquire(gb)
            t0 = time.time()
            try:
                rc = run_cmd(cmd, repo_dir, env, log_path)
            finally:
                ram.release(gb)
            dt = time.time() - t0
            text = open(log_path, errors="replace").read()
            warns = [l for l in text.splitlines() if any(p in l for p in WARN_PATTERNS)]
            if warns:
                raise RuntimeError(f"[{model}] cảnh báo nạp trọng số ({len(warns)} dòng), ví dụ: {warns[:3]}")
            found = glob.glob(os.path.join(args.base, "Log", "exps", f"exp_*_{exp}", "evaluation_score.csv"))
            if rc != 0 or not found:
                raise RuntimeError(f"[{model}/{dataset}] shard {k} lỗi (mã {rc}). Cuối log:\n{tail(log_path)}")
            exp_dir = os.path.dirname(found[0])
            shutil.copy(found[0], os.path.join(sdir, f"{k:03d}_score.csv"))
            shutil.copy(os.path.join(exp_dir, "evaluation_score.pkl"), os.path.join(sdir, f"{k:03d}_emb.pkl"))
            shutil.rmtree(exp_dir)
            audio = float(shard.Duration.sum())
            json.dump({"seconds": dt, "n": len(shard), "audio_sec": audio, "gpu": gpu},
                      open(os.path.join(sdir, f"{k:03d}.done"), "w"))
            log(f"[{model}/{dataset}] shard {k + 1}/{len(shards)}: {len(shard)} câu, "
                f"{dt:.0f} s, nhanh {audio / max(dt, 1):.1f}x thời gian thực")

    # gộp
    scores = pd.concat([pd.read_csv(os.path.join(outdir, "shards", f"{k:03d}_score.csv"))
                        for k in range(len(shards))], ignore_index=True)
    assert len(scores) == len(df) and (scores.ID.values == df.ID.values).all(), "thứ tự ID lệch"
    scores.to_csv(os.path.join(outdir, "scores.csv"), index=False)
    emb = np.concatenate([pickle.load(open(os.path.join(outdir, "shards", f"{k:03d}_emb.pkl"), "rb"))
                          for k in range(len(shards))])
    # Giữ cả embedding từng shard: nếu mất phiên sau bước này, lần sau vẫn gộp lại được.
    np.save(os.path.join(outdir, "emb.npy"), emb.astype(np.float32))
    ckpts.release(model)
    return finish(args, dataset, model, outdir, df, shards_dir=os.path.join(outdir, "shards"))


def job_aasist(args, gpu, dataset, csv_path, outdir):
    onnx_path = download(AASIST_ONNX, os.path.join(args.cache, "w2v2-aasist.onnx"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "run.log")
    t0 = time.time()
    rc = run_cmd([sys.executable, os.path.join(HERE, "score_w2v2_aasist.py"), "--onnx", onnx_path,
                  "--csv", csv_path, "--root", os.path.join(args.base, "Data"),
                  "--out", os.path.join(outdir, "scores.csv")], HERE, env, log_path)
    if rc != 0:
        raise RuntimeError(f"[w2v2_aasist/{dataset}] lỗi. Cuối log:\n{tail(log_path)}")
    df = read_csv(csv_path)
    json.dump({"seconds": time.time() - t0, "n": len(df), "audio_sec": float(df.Duration.sum()),
               "gpu": gpu}, open(os.path.join(outdir, "timing.json"), "w"))
    return finish(args, dataset, "w2v2_aasist", outdir, df)


def finish(args, dataset, model, outdir, df, shards_dir=None):
    """Tính độ đo, ghi metrics.csv và một dòng tóm tắt."""
    import pandas as pd
    score_csv = os.path.join(outdir, "scores.csv")
    cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), score_csv, "--out",
           os.path.join(outdir, "metrics.csv")]
    if model == "w2v2_aasist":
        cmd += ["--score-col", "Score"]
    if dataset == "ood_forget":
        cmd += ["--prefix", "ItW", "RADAR2026"]   # evaluate.py tự thêm dấu "-"
    if dataset == "toy":
        cmd += ["--protocol", os.path.join(args.base, "AntiDeepfake", "protocols", "toy_example_test.csv"),
                "--group-col", "Attack"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(outdir, "metrics.txt"), "w").write(out.stdout + out.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"evaluate.py lỗi: {out.stderr[-2000:]}")
    m = pd.read_csv(os.path.join(outdir, "metrics.csv"))
    pooled = m.iloc[0].to_dict()
    if shards_dir:
        done = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(shards_dir, "*.done")))]
        secs = sum(d["seconds"] for d in done)
    else:
        secs = json.load(open(os.path.join(outdir, "timing.json")))["seconds"]
    audio_h = float(df.Duration.sum()) / 3600
    row = {"dataset": dataset, "model": model, "n": len(df), "audio_h": round(audio_h, 3),
           "gpu_seconds": round(secs), "x_realtime": round(audio_h * 3600 / max(secs, 1), 1)}
    row.update({k: v for k, v in pooled.items() if k not in ("subset", "group")})
    log(f"[{model}/{dataset}] KẾT QUẢ: {json.dumps(row, ensure_ascii=False)}")
    with print_lock:
        path = os.path.join(args.out, "summary.csv")
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        if len(old):
            old = old[~((old.dataset == dataset) & (old.model == model))]
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return row


def prepare_data_dir(args):
    """$ROOT của CSV = <base>/Data. Gắn mọi thư mục con của các --data-root vào đó
    (radar/, itw/ của orpheus-prep; seaspoof_vi/ của orpheus-seaspoof-vi)."""
    data = os.path.join(args.base, "Data")
    os.makedirs(data, exist_ok=True)
    for root in args.data_root:
        for name in sorted(os.listdir(root)):
            src, dst = os.path.join(root, name), os.path.join(data, name)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(src, dst)
    return data


def find_csv(args, dataset):
    for root in args.data_root:
        p = os.path.join(root, f"{dataset}.csv")
        if os.path.exists(p):
            return p
    return None


def register_local(spec):
    """--local <tên>=<đường dẫn .ckpt>,<mô hình gốc>: chấm checkpoint cục bộ bằng YAML của
    mô hình gốc (cùng kiến trúc). File .ckpt là state_dict cùng tên khoá với checkpoint
    post-trained (ví dụ ssl.ckpt do SpeechBrain lưu khi fine-tune)."""
    name, rest = spec.split("=", 1)
    path, base = rest.rsplit(",", 1)
    yaml, _, _, ram = MODELS[base]
    MODELS[name] = (yaml, None, os.path.abspath(path), ram)
    return name


def prepare_toy(args):
    data = os.path.join(args.base, "Data")
    if not os.path.isdir(os.path.join(data, "toy_example", "eval")):
        tgz = download(TOY_URL, os.path.join(args.cache, "toy_example.tar.gz"))
        subprocess.run(["tar", "-xzf", tgz, "-C", data], check=True)
    return os.path.join(args.base, "AntiDeepfake", "protocols", "toy_example_test.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="thư mục có AntiDeepfake/ và bin/micromamba (setup_antideepfake.sh)")
    ap.add_argument("--data-root", required=True, nargs="+",
                    help="một hoặc nhiều thư mục chứa <tập>.csv và âm thanh "
                         "(prep_data.py fetch, prep_seaspoof.py fetch)")
    ap.add_argument("--local", nargs="*", default=[],
                    help="checkpoint cục bộ: <tên>=<file .ckpt>,<mô hình gốc>")
    ap.add_argument("--out", required=True, help="nơi lưu kết quả (giữ lại sau phiên)")
    ap.add_argument("--cache", required=True, help="nơi để checkpoint tạm (không nằm trong --out)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--ram-gb", type=float, default=None)
    ap.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()
    args.mm = os.path.join(args.base, "bin", "micromamba")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    os.environ["MAMBA_ROOT_PREFIX"] = os.path.join(args.base, "mamba")
    prepare_data_dir(args)
    for spec in args.local:
        log(f"Checkpoint cục bộ: {register_local(spec)}")

    jobs = []
    for spec in args.jobs:
        dataset, models = spec.split(":")
        for model in models.split(","):
            if model != "w2v2_aasist" and model not in MODELS:
                sys.exit(f"Không biết mô hình {model}")
            csv_path = prepare_toy(args) if dataset == "toy" else find_csv(args, dataset)
            if not csv_path:
                sys.exit(f"Không thấy {dataset}.csv trong {args.data_root}")
            jobs.append((dataset, model, csv_path))
    refcount = {}
    for _, m, _ in jobs:
        if m != "w2v2_aasist":
            refcount[m] = refcount.get(m, 0) + 1

    ram_gb = args.ram_gb or max(8.0, 0.85 * available_ram_gb())
    log(f"{len(jobs)} job, GPU {args.gpus}, ngân sách RAM {ram_gb:.0f} GB")
    ckpts = Checkpoints(args.cache, refcount)
    ram = RamBudget(ram_gb)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    failures = []

    def worker(gpu):
        while True:
            try:
                dataset, model, csv_path = q.get_nowait()
            except queue.Empty:
                return
            outdir = os.path.join(args.out, model, dataset)
            if os.path.exists(os.path.join(outdir, "metrics.csv")):
                log(f"[{model}/{dataset}] đã có kết quả, bỏ qua")
                if model != "w2v2_aasist":
                    ckpts.release(model)
                continue
            try:
                if model == "w2v2_aasist":
                    job_aasist(args, gpu, dataset, csv_path, outdir)
                else:
                    job_antideepfake(args, ckpts, ram, gpu, dataset, model, csv_path, outdir)
            except Exception as e:  # noqa: BLE001
                log(f"[{model}/{dataset}] THẤT BẠI: {e}")
                failures.append((model, dataset, str(e)[:500]))

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
        time.sleep(5)   # lệch nhau để hai luồng không cùng tải một checkpoint ngay lúc đầu
    for t in threads:
        t.join()
    if failures:
        json.dump(failures, open(os.path.join(args.out, "failures.json"), "w"), indent=1, ensure_ascii=False)
        log(f"{len(failures)} job thất bại, xem failures.json")
        sys.exit(1)
    log("Tất cả job xong")


if __name__ == "__main__":
    main()
