#!/usr/bin/env python3
"""Chấm baseline W2V2-AASIST (XLS-R 300M + AASIST, Tak và cộng sự, Odyssey 2022).

Dùng bản ONNX trên Hugging Face (SpeechAntiSpoofingBenchmarks/W2V2-AASIST), xuất từ
checkpoint gốc LA_model.pth (huấn luyện trên ASVspoof 2019 LA, có RawBoost).
Cách xử lý đầu vào theo model card và data_utils_SSL.py của repo gốc:
  - 16 kHz, mono, KHÔNG chuẩn hoá biên độ;
  - lấy 64.600 mẫu đầu (~4,04 s); câu ngắn hơn thì lặp lại (tile) cho đủ;
  - điểm = logit lớp 1 (bona fide), cao là thật. Không phải hiệu hai logit.

Hệ quả: mô hình chỉ nhìn 4 giây đầu của mỗi câu. Đây là cách chấm chuẩn của
baseline này, nhưng khác AntiDeepfake (nhìn cả câu); ghi rõ khi so sánh.

Chạy tiếp được: các ID đã có trong file kết quả thì bỏ qua.

  python src/score_w2v2_aasist.py --onnx w2v2-aasist.onnx --csv radar_vi.csv \
      --root <thư mục thay cho $ROOT> --out scores.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

MAX_LEN = 64600


def pad_fixed(x, max_len=MAX_LEN):
    """Như hàm pad() trong data_utils_SSL.py của SSL_Anti-spoofing."""
    if x.shape[0] >= max_len:
        return x[:max_len]
    reps = int(max_len / x.shape[0]) + 1
    return np.tile(x, reps)[:max_len]


def load(path, sr_expected=16000):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != sr_expected:
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, sr_expected).numpy()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu"], check=True)
        import onnxruntime as ort

    # Nạp thư viện CUDA/cuDNN đi kèm torch trước, để onnxruntime-gpu tìm thấy
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            print("preload_dlls:", e, flush=True)
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    print("ONNX providers:", sess.get_providers(), flush=True)
    if "CUDAExecutionProvider" not in sess.get_providers():
        print("CẢNH BÁO: không dùng được GPU, chạy trên CPU (chậm)", flush=True)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    df = pd.read_csv(args.csv)
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out).ID)
    todo = df[~df.ID.isin(done)]
    print(f"{len(df)} câu, đã có {len(done)}, còn {len(todo)}", flush=True)

    header = not os.path.exists(args.out)
    t0 = time.time()
    rows = list(todo.itertuples())
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        x = np.stack([pad_fixed(load(r.Path.replace("$ROOT", args.root))) for r in chunk])
        logits = sess.run([out_name], {in_name: x.astype(np.float32)})[0]
        out = pd.DataFrame({
            "ID": [r.ID for r in chunk],
            "Label": [1 if r.Label == "real" else 0 for r in chunk],
            "Score": logits[:, 1],
            "LogitFake": logits[:, 0],
        })
        out.to_csv(args.out, mode="a", header=header, index=False)
        header = False
        if (i // args.batch) % 100 == 0:
            print(f"  {i + len(chunk)}/{len(rows)} câu, {time.time() - t0:.0f} s", flush=True)

    res = pd.read_csv(args.out)
    order = {k: n for n, k in enumerate(df.ID)}
    res = res.drop_duplicates("ID").sort_values("ID", key=lambda s: s.map(order))
    assert len(res) == len(df), f"thiếu kết quả: {len(res)} / {len(df)}"
    res.to_csv(args.out, index=False)
    print(f"Xong {len(res)} câu sau {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
