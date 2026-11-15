# orpheus-finetune version 5

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune`, version 5 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-15T09:17:09+07:00 |
| Commit repo lúc đẩy | `a08bc23` |
| Code trong dataset `orpheus-src` | `git a08bc23`, staged 2026-11-15 09:14:16 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 02:17:23 |
| Kết thúc (UTC, dòng log cuối có giờ) | 02:45:23 |
| Thời gian chạy | 0.47 giờ (log ghi đến giây 1705) |
| Dựng môi trường | 4.0 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-15T09:48:06+07:00 |

## Job

```python
# GĐ6, lần thử đo GRPO (chore/GD6-GRPO.md mục 2; GĐ5 mục 9, L1): bộ nhớ và tốc độ của GRPO trên
#   T4, chưa huấn luyện thật. Mỗi lần đo chạy ngắn (optimizer_step_limit), không giữ checkpoint,
#   không chấm; main.py chạy qua mem_probe.py để ghi đỉnh bộ nhớ PyTorch thật. --oom-retry 0: hết
#   bộ nhớ thì ghi nhận, không tự giảm batch. "_long": chỉ 240 câu dài nhất của train (thử áp lực:
#   batch toàn câu ~9–16 s, tốn bộ nhớ attention nhất).
#   GPU 0: G1 (grpo_v1, đủ: clip + mô hình cũ + KL) ở batch 50 như R; nếu hết bộ nhớ thì thử lại
#          với PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (không đổi phép tính).
#   GPU 1: G1 ở batch 25 + tích luỹ gradient 2 (phương án giữ số bước: 1.536 so với 1.521 bước
#          mỗi epoch; inner_step_size 2000 batch = 1000 bước optimizer như cấu hình gốc), rồi G2
#          (grpo_v2, không clip, không mô hình cũ) ở batch 50.
#   Lịch learning rate như E1' (warmup 460, lr 1e-6); với số bước ngắn chỉ nằm trong warmup.
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test"]
PROBE = ["number_of_epochs=1", "warmup=460", "lr=1e-6", "valid_step=1000000000", "num_workers=2",
         "seed=1"]
B50 = ["max_batch_length=50"]
B25X2 = ["max_batch_length=25", "grad_accumulation_factor=2", "rl_config={inner_step_size: 2000}"]
PARGS = ["--probe-mem", "--oom-retry", "0", "--discard-ckpt"]
LONG = ["--train-longest", "240"]
EXP = ["--env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
_p = {"model": "mms_300m", "train": "seaspoof_vi_train", "valid": "seaspoof_vi_train_n50",
      "score": [], "probe": True}
RUNS = {
    0: [
        dict(_p, run="probe_G1_b50", method="grpo_v1", set=PROBE + B50 + ["optimizer_step_limit=150"],
             args=PARGS),
        dict(_p, run="probe_G1_b50_long", method="grpo_v1", set=PROBE + B50, args=PARGS + LONG),
        dict(_p, run="probe_G1_b50_long_exp", method="grpo_v1", set=PROBE + B50, args=PARGS + LONG + EXP,
             only_if_failed=["probe_G1_b50", "probe_G1_b50_long"]),
    ],
    1: [
        dict(_p, run="probe_G1_b25x2", method="grpo_v1",
             set=PROBE + B25X2 + ["optimizer_step_limit=150"], args=PARGS),
        dict(_p, run="probe_G1_b25x2_long", method="grpo_v1", set=PROBE + B25X2, args=PARGS + LONG),
        dict(_p, run="probe_G2_b50", method="grpo_v2", set=PROBE + B50 + ["optimizer_step_limit=100"],
             args=PARGS),
        dict(_p, run="probe_G2_b50_long", method="grpo_v2", set=PROBE + B50, args=PARGS + LONG),
    ],
}
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = []
```

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe_G1_b50 | ok | seaspoof_vi_train | 504 | 1521 | 3.07 | 150 | 50 | 14889.0 | 92.0 | 63.9 | 1 | 11.11 | 3.24e-07 |
| probe_G1_b50_long | error | seaspoof_vi_train | 43 | 63 | 3.28 |  | 50 | 13649.0 | 24.9 | 60.8 |  |  |  |
| probe_G1_b25x2 | ok | seaspoof_vi_train | 565 | 3072 | 1.73 | 150 | 25 | 13817.0 | 92.1 | 64.4 | 1 | 11.11 | 3.24e-07 |
| probe_G1_b50_long_exp | ok | seaspoof_vi_train | 222 | 63 | 2.84 | 63 | 50 | 14507.0 | 86.3 | 61.1 | 1 | 8.7 | 1.35e-07 |
| probe_G1_b25x2_long | ok | seaspoof_vi_train | 263 | 129 | 1.7 | 64 | 25 | 12785.0 | 87.9 | 53.7 | 1 | 8.7 | 1.37e-07 |
| probe_G2_b50 | ok | seaspoof_vi_train | 330 | 1521 | 2.95 | 100 | 50 | 14577.0 | 92.3 | 30.8 | 1 | 8.7 | 2.15e-07 |
| probe_G2_b50_long | ok | seaspoof_vi_train | 205 | 63 | 2.71 | 63 | 50 | 14833.0 | 87.6 | 29.7 | 1 | 8.7 | 1.35e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Dòng log đáng chú ý

Các dòng chứa Traceback, Error hoặc cảnh báo nạp trọng số (đã bỏ SyntaxWarning của nbconvert):

```
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G1_b25x2/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G1_b25x2/main.log | tail -2
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G1_b25x2_long/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G1_b25x2_long/main.log | tail -2
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G1_b50/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G1_b50/main.log | tail -2
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G1_b50_long/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G1_b50_long/main.log | tail -2
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G1_b50_long_exp/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G1_b50_long_exp/main.log | tail -2
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G2_b50/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G2_b50/main.log | tail -2
+ grep -h 'MEM_PROBE ' /kaggle/working/results/train/probe_G2_b50_long/main.log | tail -1; grep -h -E 'OutOfMemoryError|CUDA out of memory' /kaggle/working/results/train/probe_G2_b50_long/main.log | tail -2
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 280.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 240.81 MiB is free. Including non-PyTorch memory, this process has 14.32 GiB memory in use. Of the allocated memory 13.35 GiB is allocated by PyTorch, and 811.40 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

## File trong thư mục này

| File | Nội dung |
|---|---|
| `run.py`, `kernel-metadata.json` | Code kernel và cấu hình máy đúng như đã đẩy |
| `pushed.json` | Giờ đẩy, commit repo |
| `kernel.log.json` | Log gốc tải từ Kaggle |
| `kernel.log.txt` | Cùng log, dạng đọc được (giây tính từ lúc kernel bắt đầu) |
| `summary.csv` | Kết quả riêng của lần chạy này (bảng gộp ở `results/zeroshot/summary.csv`) |
| `setup.log`, `versions.txt`, `pip-freeze.txt` | Môi trường thực tế |
| `shard_logs/` | Log `main.py` và thời gian của từng shard (0 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

Viết 15/11 sau khi tải output.

**Mục đích:** GĐ6 lần thử đo (`chore/GD6-GRPO.md` mục 3; GĐ5 mục 9, L1): bộ nhớ và tốc độ GRPO trên T4 trước khi chạy G1–G3. 7 lần đo ngắn, không giữ checkpoint, không chấm; `main.py` gốc chạy qua `src/mem_probe.py` (ghi `torch.cuda.max_memory_allocated/reserved`); `--oom-retry 0`. Dev là `seaspoof_vi_train_n50` (100 câu). `_long` = 240 câu dài nhất của train (9,06–15,94 s).

**Trạng thái:** COMPLETE, 0,47 giờ (dựng môi trường 4,0 phút), được cấp máy ngay khi đẩy (9:17). Code `orpheus-src` bản `a08bc23`. 6/7 lần đo mã 0. `probe_G1_b50_long` **hết bộ nhớ** sau ~40 s (dự kiến được, là một kết quả: "Tried to allocate 280 MiB ... 240.81 MiB free ... 13.35 GiB allocated by PyTorch, 811.40 MiB reserved but unallocated"), nên lần có điều kiện `probe_G1_b50_long_exp` (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) đã chạy và qua. Không có lỗi hay cảnh báo deterministic với GRPO (`torch.multinomial` chạy được khi `use_deterministic_algorithms(True)`). Các dòng `+ grep ...` ở mục "Dòng log đáng chú ý" là lệnh in kết quả đo, không phải lỗi.

**Kiểm cấu hình đã giải (`train_logs/<run>/hyperparams.yaml`):** `probe_G1_b25x2` có `rl_config.inner_step_size: 2000` và giữ nguyên `algo: grpo3`, `beta: 0.04`, `epsilon: 0.2`, `sample_num: 64` (ghi đè lồng gộp đúng). `grpo2` không có `inner_step_size`, `epsilon`. `last_lr` 3,24e-07 sau 150 bước ở cả batch 50 và 25 × 2 (= 1e-6 × 149/460): lịch learning rate theo bước optimizer, không theo batch. Batch mỗi epoch 1.521 (50 s) và 3.072 (25 s), khớp mô phỏng.

**Quota:** 0,47 giờ; sau lần này đã dùng 18,88 / 30 giờ (còn 11,12 đến 21/11).

**Output thô:** `results/kaggle-raw/orpheus-finetune-v5/` (không commit). Không có checkpoint (`--discard-ckpt`).
