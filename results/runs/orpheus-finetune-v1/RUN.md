# orpheus-finetune version 1

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune`, version 1 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-14T10:19:12+07:00 |
| Commit repo lúc đẩy | `6edd4cc` |
| Code trong dataset `orpheus-src` | `git 6edd4cc`, staged 2026-11-14 10:10:38 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 03:19:26 |
| Kết thúc (UTC, dòng log cuối có giờ) | 04:03:47 |
| Thời gian chạy | 0.74 giờ (log ghi đến giây 2678) |
| Dựng môi trường | 4.1 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-14T12:09:25+07:00 |

## Job

```python
PILOT = ["number_of_epochs=1", "optimizer_step_limit=300", "valid_step=100000", "num_workers=2"]
RUNS = {
    0: [
        {"run": "pilot_R", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": PILOT, "score": ["seaspoof_vi_dev"]},
    ],
    1: [
        {"run": "overfit_n50", "model": "mms_300m", "train": "seaspoof_vi_train_n50",
         "valid": "seaspoof_vi_train_n50",
         "set": ["number_of_epochs=30", "valid_step=25", "use_da=False", "lr=1e-5", "warmup=1",
                 "num_workers=2"], "score": []},
        {"run": "pilot_A0", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": PILOT + ["use_da=False"],
         "score": ["seaspoof_vi_dev", "ood_forget"]},
    ],
}
```

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overfit_n50 | ok | seaspoof_vi_train_n50 | 721 | 12 | 1.7 | 360 | 50.0 | 14203.0 | 86.4 | 65.4 | 15 | 0.0 | 9.68e-06 |
| pilot_R | ok | seaspoof_vi_train | 791 | 1521 | 1.93 | 300 | 50.0 | 14645.0 | 92.8 | 64.3 | 1 | 3.81 | 3.74e-09 |
| pilot_A0 | ok | seaspoof_vi_train | 736 | 1521 | 1.92 | 300 | 50.0 | 14645.0 | 96.7 | 35.9 | 1 | 3.29 | 3.74e-09 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | pilot_R | 2070 | 2.205 | 209 | 38.0 | 1.1594 | 0.0267 | 0.1420 |
| ood_forget | pilot_A0 | 4000 | 5.457 | 666 | 29.5 | 15.2500 | 0.3244 | 0.3327 |

## Dòng log đáng chú ý

Các dòng chứa Traceback, Error hoặc cảnh báo nạp trọng số (đã bỏ SyntaxWarning của nbconvert):

Không có.

## File trong thư mục này

| File | Nội dung |
|---|---|
| `run.py`, `kernel-metadata.json` | Code kernel và cấu hình máy đúng như đã đẩy |
| `pushed.json` | Giờ đẩy, commit repo |
| `kernel.log.json` | Log gốc tải từ Kaggle |
| `kernel.log.txt` | Cùng log, dạng đọc được (giây tính từ lúc kernel bắt đầu) |
| `summary.csv` | Kết quả riêng của lần chạy này (bảng gộp ở `results/zeroshot/summary.csv`) |
| `setup.log`, `versions.txt`, `pip-freeze.txt` | Môi trường thực tế |
| `shard_logs/` | Log `main.py` và thời gian của từng shard (11 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

Viết 14/11 sau khi tải output.

**Mục đích:** pilot đo đạc của GĐ4 ngày 2 (không phải thí nghiệm GĐ5): bộ nhớ, tốc độ, CPU, pipeline có học được không, đường chấm checkpoint cục bộ. Chi tiết và cách đọc: `chore/GD4-Du-Lieu-Va-Pipeline.md` mục 4.

**Trạng thái Kaggle là ERROR** chỉ vì một bước chấm: `pilot_A0` trên `seaspoof_vi_dev`, shard 2 (70 câu cuối của CSV xếp theo ID) chỉ có một lớp; `main.py analysis` ghi xong điểm rồi lỗi `ValueError` ở `compute_metrics` (confusion matrix 1×1). `pilot_R` chấm cùng dev không lỗi, có lẽ vì dự đoán của nó có đủ hai lớp. Đã sửa ở `src/score_zeroshot.py`: xáo trộn cố định trước khi chia shard và kiểm mỗi shard đủ hai lớp. Ba lần huấn luyện đều xong (mã 0).

**Kết quả chính:**
- `max_batch_length=100` hết bộ nhớ T4 ngay bước 2 (`main.oom0.log`); `finetune.py` tự giảm xuống 50 và chạy được; bộ nhớ đỉnh 14,6 GB / 15 GB (sát).
- Toàn bộ train (17,88 giờ) = 1.521 batch mỗi epoch ở 50 s; 1,92–1,93 s mỗi bước khi hai GPU cùng chạy -> ~49 phút mỗi epoch. GPU dùng 93–97%, CPU 64% (có RawBoost) và 36% (không RawBoost): nút thắt là GPU, không phải đọc dữ liệu.
- Overfit 100 câu (lr 1e-5): loss dev (= train) giảm 0,748 -> 0,013; EER = 0 từ bước 25 (tập này vốn dễ với mô hình post-trained).
- Không có cảnh báo nạp trọng số; cả ba lần nạp `mms_300m.ckpt` từ HF.
- Đường chấm checkpoint cục bộ (`--local`, hai `--data-root`) chạy đúng: `pilot_A0` trên `ood_forget` ra In-the-Wild 2,6% / RADAR EN 25,9%, gần mốc S0 của GĐ3 (2,6 / 26,2), đúng như kỳ vọng vì learning rate pilot chỉ ~4e-9.
- EER dev: chấm lại `pilot_R` (batch 1, dev sạch) 1,16%; lúc huấn luyện 3,81% (`pilot_R`, RawBoost trên dev, batch 10 có đệm) và 3,29% (`pilot_A0`, dev sạch, batch 10 có đệm). Hai mô hình gần như chưa đổi, nên chênh 1,16 so với 3,29 là do đệm số 0 khi kiểm tra theo batch 10 (mô hình không có mặt nạ cho phần đệm).

**Quota:** 0,74 giờ (kernel 0,74 giờ; dựng môi trường 4,1 phút). Còn 29,26 / 30 giờ đến 21/11.

**Output thô:** `results/kaggle-raw/orpheus-finetune-v1/` (không commit). Checkpoint `ckpt/overfit_n50.ckpt`, `ckpt/pilot_R.ckpt`, `ckpt/pilot_A0.ckpt` (mỗi file 1.270 MB) chỉ ở output Kaggle `orpheus-finetune` version 1; không dùng tiếp (pilot).
