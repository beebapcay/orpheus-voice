# orpheus-finetune version 6

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune`, version 6 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-15T09:58:27+07:00 |
| Commit repo lúc đẩy | `444d390` |
| Code trong dataset `orpheus-src` | `git a08bc23`, staged 2026-11-15 09:14:16 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 02:58:38 |
| Kết thúc (UTC, dòng log cuối có giờ) | 08:30:13 |
| Thời gian chạy | 5.53 giờ (log ghi đến giây 19909) |
| Dựng môi trường | 3.8 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-15T15:35:26+07:00 |

## Job

?

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G2_s1 | ok | seaspoof_vi_train | 12147 | 1521 | 2.62 | 4563 | 50 | 13211.0 | 99.5 | 65.2 | 1 | 0.0 | 6.83e-07 |
| G1_s1 | ok | seaspoof_vi_train | 15414 | 1521 | 3.34 | 4563 | 50 | 14453.0 | 99.5 | 62.9 | 1 | 0.0 | 6.83e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | G2_s1 | 2070 | 2.205 | 257 | 30.8 | 0.0000 | 0.0000 | 0.0018 |
| seaspoof_vi_test | G2_s1 | 11996 | 13.511 | 1454 | 33.5 | 1.8759 | 0.0498 | 0.1391 |
| seaspoof_vi_dev | G1_s1 | 2070 | 2.205 | 199 | 40.0 | 0.0000 | 0.0000 | 0.0000 |
| radar_vi | G2_s1 | 13438 | 17.042 | 1742 | 35.2 | 23.1284 | 0.4936 | 0.8866 |
| ood_forget | G2_s1 | 4000 | 5.457 | 536 | 36.6 | 16.0000 | 0.3497 | 1.1081 |
| seaspoof_vi_test | G1_s1 | 11996 | 13.511 | 1105 | 44.0 | 2.1919 | 0.0578 | 0.1462 |
| mlaad_vi_test | G2_s1 | 7475 | 14.093 | 1212 | 41.9 | 0.1708 | 0.0043 | 0.0570 |
| radar_vi | G1_s1 | 13438 | 17.042 | 1435 | 42.8 | 23.5304 | 0.5115 | 0.8830 |
| ood_forget | G1_s1 | 4000 | 5.457 | 429 | 45.8 | 16.7000 | 0.3560 | 1.0929 |
| mlaad_vi_test | G1_s1 | 7475 | 14.093 | 1046 | 48.5 | 0.0502 | 0.0014 | 0.0182 |

## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)

| Mô hình | Tập | pooled | test_seen | test_f5 | test_minimax | test_cvreal |
|---|---|---|---|---|---|---|
| G1_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| G1_s1 | seaspoof_vi_test | 2.19 | 0.04 | 0.04 | 0.00 | 4.16 |
| G2_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| G2_s1 | seaspoof_vi_test | 1.88 | 0.04 | 0.04 | 0.00 | 4.29 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (84 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

Viết 15/11 sau khi tải output.

**Mục đích:** GĐ6 lần 1 (`chore/GD6-GRPO.md` mục 6, 10.2): G1 seed 1 (GRPO đủ, `grpo_v1`, GPU 0) và G2 seed 1 (GRPO không clip, không mô hình cũ, `grpo_v2`, GPU 1), cùng bộ ghi đè SFT (lịch E1') như R; thêm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--oom-retry 0`, `--probe-mem` (mục 3). `run.py` dùng `JOB = "grpo1"`. Chạy song song với lần 2 (`orpheus-finetune-b` v2).

**Trạng thái:** COMPLETE, mọi job mã 0, 0 cảnh báo nạp trọng số, không NaN, không hết bộ nhớ. Đẩy 9:58, được cấp máy ngay, chạy 5,53 giờ (xong 15:31). Code `orpheus-src` bản `a08bc23`. G1: 3,34 s/bước, đỉnh cấp phát 13.925 MiB; G2: 2,62 s/bước, 12.682 MiB. Mỗi lần 4.563 bước, learning rate cuối 6,83e-07.

**Kết quả chính:** RADAR VI G1 23,53%, G2 23,13% (R 5 seed 24,20 ± 1,88); ItW 5,9 / 4,3%; trong khoảng của R (mục 10.2).

**Quota:** ~5,53 giờ; lúc lưu: đã dùng 29,07 / 30 giờ (còn 0,93 đến 21/11).

**Output thô:** `results/kaggle-raw/orpheus-finetune-v6/` (không commit). Checkpoint `ckpt/G1_s1.ckpt`, `ckpt/G2_s1.ckpt` chỉ ở output Kaggle `orpheus-finetune` version 6.
