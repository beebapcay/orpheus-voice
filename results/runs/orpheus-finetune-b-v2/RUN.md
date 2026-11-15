# orpheus-finetune-b version 2

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune-b`, version 2 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-15T09:58:41+07:00 |
| Commit repo lúc đẩy | `3be49c2` |
| Code trong dataset `orpheus-src` | `git a08bc23`, staged 2026-11-15 09:14:16 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 02:58:51 |
| Kết thúc (UTC, dòng log cuối có giờ) | 07:38:11 |
| Thời gian chạy | 4.66 giờ (log ghi đến giây 16774) |
| Dựng môi trường | 4.0 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-15T14:43:07+07:00 |

## Job

?

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R_s5 | ok | seaspoof_vi_train | 9665 | 1521 | 2.07 | 4563 | 50 | 14645.0 | 99.4 | 69.8 | 1 | 0.0 | 6.83e-07 |
| R_s4 | ok | seaspoof_vi_train | 10477 | 1521 | 2.25 | 4563 | 50 | 14645.0 | 99.5 | 68.6 | 1 | 0.0 | 6.83e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | R_s5 | 2070 | 2.205 | 279 | 28.5 | 0.0966 | 0.0010 | 0.0028 |
| seaspoof_vi_dev | R_s4 | 2070 | 2.205 | 215 | 36.9 | 0.0000 | 0.0000 | 0.0058 |
| seaspoof_vi_test | R_s5 | 11996 | 13.511 | 1900 | 25.6 | 1.8581 | 0.0496 | 0.1315 |
| seaspoof_vi_test | R_s4 | 11996 | 13.511 | 1221 | 39.8 | 4.5596 | 0.1193 | 0.1225 |
| radar_vi | R_s4 | 13438 | 17.042 | 2238 | 27.4 | 26.6259 | 0.6253 | 0.7151 |
| radar_vi | R_s5 | 13438 | 17.042 | 2283 | 26.9 | 22.3174 | 0.4995 | 0.9012 |
| ood_forget | R_s4 | 4000 | 5.457 | 768 | 25.6 | 18.5500 | 0.4733 | 0.9533 |
| ood_forget | R_s5 | 4000 | 5.457 | 769 | 25.6 | 17.4000 | 0.4660 | 1.1484 |
| mlaad_vi_test | R_s5 | 7475 | 14.093 | 1569 | 32.3 | 0.2110 | 0.0049 | 0.0692 |
| mlaad_vi_test | R_s4 | 7475 | 14.093 | 1569 | 32.3 | 0.2814 | 0.0053 | 0.1268 |

## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)

| Mô hình | Tập | pooled | test_seen | test_f5 | test_minimax | test_cvreal |
|---|---|---|---|---|---|---|
| R_s4 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| R_s4 | seaspoof_vi_test | 4.56 | 0.04 | 0.07 | 0.00 | 6.10 |
| R_s5 | seaspoof_vi_dev | 0.10 | 0.10 |  |  |  |
| R_s5 | seaspoof_vi_test | 1.86 | 0.04 | 0.04 | 0.00 | 4.09 |

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

**Mục đích:** GĐ6 lần 2 (`chore/GD6-GRPO.md` mục 6, 10.1): R seed 4 (GPU 0), R seed 5 (GPU 1), cùng cấu hình với R seed 1–3 (bộ ghi đè SFT, lịch E1'; chỉ khác seed), để R đủ 5 seed (P1: đủ độ tin cậy thống kê khi so với G1). Chạy song song với lần 1 (`orpheus-finetune` v6: G1 s1, G2 s1). `run.py` dùng `JOB = "r45"`.

**Trạng thái:** COMPLETE, mọi job mã 0, 0 cảnh báo nạp trọng số. Đẩy 9:58, được cấp máy ngay, chạy 4,66 giờ (xong 14:39). Code `orpheus-src` bản `a08bc23`. Mỗi lần 4.563 bước, learning rate cuối 6,83e-07.

**Kết quả chính:** RADAR VI 26,63% (s4), 22,32% (s5); R qua 5 seed 24,20 ± 1,88 (mục 10.1).

**Quota:** ~4,66 giờ; lúc lưu: đã dùng 28,22 / 30 giờ (còn 1,78 đến 21/11, lần 1 vẫn đang chạy).

**Output thô:** `results/kaggle-raw/orpheus-finetune-b-v2/` (không commit). Checkpoint `ckpt/R_s4.ckpt`, `ckpt/R_s5.ckpt` chỉ ở output Kaggle `orpheus-finetune-b` version 2.
