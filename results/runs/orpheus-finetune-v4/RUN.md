# orpheus-finetune version 4

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune`, version 4 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-15T01:47:57+07:00 |
| Commit repo lúc đẩy | `73c316a` |
| Code trong dataset `orpheus-src` | `git 73c316a`, staged 2026-11-15 01:46:30 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 20:32:11 |
| Kết thúc (UTC, dòng log cuối có giờ) | 01:28:36 |
| Thời gian chạy | 4.94 giờ (log ghi đến giây 17807) |
| Dựng môi trường | 4.0 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-15T08:36:13+07:00 |

## Job

```python
# GĐ5, lần 4 (chore/GD5-Ablation.md mục 2): R seed 2 (GPU 0) và R seed 3 (GPU 1), cho độ lệch giữa
#   các seed của cấu hình mốc. Chấm thêm tập MLAAD fake/vi (mục 4.2), rồi chấm S0 trên MLAAD.
#   Bộ ghi đè SFT giữ NGUYÊN như lần 1 (quyết định E1', mục 4.1): `optimizer_step_limit` không vào
#   được YAML nên lịch learning rate thực tế là warmup 460 bước lên 1e-6 rồi giảm rất chậm
#   (total_steps 50000); đổi bộ ghi đè là đổi lịch và không so được với R_s1, A0_s1.
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test"]
RUNS = {
    0: [
        {"run": "R_s2", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=2"], "score": SCORE},
    ],
    1: [
        {"run": "R_s3", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=3"], "score": SCORE},
    ],
}
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = ["mlaad_vi_test:mms_300m,w2v_large,w2v_small,mms_300m_nda"]
```

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R_s3 | ok | seaspoof_vi_train | 8959 | 1521 | 1.93 | 4563 | 50 | 14645.0 | 99.5 | 71.3 | 1 | 0.0 | 6.83e-07 |
| R_s2 | ok | seaspoof_vi_train | 9584 | 1521 | 2.06 | 4563 | 50 | 14645.0 | 99.5 | 70.6 | 1 | 0.0 | 6.83e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | R_s3 | 2070 | 2.205 | 246 | 32.3 | 0.0000 | 0.0000 | 0.0048 |
| seaspoof_vi_dev | R_s2 | 2070 | 2.205 | 201 | 39.5 | 0.0000 | 0.0000 | 0.0048 |
| seaspoof_vi_test | R_s3 | 11996 | 13.511 | 1388 | 35.0 | 3.3090 | 0.0864 | 0.1143 |
| seaspoof_vi_test | R_s2 | 11996 | 13.511 | 1112 | 43.7 | 2.3922 | 0.0595 | 0.1125 |
| radar_vi | R_s3 | 13438 | 17.042 | 1636 | 37.5 | 25.7702 | 0.5834 | 0.8353 |
| radar_vi | R_s2 | 13438 | 17.042 | 1360 | 45.1 | 23.2250 | 0.5255 | 0.7742 |
| ood_forget | R_s3 | 4000 | 5.457 | 592 | 33.2 | 16.6000 | 0.3901 | 0.8895 |
| ood_forget | R_s2 | 4000 | 5.457 | 586 | 33.5 | 16.3000 | 0.3992 | 0.8727 |
| mlaad_vi_test | R_s2 | 7475 | 14.093 | 2501 | 20.3 | 0.0904 | 0.0025 | 0.0638 |
| mlaad_vi_test | R_s3 | 7475 | 14.093 | 2501 | 20.3 | 0.1608 | 0.0033 | 0.0704 |
| mlaad_vi_test | w2v_large | 7475 | 14.093 | 1024 | 49.5 | 2.2211 | 0.0469 | 0.0794 |
| mlaad_vi_test | mms_300m | 7475 | 14.093 | 1076 | 47.1 | 5.1759 | 0.1179 | 0.2016 |
| mlaad_vi_test | w2v_small | 7475 | 14.093 | 518 | 98.0 | 2.2311 | 0.0439 | 0.0870 |
| mlaad_vi_test | mms_300m_nda | 7475 | 14.093 | 1065 | 47.6 | 4.6532 | 0.0985 | 0.1684 |

## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)

| Mô hình | Tập | pooled | test_seen | test_f5 | test_minimax | test_cvreal |
|---|---|---|---|---|---|---|
| R_s2 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| R_s2 | seaspoof_vi_test | 2.39 | 0.04 | 0.04 | 0.00 | 4.56 |
| R_s3 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| R_s3 | seaspoof_vi_test | 3.31 | 0.04 | 0.04 | 0.00 | 6.15 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (116 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

Viết 15/11 sau khi tải output.

**Mục đích:** GĐ5 lần 4 (`chore/GD5-Ablation.md` mục 3.4): R seed 2 (GPU 0) và R seed 3 (GPU 1), cùng cấu hình với R seed 1 (lịch learning rate E1'); chấm thêm tập MLAAD `fake/vi` (`mlaad_vi_test`, dataset riêng tư `orpheus-mlaad-vi`); sau đó chấm S0 (MMS-300M, MMS-300M NDA, W2V-Large, W2V-Small) trên MLAAD.

**Trạng thái:** COMPLETE, mọi job mã 0, 0 cảnh báo nạp trọng số. Xếp hàng ~1 giờ 40 phút (đẩy ~1:50, chạy từ ~3:32 giờ VN), chạy 4,94 giờ. Code `orpheus-src` bản `73c316a` (có phần chấm MLAAD theo hệ thống). Phiên OAuth của Kaggle CLI hết hạn giữa chừng (~4:30), người dùng đăng nhập lại 5:00; kernel không bị ảnh hưởng.

**Quota:** ~4,94 giờ; sau lần này đã dùng 18,40 / 30 giờ (còn 11,60 đến 21/11).

**Output thô:** `results/kaggle-raw/orpheus-finetune-v4/` (không commit, không có embedding). Checkpoint `ckpt/R_s2.ckpt`, `ckpt/R_s3.ckpt` chỉ ở output Kaggle `orpheus-finetune` version 4.
