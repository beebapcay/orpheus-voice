# orpheus-finetune-b version 1

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune-b`, version 1 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-14T18:30:02+07:00 |
| Commit repo lúc đẩy | `8c227f9` |
| Code trong dataset `orpheus-src` | `git 6d3946e`, staged 2026-11-14 18:03:25 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 15:09:48 |
| Kết thúc (UTC, dòng log cuối có giờ) | 19:02:58 |
| Thời gian chạy | 3.89 giờ (log ghi đến giây 14013) |
| Dựng môi trường | 4.0 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-15T02:12:12+07:00 |

## Job

```python
# GĐ5, lần 3 (chore/GD5-Ablation.md mục 2), kernel orpheus-finetune-b (chạy song song với lần 2):
#   B1 seed 1 (GPU 0): W2V-Small post-trained (95M, tiếng Anh) -> kích thước backbone (so với B2).
#   N0 seed 1 (GPU 1): MMS-300M bản NDA (post-train không RawBoost) -> RawBoost khi post-train (so với R).
#   Bộ ghi đè SFT giữ NGUYÊN như lần 1 (quyết định E1', mục 4.1): `optimizer_step_limit` không vào
#   được YAML nên lịch learning rate thực tế là warmup 460 bước lên 1e-6 rồi giảm rất chậm
#   (total_steps 50000); đổi bộ ghi đè là đổi lịch và không so được với R_s1, A0_s1.
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget"]
RUNS = {
    0: [
        {"run": "B1_s1", "model": "w2v_small", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1"], "score": SCORE},
    ],
    1: [
        {"run": "N0_s1", "model": "mms_300m_nda", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1"], "score": SCORE},
    ],
}
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = []
```

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_s1 | ok | seaspoof_vi_train | 5097 | 1521 | 1.46 | 4563 | 50 | 14609.0 | 75.8 | 70.4 | 1 | 0.194 | 6.83e-07 |
| N0_s1 | ok | seaspoof_vi_train | 10008 | 1521 | 2.15 | 4563 | 50 | 14645.0 | 99.6 | 56.5 | 1 | 0.0967 | 6.83e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | B1_s1 | 2070 | 2.205 | 104 | 76.2 | 0.0000 | 0.0000 | 0.0000 |
| seaspoof_vi_test | B1_s1 | 11996 | 13.511 | 3040 | 16.0 | 13.8124 | 0.3711 | 0.6604 |
| seaspoof_vi_dev | N0_s1 | 2070 | 2.205 | 201 | 39.6 | 0.0000 | 0.0000 | 0.0000 |
| radar_vi | B1_s1 | 13438 | 17.042 | 2450 | 25.0 | 34.6703 | 0.7333 | 1.3495 |
| ood_forget | B1_s1 | 4000 | 5.457 | 700 | 28.1 | 17.5500 | 0.4078 | 0.8871 |
| seaspoof_vi_test | N0_s1 | 11996 | 13.511 | 1666 | 29.2 | 2.1007 | 0.0588 | 0.1002 |
| radar_vi | N0_s1 | 13438 | 17.042 | 1371 | 44.7 | 20.5016 | 0.4941 | 0.7021 |
| ood_forget | N0_s1 | 4000 | 5.457 | 441 | 44.5 | 16.0000 | 0.3549 | 0.9758 |

## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)

| Mô hình | Tập | pooled | test_seen | test_f5 | test_minimax | test_cvreal |
|---|---|---|---|---|---|---|
| B1_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| B1_s1 | seaspoof_vi_test | 13.81 | 0.00 | 0.48 | 0.00 | 8.96 |
| N0_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| N0_s1 | seaspoof_vi_test | 2.10 | 0.00 | 0.00 | 0.00 | 3.60 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (68 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

Viết 15/11 sau khi tải output.

**Mục đích:** GĐ5 lần 3 (`chore/GD5-Ablation.md` mục 3.3), kernel thứ hai `orpheus-finetune-b`, chạy song song với `orpheus-finetune` v3: B1 seed 1 (W2V-Small post-trained, 95M) và N0 seed 1 (MMS-300M bản NDA, post-train không RawBoost). Cùng bộ ghi đè SFT với lần 1 (lr cuối 6,83e-7).

**Trạng thái:** COMPLETE, mọi job mã 0. Kiểm nạp trọng số: B1 0 cảnh báo (220 tensor, 95.046.146 tham số), N0 0 cảnh báo (431 tensor). Xếp hàng ~3 giờ 40 phút (đẩy 18:30, chạy từ 22:09 giờ VN); chạy 3,89 giờ. B1 1,46 s mỗi bước nhưng GPU chỉ dùng 75,8% (CPU 70%): với mô hình 95M, đọc dữ liệu và RawBoost trên 4 nhân CPU bắt đầu là nút thắt. N0 2,15 s mỗi bước.

**Quota:** ~3,89 giờ.

**Output thô:** `results/kaggle-raw/orpheus-finetune-b-v1/` (không commit, không có embedding). Checkpoint `ckpt/B1_s1.ckpt`, `ckpt/N0_s1.ckpt` chỉ ở output Kaggle `orpheus-finetune-b` version 1.
