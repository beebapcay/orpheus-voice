# orpheus-finetune version 3

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune`, version 3 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-14T18:05:07+07:00 |
| Commit repo lúc đẩy | `6d3946e` |
| Code trong dataset `orpheus-src` | `git 6d3946e`, staged 2026-11-14 18:03:25 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 14:34:22 |
| Kết thúc (UTC, dòng log cuối có giờ) | 18:42:26 |
| Thời gian chạy | 4.14 giờ (log ghi đến giây 14902) |
| Dựng môi trường | 4.1 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-15T01:51:55+07:00 |

## Job

```python
# GĐ5, lần 2 (chore/GD5-Ablation.md mục 2): P0 seed 1 (GPU 0) và B2 seed 1 (GPU 1).
#   P0: MMS-300M khởi tạo từ trọng số fairseq gốc (chưa post-train), lớp FC ngẫu nhiên -> post-training
#       đóng góp bao nhiêu (so với R). B2: W2V-Large post-trained -> ngôn ngữ pre-training (so với R).
#   Bộ ghi đè SFT giữ NGUYÊN như lần 1 (quyết định E1', mục 4.1): `optimizer_step_limit` không vào
#   được YAML nên lịch learning rate thực tế là warmup 460 bước lên 1e-6 rồi giảm rất chậm
#   (total_steps 50000); đổi bộ ghi đè là đổi lịch và không so được với R_s1, A0_s1.
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget"]
RUNS = {
    0: [
        {"run": "P0_s1", "model": "mms_300m", "init": "fairseq", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1"], "score": SCORE},
    ],
    1: [
        {"run": "B2_s1", "model": "w2v_large", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1"], "score": SCORE},
    ],
}
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = []
```

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2_s1 | ok | seaspoof_vi_train | 9367 | 1521 | 2.0 | 4563 | 50 | 14645.0 | 99.1 | 73.1 | 1 | 0.0967 | 6.83e-07 |
| P0_s1 | ok | seaspoof_vi_train | 9486 | 1521 | 2.04 | 4563 | 50 | 14645.0 | 99.5 | 73.1 | 1 | 0.0 | 6.83e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | B2_s1 | 2070 | 2.205 | 220 | 36.0 | 0.0966 | 0.0018 | 0.0057 |
| seaspoof_vi_dev | P0_s1 | 2070 | 2.205 | 224 | 35.4 | 0.0000 | 0.0000 | 0.0000 |
| seaspoof_vi_test | P0_s1 | 11996 | 13.511 | 1847 | 26.3 | 21.3895 | 0.2777 | 0.5295 |
| seaspoof_vi_test | B2_s1 | 11996 | 13.511 | 1941 | 25.1 | 1.5177 | 0.0389 | 0.0879 |
| radar_vi | B2_s1 | 13438 | 17.042 | 2215 | 27.7 | 21.0300 | 0.4925 | 0.6325 |
| radar_vi | P0_s1 | 13438 | 17.042 | 2215 | 27.7 | 31.4555 | 0.6529 | 1.1886 |
| ood_forget | P0_s1 | 4000 | 5.457 | 732 | 26.8 | 24.1500 | 0.6001 | 1.8087 |
| ood_forget | B2_s1 | 4000 | 5.457 | 732 | 26.8 | 13.0500 | 0.2890 | 0.3452 |

## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)

| Mô hình | Tập | pooled | test_seen | test_f5 | test_minimax | test_cvreal |
|---|---|---|---|---|---|---|
| B2_s1 | seaspoof_vi_dev | 0.10 | 0.10 |  |  |  |
| B2_s1 | seaspoof_vi_test | 1.52 | 0.04 | 0.00 | 0.00 | 2.59 |
| P0_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| P0_s1 | seaspoof_vi_test | 21.39 | 0.04 | 5.94 | 0.00 | 3.26 |

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

**Mục đích:** GĐ5 lần 2 (`chore/GD5-Ablation.md` mục 3.2): P0 seed 1 (MMS-300M khởi tạo từ trọng số fairseq gốc `base_300m.pt`, chưa post-train) và B2 seed 1 (W2V-Large post-trained), cùng bộ ghi đè SFT như lần 1 (lịch learning rate E1', lr cuối 6,83e-7 như lần 1).

**Trạng thái:** COMPLETE, mọi job mã 0, không có cảnh báo nạp trọng số. Xếp hàng ~3 giờ 28 phút (đẩy 18:05, chạy từ ~21:33 giờ VN). Chạy 4,14 giờ; huấn luyện P0 9.486 s, B2 9.367 s; 4.563 bước; bộ nhớ đỉnh 14,6 GB.

**Kiểm chứng P0:** bước kiểm nạp trọng số trước khi huấn luyện (`train_logs/P0_s1/preflight.log`): 0 cảnh báo, 431 tensor, 317.392.642 tham số; `main.log` ghi `Loading pre-trained weights detector from /tmp/orpheus/cache/fairseq_base_300m.pt`, `hyperparams.yaml` có `ckpt_path` đúng file đó. Lớp FC phân loại khởi tạo ngẫu nhiên (seed 1). EER dev 0,00% sau 3 epoch: P0 học được trong miền.

**Quota:** ~4,14 giờ (chạy song song với `orpheus-finetune-b` v1 từ ~22:10).

**Output thô:** `results/kaggle-raw/orpheus-finetune-v3/` (không commit, không có embedding). Checkpoint `ckpt/P0_s1.ckpt`, `ckpt/B2_s1.ckpt` chỉ ở output Kaggle `orpheus-finetune` version 3.
