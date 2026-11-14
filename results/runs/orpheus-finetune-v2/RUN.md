# orpheus-finetune version 2

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-finetune`, version 2 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-14T12:50:29+07:00 |
| Commit repo lúc đẩy | `115d896` |
| Code trong dataset `orpheus-src` | `git 115d896`, staged 2026-11-14 12:49:11 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 05:50:40 |
| Kết thúc (UTC, dòng log cuối có giờ) | 10:31:11 |
| Thời gian chạy | 4.68 giờ (log ghi đến giây 16847) |
| Dựng môi trường | 3.9 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-14T17:55:08+07:00 |

## Job

```python
# GĐ5, lần 1 (chore/GD5-Ablation.md mục 2): R seed 1 và A0 seed 1, hai GPU song song.
#   Cấu hình SFT đã chốt ở GĐ4 (chore/GD4-Du-Lieu-Va-Pipeline.md mục 4.2, E1–E3): 3 epoch trên
#   toàn bộ train (17,88 giờ), batch 50 giây, lr đỉnh 1e-6, warmup 10% (460 / 4.600 bước), chỉ
#   kiểm tra dev một lần sau epoch cuối nên checkpoint lưu là mô hình cuối.
#   Sau khi cả hai GPU xong: chấm S0 (dùng thẳng, không fine-tune) trên SEA-Spoof dev/test cho
#   3 backbone sẽ fine-tune ở GĐ5 và MMS-300M bản NDA (mốc của N0).
SFT = ["max_batch_length=50", "number_of_epochs=3", "optimizer_step_limit=4600", "warmup=460",
       "lr=1e-6", "valid_step=1000000000", "num_workers=2"]
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget"]
RUNS = {
    0: [
        {"run": "R_s1", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1"], "score": SCORE},
    ],
    1: [
        {"run": "A0_s1", "model": "mms_300m", "train": "seaspoof_vi_train",
         "valid": "seaspoof_vi_dev", "set": SFT + ["seed=1", "use_da=False"], "score": SCORE},
    ],
}
# Chấm zero-shot sau khi mọi lần huấn luyện xong (score_zeroshot.py, hai GPU), vào results/zeroshot/
ZEROSHOT = ["seaspoof_vi_dev:mms_300m,w2v_large,w2v_small,mms_300m_nda",
            "seaspoof_vi_test:mms_300m,w2v_large,w2v_small,mms_300m_nda"]
```

## Huấn luyện (train_summary.csv)

| run | status | train | train_seconds | batches_per_epoch | sec_per_it | last_steps | max_batch_length | peak_mem_mib | gpu_util_mean | cpu_util_mean | n_valid | best_valid_eer | last_lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_s1 | ok | seaspoof_vi_train | 9321 | 1521 | 2.0 | 4563 | 50 | 14645.0 | 99.5 | 61.8 | 1 | 0.0 | 6.83e-07 |
| R_s1 | ok | seaspoof_vi_train | 9489 | 1521 | 2.04 | 4563 | 50 | 14645.0 | 99.5 | 61.9 | 1 | 0.0 | 6.83e-07 |

Đầy đủ (kể cả ghi đè YAML): `train_summary.csv`; log từng lần: `train_logs/`.

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| seaspoof_vi_dev | A0_s1 | 2070 | 2.205 | 216 | 36.8 | 0.0000 | 0.0000 | 0.0029 |
| seaspoof_vi_dev | R_s1 | 2070 | 2.205 | 201 | 39.4 | 0.0000 | 0.0000 | 0.0019 |
| seaspoof_vi_test | R_s1 | 11996 | 13.511 | 1771 | 27.5 | 1.6089 | 0.0428 | 0.0662 |
| seaspoof_vi_test | A0_s1 | 11996 | 13.511 | 1900 | 25.6 | 1.2017 | 0.0325 | 0.0649 |
| radar_vi | R_s1 | 13438 | 17.042 | 1753 | 35.0 | 23.0688 | 0.5100 | 0.7095 |
| radar_vi | A0_s1 | 13438 | 17.042 | 1753 | 35.0 | 18.3734 | 0.4282 | 0.6200 |
| ood_forget | A0_s1 | 4000 | 5.457 | 741 | 26.5 | 15.3000 | 0.3595 | 0.8416 |
| ood_forget | R_s1 | 4000 | 5.457 | 741 | 26.5 | 16.1500 | 0.4085 | 0.9982 |
| seaspoof_vi_dev | mms_300m | 2070 | 2.205 | 204 | 38.9 | 1.1594 | 0.0255 | 0.1952 |
| seaspoof_vi_dev | w2v_large | 2070 | 2.205 | 198 | 40.1 | 1.6425 | 0.0449 | 0.0594 |
| seaspoof_vi_dev | w2v_small | 2070 | 2.205 | 105 | 75.9 | 1.2560 | 0.0329 | 0.0847 |
| seaspoof_vi_dev | mms_300m_nda | 2070 | 2.205 | 202 | 39.3 | 1.5459 | 0.0338 | 0.0676 |
| seaspoof_vi_test | mms_300m | 11996 | 13.511 | 1116 | 43.6 | 3.4359 | 0.0821 | 0.1437 |
| seaspoof_vi_test | w2v_large | 11996 | 13.511 | 1100 | 44.2 | 1.9249 | 0.0516 | 0.0573 |
| seaspoof_vi_test | w2v_small | 11996 | 13.511 | 560 | 86.8 | 5.0759 | 0.1374 | 0.1675 |
| seaspoof_vi_test | mms_300m_nda | 11996 | 13.511 | 1089 | 44.7 | 1.1839 | 0.0307 | 0.0756 |

## Tập con SEA-Spoof (EER %, thấp hơn là tốt hơn; `subsets/`)

| Mô hình | Tập | pooled | test_seen | test_f5 | test_minimax | test_cvreal |
|---|---|---|---|---|---|---|
| A0_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| A0_s1 | seaspoof_vi_test | 1.20 | 0.04 | 0.00 | 0.00 | 2.79 |
| R_s1 | seaspoof_vi_dev | 0.00 | 0.00 |  |  |  |
| R_s1 | seaspoof_vi_test | 1.61 | 0.04 | 0.04 | 0.00 | 2.70 |
| mms_300m | seaspoof_vi_dev | 1.16 | 1.16 |  |  |  |
| mms_300m | seaspoof_vi_test | 3.44 | 0.81 | 0.57 | 0.00 | 7.44 |
| mms_300m_nda | seaspoof_vi_dev | 1.55 | 1.55 |  |  |  |
| mms_300m_nda | seaspoof_vi_test | 1.18 | 0.89 | 0.76 | 0.04 | 2.10 |
| w2v_large | seaspoof_vi_dev | 1.64 | 1.64 |  |  |  |
| w2v_large | seaspoof_vi_test | 1.92 | 0.73 | 0.72 | 0.04 | 3.04 |
| w2v_small | seaspoof_vi_dev | 1.26 | 1.26 |  |  |  |
| w2v_small | seaspoof_vi_test | 5.08 | 0.93 | 2.11 | 0.15 | 6.90 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (132 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

Viết 14/11 sau khi tải output.

**Mục đích:** GĐ5 lần 1 (`chore/GD5-Ablation.md` mục 2–4): R seed 1 (GPU 0) và A0 seed 1 (GPU 1) với cấu hình SFT đã chốt ở GĐ4 (E1–E3); sau đó chấm S0 (dùng thẳng) trên SEA-Spoof dev/test cho MMS-300M, W2V-Large, W2V-Small, MMS-300M NDA.

**Trạng thái:** COMPLETE, mọi job mã 0, không có cảnh báo nạp trọng số. Huấn luyện + chấm sau fine-tune 3,95 giờ (huấn luyện R 9.489 s, A0 9.321 s; 4.563 bước = đúng 3 epoch × 1.521 batch; 2,0 s mỗi bước; GPU 99,5%, bộ nhớ đỉnh 14,6 GB); chấm zero-shot 43,8 phút.

**Lệch cấu hình so với E1 (quan trọng):** lệnh truyền `--optimizer_step_limit 4600`, nhưng `train_logs/*/hyperparams.yaml` ghi `optimizer_step_limit: 50000` và `lr_scheduler.total_steps: 50000`. SpeechBrain coi `optimizer_step_limit` là tham số chạy (run option) nên không ghi đè vào YAML; `warmup` (460) thì có. Lịch learning rate thực tế: tăng tuyến tính lên 1e-6 trong 460 bước, rồi chỉ giảm còn **6,83e-7** ở bước 4.563 (`exp_log.txt`), thay vì ~1e-8. Huấn luyện vẫn dừng đúng ở 3 epoch (`number_of_epochs`). R và A0 dùng cùng lịch này nên so với nhau vẫn công bằng. Chi tiết và phương án: `chore/GD5-Ablation.md` phát hiện 1.

**Kiểm chứng:** S0 MMS-300M trên SEA-Spoof dev chấm lại ra 1,1594%, đúng bằng pilot GĐ4 (mô hình gần như chưa đổi), nên đường chấm nhất quán.

**Quota:** 4,68 giờ (5,42 / 30 giờ đã dùng sau lần này; còn 24,58 giờ đến 21/11).

**Output thô:** `results/kaggle-raw/orpheus-finetune-v2/` (không commit; tải lại không kèm embedding). Embedding (`emb.npy`, `*_emb.pkl`) và checkpoint `ckpt/R_s1.ckpt`, `ckpt/A0_s1.ckpt` (~1,27 GB mỗi file) chỉ ở output Kaggle `orpheus-finetune` version 2. Lượt tải đầu có cả embedding chạy ~0,5 MB/s nên đã dừng; từ nay `push.sh output finetune` bỏ embedding.
