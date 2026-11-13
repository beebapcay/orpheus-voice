# orpheus-zeroshot version 1

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-zeroshot`, version 1 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-13 khoảng 17:56 +07:00 |
| Commit repo lúc đẩy | `9b9fc04` |
| Code trong dataset `orpheus-src` | `git 968f43e`, staged 2026-11-13 17:14:30 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 10:57:15 |
| Kết thúc (UTC, dòng log cuối có giờ) | 13:22:39 |
| Thời gian chạy | 2.42 giờ (log ghi đến giây 8738) |
| Dựng môi trường | 4.0 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-13T22:50:12+07:00 |

## Job

- `toy:mms_300m`
- `radar_vi:mms_300m,w2v_small,w2v_large,w2v2_aasist`
- `ood_forget:mms_300m,w2v_small,w2v_large,w2v2_aasist`
- `itw_full:mms_300m,w2v2_aasist`

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| toy | mms_300m | 150 | 0.124 | 26 | 17.3 | 6.1101 | 0.0597 | 0.1710 |
| radar_vi | w2v_small | 13438 | 17.042 | 1985 | 30.9 | 27.2139 | 0.5722 | 0.5865 |
| radar_vi | mms_300m | 13438 | 17.042 | 2016 | 30.4 | 20.0848 | 0.4613 | 0.4688 |
| radar_vi | w2v2_aasist | 13438 | 17.042 | 1263 | 48.6 | 30.1162 | 0.8018 | 1.2455 |
| radar_vi | w2v_large | 13438 | 17.042 | 1428 | 42.9 | 17.8003 | 0.3843 | 0.3889 |
| ood_forget | mms_300m | 4000 | 5.457 | 674 | 29.2 | 15.5000 | 0.3315 | 0.3391 |
| ood_forget | w2v_small | 4000 | 5.457 | 505 | 38.9 | 16.2000 | 0.3763 | 0.3951 |
| ood_forget | w2v2_aasist | 4000 | 5.457 | 260 | 75.7 | 27.1000 | 0.6870 | 1.1280 |
| ood_forget | w2v_large | 4000 | 5.457 | 437 | 44.9 | 15.3000 | 0.3098 | 0.3123 |
| itw_full | w2v2_aasist | 31779 | 37.852 | 4022 | 33.9 | 11.2122 | 0.2744 | 0.7988 |
| itw_full | mms_300m | 31779 | 37.852 | 4198 | 32.5 | 2.9016 | 0.0726 | 0.0739 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (94 file) |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

**Mục đích:** (1) kiểm chứng code, dữ liệu và checkpoint bằng ba mốc công bố; (2) chấm zero-shot trên RADAR tiếng Việt cho 3 backbone sẽ fine-tune (bản Default) và baseline; (3) lấy mốc `ood_forget` cho GĐ5–6; (4) đo tốc độ để chia các lần sau.

**Kiểm chứng: qua cả ba.** Toy 6,1101% (README 6,11%; checkpoint HF giống bản Zenodo); In-the-Wild MMS-300M Default 2,9016% (Bảng II bài AntiDeepfake: 2,90); W2V2-AASIST 11,2122% (model card: 11,22).

**Quota:** sau lần chạy, `push.sh quota` báo đã dùng 2,43 / 30 giờ (đo 20:29 ngày 13/11; lúc đó chưa có `quota_after.txt` tự động).

**Output đầy đủ** (gồm embedding `emb.npy`, `*_emb.pkl`, điểm từng shard; 664 MB) ở `results/kaggle-raw/orpheus-zeroshot-v1/` trên máy, không commit. Bản gốc vẫn còn trên Kaggle (version 1 của kernel).

**Lưu trữ:** thư mục này được dựng lúc 22:48 ngày 13/11 từ output đã tải lúc 20:29; `run.py` và `kernel-metadata.json` lấy từ commit `9b9fc04` (giống bản đã đẩy).

Phân tích kết quả: `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 5.6.
