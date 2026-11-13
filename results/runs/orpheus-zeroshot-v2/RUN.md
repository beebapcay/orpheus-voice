# orpheus-zeroshot version 2

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-zeroshot`, version 2 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-13 khoảng 20:35 +07:00 |
| Commit repo lúc đẩy | `4405c15` |
| Code trong dataset `orpheus-src` | `git 968f43e`, staged 2026-11-13 17:14:30 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 13:38:01 |
| Kết thúc (UTC, dòng log cuối có giờ) | 16:20:14 |
| Thời gian chạy | 2.71 giờ (log ghi đến giây 9746) |
| Dựng môi trường | 3.8 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-13T23:30:53+07:00 |

## Job

- `radar_vi:mms_300m_nda,mms_1b,xlsr_1b,mms_1b_nda,xlsr_1b_nda,w2v_large_nda,w2v_small_nda`

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| radar_vi | mms_300m_nda | 13438 | 17.042 | 2280 | 26.9 | 18.0680 | 0.4423 | 0.4483 |
| radar_vi | mms_1b | 13438 | 17.042 | 3468 | 17.7 | 14.3178 | 0.3512 | 0.3553 |
| radar_vi | xlsr_1b | 13438 | 17.042 | 3497 | 17.5 | 14.9874 | 0.3591 | 0.3717 |
| radar_vi | mms_1b_nda | 13438 | 17.042 | 3437 | 17.9 | 14.1985 | 0.3510 | 0.3520 |
| radar_vi | w2v_large_nda | 13438 | 17.042 | 1410 | 43.5 | 16.6172 | 0.3714 | 0.3814 |
| radar_vi | w2v_small_nda | 13438 | 17.042 | 673 | 91.1 | 23.2399 | 0.5165 | 0.5229 |
| radar_vi | xlsr_1b_nda | 13438 | 17.042 | 3519 | 17.4 | 15.5975 | 0.3650 | 0.3742 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (98 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

**Mục đích:** chấm zero-shot trên RADAR tiếng Việt cho 7 checkpoint còn lại có kích thước ≤ 1B: bản NDA (không RawBoost) của MMS-300M, W2V-Large, W2V-Small, và MMS-1B, XLS-R-1B cả hai bản. MMS-300M NDA đứng đầu hàng đợi vì cần cho phát hiện 1 của GĐ3 (RawBoost có lợi trên tiếng Việt không).

**Kết quả:** đủ 7 job, không cảnh báo nạp trọng số. Bản NDA tốt hơn bản Default ở 4/5 backbone đã có đủ hai bản (MMS-300M 18,07 so với 20,08; W2V-Large 16,62 so với 17,80; W2V-Small 23,24 so với 27,21; MMS-1B 14,20 so với 14,32); XLS-R-1B ngược lại (15,60 so với 14,99). Bản 1B tốt hơn bản 300M khoảng 3–6 điểm EER.

**Tốc độ:** bản 1B ~17,5x thời gian thực (~58 phút cho `radar_vi`); tải checkpoint 1B (3,86 GB) từ HF 28–109 s. W2V-Small NDA đạt 91x vì chạy cuối, một mình trên máy, củng cố giả thuyết nút thắt ở CPU (phát hiện 14 của GĐ3).

**Quota:** sau lần chạy còn 24,84 / 30 giờ (`quota_after.txt`); lần này tốn ~2,73 giờ.

**Output đầy đủ** (embedding; 918 MB) ở `results/kaggle-raw/orpheus-zeroshot-v2/` trên máy, không commit; bản gốc còn trên Kaggle (version 2).

Phân tích: `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 5.6.
