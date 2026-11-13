# orpheus-zeroshot version 3

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-zeroshot`, version 3 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-13T23:29:18+07:00 |
| Commit repo lúc đẩy | `2c76db8` |
| Code trong dataset `orpheus-src` | `git 968f43e`, staged 2026-11-13 17:14:30 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 16:29:25 |
| Kết thúc (UTC, dòng log cuối có giờ) | 21:40:16 |
| Thời gian chạy | 5.18 giờ (log ghi đến giây 18666) |
| Dựng môi trường | 4.1 phút |
| GPU | Tesla T4, 15360 MiB (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-14T04:46:10+07:00 |

## Job

- `radar_vi:hubert_xl,hubert_xl_nda,xlsr_2b,xlsr_2b_nda`

## Kết quả của lần chạy này

| Tập | Mô hình | Câu | Giờ âm thanh | Giây GPU | x thời gian thực | EER (%) | minDCF | actDCF |
|---|---|---|---|---|---|---|---|---|
| radar_vi | hubert_xl_nda | 13438 | 17.042 | 4017 | 15.3 | 30.6743 | 0.6570 | 0.9929 |
| radar_vi | hubert_xl | 13438 | 17.042 | 4047 | 15.2 | 20.4867 | 0.4423 | 0.4591 |
| radar_vi | xlsr_2b | 13438 | 17.042 | 7100 | 8.6 | 13.7149 | 0.3323 | 0.3455 |
| radar_vi | xlsr_2b_nda | 13438 | 17.042 | 7045 | 8.7 | 14.2283 | 0.3571 | 0.3746 |

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (56 file) |
| `quota_after.txt` | Quota GPU còn lại, đo lúc lưu trữ |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

**Mục đích:** chấm zero-shot trên RADAR tiếng Việt cho 4 checkpoint lớn còn lại: HuBERT-XL và XLS-R-2B, cả bản Default và NDA. Xếp hai bản HuBERT-XL chạy song song trước; hai bản XLS-R-2B chạy lần lượt vì mỗi bản giữ 19 GB trong ngân sách RAM 27 GB.

**Kết quả:** đủ 4 job, không cảnh báo nạp trọng số (HuBERT-XL 804 tensor). XLS-R-2B Default tốt nhất trong 14 checkpoint: 13,71%; XLS-R-2B NDA 14,23%; HuBERT-XL Default 20,49%; HuBERT-XL NDA 30,67% (actDCF 0,993).

**HuBERT-XL NDA kém bất thường không phải lỗi chạy:** checkpoint nạp đủ, và Bảng II bài AntiDeepfake cũng cho thấy bản này kém hẳn trên mọi tập (In-the-Wild 17,99 so với 5,23). Trên RADAR, điểm của tiếng thật dồn về gần 0 (trung vị 0,19), 47,6% câu thật bị coi là giả ở ngưỡng 0.

**Máy:** log lần này ghi thêm CPU: 4 nhân Intel Xeon 2,0 GHz cho cả hai GPU T4.

**Tốc độ:** HuBERT-XL ~15,2x thời gian thực, XLS-R-2B ~8,6x (~2 giờ cho `radar_vi`). Tải checkpoint 2B (8,65 GB) từ HF ~84 s.

**Quota:** sau lần chạy đã dùng 10,32 / 30 giờ (`quota_after.txt`, đo 04:4x ngày 14/11, trước mốc làm mới 7 giờ sáng); lần này tốn ~5,2 giờ.

**Output đầy đủ** (embedding; 742 MB) ở `results/kaggle-raw/orpheus-zeroshot-v3/` trên máy, không commit; bản gốc còn trên Kaggle (version 3).

**Sau lần chạy:** khoảng tin cậy bootstrap cho cả 15 mô hình tính trên máy (`src/bootstrap_eer.py`, `results/zeroshot/bootstrap_radar_vi.csv`). Phân tích: `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 5.6.
