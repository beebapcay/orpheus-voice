# orpheus-prep version 1

File này do `kaggle/archive_run.py` sinh ra. Mục **Ghi chú** cuối file viết tay.

## Thông tin chạy

| | |
|---|---|
| Kernel | `orpheus-prep`, version 1 (riêng tư trên Kaggle) |
| Đẩy lúc | 2026-11-13 (chiều, giờ VN; không ghi lại chính xác) |
| Commit repo lúc đẩy | `9b9fc04` |
| Code trong dataset `orpheus-src` | `git 968f43e`, staged 2026-11-13 17:14:30 |
| Bắt đầu (UTC, dòng log đầu có giờ) | 10:15:41 |
| Kết thúc (UTC, dòng log cuối có giờ) | 10:48:40 |
| Thời gian chạy | ? (log ghi đến giây 1993) |
| Dựng môi trường | ? |
| GPU | ? (xem `kernel.log.txt` cho đủ) |
| Lưu trữ lúc | 2026-11-13T22:50:12+07:00 |

## Job

?

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
| `shard_logs/` | Log `main.py` và thời gian của từng shard (0 file) |

Không lưu ở đây: embedding (ở `results/kaggle-raw/`, không commit), điểm từng câu (ở `results/zeroshot/<mô hình>/<tập>/scores.csv`).

## Tái hiện

1. `git checkout <commit repo lúc đẩy>` (hoặc commit ghi trong dòng `orpheus-src`).
2. `bash kaggle/push.sh src` để đóng gói lại code và danh sách câu.
3. Chép `run.py` trong thư mục này vào `kaggle/<kernel>/run.py`, rồi `bash kaggle/push.sh push <kernel>`.
4. So `summary.csv` mới với file ở đây. Các bước chọn câu và dữ liệu cố định (`protocols/ids/`, seed 2026); checkpoint lấy từ Hugging Face.

## Ghi chú

**Mục đích:** tải và chuẩn bị dữ liệu kiểm tra (RADAR eval tiếng Việt và tiếng Anh, In-the-Wild) thành CSV theo định dạng AntiDeepfake. Kernel CPU có internet, không tốn quota GPU.

**Kết quả:** COMPLETE sau ~33 phút. Số câu đúng như danh sách cố định trong `protocols/ids/`: `radar_vi` 13.438 (7.000 thật / 6.438 giả), `itw_full` 31.779 (19.963 / 11.816), `itw_sub` 2.000, `radar_en_sub` 2.000, `ood_forget` 4.000. Output ~9,96 GB, nằm trên Kaggle và được gắn làm input cho `orpheus-zeroshot` (`/kaggle/input/notebooks/<user>/orpheus-prep/data`). Âm thanh không đưa lên git.

**`orpheus_data_manifest.json`:** trích từ log của `orpheus-zeroshot` v1 (kernel đó in nguyên file manifest của output prep). Không tải output prep về máy: >45.000 file, CLI liệt kê hết danh sách thì bị Kaggle giới hạn lượt gọi (lỗi 429).

**Dòng đỏ trên trang web:** `SyntaxWarning` của mistune và nbconvert, do Kaggle chuyển log sang HTML sau khi script chạy xong. Vô hại.

**Lưu trữ:** log và RUN.md được lấy về lúc 22:49 ngày 13/11 bằng `push.sh archive-log prep 1`, không phải ngay sau khi chạy. Không có quota lúc chạy xong (kernel CPU).
