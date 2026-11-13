# Bằng chứng các lần chạy trên Kaggle

Mỗi thư mục `<kernel>-v<N>` là một lần chạy (một version của kernel trên Kaggle). Dùng cho báo cáo và để tái hiện. Đọc `RUN.md` trong từng thư mục trước.

| Lần chạy | Mục đích | Kết quả |
|---|---|---|
| `orpheus-prep-v1` | Chuẩn bị dữ liệu kiểm tra (RADAR VI/EN, In-the-Wild) | Xong, ~33 phút |
| `orpheus-zeroshot-v1` | Kiểm chứng 3 mốc; zero-shot 3 backbone (Default) + baseline; mốc `ood_forget` | Xong, 2,42 giờ; qua 3 mốc |
| `orpheus-zeroshot-v2` | Zero-shot: MMS-300M NDA, MMS-1B, XLS-R-1B (hai bản), W2V-Large NDA, W2V-Small NDA | Đang chạy |

**Cách sinh:** `kaggle/push.sh` tự tạo khi đẩy kernel (`run.py`, `kernel-metadata.json`, `pushed.json`) và khi tải kết quả (`output zeroshot N` hoặc `archive-log prep N`: log, môi trường, log từng shard, `RUN.md`). Script: `kaggle/archive_run.py`. Mục **Ghi chú** trong `RUN.md` viết tay.

**Không lưu ở đây:** embedding (ở `results/kaggle-raw/`, không commit), dữ liệu âm thanh, checkpoint. Điểm từng câu và độ đo đã gộp ở `results/zeroshot/`.
