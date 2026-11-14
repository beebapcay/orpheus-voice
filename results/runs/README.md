# Bằng chứng các lần chạy GPU

Mỗi thư mục `<kernel>-v<N>` là một lần chạy trên Kaggle (một version của kernel). Mỗi thư mục `colab-<tên>-<YYYYMMDD>-<HHMM>` là một lần chạy notebook trên Colab (giờ UTC, `RUN.md` viết tay). Dùng cho báo cáo và để tái hiện. Đọc `RUN.md` trong từng thư mục trước.

| Lần chạy | Mục đích | Kết quả |
|---|---|---|
| `colab-gd2-toy-20261113-0848` | GĐ2: dựng môi trường, toy example (Colab T4) | Lỗi cài fairseq (numpy 2 kéo theo); sửa ở `46e3404` |
| `colab-gd2-toy-20261113-0909` | GĐ2: chạy lại sau khi sửa | Qua: EER 6,1101%, không cảnh báo nạp trọng số |
| `orpheus-prep-v1` | Chuẩn bị dữ liệu kiểm tra (RADAR VI/EN, In-the-Wild) | Xong, ~33 phút |
| `orpheus-zeroshot-v1` | Kiểm chứng 3 mốc; zero-shot 3 backbone (Default) + baseline; mốc `ood_forget` | Xong, 2,42 giờ; qua 3 mốc |
| `orpheus-zeroshot-v2` | Zero-shot: MMS-300M NDA, MMS-1B, XLS-R-1B (hai bản), W2V-Large NDA, W2V-Small NDA | Xong, 2,71 giờ; NDA tốt hơn Default ở 4/5 backbone |
| `orpheus-zeroshot-v3` | Zero-shot: HuBERT-XL, XLS-R-2B (hai bản) | Xong, 5,18 giờ; XLS-R-2B Default tốt nhất (13,71%) |
| `orpheus-finetune-v1` | GĐ4 pilot fine-tune: R và A0 300 bước, overfit 100 câu; đo bộ nhớ, tốc độ, CPU | ERROR do một shard chấm chỉ có một lớp (đã sửa); huấn luyện xong cả 3; 0,74 giờ |

**Cách sinh:** `kaggle/push.sh` tự tạo khi đẩy kernel (`run.py`, `kernel-metadata.json`, `pushed.json`) và khi tải kết quả (`output zeroshot N` hoặc `archive-log prep N`: log, môi trường, log từng shard, `RUN.md`). Script: `kaggle/archive_run.py`. Mục **Ghi chú** trong `RUN.md` viết tay.

**Không lưu ở đây:** embedding (ở `results/kaggle-raw/`, không commit), dữ liệu âm thanh, checkpoint. Điểm từng câu và độ đo đã gộp ở `results/zeroshot/`.
