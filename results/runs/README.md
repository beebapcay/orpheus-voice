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
| `orpheus-finetune-v6` | GĐ6 lần 1: G1 seed 1 (GRPO đủ), G2 seed 1 (GRPO đơn giản) | Xong, 5,53 giờ; RADAR VI G1 23,53%, G2 23,13% (R 24,20 ± 1,88); trong khoảng của R |
| `orpheus-finetune-b-v2` | GĐ6 lần 2 (song song lần 1): R seed 4, 5 | Xong, 4,66 giờ; RADAR VI 26,63 / 22,32%; R qua 5 seed 24,20 ± 1,88% |
| `orpheus-finetune-v5` | GĐ6 lần thử đo GRPO (7 lần đo ngắn): bộ nhớ, tốc độ G1/G2 ở batch 50 và 25 × 2, có thử áp lực 240 câu dài nhất | Xong, 0,47 giờ; G1 batch 50 hết bộ nhớ ở thử áp lực, qua với `expandable_segments` (đỉnh 13,9 / 14,6 GB); G1 3,07 s/bước, G2 2,95 s/bước (SFT ~2,0) |
| `orpheus-finetune-v4` | GĐ5 lần 4: R seed 2, 3; chấm MLAAD; S0 trên MLAAD (4 mô hình) | Xong, 4,94 giờ; RADAR VI R qua 3 seed 24,02 ± 1,52%; MLAAD: S0 2,2–5,2%, R 0,09–0,16% |
| `orpheus-finetune-b-v1` | GĐ5 lần 3 (kernel thứ hai, song song với v3): B1 seed 1 (W2V-Small), N0 seed 1 (MMS-300M NDA) | Xong, 3,89 giờ; RADAR VI: B1 34,67%, N0 20,50% |
| `orpheus-finetune-v3` | GĐ5 lần 2: P0 seed 1 (khởi tạo fairseq), B2 seed 1 (W2V-Large) | Xong, 4,14 giờ (xếp hàng ~3,5 giờ); RADAR VI: P0 31,46%, B2 21,03% |
| `orpheus-finetune-v2` | GĐ5 lần 1: R seed 1, A0 seed 1 (3 epoch); S0 trên SEA-Spoof dev/test (4 mô hình) | Xong, 4,68 giờ; RADAR VI: R 23,07%, A0 18,37% (S0 20,08%); lịch lr lệch E1 (total_steps 50000) |

**Cách sinh:** `kaggle/push.sh` tự tạo khi đẩy kernel (`run.py`, `kernel-metadata.json`, `pushed.json`) và khi tải kết quả (`output zeroshot N` hoặc `archive-log prep N`: log, môi trường, log từng shard, `RUN.md`). Script: `kaggle/archive_run.py`. Mục **Ghi chú** trong `RUN.md` viết tay.

**Không lưu ở đây:** embedding (ở `results/kaggle-raw/`, không commit), dữ liệu âm thanh, checkpoint. Điểm từng câu và độ đo đã gộp ở `results/zeroshot/`.
