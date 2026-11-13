# Tiến độ học

Ghi lại mức độ đã đọc của từng phần, để biết phần nào cần quay lại đọc kỹ. Cập nhật lần cuối: 13/11/2026.

**Mức đọc:** **Kỹ** = đọc bài gốc, tự trả lời được câu hỏi. **Lướt** = chỉ đọc tóm tắt hoặc note, chưa đối chiếu bài gốc hay code. **Chưa** = chưa đọc.

## Tổng quan

| GĐ | Mức | Trạng thái |
|---|---|---|
| 1. Bản đồ các hướng | Ngày 1 kỹ; ngày 2, 3 lướt | Đã qua, còn nợ đọc kỹ ngày 2, 3 |
| 2. Backbone SSL và môi trường | Ngày 1 lướt | **Đang làm ngày 2: chạy toy example** |
| 3–8 | Chưa | |

## GĐ1. Bản đồ các hướng

| Ngày | Nội dung | Mức | Cần quay lại |
|---|---|---|---|
| 1 | ALLM4ADD (nhóm A), PartialSpoof (nhóm C) | **Kỹ** | |
| 2 | CodecFake+, VSASV (nhóm A); Phân loại theo cấu tạo codec, VietPrism (nhóm C) | Lướt (đọc summary) | Đọc kỹ 2 bài nhóm A: phần này tính vào tiêu chí tổng quan 30% |
| 3 | GRPO, XLSR-SLS (nhóm A); Speech DF Arena, RADAR 2026, Tak 2022 (nhóm C) | Lướt (đọc summary) | Đọc kỹ GRPO (nền trực tiếp của đề tài) và XLSR-SLS; XLSR-SLS vẫn chưa có PDF |

Tài liệu: `chore/GD1-Ban-Do-Cac-Huong.md`; các note `Audio-LLM`, `Gia-Mao-Mot-Phan`, `Codec-Va-Truy-Vet-Nguon`, `Deepfake-Tieng-Viet`, `Post-Training-Va-GRPO`, `SSL-Va-Back-End`, `Benchmark-Va-Do-Ben` trong `note/`.

Việc còn nợ của GĐ1:
- [ ] Đọc kỹ CodecFake+, VSASV, GRPO, XLSR-SLS
- [ ] Tải tay PDF XLSR-SLS vào `resource/paper/`
- [ ] Tự nói mỗi hướng trong 1 phút không nhìn tài liệu

## GĐ2. Backbone SSL và môi trường

| Ngày | Nội dung | Mức | Cần quay lại |
|---|---|---|---|
| 1 sáng | wav2vec 2.0, XLS-R, MMS | Lướt (đọc summary tổng quan) | Đối chiếu bài gốc theo bảng trang cần đọc |
| 1 chiều | Lần theo code AntiDeepfake | Lướt (đọc tóm tắt kèm code) | Đọc code thật theo bảng 8 file |
| 2 | Dựng môi trường, toy example | **Đang làm** | |

Tài liệu: `note/Backbone-SSL.md`; `chore/GD2-Backbone-Va-Moi-Truong.md`; `src/setup_antideepfake.sh`; `src/evaluate.py`.

### Khi quay lại đọc kỹ ngày 1

**Bài gốc** (trong `resource/paper/`):

| Bài | Đọc | Trang |
|---|---|---|
| wav2vec 2.0 | Hình 1, mục 2, 3.1–3.2, 4.2–4.3, Bảng 4 | 2–5, 8 |
| XLS-R | Mục 2 (cuối), 3.1, Bảng 1 (dòng English, Vietnamese), Bảng 2, mục 5.2.5 + Bảng 11 | 2–6, 8, 14–15 |
| MMS | Mục 3.1.1 (1 đoạn), mục 4.2–4.3, Bảng 2, Hình 10–11 | 13–14 |

**Code** (commit `86098091`):

| File | Dòng | Câu hỏi |
|---|---|---|
| `models/W2V.py` | 39–58 | Vì sao `mask=False`? Lấy lớp nào? |
| `models/W2V.py` | 65–105 | Kích thước tensor sau từng bước? Cột nào là "thật"? |
| `models/W2V_configs.py` | 7–17 | `w2v_small`, `w2v_large`, `mms_300m` khác nhau ở đâu? |
| `dataio/dataio.py` | 47–102, 158–180 | Huấn luyện và chấm xử lý âm thanh khác gì? |
| `main.py` | 89–123 | Nạp checkpoint thế nào? |
| `utils.py` | 21–61 | Vì sao nạp sai mà không báo lỗi? |
| `evaluation.py` | 42–77, 133–154 | Từ logit ra EER thế nào? |
| `hparams/mms_300m.yaml` | toàn bộ | `lr`, `warmup`, `max_batch_length`, `rawboost.algo` |

**Tự kiểm tra:** 8 câu ở mục 11 của `note/Backbone-SSL.md`; vẽ lại luồng dữ liệu ở mục 1 của file GĐ2; nói lại nháp câu vấn đáp 3 ở mục 5 của file GĐ2.

### Ngày 2 (đang làm)

- [ ] Ô 1–2: bật GPU T4, nạp Drive, chép 2 script
- [ ] Ô 3: dựng môi trường; ghi phiên bản Python, torch, CUDA, GPU
- [ ] Ô 4: toy example; EER đúng 6,11%, không có cảnh báo nạp trọng số
- [ ] Ô 5: lưu log về Drive
- [ ] Ghi thời gian cài đặt, tốc độ `it/s`
- [ ] Kiểm tra quota GPU

Hướng dẫn chi tiết từng ô ở mục 4.2 của `chore/GD2-Backbone-Va-Moi-Truong.md`.
