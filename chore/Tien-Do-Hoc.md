# Tiến độ học

Ghi lại mức độ đã đọc của từng phần, để biết phần nào cần quay lại đọc kỹ. Cập nhật lần cuối: 13/11/2026 (GĐ3, xong lần chấm zero-shot đầu tiên).

**Mức đọc:** **Kỹ** = đọc bài gốc, tự trả lời được câu hỏi. **Lướt** = chỉ đọc tóm tắt hoặc note, chưa đối chiếu bài gốc hay code. **Chưa** = chưa đọc.

## Tổng quan

| GĐ | Mức | Trạng thái |
|---|---|---|
| 1. Bản đồ các hướng | Ngày 1 kỹ; ngày 2, 3 lướt | Đã qua, còn nợ đọc kỹ ngày 2, 3 |
| 2. Backbone SSL và môi trường | Ngày 1 lướt; ngày 2 xong | **Đã qua** (13/11). Còn nợ đọc kỹ ngày 1 |
| 3. Post-training và zero-shot | Ngày 1 chưa đọc (đã có tài liệu); ngày 2 lần chấm 1, 2 xong | Đang làm (13/11). Lần 3 (HuBERT-XL, XLS-R-2B) đang chạy |
| 4–8 | Chưa | |

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
| 2 | Dựng môi trường, toy example | **Xong**: EER 6,11%, khớp README | |

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

### Ngày 2 (xong 13/11)

- [x] Ô 1–2: bật GPU T4, nạp Drive, chép 2 script
- [x] Ô 3: dựng môi trường; ghi phiên bản Python, torch, CUDA, GPU
- [x] Ô 4: toy example; EER đúng 6,11%, không có cảnh báo nạp trọng số
- [x] Ô 5: lưu log về Drive
- [x] Ghi thời gian cài đặt, tốc độ `it/s`
- [x] Kiểm tra quota GPU: Colab miễn phí, phiên tối đa khoảng 4 giờ

Hướng dẫn chi tiết từng ô ở mục 4.2 của `chore/GD2-Backbone-Va-Moi-Truong.md`.

## GĐ3. Post-training và zero-shot

| Ngày | Nội dung | Mức | Cần quay lại |
|---|---|---|---|
| 1 | AntiDeepfake, RawBoost (nhóm B) | Chưa (tài liệu sẵn 13/11) | Đọc theo bảng trang ở mục 1 của file GĐ3: lượt 1 (kiến trúc, kết quả) trước, lượt 2 (dữ liệu, siêu tham số) sau |
| 2 | Chuẩn bị dữ liệu, chấm zero-shot trên Kaggle | Lần 1 xong (13/11): qua 3 mốc kiểm chứng; lần 2 đang chạy | Đọc kết quả ở mục 5.6 và phát hiện 10–14 (mục 3) file GĐ3 |

Tài liệu: `note/Post-Training-SSL.md`, `note/RawBoost.md`, `chore/GD3-Post-Training-Va-Zero-Shot.md`, `chore/Quy-Trinh-Chay-Kaggle.md`.

Việc ngày 1:
- [ ] Lượt 1 AntiDeepfake: Hình 1–2, Bảng II, Bảng III, mục V
- [ ] Lượt 1 RawBoost: Hình 1, Bảng 2
- [ ] Tự trả lời 7 câu cuối mỗi note
- [ ] Nói lại nháp vấn đáp 3 (mục 6 file GĐ3)

Việc ngày 2:
- [x] Cài Kaggle CLI, đăng nhập OAuth, xác minh số điện thoại, xem quota GPU (30 giờ/tuần)
- [x] Lần 1: toy 6,1101%; In-the-Wild MMS-300M 2,9016%; RADAR tiếng Việt cho 3 backbone + baseline (W2V-Large tốt nhất, 17,80%)
- [x] Lần 2 (7 checkpoint, xong 13/11): NDA tốt hơn Default ở 4/5 backbone; bản 1B ~14–16%
- [ ] Lần 3 (HuBERT-XL, XLS-R-2B, đang chạy)
