# Tiến độ học

Ghi lại mức độ đã đọc của từng phần, để biết phần nào cần quay lại đọc kỹ. Cập nhật lần cuối: 14/11/2026 (GĐ3 qua phần chạy, chuẩn bị GĐ4).

**Mức đọc:** **Kỹ** = đọc bài gốc, tự trả lời được câu hỏi. **Lướt** = chỉ đọc tóm tắt hoặc note, chưa đối chiếu bài gốc hay code. **Chưa** = chưa đọc.

## Tổng quan

| GĐ | Mức | Trạng thái |
|---|---|---|
| 1. Bản đồ các hướng | Ngày 1 kỹ; ngày 2, 3 lướt | Đã qua, còn nợ đọc kỹ ngày 2, 3 |
| 2. Backbone SSL và môi trường | Ngày 1 lướt; ngày 2 xong | **Đã qua** (13/11). Còn nợ đọc kỹ ngày 1 |
| 3. Post-training và zero-shot | Ngày 1 chưa đọc (đã có tài liệu, **hoãn** 14/11); ngày 2 xong | **Đã qua phần chạy** (14/11): đủ 14 checkpoint + baseline, phân tích lại, quyết định ở mục 8 file GĐ3. Còn nợ đọc kỹ ngày 1 |
| 4. Dữ liệu tiếng Việt và pipeline | Ngày 1 lướt bài SEA-Spoof; ngày 2 xong | **Đã qua** (14/11): chia tập đã chốt (D1–D3), pilot fine-tune xong, cấu hình huấn luyện đã chốt (E1–E3). Tổng kết ở mục 7 file GĐ4 |
| 5–8 | Chưa | |

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
| 2 | Chuẩn bị dữ liệu, chấm zero-shot trên Kaggle, phân tích lại | **Xong** (14/11) | Đọc tổng kết ở mục 8, kết quả ở mục 5.6–5.7, phát hiện 1–18 ở mục 3 file GĐ3 |

Tài liệu: `note/Post-Training-SSL.md`, `note/RawBoost.md`, `chore/GD3-Post-Training-Va-Zero-Shot.md`, `chore/Quy-Trinh-Chay-Kaggle.md`.

Việc ngày 1:
*(Hoãn 14/11 theo quyết định của người dùng: ưu tiên chạy code, đọc sau. Không chặn GĐ4.)*
- [ ] Lượt 1 AntiDeepfake: Hình 1–2, Bảng II, Bảng III, mục V
- [ ] Lượt 1 RawBoost: Hình 1, Bảng 2
- [ ] Tự trả lời 7 câu cuối mỗi note
- [ ] Nói lại nháp vấn đáp 3 (mục 6 file GĐ3)

Việc ngày 2:
- [x] Cài Kaggle CLI, đăng nhập OAuth, xác minh số điện thoại, xem quota GPU (30 giờ/tuần)
- [x] Lần 1: toy 6,1101%; In-the-Wild MMS-300M 2,9016%; RADAR tiếng Việt cho 3 backbone + baseline (W2V-Large tốt nhất, 17,80%)
- [x] Lần 2 (7 checkpoint, xong 13/11): bản 1B ~14–16%; NDA tốt hơn Default ở cả 3 bản ≤ 317M (bảng đủ ở mục 5.6 file GĐ3)
- [x] Lần 3 (HuBERT-XL, XLS-R-2B, xong 14/11): XLS-R-2B Default tốt nhất 13,71%; khoảng tin cậy bootstrap cho cả bảng
- [x] Phân tích lại trên máy (14/11): hai loại lỗi, độ dài câu, sai trùng nhau, tương quan với bài; tổng kết và quyết định (mục 8 file GĐ3)

## GĐ4. Dữ liệu tiếng Việt và pipeline

| Ngày | Nội dung | Mức | Cần quay lại |
|---|---|---|---|
| 1 | SEA-Spoof: phần mô tả dữ liệu tiếng Việt và hệ thống sinh (nhóm C, đọc lướt) | Lướt (Claude đọc bài, tóm ở mục 2 file GĐ4) | Bảng 1, 3 và mục 2.2–2.3 của bài; chỗ lệch giữa bài và dữ liệu ở mục 2.2 file GĐ4 |
| 2 | Pipeline fine-tune (`mms_300m_pre-post-sft.yaml`) | **Xong** (14/11): pilot trên Kaggle | Đọc phát hiện 8–10, 13–15 (mục 2.2, 4.1 file GĐ4): lịch learning rate, RawBoost trên dev, đệm khi kiểm tra dev |

Tài liệu: `chore/GD4-Du-Lieu-Va-Pipeline.md`.

Việc trước ngày 1 (người dùng, mục 1 file GĐ4):
- [x] Kiểm tra quyền SEA-Spoof: đã được duyệt, đã kiểm bằng tải file (14/11)
- [x] Xin quyền MLAAD (dự phòng): có quyền, kiểm 14/11
- [x] `hf auth login` trên máy (14/11)
