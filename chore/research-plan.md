# Kế hoạch nghiên cứu: Fake Voice Detection

**Đề tài:** Phân tích các yếu tố quyết định khả năng phát hiện deepfake tiếng Việt của mô hình SSL post-trained, và đánh giá fine-tune bằng học tăng cường (GRPO) so với fine-tune có giám sát.

**Câu hỏi nghiên cứu**

1. Những yếu tố nào quyết định khả năng phát hiện deepfake tiếng Việt, xét theo 3 nhóm: **mô hình nền** (kích thước, ngôn ngữ và cách pre-train), **cách train** (post-training, tăng cường dữ liệu RawBoost), và **dữ liệu** (độ phủ hệ thống sinh, lượng dữ liệu tiếng Việt khi fine-tune)? *(Đặt lại 14/11 sau GĐ3: zero-shot cho thấy ngôn ngữ ảnh hưởng ít hơn kích thước và độ phủ dữ liệu; xem `chore/GD3-Post-Training-Va-Zero-Shot.md` phát hiện 17, 18.)*
2. Fine-tune có giám sát trên tiếng Việt có làm mô hình mất khả năng tổng quát hoá không?
3. GRPO có giữ được khả năng tổng quát hoá tốt hơn trên tiếng Việt không, và thành phần nào của GRPO tạo ra khác biệt? *(Đặt lại 15/11 sau GĐ6 lần 1: GRPO đúng code gốc không giữ tổng quát hoá tốt hơn SFT khi miền fine-tune đã bão hoà; câu hỏi chuyển thành "vì sao, và bản GRPO sửa theo nguyên nhân có giúp không"; xem `chore/GD6-GRPO.md` mục 11, 12.)*

**Tổng thời gian:** 21 ngày, gồm 20 ngày làm việc và 1 ngày dự phòng.

**Giả định đã biết (không học lại):** tín hiệu, STFT, spectrogram, EER, minDCF, actDCF, LLR, khái niệm front-end/back-end.

**Nguyên tắc thông tin:** mọi số liệu và nhận định đưa vào báo cáo phải lấy từ bài gốc hoặc code gốc. Mục 9 ghi rõ phần nào đã kiểm tra, phần nào chưa.

**Cách đọc file này:** mục 1–2 là lý do và điều kiện thực hiện; mục 3 là toàn bộ việc đọc; mục 4–6 là các giai đoạn làm việc; mục 7–9 dùng để kiểm tra trước khi nộp. Phụ lục chỉ để tra cứu.

## 1. Vì sao chọn hướng này

| Tiêu chí | Bằng chứng |
|---|---|
| Baseline cũ không còn đủ | SSL-AASIST có EER 42,6% trên tập đa ngôn ngữ của [RADAR 2026](https://arxiv.org/abs/2605.09568); đội tốt nhất đạt 5,10% |
| Post-training cho kết quả tốt | [AntiDeepfake](https://arxiv.org/abs/2506.21090): zero-shot trên In-the-Wild, XLS-R-1B đạt 1,35% và XLS-R-2B đạt 1,23% (có RawBoost) |
| Có vấn đề mở rõ ràng | [Bài GRPO](https://arxiv.org/abs/2603.02914) (Interspeech 2026): fine-tune có giám sát làm XLS-R-2B giảm trên dữ liệu ngoài miền, EER trung bình từ 2,69% lên 6,28%. Tác giả nêu hạn chế: chỉ fine-tune trên một bộ dữ liệu, ablation chỉ làm với XLS-R-2B, cơ chế cần nghiên cứu thêm |
| Hiệu ứng rõ ở mô hình vừa sức GPU miễn phí | Cũng trong bài GRPO, MMS-300M trên In-the-Wild: không fine-tune 2,91%, SFT 14,17%, GRPO 5,91%. MMS-300M fine-tune được trên GPU 16 GB (mục 2) |
| Tiếng Việt còn khoảng trống | [VietPrism](https://arxiv.org/abs/2609.30005) chỉ đánh giá zero-shot 5 detector, không fine-tune, không ablation. EER rất phân tán theo hệ thống sinh, từ 0,00% đến 99,99% |
| Dữ liệu, code có sẵn và khả thi | Xem mục 2 |

**Điều đồ án không khẳng định:** không đề xuất kiến trúc mới, không nhận là đầu tiên về phát hiện deepfake tiếng Việt. Đóng góp là phân tích có kiểm soát trên tiếng Việt. XLS-R + AASIST được giữ làm baseline để so sánh.

## 2. Dữ liệu, code và tài nguyên tính toán (đã kiểm tra trực tiếp)

### Code và trọng số

| Thành phần | Trạng thái | Giấy phép | Ghi chú |
|---|---|---|---|
| [AntiDeepfake](https://github.com/nii-yamagishilab/AntiDeepfake) | Có, cập nhật 27/10/2026 | Code BSD-3 | Có train, fine-tune, inference, `evaluation.py`, RawBoost, GRPO, AASIST, script protocol cho In-the-Wild |
| 14 checkpoint post-trained ([HF](https://huggingface.co/nii-yamagishilab), [Zenodo](https://zenodo.org/records/15580543)) | Có, không giới hạn truy cập | CC BY-NC-SA 4.0 | 7 backbone, mỗi backbone có bản Default và NDA; từ 0,38 GB đến 8,65 GB |
| Trọng số fairseq gốc (W2V-Small, W2V-Large, MMS-300M, XLS-R-1B) | Tải được | Theo fairseq | 0,95 GB, 3,17 GB, 3,81 GB, 11,59 GB |
| [W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST) | Có | MIT | Baseline so sánh |
| [DF-Arena 1B](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1) | Có | Phi thương mại | Cần `trust_remote_code=True`; không nêu backbone |

**Chưa chạy thử code.** GĐ2 có bước kiểm chứng bắt buộc trước khi đi tiếp.

### Dữ liệu

| Bộ dữ liệu | Truy cập | Dung lượng | Dùng cho | Ghi chú |
|---|---|---|---|---|
| [RADAR 2026](https://zenodo.org/records/22110815) | Mở, CC BY 4.0, có nhãn | 13,9 GB | Kiểm tra | Tiếng Việt: 13.438 câu (7.000 thật, 6.438 giả). Giả từ iFlytek, ElevenLabs, Cartesia, OpenAI, Fish Audio S2 Pro, Piper. Có biến đổi âm thanh. Không có nhãn hệ thống sinh theo từng câu |
| [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) | Mở | 8,16 GB | Kiểm tra ngoài miền | AntiDeepfake không dùng để huấn luyện |
| [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) | Chờ tác giả duyệt | 81,5 GB (cả bộ) | Fine-tune chính | 81.020 câu tiếng Việt; có cột `spoof_type`. 7 hệ thống sinh cho tiếng Việt; tiếng nói thật từ VIVOS |
| [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) | Duyệt tự động | 34,9 GB | Fine-tune dự phòng | Thư mục `fake/vi`: LEMAS-TTS, OmniVoice, VieNeu-TTS, VoxCPM2, vixTTS. Chỉ có tiếng nói giả |
| [VIVOS](https://huggingface.co/datasets/AILAB-VNUHCM/vivos) | Mở, CC BY-NC-SA 4.0 | 1,47 GB | Tiếng nói thật cho bản dự phòng | 65 người nói |
| Tập tự tạo | Nhóm tự làm | Nhỏ | Kiểm tra lối tắt | Cùng người nói, cùng câu, chỉ dùng giọng thành viên có đồng ý |

**Không dùng làm dữ liệu chính**

| Bộ dữ liệu | Lý do |
|---|---|
| ASVspoof 2019/2021/5 | AntiDeepfake đã dùng để huấn luyện, nên không hợp lệ làm tập kiểm tra |
| [VSASV](https://huggingface.co/datasets/hustep-lab/VSASV-Dataset) | Chỉ có replay, voice conversion (RVC) và tấn công đối kháng, không có TTS. Nhóm tác giả cũng ghi bản công bố không đầy đủ như trong bài |
| VietPrism | Bài không có link tải, không tìm thấy trên Hugging Face |
| [Bộ cộng đồng vietkemmai](https://huggingface.co/datasets/vietkemmai/Data_voice_AI_human_scam) | Không có nhãn hệ thống sinh, nguồn gốc không rõ. Chỉ dùng để thử demo |

### Tài nguyên tính toán

GPU giả định là bản miễn phí của Kaggle hoặc Colab, khoảng 16 GB (T4 hoặc P100). Máy M1 Pro 32 GB dùng để đọc code, xử lý dữ liệu và chạy các bước trên CPU.

Fine-tune toàn bộ mô hình với AdamW cần khoảng 16 byte cho mỗi tham số (trọng số, gradient, hai trạng thái của Adam), chưa tính activation. Bảng dưới là ước lượng theo cách tính này, cần đo thực tế ở GĐ4.

| Backbone | Tham số | Bộ nhớ fine-tune | Inference | Vai trò trong đồ án |
|---|---|---|---|---|
| XLS-R-2B | 2,2B | ~35 GB | Được (fp16 khoảng 4,4 GB) | Chỉ zero-shot |
| XLS-R-1B, MMS-1B, HuBERT-XL | ~965M | ~15,4 GB, vượt 16 GB khi tính activation | Được | Chỉ zero-shot |
| MMS-300M, W2V-Large | 317M | ~5,1 GB | Được | Fine-tune; MMS-300M là cấu hình mốc |
| W2V-Small | 95M | ~1,5 GB | Được | Fine-tune |

GRPO giữ thêm hai bản mô hình chỉ chạy inference (mô hình tham chiếu và chính sách cũ), khoảng 2,5 GB với 317M tham số.

**Phương án theo tầng**

| Tầng | Cần gì | Nội dung | Đáp ứng |
|---|---|---|---|
| **0. Chắc chắn làm được** | Inference trên GPU, phần còn lại trên CPU | Chấm zero-shot; trích embedding bằng chế độ `analysis` của repo, rồi huấn luyện bộ phân loại nhẹ trên M1 để so post-training, backbone, Default/NDA | Demo và một phần ablation, không có GRPO |
| **1. Mục tiêu** | GPU 16 GB | Fine-tune MMS-300M, W2V-Large, W2V-Small; so SFT với GRPO | Đủ ablation và cải tiến |
| 2. Mở rộng | GPU từ 40 GB (Colab Pro) | Fine-tune XLS-R-1B | Không bắt buộc |

**Quota:** Colab miễn phí (kiểm tra 13/11): 0 compute unit, mỗi phiên T4 tối đa khoảng 4 giờ, không bảo đảm được cấp GPU. Kaggle (kiểm tra 13/11): 30 giờ GPU mỗi tuần, làm mới 7 giờ sáng thứ Bảy (giờ VN), máy T4 × 2, phiên tối đa 12 giờ. Mỗi thành viên dùng tài khoản riêng của mình.

**Hệ quả:** việc dài phải chạy được qua nhiều phiên: lưu file điểm lên Drive sau mỗi checkpoint (GĐ3), đặt thư mục thí nghiệm trên Drive để huấn luyện chạy tiếp từ checkpoint cuối (GĐ5–6). Checkpoint lớn không lưu trên Drive (15 GB): tải vào máy Colab mỗi phiên, dùng xong xoá.

**Bằng chứng mỗi lần chạy (bắt buộc, áp dụng mọi giai đoạn):** mọi lần chạy GPU, kể cả lần lỗi, phải lưu vào git ở `results/runs/<kernel>-v<N>/` (Kaggle) hoặc `results/runs/colab-<tên>-<YYYYMMDD>/` (Colab): code đã chạy, log, môi trường, kết quả, `RUN.md`. Dùng cho báo cáo, vấn đáp và tái hiện. Cách làm: `chore/Quy-Trinh-Chay-Kaggle.md` mục "Lưu bằng chứng"; mục lục: `results/runs/README.md`.

## 3. Danh sách đọc

Yêu cầu "ít nhất 5 bài" chỉ áp dụng cho tiêu chí tổng quan (nhóm A). Hai nhóm còn lại phục vụ việc thực hiện đồ án và trả lời vấn đáp, không tính vào 5 bài. Bài GRPO thuộc cả nhóm A và nhóm B.

| Nhóm | Mục đích | Số bài | Mức đọc |
|---|---|---|---|
| **A. Tổng quan** | Đáp ứng tiêu chí 30%: tóm tắt, phân tích ưu và nhược điểm | 5 | Toàn văn, viết thẻ |
| **B. Hiểu giải pháp** | Chạy được code, trả lời câu vấn đáp "kiến trúc bên dưới" | 6 | Đọc phần cần dùng, kèm code |
| **C. Đọc lướt** | Trả lời câu vấn đáp "khoảng 3 hướng chính", hiểu bối cảnh, dữ liệu và baseline | 7 | Tóm tắt, hình và bảng chính |
| D. Danh mục | Biết tên khi bị hỏi | Phụ lục | Không cần đọc |

### Nhóm A. 5 bài tính cho tiêu chí tổng quan

Tiêu chí chọn: mỗi bài đại diện một cách tiếp cận khác nhau, công bố từ 2024, liên quan đến đề tài, hạng tra trực tiếp trên CORE 2023 (riêng CodecFake+ là Q1 theo nguồn thứ cấp).

| Bài | Cách tiếp cận | Nơi công bố, hạng | Đọc ở | Câu hỏi cần trả lời khi đọc |
|---|---|---|---|---|
| [ALLM4ADD](https://arxiv.org/abs/2505.11079) | Audio LLM | ACM MM 2025, A* | GĐ1, ngày 1 | Bài toán được chuyển thành hỏi đáp ra sao? Mở băng encoder làm kết quả đổi thế nào (Bảng 4)? |
| [CodecFake+](https://arxiv.org/abs/2501.08238) | Dữ liệu cho TTS dựa trên codec | TASLP 2026, Q1 | GĐ1, ngày 2 | CoRS được dùng làm dữ liệu huấn luyện ra sao? Loại codec nào cho kết quả tốt nhất? |
| [VSASV](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html) | Dữ liệu tiếng Việt | Interspeech 2024, A | GĐ1, ngày 2 | Dữ liệu được tạo bằng những tấn công nào? Vì sao không dùng được cho bài toán TTS? |
| [GRPO](https://arxiv.org/abs/2603.02914) | Chiến lược fine-tune không quên | Interspeech 2026, A | GĐ1, ngày 3 | Vì sao SFT làm giảm kết quả ngoài miền? Bảng 1 nói gì về MMS-300M? |
| [XLSR-SLS](https://doi.org/10.1145/3664647.3681345) | Kiến trúc SSL + back-end | ACM MM 2024, A* | GĐ1, ngày 3 | Vì sao lấy đặc trưng từ nhiều lớp? |

**Mẫu thẻ (một trang mỗi bài):** vấn đề, ý tưởng, dữ liệu, một con số chính, ưu điểm, nhược điểm, liên hệ với đề tài.

### Nhóm B. Hiểu giải pháp

| Bài hoặc code | Đọc ở | Cần nắm |
|---|---|---|
| [wav2vec 2.0](https://arxiv.org/abs/2006.11477) | GĐ2 | Encoder CNN, học tương phản, che đặc trưng |
| [XLS-R](https://arxiv.org/abs/2111.09296) | GĐ2 | 436K giờ, 128 ngôn ngữ, trong đó tiếng Việt chỉ có 131 giờ |
| [MMS](https://arxiv.org/abs/2305.13516) | GĐ2 | Khác XLS-R ở phạm vi ngôn ngữ và cách pre-training |
| [AntiDeepfake](https://arxiv.org/abs/2506.21090) | GĐ3 | Định nghĩa post-training, dữ liệu, Bảng II và III |
| [RawBoost](https://arxiv.org/abs/2111.04433) | GĐ3 | 3 nhóm nhiễu; `algo=5` (nối tiếp thuật toán 1 và 2) là cấu hình repo đang dùng |
| [GRPO](https://arxiv.org/abs/2603.02914), đọc sâu | GĐ5, trong lúc chờ GPU | Công thức loss, các thí nghiệm ablation `beta` và phần thưởng âm |
| Code AntiDeepfake | GĐ2, GĐ3, GĐ5 | `models/W2V.py`, `models/W2V_configs.py`, `protocols/README.md`, `algo/rl.py` |

### Nhóm C. Đọc lướt

| Bài | Đọc ở | Dùng để |
|---|---|---|
| [PartialSpoof](https://arxiv.org/abs/2204.05177) | GĐ1, ngày 1 | Trình bày hướng giả mạo một phần trong câu vấn đáp 1 |
| [Phân loại theo cấu tạo codec](https://www.isca-archive.org/interspeech_2025/chen25j_interspeech.html) | GĐ1, ngày 2 | Hiểu hướng truy vết nguồn |
| [VietPrism](https://arxiv.org/abs/2609.30005) | GĐ1, ngày 2 | Tình hình tiếng Việt và khoảng trống của đề tài |
| [Speech DF Arena](https://arxiv.org/abs/2509.02859) | GĐ1, ngày 3 | Xếp hạng các detector hiện nay |
| [RADAR 2026](https://arxiv.org/abs/2605.09568) | GĐ1, ngày 3 | Bối cảnh 2026, biến đổi âm thanh, tập kiểm tra tiếng Việt |
| [Tak 2022](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.html) | GĐ1, ngày 3 (cuối cùng) | Baseline XLS-R + AASIST; dẫn sang phần backbone SSL ở GĐ2 |
| [SEA-Spoof](https://arxiv.org/abs/2509.19865) | GĐ4 | Phần mô tả dữ liệu tiếng Việt và các hệ thống sinh |

## 4. Mạch kiến thức

Mỗi giai đoạn trả lời một câu hỏi, và câu trả lời là đầu vào của giai đoạn sau.

| GĐ | Số ngày | Câu hỏi cần trả lời | Đọc (mục 3) | Dùng kết quả từ |
|---|---|---|---|---|
| 1. Bản đồ các hướng | 3 | Thế giới đang làm gì, vì sao chọn hướng này? | Nhóm A, phần lớn nhóm C | Không |
| 2. Backbone SSL và môi trường | 2 | Backbone hoạt động thế nào, code có chạy được không? | wav2vec 2.0, XLS-R, MMS | Bài Tak ở GĐ1 |
| 3. Post-training và zero-shot | 2 | Post-training là gì, mô hình có sẵn làm tốt đến đâu trên tiếng Việt? | AntiDeepfake, RawBoost | Môi trường và script chấm ở GĐ2 |
| 4. Dữ liệu tiếng Việt và pipeline | 2 | Fine-tune trên tiếng Việt như thế nào cho đúng? | SEA-Spoof | Kết quả zero-shot ở GĐ3 |
| 5. Ablation | 4 | Yếu tố nào quyết định kết quả trên tiếng Việt? | GRPO (đọc sâu) | Pipeline ở GĐ4 |
| 6. GRPO và cải tiến | 3 | Có giữ được khả năng tổng quát hoá khi fine-tune không? | | R và kết quả ablation ở GĐ5 |
| 7. Đóng gói | 3 | Trình bày kết quả và lập luận ra sao? | | Toàn bộ kết quả |
| 8. Tập vấn đáp | 1 | Có trả lời được không cần nhìn tài liệu không? | | Toàn bộ ghi chú |
| Dự phòng | 1 | Sửa lỗi cuối, nộp bài | | |

## 5. Việc chạy nền (làm ngay ngày đầu)

- [x] Xin quyền [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) (tác giả duyệt thủ công). **Đã được duyệt, kiểm 14/11.**
  - 14/11: README của SEA-Spoof yêu cầu **gửi email cho tác giả trước khi xin quyền** (wu_jinyang@a-star.edu.sg, imcc.sg@gmail.com), ghi họ tên, đơn vị, mục đích nghiên cứu, cam kết phi thương mại. Cần kiểm tra yêu cầu đã được duyệt chưa; nếu chưa, gửi email.
- [x] Xin quyền [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) (duyệt tự động), phòng trường hợp SEA-Spoof chưa được duyệt. **Đã có quyền (kiểm 14/11):** tải được file trong repo bằng tài khoản HF trên máy; 534.539 file, `fake/vi` có 5.000 file WAV (5 hệ thống × 1.000) và 5 file `meta.csv`; giấy phép CC BY-NC 4.0 (`NOTICE_NO_COMMERCIAL_USE.txt`).
- [x] ~~Xin quyền [F5-TTS Vietnamese](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice) bằng email trường, vì email cá nhân bị từ chối.~~ **Không cần nữa (kiểm 14/11):** API HF trả `gated: False`, `model_last.pt` tải được không cần đăng nhập. Giấy phép CC-BY-NC-SA-4.0; README cấm nhái giọng khi không có đồng ý. Việc còn lại: thành viên đồng ý bằng văn bản và thu ~100 câu mỗi người.
- [x] Tải [RADAR 2026](https://zenodo.org/records/22110815) và [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) vào nơi Colab/Kaggle đọc được: output kernel Kaggle `orpheus-prep` (13/11, ~10 GB).
- [x] Kiểm tra quota GPU: Kaggle 30 giờ/tuần (T4 × 2, 4 nhân CPU, phiên 12 giờ); Colab miễn phí ~4 giờ/phiên. Máy M1 Pro dùng để đọc code, xử lý dữ liệu và chạy tầng 0. (Tài khoản các thành viên khác: chưa kiểm tra.)

## 6. Chi tiết từng giai đoạn

### GĐ1. Bản đồ các hướng (3 ngày)

Triển khai **hướng 1** (post-training và fine-tune mô hình SSL) để nghiên cứu sâu. Các hướng khác vẫn là xu thế nhưng chưa nghiên cứu sâu trong đồ án này, chỉ tìm hiểu vấn đề, ý tưởng và hạn chế.

Thứ tự đọc: các hướng khác trước để có bức tranh chung, kết thúc bằng bài Tak vì bài này dẫn trực tiếp sang phần tìm hiểu backbone SSL ở GĐ2. Câu hỏi khi đọc từng bài nằm ở mục 3.

| Ngày | Nhóm A (toàn văn, viết thẻ) | Nhóm C (đọc lướt) |
|---|---|---|
| 1 | ALLM4ADD | PartialSpoof |
| 2 | CodecFake+, VSASV | Phân loại theo cấu tạo codec, VietPrism |
| 3 | GRPO, XLSR-SLS | Speech DF Arena, RADAR 2026, Tak 2022 (cuối cùng) |

**Đầu ra**
- 5 thẻ tóm tắt cho nhóm A.
- Bảng so sánh các hướng theo: đầu vào, đầu ra, dữ liệu cần có, chi phí, hạn chế. Dùng phụ lục để bổ sung tên các mô hình trong mỗi hướng.
- Bản nháp câu trả lời vấn đáp 1 và 2.

**Điều kiện qua giai đoạn:** nói được mỗi hướng trong 1 phút mà không nhìn tài liệu.

### GĐ2. Backbone SSL và môi trường (2 ngày)

**Ngày 1: Đọc nhóm B (wav2vec 2.0, XLS-R, MMS) và lần theo code**

Lần theo `models/W2V.py` và `models/W2V_configs.py` của [AntiDeepfake](https://github.com/nii-yamagishilab/AntiDeepfake):

| Thứ tự | Khối trong code | Cần hiểu |
|---|---|---|
| 1 | `SSLModel.extract_feat` | Encoder CNN và Transformer biến dạng sóng thành chuỗi vector |
| 2 | `adap_pool1d` | Lấy trung bình theo thời gian thành một vector |
| 3 | `proj_fc` | Một lớp fully connected cho ra 2 lớp `<Fake, Real>` |
| 4 | `global_configs` | So sánh các backbone theo số lớp và số chiều |

| Backbone | Số lớp Transformer | Số chiều | Pre-training |
|---|---|---|---|
| `w2v_small` | 12 | 768 | Tiếng Anh |
| `w2v_large` | 24 | 1024 | Tiếng Anh |
| `mms_300m` | 24 | 1024 | Đa ngôn ngữ |
| `mms_1b`, `xlsr_1b` | 48 | 1280 | Đa ngôn ngữ |
| `xlsr_2b` | 48 | 1920 | Đa ngôn ngữ |

**Ngày 2: Kiểm chứng code chạy được (bắt buộc)**
- Dựng môi trường theo `install.sh`: Python 3.9, torch 2.6.0, fairseq commit `862efab`, SpeechBrain 1.0.2. Colab mặc định dùng Python mới hơn, nên cần conda hoặc micromamba.
- Giảm `num_workers` từ 16 xuống mức máy có.
- Chạy `protocols/toy_example.py` và inference với `mms_300m.ckpt`. Lệnh mẫu trong README thiếu dấu `\` ở dòng `--test_csv`.
- Viết `evaluate.py`: tính EER, minDCF, actDCF từ file điểm, vì `evaluation.py` của repo không có DCF. Quy ước điểm cao là thật. RADAR quy ước ngược lại, cần đảo khi đọc.

**Điều kiện qua giai đoạn:** toy example cho EER đúng 6,11% như README, và log không có cảnh báo nạp trọng số. Tập toy chỉ có 16 câu thật, nên lệch 1 câu là EER nhảy khoảng 3 điểm; "gần" không đủ để kết luận. **Nếu không dựng được môi trường sau 1 ngày, dừng lại và đánh giá lại hướng**, có thể dùng các wrapper trên `SpeechAntiSpoofingBenchmarks` cho phần inference.

### GĐ3. Post-training và zero-shot (2 ngày)

**Ngày 1: Đọc nhóm B (AntiDeepfake, RawBoost) và `protocols/README.md`**, để biết dữ liệu nào không được dùng làm tập kiểm tra.

**Ngày 2: Chạy zero-shot**, chỉ inference, trên Kaggle (chi tiết ở `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 4–5):
- Chấm 14 checkpoint AntiDeepfake và W2V2-AASIST trên RADAR tiếng Việt (13.438 câu, 17,04 giờ).
- In-the-Wild dùng số đã công bố trong bài; chỉ chạy lại MMS-300M bản Default (và W2V2-AASIST) để đối chiếu.
- Chấm MMS-300M, W2V-Large, W2V-Small chưa fine-tune trên tập `ood_forget` (1.000 + 1.000 câu In-the-Wild, 1.000 + 1.000 câu RADAR **eval** tiếng Anh), làm mốc đo mức quên ở GĐ5. Không dùng RADAR dev: tiếng thật lấy từ LibriTTS, có trong dữ liệu post-training.
- Lưu embedding bằng chế độ `analysis` (cho điểm và embedding trong một lượt) để dùng cho tầng 0 nếu cần.

**Đầu ra:** bảng EER theo backbone và Default/NDA trên RADAR tiếng Việt.

**Điều kiện qua giai đoạn:** In-the-Wild của MMS-300M bản Default gần 2,90%. Repo đã sửa cách tính EER vào 16/10/2025, nên đối chiếu với bản mới nhất của bài.

**Trạng thái (14/11): qua phần chạy.** In-the-Wild MMS-300M Default 2,9016%. Đủ 14 checkpoint và baseline trên RADAR tiếng Việt (tốt nhất XLS-R-2B 13,71%), kèm khoảng tin cậy và phân tích thêm. Đọc kỹ ngày 1 hoãn lại. Kết quả, phát hiện và quyết định: `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 3, 5.6–5.7, 8.

### GĐ4. Dữ liệu tiếng Việt và pipeline (2 ngày)

**Ngày 1: Dữ liệu** (đọc lướt phần dữ liệu của SEA-Spoof)
- Đọc SEA-Spoof dạng streaming, lọc phần `vi` và chỉ lưu phần này, thay vì tải cả 81,5 GB. Tạo CSV theo `protocols/toy_example.py`.
- Chọn tập fine-tune khoảng 10–20 giờ.
- Chia train, dev, test không trùng người nói. Với mỗi lần thử, để riêng một hệ thống sinh làm tập chưa gặp.
- Nếu SEA-Spoof chưa được duyệt: dùng `fake/vi` của MLAAD và VIVOS. Tiếng nói thật và giả khi đó đến từ hai kho khác nhau, nên phải kiểm tra lối tắt và ghi rõ hạn chế.
- Tập tự tạo: mỗi thành viên đọc khoảng 100 câu, F5-TTS Vietnamese nhái chính giọng đó với cùng câu.

Thông tin SEA-Spoof đã kiểm (README công khai, 14/11): 553.746 câu, tiếng Việt 81.020 câu (14,63%), **trộn lẫn mọi ngôn ngữ trong 37 file parquet** (train 30 file, 64,7 GB; validation 3; evaluation 4), FLAC 16 kHz nhúng trong file. Có các cột `language`, `label`, `spoof_type`, `category`, `split`, `source_model`, `source_dataset`, `speaker_or_voice`. Cách lấy: đọc trước các cột metadata (rất nhẹ) qua HTTP Range, chọn câu, rồi chỉ tải âm thanh của phần cần, như đã làm với In-the-Wild ở GĐ3. Chi tiết: `chore/GD4-Du-Lieu-Va-Pipeline.md`.

**Ngày 2: Pipeline**, xuất phát từ `hparams/ft_grpo/mms_300m_pre-post-sft.yaml`:
- [ ] Đo bộ nhớ GPU thực tế với MMS-300M. Nếu vượt, giảm `max_batch_length` (mặc định 100 giây âm thanh mỗi batch).
- [ ] Chỉnh `warmup` theo số bước thực tế. Cấu hình gốc đặt `warmup: 80000` trong khi `optimizer_step_limit: 50000`, nên với tập nhỏ learning rate gần như không tăng tới giá trị đặt.
- [ ] Thử overfit một batch nhỏ để xác nhận pipeline học được.
- [ ] Đo thời gian một epoch và một lần chấm, rồi tính: `Tổng giờ GPU = số lần chạy × (số epoch × giờ mỗi epoch + giờ chấm điểm)`. Chấm đã đo ở GĐ3 (một T4, bản 317M): `radar_vi` 24–38 phút, `ood_forget` 7–11 phút.
- [ ] Đo tốc độ đọc dữ liệu khi huấn luyện: máy Kaggle T4 × 2 chỉ có 4 nhân CPU, và ở GĐ3 bước đọc dữ liệu có dấu hiệu là nút thắt.
- [ ] Chạy hai cấu hình song song trên hai GPU trong một phiên (như GĐ3) để quota tính một lần cho hai lần chạy, nếu bộ nhớ và CPU cho phép.

**Điều kiện qua giai đoạn:** ngân sách vừa với quota. Nếu MMS-300M vẫn không vừa sau khi giảm batch, chuyển cấu hình mốc sang W2V-Small. Nếu fine-tune không chạy được, chuyển sang tầng 0 (mục 2).

**Trạng thái (14/11): qua.** SEA-Spoof chia lại theo người nói (split có sẵn trùng cả 65 người nói VIVOS); tiếng thật có cả Common Voice nên Common Voice chỉ dùng kiểm tra; F5-TTS và MiniMax để riêng; train 17,88 giờ. Pilot trên T4: batch 50 giây (100 hết bộ nhớ), ~49 phút mỗi epoch. Đã chốt: 3 epoch, learning rate đỉnh 1e-6 với warmup 10%, chấm mô hình sau epoch cuối; ngân sách GĐ5–6 ~36 giờ. Chi tiết, phát hiện 1–15 và quyết định: `chore/GD4-Du-Lieu-Va-Pipeline.md` mục 2.2, 3, 4.1–4.2, 5, 7.

### GĐ5. Ablation (4 ngày)

**Trạng thái (15/11): qua.** 10 lần huấn luyện (R × 3 seed, A0, P0, B2, B1, N0), 17,65 giờ quota Kaggle. RADAR VI: R 24,02 ± 1,52%, A0 18,37% (tốt nhất), P0 31,46%, B1 34,67%; SFT làm quên trên ItW với MMS-300M, không với W2V-Large. Lịch learning rate thực tế lệch E1 của GĐ4 (E1'). Hệ thống để riêng SEA-Spoof và tập MLAAD thêm vào đều bão hoà; nút thắt là tiếng thật nguồn khác. Kết quả, phát hiện 1–9, quyết định chờ chốt F1–F3: `chore/GD5-Ablation.md` mục 3, 4, 8.

Đổi một yếu tố mỗi lần. Fine-tune trên tiếng Việt. R chạy 3 seed; các cấu hình khác chạy 1 seed trước, thêm seed nếu còn quota.

| Mã | Backbone | Khởi tạo | Fine-tune | RawBoost | So với | Câu hỏi trả lời |
|---|---|---|---|---|---|---|
| **R** | MMS-300M (đa ngôn ngữ, 317M), **bản Default** | Post-trained | SFT | Có | | Mốc so sánh |
| S0 | MMS-300M bản Default | Post-trained | Không | | R | Fine-tune có cần thiết? (lấy từ GĐ3: RADAR tiếng Việt 20,08%) |
| P0 | MMS-300M | Fairseq gốc (`base_300m.pt`) | SFT | Có | R | Post-training đóng góp bao nhiêu? |
| A0 | MMS-300M | Post-trained | SFT | Không | R | RawBoost giúp bao nhiêu? |
| B2 | W2V-Large (tiếng Anh, 317M) | Post-trained | SFT | Có | R | Ngôn ngữ pre-training (cùng kích thước, cùng 24 lớp, 1024 chiều) |
| B1 | W2V-Small (tiếng Anh, 95M) | Post-trained | SFT | Có | B2 | Kích thước backbone (cùng ngôn ngữ). Không sạch: W2V-Small còn khác ở dropout, layerdrop, post-LN, gradient CNN × 0,1 |
| N0 | MMS-300M **bản NDA** | Post-trained, không RawBoost | SFT | Có | R | RawBoost khi post-train: lợi thế ~2 điểm của bản NDA khi dùng thẳng (GĐ3) còn không sau fine-tune? *Thêm 14/11, 1 seed, không bắt buộc* |

Vì sao giữ R là bản Default dù bản NDA tốt hơn khi dùng thẳng: để so được với bài GRPO (bài báo MMS-300M dùng thẳng trên In-the-Wild 2,91%, gần bản Default 2,90 hơn bản NDA 2,00). A0 (bỏ RawBoost **khi fine-tune**) khác N0 (bỏ RawBoost **khi post-train**).

**Dòng tham chiếu trong mọi bảng kết quả:** XLS-R-2B dùng thẳng (RADAR tiếng Việt 13,71%), để biết fine-tune mô hình 317M có thắng mô hình 2,2B không fine-tune không.

**Tập đánh giá cho mọi cấu hình**

| Nhóm | Tập | Đo điều gì |
|---|---|---|
| Trong miền | Test SEA-Spoof `vi` | Khả năng học tiếng Việt. Dễ bão hòa gần 0% (SEA-Spoof báo 0,01% sau fine-tune), nên không dùng làm thước đo chính |
| Hệ thống sinh chưa gặp | Hệ thống được để riêng | Tổng quát hoá trong tiếng Việt |
| Ngoài miền tiếng Việt | RADAR tiếng Việt | TTS khác và biến đổi âm thanh |
| Ngoài miền khác | Tập con cân bằng của In-the-Wild và RADAR tiếng Anh | Mô hình có quên kiến thức cũ không |
| Kiểm tra lối tắt | Tập tự tạo | Có học đặc điểm tình cờ của kho dữ liệu không |

**Thước đo chính:** hệ thống sinh được để riêng, RADAR tiếng Việt và mức quên.

**Báo thêm (từ GĐ3):** tỉ lệ giả bị nhận là thật và thật bị coi là giả ở ngưỡng log-odds 0 (không chỉ EER), actDCF − minDCF, và kết quả trên tập 493 câu khó chung `results/zeroshot/analysis/hard_ids_default7.csv`. Dùng `src/bootstrap_eer.py` cho khoảng tin cậy.

RADAR có ElevenLabs và OpenAI, gần với các hệ thống trong SEA-Spoof, nên RADAR tiếng Việt không hoàn toàn là hệ thống chưa gặp.

**Yếu tố gây nhiễu cần phân tích** (sửa 14/11: tiếng thật tiếng Việt của SEA-Spoof gồm cả VIVOS và Common Voice; theo quyết định D1 ở GĐ4, train chỉ dùng VIVOS, Common Voice chỉ ở tập `test_cvreal`): tiếng nói thật dùng để fine-tune đến từ VIVOS, còn của RADAR tiếng Việt đến từ Common Voice và FOSD. Fine-tune có thể khiến mô hình coi tiếng thật từ nguồn khác là giả. Không thêm Common Voice tiếng Việt vào tập huấn luyện để khắc phục, vì RADAR lấy tiếng thật từ chính kho này. Hiện tượng này được ghi nhận như một kết quả.

**Cách chạy**
- Ngày 1–2: seed 1 cho cả 5 cấu hình cần huấn luyện (R, P0, A0, B2, B1); N0 nếu còn quota.
- Ngày 3–4: seed 2, 3 cho R; nếu còn quota, thêm seed cho P0 rồi B2.

**Việc làm trong lúc chờ GPU:** đọc sâu bài GRPO và `algo/rl.py` (nhóm B) để chuẩn bị cho GĐ6.

**Đầu ra:** bảng EER, minDCF (trung bình, độ lệch chuẩn), kèm một câu giải thích nguyên nhân cho từng kết quả. Phân tích riêng theo từng hệ thống sinh, thay vì chỉ báo một con số chung.

**Điều kiện qua giai đoạn:** có đủ seed 1 của 5 cấu hình, và R có đủ 3 seed.

### GĐ6. GRPO và cải tiến (3 ngày)

Lần theo `grpo3_loss` trong `algo/rl.py`: lấy mẫu nhãn từ xác suất dự đoán, phần thưởng 1 khi đúng và 0 khi sai, chuẩn hóa phần thưởng theo nhóm, giới hạn tỉ số chính sách bằng `epsilon`, và phạt lệch khỏi mô hình tham chiếu bằng `beta`.

| Mã | Thay đổi so với R (MMS-300M, SFT) | File cấu hình | Câu hỏi trả lời |
|---|---|---|---|
| G1 | GRPO đầy đủ (`beta` 0,04, `epsilon` 0,2), 3 seed | `mms_300m_pre-post-grpo_v1.yaml` | GRPO có giữ khả năng tổng quát hoá tốt hơn SFT? |
| G2 | GRPO đơn giản, không clip | `mms_300m_pre-post-grpo_v2.yaml` | Có cần clip và mô hình chính sách cũ? |
| G3 | GRPO với `beta` = 0 | Sửa cấu hình G1 | Phạt lệch khỏi mô hình tham chiếu có vai trò gì? |

| Ngày | Việc cần làm | Kết quả |
|---|---|---|
| 1 | Chạy G1, G2, G3 | Bảng SFT so với GRPO trên các tập đánh giá |
| 2 | So SFT và G1 với N = 50, 200, 1000 mẫu mỗi lớp | Biểu đồ số mẫu theo EER trên tập trong miền và tập chưa gặp |
| 3 | Hiệu chuẩn điểm bằng log-odds, học hệ số trên dev. Phân tích các trường hợp EER cực đoan theo hệ thống sinh, như VietPrism đã ghi nhận | actDCF trước và sau; bản nháp câu vấn đáp 4 |

**Trạng thái (15/11): đang chạy.** Thử đo trên T4 (`orpheus-finetune` v5, 0,47 giờ): G1 vừa batch 50 như R khi bật `expandable_segments` (đỉnh 13,9 / ~14,7 GB), chậm 1,55 lần SFT; kernel GRPO ~5,8 giờ. Đã chốt: Kaggle một tài khoản; R và G1 mỗi cấu hình 5 seed (thêm 2 seed để đủ độ tin cậy thống kê, P1); không thêm G1 không RawBoost; giữ ngày 2. Chi tiết: `chore/GD6-GRPO.md` mục 3, 5, 6. **Thêm 15/11:** người dùng thuê server vast.ai để chạy trọn một bộ trên cùng phần cứng, song song Kaggle: GĐ5 và GĐ6 làm lại với 3 seed mọi cấu hình, cả ngày 2 (ít mẫu, cùng 4.563 bước như bản đủ); bộ server là bộ chính, bộ Kaggle là lần lặp độc lập (`chore/GD6-GRPO.md` mục 8, 9; `server/README.md`).

*Lưu ý từ GĐ5 (15/11, chi tiết `chore/GD5-Ablation.md` mục 9):* G1 so với cả R và A0 (F2); thước đo chính RADAR VI, ItW, RADAR EN kèm FRR ở ngưỡng 0 (F3; SEA-Spoof và MLAAD bão hoà); bộ nhớ GRPO trên T4 chưa đo, không được đổi batch mà không giữ số bước; giữ lịch learning rate E1'; sau SFT actDCF − minDCF ≈ 0,2 nên hiệu chuẩn ở ngày 3 có ý nghĩa; ngày 2 (ít mẫu) cân nhắc đo ngoài miền hoặc cắt.

*Lưu ý từ GĐ3:* mô hình chưa fine-tune đã gần hiệu chuẩn tốt theo DCF (actDCF − minDCF ≤ 0,018 trên RADAR tiếng Việt). Hiệu chuẩn chỉ có ý nghĩa nếu fine-tune làm lệch; nếu không lệch thì báo đúng như vậy và dành thời gian cho phân tích theo hệ thống sinh.

actDCF dùng tham số ASVspoof 5 (β ≈ 1,9). Tập fine-tune và tập kiểm tra không được trùng người nói.

**Lưu ý khi diễn giải:** trong bài GRPO, MMS-300M trên In-the-Wild có EER 2,91% khi không fine-tune, 14,17% với SFT và 5,91% với GRPO. Tuy vậy, trung bình ngoài miền của GRPO (3,78%) vẫn hơi kém không fine-tune (3,68%), và GRPO không giúp mô hình chưa post-train. Kết quả ngược giả thuyết vẫn là kết quả hợp lệ.

**Điều kiện qua giai đoạn:** có kết luận về SFT và GRPO trên tiếng Việt, kèm số liệu trước và sau.

### GĐ7. Đóng gói (3 ngày)

| Ngày | Việc cần làm |
|---|---|
| 1 | Demo Gradio trên Colab (cần fairseq): tải âm thanh lên, trả điểm và kết luận. Viết tổng quan và phương pháp |
| 2 | Viết thí nghiệm, kết quả, phân công; làm slide theo mẫu trang đầu trong đề cương |
| 3 | Quay video 10–15 phút. Viết README rồi thử chạy lại từ đầu trên môi trường sạch |

Phần tổng quan dùng 5 thẻ của nhóm A. Hạng CORE đã tra trực tiếp; hạng Q1 của TASLP cần tra lại trên SCImago trước khi nộp. Bài Tak (Odyssey), RawBoost (ICASSP, CORE B) và các preprint vẫn trích dẫn được nhưng không tính vào 5 bài.

Dữ liệu và checkpoint có giới hạn giấy phép, nên folder nộp chỉ kèm hướng dẫn tải, không kèm bản sao.

### GĐ8. Tập vấn đáp (1 ngày)

Hỏi chéo trong nhóm theo mục 7, sau đó ôn thêm nội dung môn học. Đề cương ghi GV có thể hỏi cả phần này.

## 7. Chuẩn bị cho 4 câu hỏi vấn đáp

| Câu hỏi | Xây dựng ở | Phải nói được |
|---|---|---|
| Khoảng 3 hướng chính trên thế giới | GĐ1 (nhóm A, C), phụ lục | Trình bày 3 hướng cụ thể: post-training và fine-tune (hướng 1), SSL kết hợp back-end chuyên biệt (hướng 2), giả mạo một phần (hướng 3). Sau đó kể thêm các hướng còn lại. Đây là lựa chọn trình bày của nhóm, không phải phân loại chuẩn của lĩnh vực |
| Vì sao chọn giải pháp này | GĐ1, GĐ3 | Các bằng chứng ở mục 1, lý do loại hướng khác, khoảng trống cụ thể, tính khả thi với GPU 16 GB |
| Kiến trúc bên dưới | GĐ2, GĐ3, GĐ6 (nhóm B) | Vẽ lại luồng wav2vec 2.0 → trung bình theo thời gian → lớp FC; post-training khác fine-tune ở đâu; công thức GRPO |
| Hướng phát triển tiếp | GĐ6 | Hướng chính dựa trên kết quả thật của nhóm |

**Khoảng trống của đề tài:** VietPrism đã đánh giá zero-shot trên tiếng Việt nhưng không fine-tune và không ablation. Bài GRPO đã so SFT và GRPO nhưng không có tiếng Việt, chỉ fine-tune trên một bộ dữ liệu và chỉ ablation với XLS-R-2B. Phần đồ án bổ sung là phân tích các yếu tố và so SFT với GRPO trên tiếng Việt, với hệ thống sinh được để riêng.

**Cách chọn hướng phát triển theo kết quả**

| Nếu kết quả cho thấy | Hướng tiếp theo |
|---|---|
| SFT làm giảm kết quả ngoài miền, GRPO giữ được | Mở rộng GRPO sang các phương ngữ và code-switching |
| GRPO không giúp trên tiếng Việt | Tìm hiểu cơ chế, thử continual learning (RAWM, RF-Prompt) |
| Kém với hệ thống sinh chưa gặp | Thích nghi ít mẫu (ADD-GP), chọn dữ liệu theo độ đa dạng (DOSS) |
| Giảm mạnh trên RADAR | Tăng cường dữ liệu theo biến đổi thực tế |
| EER tốt nhưng actDCF xấu | Hiệu chuẩn trước khi triển khai |
| Đã ổn định ở mức cả câu | Định vị đoạn giả trong tiếng Việt |

## 8. Đối chiếu tiêu chí chấm và quy tắc cắt phạm vi

| Tiêu chí | Trọng số | Thực hiện ở |
|---|---|---|
| Tổng quan ít nhất 5 bài Q1/Q2 hoặc A/A*, gồm tình hình trong nước | 30% | GĐ1 (5 thẻ của nhóm A) |
| Chạy demo | 20% | GĐ2, GĐ3, GĐ7 |
| Ablation study | 30% | GĐ5 |
| Phân tích sâu, đề xuất cải tiến | 20% | GĐ6 |

**Cắt theo đúng thứ tự,** chỉ chuyển sang mức sau khi mức trước không đủ:

1. Bỏ N0 (thêm 14/11, không bắt buộc).
2. Bỏ G3.
3. Bỏ B1.
4. Fine-tune ít mẫu chỉ làm N = 50, 200.
5. ~~Zero-shot chỉ làm bản Default, không làm NDA.~~ Không còn cần: đã chấm đủ 14 checkpoint (14/11).
6. Giảm tập fine-tune xuống khoảng 10 giờ.
7. Nếu không fine-tune được, chuyển sang tầng 0 (mục 2).

**Không được cắt:** kiểm chứng code ở GĐ2, `evaluate.py`, đánh giá ngoài miền, 3 seed cho R và G1, so sánh SFT với GRPO, 5 thẻ của nhóm A.

## 9. Trạng thái kiểm tra thông tin

| Mức kiểm tra | Nội dung |
|---|---|
| Kiểm tra trực tiếp | Sự tồn tại, dung lượng, giấy phép của các repo, checkpoint và dữ liệu ở mục 2; tên tensor của checkpoint W2V-Small; số câu tiếng Việt trong nhãn RADAR; danh sách hệ thống sinh trong `fake/vi` của MLAAD; hạng CORE 2023 (Interspeech A, ICML A*, ACM MM A*, ICASSP B; Odyssey và APSIPA không có); các file cấu hình SFT và GRPO cho MMS-300M, W2V-Large, W2V-Small |
| Đọc code | `models/W2V.py`, `W2V_configs.py`, `algo/rl.py`, `install.sh`, các file `hparams`, `protocols/README.md` của AntiDeepfake. GĐ2 (13/11, commit `86098091`): thêm `main.py`, `dataio/dataio.py`, `utils.py`, `evaluation.py`, `utils_wandb.py`, `README_grpo.md`, `protocols/toy_example*`; giá trị mặc định `Wav2Vec2Config` của fairseq `862efab` |
| Kiểm tra trực tiếp (13/11, GĐ3) | Nhãn RADAR eval theo ngôn ngữ (VI 7.000 / 6.438; câu dài nhất 45,7 s); nguồn tiếng thật RADAR (Bảng II bài RADAR); RADAR dev lấy tiếng thật từ LibriTTS; meta.csv In-the-Wild (19.963 / 11.816, 16 kHz); 14 checkpoint trên HF là `model.safetensors`, Zenodo là `.ckpt`; model card W2V2-AASIST và cách cắt 4 s trong repo gốc của Tak |
| Kiểm tra trực tiếp (14/11, GĐ3 phân tích) | Phân tích lại 15 file điểm (`src/analyze_zeroshot.py`): hai loại lỗi, actDCF − minDCF, EER theo độ dài, sai trùng nhau, tương quan hạng với Bảng II của bài; khoảng tin cậy bootstrap (`src/bootstrap_eer.py`). README công khai của SEA-Spoof: số câu theo ngôn ngữ, cấu trúc file parquet, các cột metadata, yêu cầu email tác giả |
| Đọc toàn văn | RawBoost (13/11), AntiDeepfake, GRPO, VietPrism, RADAR, wav2vec 2.0 (mục 2–5), XLS-R (bảng ngôn ngữ, mục 3–4, 5.2.5), MMS (mục 3.1, 4), VSASV (phương pháp tấn công), ALLM4ADD (toàn bài, 11/11), PartialSpoof (phần dữ liệu, mô hình, kết quả, 11/11), DFALLM, ASVspoof 5, Speech DF Arena |
| Đọc phần tóm tắt (đã kiểm tra lại 11/11) | BAM, LlamaPartialSpoof, PartialEdit, FT-GRPO, SARA |
| Đọc phần tóm tắt | Các bài còn lại trong mục 3 và phụ lục. Cần đọc toàn văn trước khi trích số liệu |
| Chạy thử | `src/evaluate.py` (self-test 11/11 trên M1). **13/11: dựng môi trường AntiDeepfake trên Colab T4 và chạy toy example với `mms_300m.ckpt`: EER 6,11% khớp README, không cảnh báo nạp trọng số** (`chore/GD2-Backbone-Va-Moi-Truong.md` mục 4.2). **13/11 trên Kaggle (T4 × 2): checkpoint HF giống Zenodo (toy 6,1101%); In-the-Wild đủ bộ MMS-300M Default 2,9016% (Bảng II: 2,90), W2V2-AASIST 11,2122% (card: 11,22); zero-shot RADAR tiếng Việt cho đủ 14 checkpoint và W2V2-AASIST (13–14/11, 3 lần chạy, 10,3 giờ GPU), kèm khoảng tin cậy bootstrap; tốc độ trên T4: bản 300M ~30–45x, 1B ~15–18x, 2B ~8,6x thời gian thực; quota 30 giờ/tuần; máy Kaggle 4 nhân CPU. Bằng chứng từng lần chạy: `results/runs/`** (`chore/GD3-Post-Training-Va-Zero-Shot.md` mục 5.6) |
| **Chưa kiểm tra** | Chưa fine-tune lần nào; bộ nhớ GPU khi fine-tune (bảng bộ nhớ ở mục 2 vẫn là ước lượng). FLEURS trong dữ liệu post-training có tiếng Việt không. Nội dung bên trong SEA-Spoof và MLAAD (cần quyền truy cập). Phiên bản MLAAD mà AntiDeepfake đã dùng có chứa tiếng Việt hay không. Hạng Q trên SCImago của TASLP, IEEE SPL, TIFS (chỉ có từ nguồn thứ cấp). Nơi đăng của Codecfake |

## Phụ lục. Danh mục phương pháp và mô hình (tính đến 2026)

**Không cần đọc.** Phụ lục dùng để tra tên và ý tưởng một dòng của các phương pháp khi làm bảng so sánh ở GĐ1 hoặc khi bị hỏi ở câu vấn đáp 1. Cột "Nhóm đọc" cho biết bài nào đã có trong mục 3; dấu "—" nghĩa là chỉ cần biết tên.

**Hướng 1. Post-training và fine-tune trên mô hình SSL (triển khai)**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [AntiDeepfake](https://arxiv.org/abs/2506.21090) | Post-training: pha có giám sát giữa pre-training và fine-tune, trên khoảng 56.000 giờ thật và 18.000 giờ có artifact, hơn 100 ngôn ngữ | Preprint | B |
| [GRPO cho phát hiện deepfake](https://arxiv.org/abs/2603.02914) | Fine-tune bằng học tăng cường, phạt lệch khỏi mô hình tham chiếu để hạn chế quên | Interspeech 2026 | A, B |
| [DF-Arena 1B](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1) | Huấn luyện trên nhiều bộ dữ liệu; model card không nêu backbone | Model card | — |
| [DOSS](https://arxiv.org/abs/2512.18210) | Chọn hoặc gán trọng số dữ liệu theo độ đa dạng | Preprint | — |
| [RAPTOR](https://arxiv.org/abs/2603.06164) | So sánh backbone SSL nhỏ; quá trình pre-training quan trọng hơn kích thước | Preprint | — |

**Hướng 2. Front-end SSL kết hợp back-end chuyên biệt (dùng làm baseline)**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [XLS-R + AASIST](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.html) | Dùng wav2vec 2.0 (bản XLS-R) làm front-end, fine-tune cùng back-end AASIST | Odyssey 2022 | C |
| [XLSR-SLS](https://doi.org/10.1145/3664647.3681345) | Sensitive Layer Selection: học trọng số cho đặc trưng từ các lớp XLS-R | ACM MM 2024 | A |
| [TCM](https://arxiv.org/abs/2406.17376) | Thêm mô-đun quan hệ thời gian và kênh vào self-attention | Interspeech 2024 | — |
| [Nes2Net](https://arxiv.org/abs/2504.05657) | Bỏ lớp giảm chiều, dùng cấu trúc Res2Net lồng nhau | TIFS (đã nhận đăng theo arXiv) | — |
| [SpAArSIST](https://arxiv.org/abs/2606.11674) | Đơn giản hóa graph pooling và cách gộp nút của AASIST | Interspeech 2026 | — |
| [AASIST3](https://arxiv.org/abs/2408.17352) | Thêm mạng KAN, các lớp và encoder phụ vào AASIST | ASVspoof 2024 | — |
| [XLSR-Mamba](https://arxiv.org/abs/2411.10027) | Mamba hai cột, hai chiều trên đặc trưng wav2vec 2.0 | IEEE SPL (đã nhận đăng theo arXiv) | — |
| AST-ASVspoof5 (trong [VoxENES 2026](https://arxiv.org/abs/2607.11706)) | Transformer trên spectrogram; tốt nhất trong VoxENES với EER 28,98% | Preprint | — |

**Hướng 3. Phát hiện và định vị giả mạo một phần**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [PartialSpoof](https://arxiv.org/abs/2204.05177) | Dữ liệu nhãn đoạn ở 6 độ phân giải, học đồng thời nhãn đoạn và nhãn câu | TASLP 2023 | C |
| [BAM](https://arxiv.org/abs/2407.21611) | Dự đoán ranh giới, rồi dùng ranh giới điều khiển attention giữa các khung | Interspeech 2024 | — |
| [LlamaPartialSpoof](https://arxiv.org/abs/2409.14743) | Khoảng 130 giờ dữ liệu giả toàn phần và một phần, tạo bằng LLM và nhân bản giọng | ICASSP 2025 | — |
| [PartialEdit](https://www.isca-archive.org/interspeech_2025/zhang25g_interspeech.html) | Dữ liệu chỉnh sửa bằng mô hình neural; detector học trên PartialSpoof không phát hiện được | Interspeech 2025 | — |

**Hướng 4. Audio LLM**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [ALLM4ADD](https://arxiv.org/abs/2505.11079) | Đặt bài toán thành hỏi đáp âm thanh và fine-tune có giám sát | ACM MM 2025 | A |
| [DFALLM](https://arxiv.org/abs/2512.08403) | Encoder âm thanh là nút thắt chính, tăng cỡ LLM ít tác dụng | Preprint | — |
| [FT-GRPO](https://arxiv.org/abs/2601.02983) | Fine-tune trên chuỗi lập luận theo tần số và thời gian, rồi học tăng cường | Preprint | — |
| [SARA](https://arxiv.org/abs/2601.03615) | Kiểm tra độ tin cậy lập luận của Audio LLM | Preprint | — |

**Hướng 5. Phát hiện TTS dựa trên neural codec**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [Codecfake](https://arxiv.org/abs/2405.04880) | Hơn 1 triệu mẫu tiếng Anh, tiếng Trung; đề xuất CSAM | Cần xác minh nơi đăng | — |
| [CodecFake+](https://arxiv.org/abs/2501.08238) | Huấn luyện trên CoRS từ 31 codec, đánh giá trên CoSG từ 17 mô hình | TASLP 2026 | A |

**Hướng 6. Truy vết nguồn**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [Phân loại theo cấu tạo codec](https://www.isca-archive.org/interspeech_2025/chen25j_interspeech.html) | Truy vết hệ thống CoSG thông qua phân loại thành phần của codec | Interspeech 2025 | C |

**Hướng 7. Continual learning và thích nghi ít mẫu**

| Mô hình | Ý tưởng chính | Nơi công bố | Nhóm đọc |
|---|---|---|---|
| [RAWM](https://proceedings.mlr.press/v202/zhang23au.html) | Tính hướng cập nhật trọng số theo tỉ lệ thật/giả, thêm chính quy hóa | ICML 2023 | — |
| [RAIS](https://www.isca-archive.org/interspeech_2025/febrinanto25_interspeech.html) | Sinh nhãn phụ để chọn mẫu đa dạng cho bộ đệm ôn tập | Interspeech 2025 | — |
| [ADD-GP](https://arxiv.org/abs/2505.23619) | Bộ phân loại Gaussian Process trên đặc trưng XLS-R, thích nghi từ ít mẫu | Interspeech 2025 | — |
| [RF-Prompt](https://arxiv.org/abs/2609.37586) | Prompt dùng chung cho tiếng nói thật, expert riêng cho từng cơ chế giả mạo | Preprint | — |

**Đánh giá, tiếng Việt và các hướng bổ trợ**

| Nhóm | Tên | Nhóm đọc |
|---|---|---|
| Benchmark, cuộc thi | [Speech DF Arena](https://arxiv.org/abs/2509.02859) (C), [Spoof-SUPERB](https://arxiv.org/abs/2603.01482) (—), [VoxENES 2026](https://arxiv.org/abs/2607.11706) (—), [RADAR 2026](https://arxiv.org/abs/2605.09568) (C) | Theo từng tên |
| Trong nước, tiếng Việt | [VSASV](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html) (A), [VietPrism](https://arxiv.org/abs/2609.30005) (C), [SEA-Spoof](https://arxiv.org/abs/2509.19865) (C) | Theo từng tên |
| Watermark | [AudioSeal](https://proceedings.mlr.press/v235/san-roman24a.html) (ICML 2024) | — |
