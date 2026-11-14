# Đề cương đồ án: Fake Voice Detection

**Môn:** Xử lý tiếng nói. **Chủ đề:** 24. Fake voice detection (GV: T. Đức). **Hạn nộp:** 30/11/2026.

**Đề tài:** Phân tích các yếu tố quyết định khả năng phát hiện deepfake tiếng Việt của mô hình SSL post-trained, và đánh giá fine-tune bằng học tăng cường (GRPO) so với fine-tune có giám sát (SFT).

**Hướng thực hiện:** hướng nghiên cứu, có ablation study.

---

## 1. Đối chiếu với yêu cầu và tiêu chí chấm

| Tiêu chí | Trọng số | Đồ án đáp ứng bằng | Kết quả bàn giao |
|---|---|---|---|
| Tổng quan: ít nhất 5 bài Q1/Q2 hoặc A/A* trong 3–5 năm, tình hình thế giới và trong nước | 30% | 5 hướng nổi bật (mục 2), 5 bài tính điểm (mục 2.6), tình hình tiếng Việt (mục 2.7) | Bảng tổng hợp, thẻ tóm tắt và phân tích ưu, nhược điểm của 5 bài |
| Chạy demo | 20% | Chạy lại các mô hình có sẵn, demo web | Bảng EER zero-shot của 14 checkpoint và baseline trên RADAR tiếng Việt (**đã có**, mục 5.1); demo Gradio |
| Ablation study | 30% | Bỏ hoặc thay từng yếu tố của giải pháp (mục 4.4) | Bảng ablation trên các tập tiếng Việt và ngoài miền |
| Chọn 1 giải pháp, phân tích sâu, đề xuất cải tiến | 20% | So sánh SFT với GRPO trên tiếng Việt (mục 4.5) | Kết luận có số liệu trước và sau |

**Sản phẩm nộp:** báo cáo PDF (có danh sách nhóm và phân công), slide PDF, video 10–15 phút (thuyết trình và demo), thư mục code kèm hướng dẫn chạy lại.

---

## 2. Tình hình nghiên cứu: 5 hướng nổi bật

**Cách xác định hạng**
- **Hội nghị:** tra trực tiếp CORE 2023. Interspeech hạng A; ICML và ACM MM hạng A*; ICASSP hạng B; Odyssey và APSIPA không có trong danh sách.
- **Tạp chí:** SCImago chặn truy cập tự động, nên hạng Q lấy từ các trang tổng hợp. Cần tra lại trước khi nộp.

### 2.1. Hướng 1: Post-training và fine-tune mô hình SSL (hướng được chọn)

Ý tưởng: thêm một pha huấn luyện có giám sát trên dữ liệu rất lớn giữa pre-training và fine-tune, rồi chọn cách fine-tune sao cho mô hình không quên kiến thức cũ.

| Bài | Tác giả | Nơi công bố | Hạng | Tóm tắt |
|---|---|---|---|---|
| Post-training for Deepfake Speech Detection (AntiDeepfake) | W. Ge, X. Wang, X. Liu, J. Yamagishi | arXiv 2025 | Preprint | Post-training 7 backbone SSL trên khoảng 56.000 giờ thật và 18.000 giờ có artifact, hơn 100 ngôn ngữ. Zero-shot trên In-the-Wild: XLS-R-2B đạt EER 1,23% |
| Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection? | X. Wang, W. Ge, J. Yamagishi | Interspeech 2026 | CORE A | SFT làm XLS-R-2B giảm ngoài miền (EER trung bình 2,69% → 6,28%), GRPO giữ ở 2,69%. Hiệu quả không đồng đều giữa các backbone |

**Ưu điểm:** zero-shot rất mạnh, có code và checkpoint công khai. **Nhược điểm:** mô hình lớn, chi phí post-training không tự tái lập được, chưa có kết quả riêng cho tiếng Việt.

### 2.2. Hướng 2: Front-end SSL kết hợp back-end chuyên biệt

Ý tưởng: dùng mô hình tự giám sát (wav2vec 2.0, XLS-R) làm bộ trích đặc trưng, rồi thiết kế back-end riêng để khai thác dấu vết giả mạo.

| Bài | Tác giả | Nơi công bố | Hạng | Tóm tắt |
|---|---|---|---|---|
| Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation | H. Tak, M. Todisco, X. Wang, J. Jung, J. Yamagishi, N. Evans | Odyssey 2022 | Không có trong CORE 2023 | Ghép front-end wav2vec 2.0 với AASIST và RawBoost. Là baseline phổ biến nhất của lĩnh vực |
| Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier | Q. Zhang, S. Wen, T. Hu | ACM MM 2024 | CORE A* | Chọn lọc đặc trưng từ nhiều lớp XLS-R. EER 1,92% trên ASVspoof 2021 DF, 7,46% trên In-the-Wild |
| XLSR-Mamba: A Dual-Column Bidirectional State Space Model for Spoofing Attack Detection | Y. Xiao, R. K. Das | IEEE SPL 2025 | Q1 theo SJR, Q2 theo JCR (nguồn thứ cấp) | Thay attention bằng Mamba hai chiều; suy luận nhanh hơn, tốt nhất trên In-the-Wild trong bài |
| Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing | T. Liu, D.-T. Truong, R. K. Das, K. A. Lee, H. Li | IEEE TIFS | Q1 (nguồn thứ cấp) | Bỏ lớp giảm chiều, back-end Res2Net lồng nhau nhẹ hơn |

**Ưu điểm:** nhiều code, dễ tái lập. **Nhược điểm:** đa số chỉ huấn luyện trên ASVspoof 2019 LA. Trên RADAR 2026, baseline SSL-AASIST có EER 42,6%.

### 2.3. Hướng 3: Phát hiện và định vị giả mạo một phần

Ý tưởng: kẻ tấn công chỉ thay vài từ trong bản ghi thật, nên mô hình phải chỉ ra đoạn nào bị giả, không chỉ phân loại cả câu.

| Bài | Tác giả | Nơi công bố | Hạng | Tóm tắt |
|---|---|---|---|---|
| The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance | L. Zhang, X. Wang, E. Cooper, N. Evans, J. Yamagishi | IEEE/ACM TASLP 2023 | Q1 (nguồn thứ cấp) | Nhãn đoạn ở 6 độ phân giải 20–640 ms; học đồng thời nhãn đoạn và nhãn câu |
| Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism (BAM) | J. Zhong, B. Li, J. Yi | Interspeech 2024 | CORE A | Dự đoán ranh giới đoạn giả, rồi dùng ranh giới để điều khiển attention giữa các khung |
| PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing | Y. Zhang, B. Tian, L. Zhang, Z. Duan | Interspeech 2025 | CORE A | Detector học trên PartialSpoof thất bại với tiếng nói chỉnh sửa bằng mô hình neural |

**Ưu điểm:** sát với lừa đảo thực tế. **Nhược điểm:** cần nhãn theo đoạn; chưa tìm thấy dữ liệu giả mạo một phần tiếng Việt công khai.

### 2.4. Hướng 4: Audio LLM

Ý tưởng: đặt bài toán thành hỏi đáp trên âm thanh để mô hình vừa kết luận vừa giải thích.

| Bài | Tác giả | Nơi công bố | Hạng | Tóm tắt |
|---|---|---|---|---|
| ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection | H. Gu, J. Yi, C. Wang, J. Tao, Z. Lian, J. He, Y. Ren, Y. Chen, Z. Wen | ACM MM 2025 | CORE A* | Fine-tune Audio LLM theo dạng hỏi đáp; tốt khi ít dữ liệu. Trên In-the-Wild đạt 26,99% khi mở băng encoder |
| DFALLM: Achieving Generalizable Multitask Deepfake Detection by Optimizing Audio LLM Components | Y. Li và cộng sự | arXiv 2025 | Preprint | Encoder âm thanh là nút thắt chính; tăng cỡ LLM ít tác dụng |

**Ưu điểm:** có khả năng giải thích. **Nhược điểm:** tốn GPU (mô hình cỡ 7B), độ tin cậy của lập luận còn bị đặt nghi vấn, chưa rõ hỗ trợ tiếng Việt.

### 2.5. Hướng 5: Continual learning và thích nghi ít mẫu

Ý tưởng: cập nhật detector khi xuất hiện phương pháp sinh mới mà không quên các tấn công cũ.

| Bài | Tác giả | Nơi công bố | Hạng | Tóm tắt |
|---|---|---|---|---|
| Do You Remember? Overcoming Catastrophic Forgetting for Fake Audio Detection (RAWM) | X. Zhang, J. Yi, J. Tao, C. Wang, C. Y. Zhang | ICML 2023 | CORE A* | Điều chỉnh hướng cập nhật trọng số theo tỉ lệ thật/giả, kèm chính quy hóa giữ phân phối tiếng nói thật cũ |
| Rehearsal with Auxiliary-Informed Sampling for Audio Deepfake Detection (RAIS) | F. G. Febrinanto, K. Moore, C. Thapa, J. Ma, V. Saikrishna, F. Xia | Interspeech 2025 | CORE A | Sinh nhãn phụ để chọn mẫu đa dạng cho bộ đệm ôn tập |
| Few-Shot Speech Deepfake Detection Adaptation with Gaussian Processes (ADD-GP) | N. Glazer, D. Chernin, I. Achituve, S. Gannot, E. Fetaya | Interspeech 2025 | CORE A | Bộ phân loại Gaussian Process trên đặc trưng XLS-R, thích nghi với TTS mới từ ít mẫu |

**Ưu điểm:** sát với thực tế, vì TTS mới liên tục xuất hiện. **Nhược điểm:** cần giao thức nhiều giai đoạn; chưa tìm thấy code công khai của ADD-GP.

**Các hướng khác cần biết:** phát hiện TTS dựa trên neural codec (CodecFake+, TASLP 2026); truy vết hệ thống sinh (Interspeech 2025, CORE A); watermark chủ động (AudioSeal, ICML 2024, CORE A*).

### 2.6. 5 bài tính cho tiêu chí tổng quan

**Tiêu chí chọn**
- **Đa dạng:** mỗi bài đại diện một cách tiếp cận khác nhau.
- **Thời sự:** công bố từ 2024.
- **Liên quan:** mỗi bài giúp trả lời một phần của đề tài.
- **Đạt hạng:** ưu tiên hạng tra trực tiếp trên CORE 2023.

| # | Bài | Nơi công bố | Hạng | Cách tiếp cận | Vì sao chọn |
|---|---|---|---|---|---|
| 1 | XLSR-SLS | ACM MM 2024 | CORE A* | Kiến trúc: SSL + back-end chọn lớp | Đại diện hướng phổ biến nhất; đứng đầu các mô hình mã nguồn mở trong Speech DF Arena (Phase I) |
| 2 | ALLM4ADD | ACM MM 2025 | CORE A* | Mô hình: Audio LLM | Đại diện xu hướng mới nhất; giải thích vì sao nhóm không chọn hướng này |
| 3 | CodecFake+ | TASLP 2026 | Q1 (nguồn thứ cấp) | Dữ liệu cho TTS dựa trên neural codec | TTS đời mới phần lớn dùng codec; liên quan đến hệ thống sinh chưa gặp |
| 4 | GRPO | Interspeech 2026 | CORE A | Chiến lược fine-tune không quên | Nền trực tiếp của đề tài |
| 5 | VSASV | Interspeech 2024 | CORE A | Dữ liệu tiếng Việt | Bài tiếng Việt duy nhất đạt hạng; đáp ứng phần "trong nước" |

**Mạch trình bày:** kiến trúc (1) → mô hình mới (2) → thách thức từ TTS đời mới (3) → chiến lược huấn luyện (4) → tình hình tiếng Việt (5).

Các bài khác trong mục 2 (PartialSpoof, RAWM, BAM, ...) được trích dẫn trong phần tổng quan nhưng không tính vào 5 bài. Bài Tak (Odyssey), AASIST và RawBoost (ICASSP, CORE B) cùng các preprint cũng vậy. Nếu cần cả 5 bài có hạng tra trực tiếp, thay CodecFake+ bằng bài truy vết nguồn qua codec (Interspeech 2025, CORE A).

### 2.7. Tình hình tiếng Việt

| Công trình | Tác giả | Nơi công bố | Hạng | Tóm tắt |
|---|---|---|---|---|
| VSASV: a Vietnamese Dataset for Spoofing-Aware Speaker Verification | V. Hoang, V. T. Pham, H. N. Xuan, P. Nhi, P. Dat, T. T. T. Nguyen | Interspeech 2024 | CORE A | 174.000 câu giả, 164.000 câu thật, 1.382 người nói. Gồm replay, voice conversion (RVC) và tấn công đối kháng, **không có TTS** |
| SEA-Spoof: Bridging The Gap in Multilingual Audio Deepfake Detection for South-East Asian | J. Wu, N. Hou, Z. Pan, Q. Zhang, S. H. Bhupendra, S. Mondal | arXiv 2025 | Preprint | 6 ngôn ngữ Đông Nam Á, có tiếng Việt. Trên tập thương mại tiếng Việt, MoLEx có EER 48,50%, giảm còn 0,01% sau fine-tune |
| VietPrism: A Large-Scale Vietnamese Speech and Deepfake Corpus with Diverse Dialects and Code-Switching | M. Hoang, T. Le | arXiv 2026 | Preprint | 993,4 giờ tiếng nói thật, 1.262 người nói. Zero-shot 5 detector, EER dao động từ 0,00% đến 99,99% theo hệ thống sinh; không fine-tune, không ablation |
| RADAR Challenge 2026: Robust Audio Deepfake Recognition under Media Transformations | H.-T. Luong, X. Liu, I. Kukanov, Z. X. Chai, K. A. Lee | Gửi APSIPA 2026 | Không có trong CORE 2023 | 102.725 câu thuộc 6 biến thể ngôn ngữ, có tiếng Việt, kèm biến đổi âm thanh. Baseline SSL-AASIST có EER 42,6%, đội tốt nhất 5,10% |

---

## 3. Chọn hướng nghiên cứu

### 3.1. Tiêu chí lựa chọn

| Tiêu chí | Câu hỏi kiểm tra |
|---|---|
| Thời sự | Hướng này còn được nghiên cứu tích cực năm 2026 không? |
| Dữ liệu | Có dữ liệu tiếng Việt công khai, có nhãn không? |
| Mã nguồn | Có code và trọng số công khai không? |
| Khả thi | Chạy được với GPU của Colab/Kaggle không? |
| Khoảng trống | Có câu hỏi chưa ai trả lời cho tiếng Việt, và làm ablation được không? |

### 3.2. Đánh giá các hướng

| Hướng | Thời sự | Dữ liệu | Mã nguồn | Khả thi | Khoảng trống | Kết luận |
|---|---|---|---|---|---|---|
| 1. Post-training và fine-tune | Có (Interspeech 2026) | Có | Có, đủ train và GRPO | Có, fine-tune được backbone đến 317M trên GPU 16 GB | Có | **Chọn** |
| 2. SSL + back-end | Có, nhưng đã rất nhiều bài | Có | Có | Có | Thấp; baseline yếu trên RADAR | Dùng làm baseline |
| 3. Giả mạo một phần | Có | Chưa tìm thấy dữ liệu tiếng Việt | Có | Có | Có, nhưng thiếu dữ liệu | Không chọn |
| 4. Audio LLM | Có | Có | Có (ALLM4ADD) | Khó, mô hình cỡ 7B | Chưa rõ hỗ trợ tiếng Việt | Không chọn |
| 5. Continual learning | Có | Có | Chưa tìm thấy code ADD-GP | Có | Có | Dùng cho hướng phát triển |

### 3.3. Khoảng trống được chọn

- **VietPrism** chỉ đánh giá zero-shot trên tiếng Việt, không fine-tune, không ablation.
- **Bài GRPO** chỉ fine-tune trên một bộ dữ liệu, chỉ ablation với XLS-R-2B, và không có tiếng Việt. Tác giả cũng nêu cơ chế cần nghiên cứu thêm.

Đồ án lấp khoảng trống giữa hai bài này. Đồ án **không** đề xuất kiến trúc mới và **không** nhận là đầu tiên về phát hiện deepfake tiếng Việt.

---

## 4. Giải pháp được chọn và cách phân tích

### 4.1. Giải pháp

**AntiDeepfake** ([code](https://github.com/nii-yamagishilab/AntiDeepfake), giấy phép BSD-3; 14 checkpoint, CC BY-NC-SA 4.0).

| Thành phần | Mô tả |
|---|---|
| Kiến trúc | Dạng sóng → wav2vec 2.0 (CNN + Transformer) → trung bình theo thời gian → 1 lớp fully connected → 2 điểm `<Fake, Real>` |
| Backbone | W2V-Small, W2V-Large (tiếng Anh); MMS-300M, MMS-1B, XLS-R-1B, XLS-R-2B (đa ngôn ngữ); HuBERT-XL |
| Post-training | Huấn luyện có giám sát trên dữ liệu lớn trước khi fine-tune; có bản có và không có RawBoost |
| Fine-tune SFT | Cross-entropy trên dữ liệu đích |
| Fine-tune GRPO | Lấy mẫu nhãn từ xác suất dự đoán, thưởng 1 khi đúng và 0 khi sai, chuẩn hóa theo nhóm; giới hạn bước cập nhật (`epsilon`) và phạt lệch khỏi mô hình tham chiếu (`beta`) |

### 4.2. Câu hỏi nghiên cứu

1. Những yếu tố nào quyết định khả năng phát hiện deepfake tiếng Việt, xét theo 3 nhóm: **mô hình nền** (kích thước, ngôn ngữ và cách pre-train), **cách train** (post-training, tăng cường dữ liệu), và **dữ liệu** (độ phủ hệ thống sinh, lượng dữ liệu tiếng Việt khi fine-tune)?
2. SFT trên tiếng Việt có làm mô hình mất khả năng tổng quát hóa không?
3. GRPO có giữ khả năng tổng quát hóa tốt hơn SFT trên tiếng Việt không, và thành phần nào tạo ra khác biệt?

### 4.3. Dữ liệu và giao thức đánh giá

| Tập | Vai trò | Ghi chú |
|---|---|---|
| SEA-Spoof (phần tiếng Việt) | Fine-tune và kiểm tra trong miền | 81.020 câu, 7 hệ thống sinh; có nhãn hệ thống sinh và người nói theo từng câu. Đang chờ tác giả duyệt. Dự phòng: MLAAD tiếng Việt + VIVOS |
| Hệ thống sinh được để riêng | Kiểm tra hệ thống chưa gặp | Lần lượt để riêng một hệ thống khỏi tập fine-tune |
| RADAR 2026 tiếng Việt | Kiểm tra ngoài miền | 13.438 câu: iFlytek, ElevenLabs, Cartesia, OpenAI, Fish Audio, Piper; có biến đổi âm thanh |
| Tập con cân bằng của In-the-Wild và RADAR tiếng Anh | Đo mức quên kiến thức cũ | Không nằm trong dữ liệu huấn luyện của AntiDeepfake |
| Tập nhóm tự thu | Kiểm tra lối tắt | Cùng người nói, cùng câu, giọng thành viên có đồng ý |

**Độ đo:** EER, minDCF, actDCF (tham số ASVspoof 5). Cấu hình mốc và GRPO chạy 3 seed.

**Thước đo chính:** hệ thống sinh được để riêng, RADAR tiếng Việt và mức quên. Tập trong miền dễ bão hòa gần 0% (SEA-Spoof báo 0,01% sau fine-tune), nên không dùng để so sánh cấu hình.

**Yếu tố gây nhiễu được phân tích:** tiếng nói thật của SEA-Spoof đến từ VIVOS, còn của RADAR tiếng Việt đến từ Common Voice và FOSD. Fine-tune có thể khiến mô hình coi tiếng thật từ nguồn khác là giả. Hiện tượng này được ghi nhận như một kết quả, không khắc phục bằng cách thêm Common Voice vào tập huấn luyện vì sẽ gây rò rỉ sang RADAR.

**Loại khỏi tập kiểm tra:** ASVspoof 2019, 2021 và 5, vì AntiDeepfake đã dùng để huấn luyện.

### 4.4. Ablation study (30%)

Cấu hình mốc **R**: MMS-300M (đa ngôn ngữ, 317M tham số) post-trained bản Default (có RawBoost khi post-train), fine-tune SFT, có RawBoost. Chọn bản Default để so được với bài GRPO. Mỗi cấu hình đổi đúng một yếu tố so với cấu hình ở cột "So với".

| Mã | Thay đổi | So với | Câu hỏi trả lời |
|---|---|---|---|
| S0 | Không fine-tune (zero-shot) | R | Fine-tune trên tiếng Việt có cần thiết? (đã đo: 20,08% trên RADAR tiếng Việt) |
| P0 | Bỏ post-training, khởi tạo từ trọng số gốc | R | Post-training đóng góp bao nhiêu? |
| A0 | Bỏ RawBoost | R | Tăng cường dữ liệu giúp bao nhiêu? |
| B2 | Đổi sang W2V-Large (tiếng Anh, 317M) | R | Ngôn ngữ pre-training, khi giữ nguyên kích thước và kiến trúc |
| B1 | Đổi sang W2V-Small (tiếng Anh, 95M) | B2 | Kích thước backbone, khi giữ nguyên ngôn ngữ |
| N0 | Khởi tạo từ bản post-train không RawBoost (NDA) | R | Lợi thế của bản NDA khi dùng thẳng trên tiếng Việt có còn sau fine-tune? (không bắt buộc) |

Các backbone lớn hơn (XLS-R-1B, XLS-R-2B, MMS-1B, HuBERT-XL) chỉ đánh giá zero-shot, vì fine-tune không vừa GPU 16 GB (mục 4.6). XLS-R-2B dùng thẳng (13,71% trên RADAR tiếng Việt) là dòng tham chiếu trong mọi bảng: fine-tune mô hình 317M có thắng mô hình 2,2B không fine-tune không.

Ngoài EER, minDCF, actDCF: báo tỉ lệ tiếng giả bị nhận là thật và tiếng thật bị coi là giả ở cùng một ngưỡng, và kết quả trên nhóm câu mà mọi mô hình đều sai khi dùng thẳng (mục 5.1).

Baseline so sánh: XLS-R + AASIST (W2V2-AASIST).

### 4.5. Phân tích sâu và cải tiến (20%)

| Mã | Thay đổi so với R | Câu hỏi trả lời |
|---|---|---|
| G1 | Thay SFT bằng GRPO đầy đủ | GRPO có giữ khả năng tổng quát hóa tốt hơn? |
| G2 | GRPO không giới hạn bước cập nhật | Giới hạn bước cập nhật có cần thiết? |
| G3 | GRPO với `beta` = 0 | Phạt lệch khỏi mô hình tham chiếu có vai trò gì? |

Phân tích bổ sung:
- SFT so với GRPO khi chỉ có 50, 200, 1000 mẫu mỗi lớp.
- Hiệu chuẩn điểm số để giảm actDCF. Khi dùng thẳng, các mô hình đã gần hiệu chuẩn tốt (actDCF chỉ hơn minDCF tối đa 0,018), nên bước này chỉ cần nếu fine-tune làm lệch.
- Theo dõi nhóm 493 câu mà cả 7 mô hình đều sai khi dùng thẳng: fine-tune có sửa được không.
- Phân tích theo từng hệ thống sinh, gồm các trường hợp EER cực đoan mà VietPrism đã ghi nhận.

### 4.6. Tài nguyên tính toán

Đồ án thiết kế để chạy trên GPU miễn phí (Kaggle/Colab, khoảng 16 GB) và máy cá nhân, không cần máy chủ lớn.

| Backbone | Tham số | Bộ nhớ fine-tune ước lượng | Vai trò |
|---|---|---|---|
| XLS-R-2B | 2,2B | ~35 GB | Chỉ zero-shot |
| XLS-R-1B, MMS-1B | ~965M | ~15,4 GB, vượt 16 GB khi tính activation | Chỉ zero-shot |
| MMS-300M, W2V-Large | 317M | ~5,1 GB | Fine-tune |
| W2V-Small | 95M | ~1,5 GB | Fine-tune |

Ước lượng theo khoảng 16 byte mỗi tham số khi fine-tune với AdamW, chưa tính activation; sẽ đo thực tế trước khi chạy thí nghiệm.

**Đã đo khi chấm zero-shot** (Kaggle, 2 × T4, 4 nhân CPU, 30 giờ GPU mỗi tuần): bản 317M trở xuống chấm nhanh 27–45 lần thời gian thực, bản ~1B 15–18 lần, bản 2,2B ~9 lần. Toàn bộ bảng zero-shot tốn 10,3 giờ GPU.

**Phương án theo tầng**

| Tầng | Cần gì | Nội dung | Đáp ứng |
|---|---|---|---|
| 0. Chắc chắn làm được | Chỉ inference trên GPU, phần còn lại trên CPU | Chấm zero-shot; trích embedding rồi huấn luyện bộ phân loại nhẹ để so post-training và backbone | Demo, một phần ablation |
| 1. Mục tiêu | GPU 16 GB | Fine-tune MMS-300M, W2V-Large, W2V-Small; so SFT với GRPO | Đủ ablation và cải tiến |
| 2. Mở rộng | GPU từ 40 GB | Fine-tune XLS-R-1B | Không bắt buộc |

---

## 5. Kết quả đầu ra

### 5.1. Kết quả đã có: zero-shot trên RADAR tiếng Việt (14/11)

Pipeline được kiểm chứng bằng ba mốc công bố, đều khớp: toy example 6,11%; MMS-300M trên In-the-Wild 2,90% (đo được 2,9016%); W2V2-AASIST trên In-the-Wild 11,22% (đo được 11,21%).

| Backbone | Tham số | Pre-train | Có RawBoost khi post-train | Không RawBoost |
|---|---|---|---|---|
| W2V-Small | 95M | Anh | 27,21 | 23,24 |
| W2V-Large | 317M | Anh | 17,80 | 16,62 |
| MMS-300M | 317M | Đa ngôn ngữ | 20,08 | 18,07 |
| HuBERT-XL | 964M | Anh | 20,49 | 30,67 |
| MMS-1B | 965M | Đa ngôn ngữ | 14,32 | 14,20 |
| XLS-R-1B | 965M | Đa ngôn ngữ | 14,99 | 15,60 |
| XLS-R-2B | 2,2B | Đa ngôn ngữ | **13,71** | 14,23 |
| W2V2-AASIST (baseline) | | | 30,12 | |

EER (%) trên 13.438 câu (7.000 thật, 6.438 giả); khoảng tin cậy 95% bootstrap rộng khoảng ±0,6–0,8 điểm.

**Nhận xét ban đầu**
1. Mô hình post-trained dùng thẳng tốt hơn baseline chuyên biệt, nhưng còn xa mức trên In-the-Wild (1,2–2,9%): còn chỗ cho fine-tune.
2. Bỏ RawBoost khi post-train tốt hơn ở cả 3 bản từ 317M trở xuống (1,2–4,0 điểm); ở bản từ 1B trở lên thì ngang hoặc ngược lại.
3. Ngôn ngữ pre-train ảnh hưởng ít hơn kích thước mô hình: cặp cùng kiến trúc W2V-Large (tiếng Anh) và MMS-300M (đa ngôn ngữ) chênh 1,5–2,3 điểm theo chiều có lợi cho tiếng Anh, trong khi tăng kích thước giảm 3,9–9,4 điểm. RADAR tiếng Anh còn khó hơn tiếng Việt, và thứ hạng mô hình trên In-the-Wild dự báo đúng thứ hạng trên tiếng Việt (tương quan hạng 0,91). Giả thuyết: dữ liệu post-train chưa phủ các hệ thống sinh 2025–2026 và các biến đổi âm thanh của RADAR. Fine-tune ở phần ablation sẽ kiểm tra giả thuyết này.
4. Các mô hình sai trên cùng một nhóm câu (3,7% câu giả qua mặt cả 7 mô hình, so với 0,001% nếu sai độc lập).

### 5.2. Kết quả dự kiến

| Kết quả | Dạng trình bày | Trả lời |
|---|---|---|
| Bảng zero-shot 14 checkpoint và baseline trên RADAR tiếng Việt | Bảng EER | Mô hình có sẵn làm tốt đến đâu trên tiếng Việt (**đã có**, mục 5.1) |
| Bảng ablation 6–7 cấu hình × 5 nhóm tập kiểm tra | Bảng EER, minDCF (trung bình ± độ lệch chuẩn), kèm dòng tham chiếu XLS-R-2B dùng thẳng | Câu hỏi 1 |
| Mức thay đổi EER ngoài miền trước và sau SFT | Biểu đồ cột | Câu hỏi 2 |
| SFT so với GRPO và các biến thể | Bảng và biểu đồ số mẫu theo EER | Câu hỏi 3 |
| EER theo từng hệ thống sinh | Bảng chi tiết | Hệ thống nào khó phát hiện nhất |
| actDCF trước và sau hiệu chuẩn | Bảng | Mức sẵn sàng triển khai |
| Demo | Web Gradio: tải âm thanh lên, trả điểm và kết luận | Tiêu chí demo |

**Số liệu tham chiếu từ tài liệu**

| Mô hình | Tập | EER | Nguồn |
|---|---|---|---|
| XLS-R-2B post-trained, zero-shot | In-the-Wild | 1,23% | AntiDeepfake |
| XLS-R + SLS | In-the-Wild | 7,46% | XLSR-SLS |
| SSL-AASIST | RADAR 2026, đa ngôn ngữ | 42,6% | RADAR 2026 |
| Đội tốt nhất | RADAR 2026, đa ngôn ngữ | 5,10% | RADAR 2026 |
| XLS-R-2B: không fine-tune / SFT / GRPO | Trung bình ngoài miền | 2,69% / 6,28% / 2,69% | Bài GRPO |
| MMS-300M: không fine-tune / SFT / GRPO | In-the-Wild | 2,91% / 14,17% / 5,91% | Bài GRPO |

**Cách diễn giải:** kết quả theo hướng nào cũng có giá trị. Nếu GRPO không giúp trên tiếng Việt, đó vẫn là phát hiện hợp lệ, vì bài gốc cũng ghi nhận hiệu quả không đồng đều giữa các backbone.

---

## 6. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| ~~Môi trường cũ (Python 3.9, fairseq commit cố định) không chạy trên Colab~~ | **Đã xử lý**: chạy được trên Colab và Kaggle, ba mốc kiểm chứng khớp |
| SEA-Spoof cần tác giả duyệt và yêu cầu gửi email trước | Gửi email theo hướng dẫn trên trang dữ liệu; song song xin MLAAD (duyệt tự động) làm dự phòng |
| Máy GPU miễn phí chỉ có 4 nhân CPU, đọc dữ liệu có thể chậm hơn tính toán | Đo trước khi tính ngân sách; chạy hai cấu hình song song trên hai GPU trong một phiên |
| RADAR không có nhãn hệ thống sinh theo từng câu | Phân tích theo hệ thống sinh làm trên SEA-Spoof; trên RADAR chỉ báo chung và theo nhóm câu khó |
| SEA-Spoof không được duyệt kịp | Dùng MLAAD tiếng Việt + VIVOS, ghi rõ hạn chế về khác biệt nguồn dữ liệu |
| Không đủ GPU để fine-tune | Cấu hình mốc đã chọn MMS-300M (khoảng 5 GB khi fine-tune); nếu vẫn thiếu thì dùng W2V-Small hoặc tầng 0 |
| Tập trong miền bão hòa gần 0% | Dùng hệ thống sinh được để riêng, RADAR tiếng Việt và mức quên làm thước đo chính |
| Rò rỉ dữ liệu giữa huấn luyện và kiểm tra | Loại ASVspoof; ghi rõ RADAR có hệ thống sinh gần với SEA-Spoof |

---

## 7. Hướng phát triển

| Nếu kết quả cho thấy | Hướng tiếp theo |
|---|---|
| GRPO giữ được khả năng tổng quát hóa | Mở rộng sang phương ngữ và code-switching tiếng Việt |
| GRPO không giúp | Tìm hiểu cơ chế, thử continual learning (RAWM) |
| Kém với hệ thống sinh chưa gặp | Thích nghi ít mẫu (ADD-GP) |
| EER tốt nhưng actDCF xấu | Hiệu chuẩn trước khi triển khai |
| Đã ổn định ở mức cả câu | Định vị đoạn giả trong tiếng Việt |
