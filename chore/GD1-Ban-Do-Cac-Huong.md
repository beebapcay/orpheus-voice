# GĐ1. Bản đồ các hướng: thẻ tổng quan và bảng so sánh

Kiến thức chi tiết của từng hướng nằm trong `note/`. File này chỉ giữ phần gắn với đồ án: thẻ của 5 bài nhóm A, câu trả lời cho các câu hỏi ở mục 3 của plan, và bảng so sánh các hướng.

## Tiến độ

| Ngày | Nhóm A | Nhóm C | Note kiến thức | Trạng thái |
|---|---|---|---|---|
| 1 | ALLM4ADD | PartialSpoof | `note/Audio-LLM.md`, `note/Gia-Mao-Mot-Phan.md` | Xong bản nháp, chờ đọc lại |
| 2 | CodecFake+, VSASV | Phân loại theo cấu tạo codec, VietPrism | `note/Codec-Va-Truy-Vet-Nguon.md`, `note/Deepfake-Tieng-Viet.md` | Xong bản nháp, chờ đọc lại |
| 3 | GRPO, XLSR-SLS | Speech DF Arena, RADAR 2026, Tak 2022 | `note/Post-Training-Va-GRPO.md`, `note/SSL-Va-Back-End.md`, `note/Benchmark-Va-Do-Ben.md` | Xong bản nháp. **XLSR-SLS chưa có PDF** (ACM và OpenReview chặn tải tự động): cần tải tay vào `resource/paper/` |

## Thẻ nhóm A

### 1. ALLM4ADD (ACM MM 2025, CORE A*)

**Bài:** H. Gu, J. Yi, C. Wang, J. Tao, Z. Lian, J. He, Y. Ren, Y. Chen, Z. Wen. *ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection.* [arXiv 2505.11079](https://arxiv.org/abs/2505.11079), [code](https://github.com/ucas-hao/qwen_audio_for_add).

| Mục | Nội dung |
|---|---|
| Vấn đề | Audio LLM đã giỏi nhiều tác vụ âm thanh, nhưng chưa ai thử cho phát hiện giả. Thử zero-shot thì thất bại: Qwen-Audio-chat trả lời "thật" cho mọi đoạn |
| Ý tưởng | Đổi bài toán thành hỏi đáp: "Is this audio fake or real?" → "Fake"/"Real". SFT Qwen-Audio-chat (7,7B) với LoRA trên LLM; encoder Whisper giữ băng (★) hoặc cho học (△). Điểm số lấy từ softmax trên logit của hai token "Fake" và "Real" |
| Dữ liệu | Train trên ASVspoof 2019 LA (toàn bộ, 1/4, 1/8, 1/16, đến 1/128). Kiểm tra trên ASVspoof 2019 LA eval, In-the-Wild, SceneFake, EmoFake |
| Con số chính | ASVspoof 2019 LA: 0,41% (AASIST 0,83%). In-the-Wild: 26,99% (AASIST 43,02%) |
| Ưu điểm | Học tốt từ rất ít dữ liệu (dưới 5% với 178 giả + 20 thật). Một khung cho nhiều loại giả. Hướng tới giải thích được |
| Nhược điểm | 7,7B tham số, cần A100. Ngoài miền vẫn 26,99%. Chưa có giải thích thật sự, đầu ra chỉ 1 từ. Lý do dùng accuracy sai về mặt độ đo. Lời và bảng không khớp (36,11% không có trong Bảng 4) |
| Liên hệ đề tài | Xem bên dưới |

**Trả lời câu hỏi ở mục 3 của plan**

| Câu hỏi | Trả lời |
|---|---|
| Bài toán được chuyển thành hỏi đáp ra sao? | Đầu vào là âm thanh + câu hỏi cố định, đầu ra ràng buộc là "Fake" hoặc "Real", học bằng loss mô hình ngôn ngữ. Khi chấm, chỉ lấy logit của hai token đó, nên thực chất là phân loại hai lớp |
| Mở băng encoder làm kết quả đổi thế nào (Bảng 4)? | In-the-Wild: 32,04% → 26,99% (toàn bộ tập train), 45,12% → 33,20% (1/16). Trên ASVspoof 2019 LA, EER trung bình 4 mức dữ liệu giảm 0,81 điểm |

**Liên hệ đề tài**

- **Vì sao không chọn hướng này:** chỉ riêng trọng số 7,7B ở 16 bit đã khoảng 15,4 GB (7,7 tỉ × 2 byte), sát mức GPU 16 GB, chưa tính activation và trạng thái của LoRA. Bài huấn luyện trên A100.
- **Không so thẳng với AntiDeepfake:** XLS-R-2B post-trained đạt 1,23% zero-shot trên In-the-Wild so với 26,99% của ALLM4ADD, nhưng hai mô hình học trên lượng dữ liệu rất khác nhau: AntiDeepfake post-train trên khoảng 56.000 giờ thật và 18.000 giờ có artifact, còn ALLM4ADD chỉ học trên 25.380 câu của ASVspoof 2019 LA train. Khoảng cách này chủ yếu cho thấy vai trò của dữ liệu huấn luyện, không chứng minh kiến trúc nào tốt hơn.
- **Dùng được cho đồ án:**
  - Cách lấy điểm từ logit hai lớp giống cách lưu điểm log-odds trong `evaluate.py`.
  - Nhận xét về accuracy củng cố lựa chọn EER, minDCF, actDCF làm độ đo.
  - Thí nghiệm encoder băng/học tương tự câu hỏi "fine-tune bao nhiêu phần của backbone" ở GĐ5.
  - Thí nghiệm theo lượng dữ liệu (1/4 đến 1/128) là mẫu cho phần N = 50, 200, 1000 ở GĐ6. Lưu ý so sánh cả theo điểm tuyệt đối lẫn theo tỉ lệ.
- **Chưa kiểm tra:** khả năng của Qwen-Audio với tiếng Việt.

### 2. CodecFake+ (IEEE/ACM TASLP 2026, vol. 34)

**Bài:** X. Chen, J. Du, H. Wu, L. Zhang, I-M. Lin, I-H. Chiu, W. Ren, Y. Tseng, Y. Tsao, J.-S. R. Jang, H.-y. Lee. *CodecFake+: Codec-Based Resynthesized Data as a Proxy for Detecting CodecFake Speech.* [arXiv 2501.08238](https://arxiv.org/abs/2501.08238), [code](https://github.com/ResponsibleGenAI/CodecFake-Plus-Dataset). Bản arXiv v3 ghi đã đăng TASLP vol. 34, 2026. Hạng Q1 vẫn cần tra lại trên SCImago.

| Mục | Nội dung |
|---|---|
| Vấn đề | TTS đời mới (VALL-E, MaskGCT, ...) sinh token codec rồi dùng decoder của codec để dựng dạng sóng. Bộ phát hiện học trên dữ liệu vocoder cũ kém với loại này. Hầu hết hệ thống không công khai nên không tự sinh dữ liệu huấn luyện được |
| Ý tưởng | Cho tiếng thật đi qua codec công khai rồi gán nhãn giả (CoRS), làm dữ liệu thay thế. Phân loại codec theo 3 thuộc tính: lượng tử hoá, mục tiêu phụ, loại decoder. Dùng phân loại này để chọn dữ liệu huấn luyện |
| Dữ liệu | Train: VCTK qua 31 codec (42.965 × 31 câu giả). Test: 931 câu giả của 17 hệ thống CoSG, tải từ trang demo |
| Con số chính | W2V2-AASIST học trên ASVspoof 2019: 18,92% trên CoSG. Cân bằng theo decoder: 11,91%. Thêm ASVspoof 5 và gấp 4 dữ liệu: 8,07%. Gộp cả 31 codec: khoảng 50% |
| Ưu điểm | Tạo dữ liệu quy mô lớn không cần hệ thống CoSG thật. Giải thích được vì sao dữ liệu này tốt hơn dữ liệu kia (decoder Freq, mục tiêu phụ Disent) |
| Nhược điểm | Tập test nhỏ (nhiều hệ thống chỉ 5–11 câu), khoảng tin cậy rộng. Tiếng thật lúc test khác nguồn lúc học. Dòng "Top-1" chọn mô hình theo từng cột của tập test. Chỉ tiếng Anh, phòng thu |

**Trả lời câu hỏi ở mục 3 của plan**

| Câu hỏi | Trả lời |
|---|---|
| CoRS được dùng làm dữ liệu huấn luyện ra sao? | Tiếng thật VCTK → encoder → lượng tử hoá → decoder của từng codec công khai → gán nhãn giả. Tiếng thật gốc giữ nguyên, gán nhãn thật. Lý do: CoSG cũng dùng decoder codec ở bước cuối, nên dấu vết giống nhau |
| Loại codec nào cho kết quả tốt nhất? | Từng codec: Vocos (L1) và FACodec (G), dưới 14%. Theo thuộc tính: decoder miền tần số (Freq) và mục tiêu phụ tách thuộc tính (Disent). Theo cách chọn dữ liệu: cân bằng theo loại decoder (11,91%) |

**Liên hệ đề tài**

- **Bài học về dữ liệu, áp dụng trực tiếp cho GĐ4:** gộp nhiều nguồn chưa chắc tốt hơn (31 codec cho khoảng 50%). Học trên đúng một hệ thống sinh vẫn có thể overfit theo nguồn dữ liệu (MaskGCT: 0% trên VCTK, 12,3% trên E2). Đây cũng là lý do đồ án để riêng một hệ thống sinh làm tập chưa gặp.
- **Bằng chứng cho mối lo về tiếng thật khác nguồn:** truy vết nguồn và CodecFake+ đều thấy tiếng thật khác kho làm mô hình sai nhiều hơn. Điều này khớp với yếu tố gây nhiễu VIVOS so với Common Voice/FOSD đã ghi ở GĐ5.
- **Nên tra thêm:** các hệ thống sinh tiếng Việt trong dữ liệu của đồ án (VoxCPM2, OmniVoice, F5-TTS, ...) dùng decoder loại nào (vocoder hay codec, Time hay Freq). Phân loại này giúp giải thích EER theo từng hệ thống ở GĐ6. Chưa kiểm tra.
- **Lý do không chọn làm hướng chính:** dữ liệu và kết quả chỉ có tiếng Anh. Dựng CoRS tiếng Việt cho 31 codec vượt khối lượng của đồ án.

### 3. VSASV (Interspeech 2024, CORE A)

**Bài:** V. Hoang, V. T. Pham, H. N. Xuan, N. Pham, P. Dat, T. T. T. Nguyen. *VSASV: a Vietnamese Dataset for Spoofing-Aware Speaker Verification.* [ISCA Archive](https://www.isca-archive.org/interspeech_2024/hoang24b_interspeech.html), [dữ liệu](https://github.com/hustep-lab/VSASV-Dataset).

| Mục | Nội dung |
|---|---|
| Vấn đề | Chưa có dữ liệu chống giả mạo cho tiếng Việt. Hệ thống xác thực học trên tiếng Anh kém trên tiếng Việt |
| Ý tưởng | Lấy tiếng thật từ Vietnam-Celeb, Common Voice, VIVOS. Tạo tiếng giả từ Common Voice và VIVOS bằng phát lại, chuyển giọng RVC, và nhiễu đối kháng DoubleDeceiver (giọng TTS + nhiễu) |
| Dữ liệu | 541,02 giờ, 1.382 người nói. Khoảng 164K câu thật, 174K câu giả. Nhiễu đối kháng chỉ có ở tập test |
| Con số chính | EER của cả hệ thống xác thực: thêm AASIST làm EER với phát lại giảm từ 15,69% xuống 2,51%, nhưng với nhiễu đối kháng lại tăng từ 8,96% lên 10,31% |
| Ưu điểm | Bộ dữ liệu chống giả mạo tiếng Việt đầu tiên, nhiều người nói, có replay thật |
| Nhược điểm | Không có tấn công TTS thuần. Tiếng giả từ kho sạch, tiếng thật chủ yếu từ kho ồn, nên có nguy cơ lối tắt. Chỉ báo EER của cả hệ thống, không có EER riêng của bộ phát hiện. Mô tả hệ thống giữa phần lời và bảng không khớp. Bản công bố nhỏ hơn số trong bài |

**Trả lời câu hỏi ở mục 3 của plan**

| Câu hỏi | Trả lời |
|---|---|
| Dữ liệu được tạo bằng những tấn công nào? | Phát lại (30,04 giờ), chuyển giọng RVC (119,44 giờ), nhiễu đối kháng DoubleDeceiver (35,18 giờ, chỉ ở tập test) |
| Vì sao không dùng được cho bài toán TTS? | Không có nhóm TTS thuần. TTS chỉ nằm trong nhóm nhiễu đối kháng, chỉ ở tập test, và không rõ hệ thống TTS nào. Ngoài ra còn nguy cơ lối tắt do nhiễu nền khác nhau giữa tiếng thật và tiếng giả |

**Liên hệ đề tài**

- Đây là bài tiếng Việt duy nhất đạt hạng A, đáp ứng phần "trong nước" của tiêu chí tổng quan.
- **Cần sửa câu chữ trong plan và đề cương:** hiện ghi VSASV "không có TTS". Chính xác hơn là "không có tấn công TTS thuần; TTS chỉ dùng làm đầu vào cho nhiễu đối kháng ở tập test".
- Nguy cơ lối tắt của VSASV là ví dụ thật để giải thích vì sao đồ án cần tập tự tạo (cùng người nói, cùng câu).

### 4. GRPO (Interspeech 2026, CORE A)

**Bài:** X. Wang, W. Ge, J. Yamagishi. *Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection?* [arXiv 2603.02914](https://arxiv.org/abs/2603.02914). Code GRPO nằm trong repo AntiDeepfake (`algo/rl.py`).

| Mục | Nội dung |
|---|---|
| Vấn đề | SFT trên miền mới làm mô hình post-trained quên: XLS-R-2B trên In-the-Wild từ 1,24% lên 6,35% |
| Ý tưởng | Thay SFT bằng GRPO: mô hình tự lấy mẫu 64 nhãn, thưởng 1 khi đúng, 0 khi sai, chuẩn hoá phần thưởng, kẹp tỉ số chính sách ($\epsilon = 0{,}2$), phạt lệch khỏi mô hình tham chiếu ($\beta = 0{,}04$) |
| Dữ liệu | Fine-tune trên Deepfake-Eval-2024 (~50 giờ). Ngoài miền: ADD23, FoR, DEEP-VOICE, In-the-Wild. 3 backbone post-trained: XLS-R-2B, MMS-1B, MMS-300M |
| Con số chính | XLS-R-2B, trung bình ngoài miền: không fine-tune 2,69% / SFT 6,28% / GRPO 2,69%. Trong miền: SFT 10,26%, GRPO 9,93% |
| Ưu điểm | Giảm quên mà không cần dữ liệu mới. Ablation tách vai trò của phạt KL và phần thưởng âm. Có code và checkpoint |
| Nhược điểm | "Cải thiện ngoài miền" là so với SFT; so với không fine-tune thì chỉ ngang (XLS-R-2B) hoặc hơi kém (MMS-300M 3,78% so với 3,68%). Một bộ dữ liệu fine-tune, ablation chỉ với XLS-R-2B, cơ chế chưa giải thích. GRPO không giúp mô hình chưa post-train |

**Trả lời câu hỏi ở mục 3 của plan**

| Câu hỏi | Trả lời |
|---|---|
| Vì sao SFT làm giảm kết quả ngoài miền? | Bài chỉ mô tả hiện tượng (quên) và đo độ trôi: SFT kéo phân bố embedding của DEEP-VOICE và In-the-Wild ra xa. Nguyên nhân sâu được để cho nghiên cứu sau. Note có một giả thuyết riêng: SFT kéo mạnh nhất vào các câu mô hình đang sai nặng (lực $1-p$), còn GRPO kéo đều hơn (lực $\sqrt{p(1-p)}$) |
| Bảng 1 nói gì về MMS-300M? | Trong miền: 29,23% → SFT 12,23% / GRPO 11,94%. Ngoài miền TB: 3,68% → SFT 7,74% / GRPO 3,78%. In-the-Wild: 2,91% → SFT 14,17% / GRPO 5,91%. Hiệu ứng quên rõ nhất trong 3 backbone |

**Liên hệ đề tài**

- **Khớp cấu hình GĐ6:** G1 = `grpo_v1` (`algo: grpo3`, có kẹp và mô hình cũ). G2 = `grpo_v2` (`algo: grpo2`, tức GRPOs). G3 = G1 với $\beta = 0$. Bài cho thấy G1 và G2 gần nhau, nên G2 chủ yếu để xác nhận lại.
- **Nên cân nhắc thêm biến thể "bỏ phần thưởng âm"**, vì bài coi đây là yếu tố then chốt. Biến thể này cần sửa code loss. Chỉ làm nếu còn quota, sau G3 trong thứ tự ưu tiên.
- **Code khác công thức trong bài:** `grpo2_loss` và `grpo3_loss` chuẩn hoá phần thưởng **trên cả batch** (`reward.mean()` và `reward.std()` trên B × 64 mẫu), trong khi công thức (4) chuẩn hoá **theo nhóm của từng câu**. Cần ghi rõ trong báo cáo là dùng cách của code. Batch ở đây là batch động (`max_batch_length`), nên số câu mỗi batch thay đổi. Phân tích "lực kéo" trong note giả định chuẩn hoá theo nhóm, nên không áp thẳng được cho code.
- **Rủi ro cần chú ý từ VietPrism:** MMS-300M post-trained zero-shot trên tiếng Việt gần mức đoán bừa (51,08%). Bài GRPO cho thấy GRPO kém khi điểm xuất phát yếu (từ mô hình chỉ pre-train). Nếu MMS-300M cũng gần ngẫu nhiên trên dữ liệu tiếng Việt của nhóm, GRPO có thể học chậm hoặc kém SFT trên tiếng Việt. Đây là một kết quả đáng báo cáo, không phải thất bại.
- **P0 (khởi tạo từ fairseq) chỉ nên chạy với SFT**, đúng như plan. Bài đã cho thấy GRPO trên mô hình chưa post-train kém hơn SFT (rows 10–12).
- **$\beta$ là một đánh đổi:** $\beta = 1$ cho ngoài miền 2,04% nhưng trong miền 15,83%. Nếu G3 cho kết quả lạ, so cả hai chiều.

### 5. XLSR-SLS (ACM MM 2024, CORE A*)

**Bài:** Q. Zhang, S. Wen, T. Hu. *Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier.* [DOI 10.1145/3664647.3681345](https://doi.org/10.1145/3664647.3681345), [OpenReview PDF](https://openreview.net/pdf?id=acJMIXJg2u), [code](https://github.com/QiShanZhang/SLSforADD).

> **Thẻ tạm.** Nội dung dưới đây lấy từ code, README của tác giả và số liệu của Speech DF Arena. Phải đọc toàn văn để điền phần "vấn đề" và lý do chọn nhiều lớp theo lời của bài trước khi đưa vào báo cáo.

| Mục | Nội dung |
|---|---|
| Vấn đề | Các bộ phát hiện dùng XLS-R thường chỉ lấy lớp cuối, bỏ qua thông tin ở các lớp khác (cần xác nhận lại với bài) |
| Ý tưởng | Lấy cả 24 lớp. Mỗi lớp được trung bình theo thời gian, qua FC dùng chung và sigmoid để ra trọng số **theo từng câu**. Cộng có trọng số các lớp, rồi qua back-end nhẹ (BN, max-pool, 2 FC) |
| Dữ liệu | Train trên ASVspoof 2019 LA. Test trên ASVspoof 2021 LA, DF, In-the-Wild |
| Con số chính | 2021 DF 1,92%, 2021 LA 2,87%, In-the-Wild 7,46% (README). Speech DF Arena: EER trung bình 14 tập 13,84%, tốt nhất trong các mô hình mã nguồn mở |
| Ưu điểm | Back-end đơn giản mà tổng quát hoá tốt hơn W2V2-AASIST (In-the-Wild 7,45% so với 11,19%). Có code và trọng số |
| Nhược điểm | Vẫn chỉ học trên ASVspoof 2019 LA. Ngoài miền khó vẫn kém: CodecFake 33,43%, SONAR 24,72%, ASVspoof 2024 18,76%, và 27,02% khi thêm vang |

**Trả lời câu hỏi ở mục 3 của plan**

| Câu hỏi | Trả lời (tạm) |
|---|---|
| Vì sao lấy đặc trưng từ nhiều lớp? | Các lớp giữ thông tin khác nhau: lớp thấp gần âm học, lớp cao gần ngôn ngữ, còn lớp cuối được tối ưu cho mục tiêu tự giám sát chứ không phải để giữ dấu vết giả. SLS để mô hình tự chọn lớp hữu ích cho từng câu. **Cần đối chiếu với lập luận trong bài** |

**Liên hệ đề tài**

- AntiDeepfake dùng **lớp cuối + trung bình theo thời gian + một lớp FC**, tức chính cách mà SLS cho là chưa tối ưu. Đây là một hướng cải tiến tự nhiên cho phần "hướng phát triển" (câu vấn đáp 4), nhưng nằm ngoài phạm vi ablation của đồ án.
- Trong Speech DF Arena, XLSR-SLS là mô hình mã nguồn mở tốt nhất, nên được chọn đại diện cho hướng 2. W2V2-AASIST vẫn là baseline của đồ án vì có trên `SpeechAntiSpoofingBenchmarks` và là baseline của RADAR.

## Ghi chú nhóm C

### PartialSpoof (IEEE/ACM TASLP 2023)

**Bài:** L. Zhang, X. Wang, E. Cooper, N. Evans, J. Yamagishi. [arXiv 2204.05177](https://arxiv.org/abs/2204.05177), [code](https://github.com/nii-yamagishilab/PartialSpoof).

| Mục | Nội dung |
|---|---|
| Vấn đề | Chỉ vài đoạn trong câu thật bị thay bằng tiếng nói sinh ra. Bộ phát hiện lấy trung bình theo thời gian bỏ sót đoạn ngắn: học trên LA rồi kiểm tra trên PS có EER 14,19% |
| Ý tưởng | Dữ liệu ghép từ ASVspoof 2019 LA, nhãn ở 6 độ phân giải 20–640 ms. Mô hình wav2vec 2.0 Large + 5 gMLP chấm điểm ở mọi độ phân giải cùng lúc |
| Con số chính | Mức câu 0,49%; mức đoạn từ 12,84% (20 ms) đến 5,19% (640 ms) |
| Phát hiện đáng nhớ | Trên hệ thống sinh chưa gặp, đoạn không chứa chỗ nối khó hơn hẳn, nên mô hình dựa một phần vào dấu vết chỗ nối |
| Dùng cho vấn đáp 1 | Đại diện hướng 3 (giả mạo một phần). Lý do không chọn: chưa có dữ liệu giả mạo một phần tiếng Việt công khai |
| Lưu ý khi trích | Tóm tắt ghi 0,77% trên PartialSpoof và 0,90% trên LA; Bảng VIII cho thấy 0,77% là kết quả trên LA. Dùng số trong bảng |

Cùng nhóm tác giả với AntiDeepfake và bài GRPO (X. Wang, J. Yamagishi, NII), và cũng dùng front-end SSL fine-tune cùng back-end. Front-end Large (317M) tốt hơn hẳn Base (95M): 0,35% so với 2,22% trên dev, là một bằng chứng sớm về vai trò kích thước backbone (liên quan cấu hình B1 và B2 ở GĐ5).

### Truy vết nguồn qua phân loại codec (Interspeech 2025, CORE A)

**Bài:** X. Chen, I-M. Lin, L. Zhang, J. Du, H. Wu, H.-y. Lee, J.-S. R. Jang. *Codec-Based Deepfake Source Tracing via Neural Audio Codec Taxonomy.* [ISCA Archive](https://www.isca-archive.org/interspeech_2025/chen25j_interspeech.html), [code](https://github.com/ResponsibleGenAI/CodecFake-Source-Tracing).

| Mục | Nội dung |
|---|---|
| Vấn đề | Không chỉ hỏi "thật hay giả" mà hỏi "giả bằng gì". Đoán theo tên hệ thống thì không làm được với hệ thống chưa gặp |
| Ý tưởng | Đoán 3 thuộc tính codec (lượng tử hoá, mục tiêu phụ, decoder). Một front-end wav2vec 2.0 (XLS-R 300M) dùng chung, 4 back-end AASIST, học đa nhiệm |
| Con số chính | F1 khoảng 96–97% trên codec đã gặp. Trên hệ thống CoSG thật, F1 của mục tiêu phụ (≤ 28,18%) và decoder (≤ 27,26%) thấp hơn đoán bừa (31,04% và 34,37%) |
| Dùng cho vấn đáp 1 | Đại diện hướng truy vết nguồn. Lý do không chọn: kết quả trên hệ thống thật còn kém đoán bừa, và chỉ có tiếng Anh |
| Thay thế | Nếu cần đủ 5 bài có hạng tra trực tiếp, bài này có thể thay CodecFake+ (đã ghi trong đề cương mục 2.6) |

### VietPrism (arXiv 2026, preprint)

**Bài:** M. Hoang, T. Le. *VietPrism: A Large-Scale Vietnamese Speech and Deepfake Corpus with Diverse Dialects and Code-Switching.* [arXiv 2609.30005](https://arxiv.org/abs/2609.30005).

| Mục | Nội dung |
|---|---|
| Dữ liệu | 993,4 giờ thật, 1.262 người nói, 5 nhóm phương ngữ, 47,1% có chêm tiếng Anh. Tiếng giả từ MiniMax, OmniVoice, Higgs Audio v3, VoxCPM2, ghép cặp cùng câu cùng người nói |
| Kết quả zero-shot | AntiDeepfake XLS-R-2B tốt nhất (trung bình 14,08%). **AntiDeepfake MMS-300M trung bình 51,08%**, như đoán bừa, với 99,99% trên VoxCPM2 |
| Khoảng trống | Chỉ zero-shot, không fine-tune, không ablation. Đúng như đã ghi ở mục 1 của plan |
| Lưu ý khi trích | Tổng giờ tiếng giả không nhất quán (3,1K / 3.190 / 2.765,74). Giờ theo phương ngữ trong phần lời không khớp bảng. Không có đường dẫn tải |

**Liên hệ đề tài, cần chú ý sớm:**

- **MMS-300M là cấu hình mốc R của đồ án**, và bản post-trained của nó gần như không phát hiện được tiếng Việt khi zero-shot (51,08%, đều ở mọi phương ngữ: 48,8–52,3%). Nếu kết quả này lặp lại trên dữ liệu của nhóm thì:
  - S0 sẽ ở mức đoán bừa, nên fine-tune là bắt buộc chứ không chỉ là một lựa chọn.
  - Câu hỏi "SFT có làm quên không" vẫn đo được qua tập ngoài miền khác (In-the-Wild, RADAR tiếng Anh), nhưng trên tiếng Việt thì không có gì để quên.
  - Nên ghi nhận như một kết quả, và cân nhắc thêm XLS-R-2B (zero-shot tốt nhất) vào bảng zero-shot ở GĐ3 như plan đã định.
- **EER 99,99% là thứ tự bị đảo.** Khi chấm ở GĐ3, nếu gặp EER trên 50%, kiểm tra chiều điểm trước. Nếu chiều điểm đúng, phân tích nguyên nhân. VietPrism khử nhiễu tiếng thật bằng mạng nơ-ron, một giả thuyết đáng ghi.
- VoxCPM2 và OmniVoice cũng có trong `fake/vi` của MLAAD (dữ liệu dự phòng), nên kết quả VietPrism là tham chiếu cho các hệ thống này.

### Speech DF Arena (arXiv 2025, preprint)

**Bài:** S. Dowerah và cộng sự. [arXiv 2509.02859](https://arxiv.org/abs/2509.02859), [leaderboard](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena).

| Mục | Nội dung |
|---|---|
| Nội dung | 14 tập kiểm tra, 12 mô hình mã nguồn mở + 3 thương mại, cùng một cách chấm |
| Con số đáng nhớ | XLSR-SLS: TB 13,84%, gộp 15,68%. W2V2-AASIST: TB 18,02%, gộp 19,47%. Whispeak (thương mại, 98,9M): TB 3,05% |
| Dùng cho | Xếp hạng các detector; lý do chọn XLSR-SLS đại diện hướng 2; khái niệm EER gộp |
| Lưu ý khi trích | Phần lời nói XLSR-SLS "thấp nhất trừ ASVspoof", nhưng bảng cho thấy XLSR-Mamba thấp hơn ở In-the-Wild và ASVspoof 2024. Một số con số ghi lệch nhau giữa bảng và lời |

**Liên hệ đề tài:** khái niệm EER gộp nên dùng ở GĐ5 khi gộp nhiều tập ngoài miền. Nếu EER gộp cao hơn hẳn EER trung bình, đó là dấu hiệu thang điểm trôi, liên quan trực tiếp đến phần hiệu chuẩn ở GĐ6.

### RADAR 2026 (APSIPA ASC 2026)

**Bài:** H.-T. Luong, X. Liu, I. Kukanov, Z. X. Chai, K. A. Lee. [arXiv 2605.09568](https://arxiv.org/abs/2605.09568) (v3, 25/5/2026). Header ghi APSIPA ASC 2026. Hội nghị không có trong CORE 2023.

| Mục | Nội dung |
|---|---|
| Nội dung | Cuộc thi phát hiện giả dưới các biến đổi âm thanh (nén, vang, nhiễu, hạ tần số lấy mẫu, ...) áp lên cả tiếng thật và giả |
| Tập đánh giá | 102.726 câu, 6 biến thể ngôn ngữ, 10 TTS. Tiếng Việt: 7.000 thật (Common Voice 3.000, FOSD 4.000), 6.438 giả (iFlytek, ElevenLabs, Cartesia, OpenAI, Fish Audio S2 Pro, Piper) |
| Con số chính | Baseline SSL-AASIST: 42,6%. Tốt nhất: 5,10% |
| Lưu ý khi trích | Chưa có kết quả theo ngôn ngữ. Số đội không nhất quán (22 / 20+1 / 19) |

**Liên hệ đề tài**

- Các số liệu về RADAR trong plan và đề cương (13.438 câu, 6 hệ thống sinh, 42,6%, 5,10%) **khớp với bài**.
- Bài **không** nói về chiều điểm số trong file nhãn. Ghi chú "RADAR quy ước ngược lại" trong plan (GĐ2) chưa được kiểm chứng từ bài, cần kiểm tra khi có file nhãn.
- Vì chưa có kết quả riêng cho tiếng Việt, bảng zero-shot ở GĐ3 trên RADAR tiếng Việt sẽ là số liệu mới.
- Tiếng thật tiếng Việt của RADAR có 3.000 câu từ Common Voice. Điều này củng cố quyết định trong plan: không thêm Common Voice tiếng Việt vào tập huấn luyện.

### Tak 2022 (Odyssey 2022)

**Bài:** H. Tak, M. Todisco, X. Wang, J. Jung, J. Yamagishi, N. Evans. *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation.* [ISCA Archive](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.html).

| Mục | Nội dung |
|---|---|
| Ý tưởng | Thay front-end sinc của AASIST bằng XLS-R 300M fine-tune, thêm self-attention gộp, tăng cường bằng RawBoost |
| Con số chính | 2021 LA 0,82% (TB 1,00%), 2021 DF 2,85% (TB 3,69%). Front-end SSL một mình đưa DF từ 21,06% xuống 7,69% |
| Dùng cho | Baseline của đồ án (W2V2-AASIST). Dẫn sang GĐ2: front-end SSL là phần đóng góp lớn nhất |
| Lưu ý khi trích | Con số hay được trích là **lần chạy tốt nhất**, nên ghi kèm trung bình. Cấu hình RawBoost phải hợp miền (DF dùng cấu hình LA thì 6,64%) |

**Liên hệ đề tài:** RawBoost `algo=5` trong repo (nối tiếp loại 1 và 2) chính là cấu hình bài dùng cho LA, tức kênh truyền và thiết bị. Ablation A0 nên ghi rõ điều này. Nếu dữ liệu tiếng Việt chủ yếu bị nén kiểu DF, cấu hình loại 3 có thể hợp hơn. Đây là một ý cho hướng phát triển, không thêm vào ablation.

## Bảng so sánh các hướng (điền dần)

| Hướng | Đầu vào | Đầu ra | Dữ liệu cần có | Chi phí | Hạn chế chính |
|---|---|---|---|---|---|
| 1. Post-training và fine-tune SSL | Dạng sóng | Điểm thật/giả (logit lớp "thật") | Post-training: ~75K giờ nhiều miền (đã có sẵn checkpoint). Fine-tune: vài chục giờ của miền đích | Post-training rất đắt (không tự làm). Fine-tune 300M vừa GPU 16 GB; GRPO thêm 2 bản mô hình và 64 mẫu mỗi câu | Fine-tune làm quên; GRPO giảm quên nhưng chưa hết; GRPO cần điểm xuất phát tốt |
| 2. SSL + back-end chuyên biệt | Dạng sóng (~4 giây) | Điểm thật/giả | Thường chỉ ASVspoof 2019 LA (~25K câu) + RawBoost | ~300M tham số, GPU tầm trung | Ngoài miền kém: 7–11% In-the-Wild, 42,6% RADAR; nhiều bài chỉ báo lần chạy tốt nhất |
| 3. Giả mạo một phần | Dạng sóng | Điểm mức câu + dãy điểm mức đoạn | Nhãn theo đoạn ở nhiều độ phân giải | Như hướng 2 (front-end SSL ~317M); thêm chi phí tạo dữ liệu ghép | Đoạn 20 ms vẫn ~13%; phụ thuộc dấu vết chỗ nối; dữ liệu chỉ tiếng Anh |
| 4. Audio LLM | Dạng sóng → log-Mel + câu hỏi bằng chữ | Token "Fake"/"Real" (điểm từ logit 2 token) | Như phân loại thường; tốt khi ít dữ liệu | Cao: 7,7B tham số, A100 | Ngoài miền 26,99%; chưa giải thích thật sự |
| 5. Phát hiện tiếng nói giả từ codec | Dạng sóng | Điểm thật/giả | Tiếng thật + bản tái tổng hợp qua nhiều codec công khai; phải chọn và cân bằng theo thuộc tính codec | Như hướng 2 (W2V2-AASIST); thêm chi phí tái tổng hợp (31 codec × 43K câu) | Tập test CoSG nhỏ; chỉ tiếng Anh; "qua codec là giả" mâu thuẫn khi codec dùng để truyền tiếng thật |
| 6. Truy vết nguồn | Dạng sóng | Nhãn thật/giả + loại lượng tử hoá, mục tiêu phụ, decoder | Tiếng giả có nhãn thuộc tính của hệ thống sinh | Như hướng 2, thêm vài back-end (chạy được trên RTX 4070) | Trên hệ thống chưa gặp, 2/3 nhiệm vụ kém hơn đoán bừa |
| Tiếng Việt (dữ liệu) | | | VSASV: replay, VC, nhiễu đối kháng. VietPrism: 4 TTS đời mới, cặp đối chứng | | VSASV: không có TTS thuần, nguy cơ lối tắt do nhiễu. VietPrism: chỉ zero-shot, chưa có link tải |

## Nháp câu trả lời vấn đáp

### Câu 1. Các hướng chính trên thế giới (khoảng 3 hướng)

> Đây là cách nhóm sắp xếp các hướng để trình bày, không phải phân loại chuẩn của lĩnh vực. Mỗi hướng nói trong khoảng 1 phút: vấn đề, ý tưởng, một con số, hạn chế.

**Hướng 1. Post-training và fine-tune mô hình SSL** (hướng nhóm chọn)
- Thêm một pha huấn luyện có nhãn thật/giả trên rất nhiều dữ liệu (~75K giờ, hơn 100 ngôn ngữ) giữa pre-training và fine-tune. Mô hình có sẵn đã phát hiện tốt mà chưa cần huấn luyện thêm: XLS-R-2B đạt 1,23% trên In-the-Wild.
- Vấn đề mở: fine-tune trên miền mới làm mô hình quên (XLS-R-2B trên In-the-Wild từ 1,24% lên 6,35% sau SFT). Bài GRPO (Interspeech 2026) dùng học tăng cường để giảm quên: trung bình ngoài miền giữ ở 2,69% thay vì 6,28%.

**Hướng 2. Front-end SSL kết hợp back-end chuyên biệt**
- Dùng XLS-R làm bộ trích đặc trưng, fine-tune cùng back-end thiết kế riêng. XLS-R + AASIST (Tak 2022) đưa EER trên ASVspoof 2021 DF từ 21% xuống 2,85%. XLSR-SLS (ACM MM 2024) dùng cả 24 lớp với trọng số theo từng câu, đứng đầu các mô hình mã nguồn mở trên Speech DF Arena.
- Hạn chế: hầu hết chỉ học trên ASVspoof 2019 LA. Ngoài miền vẫn kém: SSL-AASIST có EER 42,6% trên RADAR 2026.

**Hướng 3. Phát hiện và định vị giả mạo một phần**
- Kẻ gian chỉ thay vài từ trong bản ghi thật. Mô hình lấy trung bình cả câu bỏ sót đoạn ngắn: học trên dữ liệu giả cả câu thì EER 14,19% trên dữ liệu giả một phần. PartialSpoof (TASLP 2023) gán nhãn từ 20 đến 640 ms và chấm điểm ở nhiều độ phân giải: mức câu 0,49%, mức đoạn 20 ms khoảng 13%.
- Hạn chế: cần nhãn theo đoạn; mô hình dựa một phần vào dấu vết chỗ nối.

**Các hướng khác (kể thêm nếu còn thời gian)**
- **Audio LLM** (ALLM4ADD, ACM MM 2025): đặt bài toán thành hỏi đáp "audio này thật hay giả". Học tốt từ rất ít dữ liệu (dưới 5% với ~200 câu), nhưng 7,7 tỉ tham số và ngoài miền 26,99%.
- **Tiếng nói giả từ neural codec** (CodecFake+, TASLP 2026): TTS đời mới dựng dạng sóng bằng decoder của codec. Cho tiếng thật đi qua codec công khai để tạo dữ liệu huấn luyện; chọn dữ liệu theo loại decoder được 11,91%.
- **Truy vết nguồn** (Interspeech 2025): đoán thuộc tính của codec đã dùng. Trên hệ thống thật, 2/3 nhiệm vụ còn kém đoán bừa.
- **Continual learning và thích nghi ít mẫu** (RAWM, ADD-GP): cập nhật khi có kiểu tấn công mới mà không quên kiểu cũ.
- **Benchmark và độ bền** (Speech DF Arena, RADAR 2026): không mô hình nào tốt nhất ở mọi tập; nén, vang, nhiễu làm EER tăng mạnh.

### Câu 2. Vì sao chọn hướng này

1. **Baseline cũ không còn đủ:** SSL-AASIST có EER 42,6% trên RADAR 2026 đa ngôn ngữ, đội tốt nhất đạt 5,10%.
2. **Có nền tảng mạnh, công khai:** 14 checkpoint post-trained và code đầy đủ (train, fine-tune, GRPO). Zero-shot rất tốt trên In-the-Wild.
3. **Có vấn đề mở rõ ràng, được chính tác giả nêu:** bài GRPO chỉ fine-tune trên một bộ dữ liệu, ablation chỉ với XLS-R-2B, cơ chế chưa giải thích.
4. **Tiếng Việt còn trống:** VSASV không có tấn công TTS thuần. VietPrism chỉ đánh giá zero-shot, và cho thấy mô hình có sẵn không ổn định trên tiếng Việt: cùng một mô hình, EER từ 0% đến gần 100% tuỳ hệ thống sinh. Riêng MMS-300M post-trained có trung bình 51,08%.
5. **Làm được với GPU miễn phí:** fine-tune MMS-300M (~317M tham số) vừa GPU 16 GB, và có backbone tiếng Anh cùng kích thước (W2V-Large) để tách yếu tố ngôn ngữ.

**Vì sao không chọn các hướng khác:**

| Hướng | Lý do chính |
|---|---|
| SSL + back-end | Đã rất nhiều bài; dùng làm baseline |
| Giả mạo một phần | Chưa có dữ liệu giả mạo một phần tiếng Việt công khai |
| Audio LLM | ~7,7 tỉ tham số, không vừa GPU 16 GB |
| Codec / truy vết nguồn | Dữ liệu và kết quả chỉ có tiếng Anh; dựng lại cho tiếng Việt vượt khối lượng đồ án |
| Continual learning | Cần giao thức nhiều giai đoạn; để làm hướng phát triển |

## Kiểm tra điều kiện qua GĐ1

- [x] 5 thẻ nhóm A (XLSR-SLS là thẻ tạm, cần đọc toàn văn)
- [x] Bảng so sánh các hướng
- [x] Nháp câu vấn đáp 1 và 2
- [ ] Tải tay PDF XLSR-SLS và hoàn thiện thẻ
- [ ] Tra hạng Q1 của TASLP trên SCImago
- [ ] Tự nói mỗi hướng trong 1 phút không nhìn tài liệu
