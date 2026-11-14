# GĐ5. Ablation

Câu hỏi của giai đoạn: yếu tố nào quyết định kết quả trên tiếng Việt khi fine-tune (post-training, RawBoost, ngôn ngữ và kích thước backbone), và fine-tune có giám sát có làm mô hình quên kiến thức cũ không (câu hỏi 1, 2 của plan). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ5. Đầu vào từ GĐ4: `chore/GD4-Du-Lieu-Va-Pipeline.md` mục 4.2 (cấu hình đã chốt) và mục 7. File này bắt đầu ngày 14/11.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Chuẩn bị lần chạy 1: tập con SEA-Spoof tính ngay trên Kaggle, khoá file khi hai tiến trình chấm cùng ghi, bước chấm S0 | `src/score_zeroshot.py`, `kaggle/finetune/run.py`, `kaggle/archive_run.py`, `kaggle/push.sh` | Xong (14/11), đã thử trên máy với điểm của pilot |
| 1 | **Lần chạy 1: R seed 1, A0 seed 1**, chấm S0 trên SEA-Spoof | `orpheus-finetune` v2, `results/runs/orpheus-finetune-v2/` | **Xong** (14/11, 4,68 giờ); mục 3.1 |
| 1 | Báo kết quả lần 1 cho người dùng, chờ duyệt trước khi chạy tiếp | Mục 3.1, 4 | Đã báo (14/11); chờ chốt phát hiện 1, 2 |
| 1–2 | Lần 2–3: P0, B2, B1 (seed 1); N0 nếu còn quota | | Chưa |
| 3–4 | Lần 4: R seed 2, 3 | | Chưa |
| Trong lúc chờ GPU | Đọc sâu bài GRPO và `algo/rl.py` (chuẩn bị GĐ6) | Mục 6, `src/grpo_gradient_sim.py` | Xong (14/11) |
| Trong lúc chờ GPU | Script bảng kết quả trên máy; code cho P0 (khởi tạo fairseq, kiểm nạp trọng số trước khi huấn luyện) | `src/analyze_ablation.py`, `src/finetune.py --init fairseq` | Xong (14/11), thử trên máy; P0 chưa chạy trên Kaggle |

**Lịch:** hôm nay 14/11, hạn nộp 30/11. Quota Kaggle còn 29,26 giờ đến 7 giờ sáng 21/11 (kiểm 14/11 trước lần 1).

## 0. Đầu vào từ GĐ4

| Thứ | Giá trị | Nguồn |
|---|---|---|
| Dữ liệu fine-tune | `seaspoof_vi_train` 17,88 giờ (8.910 thật VIVOS + 8.910 giả, 5 hệ thống: VITS, Edge-TTS, ChatGPT, ElevenLabs, HeyGen) | GĐ4 mục 3 |
| Tập kiểm tra SEA-Spoof | `test_seen`, `test_f5` (F5-TTS để riêng), `test_minimax` (MiniMax để riêng), `test_cvreal` (tiếng thật Common Voice) | GĐ4 mục 3 |
| Tập ngoài miền | RADAR tiếng Việt (13.438 câu), `ood_forget` (In-the-Wild 1.000 + 1.000, RADAR EN 1.000 + 1.000) | GĐ3 mục 4.1 |
| Cấu hình SFT | 3 epoch, `max_batch_length` 50, lr đỉnh 1e-6, warmup 460 / 4.600 bước, chỉ kiểm tra dev sau epoch cuối (lưu mô hình cuối) | GĐ4 mục 4.2 (E1–E3) |
| S0 (dùng thẳng) trên RADAR tiếng Việt | MMS-300M 20,08%; W2V-Large 17,80%; W2V-Small 27,21%; MMS-300M NDA 18,07% | GĐ3 mục 5.6 |
| S0 trên `ood_forget` (In-the-Wild / RADAR EN) | MMS-300M 2,6 / 26,2; W2V-Large 2,0 / 23,5; W2V-Small 4,0 / 27,2 | GĐ3 mục 5.6 |
| Dòng tham chiếu | XLS-R-2B dùng thẳng: 13,71% trên RADAR tiếng Việt | GĐ3 mục 5.6 |
| Tập câu khó chung | `results/zeroshot/analysis/hard_ids_default7.csv` (493 câu RADAR tiếng Việt: 238 giả, 255 thật) | GĐ3 phát hiện 16 |
| Chi phí đo ở pilot | ~49 phút mỗi epoch khi hai GPU cùng huấn luyện; chấm sau fine-tune ~60–70 phút | GĐ4 mục 4.1, 5 |

**Kiểm lại trước lần 1 (14/11, code gốc):**
- Với `valid_step` = 10⁹, `main.py` (`fit`, `_fit_train_customized`) không kiểm tra dev giữa chừng; `_fit_valid` gọi một lần sau epoch cuối và `save_and_keep_only(num_to_keep=1)` lưu đúng mô hình đó. E3 (chọn mô hình cuối) khớp code.
- Lịch learning rate (`WarmAndExpDecayLRSchedule`, SpeechBrain 1.0.2, `speechbrain/nnet/schedulers.py`): tăng tuyến tính `lr × bước / 460` trong 460 bước, sau đó `lr × 0,01^((bước − 460) / 4.140)`. Cuối 3 epoch (~4.563 bước) còn ~1,0 × 10⁻⁸. Khớp E1.
- Ba file `hparams/ft_grpo/{mms_300m,w2v_large,w2v_small}_pre-post-sft.yaml` giống nhau, chỉ khác tên backbone (đường dẫn thư mục, `ckpt_path`, `model_name`). B1, B2 chỉ đổi backbone, giữ nguyên mọi siêu tham số.

## 1. Các cấu hình

Đổi một yếu tố mỗi lần. Mọi cấu hình fine-tune trên `seaspoof_vi_train` với cấu hình SFT ở mục 0.

| Mã | Backbone | Khởi tạo | RawBoost khi fine-tune | So với | Câu hỏi | Lần chạy |
|---|---|---|---|---|---|---|
| **R** | MMS-300M | Post-trained, bản Default | Có | | Mốc | 1 (seed 1), 4 (seed 2, 3) |
| S0 | MMS-300M | Post-trained, bản Default | Không fine-tune | R | Fine-tune có cần thiết? | GĐ3 (RADAR, `ood_forget`); lần 1 (SEA-Spoof) |
| A0 | MMS-300M | Post-trained, bản Default | **Không** | R | RawBoost khi fine-tune giúp bao nhiêu? | 1 |
| P0 | MMS-300M | **Fairseq gốc** (`base_300m.pt`) | Có | R | Post-training đóng góp bao nhiêu? | 2 |
| B2 | **W2V-Large** (tiếng Anh, 317M) | Post-trained | Có | R | Ngôn ngữ pre-training (cùng kích thước) | 2 |
| B1 | **W2V-Small** (tiếng Anh, 95M) | Post-trained | Có | B2 | Kích thước backbone | 3 |
| N0 | MMS-300M | Post-trained, **bản NDA** | Có | R | RawBoost khi post-train (không bắt buộc) | 3, nếu còn quota |

Thước đo chính: hệ thống để riêng (`test_f5`, `test_minimax`), RADAR tiếng Việt, mức quên (`ood_forget`). Báo thêm: tỉ lệ giả bị nhận là thật (FAR) và thật bị coi là giả (FRR) ở ngưỡng log-odds 0, actDCF − minDCF, tập câu khó, khoảng tin cậy bootstrap. EER, minDCF, actDCF: thấp hơn là tốt hơn.

## 2. Các lần chạy

| Lần | Kernel | GPU 0 | GPU 1 | Sau đó | Ước tính |
|---|---|---|---|---|---|
| 1 | `orpheus-finetune` v2 | R seed 1 | A0 seed 1 | Chấm S0 trên SEA-Spoof dev, test: MMS-300M, W2V-Large, W2V-Small, MMS-300M NDA | ~4,8 giờ |
| 2 | `orpheus-finetune` v3 | P0 seed 1 | B2 seed 1 | | ~4,1 giờ (theo số đo lần 1: 3,95 giờ + dựng môi trường) |
| 3 | | B1 seed 1 | N0 seed 1 | | ~4 giờ |
| 4 | | R seed 2 | R seed 3 | | ~4,1 giờ |

**Quyết định (14/11, người dùng):** dừng báo kết quả lần 1 và chờ duyệt. Sau khi duyệt, lần 2–4 chạy nối nhau: lần sau được đẩy ngay khi lần trước xong và đã lưu bằng chứng; chỉ dừng báo khi có lỗi, kết quả bất thường, hoặc khi xong cả GĐ5.

Mỗi lần huấn luyện chấm sau fine-tune trên `seaspoof_vi_dev`, `seaspoof_vi_test` (tập con tính ngay trên Kaggle, `metrics_subsets.csv`), `radar_vi`, `ood_forget`. Chấm luôn ở batch 1 trên dữ liệu sạch, không dùng EER dev lúc huấn luyện (GĐ4 phát hiện 10, 13).

**Lần 1, ước tính thời gian** (số đo của pilot): dựng môi trường ~5 phút; huấn luyện 3 × 1.521 bước × 1,93 s ≈ 147 phút; chấm sau fine-tune ~65–80 phút (hai GPU cùng chấm, 4 nhân CPU); chấm S0 (8 job, ~63 giờ âm thanh) ~55–60 phút trên hai GPU. Tổng ~4,8 giờ, còn ~24,5 giờ quota.

Chấm S0 thêm MMS-300M NDA (ngoài 3 backbone GĐ4 đã định): ~15 phút quota, cho mốc của N0 và cho so sánh Default/NDA trên hệ thống để riêng của SEA-Spoof mà không cần fine-tune.

## 3. Kết quả

### 3.1. Lần 1 (`orpheus-finetune` v2, 14/11, 4,68 giờ; `results/runs/orpheus-finetune-v2/`)

COMPLETE, không có cảnh báo nạp trọng số. R và A0 mỗi lần 4.563 bước (đúng 3 epoch × 1.521 batch), ~2,0 s mỗi bước, ~2,6 giờ huấn luyện, GPU 99,5%, bộ nhớ đỉnh 14,6 GB. Chấm sau fine-tune ~74 phút mỗi GPU, chấm S0 43,8 phút. **Lịch learning rate lệch E1** (phát hiện 1): R và A0 cùng lịch nên so với nhau công bằng; so với kế hoạch là fine-tune mạnh hơn.

Số liệu: `results/finetune/analysis/` (`python src/analyze_ablation.py`, đầu ra `analysis_output.txt`), điểm từng câu ở `results/finetune/<lần>/<tập>/scores.csv` và `results/zeroshot/<mô hình>/seaspoof_vi_*/`.

**Bảng chính, EER (%), thấp hơn là tốt hơn.** S0 = dùng thẳng, không fine-tune.

| Cấu hình | test_seen | test_f5 (để riêng) | test_minimax (để riêng) | test_cvreal | **RADAR VI** | **ItW** (mức quên) | RADAR EN (mức quên) |
|---|---|---|---|---|---|---|---|
| S0 MMS-300M | 0,81 | 0,57 | 0,00 | 7,44 | 20,08 | 2,60 | 26,20 |
| **R** seed 1 | 0,04 | 0,04 | 0,00 | 2,70 | **23,07** | 7,60 | 26,70 |
| **A0** seed 1 (không RawBoost) | 0,04 | 0,00 | 0,00 | 2,79 | **18,37** | 6,10 | 25,00 |
| S0 MMS-300M NDA | 0,89 | 0,76 | 0,04 | 2,10 | 18,07 | | |
| S0 W2V-Large | 0,73 | 0,72 | 0,04 | 3,04 | 17,80 | 2,00 | 23,50 |
| S0 W2V-Small | 0,93 | 2,11 | 0,15 | 6,90 | 27,21 | 4,00 | 27,20 |
| Tham chiếu XLS-R-2B dùng thẳng | | | | | 13,71 | | |

**Hai loại lỗi ở ngưỡng log-odds 0** (FAR: giả bị nhận là thật; FRR: thật bị coi là giả; %, thấp hơn là tốt hơn) và lệch hiệu chuẩn:

| Cấu hình | RADAR VI FAR | RADAR VI FRR | RADAR VI actDCF − minDCF | test_cvreal FRR | test_f5 FAR |
|---|---|---|---|---|---|
| S0 MMS-300M | 37,06 | **4,81** | 0,007 | 0,75 | 9,57 |
| R seed 1 | 19,54 | **29,21** | 0,199 | 8,50 | 1,16 |
| A0 seed 1 | 13,26 | **27,09** | 0,192 | 8,30 | 0,38 |

**Chênh lệch EER, KTC 95% bootstrap ghép cặp** (1.000 lượt, theo câu; không tính độ bất định do seed huấn luyện):

| So sánh | RADAR VI | ItW | RADAR EN | test_cvreal |
|---|---|---|---|---|
| R − S0 | **+2,98** [2,41; 3,67] | **+5,00** [3,80; 6,20] | +0,50 [−1,30; 2,30] | −4,74 [−5,59; −4,16] |
| A0 − S0 | **−1,71** [−2,22; −1,12] | **+3,50** [2,50; 4,60] | −1,20 [−2,80; 0,30] | −4,65 [−5,59; −4,00] |
| A0 − R | **−4,70** [−5,24; −4,22] | −1,50 [−2,30; −0,60] | −1,70 [−2,90; −0,60] | +0,09 [−0,29; 0,36] |
| R − XLS-R-2B (dùng thẳng) | +9,35 [8,60; 10,16] | | | |
| A0 − XLS-R-2B | +4,66 [3,98; 5,34] | | | |
| S0 NDA − S0 Default | −2,02 [−2,58; −1,47] | | | −5,34 [−6,24; −4,71] |

**Theo hệ thống sinh** (SEA-Spoof test, tiếng thật VIVOS 14 người nói; EER %): S0 MMS-300M từ 0,00 (ChatGPT, MiniMax) đến 1,82 (ElevenLabs); R và A0 ≤ 0,20 ở mọi hệ thống (`table_by_system.csv`).

**Tập 493 câu khó chung của GĐ3** (RADAR VI, sai ở ngưỡng EER của chính mô hình): S0 sai 100% (theo định nghĩa); R còn sai 83,2% câu giả và 72,5% câu thật; A0 88,2% và 72,2% (`hard_ids.csv`).

Mỗi kết quả, một câu giải thích (giả thuyết, 1 seed):
1. **Trong miền SEA-Spoof bão hoà ngay từ S0**, kể cả hai hệ thống để riêng (F5-TTS 0,57%, MiniMax 0,00%); fine-tune đưa về gần 0. Hệ thống để riêng của SEA-Spoof không phân biệt được các cấu hình (phát hiện 2).
2. **Fine-tune dịch cả phân bố điểm về phía "giả":** trên RADAR VI, FAR giảm (37 → 13–20%) nhưng FRR tăng gấp ~6 (4,8 → 27–29%), actDCF − minDCF từ 0,007 lên ~0,2. Khớp yếu tố gây nhiễu đã dự báo: tiếng thật train chỉ từ VIVOS, nên tiếng thật nguồn khác (Common Voice, FOSD) bị coi là giả nhiều hơn (phát hiện 3).
3. **RawBoost khi fine-tune làm kém ngoài miền:** A0 tốt hơn R 4,70 điểm trên RADAR VI, và tốt hơn ở cả ItW, RADAR EN. Cùng chiều với GĐ3 (bản NDA tốt hơn Default ở mọi backbone ≤ 317M) và với S0 trên `test_cvreal` (NDA 2,10% so với Default 7,44%) (phát hiện 4).
4. **SFT làm quên trên In-the-Wild:** 2,6 → 7,6% (R), 6,1% (A0), cùng hiện tượng bài GRPO báo cho MMS-300M (2,91 → 14,17%, Bảng 1 hàng 17–18). RADAR EN không đổi đáng kể. Trả lời sơ bộ câu hỏi 2: có quên, rõ trên ItW.
5. **Fine-tune 317M chưa thắng 2,2B dùng thẳng:** A0 vẫn kém XLS-R-2B 4,66 điểm trên RADAR VI.

## 4. Phát hiện ảnh hưởng thí nghiệm

| # | Phát hiện | Ảnh hưởng | Việc cần làm |
|---|---|---|---|
| 1 | **`--optimizer_step_limit` không ghi đè được vào YAML.** SpeechBrain coi đây là tham số chạy (run option) của `Brain`, nên `hyperparams.yaml` của lần chạy vẫn ghi `optimizer_step_limit: 50000` và `lr_scheduler.total_steps: 50000`; `--warmup 460` thì có tác dụng. Learning rate thực tế: tăng lên 1e-6 trong 460 bước, rồi chỉ giảm còn 6,83e-7 ở bước 4.563 (`exp_log.txt`), thay vì ~1e-8. Tổng learning rate cộng dồn qua 4.563 bước: 3,64 × 10⁻³, gấp ~3,25 lần lịch E1 dự kiến (1,12 × 10⁻³); cùng cỡ với cách bài GRPO chạy thực tế (ước 2,0–3,6 × 10⁻³ cho ~18–24 nghìn bước với warmup 80.000, theo ước lượng số bước ở GĐ4 phát hiện 9), nhưng đỉnh của ta cao hơn (1e-6 so với ~2–3 × 10⁻⁷) và dồn trong ít bước hơn. Pilot GĐ4 không lộ ra vì chỉ chạy 300 bước trong warmup | R và A0 cùng lịch nên so sánh R–A0 vẫn công bằng. Lần 2–4 phải dùng **cùng một lịch** với lần 1 thì mới so được với R. So với bài GRPO, ta fine-tune mạnh hơn nữa (GĐ4 phát hiện 9) | **Đã chốt 14/11 (E1'):** giữ lịch thực tế cho cả GĐ5–6, không chạy lại (người dùng giao Claude quyết; lý do ở mục 4.1). Mọi lần sau giữ nguyên bộ ghi đè của lần 1 để YAML vẫn ra `total_steps: 50000` |
| 2 | **Hệ thống để riêng của SEA-Spoof đã dễ với mô hình dùng thẳng:** F5-TTS 0,57%, MiniMax 0,00% (S0 MMS-300M); mọi hệ thống ≤ 2,1% với cả 4 mô hình S0 | Thước đo "tổng quát hoá trong tiếng Việt sang hệ thống chưa gặp" của plan không phân biệt được cấu hình. Thước đo chính còn lại: RADAR VI, mức quên, tiếng thật nguồn khác (`test_cvreal`, FRR) | **Đã chốt 14/11:** thêm tập `fake/vi` của MLAAD (LEMAS-TTS, OmniVoice, VieNeu-TTS, VoxCPM2, vixTTS) ghép với tiếng thật VIVOS của test; chuẩn bị song song, không chặn lần 2; phải kiểm lối tắt (giả và thật từ hai kho) |
| 3 | **Fine-tune làm tiếng thật nguồn khác bị coi là giả** (FRR ở ngưỡng 0: RADAR VI 4,8 → 27–29%; `test_cvreal` 0,75 → 8,3–8,5%) và làm lệch hiệu chuẩn (actDCF − minDCF ~0,2) | Quyết định Q6 của GĐ3 ("giảm kỳ vọng vào hiệu chuẩn") cần xem lại: sau fine-tune, hiệu chuẩn có ý nghĩa. EER vẫn là thước đo không phụ thuộc ngưỡng | Báo FAR/FRR cho mọi cấu hình. GĐ6 ngày 3: hiệu chuẩn trên dev (dev cũng chỉ có tiếng thật VIVOS, nên có thể không sửa được lệch trên nguồn khác; ghi khi làm) |
| 4 | **RawBoost (khi fine-tune) làm kém ngoài miền ở MMS-300M**, cả ba tập ngoài miền, KTC theo câu không chứa 0 | Cấu hình mốc R (có RawBoost, theo bài GRPO) là cấu hình kém hơn. Một seed: chưa biết chênh lệch do seed lớn cỡ nào | R seed 2, 3 (lần 4) cho độ lệch giữa các seed. Không đổi R giữa chừng: các cấu hình khác so với R như plan |

### 4.1. Vì sao không chạy lại lần 1 (quyết định E1', 14/11)

| Câu hỏi | Trả lời |
|---|---|
| So sánh trong GĐ5–6 có còn đúng không? | Có, miễn mọi cấu hình (P0, B2, B1, N0, seed 2–3, GRPO) dùng đúng lịch của lần 1. Ablation chỉ cần "đổi một yếu tố", không cần lịch nào cụ thể |
| Lịch thực tế có vô lý không? | Không: warmup 10% rồi gần như giữ 1e-6 (còn 6,8e-7 ở cuối). Tổng learning rate cộng dồn 3,64 × 10⁻³, gần mức bài GRPO chạy thực tế (ước 2,0–3,6 × 10⁻³) hơn lịch E1 dự kiến (1,12 × 10⁻³) |
| Mất gì nếu không chạy lại? | (1) Báo cáo phải ghi lịch đúng như đã chạy, không ghi E1. (2) Mức quên tuyệt đối (ItW 2,6 → 7,6%) gắn với độ mạnh fine-tune này; không nói được nó sẽ ra sao với lịch nhẹ hơn. (3) Learning rate không giảm về gần 0 ở cuối, nên mô hình cuối có thể dao động theo seed nhiều hơn; R 3 seed sẽ đo |
| Được gì? | Tiết kiệm ~4 giờ quota và ~4 giờ chờ. Hiện tượng cần nghiên cứu (SFT làm quên) đã lộ rõ; lịch nhẹ hơn nhiều khả năng làm nó mờ đi và khó thấy khác biệt SFT–GRPO ở GĐ6 |
| Việc thêm nếu còn quota (không bắt buộc) | Một lần R seed 1 với lịch E1 đúng nghĩa, để có điểm thứ hai về "độ mạnh fine-tune so với mức quên" (biến chỗ lệch thành một ablation phụ). Sửa bằng cách ghi đè `total_steps` của `lr_scheduler`, cần kiểm cú pháp ghi đè lồng của SpeechBrain |

## 5. Ngân sách quota

| Lần | Quota dùng | Còn lại | Ghi chú |
|---|---|---|---|
| Trước GĐ5 | 0,74 giờ (pilot GĐ4) | 29,26 giờ đến 21/11 | |
| 1 (`orpheus-finetune` v2) | 4,68 giờ | 24,58 giờ | Ước 4,8 giờ; khớp |

## 6. Chuẩn bị GĐ6: đọc sâu bài GRPO và `algo/rl.py` (làm trong lúc chờ GPU, 14/11)

Bài: `resource/paper/GRPO-2603.02914.pdf` (arXiv v1, 3/3/2026), đọc toàn văn kể cả phụ lục. Code: AntiDeepfake commit `86098091`: `algo/rl.py`, `main.py` (`compute_forward`, `compute_objectives`, `__init__`), `models/W2V.py` (`forward_grpo`), `hparams/ft_grpo/mms_300m_pre-post-grpo_v{1,2}.yaml`, `README_grpo.md`.

### 6.1. Số liệu MMS-300M trong Bảng 1 của bài (trung bình 3 lần, EER %, thấp hơn là tốt hơn)

| Hàng | Fine-tune | DFE24 (trong miền, trung bình 5 độ dài) | ADD23 | FoR | DV | ItW | Trung bình ngoài miền |
|---|---|---|---|---|---|---|---|
| 17 | Không | 29,23 | 7,96 | 1,51 | 2,35 | 2,91 | 3,68 |
| 18 | SFT | 12,23 | 8,81 | 3,61 | 4,37 | 14,17 | 7,74 |
| 19 | GRPO_s (không clip, không mô hình cũ) | 11,65 | 5,70 | 1,40 | 2,91 | 8,31 | 4,58 |
| 20 | GRPO | 11,94 | 4,99 | 1,57 | 2,64 | 5,91 | 3,78 |

Bài chỉ làm ablation (β = 0, β = 1, không phần thưởng âm) và so pre-train/post-train với XLS-R-2B (hàng 7–12), không với MMS-300M.

### 6.2. Đối chiếu bài và code

| Nội dung | Bài | Code | Khớp? |
|---|---|---|---|
| Checkpoint khởi tạo | `mms_300m.ckpt` trên Zenodo (chú thích 4) | YAML ghi `ckpt_path: .../mms_300m_1000k.pt`, nhưng chỉ là giá trị mặc định, bị ghi đè bằng `--ckpt_path` | **Khớp với R của ta** (cùng file, bản Default; bản HF giống Zenodo, GĐ3 mục 2). Củng cố quyết định Q1 của GĐ3 |
| Công thức fine-tune | Như AntiDeepfake, tối đa 10 epoch, kiểm dev mỗi 20k bước, chọn EER dev thấp nhất; DFE24 ~50 giờ, cắt câu ≤ 15 s; 1 GPU H100 | YAML: `valid_step: 20000`, `number_of_epochs: 10` | Khớp. Ta khác ở lr, batch, số epoch, cách chọn checkpoint (GĐ4 mục 4.2) |
| Lấy mẫu nhãn | ỹ lấy từ mô hình "cũ" p_Θ̂ (công thức 3) | `forward_grpo`: `torch.multinomial(softmax(logits))` của mô hình **hiện tại**, G = 64 | Lệch nhỏ (Θ̂ chỉ cách Θ tối đa 1.000 bước) |
| Chuẩn hoá phần thưởng | Theo nhóm G mẫu **của từng câu** (công thức 4, 13) | `(r − r.mean()) / (r.std() + 1e-5)` trên **toàn bộ B × 64 mẫu của batch** | **Không khớp.** Hệ quả ở mục 6.3 |
| Clip tỉ số chính sách | ε = 0,2 (Bảng 2) | `grpo3_loss`: `min(r·A, clamp(r, 0,8, 1,2)·A)` | Khớp |
| Phạt lệch mô hình tham chiếu | KL xấp xỉ, β = 0,04 | Ước lượng k3 trên nhãn đã lấy mẫu: `exp(ref − logp) − (ref − logp) − 1`; β = 0,04 | Khớp |
| Cập nhật mô hình cũ | Mỗi 1k bước (Bảng 2) | `inner_step_size: 1000`, đếm theo `self.step` (số batch) | Khớp |
| GRPO_s | Bỏ clip và Θ̂ | `grpo2_loss`: `exp(logp − logp.detach())·A` | Khớp |
| GRPO không phần thưởng âm | A = δ(ỹ = y) | `grpo3_degraded_ref` (phần thưởng 0/1, không chuẩn hoá); **không có file YAML** | Có trong code, phải tự thêm cấu hình nếu dùng |
| Chú thích 7 | "SFT trên mô hình post-trained (hàng 10) tốt hơn trên mô hình pre-trained (hàng 2)" | Bảng 1: hàng 2 là post-trained, hàng 10 là chỉ pre-train | Bài ghi nhầm số hàng; nội dung vẫn đúng theo bảng (ngoài miền 6,28 so với 19,34) |
| Bộ nhớ | | `ref_detector`, `old_detector` (GRPO đủ) là hai bản đầy đủ của mô hình trên GPU | GRPO_s chỉ thêm một bản. Pilot SFT đã 14,6 / 15 GB ở batch 50 s, nên GRPO phải giảm batch (GĐ4 mục 4.1) |

### 6.3. Hệ quả của việc chuẩn hoá theo batch (tự suy ra từ code, đã kiểm bằng mô phỏng)

Gọi p = p(y_đúng | x), z là log-odds của nhãn đúng. Tại thời điểm mô hình cũ vừa cập nhật (tỉ số = 1), bỏ KL, kỳ vọng gradient theo z của mỗi câu (`src/grpo_gradient_sim.py`, Monte Carlo 4.000 lượt, B = 10, G = 64, khớp dự đoán đến 3 chữ số):

| Cách fine-tune | Kỳ vọng gradient theo z | Câu được đẩy mạnh nhất |
|---|---|---|
| SFT | 1 − p | Câu mô hình đang sai nặng (p → 0) |
| GRPO trong code (chuẩn hoá theo batch) | p(1 − p) / σ_batch | Câu ở ranh giới (p ≈ 0,5); câu đang sai nặng gần như không đóng góp |
| GRPO không phần thưởng âm | p(1 − p) | Như trên, nhưng nhỏ hơn 1/σ_batch lần |

σ_batch = độ lệch chuẩn của phần thưởng trên batch, ≈ √(r̄(1 − r̄)) với r̄ là tỉ lệ nhãn lấy mẫu đúng. Batch đúng 95% thì 1/σ ≈ 4,6; đúng 99% thì ≈ 10.

Ý nghĩa (giả thuyết cho GĐ6, chưa kiểm trên dữ liệu):
1. So với SFT, GRPO nhân trọng số mỗi câu thêm p/σ_batch: bỏ qua câu mô hình đang sai chắc chắn, học chủ yếu từ câu ở ranh giới. Đây là một cơ chế khả dĩ cho việc GRPO quên ít hơn (không bị kéo mạnh bởi câu khác hẳn dữ liệu cũ), khớp nhận xét ở phụ lục A.1 của bài ("ưu tiên các mode xác suất cao, như chính quy hoá ngầm").
2. Trong code, "phần thưởng âm" không đổi **hướng** kỳ vọng gradient so với biến thể không phần thưởng âm, chỉ đổi **độ lớn** (×1/σ_batch, một kiểu learning rate tự điều chỉnh) và phương sai. Bài kết luận phần thưởng âm là yếu tố then chốt (hàng 9 so với hàng 4, mục 3.5); số liệu của bài chưa tách được phần do độ lớn bước cập nhật.
3. Với SEA-Spoof, mô hình dùng thẳng đã gần bão hoà (dev ~1,16%, GĐ4 phát hiện 14), nên đa số câu có p gần 1 và σ_batch nhỏ: GRPO sẽ có bước cập nhật hiệu dụng lớn hơn SFT nhiều lần ở câu ranh giới, nhưng gần như không học từ câu giả bị nhận là thật một cách chắc chắn (ví dụ VITS: mô hình pilot GĐ4, gần như chưa đổi so với S0, nhận 42,0% câu giả VITS của dev là thật ở ngưỡng 0, trong khi EER chỉ 1,93%; tính lại 14/11 từ `results/finetune/pilot_R/seaspoof_vi_dev/scores.csv` bằng `src/eval_seaspoof.py`).

Việc cho GĐ6 rút ra: (a) ghi EER theo từng hệ thống cho G1 so với R, vì giả thuyết 1 dự đoán GRPO kém SFT ở hệ thống mô hình đang sai chắc chắn; (b) cân nhắc thêm một biến thể rẻ, không cần lấy mẫu: SFT với trọng số p(y|x) mỗi câu (đúng bằng kỳ vọng gradient của GRPO trong code, trừ KL và hệ số 1/σ), để kiểm giả thuyết 1–2 trực tiếp. Chưa đưa vào kế hoạch; chờ kết quả GĐ5 và người dùng chốt.

### 6.4. Mốc cho P0: Bảng III bài AntiDeepfake (arXiv v4, mục IV-D)

Fine-tune trên Deepfake-Eval-2024 (90% file train để fine-tune, 10% chọn checkpoint), **6k bước, AdamW, lr 1e-6, có RawBoost**; chấm test DFE24 cắt 4/10/13/30/50 s. EER (%), thấp hơn là tốt hơn:

| Backbone | Post-train + fine-tune (4 / 10 / 13 / 30 / 50 s) | Chỉ pre-train + fine-tune | Post-train giúp (điểm, 4 s / 50 s) |
|---|---|---|---|
| W2V-Large | 19,56 / 12,11 / 10,95 / 10,52 / 11,36 | 24,42 / 22,46 / 22,14 / 21,17 / 21,53 | −4,9 / −10,2 |
| **MMS-300M** | 17,15 / 13,38 / 12,32 / 11,18 / 10,79 | 19,77 / 13,26 / 12,72 / 12,01 / 12,30 | −2,6 / −1,5; **ở 10 s kém hơn 0,12** |
| MMS-1B | 12,11 / 10,40 / 10,02 / 8,60 / 9,35 | 19,85 / 13,57 / 11,56 / 11,04 / 11,40 | −7,7 / −2,1 |
| XLS-R-1B | 11,86 / 10,00 / 9,27 / 8,49 / 8,28 | 19,96 / 17,19 / 16,31 / 10,71 / 11,20 | −8,1 / −2,9 |
| XLS-R-2B | 12,14 / 9,80 / 9,99 / 9,46 / 9,56 | 12,88 / 10,76 / 10,34 / 9,68 / 9,84 | −0,7 / −0,3 |

Cách đọc cho GĐ5:
1. Bài viết "post-training cải thiện nhất quán và rõ ràng ở mọi mô hình", nhưng với **MMS-300M lợi ích trong miền nhỏ** (≤ 2,6 điểm, ở 10 s còn kém hơn). W2V-Large lợi nhiều nhất (~10 điểm ở 50 s). Dự đoán: P0 so với R **trên test_seen** có thể chênh ít; khác biệt nếu có sẽ ở hệ thống để riêng, RADAR và mức quên. Bài GRPO cho thấy điều này với XLS-R-2B: SFT từ bản chỉ pre-train có ngoài miền trung bình 19,34% so với 6,28% từ bản post-trained (Bảng 1 hàng 10, 2), trong khi trong miền gần như bằng nhau (10,45 so với 10,26).
2. Công thức fine-tune của Bảng III (lr 1e-6, 6k bước) gần cấu hình E1 của ta (đỉnh 1e-6, 4.600 bước) hơn là cách bài GRPO chạy thực tế (GĐ4 phát hiện 9). Bài không ghi lịch warmup.
3. Bài không nói rõ bản Default hay NDA được fine-tune ("best-performing post-trained models", mục IV-C).

## 7. Kiểm tra điều kiện qua GĐ5

- [ ] Seed 1 của 5 cấu hình cần huấn luyện: R, A0, P0, B2, B1
- [ ] R đủ 3 seed
- [ ] Bảng EER, minDCF (trung bình, độ lệch chuẩn cho R), một câu giải thích cho từng kết quả
- [ ] Phân tích riêng theo từng hệ thống sinh
- [ ] Bằng chứng mọi lần chạy trong `results/runs/`
