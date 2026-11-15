# GĐ5. Ablation

Câu hỏi của giai đoạn: yếu tố nào quyết định kết quả trên tiếng Việt khi fine-tune (post-training, RawBoost, ngôn ngữ và kích thước backbone), và fine-tune có giám sát có làm mô hình quên kiến thức cũ không (câu hỏi 1, 2 của plan). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ5. Đầu vào từ GĐ4: `chore/GD4-Du-Lieu-Va-Pipeline.md` mục 4.2 (cấu hình đã chốt) và mục 7. File này bắt đầu ngày 14/11.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Chuẩn bị lần chạy 1: tập con SEA-Spoof tính ngay trên Kaggle, khoá file khi hai tiến trình chấm cùng ghi, bước chấm S0 | `src/score_zeroshot.py`, `kaggle/finetune/run.py`, `kaggle/archive_run.py`, `kaggle/push.sh` | Xong (14/11), đã thử trên máy với điểm của pilot |
| 1 | **Lần chạy 1: R seed 1, A0 seed 1**, chấm S0 trên SEA-Spoof | `orpheus-finetune` v2, `results/runs/orpheus-finetune-v2/` | **Xong** (14/11, 4,68 giờ); mục 3.1 |
| 1 | Báo kết quả lần 1 cho người dùng, chờ duyệt trước khi chạy tiếp | Mục 3.1, 4 | Đã báo (14/11); chờ chốt phát hiện 1, 2 |
| 1–2 | Lần 2: P0, B2 seed 1 | `orpheus-finetune` v3, mục 3.2 | **Xong** (15/11 ~1:46, 4,14 giờ) |
| 1–2 | Lần 3: B1, N0 seed 1 | `orpheus-finetune-b` v1, mục 3.3 | **Xong** (15/11 ~2:03, 3,89 giờ) |
| 3–4 | Lần 4: R seed 2, 3; MLAAD | `orpheus-finetune` v4, mục 3.4 | **Xong** (15/11 8:30, 4,94 giờ) |
| 4 | Tổng kết GĐ5 | Mục 8 | Xong (15/11) |
| Trong lúc chờ GPU | Đọc sâu bài GRPO và `algo/rl.py` (chuẩn bị GĐ6) | Mục 6, `src/grpo_gradient_sim.py` | Xong (14/11) |
| Trong lúc chờ GPU | Script bảng kết quả trên máy; code cho P0 (khởi tạo fairseq, kiểm nạp trọng số trước khi huấn luyện) | `src/analyze_ablation.py`, `src/finetune.py --init fairseq` | Xong (14/11), thử trên máy; P0 chưa chạy trên Kaggle |

**Lịch:** hôm nay 14/11, hạn nộp 30/11. Quota Kaggle còn 29,26 giờ đến 7 giờ sáng 21/11 (kiểm 14/11 trước lần 1).

## 0. Đầu vào từ GĐ4

| Thứ | Giá trị | Nguồn |
|---|---|---|
| Dữ liệu fine-tune | `seaspoof_vi_train` 17,88 giờ (8.910 thật VIVOS + 8.910 giả, 5 hệ thống: VITS, Edge-TTS, ChatGPT, ElevenLabs, HeyGen) | GĐ4 mục 3 |
| Tập kiểm tra SEA-Spoof | `test_seen`, `test_f5` (F5-TTS để riêng), `test_minimax` (MiniMax để riêng), `test_cvreal` (tiếng thật Common Voice) | GĐ4 mục 3 |
| Tập ngoài miền | RADAR tiếng Việt (13.438 câu), `ood_forget` (In-the-Wild 1.000 + 1.000, RADAR EN 1.000 + 1.000) | GĐ3 mục 4.1 |
| Cấu hình SFT | 3 epoch, `max_batch_length` 50, lr đỉnh 1e-6, warmup 460 / 4.600 bước, chỉ kiểm tra dev sau epoch cuối (lưu mô hình cuối). **Lịch learning rate thực tế khác E1** (`total_steps` vẫn 50.000, lr cuối 6,83e-7): phát hiện 1, quyết định E1' mục 4.1 | GĐ4 mục 4.2 (E1–E3) |
| S0 (dùng thẳng) trên RADAR tiếng Việt | MMS-300M 20,08%; W2V-Large 17,80%; W2V-Small 27,21%; MMS-300M NDA 18,07% | GĐ3 mục 5.6 |
| S0 trên `ood_forget` (In-the-Wild / RADAR EN) | MMS-300M 2,6 / 26,2; W2V-Large 2,0 / 23,5; W2V-Small 4,0 / 27,2 | GĐ3 mục 5.6 |
| Dòng tham chiếu | XLS-R-2B dùng thẳng: 13,71% trên RADAR tiếng Việt | GĐ3 mục 5.6 |
| Tập câu khó chung | `results/zeroshot/analysis/hard_ids_default7.csv` (493 câu RADAR tiếng Việt: 238 giả, 255 thật) | GĐ3 phát hiện 16 |
| Chi phí đo ở pilot | ~49 phút mỗi epoch khi hai GPU cùng huấn luyện; chấm sau fine-tune ~60–70 phút | GĐ4 mục 4.1, 5 |

**Kiểm lại trước lần 1 (14/11, code gốc):**
- Với `valid_step` = 10⁹, `main.py` (`fit`, `_fit_train_customized`) không kiểm tra dev giữa chừng; `_fit_valid` gọi một lần sau epoch cuối và `save_and_keep_only(num_to_keep=1)` lưu đúng mô hình đó. E3 (chọn mô hình cuối) khớp code.
- Lịch learning rate (`WarmAndExpDecayLRSchedule`, SpeechBrain 1.0.2, `speechbrain/nnet/schedulers.py`): tăng tuyến tính `lr × bước / 460` trong 460 bước, sau đó `lr × 0,01^((bước − 460) / 4.140)`. Cuối 3 epoch (~4.563 bước) còn ~1,0 × 10⁻⁸. Khớp E1. **Sửa sau lần 1:** công thức đúng, nhưng lần chạy không dùng `total_steps` = 4.600 như giả định ở đây: SpeechBrain giữ `optimizer_step_limit` (nên cả `total_steps`) = 50.000 của YAML. Kiểm trước lần 1 chỉ đọc code và lệnh truyền vào, không đọc `hyperparams.yaml` đã giải; pilot GĐ4 (300 bước, nằm trong warmup) không lộ ra. Phát hiện 1, mục 4.1.
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
| 3 | `orpheus-finetune-b` v1 (kernel thứ hai, song song với lần 2) | B1 seed 1 | N0 seed 1 | | ~4 giờ |
| 4 | kernel nào rảnh trước | R seed 2 | R seed 3 | Chấm thêm `mlaad_vi_test` cho R seed 2, 3; chấm S0 trên MLAAD (MMS-300M, W2V-Large, W2V-Small, MMS-300M NDA) | ~5,4 giờ (thêm MLAAD 14,09 giờ âm thanh mỗi mô hình) |

**Chạy song song (14/11, người dùng đồng ý):** lần 2–4 độc lập nhau (không lần nào dùng kết quả của lần khác), tổng quota như nhau, nên chạy song song trên hai kernel: `orpheus-finetune` và `orpheus-finetune-b` (cùng `run.py`, `push.sh push finetune-b`). Một kernel chỉ chạy một version mỗi lúc. Giới hạn số phiên GPU cùng lúc của Kaggle chưa kiểm; lần 2 và lần 3 cùng `QUEUED` lúc 18:30. **Xếp hàng lâu (14/11):** lúc 20:15 cả hai vẫn `QUEUED` (lần 2 ~2 giờ 10 phút, lần 3 ~1 giờ 45 phút), quota không bị trừ; lần 1 sáng cùng ngày được cấp máy gần như ngay. CLI không cho biết lý do hay vị trí trong hàng đợi, và không có lệnh huỷ (chỉ huỷ được trên web), nên không đẩy lại (có thể sinh hai phiên cho cùng một lần chạy). Người dùng chọn tiếp tục chờ; phương án dự phòng: P100 (một GPU, hai lần huấn luyện chạy nối nhau) hoặc thuê server. Lần 2 được cấp máy khoảng 21:33 (xếp hàng ~3 giờ 28 phút); lần 3 vẫn `QUEUED` lúc 21:37. Lần 3 được cấp máy khoảng 22:10 (suy từ quota) trong khi lần 2 vẫn chạy: **Kaggle cho hai phiên GPU cùng lúc**, quota trừ theo từng phiên. Lúc ~1:00 ngày 15/11 phiên OAuth của Kaggle CLI lại hết hạn (lần thứ hai trong ngày), nên không tải kết quả và không đẩy lần 4 được cho tới khi người dùng đăng nhập lại; kernel trên Kaggle không bị ảnh hưởng. Người dùng hoãn thuê server (đã soạn cấu hình vast.ai: 2 × GPU ≥ 24 GB như RTX 3090/4090/A5000/A6000, không RTX 50xx, ≥ 16 CPU, ≥ 64 GB RAM, ≥ 150 GB ổ, on-demand, không đặt máy ở Trung Quốc). Lần 2 không tải lại `orpheus-src` để lần 2 (đang xếp hàng) chạy đúng bản code lúc đẩy.

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

### 3.2. Lần 2 (`orpheus-finetune` v3, 14–15/11, 4,14 giờ; `results/runs/orpheus-finetune-v3/`)

COMPLETE, không có cảnh báo nạp trọng số. P0 nạp đủ 431 tensor từ `base_300m.pt` (kiểm trước khi huấn luyện, 0 cảnh báo), lớp FC ngẫu nhiên; dev 0,00% sau 3 epoch. Cùng bộ ghi đè và lịch learning rate với lần 1.

**EER (%), thấp hơn là tốt hơn** (dòng S0 và R, A0 lặp lại để so):

| Cấu hình | test_seen | test_f5 | test_minimax | test_cvreal | **RADAR VI** | **ItW** | RADAR EN |
|---|---|---|---|---|---|---|---|
| R seed 1 (MMS-300M post-trained) | 0,04 | 0,04 | 0,00 | 2,70 | 23,07 | 7,60 | 26,70 |
| **P0** seed 1 (MMS-300M, chỉ pre-train) | 0,04 | **5,94** | 0,00 | 3,26 | **31,46** | **14,90** | **33,90** |
| **B2** seed 1 (W2V-Large post-trained) | 0,04 | 0,00 | 0,00 | 2,59 | **21,03** | **1,40** | 23,40 |
| S0 W2V-Large (dùng thẳng) | 0,73 | 0,72 | 0,04 | 3,04 | 17,80 | 2,00 | 23,50 |
| S0 MMS-300M (dùng thẳng) | 0,81 | 0,57 | 0,00 | 7,44 | 20,08 | 2,60 | 26,20 |

**Lỗi ở ngưỡng 0 (%):**

| Cấu hình | RADAR VI FAR | RADAR VI FRR | RADAR VI actDCF − minDCF | test_cvreal FRR | test_f5 FAR |
|---|---|---|---|---|---|
| R seed 1 | 19,54 | 29,21 | 0,20 | 8,50 | 1,16 |
| P0 seed 1 | 23,41 | **53,37** | **0,54** | **48,85** | **54,80** |
| B2 seed 1 | 18,36 | 25,64 | 0,14 | 12,75 | 0,51 |

**Chênh lệch EER, KTC 95% bootstrap ghép cặp (theo câu):**

| So sánh | test_f5 | RADAR VI | ItW | RADAR EN |
|---|---|---|---|---|
| P0 − R (bỏ post-training) | +5,90 [5,33; 6,59] | **+8,39** [7,44; 9,14] | **+7,30** [5,60; 8,90] | **+7,20** [5,10; 9,40] |
| B2 − R (tiếng Anh thay đa ngôn ngữ, cùng 317M) | −0,04 [−0,11; 0,00] | **−2,04** [−2,84; −1,37] | **−6,20** [−7,40; −5,00] | **−3,30** [−5,40; −1,30] |
| B2 − S0 W2V-Large (fine-tune W2V-Large) | −0,72 [−0,96; −0,48] | +3,23 [2,70; 3,81] | −0,60 [−1,10; 0,00] | −0,10 [−1,80; 1,20] |

**Tập 493 câu khó (RADAR VI, % còn sai ở ngưỡng EER của chính mô hình):** R 83,2 / 72,5 (giả / thật); B2 95,0 / 83,9; **P0 54,6 / 60,8**.

Mỗi kết quả, một câu giải thích (1 seed):
1. **Post-training đóng góp gần như toàn bộ phần ngoài miền, không đóng góp trong miền:** P0 và R bằng nhau trên `test_seen` (0,04%), nhưng P0 kém 7–8 điểm ở RADAR VI, ItW, RADAR EN và kém 5,9 điểm ở F5-TTS (hệ thống để riêng). Đúng điều đã dự đoán từ Bảng III bài AntiDeepfake và Bảng 1 bài GRPO (mục 6.4): trong miền gần như nhau, khác biệt nằm ở ngoài miền.
2. **P0 lệch hiệu chuẩn nặng:** coi 53% tiếng thật RADAR VI và 49% tiếng thật Common Voice là giả ở ngưỡng 0, và để lọt 55% câu F5-TTS. Mô hình không post-train học "thật = VIVOS" mạnh hơn hẳn (yếu tố gây nhiễu ở phát hiện 3, phóng đại).
3. **P0 sai ở những câu khác:** chỉ còn sai ~55–61% tập câu khó chung (các mô hình post-trained sai 72–95%). Tập câu khó được định nghĩa từ lỗi của các mô hình post-trained, nên kết quả này ủng hộ giả thuyết của GĐ3 (phát hiện 16): lỗi chung bắt nguồn từ dữ liệu post-training. P0 có EER chung tệ hơn nhiều, nên đây không phải ưu điểm.
4. **W2V-Large (tiếng Anh) fine-tune tốt hơn MMS-300M (đa ngôn ngữ) ở mọi tập ngoài miền**, kể cả RADAR tiếng Việt (−2,04 điểm). Cùng chiều với zero-shot GĐ3 (W2V-Large 17,80 so với MMS-300M 20,08). Ngôn ngữ pre-training "có tiếng Việt" không cho lợi thế.
5. **Mức quên phụ thuộc backbone:** W2V-Large không quên trên ItW (2,00 → 1,40%), RADAR EN giữ nguyên; MMS-300M quên rõ (2,60 → 7,60%). Nhưng trên RADAR VI cả hai đều kém bản dùng thẳng của chính mình (B2 +3,23, R +2,98): fine-tune trên SEA-Spoof làm kém RADAR VI ở cả hai backbone.

### 3.3. Lần 3 (`orpheus-finetune-b` v1, 14–15/11, 3,89 giờ; `results/runs/orpheus-finetune-b-v1/`)

COMPLETE, 0 cảnh báo nạp trọng số. B1: GPU chỉ dùng 75,8% (CPU 70% trên 4 nhân), tức với mô hình 95M đọc dữ liệu và RawBoost bắt đầu là nút thắt.

**EER (%), thấp hơn là tốt hơn:**

| Cấu hình | test_seen | test_f5 | test_cvreal | **RADAR VI** | **ItW** | RADAR EN |
|---|---|---|---|---|---|---|
| B2 seed 1 (W2V-Large, 317M) | 0,04 | 0,00 | 2,59 | 21,03 | 1,40 | 23,40 |
| **B1** seed 1 (W2V-Small, 95M) | 0,00 | 0,48 | 8,96 | **34,67** | 7,10 | 28,20 |
| S0 W2V-Small | 0,93 | 2,11 | 6,90 | 27,21 | 4,00 | 27,20 |
| R seed 1 (MMS-300M Default) | 0,04 | 0,04 | 2,70 | 23,07 | 7,60 | 26,70 |
| **N0** seed 1 (MMS-300M NDA) | 0,00 | 0,00 | 3,60 | **20,50** | 5,40 | 25,30 |
| S0 MMS-300M NDA | 0,89 | 0,76 | 2,10 | 18,07 | | |
| A0 seed 1 (Default, không RawBoost khi fine-tune) | 0,04 | 0,00 | 2,79 | 18,37 | 6,10 | 25,00 |

**Lỗi ở ngưỡng 0 (%):** B1 coi **62,1%** tiếng thật RADAR VI và **78,2%** tiếng thật Common Voice là giả (actDCF − minDCF 0,62); N0 31,3% và 13,4% (0,21).

**Chênh lệch EER, KTC 95% bootstrap ghép cặp:**

| So sánh | RADAR VI | ItW | RADAR EN | test_cvreal |
|---|---|---|---|---|
| B1 − B2 (95M thay 317M, cùng tiếng Anh) | **+13,64** [12,83; 14,53] | +5,70 [4,50; 6,80] | +4,80 [2,80; 7,20] | +6,37 [5,45; 7,40] |
| B1 − S0 W2V-Small | +7,46 [6,62; 8,34] | +3,10 [2,10; 4,20] | +1,00 [−1,10; 3,40] | +2,06 [1,19; 3,24] |
| N0 − R (post-train không RawBoost) | **−2,57** [−3,13; −2,17] | −2,20 [−3,30; −0,90] | −1,40 [−2,80; 0,20] | +0,89 [0,40; 1,45] |
| N0 − S0 NDA | +2,43 [1,72; 3,07] | | | +1,50 [1,00; 2,12] |
| N0 − A0 | +2,13 [1,56; 2,63] | −0,70 [−1,90; 0,80] | +0,30 [−1,00; 2,10] | +0,80 [0,40; 1,39] |

Mỗi kết quả, một câu giải thích (1 seed):
1. **Kích thước backbone quyết định mạnh:** W2V-Small kém W2V-Large 13,6 điểm trên RADAR VI sau fine-tune (dùng thẳng chỉ kém 9,4 điểm, GĐ3), và coi phần lớn tiếng thật nguồn khác là giả. Mô hình nhỏ bám vào đặc điểm của nguồn tiếng thật duy nhất (VIVOS) mạnh hơn.
2. **Lợi thế của bản NDA còn nguyên sau fine-tune:** N0 tốt hơn R 2,57 điểm trên RADAR VI (dùng thẳng: 2,02) và 2,20 điểm trên ItW. Cùng chiều với GĐ3: với bản 300M, RawBoost khi post-train làm kém ngoài miền.
3. **RawBoost lúc nào cũng làm kém RADAR VI ở MMS-300M:** khi post-train (N0 tốt hơn R) và khi fine-tune (A0 tốt hơn R, A0 tốt hơn N0 2,13 điểm). Cấu hình tốt nhất trên RADAR VI đến giờ là A0 (18,37%), tức không dùng RawBoost khi fine-tune.

### 3.4. Lần 4 (`orpheus-finetune` v4, 15/11, 4,94 giờ; `results/runs/orpheus-finetune-v4/`)

COMPLETE, 0 cảnh báo. R seed 2, 3 (cùng cấu hình, lịch E1'); chấm thêm MLAAD; chấm S0 trên MLAAD.

**R qua 3 seed, EER (%)** (thấp hơn là tốt hơn; trung bình ± độ lệch chuẩn, ddof 1; `table_seeds.csv`):

| Tập | Seed 1 | Seed 2 | Seed 3 | **Trung bình ± độ lệch chuẩn** | S0 MMS-300M |
|---|---|---|---|---|---|
| test_seen | 0,04 | 0,04 | 0,04 | 0,04 ± 0,00 | 0,81 |
| test_cvreal | 2,70 | 4,56 | 6,15 | 4,47 ± 1,72 | 7,44 |
| **RADAR VI** | 23,07 | 23,23 | 25,77 | **24,02 ± 1,52** | 20,08 |
| **ItW** | 7,60 | 4,50 | 4,80 | **5,63 ± 1,71** | 2,60 |
| RADAR EN | 26,70 | 28,10 | 28,10 | 27,63 ± 0,81 | 26,20 |
| RADAR VI FRR ở ngưỡng 0 (%) | 29,21 | 33,83 | 35,06 | 32,70 ± 3,08 | 4,81 |
| MLAAD | | 0,09 | 0,16 | | 5,18 |

**MLAAD `fake/vi`** (5 hệ thống TTS tiếng Việt chưa gặp, tiếng thật VIVOS của test; EER %): S0 MMS-300M 5,18; S0 MMS-300M NDA 4,65; S0 W2V-Large 2,22; S0 W2V-Small 2,23; R seed 2 **0,09**; R seed 3 **0,16** (R − S0: −5,09 [−5,58; −4,53]).

Mỗi kết quả, một câu giải thích:
1. **Độ lệch do seed huấn luyện lớn:** cùng cấu hình, RADAR VI chênh tới 2,70 điểm giữa seed 1 và 3 (KTC theo câu [2,21; 3,10], không chứa 0), ItW chênh tới 3,1 điểm. KTC bootstrap theo câu (rộng ~±0,5 điểm) **đánh giá thấp** độ bất định thật; chênh lệch 1 seed dưới ~3 điểm (≈ 2 độ lệch chuẩn) không đủ để kết luận (phát hiện 8).
2. **Fine-tune làm kém RADAR VI ở cả 3 seed** (+2,98, +3,14, +5,69 so với S0) và làm quên trên ItW ở cả 3 seed (+1,9 đến +5,0). Kết luận của câu hỏi 2 (SFT làm kém ngoài miền với MMS-300M) đứng vững qua seed.
3. **Fine-tune bắt gần hết TTS tiếng Việt chưa gặp khi tiếng thật cùng nguồn với train:** MLAAD từ 5,18% xuống 0,09–0,16%. Trái với RADAR VI (kém đi). Khác biệt chính giữa hai tập là nguồn tiếng thật: MLAAD dùng VIVOS (cùng nguồn train), RADAR dùng Common Voice và FOSD. Trên RADAR, fine-tune giảm FAR (37 → 17–21%) nhưng tăng FRR (4,8 → 29–35%). Vậy phần kém đi trên RADAR VI chủ yếu do **tiếng thật nguồn khác**, không do hệ thống sinh lạ (phát hiện 9).

**Đọc lại các so sánh 1 seed với độ lệch giữa seed của R** (RADAR VI, R = 24,02 ± 1,52):

| Cấu hình | RADAR VI | So với trung bình R | Cách 2 độ lệch chuẩn (~3 điểm)? | Kết luận |
|---|---|---|---|---|
| A0 (không RawBoost khi fine-tune) | 18,37 | −5,65 | Có | RawBoost khi fine-tune làm kém: vững |
| N0 (bản NDA) | 20,50 | −3,52 | Vừa qua | NDA tốt hơn: khả dĩ, cần thêm seed |
| B2 (W2V-Large) | 21,03 | −2,99 | Sát ngưỡng | W2V-Large tốt hơn: chưa chắc |
| P0 (chỉ pre-train) | 31,46 | +7,44 | Có | Post-training quan trọng: vững |
| B1 (W2V-Small) | 34,67 | +10,65 | Có | Backbone nhỏ kém hẳn: vững |

Trên ItW (R = 5,63 ± 1,71): A0 6,10 và N0 5,40 nằm trong khoảng của R (các chênh lệch ItW của A0, N0 ở mục 3.1, 3.3 không còn đáng tin); B2 1,40 và P0 14,90 vẫn cách xa.

**Sửa 15/11 (GĐ6 mục 7.1):** kiểm định t chính thức (khoảng dự đoán từ 3 seed của R, bậc tự do 2) chặt hơn quy tắc "2 độ lệch chuẩn" ở bảng trên: A0 − R trên RADAR VI −5,65 điểm, p = 0,08 (chưa có ý nghĩa ở mức 5%); chỉ P0 − R có ý nghĩa (ItW, RADAR EN, FRR). Cột "Kết luận" ở bảng trên là đọc sơ bộ; R seed 4, 5 (GĐ6 lần 2) sẽ cho bậc tự do 4. **Cập nhật sau GĐ6 lần 2 (R 5 seed, GĐ6 mục 10.1):** R = 24,20 ± 1,88 (RADAR VI); A0 − R −5,83, p = 0,047; P0 − R +7,25, p = 0,024; N0, B2 chưa có ý nghĩa; R − S0 +4,12, p = 0,008 (SFT làm kém RADAR VI có ý nghĩa thống kê).

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

### 4.2. Tập kiểm tra MLAAD `fake/vi` (dựng 14/11)

`src/prep_mlaad.py build` / `check`; danh sách ID `protocols/ids/mlaad_vi_test.tsv`; dataset Kaggle riêng tư `orpheus-mlaad-vi` (`push.sh data-mlaad`). Kết quả kiểm: `results/mlaad/`.

| Thứ | Giá trị |
|---|---|
| Tiếng giả | 5 hệ thống × 1.000 câu: LEMAS-TTS (kiến trúc F5-TTS), OmniVoice, VieNeu-TTS, VoxCPM2, vixTTS; 10,06 giờ; WAV 22,05 kHz, giữ nguyên (code gốc tự resample về 16 kHz khi chấm) |
| Tiếng thật | `real_vivos` của test SEA-Spoof (2.475 câu, 14 người nói, 4,03 giờ), đúng phần thật của `test_f5`, `test_minimax`, nên so được trực tiếp |
| Tổng | 7.475 câu, 14,09 giờ |
| Đặc điểm | LEMAS-TTS dùng giọng tham chiếu tiếng Anh (LibriVox) đọc văn bản tiếng Việt (cột `reference_speaker`, `original_file` của `meta.csv`) |
| Lối tắt (EER khi chỉ dùng một đặc trưng thô; 50% = không tách được) | Thấp nhất 26,5% (âm lượng, vixTTS), 28,1% (âm lượng, VoxCPM2), 28,2% (năng lượng dải 4–8 kHz, VoxCPM2). Không có lối tắt mạnh; cùng cỡ `test_f5` (âm lượng 28,7%, GĐ4 mục 3.1) |
| Lỗi đã sửa khi dựng | `meta.csv` có dấu ngoặc kép trong transcript; đọc mặc định gộp dòng (VieNeu-TTS ra 841 / 1.000). Đọc với `QUOTE_NONE` |

Kế hoạch dùng: chấm S0 (4 mô hình) ở lần 4. Nếu S0 cũng gần 0% thì MLAAD không phân biệt được cấu hình và dừng ở đó; nếu đủ khó thì chấm lại checkpoint các lần 1–3 (đang ở output Kaggle).

| 5 | **Fine-tune trên SEA-Spoof làm kém RADAR tiếng Việt ở mọi cấu hình có RawBoost đã chạy** (R +2,98, B2 +3,23 so với bản dùng thẳng của chính backbone; P0 còn tệ hơn); chỉ A0 (không RawBoost) tốt hơn S0 (−1,71) | Câu hỏi 1 phần "dữ liệu": có thêm dữ liệu tiếng Việt cùng hệ thống sinh (ElevenLabs, ChatGPT) vẫn không giúp RADAR VI; tiếng thật một nguồn (phát hiện 3) và RawBoost (phát hiện 4) lấn át lợi ích | Chờ N0, B1, R seed 2–3 để xem có nhất quán; cân nhắc ở GĐ6: so GRPO với cả R và A0 |
| 8 | **Độ lệch do seed huấn luyện cỡ 1,5 điểm (RADAR VI) và 1,7 điểm (ItW)**, lớn hơn KTC bootstrap theo câu (~±0,5); hai seed của cùng cấu hình có thể khác nhau "có ý nghĩa" theo KTC theo câu (R seed 3 − seed 1 = +2,70 [2,21; 3,10]) | KTC theo câu không dùng để kết luận chênh lệch giữa cấu hình 1 seed. Ngưỡng thực tế: ~2 độ lệch chuẩn của R (~3 điểm RADAR VI) | Báo cáo dùng trung bình ± độ lệch chuẩn của R làm thước. Nếu còn quota: thêm seed cho N0, B2 (gần ngưỡng). GĐ6: G1 cần 3 seed như plan |
| 9 | **Phần kém đi trên RADAR VI do tiếng thật nguồn khác, không do hệ thống sinh lạ:** trên MLAAD (TTS tiếng Việt chưa gặp, tiếng thật VIVOS) fine-tune giảm EER 5,18 → ~0,1%; trên RADAR VI (tiếng thật Common Voice, FOSD) EER tăng, FRR tăng 4,8 → ~33% | Trả lời phần "dữ liệu" của câu hỏi 1: độ phủ hệ thống sinh của dữ liệu fine-tune đủ cho TTS tiếng Việt mới; nút thắt là độ đa dạng tiếng thật. MLAAD sau fine-tune cũng bão hoà, nên không cần chấm lại các checkpoint khác trên MLAAD | Ghi vào báo cáo. Hướng tiếp: thêm tiếng thật đa nguồn khi fine-tune (lưu ý không dùng Common Voice vì trùng nguồn RADAR), hoặc hiệu chuẩn theo nguồn |
| 7 | **Mọi cấu hình fine-tune có RawBoost đều kém bản dùng thẳng của chính backbone trên RADAR VI**: R +2,98, B2 +3,23, B1 +7,46, N0 +2,43 điểm; chỉ A0 (không RawBoost) tốt hơn (−1,71). B1 lệch hiệu chuẩn nặng nhất (FRR ở ngưỡng 0: 62% RADAR VI, 78% Common Voice) | Kết luận chính cho câu hỏi 2 (SFT làm hại ngoài miền tiếng Việt) và cho câu hỏi 1 phần "cách train" (RawBoost). Tất cả 1 seed; R seed 2–3 (lần 4) cho biết độ lệch giữa seed | Báo cáo cả EER lẫn FRR. GĐ6: cân nhắc dùng A0 làm thêm một mốc so với GRPO (cần người dùng chốt) |
| 6 | **Post-training gần như không giúp trong miền nhưng quyết định ngoài miền** (P0 − R: test_seen 0,00; RADAR VI +8,39; ItW +7,30; F5 +5,90) | Trả lời trực tiếp "post-training đóng góp bao nhiêu" của câu hỏi 1. Khớp các bài gốc (mục 6.4) | Ghi vào báo cáo kèm cảnh báo 1 seed |

## 5. Ngân sách quota

| Lần | Quota dùng | Còn lại | Ghi chú |
|---|---|---|---|
| Trước GĐ5 | 0,74 giờ (pilot GĐ4) | 29,26 giờ đến 21/11 | |
| 1 (`orpheus-finetune` v2) | 4,68 giờ | 24,58 giờ | Ước 4,8 giờ; khớp |
| 2 (`orpheus-finetune` v3) | 4,14 giờ | | Xếp hàng ~3,5 giờ không tốn quota |
| 3 (`orpheus-finetune-b` v1) | 3,89 giờ | 16,54 giờ (sau lần 2 và 3) | Chạy song song với lần 2 |
| 4 (`orpheus-finetune` v4) | 4,94 giờ | **11,60 giờ** đến 21/11 | Thêm chấm MLAAD (~0,7 giờ) |
| **Tổng GĐ5** | **17,65 giờ** (4 lần, 10 lần huấn luyện) | | Ước ban đầu ~15 giờ; thêm MLAAD, S0 trên SEA-Spoof |

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

- [x] Seed 1 của 5 cấu hình cần huấn luyện: R, A0, P0, B2, B1 (thêm N0)
- [x] R đủ 3 seed
- [x] Bảng EER, minDCF (trung bình, độ lệch chuẩn cho R), một câu giải thích cho từng kết quả (mục 3, `results/finetune/analysis/`)
- [x] Phân tích riêng theo từng hệ thống sinh (`table_by_system.csv`; MLAAD theo hệ thống ở `results/zeroshot/*/mlaad_vi_test/metrics.csv`)
- [x] Bằng chứng mọi lần chạy trong `results/runs/` (`orpheus-finetune-v2`, `-v3`, `-v4`, `orpheus-finetune-b-v1`)

## 8. Tổng kết GĐ5 và mang sang GĐ6 (15/11)

**Trạng thái:** GĐ5 **qua**, đủ 5 điều kiện ở mục 7. 10 lần huấn luyện (R × 3, A0, P0, B2, B1, N0) + S0 trên SEA-Spoof và MLAAD, 4 lần chạy Kaggle, 17,65 giờ quota. Còn 11,60 giờ đến 21/11.

**Năm kết quả chính** (EER RADAR VI, thấp hơn là tốt hơn; R = 24,02 ± 1,52 qua 3 seed)
1. **SFT trên tiếng Việt làm kém ngoài miền với MMS-300M** (câu hỏi 2): RADAR VI kém bản dùng thẳng ở cả 3 seed (+3,0 đến +5,7 điểm), ItW 2,6 → 5,6 ± 1,7%. Nhưng W2V-Large không quên trên ItW (1,4%): mức quên phụ thuộc backbone.
2. **Post-training là yếu tố lớn nhất ngoài miền, không đóng góp trong miền:** P0 bằng R trong miền, kém 7–8 điểm ngoài miền.
3. **Kích thước quan trọng:** W2V-Small kém W2V-Large 13,6 điểm RADAR VI sau fine-tune.
4. **RawBoost làm kém ngoài miền tiếng Việt với bản 300M**, cả khi fine-tune (A0 18,37%, tốt nhất GĐ5, cách R ~3,7 độ lệch chuẩn) lẫn khi post-train (N0 20,50%, khả dĩ). Ngôn ngữ pre-training "có tiếng Việt" không giúp (B2 ≤ R).
5. **Nút thắt là tiếng thật nguồn khác, không phải hệ thống sinh lạ:** fine-tune bắt gần hết TTS tiếng Việt chưa gặp (MLAAD 5,18 → ~0,1%), nhưng coi ~33% tiếng thật RADAR VI là giả ở ngưỡng 0 (S0: 4,8%).

**Giới hạn cần ghi trong báo cáo:** 1 seed cho A0, P0, B2, B1, N0; độ lệch giữa seed ~1,5 điểm nên chênh lệch < ~3 điểm chưa kết luận được (N0, B2 sát ngưỡng). Lịch learning rate E1' (lệch E1, mục 4.1). R seed 1 và các lần khác cùng chạy trên T4 (không đổi GPU). Hệ thống để riêng của SEA-Spoof và MLAAD đều bão hoà, nên "tổng quát hoá sang hệ thống sinh mới" chỉ đo được ở mức S0 và qua RADAR.

**Câu hỏi chờ người dùng chốt trước GĐ6**
| # | Câu hỏi | Đề xuất | Lý do |
|---|---|---|---|
| F1 | Dùng 11,60 giờ quota còn lại đến 21/11 thế nào | Dành cho GĐ6 (G1 seed 1 + đo GRPO): GRPO chưa đo bộ nhớ, tốc độ; đo sớm để biết ngân sách tuần sau | Thêm seed cho N0, B2 (~8 giờ) làm chắc phát hiện 4 nhưng chiếm gần hết quota tuần này |
| F2 | Mốc so sánh cho GRPO ở GĐ6 | So G1 với **cả R và A0** (cùng có RawBoost thì so R; GRPO không RawBoost thì so A0) | A0 là SFT tốt nhất; nếu GRPO chỉ hơn R mà không hơn A0, thì "tắt RawBoost" là cách rẻ hơn |
| F3 | Thước đo chính của GĐ6 | RADAR VI (EER + FRR ở ngưỡng 0), ItW, RADAR EN; bỏ hệ thống để riêng SEA-Spoof và MLAAD khỏi thước đo chính (bão hoà) | Phát hiện 2, 9 |

**Đã chốt (15/11, người dùng):** F2: so G1 với cả R và A0. F3: thước đo chính RADAR VI, ItW, RADAR EN (EER kèm FRR ở ngưỡng 0); SEA-Spoof và MLAAD vẫn chấm và báo nhưng không dùng để kết luận. F1: chưa chạy GĐ6; người dùng yêu cầu đánh giá lại plan và kết quả trước.

**Mang sang GĐ6:** cấu hình SFT và lịch E1' (mục 4.1); bảng R 3 seed làm thước độ lệch; `src/analyze_ablation.py` (thêm cặp G1:R, G1:A0); phân tích GRPO ở mục 6 (giả thuyết GRPO bỏ qua câu sai chắc chắn: kiểm bằng FRR trên RADAR VI và theo hệ thống). Kernel `orpheus-finetune` và `orpheus-finetune-b` chạy song song được; phiên OAuth Kaggle CLI chỉ sống ~3 giờ (nên dùng API token).

## 9. Đánh giá plan sau GĐ5 và lưu ý cho GĐ6 (15/11)

**Tiến độ:** GĐ5 xong 15/11 (plan dành 4 ngày). Còn 15 ngày đến hạn 30/11; GĐ6–8 và dự phòng cần ~8 ngày.

| Câu hỏi nghiên cứu | Trạng thái sau GĐ5 |
|---|---|
| 1. Yếu tố quyết định | Trả lời tốt: post-training > kích thước > RawBoost > độ đa dạng tiếng thật; ngôn ngữ pre-training không giúp |
| 2. SFT có làm quên | Có với MMS-300M (vững qua 3 seed), không với W2V-Large |
| 3. GRPO | Chưa làm (GĐ6) |

**Điểm yếu cần ghi trong báo cáo:** (1) kết quả RADAR VI trộn "quên" với "lệch nguồn tiếng thật" (tiếng thật train chỉ VIVOS); MLAAD tách được một phần (phát hiện 9). (2) 5 cấu hình 1 seed; N0, B2 sát ngưỡng độ lệch seed. (3) Thước đo trong miền bão hoà (SEA-Spoof, MLAAD).

**Lưu ý cho GĐ6**
| # | Lưu ý | Việc cần làm |
|---|---|---|
| L1 | **Bộ nhớ GRPO chưa đo.** SFT đã 14,6 / 15 GB trên T4 ở batch 50 s; GRPO giữ thêm `ref_detector` và `old_detector` (~2,5 GB). Giảm batch sẽ đổi số bước và lịch learning rate, không so được với R, A0 | Chạy thử ngắn trước (đo bộ nhớ, giây mỗi bước). Phương án giữ số bước: batch 25 + tích luỹ gradient 2 (`grad_accumulation_factor` của SpeechBrain, **cần kiểm** có nhận qua dòng lệnh không và có làm lệch lịch learning rate không). Hoặc chạy trên server GPU 24 GB (L5) |
| L2 | **Mốc so sánh (F2):** G1 so với cả R (3 seed, có RawBoost) và A0 (không RawBoost) | `src/analyze_ablation.py --pairs G1_s1:R_s1 G1_s1:A0_s1 ...`; dùng R = 24,02 ± 1,52 (RADAR VI) làm thước độ lệch seed |
| L3 | **Thước đo chính (F3):** RADAR VI, ItW, RADAR EN, kèm FRR ở ngưỡng 0; SEA-Spoof và MLAAD chỉ báo | |
| L4 | **Cùng bộ ghi đè SFT** (lịch E1', mục 4.1) cho mọi lần GRPO: `max_batch_length=50 number_of_epochs=3 optimizer_step_limit=4600 warmup=460 lr=1e-6 valid_step=1000000000 num_workers=2 seed=<n>`, file `hparams/ft_grpo/mms_300m_pre-post-grpo_v1.yaml` (G1), `_v2` (G2); G3 = G1 với `beta=0` | `src/finetune.py --method grpo_v1` đã có sẵn tham số `--method` |
| L5 | **Server thuê là phương án được** (người dùng hỏi 15/11): GPU 24 GB giải L1, không xếp hàng, không giới hạn quota. Nếu dùng: chạy lại R và A0 trên server để so cùng phần cứng; xoá SEA-Spoof khi trả máy. Cấu hình vast.ai: mục 2 (ghi chú xếp hàng) | Người dùng quyết; hiện plan là Kaggle |
| L6 | **Không chạy tiếp giữa chừng được:** checkpoint chỉ lưu sau epoch cuối (E3), không lưu optimizer, bộ đếm epoch, lịch learning rate. Chỉ đẩy một lần chạy khi quota còn lại > thời gian ước tính + dự phòng; phần thiếu để tuần sau, chạy tiếp từ lần kế tiếp với tham số đã lưu | Kaggle làm gì khi hết quota giữa phiên: chưa kiểm |
| L7 | **Ngày 2 GĐ6 (ít mẫu N = 50/200/1000):** trong miền chắc bão hoà | Đo ngoài miền, hoặc cắt để dành quota cho G1 đủ 3 seed (G1 3 seed không được cắt theo plan) |
| L8 | **Ngày 3 GĐ6 (hiệu chuẩn) có ý nghĩa:** sau fine-tune actDCF − minDCF ≈ 0,2 (GĐ3: ≤ 0,018). Hạn chế: dev chỉ có tiếng thật VIVOS | Báo actDCF trước và sau trên RADAR VI; ghi rõ hạn chế |
| L9 | **Giả thuyết cơ chế GRPO** (mục 6.3): GRPO trong code bỏ qua câu mô hình sai chắc chắn. Lỗi chính sau SFT là tiếng thật nguồn khác bị coi là giả | Kiểm: GRPO có giảm FRR trên RADAR VI và `test_cvreal` so với R không |
| L10 | **Quota:** 11,60 giờ còn lại tuần này mất sau 7 giờ sáng 21/11; tuần sau 30 giờ. GĐ6 (G1 × 3, G2, G3; GRPO ước chậm ~1,7 lần SFT) ~25–30 giờ | Người dùng chưa chốt dùng 11,60 giờ (F1) |
| L11 | **Kaggle CLI:** phiên OAuth chỉ sống ~3 giờ sau `kaggle auth login --force`; nên dùng API token `~/.kaggle/access_token`. Hai kernel `orpheus-finetune`, `orpheus-finetune-b` chạy song song được; xếp hàng có thể vài giờ (không tốn quota) | `chore/Quy-Trinh-Chay-Kaggle.md` |
| L12 | **Checkpoint các lần GĐ5** chỉ ở output Kaggle: `orpheus-finetune` v2 (R_s1, A0_s1), v3 (P0_s1, B2_s1), v4 (R_s2, R_s3); `orpheus-finetune-b` v1 (B1_s1, N0_s1). Điểm từng câu đã ở `results/finetune/` | Hiệu chuẩn (ngày 3) chỉ cần điểm, không cần GPU |

