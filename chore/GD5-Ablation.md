# GĐ5. Ablation

Câu hỏi của giai đoạn: yếu tố nào quyết định kết quả trên tiếng Việt khi fine-tune (post-training, RawBoost, ngôn ngữ và kích thước backbone), và fine-tune có giám sát có làm mô hình quên kiến thức cũ không (câu hỏi 1, 2 của plan). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ5. Đầu vào từ GĐ4: `chore/GD4-Du-Lieu-Va-Pipeline.md` mục 4.2 (cấu hình đã chốt) và mục 7. File này bắt đầu ngày 14/11.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Chuẩn bị lần chạy 1: tập con SEA-Spoof tính ngay trên Kaggle, khoá file khi hai tiến trình chấm cùng ghi, bước chấm S0 | `src/score_zeroshot.py`, `kaggle/finetune/run.py`, `kaggle/archive_run.py`, `kaggle/push.sh` | Xong (14/11), đã thử trên máy với điểm của pilot |
| 1 | **Lần chạy 1: R seed 1, A0 seed 1**, chấm S0 trên SEA-Spoof | `orpheus-finetune` v2, `results/runs/orpheus-finetune-v2/` | Đang chạy |
| 1 | Báo kết quả lần 1 cho người dùng, chờ duyệt trước khi chạy tiếp | Mục 3 | |
| 1–2 | Lần 2–3: P0, B2, B1 (seed 1); N0 nếu còn quota | | Chưa |
| 3–4 | Lần 4: R seed 2, 3 | | Chưa |
| Trong lúc chờ GPU | Đọc sâu bài GRPO và `algo/rl.py` (chuẩn bị GĐ6) | | Chưa |

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
| 2 | | P0 seed 1 | B2 seed 1 | | ~3,7 giờ |
| 3 | | B1 seed 1 | N0 seed 1 | | ~3,5 giờ |
| 4 | | R seed 2 | R seed 3 | | ~3,7 giờ |

Mỗi lần huấn luyện chấm sau fine-tune trên `seaspoof_vi_dev`, `seaspoof_vi_test` (tập con tính ngay trên Kaggle, `metrics_subsets.csv`), `radar_vi`, `ood_forget`. Chấm luôn ở batch 1 trên dữ liệu sạch, không dùng EER dev lúc huấn luyện (GĐ4 phát hiện 10, 13).

**Lần 1, ước tính thời gian** (số đo của pilot): dựng môi trường ~5 phút; huấn luyện 3 × 1.521 bước × 1,93 s ≈ 147 phút; chấm sau fine-tune ~65–80 phút (hai GPU cùng chấm, 4 nhân CPU); chấm S0 (8 job, ~63 giờ âm thanh) ~55–60 phút trên hai GPU. Tổng ~4,8 giờ, còn ~24,5 giờ quota.

Chấm S0 thêm MMS-300M NDA (ngoài 3 backbone GĐ4 đã định): ~15 phút quota, cho mốc của N0 và cho so sánh Default/NDA trên hệ thống để riêng của SEA-Spoof mà không cần fine-tune.

## 3. Kết quả

(Điền sau lần 1.)

## 4. Phát hiện ảnh hưởng thí nghiệm

| # | Phát hiện | Ảnh hưởng | Việc cần làm |
|---|---|---|---|

## 5. Ngân sách quota

| Lần | Quota dùng | Còn lại | Ghi chú |
|---|---|---|---|
| Trước GĐ5 | 0,74 giờ (pilot GĐ4) | 29,26 giờ đến 21/11 | |

## 6. Kiểm tra điều kiện qua GĐ5

- [ ] Seed 1 của 5 cấu hình cần huấn luyện: R, A0, P0, B2, B1
- [ ] R đủ 3 seed
- [ ] Bảng EER, minDCF (trung bình, độ lệch chuẩn cho R), một câu giải thích cho từng kết quả
- [ ] Phân tích riêng theo từng hệ thống sinh
- [ ] Bằng chứng mọi lần chạy trong `results/runs/`
