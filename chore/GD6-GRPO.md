# GĐ6. GRPO và cải tiến

Câu hỏi của giai đoạn: fine-tune bằng GRPO trên tiếng Việt có giữ khả năng tổng quát hoá tốt hơn SFT không, và thành phần nào của GRPO tạo ra khác biệt (câu hỏi 3 của plan). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ6 kèm "Lưu ý từ GĐ5". Đầu vào từ GĐ5: `chore/GD5-Ablation.md` mục 8 (tổng kết), mục 9 (lưu ý L1–L12), mục 6 (đối chiếu bài GRPO và code), mục 4.1 (lịch learning rate E1'). File này bắt đầu ngày 15/11.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Chuẩn bị lần thử đo GRPO: đọc code GRPO (YAML v1/v2, `main.py`, `algo/rl.py`), kiểm ghi đè lồng `rl_config`, tích luỹ gradient, mô phỏng số batch | Mục 2; `src/mem_probe.py`; `src/finetune.py` (`--probe-mem`, `--oom-retry`, `--train-longest`, `--discard-ckpt`, `--env`); `kaggle/finetune/run.py` | Xong (15/11), đã thử trên máy |
| 1 | **Lần thử đo bộ nhớ và tốc độ GRPO trên T4** (L1) | `orpheus-finetune` v5, mục 3 | **Xong** (15/11, 0,47 giờ): G1 batch 50 vừa với `expandable_segments`, 3,07 s/bước |
| 1 | Báo kết quả lần thử; người dùng chốt: Kaggle hay thuê server, dùng quota còn lại tuần này thế nào | Mục 3, 5 | **Đã chốt** (15/11, mục 5): Kaggle; batch 50 + `expandable_segments`; P1 làm, P2 không, P3 không cắt |
| 1 | Lần 1: G1 s1, G2 s1; lần 2 (song song): R s4, R s5 | `orpheus-finetune` v6, `orpheus-finetune-b` v2; mục 6 | Đang đẩy (15/11) |
| 2–3 | G1 s2–s5, G3 (từ 21/11) | Mục 6 | Chưa |
| 2 | Ít mẫu (N = 50, 200, 1000), hoặc cắt (L7) | | Chưa |
| 3 | Hiệu chuẩn điểm, phân tích theo hệ thống sinh | | Chưa |

**Lịch:** hôm nay 15/11, hạn nộp 30/11. Quota Kaggle còn 11,60 giờ đến 7 giờ sáng 21/11 (kiểm 15/11 trước lần thử), tuần sau 30 giờ.

## 0. Đầu vào từ GĐ5 (đã chốt)

| Thứ | Giá trị | Nguồn |
|---|---|---|
| Mốc so sánh (F2) | G1 so với **cả R và A0** | GĐ5 mục 8 |
| Thước đo chính (F3) | RADAR VI, ItW, RADAR EN (EER) kèm FRR ở ngưỡng log-odds 0; SEA-Spoof và MLAAD vẫn chấm và báo, không dùng để kết luận (bão hoà) | GĐ5 mục 8, phát hiện 2, 9 |
| Cấu hình huấn luyện | Bộ ghi đè SFT giữ nguyên (lịch E1'): `max_batch_length=50 number_of_epochs=3 optimizer_step_limit=4600 warmup=460 lr=1e-6 valid_step=1000000000 num_workers=2 seed=<n>` | GĐ5 mục 4.1, L4 |
| Thước độ lệch seed | R qua 3 seed: RADAR VI 24,02 ± 1,52; ItW 5,63 ± 1,71; RADAR EN 27,63 ± 0,81; FRR RADAR VI ở ngưỡng 0 32,70 ± 3,08 (%). Chênh lệch < ~3 điểm RADAR VI chưa kết luận được với 1 seed | GĐ5 mục 3.4, phát hiện 8 |
| A0 seed 1 (không RawBoost) | RADAR VI 18,37; ItW 6,10; RADAR EN 25,00; FRR 27,09 | GĐ5 mục 3.1 |
| S0 MMS-300M (dùng thẳng) | RADAR VI 20,08; ItW 2,60; RADAR EN 26,20; FRR 4,81 | GĐ3, GĐ5 mục 3.1 |
| Giả thuyết cơ chế | GRPO trong code chuẩn hoá phần thưởng theo batch: kỳ vọng gradient theo log-odds nhãn đúng là p(1 − p)/σ_batch (SFT: 1 − p), tức gần như bỏ qua câu mô hình đang sai chắc chắn | GĐ5 mục 6.3, L9 |

EER, minDCF, actDCF, FAR, FRR: thấp hơn là tốt hơn.

## 1. Các cấu hình

Mọi cấu hình fine-tune MMS-300M post-trained (bản Default) trên `seaspoof_vi_train`, cùng bộ ghi đè ở mục 0, chỉ đổi file YAML hoặc khoá trong `rl_config`.

| Mã | File cấu hình (`hparams/ft_grpo/`) | Khác SFT ở | Ghi đè thêm | So với | Câu hỏi |
|---|---|---|---|---|---|
| G1 | `mms_300m_pre-post-grpo_v1.yaml` (`algo: grpo3`) | GRPO đủ: G = 64 nhãn lấy mẫu mỗi câu, clip ε = 0,2, mô hình cũ cập nhật mỗi 1.000 batch, KL β = 0,04 | | R (3 seed), A0 | GRPO có giữ tổng quát hoá tốt hơn SFT? |
| G2 | `mms_300m_pre-post-grpo_v2.yaml` (`algo: grpo2`) | Bỏ clip và mô hình cũ | | G1 | Có cần clip và mô hình cũ? |
| G3 | như G1 | β = 0 | `rl_config={beta: 0.0}` | G1 | Phạt lệch khỏi mô hình tham chiếu có vai trò gì? |

Theo plan, G1 chạy 3 seed (không được cắt); G2, G3 một seed; G3 là mục cắt thứ 2.

## 2. Kiểm code trước lần thử (15/11, AntiDeepfake commit `86098091`, SpeechBrain 1.0.2, HyperPyYAML 1.2.3)

| Nội dung | Kết quả kiểm | Hệ quả |
|---|---|---|
| YAML GRPO so với SFT | `grpo_v1` chỉ thêm `rl_config` (`algo: grpo3`, `sample_num: 64`, `std_floor: 1e-5`, `beta: 0.04`, `inner_step_size: 1000`, `epsilon: 0.2`) và hai mô hình `ref_detector`, `old_detector`; `grpo_v2` có `algo: grpo2`, bỏ `inner_step_size`, `epsilon`, `old_detector`. Mọi khoá khác (lr, warmup, batch, RawBoost `algo: 5`) giống file SFT | Dùng nguyên bộ ghi đè SFT là đúng "đổi một yếu tố": chỉ đổi hàm loss |
| Hai mô hình phụ (`main.py`, `SSLBrain.__init__`) | `ref_detector` và `old_detector` là hai bản đầy đủ của mô hình, chép trọng số từ `detector` sau khi nạp checkpoint, đặt trên GPU, chế độ eval, chỉ chạy forward trong `torch.no_grad()` | Thêm 2 × 317M tham số fp32 ≈ 2,5 GB cho G1, ≈ 1,3 GB cho G2, chưa tính activation tạm của hai lượt forward |
| fp16 | `main.py` ghi đè `fit_batch` của SpeechBrain và **không bọc `torch.autocast`** (bản gốc của SpeechBrain có, `core.py` dòng 1227); mô hình cũng không tự bật. Vậy dù YAML ghi `precision: fp16`, forward và backward chạy fp32; GradScaler vẫn nhân loss | Áp dụng như nhau cho SFT và GRPO (không ảnh hưởng so sánh). Giải thích vì sao SFT đã sát trần 15 GB ở batch 50 s |
| Cập nhật mô hình cũ | `self.step % inner_step_size == 0`, với `self.step` đếm **batch**; `README_grpo.md` ghi "optimizer steps". Hai cách bằng nhau khi không tích luỹ gradient | Nếu tích luỹ gradient 2 thì đặt `inner_step_size: 2000` để giữ 1.000 bước optimizer |
| Tích luỹ gradient | `--grad_accumulation_factor` là tham số chạy của SpeechBrain (nhận qua dòng lệnh, như `optimizer_step_limit`). `fit_batch` của `main.py` chia loss cho hệ số, chỉ gọi optimizer và lịch learning rate (`on_fit_batch_end`) ở batch thứ `k` × hệ số; `optimizer_step_limit` đếm bước optimizer | Lịch learning rate theo bước optimizer giữ nguyên khi tích luỹ |
| Ghi đè khoá lồng | `--rl_config "{beta: 0.0}"` qua `_convert_to_yaml` của SpeechBrain thành `rl_config: {beta: 0.0}`; HyperPyYAML `recursive_update` **gộp** vào `rl_config` của YAML, giữ các khoá khác. Đã thử trên máy với đúng hai hàm này và file `grpo_v1` | G3 làm được bằng ghi đè, không cần file YAML mới |
| Số batch mỗi epoch | Biên bucket của `DynamicBatchSampler` không phụ thuộc dữ liệu (phân vị lognormal co theo `max_batch_length`, 20 bucket). Mô phỏng trên `seaspoof_vi_train`: batch 50 s cho **1.521** batch (đúng số đo của R), trung bình 11,7 câu, 42,3 s âm thanh; batch 25 s cho **3.072** batch, 5,8 câu, 21,0 s | Batch 25 + tích luỹ 2 cho 1.536 bước mỗi epoch; với `optimizer_step_limit` 4600 dừng ở bước 4.600 (R: 4.563), bỏ 16 batch cuối epoch 3; âm thanh mỗi bước 42,0 s (R 42,3 s). Gần tương đương; khác ở chỗ phần thưởng được chuẩn hoá trên nửa batch |
| Câu dài | Câu train từ 0,55 đến 15,94 s, trung vị 3,38 s, 99% dưới 9,38 s; 2 câu > 13 s (bị cắt còn 10–13 s khi huấn luyện) | Batch toàn câu dài tốn bộ nhớ attention nhất (bậc hai theo số khung) dù tổng thời lượng như nhau; 300 bước ngẫu nhiên chưa chắc gặp. Thêm bài thử áp lực trên 240 câu dài nhất (9,06–15,94 s) |

## 3. Lần thử đo (`orpheus-finetune` v5)

Mục đích (L1): biết GRPO có vừa T4 ở batch 50 s như R không, cần bao nhiêu giây mỗi bước, để chọn cấu hình và tính ngân sách G1–G3 trước khi chạy thật. Một lần GRPO dài hết bộ nhớ giữa chừng là mất cả phiên (không chạy tiếp được, L6), và `finetune.py` mặc định tự giảm một nửa batch khi hết bộ nhớ (đổi số bước, không so được với R). Vì vậy:
- `--oom-retry 0`: hết bộ nhớ thì ghi nhận, không tự giảm batch.
- `--probe-mem`: `main.py` gốc chạy qua `src/mem_probe.py` (`runpy`, không đổi phép tính), ghi `torch.cuda.max_memory_allocated` và `max_memory_reserved`. nvidia-smi chỉ thấy bộ nhớ PyTorch giữ sẵn nên luôn gần đầy (GĐ4: 14,6 GB cả khi overfit 100 câu).
- `--discard-ckpt`: không giữ mô hình, không chấm. Dev là `seaspoof_vi_train_n50` (100 câu) để bước kiểm tra cuối ngắn.

| GPU | Lần đo | Cấu hình | Số bước | Đo gì |
|---|---|---|---|---|
| 0 | `probe_G1_b50` | G1, batch 50 | 150 | G1 có vừa ở batch của R không; giây mỗi bước |
| 0 | `probe_G1_b50_long` | G1, batch 50, 240 câu dài nhất | 1 epoch (~60) | Trường hợp xấu nhất về bộ nhớ |
| 0 | `probe_G1_b50_long_exp` | Như trên, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ~60 | Chỉ chạy nếu một trong hai lần trên hết bộ nhớ: giảm phân mảnh có đủ không |
| 1 | `probe_G1_b25x2` | G1, batch 25, tích luỹ 2, `inner_step_size: 2000` | 150 bước optimizer (300 batch) | Phương án giữ số bước nếu batch 50 không vừa; kiểm luôn ghi đè lồng và tích luỹ gradient trên máy thật |
| 1 | `probe_G1_b25x2_long` | Như trên, 240 câu dài nhất | 1 epoch | Trường hợp xấu nhất của phương án này |
| 1 | `probe_G2_b50` | G2, batch 50 | 100 | Bộ nhớ, tốc độ G2 |
| 1 | `probe_G2_b50_long` | G2, batch 50, 240 câu dài nhất | 1 epoch | Trường hợp xấu nhất của G2 |

Ước tính: dựng môi trường ~5 phút; mỗi lần đo ~1,5–2 phút khởi tạo (ba bản mô hình, nạp checkpoint) cộng thời gian huấn luyện (GRPO ước ~1,7 lần SFT, tức ~3,3 s mỗi bước ở batch 50). Tổng ~45 phút, ~0,75 giờ quota.

### 3.1. Kết quả (`orpheus-finetune` v5, 15/11, 0,47 giờ; `results/runs/orpheus-finetune-v5/`)

COMPLETE, được cấp máy ngay khi đẩy (9:17), chạy 28 phút. Không có lỗi hay cảnh báo deterministic với GRPO (`torch.multinomial` chạy được khi `use_deterministic_algorithms(True)`). Số liệu: `results/runs/orpheus-finetune-v5/train_summary.csv`, log `train_logs/<run>/main.log` (dòng `MEM_PROBE`).

**Bộ nhớ (MiB) và tốc độ.** PyTorch thấy T4 có 14.912 MiB (14,56 GiB); theo thông báo hết bộ nhớ, phần ngoài PyTorch (CUDA context) ~180 MiB, nên PyTorch giữ được tối đa ~14.700 MiB. "Đỉnh cấp phát" = `max_memory_allocated` (tensor thật sự dùng); "đỉnh giữ" = `max_memory_reserved` (gồm phần allocator giữ sẵn, gần với số nvidia-smi thấy). Giây mỗi bước tính theo bước optimizer; SFT (R, GĐ5) 1,93–2,06 s.

| Lần đo | Cấu hình | Kết quả | Bước optimizer | Giây mỗi bước | Đỉnh cấp phát | Đỉnh giữ | GPU dùng (%) |
|---|---|---|---|---|---|---|---|
| `probe_G1_b50` | G1, batch 50 | Qua | 150 | **3,07** | 13.826 | 14.696 | 92 |
| `probe_G1_b50_long` | G1, batch 50, 240 câu dài nhất | **Hết bộ nhớ** sau ~40 s | | | 13.667 (13,35 GiB lúc lỗi) | 14.654 | |
| `probe_G1_b50_long_exp` | Như trên + `expandable_segments:True` | Qua (đủ 63 batch) | 63 | 2,84 | **13.925** | **14.314** | 86 |
| `probe_G1_b25x2` | G1, batch 25 × 2 | Qua | 150 (300 batch) | 3,46 | 11.432 | 14.220 | 92 |
| `probe_G1_b25x2_long` | Như trên, 240 câu dài nhất | Qua | 64 (129 batch) | 3,40 | 11.277 | 14.208 | 88 |
| `probe_G2_b50` | G2, batch 50 | Qua | 100 | **2,95** | 12.580 | 14.696 | 92 |
| `probe_G2_b50_long` | G2, batch 50, 240 câu dài nhất | Qua | 63 | 2,71 | 12.704 | 14.640 | 88 |

Mỗi kết quả, một câu giải thích:
1. **G1 ở batch 50 vừa về dung lượng nhưng hỏng vì phân mảnh:** lúc lỗi, PyTorch mới cấp 13,35 GiB và còn 811 MiB "giữ nhưng chưa dùng" (các khối rời, không ghép được cho yêu cầu 280 MiB). Bật `expandable_segments` (cho allocator nối dài vùng nhớ thay vì cắt khối cố định; không đổi phép tính) thì cùng 240 câu dài nhất chạy qua, đỉnh giữ 14.314 MiB, còn ~400 MiB so với ~14.700 MiB (≈ 1,4 lần yêu cầu lớn nhất gặp được, 280 MiB).
2. **Bộ nhớ gần như không đổi theo batch:** G1 ở batch thường cần 13.826 MiB, batch toàn câu dài 13.925 MiB (+100 MiB). Giới hạn tổng thời lượng 50 s mỗi batch giữ phần lớn activation cố định; câu dài chỉ thêm phần attention. Thử áp lực vì vậy đại diện được cho trường hợp xấu nhất của lần chạy đủ.
3. **Batch 25 × 2 an toàn hơn nhiều** (11,4 GB, dư ~3 GB) nhưng chậm hơn 13% (3,46 so với 3,07 s mỗi bước) vì batch nhỏ dùng GPU kém hiệu quả hơn.
4. **GRPO chậm 1,55 lần SFT (G1) và 1,49 lần (G2)**, ít hơn ước 1,7 của GĐ5 (L10). G2 chỉ bỏ một lượt forward của mô hình cũ nên nhanh hơn G1 có 4%. G2 vừa batch 50 mà không cần `expandable_segments` (12,7 GB ở trường hợp xấu nhất).
5. **Cơ chế đã kiểm trên máy thật:** cấu hình đã giải của `probe_G1_b25x2` có `inner_step_size: 2000`, giữ nguyên các khoá khác; `last_lr` 3,24e-07 sau 150 bước ở cả batch 50 và 25 × 2 (= 1e-6 × 149/460): lịch learning rate theo bước optimizer; 1.521 và 3.072 batch mỗi epoch, khớp mô phỏng ở mục 2.

### 3.2. Ước tính một lần chạy đủ (3 epoch)

Chấm sau fine-tune (5 tập, có MLAAD) lấy từ GĐ5 lần 4: 256 phút mỗi lần trừ 149–160 phút huấn luyện = 96–107 phút. Hai lần huấn luyện chạy song song trên hai GPU của một kernel (như lúc đo).

| Cấu hình | Huấn luyện | Chấm | Một kernel (2 lần song song, cả dựng môi trường) |
|---|---|---|---|
| G1 hoặc G3, batch 50 + `expandable_segments` | 4.563 × 3,07 s ≈ 3,9 giờ (+ ~5 phút khởi tạo, kiểm tra dev cuối) | ~1,7 giờ | **~5,8 giờ** |
| G1, batch 25 × 2 | 4.600 × 3,46 s ≈ 4,4 giờ | ~1,7 giờ | ~6,3 giờ |
| G2, batch 50 | 4.563 × 2,95 s ≈ 3,7 giờ | ~1,7 giờ | ~5,5 giờ (cặp với G1 thì theo G1: ~5,8) |
| SFT (R thêm seed) | ~2,6 giờ | ~1,7 giờ | ~4,4 giờ (GĐ5 lần 4: 4,34 giờ, không tính chấm zero-shot) |

Kiểm tra dev cuối lần chạy (`seaspoof_vi_dev`, batch 10 câu, câu dài nhất 12,12 s, không gradient) giữ nguyên như R: bộ nhớ tĩnh lúc đó ~6,4 GB (ba bản mô hình + trạng thái Adam; gradient đã xoá), cộng activation tạm ước ≤ 3 GB, xa giới hạn.

## 4. Phát hiện ảnh hưởng thí nghiệm

| # | Phát hiện | Ảnh hưởng | Việc cần làm |
|---|---|---|---|
| 1 | **Huấn luyện chạy fp32 dù YAML ghi `precision: fp16`** (`fit_batch` của `main.py` không có `autocast`; đọc code, chưa kiểm khi chạy) | Không đổi so sánh (SFT và GRPO cùng code). Giải thích bộ nhớ sát trần. Ghi vào báo cáo phần cấu hình | Không sửa code gốc |
| 2 | **`inner_step_size` đếm theo batch, không theo bước optimizer** như `README_grpo.md` viết | Chỉ khác khi tích luỹ gradient | Đặt 2000 nếu dùng batch 25 × 2 (đã kiểm, mục 3.1 kết quả 5) |
| 3 | **G1 ở batch 50 hết bộ nhớ trên batch toàn câu dài do phân mảnh, không do thiếu dung lượng**; qua khi bật `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, đỉnh 13,9 / ~14,7 GB | Giữ được đúng batch, số bước, lịch learning rate của R (L1) mà không đổi phép tính. Biên còn mỏng (~400 MiB giữ). `finetune.py` mặc định sẽ tự giảm batch khi hết bộ nhớ: với GRPO phải đặt `--oom-retry 0` | Đề xuất: mọi lần GRPO dùng batch 50 + `expandable_segments` + `--oom-retry 0`; nếu một lần hết bộ nhớ giữa chừng thì báo, cân nhắc chuyển cả nhóm sang 25 × 2 |
| 4 | **GRPO chậm 1,55 lần SFT** (G1 3,07 s/bước; G2 2,95) | Kernel GRPO hai lần ~5,8 giờ, gần một nửa quota còn lại tuần này | Mục 5 |
| 5 | **Quy tắc "2 độ lệch chuẩn" của GĐ5 lạc quan so với kiểm định t khi R chỉ có 3 seed:** A0 − R RADAR VI −5,65 điểm, p = 0,08; chỉ P0 − R có ý nghĩa (ItW, RADAR EN, FRR) | Kết luận RawBoost (GĐ5 phát hiện 4) là khả dĩ, chưa có ý nghĩa thống kê. Mọi so sánh G1 phải báo KTC theo seed, không chỉ KTC bootstrap theo câu | R seed 4, 5 (lần 2) đưa bậc tự do lên 4; chạy lại mục 5 khi có |
| 6 | **Hiệu chuẩn trên dev (chỉ tiếng thật VIVOS) làm actDCF ngoài miền tệ hơn**, ở mọi biến thể; chỉ giúp tập cùng nguồn tiếng thật | Trả lời ngày 3 của plan: lệch hiệu chuẩn sau fine-tune là do lệch nguồn tiếng thật, không sửa được bằng dev cùng nguồn | Báo cáo kết quả này; chấm lại với G1 khi có để xem GRPO có lệch ít hơn không |

## 5. Ngân sách quota

| Lần | Quota dùng | Còn lại | Ghi chú |
|---|---|---|---|
| Trước GĐ6 | | 11,60 giờ đến 21/11 | Kiểm 15/11 |
| Thử đo (`orpheus-finetune` v5) | 0,47 giờ | **11,12 giờ** đến 21/11 | Ước ~0,75 giờ; đẩy 9:17, được cấp máy ngay |

**Ngân sách sau lần thử (mục 3.2; kernel GRPO ~5,8 giờ, kernel SFT ~4,4 giờ):**

| Gói | Lần chạy | Kernel | Quota |
|---|---|---|---|
| Theo plan | G1 × 3, G2, G3 | 3 | ~17,4 giờ |
| + P2 | + G1 không RawBoost (ghép kernel thứ ba với G3) | 3 | ~17,4 giờ |
| + P1 | + G1 seed 4, 5 (GRPO) và R seed 4, 5 (SFT) | 5 | ~27,6 giờ |
| Có | 11,12 giờ tuần này (chỉ đủ **một** kernel GRPO; dư ~5,3 giờ, vừa một kernel SFT) + 30 giờ từ 21/11 | | 41,1 giờ |

Quota tuần này mất lúc 7 giờ 21/11. Hai kernel chạy song song được (quota tính theo từng phiên).

**Đã chốt (15/11, người dùng, sau lần thử):**
- Mọi lần GRPO: **batch 50 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**, `--oom-retry 0`, `--probe-mem`; còn lại đúng bộ ghi đè SFT (E1'). Nếu một lần hết bộ nhớ giữa chừng: báo người dùng, cân nhắc chuyển cả nhóm GRPO sang 25 × 2.
- Tuần này: **G1 s1 + G2 s1** (`orpheus-finetune` v6) **song song R s4 + R s5** (`orpheus-finetune-b` v2): ~5,8 + ~4,4 = ~10,2 giờ trong 11,12 giờ (dư ~0,9 giờ).
- **P1: làm** (R và G1 mỗi cấu hình 5 seed). **P2: không làm** (không có G1 không RawBoost; so G1 với A0 phải ghi rõ hai yếu tố khác nhau). **P3: không cắt** (ngày 2 ít mẫu giữ như plan).

## 6. Các lần chạy

| Lần | Kernel | GPU 0 | GPU 1 | Ước tính | Tuần quota |
|---|---|---|---|---|---|
| 1 | `orpheus-finetune` v6 | G1 seed 1 | G2 seed 1 | ~5,8 giờ | đến 21/11 |
| 2 | `orpheus-finetune-b` v2 (song song lần 1) | R seed 4 | R seed 5 | ~4,4 giờ | đến 21/11 |
| 3 | kernel rảnh trước | G1 seed 2 | G1 seed 3 | ~5,8 giờ | từ 21/11 |
| 4 | kernel còn lại | G3 seed 1 (`rl_config={beta: 0.0}`) | G1 seed 4 | ~5,8 giờ | từ 21/11 |
| 5 | | G1 seed 5 | (trống: ngày 2 ít mẫu hoặc chốt sau) | ~5,8 giờ | từ 21/11 |

Mỗi lần huấn luyện chấm sau fine-tune trên `seaspoof_vi_dev`, `seaspoof_vi_test`, `radar_vi`, `ood_forget`, `mlaad_vi_test` (như R seed 2, 3). Lần 3–5 tổng ~17,4 giờ trong 30 giờ tuần sau.

## 7. Phân tích không cần GPU (làm trong lúc chờ lần 1, 2; 15/11)

### 7.1. So sánh qua seed bằng kiểm định t (`src/analyze_ablation.py` mục 5, `results/finetune/analysis/seed_tests.csv`)

GĐ5 đọc các cấu hình 1 seed bằng quy tắc "cách trung bình R ~2 độ lệch chuẩn" (GĐ5 mục 3.4). Kiểm định chính thức: một bên 1 seed thì so với khoảng dự đoán của bên kia, t = (A − B)/(s·√(1 + 1/n)), bậc tự do n − 1 (giả định cấu hình 1 seed có cùng độ lệch seed với R); hai bên ≥ 2 seed thì Welch. Chênh lệch điểm EER (%), KTC 95%, p hai phía; R có 3 seed (bậc tự do 2):

| So sánh | RADAR VI | ItW | RADAR EN | FRR RADAR VI ở ngưỡng 0 |
|---|---|---|---|---|
| A0 − R | −5,65 [−13,18; 1,89], p = 0,08 | +0,47 [−8,03; 8,96], p = 0,84 | −2,63 [−6,65; 1,38], p = 0,11 | −5,61, p = 0,26 |
| N0 − R | −3,52 [−11,05; 4,01], p = 0,18 | −0,23, p = 0,92 | −2,33, p = 0,13 | −1,41, p = 0,73 |
| B2 − R | −2,99 [−10,53; 4,54], p = 0,23 | −4,23, p = 0,17 | **−4,23 [−8,25; −0,22], p = 0,045** | −7,06, p = 0,19 |
| P0 − R | +7,43 [−0,10; 14,97], p = 0,051 | **+9,27 [0,77; 17,76], p = 0,043** | **+6,27 [2,25; 10,28], p = 0,021** | **+20,67, p = 0,028** |

Đọc: với R chỉ 3 seed, hệ số t ở bậc tự do 2 là 4,30 (so với 1,96), nên khoảng tin cậy rất rộng. Chỉ P0 (bỏ post-training) kém R có ý nghĩa ở mức 5% (ItW, RADAR EN, FRR), và B2 tốt hơn R trên RADAR EN. A0 tốt hơn R 5,65 điểm RADAR VI vẫn chỉ ở p = 0,08: kết luận "RawBoost khi fine-tune làm kém" của GĐ5 là khả dĩ nhưng **chưa có ý nghĩa thống kê** theo kiểm định này (phát hiện 5). R seed 4, 5 (lần 2) đưa bậc tự do lên 4: với s = 1,52, nửa khoảng tin cậy cho cấu hình 1 seed từ 7,2 giảm còn ~4,6 điểm RADAR VI, đủ để A0 − R (−5,65) có ý nghĩa nếu độ lệch seed giữ nguyên.

### 7.2. Hiệu chuẩn điểm trên dev (ngày 3 của plan; `src/calibrate.py`, `results/calibration/`)

Cách làm: học s' = a·s + b trên `seaspoof_vi_dev` (1.035 thật VIVOS + 1.035 giả 5 hệ thống train) bằng hồi quy logistic có trọng số prior hiệu dụng của DCF ASVspoof 5 (β = 1,9 ⇒ P(thật) hiệu dụng = β/(1 + β) = 0,655; ngưỡng Bayes −ln β = −0,64). Ba biến thể: `affine` (học a, b); `shift` (a = 1, chỉ dời ngưỡng); `platt` (affine với nhãn làm mềm kiểu Platt, tránh hệ số phình khi dev tách được hoàn toàn). EER và minDCF không đổi (biến đổi đơn điệu); chỉ actDCF đổi. Kiểm chéo: actDCF − minDCF trước hiệu chuẩn trên RADAR VI khớp GĐ5 (S0 0,007; R s1 0,200; A0 0,192).

**actDCF trước và sau hiệu chuẩn** (thấp hơn là tốt hơn; minDCF là mức tốt nhất đạt được bằng chọn ngưỡng):

| Cấu hình | Tập | minDCF | actDCF gốc | affine | shift | platt |
|---|---|---|---|---|---|---|
| S0 MMS-300M | RADAR VI | 0,461 | **0,469** | 0,715 | 0,583 | 0,700 |
| R s1 | RADAR VI | 0,510 | **0,710** | 0,755 | 0,771 | 0,781 |
| R s2 | RADAR VI | 0,525 | **0,774** | 0,940 | 0,880 | 0,894 |
| R s3 | RADAR VI | 0,583 | **0,835** | 0,957 | 0,927 | 0,937 |
| A0 s1 | RADAR VI | 0,428 | **0,620** | 0,680 | 0,662 | 0,666 |
| S0 MMS-300M | ItW | 0,069 | **0,077** | 0,360 | 0,208 | 0,336 |
| R s1 | ItW | 0,211 | **0,777** | 0,827 | 0,858 | 0,867 |
| S0 MMS-300M | MLAAD | 0,118 | 0,202 | **0,136** | 0,161 | 0,139 |
| R s2 | MLAAD | 0,003 | 0,064 | **0,038** | 0,047 | 0,046 |

Toàn bộ 12 cấu hình × 7 tập (kể cả FRR, FAR ở ngưỡng Bayes): `results/calibration/calibration_output.txt`.

Mỗi kết quả, một câu giải thích:
1. **Hiệu chuẩn học trên dev làm actDCF ngoài miền tệ hơn ở gần như mọi cấu hình và cả ba biến thể** (RADAR VI, ItW, RADAR EN, `test_cvreal`), kể cả biến thể ổn định nhất (`shift`). Vậy nguyên nhân không phải do phép khớp suy biến.
2. **Nó chỉ giúp khi tiếng thật của tập kiểm tra cùng nguồn với dev** (VIVOS): MLAAD, `test_seen`, dev. Độ dời học được gần như luôn âm (b < 0, tức nâng ngưỡng): trên dev, tiếng thật VIVOS được chấm "thật" rất chắc, còn một phần câu giả lọt qua, nên dev đòi khắt khe hơn với tiếng thật; ngoài miền thì ngược lại, tiếng thật nguồn khác đã bị loại quá nhiều (FRR RADAR VI của R: 26–33% trước, 30–43% sau `affine`).
3. **Sau fine-tune dev tách được hoàn toàn** (EER dev 0% ở R, A0, N0, P0, B1), nên `affine` cho hệ số dốc rất lớn (A0 a = 45,9; R s2 a = 33,9) và không xác định rõ; `shift` và `platt` cho a ≈ 1, b ≈ −1 đến −2.
4. Hệ quả cho báo cáo: lệch hiệu chuẩn sau fine-tune (actDCF − minDCF ≈ 0,2 trên RADAR VI) **không sửa được bằng dev cùng nguồn với train**; cần dev có tiếng thật đa nguồn (không dùng Common Voice vì trùng nguồn RADAR). Khớp hạn chế đã dự báo ở GĐ5 L8 và phát hiện 9.

**Đã chốt (15/11, người dùng):** chạy GĐ6 trên Kaggle như hiện tại, một tài khoản; không thuê server (nên không cần chạy lại R, A0). Không tạo tài khoản Kaggle thứ hai: điều khoản Kaggle cấm một người có hơn một tài khoản đang hoạt động (theo kết quả tìm kiếm trích kaggle.com/terms; trang render bằng JavaScript nên chưa đọc trực tiếp), rủi ro mất cả tài khoản chính (dataset riêng tư, output và checkpoint GĐ5). Theo ước tính trước lần thử, tổng quota đủ (GĐ6 ~17 giờ; ~31 giờ nếu thêm 2 seed cho R và G1 cùng G1 không RawBoost; có ~10,85 giờ tuần này + 30 giờ từ 21/11); chỉ là xong GPU muộn hơn (~22–23/11).

**Đánh giá plan theo số liệu GĐ5 (15/11, đề xuất, chờ người dùng chốt sau lần thử):**

| # | Vấn đề | Số liệu | Đề xuất |
|---|---|---|---|
| P1 | 3 seed có thể không đủ để kết luận G1 so với R | Ngưỡng chênh lệch có ý nghĩa (t-test hai mẫu, 3 seed mỗi bên, giả định độ lệch seed của G1 như R): RADAR VI ~3,4 điểm (s = 1,52), ItW ~3,9 điểm (s = 1,71). Phần SFT làm hỏng mà GRPO có thể gỡ lại: RADAR VI R − S0 = 3,9 điểm; ItW 5,63 − 2,60 = 3,0 điểm (bài GRPO, MMS-300M: 14,17 − 2,91 = 11,3 điểm). Trên ItW, kể cả GRPO gỡ lại hết vẫn dưới ngưỡng | Thêm 2 seed cho cả R và G1 (~10–11 giờ, tuần sau): ngưỡng còn ~2,2 điểm RADAR VI, ~2,5 điểm ItW |
| P2 | G1 so với A0 lệch hai yếu tố (hàm loss và RawBoost) | A0 − R = −5,65 điểm RADAR VI (RawBoost khi fine-tune làm kém, GĐ5 phát hiện 4) | Thêm "G1 không RawBoost" (1 seed, `use_da=False`): bảng 2 × 2 {SFT, GRPO} × {có, không RawBoost} |
| P3 | Ngày 2 (ít mẫu N = 50/200/1000) không đo được gì | SEA-Spoof bão hoà từ S0 (mọi hệ thống ≤ 2,1%) | Cắt (mục cắt thứ 4 của plan), dành quota cho P1, P2 |
