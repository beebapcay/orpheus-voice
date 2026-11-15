# GĐ6. GRPO và cải tiến

Câu hỏi của giai đoạn: fine-tune bằng GRPO trên tiếng Việt có giữ khả năng tổng quát hoá tốt hơn SFT không, và thành phần nào của GRPO tạo ra khác biệt (câu hỏi 3 của plan). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ6 kèm "Lưu ý từ GĐ5". Đầu vào từ GĐ5: `chore/GD5-Ablation.md` mục 8 (tổng kết), mục 9 (lưu ý L1–L12), mục 6 (đối chiếu bài GRPO và code), mục 4.1 (lịch learning rate E1'). File này bắt đầu ngày 15/11.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Chuẩn bị lần thử đo GRPO: đọc code GRPO (YAML v1/v2, `main.py`, `algo/rl.py`), kiểm ghi đè lồng `rl_config`, tích luỹ gradient, mô phỏng số batch | Mục 2; `src/mem_probe.py`; `src/finetune.py` (`--probe-mem`, `--oom-retry`, `--train-longest`, `--discard-ckpt`, `--env`); `kaggle/finetune/run.py` | Xong (15/11), đã thử trên máy |
| 1 | **Lần thử đo bộ nhớ và tốc độ GRPO trên T4** (L1) | `orpheus-finetune` v5, mục 3 | **Xong** (15/11, 0,47 giờ): G1 batch 50 vừa với `expandable_segments`, 3,07 s/bước |
| 1 | Báo kết quả lần thử; người dùng chốt: Kaggle hay thuê server, dùng quota còn lại tuần này thế nào | Mục 3, 5 | **Đã chốt** (15/11, mục 5): Kaggle; batch 50 + `expandable_segments`; P1 làm, P2 không, P3 không cắt |
| 1 | Lần 1: G1 s1, G2 s1; lần 2 (song song): R s4, R s5 | `orpheus-finetune` v6, `orpheus-finetune-b` v2; mục 6, 10 | **Xong**: lần 2 (14:39, 4,66 giờ, mục 10.1), lần 1 (15:31, 5,53 giờ, mục 10.2) |
| 2–3 | G1 s2–s5, G3 (từ 21/11) | Mục 6 | Chưa |
| 1 | Phân tích không cần GPU: kiểm định qua seed, hiệu chuẩn, theo hệ thống sinh, mốc cơ chế | Mục 7 | Xong (15/11) |
| 1 | **Bộ đầy đủ trên server vast.ai** (76 lần, cùng phần cứng, 3 seed mọi cấu hình): script, chọn thí nghiệm | Mục 9, 12, `server/` | Script xong, thử trên máy (15/11); người dùng thuê máy **sau khi** xem kết quả lần 1, 2 Kaggle (mục 9) |
| 1 | Đánh giá toàn diện hướng thực nghiệm (người dùng yêu cầu): đối chiếu chế độ của bài, độ mạnh thống kê, nguồn ngẫu nhiên của seed, lực của phạt KL; sửa lỗi xoá checkpoint của bộ server; dấu vân tay batch ở tầng 0 | Mục 13 | Xong (15/11); chờ người dùng chốt các đề xuất ở 13.5 |
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
| 5 | **Quy tắc "2 độ lệch chuẩn" của GĐ5 lạc quan so với kiểm định t khi R chỉ có 3 seed:** A0 − R RADAR VI −5,65 điểm, p = 0,08; chỉ P0 − R có ý nghĩa (ItW, RADAR EN, FRR) | Kết luận RawBoost (GĐ5 phát hiện 4) là khả dĩ, chưa có ý nghĩa thống kê. Mọi so sánh G1 phải báo KTC theo seed, không chỉ KTC bootstrap theo câu | **Cập nhật sau lần 2 (R 5 seed, bậc tự do 4): A0 − R −5,83 điểm RADAR VI, p = 0,047 (vừa có ý nghĩa); R − S0 +4,12, p = 0,008** (mục 10.1) |
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

*Bảng dưới tính với R 3 seed (trước lần 2). Cập nhật với R 5 seed: mục 10.1.*

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
1. **Hiệu chuẩn học trên dev làm actDCF ngoài miền tệ hơn ở gần như mọi cấu hình và cả ba biến thể** (ngoại lệ duy nhất trên RADAR VI: R s5 với `affine` gần như không đổi, 0,901 → 0,899) (RADAR VI, ItW, RADAR EN, `test_cvreal`), kể cả biến thể ổn định nhất (`shift`). Vậy nguyên nhân không phải do phép khớp suy biến.
2. **Nó chỉ giúp khi tiếng thật của tập kiểm tra cùng nguồn với dev** (VIVOS): MLAAD, `test_seen`, dev. Độ dời học được gần như luôn âm (b < 0, tức nâng ngưỡng): trên dev, tiếng thật VIVOS được chấm "thật" rất chắc, còn một phần câu giả lọt qua, nên dev đòi khắt khe hơn với tiếng thật; ngoài miền thì ngược lại, tiếng thật nguồn khác đã bị loại quá nhiều (FRR RADAR VI của R: 26–33% trước, 30–43% sau `affine`).
3. **Sau fine-tune dev tách được hoàn toàn** (EER dev 0% ở R, A0, N0, P0, B1), nên `affine` cho hệ số dốc rất lớn (A0 a = 45,9; R s2 a = 33,9) và không xác định rõ; `shift` và `platt` cho a ≈ 1, b ≈ −1 đến −2.
4. Hệ quả cho báo cáo: lệch hiệu chuẩn sau fine-tune (actDCF − minDCF ≈ 0,2 trên RADAR VI) **không sửa được bằng dev cùng nguồn với train**; cần dev có tiếng thật đa nguồn (không dùng Common Voice vì trùng nguồn RADAR). Khớp hạn chế đã dự báo ở GĐ5 L8 và phát hiện 9.

### 7.3. EER cực đoan theo hệ thống sinh (ngày 3 của plan; `results/finetune/analysis/table_by_system_mlaad.csv`)

RADAR VI không có nhãn hệ thống theo câu, nên chỉ phân tích được trên SEA-Spoof (mọi hệ thống ≤ 2,1% với S0, ≤ 0,2% sau fine-tune: bão hoà, GĐ5 mục 3.1) và MLAAD `fake/vi` (5 TTS tiếng Việt chưa gặp, tiếng thật VIVOS). EER (%) theo hệ thống; trong ngoặc: % câu giả bị nhận là thật ở ngưỡng log-odds 0:

| Cấu hình | LEMAS-TTS | OmniVoice | VieNeu-TTS | **VoxCPM2** | vixTTS |
|---|---|---|---|---|---|
| S0 MMS-300M | 0,02 (0,4) | 0,09 (2,7) | 0,09 (0,1) | **15,78 (95,6)** | 0,00 (0,0) |
| S0 MMS-300M NDA | 0,00 (0,0) | 0,11 (0,9) | 0,00 (0,0) | **12,71 (79,7)** | 0,00 (0,0) |
| S0 W2V-Large | 0,02 (0,1) | 0,00 (0,0) | 0,00 (0,0) | **6,90 (36,5)** | 0,00 (0,0) |
| S0 W2V-Small | 0,29 (1,2) | 0,60 (1,9) | 0,02 (0,0) | **6,61 (35,3)** | 0,00 (0,0) |
| R s2 | 0,00 (0,0) | 0,00 (0,0) | 0,00 (0,0) | **0,29 (28,7)** | 0,00 (0,0) |
| R s3 | 0,00 (0,0) | 0,00 (0,0) | 0,00 (0,0) | **0,40 (31,7)** | 0,00 (0,0) |

Mỗi kết quả, một câu giải thích:
1. **Một hệ thống chiếm gần hết lỗi:** với mọi mô hình dùng thẳng, VoxCPM2 có EER 6,6–15,8% trong khi bốn hệ thống còn lại ≤ 0,6%; MMS-300M Default nhận 95,6% câu VoxCPM2 là thật. Cùng kiểu "EER phân tán cực đoan theo hệ thống sinh" mà VietPrism ghi nhận (0,00% đến 99,99%).
2. **Fine-tune sửa thứ hạng nhưng không sửa ngưỡng cho VoxCPM2:** EER còn 0,29–0,40%, nhưng 29–32% câu VoxCPM2 vẫn lọt qua ở ngưỡng 0 (bốn hệ thống khác 0%). Điểm của VoxCPM2 nằm giữa tiếng thật VIVOS và các hệ thống khác: tách được nếu chọn lại ngưỡng, nhưng ngưỡng mặc định thì không.
3. Hai bản 317M tiếng Anh (W2V-Large, W2V-Small) dùng thẳng chỉ kém trên VoxCPM2 bằng nửa hai bản MMS-300M (6,6–6,9% so với 12,7–15,8%), cùng chiều với W2V-Large ≤ MMS-300M ở RADAR VI (GĐ3, GĐ5 B2).

Khi có G1: so cột VoxCPM2 (cả EER và % lọt ở ngưỡng 0) của G1 với R. Giả thuyết cơ chế (GĐ5 mục 6.3) dự đoán GRPO ít kéo các câu mô hình đang sai chắc chắn, nên nếu VoxCPM2 giống câu "sai chắc chắn" của train thì G1 có thể để lọt nhiều hơn R.

### 7.4. Mốc SFT cho kiểm giả thuyết cơ chế (L9; `src/analyze_mechanism.py`, `results/finetune/analysis/mechanism_output.txt`)

R, A0 (và G1, G2, G3) cùng xuất phát từ S0 MMS-300M, nên so được từng câu: z0 = log-odds của nhãn đúng theo S0 (dương: S0 đúng), Δz = z sau fine-tune − z0 (dương: tiến về đúng). Chia câu theo z0: "sai chắc" (z0 < −3, tức p < 0,05), "sai", "đúng", "đúng chắc" (z0 ≥ 3). Mốc SFT (R 3 seed, A0):

| Tập | Lớp, nhóm theo S0 | Số câu | Δz trung vị (R s1–s3; A0) | % lật nhãn ở ngưỡng 0 (R s1–s3; A0) |
|---|---|---|---|---|
| dev | giả, **sai chắc** | 56 | **10,8–11,2; 11,6** | 93–100 thành đúng; 95 |
| dev | giả, sai | 110 | 8,5; 8,9 | 100; 100 |
| dev | giả, đúng chắc | 646 | 1,3; 1,6 | 0 |
| RADAR VI | giả, sai chắc | 1.501 | 3,1–3,8; 4,5 | 39–43 thành đúng; 47 |
| RADAR VI | thật, đúng | 756 | −6,2 đến −7,2; −6,7 | **86–93 thành sai**; 87 |
| RADAR VI | thật, đúng chắc | 5.907 | −0,6 đến −1,2; −0,3 | **19–24 thành sai**; 16 |
| ItW | thật, đúng chắc | 915 | −1,8 đến −4,1; −2,6 | 25–38 thành sai; 31 |
| RADAR EN | thật, đúng chắc | 809 | −5,9 đến −6,8; −3,7 | 49–51 thành sai; 38 |

Đọc:
1. **Trên dev, SFT kéo mạnh nhất đúng những câu S0 sai chắc** (Δz ~11, gấp ~8 lần câu S0 đã đúng chắc), khớp trọng số 1 − p của SFT (GĐ5 mục 6.3).
2. **Ngoài miền, SFT đẩy tiếng thật về phía "giả" ở mọi nhóm**, kể cả câu S0 đúng chắc (RADAR VI 19–24%, RADAR EN ~50% bị lật). Đây là cơ chế của FRR tăng (GĐ5 phát hiện 3, 9). A0 (không RawBoost) đẩy ít hơn R ở nhóm đúng chắc (RADAR VI 16% so với 19–24%; RADAR EN 38% so với 49–51%).

Dự đoán cho G1 (giả thuyết, sẽ kiểm khi có điểm): (a) trên dev, Δz của nhóm "giả, sai chắc" nhỏ hơn R nhiều so với các nhóm khác (GRPO trọng số p(1 − p)/σ, gần 0 khi p → 0); (b) nếu GRPO quên ít hơn thì tỉ lệ lật "thật, đúng chắc" ngoài miền thấp hơn R.

## 8. Ngày 2 (ít mẫu): phương án

**Đã chốt (15/11, người dùng: "hợp lý và cho kết quả tốt nhất, không ngại thời gian, thuê server được"):** phương án C (cùng 4.563 bước và lịch E1' như bản đủ, chỉ đổi lượng dữ liệu; điểm bản đủ chính là R và G1), **3 seed mỗi điểm, chạy trên server** trong bộ đầy đủ ở mục 9. Bảng dưới giữ để ghi lý do chọn.

Plan: so SFT và G1 với N = 50, 200, 1000 câu mỗi lớp (P3 không cắt). Tập con có sẵn từ GĐ4, lồng nhau, giả chia đều 5 hệ thống train: `seaspoof_vi_train_n50` (0,10 giờ), `_n200` (0,42 giờ), `_n1000` (2,03 giờ). Số batch mỗi epoch ở batch 50 s (mô phỏng `DynamicBatchSampler`, khớp pilot GĐ4 với n50): **12 / 42 / 178**. Điểm N đủ (8.910 mỗi lớp) là R và G1 đã có.

| Phương án | Số bước | Lịch learning rate | Đo được gì | Chi phí |
|---|---|---|---|---|
| A. 3 epoch như bản đủ | 36 / 126 / 534 | Giữ E1' (warmup 460): N = 50, 200 không ra khỏi warmup, lr tối đa 7,8e-8 / 2,7e-7 | Gần như không fine-tune ở N nhỏ: lẫn "ít dữ liệu" với "ít bước" | Rẻ |
| **B. Cùng 534 bước cho mọi N** (= 3 epoch của n1000; đề xuất) | 534 (45 / 13 / 3 epoch) | warmup 53 (10%), lr đỉnh 1e-6 rồi gần như giữ nguyên (cùng dạng E1') | Chỉ đổi lượng dữ liệu, giữ số lần cập nhật; N nhỏ lặp lại dữ liệu nhiều (dễ overfit, đó cũng là điều cần đo) | Huấn luyện ~18 phút (SFT), ~27 phút (G1) |
| C. Cùng 4.563 bước như bản đủ | 4.563 | E1' | Như B nhưng sát bản đủ | ~30 giờ cho 6 lần: không vừa |

Đề xuất thêm cho B: chỉ chấm `seaspoof_vi_test` (trong miền, kể cả F5-TTS, MiniMax để riêng), `radar_vi`, `ood_forget` (thước đo chính F3); bỏ dev và MLAAD (bão hoà) để chấm còn ~1 giờ mỗi lần. Mỗi lần ~1,5 giờ; 6 lần (SFT, G1 × 3 N, 1 seed) ≈ 3 lần trên GPU trống của lần 5 + 3 lần ở một kernel nữa (~3,1 giờ). Tổng tuần sau ~20,5 giờ trong 30. Hạn chế: 1 seed mỗi N, nên chênh lệch < ~3 điểm RADAR VI không kết luận được (GĐ5 phát hiện 8).

**Đã chốt (15/11, người dùng):** chạy GĐ6 trên Kaggle như hiện tại, một tài khoản; không thuê server (nên không cần chạy lại R, A0). Không tạo tài khoản Kaggle thứ hai: điều khoản Kaggle cấm một người có hơn một tài khoản đang hoạt động (theo kết quả tìm kiếm trích kaggle.com/terms; trang render bằng JavaScript nên chưa đọc trực tiếp), rủi ro mất cả tài khoản chính (dataset riêng tư, output và checkpoint GĐ5). Theo ước tính trước lần thử, tổng quota đủ (GĐ6 ~17 giờ; ~31 giờ nếu thêm 2 seed cho R và G1 cùng G1 không RawBoost; có ~10,85 giờ tuần này + 30 giờ từ 21/11); chỉ là xong GPU muộn hơn (~22–23/11).

**Đánh giá plan theo số liệu GĐ5 (15/11, đề xuất, chờ người dùng chốt sau lần thử):**

| # | Vấn đề | Số liệu | Đề xuất |
|---|---|---|---|
| P1 | 3 seed có thể không đủ để kết luận G1 so với R | Ngưỡng chênh lệch có ý nghĩa (t-test hai mẫu, 3 seed mỗi bên, giả định độ lệch seed của G1 như R): RADAR VI ~3,4 điểm (s = 1,52), ItW ~3,9 điểm (s = 1,71). Phần SFT làm hỏng mà GRPO có thể gỡ lại: RADAR VI R − S0 = 3,9 điểm; ItW 5,63 − 2,60 = 3,0 điểm (bài GRPO, MMS-300M: 14,17 − 2,91 = 11,3 điểm). Trên ItW, kể cả GRPO gỡ lại hết vẫn dưới ngưỡng | Thêm 2 seed cho cả R và G1 (~10–11 giờ, tuần sau): ngưỡng còn ~2,2 điểm RADAR VI, ~2,5 điểm ItW |
| P2 | G1 so với A0 lệch hai yếu tố (hàm loss và RawBoost) | A0 − R = −5,65 điểm RADAR VI (RawBoost khi fine-tune làm kém, GĐ5 phát hiện 4) | Thêm "G1 không RawBoost" (1 seed, `use_da=False`): bảng 2 × 2 {SFT, GRPO} × {có, không RawBoost} |
| P3 | Ngày 2 (ít mẫu N = 50/200/1000) không đo được gì | SEA-Spoof bão hoà từ S0 (mọi hệ thống ≤ 2,1%) | Cắt (mục cắt thứ 4 của plan), dành quota cho P1, P2 |

## 9. Bộ thí nghiệm đầy đủ trên server (quyết định 15/11)

**Người dùng (15/11):** thuê server vast.ai, chạy song song với Kaggle; vì kết quả hai phần cứng không nên trộn trong một phép so sánh, chọn các bước cần làm lại để chạy **trọn một bộ** trên server.

**Vì sao làm lại cả GĐ5, không chỉ GĐ6:** mọi bảng của báo cáo so các cấu hình với nhau (G1 với R, A0; R với P0, B2, B1, N0; đường cong ít mẫu với bản đủ). Nếu một phía chạy trên T4 và phía kia trên GPU khác thì chênh lệch lẫn cả yếu tố phần cứng: GPU Ampere/Ada mặc định dùng TF32 cho convolution (T4 không có), tốc độ đọc dữ liệu khác nên chuỗi ngẫu nhiên của RawBoost khác. Độ lệch giữa seed đã ~1,5 điểm RADAR VI (GĐ5 phát hiện 8), nên một độ lệch hệ thống nhỏ do phần cứng cũng đủ làm sai kết luận. Làm lại cả bộ cũng sửa điểm yếu 1 seed của GĐ5 (mục 7.1: A0 − R chỉ p = 0,08).

*Bảng tầng dưới là bản đầu (54 lần). **Đã cập nhật 15/11 thành 73 lần theo hướng "chẩn đoán và sửa" (mục 12); bảng hiện hành: `server/README.md` mục 1.***

**Bộ server (54 lần, `server/experiments.py`), mọi cấu hình 3 seed:**

| Tầng | Nội dung | Số lần |
|---|---|---|
| 0 | Thử đo trên máy thuê: SFT và G1 × {2, 6} worker đọc dữ liệu, 200 bước (tốc độ, bộ nhớ, CPU) | 4 |
| 1 | S0 (4 backbone dùng thẳng, chấm 5 tập); R, G1, A0 × 3 seed | 13 |
| 2 | G2, G3 × 3 seed | 6 |
| 3 | P0, B2, N0, B1 × 3 seed | 12 |
| 4 | Ít mẫu (mục 8, phương án C): SFT, G1 × N = 50/200/1000 × 3 seed | 18 |
| 5 | R, G1 seed 4, 5 (P1); mốc XLS-R-2B dùng thẳng trên RADAR VI | 5 |

Công thức: đúng như Kaggle (đã đối chiếu từng tham số với bản đẩy `orpheus-finetune` v6 và `orpheus-finetune-b` v2: G1_s1, R_s4 khớp), thêm `NVIDIA_TF32_OVERRIDE=0` (fp32 như T4) và `--oom-retry 0` cho mọi lần. Ít mẫu: dừng bằng `optimizer_step_limit=4563`, số epoch = ⌈4.563 / batch mỗi epoch⌉ + 2 (383 / 111 / 28). Số worker đọc dữ liệu chốt sau tầng 0 và giữ nguyên cho cả bộ (chỉ ảnh hưởng tốc độ và chuỗi ngẫu nhiên của RawBoost, như một seed).

**Vai trò hai bộ:** bộ server là bộ chính cho bảng kết quả (cùng phần cứng, 3 seed mọi cấu hình). Bộ Kaggle (T4: R × 5, G1 × 5, G2, G3; A0, P0, B2, B1, N0 × 1) là lần lặp độc lập: chỉ so ở mức "kết luận có giữ nguyên không" (cùng chiều, cùng cỡ), không trộn số. Kết quả server lưu riêng ở `results/server/` (không lẫn `results/finetune/`, `results/zeroshot/`).

**Code (`server/`, hướng dẫn `server/README.md`):**

| File | Chạy ở | Việc |
|---|---|---|
| `experiments.py` | server | Danh sách 54 lần theo tầng; nguồn duy nhất |
| `run.py` | server | Điều phối: mỗi GPU một hàng đợi theo tầng, chạy tiếp được sau khi mất kết nối (lần xong thì bỏ qua), trạng thái `results/jobs/`, `results/status.txt` mỗi phút, `--dry-run` |
| `bootstrap.sh` | server | Gói hệ thống, môi trường như Kaggle + toy example (6,11%), môi trường điều phối, tải RADAR/ItW bằng `src/prep_data.py`, kiểm dữ liệu, tải trước checkpoint |
| `check_data.py` | server | Số câu và ID của mọi CSV khớp `protocols/ids/`, đủ file âm thanh |
| `vast.sh` | máy | SSH/rsync: `init`, `check`, `code`, `data`, `bootstrap`, `start`, `status`, `log`, `fetch`, `collect`, `stop`, `cleanup` |
| `collect.py` | máy | Gộp kết quả vào `results/server/`, bằng chứng vào `results/runs/server-<nhãn>/` (code, môi trường, log, `RUN.md`) |

Đã thử trên máy (15/11): danh sách 54 lần và tham số khớp Kaggle; chạy khô bộ điều phối với 4 GPU (54 lần, không lần nào bị hai GPU nhận trùng); `check_data.py` trên SEA-Spoof và MLAAD ở máy (khớp danh sách câu, đủ file); `collect.py` và các script phân tích trên cây kết quả giả lập từ output Kaggle v4. **Chưa thử trên server thật.** Sửa kèm: `finetune.py` khoá file khi ghi `train_summary.csv` (nhiều tiến trình cùng ghi).

**Ước tính:** ~245 giờ GPU tính theo T4 (SFT ~4,3 giờ, GRPO ~5,6 giờ mỗi lần, cả chấm). Tốc độ máy thuê so với T4 chỉ biết sau tầng 0; với 4 GPU nhanh hơn T4 2,5 lần: ~25 giờ.

**Cần người dùng:** thuê máy theo `server/README.md` mục 2 (4 × GPU ≥ 24 GB cùng loại, không RTX 50xx; ≥ 8 nhân CPU và ≥ 16 GB RAM mỗi GPU; ≥ 250 GB ổ; on-demand), thêm khoá SSH vào vast.ai, gửi chuỗi kết nối SSH.

**Quyết định (15/11, người dùng):** chờ kết quả Kaggle lần 1, 2 (G1 s1, G2 s1, R s4, R s5; dự kiến ~15:45) rồi mới thuê server. Tiêu chí xem G1 s1 trước khi chạy GRPO ở quy mô lớn: (1) chạy hết 4.563 bước, không hết bộ nhớ, loss không NaN; (2) đỉnh bộ nhớ và giây mỗi bước khớp lần thử đo (13,9 GB, 3,07 s); (3) trong miền (dev, `test_seen`) tốt hơn S0, tức GRPO có học; (4) RADAR VI, ItW, FRR ở ngưỡng 0 nằm trong khoảng hợp lý so với R (5 seed). Không đạt (1)–(3) thì sửa cấu hình GRPO trước khi chạy trên server.

## 10. Kết quả các lần chạy GĐ6

### 10.1. Lần 2 (`orpheus-finetune-b` v2, 15/11, 4,66 giờ; `results/runs/orpheus-finetune-b-v2/`)

COMPLETE, 0 cảnh báo nạp trọng số, code `orpheus-src` bản `a08bc23`. R s4, R s5: mỗi lần 4.563 bước (3 × 1.521), learning rate cuối 6,83e-07 (đúng lịch E1'), 2,25 và 2,07 s mỗi bước, GPU 99,5%. Cấu hình chỉ khác R s1–s3 ở seed (mục 6).

**R qua 5 seed, EER (%)** (thấp hơn là tốt hơn; `results/finetune/analysis/table_main.csv`, `table_seeds.csv`):

| Tập | s1 | s2 | s3 | s4 | s5 | **Trung bình ± độ lệch chuẩn** | S0 MMS-300M |
|---|---|---|---|---|---|---|---|
| RADAR VI | 23,07 | 23,23 | 25,77 | 26,63 | 22,32 | **24,20 ± 1,88** | 20,08 |
| ItW | 7,60 | 4,50 | 4,80 | 6,20 | 9,10 | **6,44 ± 1,93** | 2,60 |
| RADAR EN | 26,70 | 28,10 | 28,10 | 31,60 | 27,50 | **28,40 ± 1,88** | 26,20 |
| FRR RADAR VI ở ngưỡng 0 (%) | 29,21 | 33,83 | 35,06 | 23,71 | 42,80 | **32,92 ± 7,10** | 4,81 |
| `test_cvreal` | 2,70 | 4,56 | 6,15 | 6,10 | 4,09 | 4,72 ± 1,45 | 7,44 |
| MLAAD | | 0,09 | 0,16 | 0,28 | 0,21 | 0,19 ± 0,08 | 5,18 |

**Kiểm định qua seed với R 5 seed** (chênh lệch điểm EER %, KTC 95%, p hai phía; `seed_tests.csv`). R so với S0: t-test một mẫu (S0 dùng thẳng, không có độ lệch seed). Cấu hình 1 seed so với R: khoảng dự đoán từ 5 seed của R (bậc tự do 4).

| So sánh | RADAR VI | ItW | RADAR EN | FRR RADAR VI ở ngưỡng 0 |
|---|---|---|---|---|
| **R − S0** (SFT so với dùng thẳng) | **+4,12 [1,78; 6,45], p = 0,008** | **+3,84 [1,44; 6,24], p = 0,011** | +2,20 [−0,13; 4,53], p = 0,059 | **+28,11 [19,30; 36,92], p < 0,001** |
| A0 − R | **−5,83 [−11,54; −0,11], p = 0,047** | −0,34, p = 0,88 | −3,40, p = 0,17 | −5,84, p = 0,50 |
| N0 − R | −3,70 [−9,42; 2,02], p = 0,15 | −1,04, p = 0,65 | −3,10, p = 0,21 | −1,64, p = 0,84 |
| B2 − R | −3,17 [−8,89; 2,54], p = 0,20 | −5,04, p = 0,076 | −5,00, p = 0,072 | −7,28, p = 0,40 |
| P0 − R | **+7,25 [1,54; 12,97], p = 0,024** | **+8,46 [2,58; 14,34], p = 0,016** | +5,50, p = 0,056 | +20,45, p = 0,058 |

Mỗi kết quả, một câu giải thích:
1. **SFT làm kém ngoài miền với MMS-300M là kết luận có ý nghĩa thống kê** (câu hỏi 2): qua 5 seed, RADAR VI kém S0 4,12 điểm (p = 0,008), ItW 3,84 điểm (p = 0,011), và tỉ lệ thật bị coi là giả trên RADAR VI tăng 28 điểm (p < 0,001). Đồng thời MLAAD (tiếng thật cùng nguồn VIVOS) tốt hơn S0 ~5 điểm ở cả 4 seed.
2. **Độ lệch giữa seed lớn hơn ước tính từ 3 seed:** RADAR VI 1,52 → 1,88; RADAR EN 0,81 → 1,88; FRR ở ngưỡng 0 dao động 23,7–42,8% (độ lệch chuẩn 7,1). EER ổn định hơn nhiều so với lỗi ở ngưỡng cố định: vị trí ngưỡng (hiệu chuẩn) sau fine-tune phụ thuộc mạnh vào seed.
3. **RawBoost khi fine-tune làm kém RADAR VI: vừa có ý nghĩa** (A0 − R −5,83, p = 0,047, A0 1 seed). Post-training quan trọng: có ý nghĩa (P0 − R). Ngôn ngữ pre-training (B2) và bản NDA (N0): chưa có ý nghĩa với 1 seed.
4. Mốc cơ chế (mục 7.4) và hiệu chuẩn (mục 7.2) của R s4, s5 cùng mẫu hình với s1–s3: trên dev, câu giả S0 sai chắc được kéo Δz 9,8–11,2; trên RADAR VI, 13–33% câu thật S0 đúng chắc bị lật thành sai; hiệu chuẩn trên dev không giúp RADAR VI.

### 10.2. Lần 1 (`orpheus-finetune` v6, 15/11, 5,53 giờ; `results/runs/orpheus-finetune-v6/`)

COMPLETE, 0 cảnh báo nạp trọng số, code `a08bc23`. Cả hai lần 4.563 bước, learning rate cuối 6,83e-07, không hết bộ nhớ, không NaN (dòng duy nhất khớp "nonfinite" là thông tin cấu hình `skip_nonfinite_grads ... False`, R cũng có). G1: 3,34 s/bước (lần thử đo 3,07), đỉnh cấp phát 13.925 MiB (đúng bằng trường hợp xấu nhất của lần thử đo), giữ 14.260 MiB; G2: 2,62 s/bước, 12.682 MiB. Kernel 5,53 giờ (ước 5,8).

**Đánh giá theo 4 tiêu chí (mục 9):** (1) đạt; (2) đạt (bộ nhớ khớp; tốc độ G1 chậm hơn lần đo 9%, G2 nhanh hơn 11%); (3) đạt: GRPO có học (dev 0,00%, `test_seen` 0,04%, F5-TTS 0,04%, MLAAD 0,05% so với S0 1,16 / 0,81 / 0,57 / 5,18%); (4) đạt: mọi thước đo ngoài miền nằm trong khoảng của R 5 seed hoặc tốt hơn.

**EER (%), thấp hơn là tốt hơn** (R: trung bình ± độ lệch chuẩn 5 seed, khoảng [nhỏ nhất; lớn nhất]):

| Cấu hình | `test_seen` | `test_cvreal` | MLAAD | **RADAR VI** | **ItW** | **RADAR EN** | FRR RADAR VI ở ngưỡng 0 | actDCF − minDCF RADAR VI |
|---|---|---|---|---|---|---|---|---|
| S0 MMS-300M | 0,81 | 7,44 | 5,18 | 20,08 | 2,60 | 26,20 | 4,81 | 0,01 |
| R (5 seed) | 0,04 | 4,72 ± 1,45 | 0,19 | **24,20 ± 1,88** [22,32; 26,63] | **6,44 ± 1,93** [4,5; 9,1] | **28,40 ± 1,88** [26,7; 31,6] | 32,92 ± 7,10 [23,7; 42,8] | 0,09–0,40 |
| A0 s1 | 0,04 | 2,79 | | 18,37 | 6,10 | 25,00 | 27,09 | 0,19 |
| **G1 s1** (GRPO đủ) | 0,04 | 4,16 | 0,05 | **23,53** | **5,90** | **25,70** | 39,47 | 0,37 |
| **G2 s1** (không clip, không mô hình cũ) | 0,04 | 4,29 | 0,17 | **23,13** | **4,30** | **26,00** | 39,90 | 0,39 |

So với R (khoảng dự đoán từ 5 seed, bậc tự do 4): G1 − R RADAR VI −0,67 (p = 0,76), ItW −0,54 (p = 0,81), RADAR EN −2,70 (p = 0,26), FRR +6,55 (p = 0,45); G2 − R RADAR VI −1,07 (p = 0,63), ItW −2,14 (p = 0,37), RADAR EN −2,40 (p = 0,31). Không chênh lệch nào có ý nghĩa.

**Kiểm giả thuyết cơ chế (mục 7.4; trung bình theo seed cho R, `mechanism_output.txt`):**

| Tập | Lớp, nhóm theo S0 | Δz trung vị R / G1 / G2 | % lật nhãn ở ngưỡng 0 R / G1 / G2 |
|---|---|---|---|
| dev | giả, S0 sai chắc (56 câu) | 10,8 / **11,6** / 11,9 | 95 / 100 / 100 thành đúng |
| dev | giả, S0 đúng chắc | 1,3 / 1,7 / 2,0 | |
| RADAR VI | giả, S0 sai chắc (1.501) | 3,7 / **6,0** / 5,1 | 42 / 51 / 49 thành đúng |
| RADAR VI | thật, S0 đúng chắc (5.907) | −1,1 / 0,0 / 0,2 | 22 / **30** / 30 thành sai |
| ItW | thật, S0 đúng chắc (915) | −3,4 / −2,8 / −3,7 | 33 / **39** / 44 thành sai |
| RADAR EN | thật, S0 đúng chắc (809) | −6,9 / −9,6 / −9,1 | 53 / **63** / 58 thành sai |

**MLAAD VoxCPM2** (% câu giả lọt ở ngưỡng 0): R 29–59%, **G1 8,1%**, G2 27,4%. **Hiệu chuẩn:** actDCF RADAR VI G1 0,883, G2 0,887 (R 0,71–0,90); hiệu chuẩn trên dev vẫn không giúp.

Mỗi kết quả, một câu giải thích (1 seed cho G1, G2):
1. **Chưa thấy GRPO giữ tổng quát hoá tốt hơn SFT trên tiếng Việt:** G1 và G2 nằm trong khoảng của R ở mọi thước đo ngoài miền, và cũng kém S0 trên RADAR VI (+3,5 và +3,1 điểm) như SFT. Khác bài GRPO (MMS-300M: SFT 14,17%, GRPO 5,91% trên ItW), nhưng ở ta SFT chỉ làm quên ~3,8 điểm ItW nên khoảng để GRPO gỡ lại nhỏ hơn nhiều (P1).
2. **Giả thuyết cơ chế (a) không được ủng hộ:** trên dev, G1 kéo câu giả S0 sai chắc mạnh **bằng hoặc hơn** SFT (Δz 11,6 so với 10,8), không "bỏ qua" như dự đoán từ kỳ vọng gradient p(1 − p)/σ_batch. Giả thuyết (b) cũng không: G1 lật tiếng thật đúng chắc sang giả **nhiều hơn** R ngoài miền (RADAR VI 30% so với 22%, RADAR EN 63% so với 53%), nên FRR cao hơn (39,5% so với 32,9%).
3. **G1 dịch cả phân bố về phía "giả" mạnh hơn SFT:** câu giả được kéo về đúng nhiều hơn (RADAR VI Δz 6,0 so với 3,7; VoxCPM2 chỉ lọt 8% so với 29–59%), câu thật bị đẩy sang giả nhiều hơn; EER gần như không đổi. Khớp dự đoán thứ 3 ở GĐ5 mục 6.3: khi batch gần như đúng hết, σ_batch nhỏ nên 1/σ_batch khuếch đại bước cập nhật (batch đúng 99% thì ×10), tức GRPO trong code là một bản fine-tune "mạnh tay" hơn SFT chứ không nhẹ tay hơn; phạt KL β = 0,04 không đủ giữ.
4. **G2 ≈ G1** (bỏ clip và mô hình cũ không đổi kết quả), như bài báo kết luận (GRPO_s gần GRPO).
5. Pipeline GRPO chạy đúng và ổn định; đủ điều kiện chạy GRPO ở quy mô lớn (server hoặc Kaggle tuần sau).

## 11. Đánh giá lại hướng thực nghiệm (15/11, sau lần 1, 2; người dùng yêu cầu)

**Kết luận:** hướng đúng và vững cho câu hỏi 1, 2. Với câu hỏi 3, thiết kế hiện tại đúng kỹ thuật (đúng code gốc, đúng công thức, so sánh công bằng) nhưng đặt GRPO vào một chế độ khác hẳn bài báo, nên nhiều khả năng chỉ cho kết quả "GRPO ≈ SFT" mà không trả lời được vì sao. Nên chỉnh phần GRPO trước khi chạy quy mô lớn.

**Phần vững** (số liệu mục 10): pipeline đã kiểm (toy 6,11%, 4 tiêu chí của G1, R 5 seed); câu hỏi 2: SFT làm kém ngoài miền có ý nghĩa thống kê (R − S0: RADAR VI +4,12, p = 0,008; ItW +3,84, p = 0,011; FRR +28, p < 0,001); câu hỏi 1: post-training (P0 − R, p ≤ 0,024), RawBoost khi fine-tune (A0 − R −5,83, p = 0,047), kích thước (B1 kém B2 13,6 điểm, 1 seed); ngôn ngữ pre-training không giúp.

**Vấn đề 1: miền fine-tune đã bão hoà, nên GRPO chạy ngoài vùng nó được thiết kế.** Kiểm trên điểm dev (mô phỏng batch 12 câu × 64 nhãn lấy mẫu):

| Mô hình | p(nhãn đúng) trung bình trên dev | 1/σ_batch trung vị | % batch mọi nhãn lấy mẫu đều đúng (lợi thế = 0, chỉ còn KL) | % câu dev đúng với p > 0,999 | % câu RADAR VI đúng với p > 0,999 |
|---|---|---|---|---|---|
| S0 (đầu fine-tune) | 0,909 | 3,6 | 1,6 | 48,4 | 23,4 |
| R s1–s5 (cuối) | 0,999 | rất lớn | 66 (R s1) | 91–97 | 39–43 |
| G1 s1 (cuối) | 0,9995 | rất lớn | 76 | **99,4** | **52,9** |
| G2 s1 (cuối) | | | | **99,7** | **57,5** |

Theo kỳ vọng gradient (note `Post-Training-Va-GRPO.md` mục 8): với câu đã đúng chắc (p = 0,999), SFT kéo tiếp với lực 1 − p = 0,001 (gần như dừng), còn GRPO (chuẩn hoá theo nhóm hoặc theo batch) với lực ≈ √(p(1 − p)) ≈ 0,032, tức **~30 lần SFT**. Ở SEA-Spoof, phần lớn câu train đã đúng chắc từ sớm, nên cập nhật của GRPO bị chi phối bởi việc **tiếp tục làm sắc các câu đã đúng**: mô hình tự tin hơn R (99,4% câu dev đúng với p > 0,999 so với 91–97%), cả phân bố điểm dịch xa hơn, tiếng thật nguồn khác bị đẩy sang "giả" nhiều hơn (mục 10.2). Bài báo fine-tune trên DFE24, nơi MMS-300M dùng thẳng có EER 29,23% (GĐ5 mục 6.1): phần lớn câu ở ranh giới, nơi GRPO và SFT kéo tương đương, nên không gặp chế độ này. Giả thuyết cũ (GRPO "bỏ qua câu sai chắc", GĐ5 mục 6.3) nhìn nhầm phía: điều quyết định ở miền bão hoà là phía câu **đã đúng**.

**Vấn đề 2: "tổng quát hoá" đo được bị chi phối bởi tiếng thật một nguồn.** Lỗi ngoài miền chính sau fine-tune ở mọi cấu hình là tiếng thật không phải VIVOS bị coi là giả (FRR RADAR VI +28 điểm, có ý nghĩa), trong khi MLAAD (tiếng thật VIVOS) tốt lên. Lối tắt "thật = VIVOS" nằm trong dữ liệu; đổi hàm loss từ SFT sang GRPO không xoá được lối tắt dữ liệu, chỉ đổi mức mô hình bám vào nó. Vì vậy chênh lệch SFT–GRPO bị nén lại.

**Vấn đề 3: cường độ fine-tune.** Lịch E1' giữ lr ~1e-6 gần như suốt 4.563 bước (bài báo thực tế ~2–3e-7, GĐ4 phát hiện 9); cùng lr, GRPO trong miền bão hoà còn mạnh tay hơn SFT (vấn đề 1). Không sai (công bằng giữa các cấu hình), nhưng xa chế độ của bài báo.

**Hệ quả cho câu hỏi 3:** chạy thêm seed của G1, G2, G3 như kế hoạch sẽ cho kết luận "GRPO không giữ tổng quát hoá tốt hơn SFT trong điều kiện này" có khoảng tin cậy: hợp lệ (plan: kết quả ngược giả thuyết vẫn là kết quả), nhưng chưa trả lời "thành phần nào tạo khác biệt" và "vì sao". Đề xuất đổi trọng tâm câu hỏi 3 thành: **vì sao GRPO không giúp khi miền đích đã bão hoà và tiếng thật một nguồn, và cần gì để nó giúp**, với các thí nghiệm kiểm trực tiếp cơ chế (chờ người dùng chốt):

| Thí nghiệm | Kiểm điều gì | Dự đoán nếu cách giải thích đúng |
|---|---|---|
| G1 với β = 1 (phạt KL mạnh; bài báo: ngoài miền tốt nhất) | Giữ mô hình gần điểm xuất phát có chặn được việc làm sắc không | Ít tự tin hơn G1, FRR và ItW gần S0 hơn, trong miền vẫn ~0% (đã bão hoà) |
| G1 với lr thấp (ví dụ 1e-7) | Do cường độ cập nhật hay do bản chất GRPO | Gần S0 hơn; nếu G1 lr thấp vẫn kém S0 thì do dữ liệu (vấn đề 2) |
| GRPO chuẩn hoá theo câu (công thức của bài, code dùng theo batch) | Chênh lệch giữa bài và code có quan trọng không | Gần G1 (cả hai đều làm sắc câu đã đúng); khác biệt nhỏ |
| SFT với "trọng số kiểu GRPO" (cross-entropy nhân p(y\|x), GĐ5 mục 6.3 b) | Cơ chế nằm ở trọng số theo p chứ không ở lấy mẫu | Giống G1 |

Phần ít mẫu (tầng 4) trong chế độ này chủ yếu đo "ít dữ liệu thì bão hoà nhanh hơn, quên ra sao", nên xếp sau các thí nghiệm cơ chế. Vấn đề 2 không sửa được bằng hàm loss; muốn kiểm phải thêm tiếng thật đa nguồn vào dữ liệu fine-tune (không dùng Common Voice, FOSD vì trùng RADAR), là thay đổi dữ liệu lớn: ghi làm hướng phát triển, hoặc làm nếu người dùng chốt.

## 12. Đã chốt sau đánh giá lại (15/11, người dùng đồng ý mọi đề xuất) và tổng hợp phát hiện GĐ6

**Quyết định**

| # | Việc | Đã chốt |
|---|---|---|
| Q1 | Trọng tâm câu hỏi 3 | **Chẩn đoán và sửa:** vì sao GRPO (đúng code gốc) không giữ tổng quát hoá tốt hơn SFT khi miền fine-tune đã bão hoà; thiết kế bản GRPO sửa theo nguyên nhân và kiểm so với SFT, GRPO gốc. Đây là phần "phân tích sâu + đề xuất cải tiến" (20%) |
| Q2 | Thí nghiệm sửa và kiểm cơ chế, mỗi cái 3 seed | (a) **G1s**: bỏ khuếch đại, `rl_config={std_floor: 1.0}` (kỳ vọng gradient p(1 − p), gần như bỏ chia cho độ lệch chuẩn; ý tưởng gần Dr. GRPO, phải đọc bài gốc trước khi trích); (b) **G1b**: β = 1 (núm của bài báo); (c) bản sửa tốt hơn trong (a), (b) ghép **không RawBoost** (G1sA hoặc G1bA; so với A0); (d) **G1lr**: lr 1e-7, đối chứng cường độ. Giữ G2, G3 của plan |
| Q3 | Phạm vi bộ server | Lõi (S0, R, G1, A0) + sửa và cơ chế + các yếu tố GĐ5 (P0, B2, N0, B1), mọi cấu hình 3 seed; ít mẫu và seed 4, 5 xếp cuối. 73 lần, `server/experiments.py`, `server/README.md` mục 1 |
| Q4 | Nơi chạy | **Thuê server vast.ai** (máy 2 × RTX 3090, 20 vCPU, 81 GB RAM, 775 GB đã xem là đủ; khuyến nghị máy verified nếu có). Bộ server là bộ chính; Kaggle (quota mới từ 7 giờ 21/11) chạy lặp độc lập hoặc bổ sung, chốt sau |
| Q5 | Lối tắt tiếng thật một nguồn (VIVOS) | Ghi làm hạn chế và hướng phát triển (thêm tiếng thật đa nguồn, không dùng Common Voice, FOSD vì trùng RADAR); không làm trong đồ án |
| Q6 | Tách hai tác dụng của G1s (mục 12.1, hệ quả 3) | **Thêm G1s0** (`rl_config={std_floor: 1.0, beta: 0.0}`, 3 seed, cuối tầng 2): cùng G1, G3, G1s thành bảng 2 × 2 {chia σ_batch hay không} × {β 0,04 hay 0}. Bộ server 76 lần |
| Q7 | Quota Kaggle từ 21/11 (30 giờ) | Ưu tiên 1 seed mỗi bản sửa trên T4 (G1s, G1b, G1lr: 2 kernel, ~11,6 giờ) để xem kết luận có giữ trên phần cứng khác; còn quota thì G1 s2, s3 (~5,8 giờ). Chốt cụ thể sau khi có tầng 2 của server |

**Phát hiện chính của GĐ6 đến 15/11** (chi tiết ở các mục được trỏ)

| # | Phát hiện | Mục |
|---|---|---|
| 1 | GRPO (G1) ở batch 50 vừa T4 chỉ khi bật `expandable_segments` (hết bộ nhớ do phân mảnh); đỉnh 13,9 / ~14,7 GB; chậm 1,55 lần SFT | 3.1 |
| 2 | `main.py` huấn luyện fp32 dù YAML ghi fp16; `inner_step_size` đếm theo batch | 2, 4 |
| 3 | SFT làm kém ngoài miền có ý nghĩa thống kê (R 5 seed − S0: RADAR VI +4,12, p = 0,008; ItW +3,84, p = 0,011; FRR +28, p < 0,001); RawBoost khi fine-tune làm kém (A0 − R −5,83, p = 0,047); post-training quan trọng (P0 − R, p ≤ 0,024) | 10.1 |
| 4 | Độ lệch giữa seed lớn (RADAR VI 1,88; FRR ở ngưỡng 0 dao động 23,7–42,8%): mọi so sánh phải theo seed, không theo KTC bootstrap theo câu | 7.1, 10.1 |
| 5 | Hiệu chuẩn học trên dev (chỉ tiếng thật VIVOS) làm actDCF ngoài miền tệ hơn ở mọi biến thể; chỉ giúp tập cùng nguồn tiếng thật | 7.2 |
| 6 | MLAAD: VoxCPM2 chiếm gần hết lỗi (S0 15,8%, nhận 95,6% là thật); fine-tune sửa thứ hạng nhưng không sửa ngưỡng (R lọt 29–59%, G1 8%) | 7.3, 10.2 |
| 7 | G1, G2 (1 seed) nằm trong khoảng của R ở mọi thước đo ngoài miền; không thấy GRPO giữ tổng quát hoá tốt hơn | 10.2 |
| 8 | Giả thuyết cơ chế cũ (GRPO bỏ qua câu sai chắc) bị bác: G1 kéo câu sai chắc mạnh bằng SFT và lật tiếng thật sang giả nhiều hơn | 10.2 |
| 9 | Nguyên nhân: miền fine-tune bão hoà; cuối huấn luyện 66–76% batch có mọi nhãn lấy mẫu đều đúng; chuẩn hoá phần thưởng khiến GRPO tiếp tục làm sắc câu đã đúng (~30 lần SFT ở p = 0,999): G1 99,4% câu dev đúng với p > 0,999 (R 91–97%) | 11 |

**Việc tiếp theo** (session sau): người dùng thuê máy và gửi chuỗi SSH; Claude chạy `server/vast.sh` (init, check, code, data, bootstrap, tầng 0), báo tốc độ, chốt số worker, chạy tầng 1–6; sau tầng 2 chọn bản ghép ở tầng 4 (`skip` bản kia); gom kết quả (`fetch`, `collect`) và phân tích trên `results/server/`. Kiểm bài Dr. GRPO trước khi trích (đã kiểm, mục 12.1).

### 12.1. Đối chiếu G1s với bài Dr. GRPO (kiểm 15/11, trước khi trích)

Bài: Liu và cộng sự, "Understanding R1-Zero-Like Training: A Critical Perspective", COLM 2025, arXiv 2503.20783 (`resource/paper/DrGRPO-2503.20783.pdf`). Code ta: AntiDeepfake `algo/rl.py` commit `86098091`, `grpo3_loss` dòng 172: `reward_norm = (reward - reward.mean())/(reward.std() + rl_config['std_floor'])`, tính trên **cả B × 64 mẫu của batch** (đọc lại 15/11).

| Nội dung | Bài Dr. GRPO | Ở ta |
|---|---|---|
| Chuẩn hoá phần thưởng | GRPO chia cho độ lệch chuẩn phần thưởng của G câu trả lời **của cùng một câu hỏi** (mục 3.1, công thức lợi thế) | Chia cho độ lệch chuẩn của **cả batch** |
| Thiên lệch bài nêu | "Question-level difficulty bias": câu hỏi có độ lệch chuẩn thấp (quá dễ hoặc quá khó, phần thưởng gần như toàn 1 hoặc toàn 0) nhận trọng số cao hơn. Bài ghi rõ: chuẩn hoá lợi thế thường tính trên cả batch là kỹ thuật phổ biến; vấn đề là chuẩn hoá theo từng câu hỏi | Mọi câu trong batch chung một 1/σ_batch, nên không có thiên lệch giữa các câu cùng batch. Tác động là khác: 1/σ_batch nhân cả bước cập nhật và tăng dần khi mô hình bão hoà, như một learning rate tự tăng |
| Thiên lệch độ dài (chia cho \|o_i\|) | Có (câu trả lời nhiều token) | Không áp dụng (một nhãn mỗi câu) |
| Cách sửa | Dr. GRPO bỏ cả hai số hạng (mục 3.2); lợi thế còn lại bằng RLOO nhân một hệ số (phụ lục) | G1s: `std_floor` 1,0. Phần thưởng 0/1 có σ ≤ 0,5, nên mẫu số σ_batch + 1 nằm trong [1; 1,5]: gần như bỏ chia |
| Bằng chứng | Phụ lục C, hình 8 (Qwen2.5-1.5B, bài toán): bỏ riêng độ lệch chuẩn hoặc riêng độ dài đều cho phần thưởng huấn luyện và độ chính xác cao hơn GRPO gốc. Bài không đo quên hay tổng quát hoá ngoài miền | Kiểm ở tầng 2 của bộ server |

**Cách trích trong báo cáo:** "G1s dùng cùng phép sửa với Dr. GRPO (bỏ chia cho độ lệch chuẩn), nhưng vì một lý do khác: code AntiDeepfake chuẩn hoá theo batch, nên phép chia không tạo thiên lệch độ khó giữa các câu như bài phân tích, mà phóng đại bước cập nhật khi miền fine-tune đã bão hoà." Không viết rằng Dr. GRPO chứng minh việc bỏ chuẩn hoá giúp giữ tổng quát hoá.

**Kiểm lực kéo bằng mô phỏng** (cùng cách `src/grpo_gradient_sim.py`: 10 câu × 64 mẫu, 4.000 lượt, bỏ KL và clip; khớp công thức p(1 − p)/(σ_batch + c) trong 5%). Lực kéo lên log-odds nhãn đúng, tính theo số lần lực của SFT (1 − p):

| Batch | σ_batch | Câu | G1 (c = 1e-5) | G1s (c = 1) |
|---|---|---|---|---|
| Hỗn hợp (p từ 0,02 đến 0,999) | 0,48 | sai chắc, p = 0,02 | 0,04 | 0,01 |
| | | phân vân, p = 0,5 | 1,04 | 0,34 |
| | | đúng chắc, p = 0,999 | 1,9 | 0,7 |
| Bão hoà (p từ 0,9 đến 0,9999) | 0,11 | p = 0,9 | 7,9 | 0,80 |
| | | đúng chắc, p = 0,999 | 9,2 | 0,90 |

Ba hệ quả cho việc diễn giải tầng 2:
1. Ở batch bão hoà, G1s kéo câu đúng chắc gần bằng SFT (0,9 lần), G1 kéo gấp ~9 lần: G1s bỏ được đúng hiệu ứng "tiếp tục làm sắc câu đã đúng" (mục 11, vấn đề 1).
2. G1s kéo nhẹ hơn SFT ở mọi câu (0,34 lần ở câu phân vân), nên một phần khác biệt có thể chỉ do bước cập nhật nhỏ hơn: so thêm với G1lr (lr 1e-7).
3. Biên độ lợi thế (căn trung bình bình phương) của G1 luôn ≈ 1; của G1s là 0,32 (batch hỗn hợp) và 0,10 (batch bão hoà), còn số hạng KL không đổi. Vậy khi bão hoà, phạt KL β = 0,04 của G1s nặng tương đối gấp ~10 lần so với G1 (tương đương β ≈ 0,4 nếu giữ biên độ lợi thế của G1): **G1s cũng mạnh lên về phía "kéo về mô hình tham chiếu", giống G1b**. Khác G1b ở chỗ G1s không phóng đại bước cập nhật. Đây là phân tích, chưa có số đo.

## 13. Đánh giá toàn diện hướng thực nghiệm (15/11 tối, người dùng yêu cầu)

Căn cứ: plan (câu hỏi 1–3); bài GRPO (Bảng 1, mục 3.1, 3.5, phụ lục A.1); code AntiDeepfake `86098091` (`main.py`, `algo/rl.py`, `models/W2V.py`, `dataio/dataio.py`, `utils.py`, `hparams/ft_grpo/mms_300m_pre-post-sft.yaml`); SpeechBrain 1.0.2 (`dataio/sampler.py`, `dataio/dataloader.py`, `core.py`); số liệu mục 7, 10, 11; mô phỏng Monte Carlo (mục 12.1 và dưới đây).

**Kết luận**
1. Câu hỏi 1 và 2: hướng đúng, số liệu vững; bộ server 3 seed sẽ củng cố. Giữ nguyên.
2. Câu hỏi 3 ("chẩn đoán và sửa"): đúng hướng, nhưng thiết kế hiện tại có ba lỗ hổng khiến kết luận dễ rơi vào "không phân biệt được": (a) độ mạnh thống kê thấp hơn cỡ hiệu ứng có thể có; (b) thiếu đối chứng để quy một cải thiện (nếu có) cho GRPO, thay vì cho "fine-tune nhẹ tay hơn" hay "có phạt KL" (chính bài GRPO ghi chưa làm phần này); (c) chưa có phương án cải tiến nào có khả năng cao thật sự giảm mức quên, nên phần "đề xuất cải tiến" sẽ trống nếu mọi bản sửa GRPO ≈ R.
3. Cả ba sửa được mà không đổi công thức chính: một cách phân tích qua seed khác (không tốn GPU), hai đối chứng (6 lần chạy), một thành phần không cần huấn luyện (WiSE-FT), một phép đo (độ trôi trọng số). Một thí nghiệm lớn hơn (kiểm chế độ trên RADAR VI) để người dùng cân nhắc. Mục 13.5.
4. Trong lúc kiểm đã thấy và sửa một lỗi của bộ server (`bc94828`), và thêm một phép kiểm vào tầng 0 (`c62b211`). Mục 13.3.

### 13.1. Phần vững, giữ nguyên

| Thứ | Căn cứ |
|---|---|
| Pipeline, công thức (lịch E1', batch 50, GRPO với `expandable_segments`) | Toy 6,11%; 12 lần Kaggle ổn định; G1 đạt 4 tiêu chí (mục 10.2) |
| Câu hỏi 2: SFT làm kém ngoài miền | R − S0 (5 seed): RADAR VI +4,12 (p = 0,008), ItW +3,84 (p = 0,011), FRR +28,1 (p < 0,001) |
| Cơ chế câu hỏi 2: tiếng thật một nguồn | MLAAD (tiếng thật VIVOS) 5,18 → ~0,2%; FRR RADAR VI 4,8 → ~33%; Δz tiếng thật "S0 đúng chắc" âm ở mọi tập ngoài miền (mục 7.4) |
| Câu hỏi 1: post-training, RawBoost, kích thước | P0 − R p ≤ 0,024; A0 − R −5,83 (p = 0,047, A0 1 seed); B1 kém B2 13,6 điểm (1 seed) |
| Lịch E1' cho cả bộ | Tổng lr cộng dồn 3,64 × 10⁻³, cùng cỡ bài (2,0–3,6 × 10⁻³); đổi lịch thì mất so sánh với 12 lần Kaggle. Điểm "nhẹ tay" có G1lr (và Rlr đề xuất) |
| Hướng "chẩn đoán và sửa" | Chẩn đoán bão hoà có cơ sở định lượng (mục 11, 12.1) và tiền lệ trong bài: hàng 9 ("GRPO w/o neg.", XLS-R-2B), về kỳ vọng gradient là dạng p(1 − p) cộng KL như G1s, đã gỡ lại phần lớn mức quên ở chế độ khó (ngoài miền 3,44% so với SFT 6,28%, GRPO 2,69%) |

### 13.2. Chế độ của bài GRPO so với của ta

| | Bài (DFE24, MMS-300M) | Ta (SEA-Spoof vi, MMS-300M) | Hệ quả |
|---|---|---|---|
| Độ khó dữ liệu fine-tune với S0 | EER trong miền 29,23% | dev 1,16% | Ta ở chế độ bão hoà (mục 11) |
| Tiếng thật | Mạng xã hội, đa dạng | Chỉ VIVOS | Lối tắt nguồn tiếng thật (Q5) |
| lr mỗi bước | ≤ ~2–3 × 10⁻⁷ (warmup 80.000 không tới đỉnh, GĐ4 phát hiện 9) | ~1e-6 (warmup 460) | Mỗi bước mạnh hơn 3–5 lần |
| lr cộng dồn | 2,0–3,6 × 10⁻³ (ước) | 3,64 × 10⁻³ | Tương đương |
| Mức quên của SFT trên ItW | +11,3 điểm | +3,8 điểm | Khoảng để GRPO gỡ lại nhỏ |
| β = 1 (hàng 8, XLS-R-2B) | Ngoài miền tốt nhất (2,04) nhưng trong miền kém (15,83 so với 10,26: thiếu khớp) | Trong miền đã bão hoà, nên cái giá "thiếu khớp" nhỏ | G1b có thể là cách sửa rẻ nhất |
| Phần bài tự ghi chưa làm | "SFT may also be enhanced with proper regularization, but it is not the focus of our current work" (mục 3.5) | | Đối chứng SFT + phạt KL còn thiếu ở cả bài lẫn bộ của ta |

### 13.3. Vấn đề thiết kế và điều kiểm được trong code

**1. Độ mạnh thống kê không đủ cho cỡ hiệu ứng có thể có.** Độ lệch seed của R (5 seed): RADAR VI 1,88; ItW 1,93; RADAR EN 1,88; FRR 7,10. Chênh lệch nhỏ nhất phát hiện được với độ mạnh 80% (t-test hai mẫu, α = 0,05), điểm %:

| Thước đo | 3 v 3 | 3 v 5 | 5 v 5 |
|---|---|---|---|
| RADAR VI, ItW, RADAR EN (EER) | 5,8–5,9 | 4,6–4,7 | 3,8–3,9 |
| FRR RADAR VI ở ngưỡng 0 | 21,8 | 17,5 | 14,4 |

Toàn bộ phần SFT làm hỏng chỉ ~4 điểm (R − S0). Nếu một bản sửa gỡ lại đủ 4 điểm, so 3 seed với 3 seed chỉ phát hiện được 51% số lần (5 v 5: 84%). FRR không dùng làm phép thử được, chỉ để mô tả.

**2. "Seed" đổi những gì (đọc code).**
- `dynamic_batch_sampler_train` trong YAML không truyền `seed`, nên `DynamicBatchSampler` dùng mặc định 42 (xáo theo `seed + epoch`). Mọi lần chạy, kể cả 12 lần Kaggle, cùng một thứ tự batch.
- `main.py` gọi `load_hyperpyyaml` (dựng `detector`, `ref_detector`, `old_detector`) **trước** `set_random_seed`; SpeechBrain không đặt `worker_init_fn`, nên worker đọc dữ liệu lấy seed từ RNG CPU lúc tạo iterator. Vậy cùng seed và cùng số worker thì SFT và mọi biến thể GRPO có RawBoost nhiều khả năng nhận **cùng phép tăng cường cho từng câu**; chỉ khác dropout và phép lấy mẫu nhãn của GRPO (RNG GPU). Thử với SpeechBrain 1.0.2 (DataLoader 2 worker, kiểu fork như Linux): dựng thêm 2 mô hình trước khi đặt seed không đổi dấu vân tay batch.
- Số liệu Kaggle khớp (một quan sát): RADAR VI seed 1 của R, G1, G2 là 23,07 / 23,53 / 23,13, trong khi R qua 5 seed trải 22,32–26,63.
- Hệ quả: (a) phần lớn độ lệch seed có thể **chung** giữa các cấu hình cùng đường dữ liệu, nên phân tích có khối seed tăng độ mạnh nhiều (bảng dưới); (b) tầng 0 kiểm trực tiếp: `probe_R_w2` và `probe_G1_w2` (cùng seed 1, 2 worker) phải trùng dấu vân tay batch (`BATCH_HASH_N`, `server/probe_report.py`; `c62b211`); (c) kết luận chỉ đúng với một thứ tự batch: ghi làm hạn chế.

Chênh lệch nhỏ nhất phát hiện được (80%), RADAR VI, khi phân tích bằng mô hình hai yếu tố (cấu hình + khối seed), gộp phương sai dư của các cấu hình cùng đường dữ liệu; ρ = tương quan của hiệu ứng seed giữa hai cấu hình (chưa biết, đo được sau tầng 1):

| Thiết kế | ρ = 0 | ρ = 0,5 | ρ = 0,75 | ρ = 0,9 |
|---|---|---|---|---|
| Welch từng cặp, 3 v 3 (cách hiện tại) | 5,8 | | | |
| Khối seed, 8–12 cấu hình × 3 seed | 4,5–4,6 | 3,2–3,3 | 2,3 | 1,4–1,5 |
| Khối seed, 8–12 cấu hình × 5 seed | 3,4–3,5 | 2,4 | 1,7 | 1,1 |
| Ghép cặp từng đôi, 3 seed (bậc tự do 2) | | 6,1 | 4,3 | 2,7 |

Kể cả khi ρ = 0, gộp phương sai vẫn giảm 5,8 → 4,5 nhờ thêm bậc tự do (giả định độ lệch seed như nhau giữa các cấu hình; kiểm được và báo kèm). Ghép cặp từng đôi với 3 seed thì không đáng (bậc tự do 2).

**3. Phạt KL trong code kéo về đâu.** Số hạng KL của `grpo3_loss`, `grpo2_loss` là ước lượng k3 trên nhãn lấy mẫu từ mô hình hiện tại: `exp(r) − r − 1`, r = log π_ref(ỹ) − log π_θ(ỹ). Kỳ vọng gradient của nó theo log-odds nhãn đúng là **β(p − p_ref)**, tức gradient của KL(π_ref ‖ π_θ) (kiểm Monte Carlo 2 triệu mẫu, khớp trong 2%). Lực này không tắt khi mô hình bão hoà (khác KL(π_θ ‖ π_ref), có gradient p(1 − p)(z − z_ref) → 0). Mọi biến thể có KL vì vậy có một điểm cân bằng cho câu đúng chắc: lực lợi thế (1 − p)/s cân với β(1 − p_ref), cho Δz* ≈ ln(1/(β·s)), với s = σ_batch (G1, G1b), σ_batch + 1 (G1s), 1 (SFT + KL):

| Cấu hình | β | s (batch bão hoà, σ_batch ≈ 0,11) | Δz* |
|---|---|---|---|
| G1 | 0,04 | 0,11 | 5,4 |
| G1b | 1 | 0,11 | 2,2 |
| G1s | 0,04 | 1,11 | 3,1 |
| SFT + KL (đề xuất, mục 13.5) | 0,04 | 1 | 3,2 |
| R, G3, G1s0 | 0 | | Không có: tiếp tục làm sắc, chậm dần |

G1s và SFT + KL có cùng điểm cân bằng nhưng khác lực ở câu phân vân: SFT kéo 0,5, còn G1s ≤ 0,25. So hai cấu hình này tách được "trọng số kiểu RL" khỏi "phạt KL". Quan sát sau 4.563 bước: Δz trung vị nhóm giả "S0 đúng chắc" trên dev là R 1,3 / G1 1,7 / G2 2,0, chưa tới điểm cân bằng, nên tốc độ (G1 kéo ~9 lần SFT ở batch bão hoà) quan trọng không kém điểm cân bằng.

**4. Đối chứng còn thiếu.** (a) G1lr không có SFT cùng lr, nên nếu G1lr tốt hơn G1 thì chưa biết đó là tính chất riêng của GRPO hay chỉ do fine-tune nhẹ tay. (b) Không có SFT + phạt KL (khoảng trống bài tự nêu, mục 13.2): nếu G1b tốt hơn R, chưa biết do GRPO hay do KL.

**5. Chưa có phương án cải tiến chắc có tác dụng.** WiSE-FT (Wortsman và cộng sự, "Robust fine-tuning of zero-shot models", CVPR 2022, arXiv 2109.01903; `resource/paper/WiSE-FT-2109.01903.pdf`): θ_α = (1 − α)·θ_zero-shot + α·θ_fine-tuned. Trên ImageNet và 5 dịch chuyển phân bố, cách này tăng độ chính xác dưới dịch chuyển 4–6 điểm phần trăm so với các cách trước, đồng thời tăng 1,6 điểm trên ImageNet, không tốn thêm tính toán (tóm tắt của bài). Vì sao hợp với ta: nhắm đúng cơ chế đo được (điểm của tiếng thật nguồn khác bị trôi); S0 hiệu chuẩn tốt ngoài miền (actDCF − minDCF 0,007 trên RADAR VI); mọi checkpoint fine-tune cùng gốc S0 nên nội suy hợp lệ; áp được cho cả SFT và GRPO. Bài chỉ đo trên ảnh (CLIP): kiểm trên phát hiện giọng giả là phần đóng góp. Rủi ro: có thể chỉ cho các điểm nằm giữa S0 và R (một đường đánh đổi), không trội cả hai; vẫn là kết quả báo được.

**6. Kỹ thuật.**
- **Lỗi đã sửa (`bc94828`):** lần chấm S0 (tầng 1) xoá `cache/<mô hình>.ckpt` sau tập cuối, trong khi R, G1, A0 ở GPU kia khởi tạo từ chính file đó: `main.py` báo "Not find" rồi dừng, hoặc hai tiến trình cùng tải lại một file. Sửa: `score_zeroshot.py --keep-ckpt` (server dùng; Kaggle giữ hành vi cũ), khoá file giữa các tiến trình khi tải (thử: 3 tiến trình cùng xin, chỉ tải 1 lần).
- **TF32:** SpeechBrain 1.0.2 tự bật quirk `allow_tf32` khi import (thấy khi import trên máy), tức bật TF32 cho cả matmul lẫn convolution trên GPU Ampere. `NVIDIA_TF32_OVERRIDE=0` của bộ server ghi đè ở mức cuBLAS/cuDNN, nên `NVIDIA_TF32_OVERRIDE=0` là bắt buộc, không chỉ phòng xa. Kiểm bằng số trên server sau bootstrap (sai số matmul fp32 so với fp64).
- **Thứ tự dọn máy:** checkpoint chỉ có trên server; mọi phép đo cần checkpoint (WiSE-FT, độ trôi trọng số) phải xong trước `cleanup`.

### 13.4. Dự đoán ghi trước (kiểm khi có tầng 2)

Hai thước đo kiểm cơ chế, độ lệch seed rất nhỏ: (i) Δz trung bình của tiếng thật dev nhóm "S0 đúng chắc": R 0,54 ± 0,02 (5 seed), G1 0,90, G2 1,07 (1 seed), tức G1 cách R ~18 độ lệch chuẩn, 1 seed đủ phân biệt; (ii) % câu dev đúng với p > 0,999: R 91–97, G1 99,4, G2 99,7.

| Cấu hình | (i), (ii) so với R | Ngoài miền (RADAR VI, ItW, FRR) | Căn cứ |
|---|---|---|---|
| G3 (β = 0) | ≥ G1 | ≈ G1 | Ở β = 0,04 lực KL đã nhỏ so với lợi thế phóng đại |
| G2 | ≈ G1 | ≈ G1 | Đã thấy ở seed 1 |
| G1b (β = 1) | < G1, có thể < R | Tốt hơn G1 | Δz* 2,2 |
| G1s | ≤ R | Tốt hơn G1; so với R chưa chắc | Không phóng đại; Δz* 3,1; kéo câu phân vân yếu hơn SFT |
| G1s0 | ≈ R | ≈ G1s hoặc kém hơn chút | ≈ SFT có trọng số p, không KL |
| G1lr | < R | Tốt hơn G1, gần S0 hơn | Bước nhỏ 10 lần (vẫn phóng đại 3,6–10 lần) |
| Rlr (đề xuất) | < R | Tốt hơn R | Bước nhỏ 10 lần |
| SFT + KL (đề xuất) | < R, ≈ G1s | ≈ G1s | Cùng Δz* |

Dự đoán nào sai thì phân tích lực (mục 12.1, 13.3) thiếu một yếu tố; báo đúng như vậy.

### 13.5. Đề xuất (xếp theo giá trị trên chi phí)

| # | Đề xuất | Loại | Trả lời gì | Chi phí | Khuyến nghị |
|---|---|---|---|---|---|
| A1 | Phép thử chính qua seed: mô hình hai yếu tố (cấu hình + khối seed, gộp phương sai dư của các cấu hình cùng đường dữ liệu); Welch từng cặp làm kiểm tra độ vững. Ghi trước 3 so sánh chính (G1 − R; bản sửa tốt nhất − R; bản sửa tốt nhất − G1) trên RADAR VI và ItW, hiệu chỉnh Holm; thước đo cơ chế (13.4); FRR chỉ mô tả; báo ngưỡng phát hiện cạnh mọi kết luận "không khác biệt" | Phân tích | Ngưỡng phát hiện RADAR VI từ 5,8 xuống 4,5 (ρ = 0) đến 2,3 (ρ = 0,75) | Không tốn GPU | Làm; dạng cuối theo kết quả dấu vân tay ở tầng 0 |
| A2 | Độ trôi trọng số ‖θ − θ_S0‖ theo nhóm lớp (CNN, 24 lớp transformer, lớp FC) cho mọi checkpoint; đặt cạnh mức quên qua mọi lần chạy và qua α của WiSE-FT | Phân tích | Quên do đi xa bao nhiêu hay do đi theo cách nào. Nếu mức quên chỉ theo độ trôi, bất kể SFT hay GRPO, thì GRPO chỉ là một núm cường độ | CPU, ~1 phút mỗi checkpoint, chạy trước `cleanup` | Làm |
| B1 | Rlr: SFT với lr 1e-7 × 3 seed | Tham số | Cặp cùng cường độ với G1lr | 3 lần SFT | Làm, tầng 2 |
| B2 | SFT + KL: cross-entropy như R cộng đúng số hạng KL k3 của GRPO, β = 0,04, × 3 seed | Công thức | Khoảng trống bài tự nêu; so với G1s tách "trọng số kiểu RL" khỏi KL; so với R cho tác dụng của KL lên SFT | 3 lần (tốc độ như G2); một lớp bọc loss nhỏ (runpy như `mem_probe.py`, không sửa code gốc) | Làm, tầng 2 |
| C1 | WiSE-FT: α ∈ {0,25; 0,5; 0,75} cho R, A0, G1 và bản sửa tốt nhất (α = 0,5 cả 3 seed; 0,25 và 0,75 ở seed 1) | Thành phần | Phương án cải tiến không cần huấn luyện; đường đánh đổi giữa trong miền (MLAAD, `test_cvreal`) và ngoài miền | ~20 lần chấm | Làm, sau tầng 2 |
| C2 | Seed 4, 5 cho bản sửa tốt nhất; đưa R, G1 seed 4, 5 lên ngay sau tầng 2 | Thiết kế | 5 v 5 cho các so sánh chính | +2 lần GRPO; phần còn lại chỉ đổi thứ tự | Làm |
| D1 | Kiểm chế độ: fine-tune trên nửa RADAR VI (chia ngẫu nhiên theo nhãn: không có nhãn người nói; S0 EER 20%; tiếng thật đa nguồn), R và G1 × 3 seed; chấm nửa còn lại (trong miền), ItW, RADAR EN, SEA-Spoof, MLAAD | Thiết kế | Trả lời trực tiếp "vì sao": bỏ bão hoà mà giữ ngôn ngữ, mô hình, code. G1 hơn R ở đây mà không hơn trên SEA-Spoof thì chẩn đoán được ủng hộ | 6 lần; chuẩn bị danh sách câu, CSV, kiểm dữ liệu | Cân nhắc: làm nếu tầng 0 cho thấy còn dư ≥ 1 ngày |
| D2 | Ít mẫu: thêm bản sửa tốt nhất cạnh SFT và G1 | Thiết kế | Ít mẫu là bão hoà nặng nhất: chẩn đoán dự đoán G1 hỏng nhiều hơn SFT khi N nhỏ, bản sửa thì không | +9 lần GRPO | Chốt sau tầng 2 |
| D3 | Nếu độ lệch seed của A0 nhỏ hơn hẳn R: thêm G1A (G1 không RawBoost), giữ cả G1sA và G1bA | Thiết kế | Bảng 2 × 2 {SFT, GRPO} × {có, không RawBoost} ở nhóm ít nhiễu hơn | +3 đến +6 lần | Chốt sau tầng 1 |

### 13.6. Đã xét, không đề xuất

| Ý | Vì sao không |
|---|---|
| Đổi lịch lr cho cả bộ server (giảm về ~0 ở cuối, E1 đúng nghĩa) | Mất so sánh với 12 lần Kaggle; có thể giảm độ lệch seed nhưng cũng đổi cỡ hiệu ứng. Rlr, G1lr đã cho điểm "nhẹ tay" |
| Chuẩn hoá phần thưởng theo từng câu như công thức (4) của bài | Suy ra được: ở chế độ bão hoà còn phóng đại hơn chuẩn hoá theo batch (√(p(1 − p)): ~30 lần SFT ở p = 0,999); thêm thí nghiệm ít thông tin |
| Đổi seed của sampler theo seed huấn luyện | Mất thứ tự batch chung giữa các cấu hình (đang có lợi cho so sánh), tăng phương sai |
| FRR theo nguồn tiếng thật của RADAR VI | Danh sách câu không có nhãn nguồn hay người nói (`Speaker` toàn "-") |
| Độ trôi embedding như hình 2 của bài | Cần ASVspoof 2019 dev và embedding; A2 và Δz đo trực tiếp hơn |
| Thêm tiếng thật đa nguồn vào dữ liệu fine-tune | Đã chốt Q5 (hạn chế, hướng phát triển); D1 có tiếng thật đa nguồn nhưng chỉ là phép kiểm chế độ |
| Label smoothing, LoRA, đóng băng lớp | Ngoài câu hỏi SFT so với GRPO; ghi hướng phát triển. A2 cho biết lớp nào trôi nhiều nhất, làm căn cứ |

### 13.7. Chi phí và lịch

| Gói | Giờ GPU tính theo T4 | 2 × 3090 (nhanh 2,5–3,5 lần T4) |
|---|---|---|
| Bộ hiện tại (76 lần, bỏ một bản ghép tầng 4) | ~330 | 47–66 giờ |
| B1, B2, C1, C2 | ~70 | +10–14 giờ |
| D1 | ~30 | +4–6 giờ |
| D2 | ~50 | +7–10 giờ |
| Tất cả (trừ D3) | ~480 | 2,8–4 ngày; 4 GPU: 1,4–2 ngày |

vast.ai tính tiền theo giờ GPU, nên 4 GPU tốn gần bằng 2 GPU nhưng xong sớm gấp đôi. Thí nghiệm nên xong trước ~25/11 để còn GĐ7.

**Cần người dùng chốt:** B1, B2, C1, C2 (đề xuất làm); D1 (làm, hay để hướng phát triển); 2 hay 4 GPU. A1, A2 không đổi thí nghiệm (A1 đổi phép thử chính, Welch vẫn báo kèm). D2, D3 chốt sau tầng 1–2.
