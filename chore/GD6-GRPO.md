# GĐ6. GRPO và cải tiến

Câu hỏi của giai đoạn: fine-tune bằng GRPO trên tiếng Việt có giữ khả năng tổng quát hoá tốt hơn SFT không, và thành phần nào của GRPO tạo ra khác biệt (câu hỏi 3 của plan). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ6 kèm "Lưu ý từ GĐ5". Đầu vào từ GĐ5: `chore/GD5-Ablation.md` mục 8 (tổng kết), mục 9 (lưu ý L1–L12), mục 6 (đối chiếu bài GRPO và code), mục 4.1 (lịch learning rate E1'). File này bắt đầu ngày 15/11.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Chuẩn bị lần thử đo GRPO: đọc code GRPO (YAML v1/v2, `main.py`, `algo/rl.py`), kiểm ghi đè lồng `rl_config`, tích luỹ gradient, mô phỏng số batch | Mục 2; `src/mem_probe.py`; `src/finetune.py` (`--probe-mem`, `--oom-retry`, `--train-longest`, `--discard-ckpt`, `--env`); `kaggle/finetune/run.py` | Xong (15/11), đã thử trên máy |
| 1 | **Lần thử đo bộ nhớ và tốc độ GRPO trên T4** (L1) | `orpheus-finetune` v5, mục 3 | Đang chạy |
| 1 | Báo kết quả lần thử; người dùng chốt: Kaggle hay thuê server, dùng quota còn lại tuần này thế nào | Mục 3, 5 | **Kaggle: đã chốt** (15/11, mục 5); quota tuần này: chờ kết quả lần thử |
| 1–2 | G1 (3 seed), G2, G3 | | Chưa |
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

Kết quả: (chờ lần chạy).

## 4. Phát hiện ảnh hưởng thí nghiệm

| # | Phát hiện | Ảnh hưởng | Việc cần làm |
|---|---|---|---|
| 1 | **Huấn luyện chạy fp32 dù YAML ghi `precision: fp16`** (`fit_batch` của `main.py` không có `autocast`; đọc code, chưa kiểm khi chạy) | Không đổi so sánh (SFT và GRPO cùng code). Giải thích bộ nhớ sát trần. Ghi vào báo cáo phần cấu hình | Không sửa code gốc |
| 2 | **`inner_step_size` đếm theo batch, không theo bước optimizer** như `README_grpo.md` viết | Chỉ khác khi tích luỹ gradient | Đặt 2000 nếu dùng batch 25 × 2 |

## 5. Ngân sách quota

| Lần | Quota dùng | Còn lại | Ghi chú |
|---|---|---|---|
| Trước GĐ6 | | 11,60 giờ đến 21/11 | Kiểm 15/11 |
| Thử đo (`orpheus-finetune` v5) | | | Ước ~0,75 giờ; đẩy 9:17, được cấp máy ngay |

**Đã chốt (15/11, người dùng):** chạy GĐ6 trên Kaggle như hiện tại, một tài khoản; không thuê server (nên không cần chạy lại R, A0). Không tạo tài khoản Kaggle thứ hai: điều khoản Kaggle cấm một người có hơn một tài khoản đang hoạt động (theo kết quả tìm kiếm trích kaggle.com/terms; trang render bằng JavaScript nên chưa đọc trực tiếp), rủi ro mất cả tài khoản chính (dataset riêng tư, output và checkpoint GĐ5). Theo ước tính trước lần thử, tổng quota đủ (GĐ6 ~17 giờ; ~31 giờ nếu thêm 2 seed cho R và G1 cùng G1 không RawBoost; có ~10,85 giờ tuần này + 30 giờ từ 21/11); chỉ là xong GPU muộn hơn (~22–23/11).

**Đánh giá plan theo số liệu GĐ5 (15/11, đề xuất, chờ người dùng chốt sau lần thử):**

| # | Vấn đề | Số liệu | Đề xuất |
|---|---|---|---|
| P1 | 3 seed có thể không đủ để kết luận G1 so với R | Ngưỡng chênh lệch có ý nghĩa (t-test hai mẫu, 3 seed mỗi bên, giả định độ lệch seed của G1 như R): RADAR VI ~3,4 điểm (s = 1,52), ItW ~3,9 điểm (s = 1,71). Phần SFT làm hỏng mà GRPO có thể gỡ lại: RADAR VI R − S0 = 3,9 điểm; ItW 5,63 − 2,60 = 3,0 điểm (bài GRPO, MMS-300M: 14,17 − 2,91 = 11,3 điểm). Trên ItW, kể cả GRPO gỡ lại hết vẫn dưới ngưỡng | Thêm 2 seed cho cả R và G1 (~10–11 giờ, tuần sau): ngưỡng còn ~2,2 điểm RADAR VI, ~2,5 điểm ItW |
| P2 | G1 so với A0 lệch hai yếu tố (hàm loss và RawBoost) | A0 − R = −5,65 điểm RADAR VI (RawBoost khi fine-tune làm kém, GĐ5 phát hiện 4) | Thêm "G1 không RawBoost" (1 seed, `use_da=False`): bảng 2 × 2 {SFT, GRPO} × {có, không RawBoost} |
| P3 | Ngày 2 (ít mẫu N = 50/200/1000) không đo được gì | SEA-Spoof bão hoà từ S0 (mọi hệ thống ≤ 2,1%) | Cắt (mục cắt thứ 4 của plan), dành quota cho P1, P2 |
