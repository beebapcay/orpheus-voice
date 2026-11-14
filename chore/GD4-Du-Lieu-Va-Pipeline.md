# GĐ4. Dữ liệu tiếng Việt và pipeline

Câu hỏi của giai đoạn: fine-tune trên tiếng Việt như thế nào cho đúng (dữ liệu nào, chia tập ra sao, pipeline có chạy và vừa quota không). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ4. Đầu vào từ GĐ3: `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 8. File này chuẩn bị ngày 14/11, trước khi bắt đầu.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 0 | Việc người dùng làm trước (mục 1): quyền SEA-Spoof, MLAAD, đăng nhập Hugging Face | | **SEA-Spoof: có quyền** (14/11, đã kiểm). MLAAD: chưa |
| 1 | Lấy metadata SEA-Spoof, thống kê phần tiếng Việt, chọn và chia tập, tải âm thanh | `protocols/ids/seaspoof_vi_*.tsv`, CSV theo định dạng AntiDeepfake | Chưa |
| 2 | Pipeline fine-tune: bộ nhớ, warmup, overfit một batch, thời gian một epoch | Kernel Kaggle fine-tune, `results/runs/...` | Chưa |
| 2 | Ngân sách quota GĐ5–6 | Mục 5 | Chưa |

**Lịch:** hôm nay 14/11, hạn nộp 30/11 (còn 16 ngày). Plan cần 14 ngày cho GĐ4–8 và dự phòng, còn dư khoảng 2 ngày. Quota Kaggle 30 giờ/tuần, làm mới 7 giờ sáng thứ Bảy: tuần 14–20/11, tuần 21–27/11, rồi 28–30/11.

## 0. Đầu vào từ GĐ3

| Thứ | Giá trị | Dùng ở |
|---|---|---|
| Cấu hình mốc R | MMS-300M **bản Default** | GĐ5 |
| Cấu hình thêm N0 | MMS-300M bản NDA, 1 seed, không bắt buộc | GĐ5 |
| S0 (dùng thẳng) trên RADAR tiếng Việt | MMS-300M 20,08%; W2V-Large 17,80%; W2V-Small 27,21% (bản NDA: 18,07 / 16,62 / 23,24) | Mốc của R, B2, B1, N0 |
| S0 trên `ood_forget` (In-the-Wild / RADAR EN) | MMS-300M 2,6 / 26,2; W2V-Large 2,0 / 23,5; W2V-Small 4,0 / 27,2 | Đo mức quên |
| Dòng tham chiếu | XLS-R-2B dùng thẳng: 13,71% trên RADAR tiếng Việt | Mọi bảng GĐ5–6 |
| Tập câu khó chung | `results/zeroshot/analysis/hard_ids_default7.csv` (493 câu) | Theo dõi trước và sau fine-tune |
| Chấm đã đo (một T4, bản ≤ 317M) | `radar_vi` 24–38 phút; `ood_forget` 7–11 phút | Ngân sách (mục 5) |
| Máy Kaggle | 2 × T4 15 GB, **4 nhân CPU**, 31 GB RAM, phiên 12 giờ | Đo tốc độ đọc dữ liệu |
| Môi trường | `src/setup_antideepfake.sh` dựng trong ~4 phút trên Kaggle | Kernel fine-tune |
| Báo thêm ngoài EER | Tỉ lệ giả bị nhận là thật và thật bị coi là giả ở ngưỡng 0; actDCF − minDCF; khoảng tin cậy (`src/bootstrap_eer.py`) | GĐ5–6 |

## 1. Việc người dùng cần làm trước

1. ~~**Kiểm tra quyền SEA-Spoof.**~~ **Xong 14/11:** tác giả đã duyệt. Đã kiểm trên máy: `hf auth whoami` ra đúng tài khoản; `hf download` tải được `metadata/dataset_metadata.json`; đọc được đoạn cuối của một file parquet qua `HfFileSystem` (đọc theo khoảng byte, không tải cả file). Đăng nhập Hugging Face, mở [trang dữ liệu](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof). Nếu vẫn chờ duyệt: README hiện yêu cầu **gửi email cho tác giả trước** (wu_jinyang@a-star.edu.sg, imcc.sg@gmail.com), ghi họ tên, đơn vị, loại đơn vị (trường đại học), mục đích nghiên cứu, và cam kết phi thương mại, không chia sẻ lại. Claude soạn được bản nháp email nếu cần.
2. **Xin quyền MLAAD** ([trang dữ liệu](https://huggingface.co/datasets/mueller91/MLAAD), duyệt tự động) làm dự phòng. Làm ngay, kể cả khi SEA-Spoof được duyệt.
3. ~~**Đăng nhập Hugging Face trên máy.**~~ **Xong 14/11:** CLI `hf` (huggingface_hub 2.4.0) cài bằng `uv tool install --python 3.12 huggingface_hub`; người dùng đã `hf auth login`. Claude không đọc token; script dùng Python của CLI (`head -1 $(command -v hf)`) để có sẵn thông tin đăng nhập. Kết nối tới huggingface.co hiện đi được cả IPv4 lẫn IPv6.
4. (Không chặn) Xin quyền F5-TTS Vietnamese bằng email trường, cho tập tự tạo.

## 2. SEA-Spoof: những gì đã biết

Kiểm từ README công khai ngày 14/11; chưa có quyền đọc file dữ liệu.

| Thứ | Giá trị |
|---|---|
| Tổng | 553.746 câu, 81,5 GB, 7 ngôn ngữ (en, hi, id, ms, ta, th, vi) |
| Tiếng Việt | 81.020 câu (14,63%) |
| Chia sẵn | train 439.362 / validation 57.158 / evaluation 57.226 (mọi ngôn ngữ) |
| File | 37 file parquet: train 30 file (64,7 GB), validation 3, evaluation 4. **Các ngôn ngữ trộn lẫn, không tách file theo ngôn ngữ** |
| Âm thanh | FLAC 16 kHz, nhúng trong cột `audio` |
| Cột metadata | `row_id`, `utterance_id`, `text`, `language`, `label` (bonafide/spoof), `spoof_type`, `category` (bonafide, offline_spoof, online_spoof), `split`, `source_model`, `source_dataset`, `speaker_or_voice`, `sampling_rate`, `audio_was_resampled`, và các cột nguồn gốc transcript |
| Không có | Cột độ dài câu: chỉ biết sau khi đọc âm thanh (hoặc header FLAC) |
| Giấy phép | Phi thương mại, không phân phối lại: không đưa âm thanh lên git hay dataset Kaggle công khai |

**Cần kiểm khi có quyền:** 7 hệ thống sinh tiếng Việt (theo bài) có đúng là các giá trị của `source_model` không; tiếng thật có đúng từ VIVOS không (`source_dataset`); `speaker_or_voice` có đủ để chia không trùng người nói không; các split có sẵn có trùng người nói không.

### 2.1. Cách lấy phần tiếng Việt mà không tải 81,5 GB

| Lượt | Làm gì | Dung lượng |
|---|---|---|
| 1. Metadata | Đọc chỉ các cột nhẹ của 37 file parquet qua HTTP Range (như cách đọc In-the-Wild ở GĐ3, `src/prep_data.py`) | Nhỏ (ước vài chục MB; cần đo) |
| 2. Chọn câu | Thống kê phần `vi` theo split, nhãn, hệ thống sinh, người nói; chọn tập fine-tune 10–20 giờ và các tập kiểm tra; cố định danh sách câu vào `protocols/ids/` (seed 2026) | Trên máy |
| 3. Âm thanh | Chỉ tải các *row group* parquet chứa câu đã chọn | Tuỳ cách dữ liệu được xếp: nếu câu tiếng Việt nằm gọn trong vài row group thì nhỏ; nếu rải đều thì gần như phải đọc cả 81,5 GB |

Bước 3 chạy ở đâu: chọn sau khi biết kết quả lượt 1.

| | Mac | Kaggle (kernel CPU, như `orpheus-prep`) |
|---|---|---|
| Đăng nhập HF | Có sẵn sau `hf auth login` | Cần token trong Kaggle Secrets; **chưa kiểm** kernel đẩy bằng CLI có đọc được secret không |
| Băng thông | Mạng nhà | Nhanh (tải HF ~95 MB/s ở GĐ3) |
| Lưu kết quả | Ổ máy, rồi phải đưa lên Kaggle cho kernel GPU | Output kernel dùng thẳng cho kernel fine-tune |
| Giấy phép | Dataset Kaggle phải để riêng tư | Output kernel riêng tư |

## 3. Chia tập (nháp, chốt sau lượt 1)

| Tập | Nguồn | Yêu cầu |
|---|---|---|
| Train | SEA-Spoof `vi`, 10–20 giờ | Không trùng người nói với dev, test |
| Dev | SEA-Spoof `vi` | Chọn checkpoint tốt nhất; học hệ số hiệu chuẩn ở GĐ6 |
| Test trong miền | SEA-Spoof `vi` | Dễ bão hòa gần 0%, không dùng làm thước đo chính |
| Hệ thống sinh để riêng | Một (hoặc vài) `source_model` bỏ khỏi train | Thước đo chính. **Cần chốt:** để riêng một hệ thống cố định cho mọi cấu hình, hay luân phiên (tốn quota hơn) |
| RADAR tiếng Việt, `ood_forget`, tập câu khó | Đã có từ GĐ3 | Giữ nguyên danh sách câu |
| Tập tự tạo | Thành viên đọc và F5-TTS nhái giọng | Không chặn GĐ5 |

Yếu tố gây nhiễu (từ plan): tiếng thật của SEA-Spoof từ VIVOS, của RADAR từ Common Voice và FOSD. Không thêm Common Voice vào train. Theo dõi tỉ lệ tiếng thật bị coi là giả trên RADAR trước và sau fine-tune (S0 của MMS-300M: 4,8% ở ngưỡng 0).

Nếu dùng phương án dự phòng (MLAAD `fake/vi` + VIVOS): tiếng thật và tiếng giả từ hai kho khác nhau, phải kiểm tra lối tắt (ví dụ độ dài, mức âm lượng, khoảng lặng) trước khi huấn luyện.

## 4. Pipeline (ngày 2)

Xuất phát từ `hparams/ft_grpo/mms_300m_pre-post-sft.yaml` của AntiDeepfake (khởi tạo từ checkpoint post-trained; RawBoost `algo: 5` khi fine-tune).

- [ ] Kernel Kaggle fine-tune: dùng lại cách dựng môi trường và lưu bằng chứng của GĐ3 (`results/runs/<kernel>-v<N>/`); mở rộng `push.sh output` cho kernel mới.
- [ ] Lưu checkpoint sau mỗi epoch vào `/kaggle/working` để chạy tiếp được qua nhiều phiên; ghi tên, dung lượng, nơi lưu checkpoint vào `RUN.md`.
- [ ] Đo bộ nhớ GPU với MMS-300M. Nếu vượt 15 GB, giảm `max_batch_length` (mặc định 100 giây âm thanh mỗi batch).
- [ ] Chỉnh `warmup` theo số bước thật: cấu hình gốc `warmup: 80000` nhưng `optimizer_step_limit: 50000`, nên với tập nhỏ learning rate gần như không tăng tới giá trị đặt.
- [ ] Overfit một batch nhỏ để xác nhận pipeline học được.
- [ ] Đo thời gian một epoch và **tốc độ đọc dữ liệu** (4 nhân CPU cho 2 GPU; chỉnh `num_workers`).
- [ ] Thử chạy hai cấu hình song song trên hai GPU trong một phiên.
- [ ] Chấm sau fine-tune bằng `src/score_zeroshot.py` (đã có, chia shard, chạy tiếp được) trên dev, test, hệ thống để riêng, RADAR tiếng Việt, `ood_forget`.

## 5. Ngân sách quota (nháp, điền số sau ngày 2)

`Giờ quota = số phiên × thời gian phiên`, mỗi phiên chạy được 2 cấu hình song song (nếu ngày 2 xác nhận được).

| Giai đoạn | Lần huấn luyện | Ghi chú |
|---|---|---|
| GĐ5 | R × 3 seed, P0, A0, B2, B1 = 7; N0 = 1 (không bắt buộc) | |
| GĐ6 | G1 × 3 seed, G2, G3 = 5; ít mẫu (SFT và G1, N = 50, 200, 1000) = 6 lần ngắn | |
| Tổng | ~19 lần | |

| Thành phần cho mỗi lần | Đã biết | Chưa biết |
|---|---|---|
| Huấn luyện | | Thời gian một epoch, số epoch |
| Chấm RADAR tiếng Việt + `ood_forget` | ~30–50 phút trên một T4 (bản 317M) | |
| Chấm dev, test, hệ thống để riêng | | Phụ thuộc kích thước tập |

Quota còn lại đến hạn nộp: khoảng 30 + 30 + 30 giờ (ba lần làm mới: 14/11, 21/11, 28/11), nhưng tuần cuối chỉ còn 2–3 ngày dùng được trước khi đóng gói. Nếu không vừa: theo thứ tự cắt ở mục 8 của plan (bỏ N0, G3, B1, ...).

## 6. Kiểm tra điều kiện qua GĐ4

- [ ] Có quyền SEA-Spoof (hoặc đã chuyển sang MLAAD + VIVOS và ghi rõ hạn chế)
- [ ] Danh sách câu train, dev, test, hệ thống để riêng cố định trong `protocols/ids/`, không trùng người nói
- [ ] Pipeline overfit được một batch; bộ nhớ vừa T4
- [ ] Đã đo thời gian một epoch và tốc độ đọc dữ liệu; bảng ngân sách mục 5 điền đủ và vừa quota
- [ ] Bằng chứng mọi lần chạy trong `results/runs/`

Điều kiện của plan: ngân sách vừa quota. Nếu MMS-300M không vừa sau khi giảm batch, chuyển cấu hình mốc sang W2V-Small. Nếu không fine-tune được, chuyển sang tầng 0.
