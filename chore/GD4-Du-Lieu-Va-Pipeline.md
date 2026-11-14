# GĐ4. Dữ liệu tiếng Việt và pipeline

Câu hỏi của giai đoạn: fine-tune trên tiếng Việt như thế nào cho đúng (dữ liệu nào, chia tập ra sao, pipeline có chạy và vừa quota không). Kế hoạch gốc: `chore/research-plan.md`, mục GĐ4. Đầu vào từ GĐ3: `chore/GD3-Post-Training-Va-Zero-Shot.md` mục 8. File này chuẩn bị ngày 14/11, trước khi bắt đầu.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 0 | Việc người dùng làm trước (mục 1): quyền SEA-Spoof, MLAAD, đăng nhập Hugging Face | | **SEA-Spoof: có quyền** (14/11, đã kiểm). MLAAD: chưa |
| 1 | Lượt 1: đọc metadata SEA-Spoof (38 file, chỉ cột nhẹ), thống kê phần tiếng Việt | `src/prep_seaspoof.py meta/stats`, `results/seaspoof/stats_vi.txt` | **Xong** (14/11, 7,5 phút, 38 MB); phát hiện ở mục 2.2 |
| 1 | Đọc lướt bài SEA-Spoof | `resource/paper/SEA-Spoof-2509.19865.pdf` | Xong (mục 2, 2.2) |
| 1 | Chọn và chia tập | `protocols/ids/seaspoof_vi_*.tsv`, mục 3 | **Xong** (người dùng chốt D1–D3 14/11); không trùng người nói, giọng, video |
| 1 | Tải âm thanh, tạo CSV theo định dạng AntiDeepfake, kiểm lối tắt | `Data/seaspoof/data/` (máy), `results/seaspoof/` | **Xong** (14/11, 22,6 phút); train 17,88 giờ |
| 1 | Đưa lên Kaggle | Dataset riêng tư `orpheus-seaspoof-vi` | Đang tải lên |
| 2 | Code fine-tune và kernel | `src/finetune.py`, `kaggle/finetune/`, mục 4 | Xong code; pilot chưa chạy |
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

Kiểm 14/11 trên metadata thật (revision HF `132f5dca`, ngày 21/07/2026; `src/prep_seaspoof.py`, đầu ra `results/seaspoof/stats_vi.txt`) và bài gốc ([arXiv 2509.19865v2](https://arxiv.org/abs/2509.19865), `resource/paper/SEA-Spoof-2509.19865.pdf`, 4 trang nội dung).

| Thứ | Giá trị | Nguồn |
|---|---|---|
| Tổng | 553.746 câu, 81,5 GB, 7 ngôn ngữ (en, hi, id, ms, ta, th, vi). Bài chỉ mô tả 6 ngôn ngữ Đông Nam Á; bản HF thêm tiếng Anh (130.776 câu) | Metadata; bài mục 1 |
| Tiếng Việt | 81.020 câu: **25.804 thật, 55.216 giả** (1 : 2,14) | Metadata |
| Chia sẵn (tiếng Việt) | train 60.905 / validation 10.058 / evaluation 10.057 | Metadata |
| File | **38** file parquet (README ghi validation 3 file, thực tế 4). Mỗi row group 200 câu | Footer parquet |
| Vị trí tiếng Việt | Gọn trong **411 row group** của 11 file (9 row group trộn ngôn ngữ ở ranh giới), **7,18 GB** âm thanh | Footer + metadata |
| Âm thanh | FLAC 16 kHz, mono, không resample (`audio_was_resampled` = False cho cả 81.020 câu) | Metadata |
| Tiếng thật | **VIVOS 12.420 + Common Voice 13.384** (không chỉ VIVOS như plan ghi). Common Voice lấy từ `other.tsv` (clip chưa đủ lượt duyệt) và 158 clip từ `invalidated.tsv` | Metadata (`source_dataset`, `text_source`); bài mục 2.3 |
| Tiếng giả mã nguồn mở | VITS 14.616, Edge-TTS 14.599, F5-TTS 14.600. Đọc văn bản của câu thật (VIVOS 60%, Common Voice 40%) | Metadata |
| Tiếng giả thương mại | ChatGPT-TTS 3.267, ElevenLabs 3.032 (đọc văn bản Common Voice); HeyGen 2.983, MiniMax 2.119 (đoạn cắt từ video hoặc bản ghi dài, văn bản do WhisperX nhận dạng) | Metadata |
| Giờ (ước lượng) | ~108 giờ: VIVOS ~15, Common Voice ~24, VITS ~27, Edge-TTS ~20, F5-TTS ~11, thương mại ~10,5 | Dung lượng FLAC ÷ 18,1 KB/s (hiệu chỉnh theo VIVOS ~15 giờ). Sai ±15–20%; số thật đo sau khi tải |
| Kết quả của bài trên tiếng Việt | MoLEx fine-tune: EER 0,01% trên phần thương mại tiếng Việt; trước fine-tune 48,5% (Bảng 3) | Bài mục 3.3 |
| Giấy phép | Phi thương mại, không phân phối lại: không đưa âm thanh (và transcript) lên git hay dataset Kaggle công khai | README dataset |

Đối chiếu ước lượng giờ với Bảng 1 của bài (chỉ tiếng giả, tập train tiếng Việt): thương mại 8,0 giờ, bài 8,0; mã nguồn mở 47 giờ, bài 40,0. Đủ tin để lập kế hoạch, không đủ để báo cáo.

### 2.1. Cách lấy phần tiếng Việt

| Lượt | Làm gì | Kết quả |
|---|---|---|
| 1. Metadata | `prep_seaspoof.py meta`: đọc mọi cột trừ `audio.bytes` (49 MB trên 81,5 GB) qua HTTP Range, 24 luồng; ghi vị trí từng câu (file, row group, thứ tự) | **Xong:** 7,5 phút trên máy, `Data/seaspoof/meta.parquet` 38 MB (không commit vì có transcript) |
| 2. Chọn câu | `prep_seaspoof.py select` theo cách chia đã chốt (mục 3), seed 2026 | **Xong:** `protocols/ids/seaspoof_vi_{train,dev,test}.tsv`, `seaspoof_vi_split.json` (người nói, giọng, số câu, kết quả kiểm tra) |
| 3. Âm thanh | `prep_seaspoof.py fetch`: tải 384 row group chứa câu đã chọn, kiểm `utterance_id` đúng vị trí, ghi FLAC, đo độ dài, tạo CSV định dạng AntiDeepfake | **Xong:** 22,6 phút trên máy (~5,4 MB/s), 7,38 GB đọc, giữ 31.886 file FLAC (2,5 GB) ở `Data/seaspoof/data/` (không commit) |
| 4. Kiểm lối tắt | `prep_seaspoof.py check`: giờ thật theo tập và hệ thống; EER khi chỉ dùng một đặc trưng thô | **Xong:** `results/seaspoof/check_output.txt`, `hours_by_set.csv`, `shortcut_eer.csv` (mục 3.1) |
| 5. Lên Kaggle | `push.sh data` -> dataset **riêng tư** `orpheus-seaspoof-vi` (FLAC + CSV) | Đang tải lên |

Lượt 3 chạy trên **máy** (đăng nhập HF có sẵn; kernel Kaggle cần token HF trong Kaggle Secrets, chưa kiểm được với kernel đẩy bằng CLI).

### 2.2. Phát hiện ảnh hưởng thí nghiệm

| # | Phát hiện | Ảnh hưởng | Việc cần làm |
|---|---|---|---|
| 1 | **Split có sẵn trùng người nói hoàn toàn:** cả 65 người nói VIVOS đều có ở train, validation và evaluation; tiếng thật mỗi split chỉ là chia ngẫu nhiên theo câu. Ngoài ra 3.775 văn bản vừa ở train vừa ở evaluation (câu thật ở split này, câu giả cùng nội dung ở split kia) | Không dùng được split có sẵn làm tập kiểm tra. EER 0,01% của bài (Bảng 3) đo trên split này nên có thể lạc quan | Chia lại theo người nói (mục 3). Ghi vào báo cáo khi so với số của bài |
| 2 | **Tiếng thật có cả Common Voice (13.384 câu), không chỉ VIVOS.** RADAR tiếng Việt cũng lấy 3.000 câu thật từ "Common Voice Scripted Speech" (Bảng II bài RADAR, không ghi phiên bản). Common Voice trong SEA-Spoof không có nhãn người nói, và 5.394 văn bản cho 13.384 clip (một câu được nhiều người đọc) | Nếu đưa Common Voice vào train: (a) có thể trùng chính các clip thật của RADAR tiếng Việt, làm đẹp kết quả RADAR; (b) không bảo đảm được không trùng người nói. Yếu tố gây nhiễu của plan ("SEA-Spoof thật từ VIVOS, RADAR từ Common Voice") phải viết lại | Người dùng chốt (mục 3, D1). Nếu dùng Common Voice ở bất kỳ đâu: kiểm trùng clip với RADAR tiếng Việt (so số mẫu chính xác rồi so dạng sóng) |
| 3 | **Hệ thống mã nguồn mở tiếng Việt là VITS, Edge-TTS, F5-TTS**; Bảng 1 của bài ghi A1 (VITS-MMS), A4 (FastSpeech2), A6 (F5-TTS). Dữ liệu không có FastSpeech2 mà có Edge-TTS (A2) | Đủ 7 hệ thống như plan, nhưng tên theo dữ liệu, không theo Bảng 1 | Báo cáo dùng tên trong `source_model`, ghi chú chỗ lệch với bài |
| 4 | **Tiếng Việt lệch nhãn: 1 thật : 2,14 giả**, khác tỉ lệ 1 : 1,004 của Bảng 1 (tính cho cả bộ) | Train lấy nguyên sẽ lệch về lớp giả | Cân bằng số câu thật và giả khi chọn train, dev |
| 5 | **Câu giả của VITS, Edge-TTS, F5-TTS ghép được với câu thật qua văn bản** (khớp 100%, mỗi văn bản một câu giả cho mỗi hệ thống). Nhờ đó gán được người nói VIVOS (và tập) cho câu giả đọc văn bản VIVOS | Chia được tiếng giả theo đúng người nói của câu thật cùng văn bản, tránh trùng nội dung giữa train và test | Dùng trong `select` |
| 6 | **Thông tin giọng không đều:** VIVOS 65 người nói (46 có ~250 câu, 19 chỉ 40 câu); Common Voice, VITS, Edge-TTS, F5-TTS, ChatGPT không ghi giọng; ElevenLabs 12 giọng, phần lớn là giọng tiếng Anh, Indonesia, Malay, Hindi, Tamil đọc tiếng Việt (bài mục 2.2.3: cố ý dùng người nói không bản ngữ); HeyGen 290 video; **MiniMax gần như một giọng** (`vie_2` 2.032 / 2.119 câu) | MiniMax không tách giọng giữa train và test được. ElevenLabs, HeyGen tách được theo giọng hoặc video | MiniMax nên để riêng toàn bộ (mục 3, D2) |
| 7 | **Toàn bộ tiếng Việt chỉ 7,18 GB** (411 row group liền nhau) | Lượt 3 không phải chọn trước rồi mới tải; tải trên máy được | |
| 8 | **Pipeline:** `main.py` chỉ chấm dev và lưu checkpoint mỗi `valid_step` bước (cấu hình 20.000) và một lần sau epoch cuối; checkpoint giữ 1 bản tốt nhất theo EER dev. Với tập 10–20 giờ và batch 100 giây (vài trăm bước mỗi epoch), cả lần huấn luyện chỉ chấm dev một lần ở cuối | Không chọn được checkpoint tốt nhất; mất phiên giữa chừng là mất hết | Ngày 2: đặt `valid_step` theo số bước một epoch (đo ở pilot). Dev nhỏ (2.070 câu, 2,2 giờ) vì chạy nhiều lần. Mốc của bài GRPO: ~50 giờ DFE24, tối đa 10 epoch, chấm dev mỗi 20.000 bước, chọn checkpoint có EER dev thấp nhất (bài GRPO mục 3.1–3.2) |
| 9 | **Learning rate thực tế thấp hơn nhiều so với giá trị đặt.** `WarmAndExpDecayLRSchedule` của SpeechBrain 1.0.2 tăng tuyến tính `lr × bước / warmup`; với `lr: 1e-6`, `warmup: 80000`, `optimizer_step_limit: 50000` thì không bao giờ tới đỉnh. Ước tính: bài GRPO (~50 giờ, 10 epoch, ~18–24 nghìn bước) chỉ lên tới ~2–3 × 10⁻⁷; tập 17,9 giờ của ta (10 epoch, ~6–7 nghìn bước) chỉ ~0,8 × 10⁻⁷, gần bằng learning rate post-train (1e-7) | "Sửa warmup" theo plan sẽ làm ta fine-tune mạnh hơn bài khoảng 3–10 lần. Mức quên của SFT (câu hỏi 2) phụ thuộc trực tiếp vào điều này | **Cần người dùng chốt sau pilot** (mục 4): giữ lịch gốc (giống bài GRPO) hay chỉnh warmup theo số bước |
| 10 | **RawBoost áp lên cả dev khi `use_da: True`**: `audio_pipeline` trong `dataio/dataio.py` dùng chung cho train và dev (kể cả cắt ngẫu nhiên câu > 13 s) | EER dev dùng để chọn checkpoint của R (có RawBoost) đo trên dev đã làm méo ngẫu nhiên; của A0 (không RawBoost) đo trên dev sạch. Chọn checkpoint của hai cấu hình không cùng điều kiện | **Cần người dùng chốt sau pilot:** giữ code gốc (ghi rõ hạn chế) hay vá để RawBoost chỉ áp lên train. Mọi số báo cáo vẫn chấm trên dev/test sạch bằng `score_zeroshot.py` |
| 11 | **HeyGen và MiniMax là đoạn cắt đúng 4,00 s**, không có khoảng lặng hai đầu (đoạn cắt từ video hoặc bản ghi dài, văn bản do WhisperX nhận dạng) | Hệ thống để riêng MiniMax chung kiểu cắt đoạn với HeyGen trong train, nên "chưa gặp" chỉ ở mức hệ thống sinh, không ở mức cách cắt | Ghi khi diễn giải kết quả MiniMax; so với F5-TTS (khác hẳn) |
| 12 | **Khoảng lặng hai đầu khác nhau theo nguồn**: tiếng thật VIVOS đã cắt sát (trung vị 0,06 s đầu, 0,04 s cuối); Common Voice có nhiều (0,62 / 0,52 s); Edge-TTS cuối 0,82 s, VITS đầu 0,36 s. Một đặc trưng khoảng lặng tách tiếng thật Common Voice khỏi tiếng giả đã thấy với EER 28% | Nếu mô hình học "có khoảng lặng thì giả", tiếng thật Common Voice (và RADAR) sẽ bị coi là giả nhiều hơn sau fine-tune | Theo dõi tỉ lệ tiếng thật bị coi là giả trên `test_cvreal` và RADAR tiếng Việt trước và sau fine-tune. Không cắt khoảng lặng (giữ dữ liệu gốc) |

## 3. Chia tập (**đã chốt 14/11**: người dùng chọn đề xuất ở cả D1, D2, D3)

Không dùng split có sẵn (phát hiện 1). Chia lại từ toàn bộ 81.020 câu tiếng Việt, seed 2026.

**Kết quả** (`protocols/ids/seaspoof_vi_split.json`; giờ đo thật từ file đã tải, `results/seaspoof/hours_by_set.csv`):

| Tập | Câu thật + giả | Giờ thật + giả | Thành phần |
|---|---|---|---|
| `train` | 8.910 + 8.910 | 10,31 + 7,57 = **17,88** | VIVOS 46 người nói; giả 1.782 câu mỗi hệ thống × 5 (VITS, Edge-TTS, ChatGPT, ElevenLabs, HeyGen) |
| `train_10h` (lồng trong `train`) | 4.454 + 4.455 | 8,94 | Một nửa, giữ tỉ lệ theo người nói và hệ thống |
| `train_n1000`, `n200`, `n50` (lồng nhau, trong `train_10h`) | 1.000 / 200 / 50 mỗi lớp | 2,03 / 0,42 / 0,10 | Cho thí nghiệm ít mẫu ở GĐ6; giả chia đều 5 hệ thống |
| `dev` | 1.035 + 1.035 | 1,32 + 0,88 = 2,20 | VIVOS 5 người nói; giả 207 câu mỗi hệ thống |
| `test` (một file, cột `group`) | 4.475 + 7.521 | 13,51 | `real_vivos` 2.475 (14 người nói), `seen` 495 × 5, `f5_tts` 2.927, `minimax` 2.119, `real_cv` 2.000 |

Tập con của `test` (tính độ đo trên cùng một lần chấm): `test_seen` = real_vivos + seen; `test_f5` = real_vivos + f5_tts; `test_minimax` = real_vivos + minimax; `test_cvreal` = real_cv + seen.

**Kiểm tra** (`seaspoof_vi_split.json`, mục `check`): trùng người nói VIVOS, giọng ElevenLabs, video HeyGen giữa train, dev, test: **0**. Văn bản vừa là câu thật ở tập này vừa là câu giả ở tập khác: **0** (đã bỏ 77 clip Common Voice ở dev, 145 ở test, và 7 câu giả ở test vì trùng văn bản khác nhãn). Trùng văn bản cùng nhãn vẫn còn (371 văn bản train–dev, 487 train–test): 12 giọng ElevenLabs đọc chung một bộ câu, câu ngắn phổ biến do ASR nhận dạng (HeyGen, MiniMax), câu nhắc VIVOS do nhiều người đọc. Bỏ 306 văn bản VIVOS do người nói ở hai tập khác nhau cùng đọc.

### 3.1. Kiểm lối tắt (14/11, `results/seaspoof/check_output.txt`)

EER (%) khi chỉ dùng **một** đặc trưng thô làm điểm (50% = không tách được):

| Tập | Độ dài | Âm lượng (RMS) | Biên độ đỉnh | Lặng đầu | Lặng cuối |
|---|---|---|---|---|---|
| train | 39,8 | 42,0 | 49,3 | 49,7 | 39,1 |
| dev | 37,7 | 38,8 | 46,2 | 45,9 | 42,3 |
| test_seen | 37,7 | 33,1 | 45,5 | 48,2 | 37,8 |
| test_f5 | 34,8 | 28,7 | 33,8 | 47,6 | 49,0 |
| test_minimax | 40,4 | 48,4 | 40,4 | 36,0 | 45,4 |
| test_cvreal | 41,6 | 38,0 | 45,6 | 28,4 | 30,4 |

Không có lối tắt mạnh (train ≥ 39%). Đáng chú ý: F5-TTS to hơn tiếng thật test (RMS −20,1 so với −23,6 dB); tiếng thật Common Voice khác tiếng giả ở khoảng lặng (phát hiện 12). Một đặc trưng đơn lẻ không bắt được tương tác giữa các đặc trưng, nên đây chỉ là kiểm tra tối thiểu.

**Đề xuất ban đầu** (giữ để đối chiếu; số là ước lượng trước khi tải):

| Tập | Tiếng thật | Tiếng giả | Câu thật + giả | Giờ (ước) |
|---|---|---|---|---|
| `train` | VIVOS, 46 người nói | 5 hệ thống "đã thấy": VITS, Edge-TTS, ChatGPT, ElevenLabs, HeyGen; lấy số câu bằng tiếng thật, chia đều theo hệ thống | 8.910 + 8.910 | ~20 |
| `train_10h` | Tập con lồng trong `train` (một nửa, giữ tỉ lệ theo người nói và hệ thống) | | ~4.450 + 4.450 | ~10 |
| `dev` | VIVOS, 5 người nói | 5 hệ thống đã thấy, cùng số câu | ~1.035 + 1.035 | ~2,5 |
| `test_seen` | VIVOS, 14 người nói | 5 hệ thống đã thấy | ~2.475 + 2.475 | ~6 |
| `test_heldout` | Như `test_seen` | **F5-TTS** (câu thuộc văn bản của tập test) và **MiniMax** (toàn bộ); báo riêng từng hệ thống | ~2.475 + ~4.900 | ~8 |
| `test_cvreal` | 2.000 clip Common Voice | So với tiếng giả của `test_seen` | 2.000 + 2.475 | ~6 |
| RADAR tiếng Việt, `ood_forget`, tập câu khó | Đã có từ GĐ3 | | Giữ nguyên | |

Người nói VIVOS chia theo tỉ lệ số câu khoảng 70 / 10 / 20 (train / dev / test), xếp ngẫu nhiên theo seed. Cách gán tiếng giả vào tập, để không trùng người nói và nội dung:

| Tiếng giả | Gán theo |
|---|---|
| VITS, Edge-TTS, F5-TTS đọc văn bản VIVOS | Tập của người nói VIVOS đọc văn bản đó (phát hiện 5). Văn bản do người nói ở hai tập khác nhau cùng đọc thì bỏ |
| VITS, Edge-TTS, F5-TTS, ChatGPT đọc văn bản Common Voice | Theo nhóm văn bản, 70 / 10 / 20 (cùng văn bản thì cùng tập) |
| ElevenLabs | Theo giọng (12 giọng) |
| HeyGen | Theo video (290 video) |
| F5-TTS, MiniMax | Không có trong `train`, `dev` |

Ước tính chi phí chấm mỗi lần chạy (một T4, bản 317M, ~35x thời gian thực): các tập SEA-Spoof ~25 phút (~14 giờ âm thanh không trùng), RADAR tiếng Việt 24–38 phút, `ood_forget` 7–11 phút; tổng ~1 giờ.

**Đã chốt** (14/11, người dùng chọn cột "Đề xuất" ở cả ba dòng)

| # | Câu hỏi | Đề xuất | Lý do | Phương án khác |
|---|---|---|---|---|
| D1 | Tiếng thật Common Voice | **Không đưa vào train, dev**; lấy 2.000 clip làm `test_cvreal` | Plan đã đặt nguyên tắc không thêm Common Voice vào train vì RADAR lấy tiếng thật từ kho này; có thể trùng clip với RADAR; không có nhãn người nói. `test_cvreal` đo thẳng yếu tố gây nhiễu: tiếng thật từ nguồn khác có bị coi là giả sau fine-tune không | Đưa vào train: tiếng thật đa dạng hơn (~2 nguồn), nhưng phải kiểm trùng clip với RADAR và chấp nhận có thể trùng người nói |
| D2 | Hệ thống để riêng | **F5-TTS và MiniMax, cố định cho mọi cấu hình** | F5-TTS: mã nguồn mở, nhái giọng zero-shot, đủ lớn, không có trong RADAR. MiniMax: thương mại, không có trong RADAR, và không tách giọng được (phát hiện 6). Giữ ElevenLabs và ChatGPT trong train để kiểm phát hiện 18 của GĐ3 (RADAR có ElevenLabs, OpenAI). Luân phiên tốn quota gấp nhiều lần | Chỉ F5-TTS; hoặc để riêng ElevenLabs + ChatGPT (khi đó RADAR gần như toàn hệ thống chưa thấy) |
| D3 | Kích thước train | **~20 giờ (`train`), kèm `train_10h` lồng bên trong làm dự phòng**; số epoch chốt sau khi đo tốc độ ở ngày 2 | Dùng hết người nói VIVOS của train cho tiếng thật đa dạng nhất. Bài GRPO dùng ~50 giờ. `train_10h` là mức cắt số 6 của plan, chuẩn bị sẵn để không phải chia lại | Chỉ ~10 giờ |

Nếu dùng phương án dự phòng (MLAAD `fake/vi` + VIVOS): tiếng thật và tiếng giả từ hai kho khác nhau, phải kiểm tra lối tắt (ví dụ độ dài, mức âm lượng, khoảng lặng) trước khi huấn luyện. Với SEA-Spoof theo D1, train cũng có dạng này ở mức nhẹ hơn (tiếng thật chỉ từ VIVOS); kiểm ở mục 3.1.

## 4. Pipeline (ngày 2)

Xuất phát từ `hparams/ft_grpo/mms_300m_pre-post-sft.yaml` của AntiDeepfake (khởi tạo từ checkpoint post-trained; RawBoost `algo: 5` khi fine-tune).

**Code (14/11):**

| File | Vai trò |
|---|---|
| `src/finetune.py` | Bọc `main.py` gốc (không sửa code AntiDeepfake): ghi đè siêu tham số (`--set`), theo dõi GPU và CPU mỗi 5 s (`monitor.csv`), hết bộ nhớ thì giảm nửa `max_batch_length` rồi chạy lại, đọc số batch mỗi epoch, giây mỗi bước, EER dev; lấy checkpoint tốt nhất (`ssl.ckpt` -> `ckpt/<run>.ckpt`), chấm bằng `score_zeroshot.py --local` |
| `src/score_zeroshot.py` | Thêm `--local <tên>=<ckpt>,<mô hình gốc>` (chấm checkpoint cục bộ) và nhiều `--data-root` (RADAR từ `orpheus-prep`, SEA-Spoof từ `orpheus-seaspoof-vi`). Hành vi cũ giữ nguyên |
| `kaggle/finetune/` | Kernel `orpheus-finetune` (GPU T4 × 2): các lần huấn luyện theo từng GPU ở biến `RUNS` |
| `kaggle/push.sh` | Thêm `data` (dataset SEA-Spoof riêng tư), `output finetune N` (không tải `*.ckpt`) |
| `kaggle/archive_run.py` | Lưu thêm `train_summary.csv`, `train_logs/`; `merge-finetune` gộp kết quả vào `results/finetune/` |

**Lần chạy 1 (pilot đo đạc, không phải thí nghiệm GĐ5)**, hai GPU song song:

| GPU | Lần | Cấu hình | Để đo |
|---|---|---|---|
| 0 | `pilot_R` | R (MMS-300M Default, SFT, RawBoost), dừng ở 300 bước, `num_workers=2` | Giây mỗi bước, số batch mỗi epoch, bộ nhớ đỉnh, mức dùng GPU và CPU, thời gian chấm dev |
| 1 | `overfit_n50` | 100 câu (`train_n50`), train = dev, không RawBoost, `lr=1e-5`, `warmup=1`, 30 epoch, chấm mỗi 25 bước | Pipeline có học được không (EER trên chính tập train về 0) |
| 1 | `pilot_A0` | Như `pilot_R` nhưng không RawBoost; chấm lại dev và `ood_forget` qua `score_zeroshot.py --local` | Chi phí RawBoost trên CPU; EER dev lúc huấn luyện có khớp EER chấm lại không (cùng trọng số, dev sạch); đường chấm với nhiều nguồn dữ liệu |

Với lịch gốc, 300 bước chỉ đưa learning rate lên ~4 × 10⁻⁹ (phát hiện 9), nên mô hình gần như không đổi: EER dev của pilot xấp xỉ EER dùng thẳng (S0) trên dev SEA-Spoof.

**Cần chốt sau pilot** (phát hiện 9, 10): lịch learning rate; RawBoost trên dev; số epoch và `valid_step`.

- [x] Kernel Kaggle fine-tune: dùng lại cách dựng môi trường và lưu bằng chứng của GĐ3 (`results/runs/<kernel>-v<N>/`); mở rộng `push.sh output` cho kernel mới.
- [ ] Lưu checkpoint tốt nhất vào `/kaggle/working/results/ckpt/` (đã làm). **Chạy tiếp qua nhiều phiên không làm được đúng với code gốc:** checkpoint chỉ giữ 1 bản tốt nhất, không lưu bộ đếm epoch và lịch learning rate (chỉ `ssl`, cộng `optimizer`, `brain` do SpeechBrain tự thêm). Vì vậy mỗi lần huấn luyện phải vừa một phiên 12 giờ.
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

- [x] Có quyền SEA-Spoof (hoặc đã chuyển sang MLAAD + VIVOS và ghi rõ hạn chế): có từ 14/11
- [x] Danh sách câu train, dev, test, hệ thống để riêng cố định trong `protocols/ids/`, không trùng người nói (14/11, mục 3)
- [ ] Pipeline overfit được một batch; bộ nhớ vừa T4
- [ ] Đã đo thời gian một epoch và tốc độ đọc dữ liệu; bảng ngân sách mục 5 điền đủ và vừa quota
- [ ] Bằng chứng mọi lần chạy trong `results/runs/`

Điều kiện của plan: ngân sách vừa quota. Nếu MMS-300M không vừa sau khi giảm batch, chuyển cấu hình mốc sang W2V-Small. Nếu không fine-tune được, chuyển sang tầng 0.
