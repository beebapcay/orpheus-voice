# Benchmark và độ bền của bộ phát hiện

**Phạm vi:** đánh giá bộ phát hiện trên nhiều bộ dữ liệu cùng lúc, và trên âm thanh đã qua các biến đổi ngoài đời thực (nén, thêm nhiễu, vang, ...). Hai công trình: Speech DF Arena (bảng xếp hạng, 2025) và RADAR Challenge 2026 (cuộc thi APSIPA). Mọi bảng số liệu là **EER (%), thấp hơn là tốt hơn**.

## 1. Vì sao cần benchmark chung

Mỗi bài báo thường tự chọn tập kiểm tra, cách tính EER và cách cắt âm thanh riêng, nên các con số khó so sánh với nhau. Benchmark chung chấm mọi mô hình trên **cùng dữ liệu, cùng cách tính**.

## 2. Speech DF Arena

### Thiết lập

| Mục | Nội dung |
|---|---|
| Tập kiểm tra | 14 tập: In-the-Wild, ASVspoof 2019/2021 LA/2021 DF/2024, FoR, CodecFake, ADD 2022 và 2023, DFADD, LibriSeVoc, SONAR |
| Mô hình | 12 mã nguồn mở + 3 thương mại (Whispeak, Syntra, Resemble AI) |
| Dữ liệu huấn luyện | Các mô hình mã nguồn mở dùng trọng số gốc, hầu hết học trên ASVspoof 2019 LA |
| Độ đo | EER từng tập, EER trung bình, **EER gộp** (pooled) |

**EER trung bình và EER gộp khác nhau thế nào:**
- **EER trung bình:** tính EER riêng từng tập, mỗi tập được chọn ngưỡng riêng, rồi lấy trung bình.
- **EER gộp:** dồn điểm của mọi tập vào chung, rồi tìm **một ngưỡng duy nhất**.

EER gộp cao hơn hẳn EER trung bình nghĩa là thang điểm của mô hình **trôi** giữa các tập: ngưỡng tốt cho tập này lại tệ cho tập khác. Đây cũng là vấn đề hiệu chuẩn (Kiến thức nền, mục 6.6).

### Kết quả chính

| Mô hình | Tham số | In-the-Wild | 2021 DF | ASVspoof 2024 | CodecFake | EER TB | EER gộp |
|---|---|---|---|---|---|---|---|
| Whispeak (thương mại) | 98,9M | 1,26 | 3,23 | 9,92 | 0,86 | **3,05** | **3,00** |
| XLSR-SLS | 340M | 7,45 | 1,91 | 18,76 | 33,43 | 13,84 | 15,68 |
| XLSR-Mamba | 319M | 6,70 | 1,88 | 14,40 | 35,26 | 14,21 | 20,12 |
| W2V2-AASIST | 318M | 11,19 | 6,63 | 16,24 | 43,36 | 18,02 | 19,47 |
| AASIST (không SSL) | 0,3M | 43,00 | 21,07 | 35,53 | 51,05 | 34,49 | 33,16 |
| RawNet2 | 17,6M | 49,00 | 22,38 | 40,67 | 49,75 | 35,75 | 35,66 |

**Những gì rút ra:**
- **Không mô hình nào tốt nhất ở mọi tập.**
- Các mô hình **có front-end SSL** vượt hẳn các mô hình không có.
- **Mô hình to hơn chưa chắc tốt hơn:** Whispeak (98,9M) vượt Resemble AI (2.112M).
- **Dữ liệu codec là điểm yếu chung:** các mô hình mã nguồn mở đều trên 33% ở CodecFake.

**Độ bền với biến đổi âm thanh** (ASVspoof 2024):

| Mô hình | Gốc | + Nhạc nền | + Nhiễu | + Tiếng nói khác | + Vang |
|---|---|---|---|---|---|
| XLSR-SLS | 18,76 | 19,92 | 21,88 | 20,94 | 27,02 |
| TCM | 18,85 | 19,28 | 21,25 | 27,32 | 27,78 |

Riêng tiếng vang đã làm EER tăng khoảng 50% tương đối.

### Đọc kỹ

- **Bài nói XLSR-SLS "thấp nhất trừ chuỗi ASVspoof"**, nhưng bảng cho thấy XLSR-Mamba thấp hơn ở In-the-Wild (6,70 so với 7,45) và ASVspoof 2024 (14,40 so với 18,76).
- **Một con số được ghi 3 cách:** EER của XLSR-SLS trên LibriSeVoc là 1,72 (bảng chính), 1,96 (bảng biến đổi) và 1,86 (phần lời).
- **Không biết dữ liệu huấn luyện của mô hình thương mại,** nên không so sánh công bằng được với mô hình mã nguồn mở.
- Chỉ có tiếng Anh và tiếng Trung.

## 3. RADAR Challenge 2026

### Ý tưởng

Âm thanh ngoài đời ít khi "sạch". Trước khi tới bộ phát hiện, nó đã bị nén (Opus, MP3, AAC), đổi tần số lấy mẫu, thêm nhiễu, vang, nhạc nền, cắt khoảng lặng, ... Các biến đổi này có thể **xoá dấu vết giả** (ví dụ nén làm mất dải tần cao), hoặc **thêm dấu vết lạ** khiến tiếng thật trông như giả. RADAR áp các biến đổi này lên **cả tiếng thật lẫn tiếng giả**.

### Hai giai đoạn

| Giai đoạn | Dữ liệu | Mục đích |
|---|---|---|
| 1. Phát triển | 44.034 câu tiếng Anh (thật từ LibriTTS; giả từ 6 hệ thống của LlamaPartialSpoof), biến đổi ở mức vừa phải | Phân tích, chỉnh mô hình |
| 2. Đánh giá | 102.726 câu, 6 biến thể ngôn ngữ, 10 hệ thống TTS, biến đổi mạnh và đa dạng hơn, **không công bố nhãn** | Xếp hạng cuối |

**Tập đánh giá:**

| Ngôn ngữ | Thật | Giả | Tổng |
|---|---|---|---|
| Tiếng Anh | 15.000 | 17.923 | 32.923 |
| Tiếng Anh Singapore | 3.500 | 6.711 | 10.211 |
| Tiếng Trung | 11.500 | 11.234 | 22.734 |
| Tiếng Trung Đài Loan | 3.500 | 1.678 | 5.178 |
| Tiếng Nhật | 9.500 | 8.742 | 18.242 |
| **Tiếng Việt** | **7.000** | **6.438** | **13.438** |

- **Tiếng Việt thật** lấy từ Common Voice (3.000) và FOSD (4.000).
- **Tiếng Việt giả** sinh từ iFlytek (1.474), ElevenLabs (618), Cartesia (868), OpenAI (1.187), Fish Audio S2 Pro (841), Piper (1.450).

**Một số biến đổi trong tập đánh giá** (mỗi biến đổi được áp với xác suất riêng, có thể nối tiếp nhau):

| Xác suất | Biến đổi |
|---|---|
| 75% | Cắt khoảng lặng đầu và cuối |
| 50% | Nén Opus (24–96 kbps), MP3 (48–128 kbps) hoặc AAC (64–96 kbps) |
| 20% | Vang, nhạc nền (SNR 8–20 dB), hiệu ứng âm thanh (SNR 4–12 dB), nén dải động |
| 10% | Giới hạn băng thông 300–3400 Hz (như điện thoại), hạ xuống 8 kHz, rớt gói, đổi tốc độ 0,95–1,05 |

**Cắt khoảng lặng có chủ đích:** khoảng lặng từng là một lối tắt nổi tiếng (Kiến thức nền, mục 8). Cắt đi buộc mô hình không dựa vào nó được.

### Kết quả

| | Phát triển (tiếng Anh) | Đánh giá (đa ngôn ngữ) |
|---|---|---|
| Baseline SSL-AASIST (học trên ASVspoof 2019 LA) | 37,71 | 42,6 |
| Đội A | **1,27** (hạng 1) | 5,67 (hạng 2) |
| Đội C | 4,63 (hạng 3) | **5,10** (hạng 1) |
| Đội J | 16,03 (hạng 10) | 8,04 (hạng 3) |

- Mô hình học trên dữ liệu sạch, tiếng Anh, gần như **đoán bừa** khi gặp dữ liệu đa ngôn ngữ đã qua biến đổi.
- Các đội tốt nhất xuống được khoảng 5%. Làm được, nhưng cần dữ liệu đa ngôn ngữ và tăng cường mạnh.
- **Tốt ở giai đoạn phát triển chưa chắc tốt ở giai đoạn đánh giá:** đội J từ hạng 10 lên hạng 3.

### Đọc kỹ

- **Chưa có kết quả theo từng ngôn ngữ hay từng loại biến đổi.** Bài chỉ có bảng xếp hạng tổng.
- **51% tiếng thật lấy từ Common Voice,** bộ mà nhiều đội có thể đã dùng để huấn luyện. Bài tự nêu rủi ro này.
- **Giả định "tiếng thật đã biến đổi vẫn là thật"** có thể không đúng khi biến đổi quá mạnh. Bài tự nêu.
- **Số đội không nhất quán:** tóm tắt ghi 22 đội ở giai đoạn 2, phần lời ghi 20 đội cộng 1 đội khách, hình ghi 19 đội.
- Biến đổi là mô phỏng, chưa phải chuỗi xử lý thật của mạng xã hội hay ứng dụng gọi điện.

## 4. Những điều rút ra

| Bài học | Bằng chứng |
|---|---|
| Phải đánh giá trên nhiều tập, không chỉ tập cùng miền | Không mô hình nào tốt nhất ở mọi tập trong 14 tập |
| Báo cả EER gộp hoặc actDCF, không chỉ EER từng tập | EER gộp cao hơn EER trung bình khi thang điểm trôi giữa các tập |
| Biến đổi âm thanh là thách thức riêng | Vang làm EER tăng khoảng 50%; baseline 42,6% trên RADAR |
| Dữ liệu huấn luyện quyết định hơn kích thước mô hình | 98,9M vượt 2.112M; các đội đứng đầu RADAR dùng dữ liệu đa ngôn ngữ |

## 5. Câu hỏi tự kiểm tra

1. EER trung bình và EER gộp khác nhau ở đâu? EER gộp cao hơn hẳn nói lên điều gì?
2. Vì sao RADAR áp biến đổi lên cả tiếng thật, không chỉ tiếng giả?
3. Vì sao cắt khoảng lặng là một biến đổi có chủ đích?
4. Phần tiếng Việt của RADAR gồm những nguồn tiếng thật và hệ thống sinh nào?
5. Đội A và đội J cho thấy điều gì về việc chọn mô hình theo tập phát triển?

## Độ bất định do seed huấn luyện (bổ sung)

Khoảng tin cậy bootstrap lấy lại câu chỉ đo độ bất định do **chọn câu kiểm tra**. Khi so hai mô hình đã huấn luyện, còn độ bất định do **seed huấn luyện** (thứ tự batch, khởi tạo, tăng cường dữ liệu ngẫu nhiên).

Quan sát (một cấu hình fine-tune 300M, 3 seed): EER trên một tập ngoài miền 23,1 / 23,2 / 25,8%, độ lệch chuẩn ~1,5 điểm; trên In-the-Wild 7,6 / 4,5 / 4,8%. Khoảng tin cậy theo câu chỉ rộng ~±0,5 điểm, nên hai seed của **cùng một cấu hình** đã "khác nhau có ý nghĩa" theo khoảng này.

| Nguyên tắc | Lý do |
|---|---|
| Chạy cấu hình mốc ít nhất 3 seed, báo trung bình ± độ lệch chuẩn | Có thước đo độ lệch do huấn luyện |
| Chênh lệch của cấu hình 1 seed dưới ~2 độ lệch chuẩn của mốc thì chưa kết luận | Khoảng tin cậy theo câu đánh giá thấp độ bất định |
| Không dùng khoảng tin cậy theo câu làm bằng chứng duy nhất khi so các lần huấn luyện | Chỉ phù hợp khi so các mô hình cố định (ví dụ checkpoint công bố) |


### Bao nhiêu seed thì đủ

Với σ là độ lệch chuẩn qua seed của một cấu hình, chênh lệch nhỏ nhất mà t-test hai mẫu phát hiện được với độ mạnh 80% (α = 0,05) là khoảng:

| Số seed mỗi bên | 3 v 3 | 3 v 5 | 5 v 5 |
|---|---|---|---|
| Chênh lệch nhỏ nhất phát hiện được | ~3,1σ | ~2,5σ | ~2,0σ |

Ví dụ: σ ≈ 1,9 điểm EER thì 3 seed mỗi bên chỉ phát hiện chắc chênh lệch ≥ ~5,8 điểm. Nếu cả hiệu ứng muốn đo chỉ cỡ 4 điểm, 3 v 3 bắt được khoảng một nửa số lần. Trước khi chạy, so cỡ hiệu ứng mong đợi với bảng này. Một tỉ lệ lỗi ở ngưỡng cố định (FRR, FAR) thường dao động theo seed mạnh hơn EER nhiều (ví dụ độ lệch chuẩn 7 điểm so với 1,9), nên chỉ dùng để mô tả, không làm phép thử.

### "Seed" thật ra đổi những gì

Đừng giả định seed đổi mọi nguồn ngẫu nhiên. Đọc code để biết:

| Nguồn ngẫu nhiên | Có thể không theo seed khi | Cách kiểm |
|---|---|---|
| Thứ tự batch | Bộ chia batch có seed riêng với giá trị mặc định, file cấu hình không truyền seed huấn luyện vào | Tìm tham số `seed` của sampler trong file cấu hình đã giải |
| Tăng cường dữ liệu trong worker | Worker lấy seed từ bộ sinh số của tiến trình chính lúc tạo iterator; nếu seed được đặt sau khi dựng mô hình thì hai phương pháp khác nhau (dựng 1 hay 3 bản mô hình) vẫn cùng chuỗi tăng cường | In mã băm danh sách câu và sóng âm sau tăng cường của vài batch đầu, so giữa hai lần chạy |
| Dropout, lấy mẫu trên GPU | Luôn theo seed, nhưng chuỗi lệch nhau ngay khi một phương pháp dùng thêm số ngẫu nhiên trên GPU | |

Nếu các cấu hình cùng seed dùng chung thứ tự batch và chuỗi tăng cường thì đó là thiết kế "số ngẫu nhiên chung": phần nhiễu dùng chung bị khử khi so sánh. Khi đó phân tích bằng mô hình hai yếu tố (cấu hình + khối seed), gộp phương sai dư của nhiều cấu hình, mạnh hơn hẳn t-test từng cặp. Ví dụ với σ ≈ 1,9 và 8–12 cấu hình × 3 seed: ngưỡng phát hiện ~4,5 điểm nếu hiệu ứng seed không dùng chung (nhờ thêm bậc tự do), ~2,3 điểm nếu tương quan 0,75. Ghép cặp từng đôi chỉ với 3 seed thì không đáng (bậc tự do 2). Đổi lại, kết luận chỉ đúng với một thứ tự batch; ghi rõ khi báo cáo.
