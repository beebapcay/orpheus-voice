# GĐ3. Post-training và zero-shot

Kiến thức chung nằm trong `note/Post-Training-SSL.md` (bài AntiDeepfake) và `note/RawBoost.md`. File này chỉ giữ phần gắn với đồ án: trang cần đọc kỹ, đối chiếu bài với code và checkpoint, các phát hiện ảnh hưởng thí nghiệm, cách chuẩn bị dữ liệu và chấm zero-shot trên Kaggle, nháp câu vấn đáp 3.

## Tiến độ

| Ngày | Việc | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Tóm tắt AntiDeepfake, RawBoost | `note/Post-Training-SSL.md`, `note/RawBoost.md`, mục 1–3 dưới đây | Có tài liệu (13/11), **chưa đọc** |
| 1 | Tải PDF RawBoost | `resource/paper/RawBoost-2111.04433.pdf` | Xong |
| 2 | Chọn tập kiểm tra | `protocols/ids/*.tsv`, `stats.json` (chạy `prep_data.py select` trên máy) | Xong |
| 2 | Code chuẩn bị dữ liệu và chấm | `src/prep_data.py`, `src/score_zeroshot.py`, `src/score_w2v2_aasist.py`, `kaggle/` | Viết xong, thử trên máy với `main.py` giả; **chưa chạy trên Kaggle** |
| 2 | Chạy trên Kaggle | `results/zeroshot/` | Chờ người dùng cài Kaggle CLI (mục 5.5) |

Bài AntiDeepfake đọc ở bản **arXiv v4 (21/10/2025)**, tức sau khi repo sửa cách tính EER (16/10/2025). Số MMS-300M trong Bảng II khớp từng ô với bảng trên trang Hugging Face của checkpoint. Code đối chiếu ở commit `86098091` như GĐ2.

## 1. Trang cần đọc kỹ

Đọc kiến trúc tổng thể và kết quả trước; phần dữ liệu chi tiết và siêu tham số để sau.

**AntiDeepfake** (`resource/paper/AntiDeepfake-2506.21090.pdf`, 8 trang, phần chính 5,5 trang):

| Lượt | Phần | Trang | Câu hỏi khi đọc |
|---|---|---|---|
| 1. Tổng thể | Tóm tắt, Hình 1, mục I | 1 | Post-training nằm ở đâu, khác fine-tune thế nào? |
| 1. Tổng thể | Hình 2, mục III-B | 3 | Kiến trúc gồm những khối nào? Khối nào được cập nhật? |
| 1. Kết quả | Mục IV-B, IV-C, **Bảng II** | 3–4 | 6 tập kiểm tra là gì? Mô hình nào tốt nhất trên In-the-Wild? RawBoost giúp ở đâu? |
| 1. Kết quả | Mục IV-D, Hình 3, **Bảng III** | 4–5 | Post-training giúp fine-tune bao nhiêu? Vì sao P3 hơn MMS-300M ở 4 giây? |
| 1. Kết quả | Mục V (hạn chế) | 5–6 | Tác giả tự nhận những điểm yếu nào? |
| 2. Chi tiết | Mục III-A, **Bảng I** | 2 | 5 nhóm dữ liệu; nhóm nào chiếm nhiều giờ nhất? |
| 2. Chi tiết | Mục IV-A | 3 | Learning rate, batch, số bước, cắt câu |
| 2. Chi tiết | Mục IV-E, Hình 4–5 | 5–6 | t-SNE nói gì? |

**RawBoost** (`resource/paper/RawBoost-2111.04433.pdf`, 4 trang nội dung):

| Lượt | Phần | Trang | Câu hỏi khi đọc |
|---|---|---|---|
| 1. Tổng thể | Tóm tắt, mục 1 | 1 | Vì sao cần tăng cường trên dạng sóng? |
| 1. Tổng thể | Hình 1 | 3 | Ba loại nhiễu là gì? |
| 1. Kết quả | **Bảng 2**, mục 4.4 | 4 | Cấu hình nào tốt nhất? Vì sao thêm (3) không giúp? |
| 2. Chi tiết | Mục 3.1–3.3, công thức (1)–(6) | 2 | Mỗi loại nhiễu tính thế nào? |
| 2. Chi tiết | Bảng 1, Hình 2, mục 4.3 | 3 | Khoảng giá trị tham số; nối tiếp và song song |

**Code** (commit `86098091`):

| File | Dòng | Câu hỏi |
|---|---|---|
| `dataio/rawboost.py` | 63–98, 102–206 | `algo = 5` gọi những hàm nào, theo thứ tự nào? |
| `hparams/mms_300m.yaml` | khối `rawboost` | Tham số có khớp Bảng 1 của bài không? |
| `main.py` | `analyze()` | Chế độ `analysis` ghi ra những file nào? |
| `models/W2V.py` | 137–145 | `analysis` trả về embedding ở đâu trong mạng? |

**Tự kiểm tra:** câu hỏi cuối hai note; nháp vấn đáp ở mục 6.

## 2. Đối chiếu bài, code và checkpoint

| Nội dung | Bài | Code / checkpoint | Khớp? |
|---|---|---|---|
| Kiến trúc | SSL → GAP → FC (Hình 2) | `models/W2V.py`: `extract_feat` → `AdaptiveAvgPool1d(1)` → `Linear(D, 2)` | Khớp |
| Cắt câu khi huấn luyện | > 13 s cắt ngẫu nhiên 10–13 s | `dataio.py` | Khớp |
| RawBoost | "cấu hình tốt nhất" của bài RawBoost = (1)+(2) nối tiếp | `algo: 5` = `LnL_convolutive_noise` rồi `ISD_additive_noise`; tham số khớp Bảng 1 của RawBoost | Khớp |
| Learning rate | Tăng tới 1e-7 trong 80K bước, **giảm tuyến tính về 0** ở bước 800K | YAML: `WarmAndExpDecayLRSchedule`, `decay_factor: 0.01` (giảm theo hàm mũ) | **Không khớp**; ảnh hưởng GĐ4 khi chỉnh cấu hình fine-tune |
| Bản Default / NDA | Bảng II: "with RawBoost / no RawBoost" | Model card HF: Default = có tăng cường, NDA = không; card ghi "RawBoost series: (1)+(2)" | Khớp |
| Bảng fine-tune | Bảng III (v4) | README repo có bảng khác ở vài ô, ví dụ MMS-1B không post-train, 10 s: bài 13,57, README 10,32 | **Không khớp**; dùng số trong bài v4 |
| XLS-R + SLS trên In-the-Wild | Bảng II của AntiDeepfake ghi 7,45 | Plan và đề cương ghi 7,46 (lấy từ bài XLSR-SLS) | Cần đối chiếu bài XLSR-SLS (chưa có PDF) |
| Ghi chú `algo` trong YAML | | Ghi "8: 1+3", nhưng code `algo == 8` là (1)+(2) **song song** | Comment sai, code đúng với bài |

**Checkpoint trên Zenodo và Hugging Face:**

| | Zenodo (record 15580543) | Hugging Face (`nii-yamagishilab/*-anti-deepfake[-nda]`) |
|---|---|---|
| Định dạng | `.ckpt` (state_dict PyTorch), repo nạp trực tiếp | `model.safetensors` + `config.json`, nạp bằng `PyTorchModelHubMixin` |
| Tên khoá | `m_ssl.model.*`, `proj_fc.*` | Cùng lớp `DeepfakeDetector` với cùng tên module trong script inference trên card |
| Dung lượng 14 file | 0,38–8,65 GB, tổng ~46 GB | Như Zenodo |
| Tốc độ tải đo được | ~2,4 MB/s trên Colab (GĐ2) → ~5,4 giờ cho 46 GB | Chưa đo trên Kaggle |

Cách dùng: tải `model.safetensors` từ Hugging Face, chuyển sang `.ckpt` (`torch.save` của state_dict), rồi nạp bằng code gốc. **Kiểm chứng bắt buộc:** chấm toy example với `mms_300m` lấy từ HF phải ra đúng EER 6,11% như GĐ2, và log không có cảnh báo nạp trọng số. Nếu HF lỗi, script tự chuyển sang Zenodo.

## 3. Phát hiện ảnh hưởng thí nghiệm

| # | Phát hiện | Ảnh hưởng | Việc cần làm |
|---|---|---|---|
| 1 | **MMS-300M có RawBoost kém hơn bản không RawBoost trên In-the-Wild** (2,90 so với 2,00) và FoR-norm (5,92 so với 2,70). Trên In-the-Wild, 6/7 backbone được RawBoost giúp; MMS-300M là ngoại lệ duy nhất | Cấu hình mốc R dùng MMS-300M có RawBoost. A0 (bỏ RawBoost khi fine-tune) có thể ra kết quả ngược kỳ vọng | Chấm zero-shot cả Default lẫn NDA của MMS-300M trên RADAR tiếng Việt để biết RawBoost có lợi trên tiếng Việt không, **trước** khi chạy A0 |
| 2 | **RADAR dev không dùng được:** tiếng thật lấy từ LibriTTS dev/test (bài RADAR mục III-A); LibriTTS, LibriTTS-R, LibriTTS-Vocoded nằm trong dữ liệu post-training | Nếu lấy "RADAR tiếng Anh" từ dev, mô hình có thể đã thấy chính các câu thật đó | "RADAR tiếng Anh" lấy từ **tập eval** (thật: Common Voice 8.000 + People's Speech 7.000; giả: 10 hệ thống) |
| 3 | RADAR tiếng Việt: thật = Common Voice 3.000 + FOSD 4.000 (Bảng II bài RADAR). Common Voice có trong dữ liệu **pre-train** của XLS-R (v6.1), MMS (v9.0) | Không phải rò rỉ nhãn (pre-train không có nhãn thật/giả), nhưng một phần tiếng thật có thể "quen" với backbone | Ghi chú khi diễn giải; không loại bỏ |
| 4 | RADAR **không có nhãn hệ thống sinh và nguồn tiếng thật theo từng câu**; chỉ có nhãn ngôn ngữ | Không tính được EER theo từng hệ thống trên RADAR | Phân tích theo hệ thống (GĐ5, GĐ6) chỉ làm trên SEA-Spoof và tập tự tạo |
| 5 | Dữ liệu post-training có FLEURS và FLEURS-R (102 ngôn ngữ) và MLAAD (38 ngôn ngữ). Theo danh sách của FLEURS có tiếng Việt; chưa kiểm tra | Mô hình post-trained có thể đã thấy tiếng Việt thật (FLEURS) và tiếng Việt "phục hồi" bị gán nhãn có artifact (FLEURS-R) | Kiểm tra danh sách ngôn ngữ FLEURS và phiên bản MLAAD đã dùng (đã có trong mục 9 của plan) |
| 6 | Độ dài câu: RADAR tiếng Việt dài nhất 45,7 s, RADAR tiếng Anh (tập con) 51,4 s, In-the-Wild 25,0 s | Bẫy số 4 của GĐ2 (câu vài phút tràn bộ nhớ) không xảy ra với các tập này | Không cần cắt đoạn |
| 7 | Chế độ `analysis` ghi cả file điểm lẫn embedding sau lớp trung bình (`.pkl`), nhưng **chỉ ghi khi chạy xong cả tập** | Mất phiên giữa chừng là mất hết | `score_zeroshot.py` chia shard 2.000 câu; shard xong thì lưu ngay |
| 8 | W2V2-AASIST (baseline): repo HF có `LA_model.pth` và bản ONNX; **không có** `w2v2_aasist.py`, `_net.py` mà model card nhắc tới. Chỉ nhìn **4,04 s đầu** của câu (64.600 mẫu, câu ngắn thì lặp lại), không chuẩn hoá biên độ, điểm = logit lớp thật (đã đối chiếu `data_utils_SSL.py` và `main_SSL_LA.py` của repo gốc) | So với AntiDeepfake (nhìn cả câu) là không cùng điều kiện đầu vào. actDCF của baseline vô nghĩa vì điểm không phải log-odds | Dùng ONNX; báo EER và minDCF; ghi rõ khác biệt. Đối chiếu: In-the-Wild đủ bộ phải gần **11,22%** như card công bố |
| 9 | Kết quả có RawBoost / không RawBoost đổi chiều theo tập và theo backbone (note, mục 5) | Không suy được tác dụng trên tiếng Việt từ số tiếng Anh | Bảng zero-shot RADAR tiếng Việt cần đủ Default và NDA (thứ tự cắt số 4 của plan chỉ dùng khi thiếu quota) |

## 4. Dữ liệu kiểm tra

### 4.1. Các tập

Chọn ngẫu nhiên có cố định seed (2026), danh sách ở `protocols/ids/` (đã commit, nên mọi lần chạy dùng đúng một tập).

| Tập | Câu (thật / giả) | Giờ | Nguồn | Dùng cho |
|---|---|---|---|---|
| `radar_vi` | 13.438 (7.000 / 6.438) | 17,04 | RADAR eval, `VI` | Bảng zero-shot chính; ngoài miền tiếng Việt ở GĐ5–6 |
| `radar_en_sub` | 2.000 (1.000 / 1.000) | 3,02 | RADAR eval, `EN` | Một nửa tập đo mức quên |
| `itw_sub` | 2.000 (1.000 / 1.000) | ~2,43 | In-the-Wild | Nửa còn lại của tập đo mức quên |
| `ood_forget` | 4.000 | ~5,45 | `itw_sub` + `radar_en_sub` | Đo mức quên ở GĐ5–6, báo riêng từng nửa |
| `itw_full` | 31.779 (19.963 / 11.816) | ~37,85 | In-the-Wild đủ bộ | Chỉ để đối chiếu số công bố (MMS-300M 2,90%; W2V2-AASIST 11,22%) |

Giờ của In-the-Wild tính từ kích thước file WAV (16 kHz, float 32-bit) nên có dấu "~"; số đếm câu khớp số công bố. Tập con 1.000 câu mỗi lớp: với EER khoảng 3%, sai số chuẩn khoảng 0,5 điểm, đủ để so cấu hình nhưng không đủ phân biệt chênh lệch dưới 1 điểm.

### 4.2. Lưu trữ

| Thứ | Dung lượng | Cách lấy | Giữ ở đâu |
|---|---|---|---|
| RADAR eval (nén) | 8,75 GB | **Không lưu**: đọc luồng `tar.gz` từ Zenodo, chỉ ghi file cần | |
| RADAR tiếng Việt + tiếng Anh (FLAC) | ~1 GB (ước theo 8,75 GB cho 171 giờ) | Như trên | Output kernel `orpheus-prep` |
| In-the-Wild (zip) | 8,16 GB | Tải vào `/tmp` của kernel, giải nén rồi xoá | |
| In-the-Wild (WAV) | 8,72 GB | | Output kernel `orpheus-prep` |
| Checkpoint | 0,38–8,65 GB mỗi file | Tải từ HF trong kernel GPU, xoá sau khi chấm xong mô hình đó | `/tmp` của kernel (không vào output) |
| Kết quả | Điểm ~1 MB/tập; embedding đến ~100 MB/tập (bản 2B) | | Output kernel `orpheus-zeroshot` → `results/` trên máy; điểm và độ đo commit lên git, embedding thì không |

Output kernel `orpheus-prep` khoảng 10 GB, dưới giới hạn 20 GB của `/kaggle/working`. Không cần Drive.

Nếu phải quay về Colab (Drive 15 GB): bỏ `itw_full`, chỉ lấy `radar_vi`, `radar_en_sub`, `itw_sub` (~1,6 GB). `prep_data.py` lấy `itw_sub` bằng cách đọc từng file trong zip qua HTTP Range, không tải 8 GB.

## 5. Chạy trên Kaggle

### 5.1. Vì sao Kaggle

| | Colab miễn phí | Kaggle |
|---|---|---|
| GPU | 1 × T4, không bảo đảm được cấp | **2 × T4** (`NvidiaTeslaT4`) |
| Phiên | ~4 giờ, phải giữ tab mở | 12 giờ, chạy nền ("Save & Run"), đóng trình duyệt vẫn chạy |
| Quota | Không rõ | Thường báo ~30 giờ/tuần; cần xem trên trang Settings của tài khoản |
| Điều khiển từ máy | CLI chập chờn (GĐ2) | Kaggle CLI: đẩy kernel, xem trạng thái, tải kết quả |
| Ổ đĩa | Drive 15 GB | `/kaggle/working` ~20 GB giữ lại sau phiên |

Giới hạn 12 giờ/phiên và ~30 giờ/tuần lấy từ tài liệu thứ cấp; quota thật xem trên tài khoản.

### 5.2. Luồng

```mermaid
flowchart LR
    M["Máy: sửa code<br/>push.sh src"] --> D["Dataset riêng tư<br/>orpheus-src<br/>(code + danh sách câu)"]
    D --> P["Kernel orpheus-prep<br/>CPU, không tốn quota GPU<br/>tải RADAR, In-the-Wild"]
    P --> Z["Kernel orpheus-zeroshot<br/>GPU T4 × 2<br/>dựng env, chấm theo shard"]
    D --> Z
    Z --> R["push.sh output<br/>results/zeroshot/ trên máy<br/>commit điểm và độ đo"]
```

Code đưa lên Kaggle dưới dạng **dataset riêng tư** thay vì clone GitHub, nên không cần token trên Kaggle. Hai kernel cũng riêng tư.

### 5.3. Kế hoạch các lần chạy

**Lần 1: kiểm chứng và đo tốc độ** (`kaggle/zeroshot/run.py`, biến `JOBS`):

| Job | Mục đích | Điều kiện qua |
|---|---|---|
| `toy:mms_300m` | Checkpoint lấy từ HF giống Zenodo | EER = 6,11%, không cảnh báo nạp trọng số |
| `itw_full:mms_300m` | Code và dữ liệu khớp bài | Gần 2,90% (Bảng II, có RawBoost) |
| `itw_full:w2v2_aasist` | Baseline chạy đúng | Gần 11,22% (model card) |
| `radar_vi:mms_300m,w2v_small,w2v_large,w2v2_aasist` | Bảng zero-shot, 3 backbone sẽ fine-tune + baseline | |
| `ood_forget:mms_300m,w2v_small,w2v_large,w2v2_aasist` | Mốc đo mức quên cho R, B1, B2 (S0 của từng backbone) | |

Ước lượng thô từ GĐ2 (~0,22 s/câu với bản 300M, gồm cả nạp mô hình): `radar_vi` ~50 phút, `itw_full` ~2 giờ, `ood_forget` ~15 phút cho MMS-300M. Lần 1 khoảng 3–3,5 giờ kể cả dựng môi trường (~5 phút). Mỗi job ghi tốc độ thực tế (`x_realtime` trong `summary.csv`).

**Lần 2 trở đi:** 11 checkpoint còn lại trên `radar_vi` (bản NDA của 3 backbone nhỏ; MMS-1B, XLS-R-1B, HuBERT-XL, XLS-R-2B cả hai bản). Chia số job theo tốc độ đo ở lần 1, mỗi lần dưới ~9 giờ. Nếu thiếu quota: bỏ NDA của các bản 1B/2B trước (thứ tự cắt số 4 của plan), giữ NDA của MMS-300M (phát hiện 1).

**Chạy hai GPU song song:** mỗi GPU lấy job kế tiếp trong hàng đợi. Script giới hạn RAM để không nạp hai bản 2B cùng lúc (mỗi bản cần ~17 GB RAM khi nạp).

### 5.4. Kết quả và cách đọc

| File | Nội dung |
|---|---|
| `results/zeroshot/summary.csv` | Mỗi dòng một (tập, mô hình): số câu, giờ âm thanh, giây GPU, tốc độ, EER (%), minDCF, actDCF |
| `results/zeroshot/<mô hình>/<tập>/scores.csv` | Điểm từng câu, định dạng của repo `[logit giả, logit thật]` |
| `.../metrics.csv` | Bảng của `evaluate.py`; `ood_forget` có thêm dòng riêng cho In-the-Wild và RADAR |
| `.../emb.npy` | Embedding sau lớp trung bình (không commit), dùng cho tầng 0 nếu cần |
| `results/zeroshot/setup.log`, `versions.txt` | Môi trường thực tế trên Kaggle |

Điểm cao là thật (log-odds); EER thấp hơn là tốt hơn.

### 5.5. Việc người dùng cần làm trước lần chạy đầu

1. Cài Kaggle CLI trên máy: `uv tool install kaggle` (hoặc `pip install --user kaggle`).
2. Kaggle → Settings → **API → Create New Token**, đặt file vào `~/.kaggle/kaggle.json`, rồi `chmod 600 ~/.kaggle/kaggle.json`.
3. Xác minh số điện thoại trên Kaggle (bắt buộc để bật internet và GPU cho kernel).
4. Xem quota GPU còn lại (Settings hoặc trang Notebooks).
5. Báo Claude. Claude sẽ chạy: `push.sh src` → `push.sh push prep` → chờ xong → `push.sh push zeroshot` → `push.sh status` → `push.sh output zeroshot`.

Hướng dẫn chung: `chore/Quy-Trinh-Chay-Kaggle.md`.

## 6. Nháp câu vấn đáp 3: kiến trúc bên dưới (phần post-training)

> Phần backbone ở file GĐ2 mục 5. Công thức GRPO bổ sung ở GĐ6.

**Trình bày khoảng 1 phút:**

1. Backbone SSL (ví dụ MMS-300M) được pre-train tự giám sát trên tiếng thật, chưa hề thấy tiếng giả.
2. Post-training: gắn thêm lớp trung bình theo thời gian và một lớp FC ra 2 logit, rồi huấn luyện **cả mạng** bằng cross-entropy trên 74K giờ: 56K giờ thật, 18K giờ có artifact (TTS, VC, vocoded, restored, codec), hơn 100 ngôn ngữ, có RawBoost.
3. Learning rate rất nhỏ (1e-7), 800K bước trên 8 GPU H100. Nhóm không tự làm lại được, nên dùng checkpoint công bố.
4. Kết quả: dùng thẳng không học thêm đã đạt 1,23% EER trên In-the-Wild (XLS-R-2B), trong khi XLS-R + SLS 7,45%, AASIST 43%.
5. Fine-tune từ bản post-trained tốt hơn fine-tune từ bản pre-train, rõ nhất với đoạn ngắn.

**Câu hỏi có thể bị hỏi thêm:**

| Câu hỏi | Trả lời ngắn |
|---|---|
| Post-training khác fine-tune ở đâu? | Cùng loss, cùng kiến trúc. Post-training: dữ liệu rất lớn, nhiều miền, mục tiêu là biểu diễn chung. Fine-tune: dữ liệu nhỏ, một miền |
| Vì sao coi tiếng qua vocoder hay codec là "giả"? | Bài coi là "có artifact", không phải deepfake. Mục đích là dạy mô hình nhận ra sai khác nhỏ ở dạng sóng. Loại dữ liệu này rẻ: có tiếng thật là tạo được |
| RawBoost là gì? | Tăng cường dữ liệu trên dạng sóng: lọc nhiều dải + méo hài, và làm méo ngẫu nhiên vài % mẫu. Mô phỏng kênh điện thoại, không cần dữ liệu nhiễu ngoài |
| RawBoost có luôn giúp không? | Không. Trong bài AntiDeepfake, giúp trên In-the-Wild ở 6/7 backbone nhưng thường làm kém đi trên FakeOrReal và Deepfake-Eval-2024. Với MMS-300M trên In-the-Wild: 2,90% có RawBoost, 2,00% không |
| Vì sao không tự post-train? | 74K giờ dữ liệu, 8 GPU H100; vượt xa GPU miễn phí. Đồ án dùng checkpoint công bố và tập trung vào fine-tune |
| Hạn chế của post-training? | Kém trên dữ liệu mạng xã hội mới (EER trên 25%); dữ liệu lệch (~90% giờ thật từ MLS); lợi ích không đều giữa backbone |
| Mô hình đã thấy tiếng Việt chưa? | Có thể qua FLEURS, FLEURS-R, MLAAD (đều đa ngôn ngữ); cần kiểm tra. Pre-train của MMS, XLS-R có tiếng Việt |

## 7. Kiểm tra điều kiện qua GĐ3

- [ ] Đọc kỹ AntiDeepfake và RawBoost theo mục 1; tự trả lời câu hỏi cuối hai note
- [x] Chọn tập kiểm tra, cố định danh sách câu (`protocols/ids/`)
- [x] Script chuẩn bị dữ liệu và chấm, thử trên máy với `main.py` giả (chia shard, chạy tiếp, gộp, độ đo theo từng nguồn)
- [ ] Kaggle CLI và token (mục 5.5)
- [ ] Kernel `orpheus-prep` chạy xong: `radar_vi` đúng 7.000 / 6.438, `itw_full` đúng 19.963 / 11.816
- [ ] Toy với checkpoint từ HF: EER đúng 6,11%, không cảnh báo nạp trọng số
- [ ] **In-the-Wild của MMS-300M bản Default gần 2,90%** (điều kiện của plan)
- [ ] W2V2-AASIST trên In-the-Wild gần 11,22%
- [ ] Bảng EER zero-shot trên RADAR tiếng Việt: 14 checkpoint và baseline
- [ ] Mốc `ood_forget` cho MMS-300M, W2V-Large, W2V-Small
