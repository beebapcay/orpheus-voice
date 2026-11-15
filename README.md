# Orpheus Voice: phát hiện deepfake tiếng Việt

Đồ án môn Xử lý tiếng nói, chủ đề 24 (Fake voice detection). Nghiên cứu có kiểm soát trên tiếng Việt với các mô hình SSL đã post-train cho phát hiện giọng giả ([AntiDeepfake](https://github.com/nii-yamagishilab/AntiDeepfake)):

1. Yếu tố nào quyết định khả năng phát hiện deepfake tiếng Việt: mô hình nền (kích thước, ngôn ngữ pre-train), cách train (post-training, tăng cường RawBoost), dữ liệu (lượng dữ liệu tiếng Việt khi fine-tune)?
2. Fine-tune có giám sát (SFT) trên tiếng Việt có làm mô hình mất khả năng tổng quát hoá không?
3. Fine-tune bằng học tăng cường GRPO có giữ tổng quát hoá tốt hơn SFT không; nếu không, vì sao, và bản GRPO sửa theo nguyên nhân có giúp không?

## Chạy thí nghiệm

| Bạn muốn | Đọc |
|---|---|
| Chạy toàn bộ bộ thí nghiệm trên máy có GPU của bạn | [`server/HUONG-DAN-CHAY.md`](server/HUONG-DAN-CHAY.md) (lệnh `server/local.sh`) |
| Hiểu bộ thí nghiệm (122 lần, theo tầng), cấu hình máy, chạy trên máy thuê vast.ai | [`server/README.md`](server/README.md) |
| Danh sách lần chạy và tham số chính xác | [`server/experiments.py`](server/experiments.py) |

Mọi lần chạy dùng cùng một công thức: MMS-300M post-trained (bản Default), fine-tune trên SEA-Spoof tiếng Việt, batch tối đa 50 giây âm thanh, learning rate 1e-6 với warmup 460 bước, 4.563 bước (3 epoch), chấm trên 5 tập (dev và test SEA-Spoof, RADAR tiếng Việt, In-the-Wild và RADAR tiếng Anh, MLAAD tiếng Việt). Cả bộ phải chạy trên **một loại GPU** để các cấu hình so được với nhau.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/` | Code: chấm điểm (`score_zeroshot.py`, `evaluate.py`), fine-tune (`finetune.py`, gọi `main.py` gốc của AntiDeepfake, không sửa code gốc), biến thể loss (`loss_variants.py`), WiSE-FT (`interp_score.py`), chuẩn bị dữ liệu (`prep_*.py`, `split_radar_regime.py`), phân tích (`analyze_*.py`, `calibrate.py`, `weight_drift.py`) |
| `server/` | Điều phối bộ thí nghiệm: `experiments.py` (danh sách), `run.py` (hàng đợi trên nhiều GPU), `bootstrap.sh` (dựng môi trường), `local.sh` (chạy trên máy của mình), `vast.sh` (điều khiển máy thuê qua SSH), `collect.py`, `analyze.sh` |
| `protocols/ids/` | Danh sách câu cố định của mọi tập (chỉ ID và nhãn, không có âm thanh hay transcript) |
| `results/` | Kết quả đã gộp: `zeroshot/` (mô hình dùng thẳng), `finetune/` (sau fine-tune, bộ Kaggle T4) kèm `analysis/`, `calibration/`, `mlaad/`, `seaspoof/` (kiểm dữ liệu). Kết quả bộ chạy chung sẽ ở `results/server/` |

Một số chú thích trong code trỏ tới `chore/...`: đó là ghi chép nội bộ của chủ đồ án, không có trong repo.

## Mã cấu hình (tên lần chạy là `<mã>_s<seed>`)

| Mã | Cấu hình |
|---|---|
| S0 | Mô hình post-trained dùng thẳng, không fine-tune (MMS-300M; bản NDA; W2V-Large; W2V-Small) |
| R | SFT (cross-entropy), có RawBoost: cấu hình mốc |
| A0 | R không RawBoost |
| P0 | R nhưng khởi tạo từ trọng số chỉ pre-train (chưa post-train) |
| B2, B1, N0 | R với backbone W2V-Large, W2V-Small, MMS-300M bản NDA |
| G1 | GRPO đầy đủ (64 nhãn lấy mẫu mỗi câu, clip 0,2, mô hình cũ cập nhật mỗi 1.000 batch, phạt KL β = 0,04) |
| G2 | GRPO đơn giản (không clip, không mô hình cũ) |
| G3 | G1 với β = 0 |
| G1s | G1 cộng 1 vào mẫu số chuẩn hoá phần thưởng (gần như bỏ chia cho độ lệch chuẩn) |
| G1b | G1 với β = 1 |
| G1s0 | G1s với β = 0 |
| G1lr, Rlr | G1, R với learning rate 1e-7 |
| SFTKL | Cross-entropy như R cộng đúng số hạng KL của GRPO (β = 0,04) |
| G1A, G1sA, G1bA | G1, G1s, G1b không RawBoost |
| `<mã>w<α×100>` | WiSE-FT: nội suy trọng số giữa S0 và bản fine-tune, θ = (1 − α)·θ_S0 + α·θ_FT |
| Rrv, G1rv | R, G1 fine-tune trên nửa RADAR tiếng Việt (kiểm khi miền fine-tune không dễ) |
| SFTn`N`, G1n`N` | R, G1 chỉ với N câu mỗi lớp (N = 50, 200, 1000), cùng số bước |

EER, minDCF, actDCF, tỉ lệ lỗi FAR/FRR: thấp hơn là tốt hơn.

## Dữ liệu và giấy phép

| Bộ | Dùng cho | Giấy phép, lưu ý |
|---|---|---|
| [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) (tiếng Việt) | Fine-tune, kiểm tra trong miền | Phi thương mại, **không phân phối lại**: không chia sẻ âm thanh, xoá khi xong. Mỗi người tự xin quyền trên Hugging Face |
| [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) `fake/vi` | Kiểm tra TTS tiếng Việt chưa gặp | CC BY-NC 4.0 |
| [RADAR 2026](https://zenodo.org/records/22110815) | Kiểm tra ngoài miền (tiếng Việt, tiếng Anh) | CC BY 4.0; script tự tải đúng danh sách câu |
| [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) | Kiểm tra ngoài miền | Script tự tải tập con |
| Checkpoint AntiDeepfake ([Zenodo](https://zenodo.org/records/15580543), Hugging Face) | Mô hình xuất phát | CC BY-NC-SA 4.0; script tự tải |

Repo chỉ chứa danh sách câu, điểm số và độ đo; không chứa âm thanh, checkpoint hay embedding.

## Đánh giá trên máy

```bash
python src/evaluate.py --selftest
python src/evaluate.py <scores.csv> --protocol <test.csv> --group-col Attack
bash server/analyze.sh      # phân tích kết quả bộ chạy chung (cần pandas, numpy, scipy)
```

## Phiên bản cố định

| Thành phần | Phiên bản |
|---|---|
| AntiDeepfake | commit `86098091c52c5d0e38c44396c95608f93b0f648d` |
| fairseq | commit `862efab86f649c04ea31545ce28d13c59560113d` |
| Python / torch | 3.9 / 2.6.0 (CUDA 11.8) |
| SpeechBrain | 1.0.2 |
| SEA-Spoof | revision Hugging Face `132f5dca9b6efe39cf1d3b54a858f167f5a421fc` |
