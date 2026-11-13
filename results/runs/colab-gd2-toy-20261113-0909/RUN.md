# Colab GĐ2 toy example, 13/11/2026 09:09 UTC (lần qua)

Viết tay từ thư mục run mà notebook lưu trên Drive (`MyDrive/orpheus/runs/gd2_toy_20261113-0909/`).

## Thông tin chạy

| | |
|---|---|
| Notebook | `colab/GD2_toy_example.ipynb`, chạy từ trình duyệt (Runtime → Run all) |
| Commit code | `46e3404` (Setup script: kiểm tra import fairseq sau khi cố định numpy) |
| GPU | Tesla T4, 15360 MiB (Colab miễn phí) |
| Giờ (UTC, theo `setup.log`) | Bước env 08:55:35–08:56:35; bước toy 08:57:05–09:07:10 |
| Môi trường | Python 3.9.23; torch 2.6.0+cu118; fairseq 1.0.0a0+862efab; speechbrain 1.0.2; numpy 1.21.2; pandas 1.4.3; AntiDeepfake `86098091` (đủ ở `versions.txt`, `pip-freeze.txt`) |
| Checkpoint | `mms_300m.ckpt` từ Zenodo, cache trên Drive |

Cùng một runtime với lần lỗi `colab-gd2-toy-20261113-0848`: micromamba và torch đã cài từ lần trước, nên bước env chỉ mất 1 phút. `setup.log` ở đây là log nối tiếp, có cả các dòng của lần lỗi (08:44–08:48).

## Kết quả

| Kiểm tra | Kết quả |
|---|---|
| `evaluation.py` của repo | eer 0.0611, khớp README |
| `evaluate.py` của nhóm | EER 6,1101%, minDCF 0,0597, actDCF 0,1710; ngưỡng EER −1,2741 (log-odds) |
| Cảnh báo nạp trọng số | Không có |
| Số câu | 150 (16 thật, 134 giả) |
| Inference | 33 giây gồm nạp mô hình; vòng chấm 16 giây (~9 câu/giây, fp16) |
| Tải về | Toy 25 MB ~40 giây; checkpoint 1,27 GB ~9 phút |

Theo từng kiểu tấn công: `metrics.csv` (dòng `group:A07`–`group:A19`).

## File

| File | Nội dung |
|---|---|
| `run-info.txt` | Commit code, GPU |
| `setup.log` | Log đầy đủ của script (nối tiếp từ lần 08:48) |
| `toy_inference.log` | Log `main.py` lúc inference |
| `evaluation_score.csv` | Logit `[fake, real]` của từng câu |
| `metrics.csv` | Đầu ra `src/evaluate.py` (gộp và theo kiểu tấn công) |
| `versions.txt`, `pip-freeze.txt` | Môi trường thực tế |

Không có: notebook `.ipynb` kèm output các ô (chưa tải về).

## Tái hiện

1. `git checkout 46e3404`.
2. Mở `colab/GD2_toy_example.ipynb` trên Colab, chọn T4, Run all.
3. So `metrics.csv` mới với file ở đây: EER phải đúng 6,1101%.

## Ghi chú

- Kiểm chứng GĐ2 **qua**: EER khớp README đến 4 chữ số, không cảnh báo nạp trọng số. Toy thuộc ASVspoof 2019 LA mà AntiDeepfake đã dùng để huấn luyện, nên đây là bằng chứng code chạy đúng, không phải kết quả đánh giá.
- actDCF 0,171 so với minDCF 0,060: điểm log-odds chưa hiệu chuẩn ngay trong miền.
- Kết quả này được chạy lại trên Kaggle (`orpheus-zeroshot-v1`, job `toy:mms_300m`) với checkpoint lấy từ Hugging Face: cùng 6,1101%.
- Phân tích đầy đủ: `chore/GD2-Backbone-Va-Moi-Truong.md` mục 4.2.
