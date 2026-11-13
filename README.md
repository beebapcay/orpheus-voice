# Orpheus Voice: phát hiện deepfake tiếng Việt

Đồ án môn Xử lý tiếng nói, chủ đề 24 (Fake voice detection). Phân tích các yếu tố quyết định khả năng phát hiện deepfake tiếng Việt của mô hình SSL post-trained (AntiDeepfake), và so fine-tune bằng GRPO với fine-tune có giám sát.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/` | Code của nhóm: `evaluate.py` (EER, minDCF, actDCF), `setup_antideepfake.sh` (dựng môi trường), `prep_data.py` (dữ liệu kiểm tra), `score_zeroshot.py` và `score_w2v2_aasist.py` (chấm zero-shot) |
| `protocols/ids/` | Danh sách câu cố định của từng tập kiểm tra (RADAR, In-the-Wild) |
| `kaggle/` | Kernel Kaggle (`prep`, `zeroshot`) và `push.sh` để đẩy, xem trạng thái, tải kết quả |
| `results/` | Kết quả chạy: `zeroshot/` (điểm, độ đo đã gộp), `runs/` (bằng chứng từng lần chạy Kaggle: code đã đẩy, log, môi trường, `RUN.md`); embedding và file thô không đưa vào git |
| `colab/` | Notebook chạy trên Google Colab |
| `chore/` | Kế hoạch, đề cương, ghi chép từng giai đoạn |
| `note/` | Ghi chú kiến thức |
| `resource/` | PDF tham khảo, **không đưa lên git** |

## Cách làm việc

- **Sửa code:** trên máy (VS Code), rồi commit và push lên GitHub.
- **Chạy GPU:** trên Colab. Notebook tự clone hoặc pull code mới nhất.
- **Lưu kết quả:** Google Drive, thư mục `MyDrive/orpheus/`:
  - `cache/`: checkpoint và dữ liệu tải về, dùng lại giữa các phiên;
  - `runs/<tên>_<thời gian>/`: log, file điểm, phiên bản gói, commit code.

Không sửa code trực tiếp trên Colab: thay đổi sẽ mất khi hết phiên và lệch với GitHub.

## Chạy toy example trên Colab (GĐ2)

1. Mở `colab/GD2_toy_example.ipynb` trên Colab: File → Open notebook → tab **GitHub** → tích *Include private repos* → chọn repo.
2. Runtime → Change runtime type → **T4 GPU**.
3. Secrets (biểu tượng chìa khoá): thêm `GITHUB_TOKEN`, bật *Notebook access*.
4. Sửa `REPO_URL` ở ô Cấu hình, rồi Runtime → Run all.

Kết quả mong đợi: EER = 6,11%, không có cảnh báo nạp trọng số.

## Chấm zero-shot trên Kaggle (GĐ3)

Xem `chore/Quy-Trinh-Chay-Kaggle.md`. Tóm tắt:

```bash
bash kaggle/push.sh src             # đóng gói code thành dataset riêng tư
bash kaggle/push.sh push prep       # tải RADAR tiếng Việt, In-the-Wild (CPU)
bash kaggle/push.sh push zeroshot   # chấm trên 2 GPU T4
bash kaggle/push.sh output zeroshot N   # version N: kết quả về results/zeroshot/, bằng chứng về results/runs/
```

## Chạy `evaluate.py` trên máy

```bash
python src/evaluate.py --selftest
python src/evaluate.py <evaluation_score.csv> --protocol <test.csv> --group-col Attack
```

Chỉ cần `numpy` và `pandas`.

## Phiên bản cố định

| Thành phần | Phiên bản |
|---|---|
| AntiDeepfake | commit `86098091c52c5d0e38c44396c95608f93b0f648d` |
| fairseq | commit `862efab86f649c04ea31545ce28d13c59560113d` |
| Python / torch | 3.9 / 2.6.0 (CUDA 11.8) |
| SpeechBrain | 1.0.2 |

Checkpoint AntiDeepfake (CC BY-NC-SA 4.0) và dữ liệu không nằm trong repo; script tự tải từ Zenodo.
