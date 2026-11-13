# Colab GĐ2 toy example, 13/11/2026 08:48 UTC (lần lỗi)

Viết tay từ thư mục run mà notebook lưu trên Drive (`MyDrive/orpheus/runs/gd2_toy_20261113-0848/`).

## Thông tin chạy

| | |
|---|---|
| Notebook | `colab/GD2_toy_example.ipynb`, chạy từ trình duyệt |
| Commit code | `8e5f337` (Notebook GĐ2: điền REPO_URL, không để lộ token khi git lỗi) |
| GPU | Tesla T4, 15360 MiB (Colab miễn phí) |
| Giờ (UTC, theo `setup.log`) | 08:44:34 bắt đầu; 08:48:03 lỗi |

## Kết quả

**Lỗi ở bước env:** `LỖI: cài fairseq thất bại`, dù `pip install` báo thành công. Bước toy dừng ngay (`chưa có môi trường`).

**Nguyên nhân:** pip kéo theo numpy 2.0.2 khi cài fairseq. fairseq ở commit `862efab` dùng `np.float` (đã bị xoá từ numpy 1.24), nên `import fairseq` lỗi trước khi script kịp hạ numpy về 1.21.2. Script lúc đó dùng phép `import` để kiểm tra "đã cài", nên báo nhầm là cài thất bại.

**Sửa:** commit `46e3404`: kiểm tra "đã cài" bằng `pip show`, chỉ thử `import fairseq.models.wav2vec` sau khi cài các gói cố định (numpy 1.21.2), và in traceback nếu vẫn lỗi. Lần chạy lại `colab-gd2-toy-20261113-0909` qua.

## File

| File | Nội dung |
|---|---|
| `run-info.txt` | Commit code, GPU |
| `setup.log` | Log script đến lúc lỗi; cuối log là dòng `Successfully installed ... numpy-2.0.2 ...` rồi `LỖI` |

## Ghi chú

- Ghi trong bảng "Lịch sử lỗi đã sửa" của `chore/Quy-Trinh-Chay-Colab.md`.
- Bài học: kiểm tra một gói đã cài hay chưa không nên dựa vào `import` khi gói đó còn phụ thuộc phiên bản của gói khác chưa được cố định.
