# Chạy bộ thí nghiệm trên server thuê (vast.ai)

Một bộ thí nghiệm **hoàn chỉnh** chạy trên một loại máy, để mọi phép so sánh trong báo cáo dùng kết quả cùng phần cứng. Bộ Kaggle (T4) vẫn chạy song song và giữ làm lần lặp độc lập trên phần cứng khác. Lý do và danh sách đầy đủ: `chore/GD6-GRPO.md` mục 9; danh sách lần chạy: `server/experiments.py`.

## 1. Bộ thí nghiệm (54 lần, theo tầng ưu tiên)

| Tầng | Nội dung | Lần chạy |
|---|---|---|
| 0 | Thử đo: tốc độ, bộ nhớ, số worker đọc dữ liệu (2 hoặc 6) cho SFT và G1, 200 bước | 4 (ngắn) |
| 1 | Câu hỏi chính: S0 (dùng thẳng: MMS-300M, MMS-300M NDA, W2V-Large, W2V-Small), R, G1, A0 × 3 seed | 4 + 9 |
| 2 | Thành phần GRPO: G2, G3 × 3 seed | 6 |
| 3 | Các yếu tố GĐ5: P0, B2, N0, B1 × 3 seed | 12 |
| 4 | Ít mẫu: SFT và G1 với N = 50 / 200 / 1000 mỗi lớp × 3 seed, cùng 4.563 bước như bản đủ | 18 |
| 5 | Thêm seed 4, 5 cho R và G1; mốc XLS-R-2B dùng thẳng trên RADAR VI | 5 |

Công thức giữ nguyên như Kaggle (lịch E1', batch 50, GRPO với `expandable_segments`), thêm `NVIDIA_TF32_OVERRIDE=0` để GPU Ampere/Ada tính fp32 như T4. Mỗi lần huấn luyện chấm trên 5 tập: `seaspoof_vi_dev`, `seaspoof_vi_test`, `radar_vi`, `ood_forget`, `mlaad_vi_test`.

## 2. Cấu hình máy cần thuê

| Thứ | Yêu cầu | Lý do |
|---|---|---|
| GPU | **4 × cùng loại**, ≥ 24 GB: RTX 4090, RTX 3090, A5000, A6000, L40S, A100, H100 | G1 cần ~14 GB (đo trên T4). Nhiều GPU thì xong sớm; các GPU cùng loại để cả bộ cùng phần cứng |
| Không dùng | RTX 50xx (Blackwell) | torch 2.6.0 bản CUDA 11.8 của môi trường gốc không hỗ trợ |
| CPU | ≥ 8 nhân mỗi GPU (32 cho 4 GPU) | RawBoost và đọc âm thanh chạy trên CPU; GĐ5 đã thấy CPU là nút thắt với 4 nhân cho 2 GPU |
| RAM | ≥ 16 GB mỗi GPU (64 GB cho 4 GPU) | Mỗi lần nạp 1–3 bản mô hình và tiến trình đọc dữ liệu |
| Ổ đĩa | ≥ 250 GB | Dữ liệu ~15 GB, môi trường ~15 GB, checkpoint gốc ~17 GB, 49 checkpoint sau fine-tune ~60 GB |
| Mạng | Tải xuống nhanh (≥ 500 Mbps nếu có) | RADAR 8,75 GB từ Zenodo, checkpoint từ Hugging Face |
| Loại thuê | **On-demand**, không interruptible | Bị ngắt giữa chừng là mất lần đang chạy (không chạy tiếp được giữa chừng, GĐ5 L6) |
| Ảnh máy | Ubuntu 22.04, driver NVIDIA ≥ 520 (CUDA ≥ 11.8); ví dụ `nvidia/cuda:11.8.0-devel-ubuntu22.04` hoặc mẫu PyTorch của vast.ai | Môi trường tự cài torch riêng bằng micromamba |
| Vị trí | Không đặt ở Trung Quốc | Cần GitHub, Hugging Face, Zenodo |

Thời gian: bộ đầy đủ ~245 giờ GPU tính theo T4 (SFT ~4,3 giờ, GRPO ~5,6 giờ mỗi lần, cả chấm). Tốc độ GPU thuê so với T4 chỉ biết sau tầng 0; nếu nhanh hơn 2,5 lần thì 4 GPU mất ~25 giờ.

## 3. Người dùng làm

1. Thêm khoá SSH công khai của máy (`~/.ssh/id_ed25519.pub` hoặc `id_rsa.pub`) vào tài khoản vast.ai (Account → SSH Keys).
2. Thuê máy theo mục 2, đợi trạng thái Running.
3. Gửi Claude chuỗi kết nối SSH vast.ai hiển thị (dạng `ssh -p 12345 root@ssh5.vast.ai`). Claude không đọc khoá.
4. Khi xong: Claude chạy `cleanup --yes` (xoá SEA-Spoof, MLAAD, checkpoint); người dùng huỷ (destroy) máy trên vast.ai.

## 4. Claude làm (từ máy, `bash server/vast.sh ...`)

| Bước | Lệnh | Ghi chú |
|---|---|---|
| 1 | `init <host> <port>`, `check` | Ghi `server/host.env` (không commit), kiểm GPU, CPU, RAM, ổ |
| 2 | `code`, `data` | Chép `src/`, `protocols/ids/`, `server/`; SEA-Spoof vi (2,5 GB) và MLAAD vi (1,5 GB). SEA-Spoof không được phân phối lại: chỉ chép lên máy riêng đã thuê, xoá khi trả máy |
| 3 | `bootstrap`, `bootlog` | Dựng môi trường như Kaggle, toy example (EER phải 6,11%), tải RADAR/ItW đúng danh sách câu, kiểm dữ liệu, tải trước checkpoint |
| 4 | `start --tiers 0`, `status` | Thử đo; chốt số worker (`experiments.NUM_WORKERS`) và ước thời gian, báo người dùng |
| 5 | `start --tiers 1 2 3 4 5 [--workers N]` | Chạy nền (`setsid nohup`), không phụ thuộc SSH; chạy lại lệnh là chạy tiếp các lần chưa xong |
| 6 | `status`, `log <tên>` | Theo dõi: `results/status.txt` cập nhật mỗi phút |
| 7 | `fetch`, `collect <nhãn>` | Tải kết quả (không checkpoint) về `results/server/raw/`, gộp vào `results/server/`, bằng chứng vào `results/runs/server-<nhãn>/` |
| 8 | Phân tích | `src/analyze_ablation.py --finetune results/server/finetune --zeroshot results/server/zeroshot --out results/server/analysis` (tương tự `calibrate.py`, `analyze_mechanism.py`) |
| 9 | `cleanup --yes` | Trước khi trả máy |

## 5. Trên server

```
/workspace/orpheus/
  code/      src/*.py, src/*.sh, VERSION (commit git lúc chép)
  ids/       protocols/ids/ (danh sách câu)
  server/    experiments.py, run.py, bootstrap.sh, check_data.py
  work/      môi trường (micromamba: antideepfake, orpheus-driver), AntiDeepfake, fairseq
  cache/     checkpoint gốc
  data/      seaspoof/, mlaad/ (chép từ máy), prep/ (RADAR, In-the-Wild tải trên server)
  results/   train/, scores/, zeroshot/, ckpt/, jobs/<lần>.json|log, status.txt, driver.log, env/
```

Lần lỗi không tự chạy lại: xem `log <tên>`, sửa, rồi `start --tiers ... --retry-failed`. Dừng hết: `stop --yes`.
