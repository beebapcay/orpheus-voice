# Chạy bộ thí nghiệm trên server thuê (vast.ai)

**Chạy trên máy của người khác (không qua vast.ai, người đó tự gõ lệnh):** `server/local.sh`, hướng dẫn cho người chạy: `server/HUONG-DAN-CHAY.md`. Kết quả gửi về (`local.sh pack`) nhận bằng `bash server/vast.sh import <file.tar.gz>` rồi `collect` như thường.

Một bộ thí nghiệm **hoàn chỉnh** chạy trên một loại máy, để mọi phép so sánh trong báo cáo dùng kết quả cùng phần cứng. Bộ Kaggle (T4) vẫn chạy song song và giữ làm lần lặp độc lập trên phần cứng khác. Lý do và danh sách đầy đủ: `chore/GD6-GRPO.md` mục 9; danh sách lần chạy: `server/experiments.py`.

## 1. Bộ thí nghiệm (122 lần; người dùng chốt 15/11: làm đầy đủ, chia giai đoạn có điểm kiểm, `chore/GD6-GRPO.md` mục 14)

| Tầng | Giai đoạn | Nội dung | Lần chạy |
|---|---|---|---|
| 0 | A | Thử đo: tốc độ, bộ nhớ, số worker (2 hoặc 6) cho SFT và G1, 200 bước; dấu vân tay batch; SFT + KL thử 100 bước | 5 (ngắn) |
| 1 | B | Lõi: S0 (dùng thẳng: MMS-300M, MMS-300M NDA, W2V-Large, W2V-Small), R, G1, A0 × 3 seed | 4 + 9 |
| 2 | C | Chẩn đoán và sửa × 3 seed: G1s (`std_floor` 1.0), G1b (β = 1); G1lr, Rlr (lr 1e-7: cường độ); G3 (β = 0); SFTKL (cross-entropy + KL k3 của GRPO, β 0,04), G1s0 (G1s với β = 0); G2 | 24 |
| 3 | D | WiSE-FT: nội suy S0 ↔ bản fine-tune rồi chấm (`src/interp_score.py`, chờ lần nguồn xong): R, A0, G1, G1s, G1b; α 0,5 cả 3 seed, α 0,25 và 0,75 seed 1 | 25 (chỉ chấm) |
| 4 | D | Không RawBoost × 3 seed: G1sA, G1bA (bản sửa ghép), G1A (G1 gốc) | 9 |
| 5 | D | Seed 4, 5: R, G1, G1s, G1b | 8 |
| 6 | E | Các yếu tố GĐ5: P0, B2, N0, B1 × 3 seed | 12 |
| 7 | E | Kiểm chế độ D1: fine-tune trên nửa RADAR VI (`src/split_radar_regime.py`), R, G1 × 3 seed; S0 trên hai tập chia | 7 |
| 8 | E | Ít mẫu: SFT và G1 với N = 50 / 200 / 1000 mỗi lớp × 3 seed, cùng 4.563 bước; bản sửa tốt nhất thêm ở điểm kiểm C | 18 (+9) |
| 9 | E | Mốc XLS-R-2B dùng thẳng trên RADAR VI | 1 |

Công thức giữ nguyên như Kaggle (lịch E1', batch 50, GRPO với `expandable_segments`), thêm `NVIDIA_TF32_OVERRIDE=0` (SpeechBrain 1.0.2 tự bật TF32 khi import; biến này ghi đè ở mức cuBLAS/cuDNN, `bootstrap.sh` kiểm bằng số). Mỗi lần huấn luyện chấm trên 5 tập: `seaspoof_vi_dev`, `seaspoof_vi_test`, `radar_vi`, `ood_forget`, `mlaad_vi_test` (D1: `radar_vi_rv_dev`, `radar_vi_rv_test` thay cho dev và RADAR VI).

**Hàng đợi động:** `run.py` đọc lại `experiments.py` mỗi lần một GPU nhận việc, nên thêm, bớt lần chạy ở điểm kiểm (sửa file, `vast.sh code`) có hiệu lực ngay, không dừng lần đang chạy. Lần có `needs` (WiSE-FT) chờ lần nguồn xong; lần nguồn lỗi hoặc bị bỏ thì lần phụ thuộc thành "skipped".

## 2. Cấu hình máy cần thuê

| Thứ | Yêu cầu | Lý do |
|---|---|---|
| GPU | **4 × cùng loại**, ≥ 24 GB: RTX 4090, RTX 3090, A5000, A6000, L40S, A100, H100 | G1 cần ~14 GB (đo trên T4). Nhiều GPU thì xong sớm; các GPU cùng loại để cả bộ cùng phần cứng |
| Không dùng | RTX 50xx (Blackwell) | torch 2.6.0 bản CUDA 11.8 của môi trường gốc không hỗ trợ |
| CPU | ≥ 8 nhân mỗi GPU (32 cho 4 GPU) | RawBoost và đọc âm thanh chạy trên CPU; GĐ5 đã thấy CPU là nút thắt với 4 nhân cho 2 GPU |
| RAM | ≥ 16 GB mỗi GPU (64 GB cho 4 GPU) | Mỗi lần nạp 1–3 bản mô hình và tiến trình đọc dữ liệu |
| Ổ đĩa | **256 GB** (đỉnh ước ~200–225 GB; bảng dưới) | Phần lớn là ~95 checkpoint sau fine-tune (~118 GB), giữ đến cuối để chạy WiSE-FT, đo độ trôi trọng số và chấm lại khi cần |
| Mạng | Tải xuống nhanh (≥ 500 Mbps nếu có) | RADAR 8,75 GB từ Zenodo, checkpoint từ Hugging Face |
| Loại thuê | **On-demand**, không interruptible | Bị ngắt giữa chừng là mất lần đang chạy (không chạy tiếp được giữa chừng, GĐ5 L6) |
| Ảnh máy | Ubuntu 22.04, driver NVIDIA ≥ 520 (CUDA ≥ 11.8); ví dụ `nvidia/cuda:11.8.0-devel-ubuntu22.04` hoặc mẫu PyTorch của vast.ai | Môi trường tự cài torch riêng bằng micromamba |
| Vị trí | Không đặt ở Trung Quốc | Cần GitHub, Hugging Face, Zenodo |

**Ước dung lượng ổ (15/11, theo số đo Kaggle):**

| Phần | GB | Căn cứ |
|---|---|---|
| Ảnh máy (vast.ai tính vào ổ được cấp) | ~10–15 | Ảnh CUDA của vast.ai (ước) |
| Môi trường và cache cài đặt (`work/`, pip, micromamba) | ~12–15 | torch 2.6.0 cu118 (~2,3 GB tải, ~5 GB cài), fairseq, SpeechBrain; môi trường điều phối ~2 GB |
| Dữ liệu | ~6,5 | SEA-Spoof vi 2,5 và MLAAD vi 1,5 (đo trên máy); RADAR VI, RADAR EN tập con, In-the-Wild tập con ~2,5 (output prep Kaggle 9,96 GB trừ In-the-Wild đầy đủ ~8,2 GB) |
| Checkpoint gốc (`cache/`) | ~16,7 | MMS-300M, bản NDA, W2V-Large 1,27 mỗi bản; W2V-Small 0,38; XLS-R-2B 8,65; fairseq base_300m 3,81 (lúc chuyển XLS-R-2B thêm 8,65 tạm thời, ở đầu bộ) |
| Checkpoint sau fine-tune (`results/ckpt/`) | ~118 | 92 × 1,27 GB (đo: 1.270 MB) + 3 × 0,38 (B1); file nội suy WiSE-FT xoá sau khi chấm |
| Đang huấn luyện (2 lần cùng lúc) | ~8 | Mỗi lần: `save/` của SpeechBrain gồm mô hình 1,27 và trạng thái AdamW ~2,5; xoá sau khi chép checkpoint |
| Embedding | ~20 | ~125 lần chấm × `emb.npy` ~160 MB (đo trên output Kaggle); bản sao theo shard (~20 GB nữa) `run.py` xoá sau mỗi lần chấm xong |
| Điểm, log | ~3 | |
| **Đỉnh (cuối bộ)** | **~195–205** | Còn ~50–60 GB trong 256 GB. `run.py` dừng nhận việc mới khi ổ còn < 25 GB (báo trong `status.txt`) |

Thời gian (tính theo T4; SFT ~4,3 giờ, GRPO ~5,6 giờ mỗi lần, cả chấm; chấm riêng ~1,7 giờ): cả bộ ~530 giờ. Tốc độ GPU thuê so với T4 chỉ biết sau tầng 0; nhanh hơn T4 2,5–3,5 lần thì 2 GPU mất ~3,2–4,4 ngày, 4 GPU ~1,6–2,2 ngày. vast.ai tính theo giờ GPU, nên 4 GPU tốn gần bằng 2 GPU mà xong sớm gấp đôi.

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
| 4 | `start --tiers 0`, `status`; sau đó `fetch` và `python3 server/probe_report.py --train results/server/raw/train --summary results/server/raw/train_summary.csv` | Thử đo; chốt số worker (`experiments.NUM_WORKERS`) và ước thời gian, báo người dùng. `probe_report.py` tính GPU, CPU dùng chỉ trong khoảng huấn luyện (`gpu_util_mean` của `train_summary.csv` gồm cả lúc nạp mô hình, thấp hơn thực tế 5–10 điểm ở lần đo ngắn), và so dấu vân tay batch (`BATCH_HASH_N`): cùng seed, cùng số worker thì R và G1 có nhận cùng batch, cùng phép tăng cường RawBoost không (quyết định ghép cặp theo seed khi phân tích, `chore/GD6-GRPO.md` mục 13) |
| 5 | `start --workers N` (mọi tầng ≥ 1) | Chạy nền (`setsid nohup`), không phụ thuộc SSH; chạy lại lệnh là chạy tiếp các lần chưa xong |
| 6 | `status`, `log <tên>`, `skip <tên>...`, `hold`, `release` | Theo dõi: `results/status.txt` cập nhật mỗi phút; `skip` bỏ qua lần đang chờ; `hold` để GPU rảnh không nhận việc mới (lần đang chạy vẫn chạy), `release` nhận lại. Điểm kiểm A–E: ghi mục mới trong `chore/GD6-GRPO.md` |
| 7 | `fetch`, `collect <nhãn>` | Tải kết quả (không checkpoint) về `results/server/raw/`, gộp vào `results/server/`, bằng chứng vào `results/runs/server-<nhãn>/` |
| 8 | Phân tích | `bash server/analyze.sh` (trên máy, cần `.venv` có pandas, numpy, scipy): chạy `analyze_ablation.py` (kèm các cặp so qua seed của bản sửa), `calibrate.py`, `analyze_mechanism.py` trên `results/server/`, ghi `results/server/analysis/`, `results/server/calibration/` |
| 9 | Đo độ trôi trọng số trên mọi checkpoint (trước `cleanup`) | Checkpoint chỉ có trên server |
| 10 | `cleanup --yes` | Trước khi trả máy |

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

Lần lỗi không tự chạy lại: xem `log <tên>`, sửa, rồi xoá `results/jobs/<tên>.json` (điều phối đang chạy sẽ nhận lại; không cần khởi động lại). Dừng hết, kể cả lần đang chạy: `stop --yes`.
