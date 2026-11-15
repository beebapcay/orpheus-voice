# Hướng dẫn chạy bộ thí nghiệm trên máy của bạn

Cảm ơn bạn đã giúp chạy. Bộ này fine-tune lại một mô hình phát hiện giọng nói giả (~300 triệu tham số) khoảng 100 lần, mỗi lần một cấu hình, rồi chấm điểm trên vài tập kiểm tra. Mọi thứ chạy tự động bằng vài lệnh; bạn chỉ cần dựng máy, khởi động, thỉnh thoảng xem trạng thái và gửi lại file kết quả.

## 1. Máy cần có

| Thứ | Yêu cầu |
|---|---|
| Hệ điều hành | Linux x86_64 (Ubuntu 20.04/22.04 là chắc chạy), có `sudo` để cài gói (git, curl, wget, bzip2, build-essential, rsync, tmux) |
| GPU | NVIDIA, **≥ 24 GB mỗi GPU, mọi GPU cùng loại** (RTX 3090/4090, A5000/A6000, A100, L40S…). Không dùng RTX 50xx. Driver hỗ trợ CUDA ≥ 11.8 (`nvidia-smi` báo CUDA Version ≥ 11.8) |
| CPU, RAM | ≥ 8 luồng CPU và ≥ 16 GB RAM cho mỗi GPU |
| Ổ trống | **~256 GB** ở thư mục chạy (mặc định `~/orpheus-run`) |
| Mạng | Vào được GitHub, Hugging Face, Zenodo, download.pytorch.org, conda-forge |
| Thời gian | Với 2 × RTX 3090: ~3–4 ngày chạy liên tục. 1 GPU: gấp đôi (báo chủ đồ án trước khi chạy) |

Trong suốt thời gian chạy: không dùng các GPU này cho việc khác, không để máy ngủ hay tắt.

## 2. Quy tắc dữ liệu (quan trọng)

- Bộ dữ liệu SEA-Spoof có giấy phép **không cho phân phối lại**, MLAAD chỉ dùng phi thương mại. Không chép, không chia sẻ âm thanh của hai bộ này cho ai.
- Khi xong việc, **xoá dữ liệu** bằng lệnh ở bước 9.
- File kết quả gửi lại (bước 8) chỉ có điểm số, log, cấu hình; không có âm thanh hay mô hình.

## 3. Lấy code

Chủ đồ án mời bạn vào repo GitHub, rồi:

```bash
git clone git@github.com:beebapcay/orpheus-voice.git ~/orpheus-voice
cd ~/orpheus-voice
```

Mọi lệnh dưới đây chạy **trong thư mục `~/orpheus-voice`**, nên mở trong `tmux` để không mất khi đóng cửa sổ (`tmux new -s orpheus`; vào lại: `tmux attach -t orpheus`).

Thư mục chạy (dữ liệu, môi trường, kết quả) mặc định là `~/orpheus-run`. Nếu ổ chứa thư mục home không đủ ~256 GB, chọn chỗ khác và đặt **trước mọi lệnh** (nên thêm dòng này vào `~/.bashrc`):

```bash
export ORPHEUS_ROOT=/duong/dan/o/lon/orpheus-run
```

## 4. Dựng máy (~1 giờ)

```bash
bash server/local.sh check       # xem GPU, CPU, RAM, ổ trống: gửi kết quả cho chủ đồ án
bash server/local.sh setup       # tạo thư mục chạy, chép code
bash server/local.sh env         # cài môi trường, tải dữ liệu kiểm tra (~30–60 phút)
```

`env` kết thúc bằng dòng "Dựng môi trường và dữ liệu kiểm tra xong". Trong log có hai dòng cần thấy: "Toy example: EER 6,11%" và "TF32: NVIDIA_TF32_OVERRIDE=0 cho kết quả fp32". Nếu thấy dòng CẢNH BÁO thì dừng, gửi chủ đồ án `~/orpheus-run/results/env/bootstrap.log`.

## 5. Dữ liệu fine-tune (chọn một cách)

**Cách A: bạn tự tạo (khuyến nghị).**
1. Tạo tài khoản Hugging Face; vào trang dataset `Jack-ppkdczgx/SEA-Spoof` bấm xin quyền (tác giả duyệt, có thể mất vài ngày: làm việc này **sớm nhất có thể**); vào `mueller91/MLAAD` chấp nhận điều khoản (duyệt tự động).
2. Tạo token đọc (Settings → Access Tokens → Read) rồi:

```bash
HF_TOKEN=hf_xxxxxxxx bash server/local.sh fetch-data
```

**Cách B: chủ đồ án tự chép dữ liệu lên máy** (nếu bạn cho chủ đồ án truy cập máy), rồi bạn chạy `bash server/local.sh copy-data <thư mục SEA-Spoof> <thư mục MLAAD>` theo đường dẫn chủ đồ án đưa.

Sau đó kiểm dữ liệu và tải trước mô hình gốc:

```bash
bash server/local.sh bootstrap
```

Phải thấy **"DỮ LIỆU OK"** và dòng cuối "Dựng xong". Nếu "DỮ LIỆU LỖI", gửi chủ đồ án `~/orpheus-run/results/env/check_data.txt`.

## 6. Thử đo (~30–40 phút)

```bash
bash server/local.sh probe
bash server/local.sh pack        # tạo ~/orpheus-results-<ngày-giờ>.tar.gz
```

`probe` chạy 5 lần ngắn để đo tốc độ, bộ nhớ, và in **"Số worker khuyến nghị: N"**. Gửi file `.tar.gz` cho chủ đồ án và chờ trả lời "chạy tiếp" (thường trong vài giờ).

## 7. Chạy cả bộ (~3–4 ngày, chạy nền)

```bash
bash server/local.sh start       # dùng số worker khuyến nghị ở bước 6
```

Lệnh chạy nền, đóng terminal vẫn chạy. Xem tình hình:

```bash
bash server/local.sh status      # tiến độ, GPU, số lần xong/lỗi, ổ trống
```

| Khi | Làm |
|---|---|
| Mỗi ngày một lần (hoặc khi chủ đồ án nhắn) | `bash server/local.sh pack` rồi gửi file `.tar.gz` |
| `status` có dòng **"LỖI: <tên>"** | Gửi chủ đồ án kết quả `bash server/local.sh status` và `bash server/local.sh log <tên> 80`. Các lần khác vẫn chạy tiếp; không cần dừng |
| Chủ đồ án nhắn "cập nhật code" | `git pull && bash server/local.sh setup` (không cần dừng; lần chạy sau tự dùng code mới) |
| Phải tắt máy gấp | `bash server/local.sh stop --yes`; khi bật lại: `bash server/local.sh start` (lần đã xong không chạy lại; lần đang chạy dở chạy lại từ đầu) |

Không tự sửa code, không xoá gì trong `~/orpheus-run/results`.

## 8. Khi chạy xong

`status` báo "điều phối KHÔNG chạy" và không còn lần nào chờ. Khi đó:

```bash
bash server/local.sh drift       # đo độ thay đổi trọng số của mọi mô hình (~30–60 phút)
bash server/local.sh pack        # gửi file .tar.gz cuối cho chủ đồ án
```

## 9. Dọn máy (chỉ sau khi chủ đồ án xác nhận đã nhận đủ file)

```bash
bash server/local.sh cleanup --yes      # xoá SEA-Spoof, MLAAD, mô hình đã huấn luyện, mô hình gốc
rm -rf ~/orpheus-run                    # (tuỳ bạn) xoá hết thư mục chạy
```
