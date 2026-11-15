Tiếp tục GĐ6 (GRPO và cải tiến) của đồ án: chạy bộ thí nghiệm đầy đủ trên server vast.ai (bộ chính), song song với Kaggle.

Đọc trước: chore/GD6-GRPO.md (mục 12 tổng hợp quyết định và phát hiện; mục 11 đánh giá lại hướng thực nghiệm; mục 9 bộ server; mục 10 kết quả Kaggle lần 1, 2; mục 3 lần thử đo; mục 7 phân tích không cần GPU); server/README.md (bộ 73 lần, cấu hình máy, các bước); server/experiments.py; chore/research-plan.md (GĐ6, câu hỏi 3 đã đặt lại); chore/Quy-Trinh-Chay-Kaggle.md (quy tắc quota, phiên OAuth).

Đã chốt (15/11):
- Câu hỏi 3 theo hướng "chẩn đoán và sửa": GRPO đúng code gốc không giữ tổng quát hoá tốt hơn SFT vì miền fine-tune (SEA-Spoof) đã bão hoà, chia phần thưởng cho độ lệch chuẩn của batch khiến GRPO tiếp tục làm sắc câu đã đúng (~30 lần SFT ở p = 0,999). Thiết kế bản GRPO sửa theo nguyên nhân và kiểm so với SFT, GRPO gốc: đây là phần "phân tích sâu + đề xuất cải tiến".
- Thí nghiệm sửa và kiểm cơ chế, mỗi cái 3 seed: G1s (rl_config std_floor 1.0, gần như bỏ chia cho độ lệch chuẩn), G1b (beta 1.0), G1lr (lr 1e-7, đối chứng cường độ), bản ghép không RawBoost của cách sửa tốt hơn (G1sA hoặc G1bA, so với A0); giữ G2, G3.
- Bộ server là bộ chính: 73 lần, mọi cấu hình 3 seed, cùng phần cứng, NVIDIA_TF32_OVERRIDE=0, công thức khớp Kaggle (lịch E1', batch 50, GRPO với expandable_segments). Bộ Kaggle là lần lặp độc lập trên T4: chỉ so ở mức "kết luận có giữ nguyên không", không trộn số. Kết quả server lưu riêng ở results/server/.
- Lối tắt tiếng thật một nguồn (VIVOS): ghi làm hạn chế và hướng phát triển, không làm trong đồ án.
- Thước đo chính: RADAR VI, ItW, RADAR EN (EER) kèm FRR ở ngưỡng 0; SEA-Spoof và MLAAD vẫn chấm và báo, không dùng để kết luận. So sánh giữa cấu hình theo seed (t-test, src/analyze_ablation.py mục 5), không kết luận bằng KTC bootstrap theo câu.

Trạng thái lúc chuyển (15/11 tối):
- Kaggle: lần 1 (G1 s1, G2 s1, orpheus-finetune v6) và lần 2 (R s4, s5, orpheus-finetune-b v2) xong, đã lưu bằng chứng và phân tích. Quota tuần này còn ~0,9 giờ (hết 7 giờ sáng 21/11); từ 21/11 có 30 giờ. kaggle/finetune/run.py có sẵn JOBS grpo2, grpo3, grpo4 (G1 s2–s5, G3). Việc dùng quota Kaggle tuần sau CHƯA chốt: hỏi tôi (lặp độc lập G1/R trên T4, hay chạy 1 seed các bản sửa để đối chiếu phần cứng).
- Server: script trong server/ đã xong và thử trên máy (chạy khô bộ điều phối, kiểm dữ liệu, gom kết quả, phân tích trên cây giả lập), CHƯA chạy trên server thật. Tôi sẽ thuê máy (2 × RTX 3090 hoặc theo server/README.md mục 2; nhớ chỉnh ổ ≥ 250 GB, on-demand) và gửi chuỗi SSH dạng `ssh -p <port> root@<host>`.

Việc cần làm, theo thứ tự:
1. Khi có chuỗi SSH: `bash server/vast.sh init <host> <port>`, `check` (đối chiếu GPU, CPU, RAM, ổ với server/README.md mục 2), `code`, `data`, `bootstrap`; theo dõi `bootlog` đến khi toy example ra EER 6,11%, check_data báo DỮ LIỆU OK, tải trước checkpoint xong. Báo tôi.
2. `start --tiers 0` (thử đo): đọc giây mỗi bước, đỉnh bộ nhớ (MEM_PROBE), GPU dùng (%), CPU với 2 và 6 worker; chốt số worker (nhỏ nhất cho GPU ≥ 90%, giữ nguyên cho cả bộ); ước thời gian tầng 1–6. Dừng lại báo tôi trước khi chạy cả bộ.
3. `start --tiers 1 2 3 4 5 6 --workers N`; theo dõi (`status`), báo khi xong mỗi tầng hoặc khi lỗi. Khi đủ 3 seed của G1s và G1b (tầng 2): so hai cách sửa (RADAR VI, ItW, FRR, độ tự tin trên dev), rồi `skip` bản ghép tầng 4 của cách kém hơn; hỏi tôi nếu chênh lệch không rõ.
4. Định kỳ `fetch` rồi `collect <nhãn>`, phân tích trên results/server/: `src/analyze_ablation.py --finetune results/server/finetune --zeroshot results/server/zeroshot --out results/server/analysis --group-pairs ...` (thêm G1s:R, G1b:R, G1s:G1, G1b:G1, G1lr:G1, G1sA:A0, G1bA:A0 vào cặp mặc định); `src/calibrate.py`, `src/analyze_mechanism.py` với cùng thư mục (SFT ít mẫu SFTn* cần --configs). Ghi kết quả và phát hiện vào chore/GD6-GRPO.md (mục mới, theo mẫu các mục 3, 10), phần kiến thức chung vào note/.
5. Khi xong: `fetch`, `collect`, `cleanup --yes`, rồi báo tôi huỷ máy trên vast.ai.

Lưu ý kỹ thuật:
- Python hệ thống của máy không có pandas, scipy: tạo venv riêng (ví dụ .venv, đã gitignore) với pandas, numpy, scipy để chạy các script phân tích.
- Quy trình server chưa thử trên máy thật: có thể lỗi ở apt, build fairseq, ảnh máy, mạng tới Zenodo/Hugging Face. Sửa, commit, `vast.sh code` lại; bootstrap.sh và run.py chạy lại được (bỏ qua bước đã xong).
- G1s cũng làm bước cập nhật ở câu ranh giới nhỏ hơn SFT (lực 0,25 so với 0,5): khi diễn giải, so thêm với G1lr để tách "đổi cơ chế" khỏi "giảm cường độ".
- Ý tưởng G1s gần với Dr. GRPO (bỏ chia cho độ lệch chuẩn): kiểm bài gốc trước khi trích vào báo cáo.

Ràng buộc:
- Bạn tự commit và push. GPU chạy trên Kaggle qua kaggle/push.sh và trên server qua server/vast.sh; tôi không chạy tay.
- Bạn theo dõi các lần chạy và tự xử lý tiếp khi xong. Báo tôi khi bắt đầu, xong, hoặc lỗi.
- Không đọc token (Kaggle, Hugging Face) hay khoá SSH. Phiên OAuth Kaggle CLI chỉ sống ~3 giờ; nếu hết hạn thì nhắn tôi đăng nhập lại.
- SEA-Spoof không được phân phối lại: âm thanh không đưa lên git; dataset và kernel Kaggle để riêng tư; trên server chỉ chép lên máy đã thuê và xoá trước khi trả máy.
- Mọi lần chạy GPU phải lưu bằng chứng vào results/runs/ (Kaggle: <kernel>-v<N>; server: server-<nhãn>).
- Số liệu lấy từ bài gốc hoặc code gốc.
- Ghi kết quả và phát hiện ngay khi có vào chore/GD6-GRPO.md; phần có tính kiến thức chung thì bổ sung vào note/.
- Chỉ tập trung kết quả thực nghiệm; không nhắc việc đọc hay học của tôi.
