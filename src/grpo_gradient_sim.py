#!/usr/bin/env python
"""Kiểm bằng Monte Carlo kỳ vọng gradient của loss GRPO trong AntiDeepfake (`algo/rl.py`, `grpo3_loss`)
cho bộ phân loại hai lớp, so với SFT (cross-entropy). Chạy trên máy, chỉ cần numpy:

  python src/grpo_gradient_sim.py

Giả định: tại thời điểm mô hình "cũ" vừa được cập nhật (tỉ số chính sách = 1, clip không tác dụng),
bỏ số hạng KL (β = 0). Phần thưởng 1 nếu nhãn lấy mẫu đúng, 0 nếu sai; chuẩn hoá (r − trung bình)/(độ
lệch chuẩn + 1e-5) trên **cả batch × G mẫu** như code (không theo từng câu như công thức (4) của bài).
z là log-odds của nhãn đúng, p = sigmoid(z) = p(y_đúng | x).

Suy ra (cột "dự đoán"): E[∂(−loss)/∂z] = p(1 − p)/σ_batch với GRPO; p(1 − p) với biến thể không có
phần thưởng âm (`grpo3_degraded_ref`); SFT: 1 − p. Tất cả chia cho B (loss lấy trung bình trên batch).
"""
import numpy as np
rng = np.random.default_rng(0)
G, B, R = 64, 10, 4000
p = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.999])   # p(y_đúng|x) của 10 câu
g_grpo, g_neg, sig = np.zeros(B), np.zeros(B), []
for _ in range(R):
    correct = rng.random((B, G)) < p[:, None]          # nhãn lấy mẫu trùng nhãn đúng?
    r = correct.astype(float)
    A = (r - r.mean()) / (r.std(ddof=1) + 1e-5)        # chuẩn hoá trên cả batch (torch.std: không chệch)
    sig.append(r.std(ddof=1))
    dlogp_dz = np.where(correct, 1 - p[:, None], -p[:, None])
    # loss = -mean(A * ratio); d ratio/dz = d logp/dz khi ratio = 1. Gradient (để giảm loss) theo hướng +z:
    g_grpo += (A * dlogp_dz).sum(1) / (B * G)
    g_neg += (r * dlogp_dz).sum(1) / (B * G)
g_grpo /= R; g_neg /= R; s = np.mean(sig)
print(f"sigma_batch trung bình {s:.3f}")
print("p      GRPO_mô_phỏng  p(1-p)/σ/B   w/o_neg_mô_phỏng  p(1-p)/B   SFT (1-p)/B")
for i in range(B):
    print(f"{p[i]:.3f}  {g_grpo[i]:+.5f}      {p[i]*(1-p[i])/s/B:+.5f}     {g_neg[i]:+.5f}          {p[i]*(1-p[i])/B:+.5f}   {(1-p[i])/B:+.5f}")
