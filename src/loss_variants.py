#!/usr/bin/env python3
"""Chạy main.py gốc của AntiDeepfake với một hàm loss thay thế, KHÔNG sửa code gốc (runpy, như mem_probe.py).

  python loss_variants.py <biến thể> [mem_probe.py] main.py <tham số của main.py>
  (cwd = thư mục repo AntiDeepfake, như `python main.py ...`; src/finetune.py --loss-variant tự ghép lệnh)

Biến thể:
  sft_kl  Cross-entropy trên nhãn thật, đúng hàm loss của SFT (models/W2V.py get_loss: F.cross_entropy), cộng
          β·KL với đúng số hạng KL của GRPO (algo/rl.py grpo3_loss, grpo2_loss: ước lượng k3 = e^r − r − 1,
          r = log π_ref(ỹ) − log π_θ(ỹ), trên G nhãn ỹ lấy mẫu từ mô hình hiện tại; β = rl_config['beta']).
          Dùng với YAML grpo_v2 (algo: grpo2: có ref_detector, không có old_detector): thay rl.grpo2_loss,
          nên đường dữ liệu, mô hình tham chiếu, lấy mẫu nhãn, optimizer giữ nguyên như G2. Kỳ vọng gradient
          của số hạng KL theo log-odds nhãn đúng là β(p − p_ref) (chore/GD6-GRPO.md mục 13.3).

main.py gọi `rl.grpo2_loss(...)` qua module `algo.rl` (`from algo import rl`), nên vá thuộc tính của module
trong sys.modules trước khi chạy main.py là đủ.
"""
import os
import runpy
import sys

VARIANTS = ("sft_kl",)


def sft_kl_loss(logits, logits_ref, sa_labels, gt_labels, rl_config):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, gt_labels)                     # như SFT (get_loss)
    samp_num = sa_labels.shape[-1]
    logits_ = torch.repeat_interleave(logits, samp_num, dim=0)  # như grpo2_loss / grpo3_loss
    logits_ref_ = torch.repeat_interleave(logits_ref, samp_num, dim=0)
    sa_ = torch.flatten(sa_labels)
    logps = torch.gather(F.log_softmax(logits_, dim=-1), dim=1, index=sa_.unsqueeze(1)).squeeze(1)
    logps_ref = torch.gather(F.log_softmax(logits_ref_, dim=-1), dim=1, index=sa_.unsqueeze(1)).squeeze(1)
    kl = torch.exp(logps_ref - logps) - (logps_ref - logps) - 1
    beta = rl_config['beta'] if 'beta' in rl_config else 1.0     # như grpo3_loss
    return ce + beta * kl.mean()


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in VARIANTS:
        sys.exit(f"Cách dùng: loss_variants.py <{'|'.join(VARIANTS)}> [mem_probe.py] main.py ...")
    variant, rest = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, os.getcwd())          # repo AntiDeepfake: import đúng algo/rl.py của repo
    from algo import rl
    if variant == "sft_kl":
        rl.grpo2_loss = sft_kl_loss
        print("LOSS_VARIANT sft_kl: algo.rl.grpo2_loss -> cross-entropy(nhãn thật) + beta*KL k3 (nhãn lấy mẫu)",
              flush=True)
    script = os.path.abspath(rest[0])
    sys.argv = rest
    sys.path[0] = os.path.dirname(script)    # như `python <script>` (mem_probe.py tự đặt lại cho main.py)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
