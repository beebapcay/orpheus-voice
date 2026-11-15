#!/usr/bin/env python3
"""WiSE-FT (Wortsman và cộng sự, "Robust fine-tuning of zero-shot models", CVPR 2022, arXiv 2109.01903):
nội suy trọng số giữa mô hình chưa fine-tune và bản fine-tune từ chính nó,
    θ_α = (1 − α)·θ_S0 + α·θ_FT,
rồi chấm θ_α như một checkpoint sau fine-tune (score_zeroshot.py --local). Không huấn luyện thêm.

  python interp_score.py --base B --data-root D1 D2 D3 --out O --cache C --gpu 0 \\
      --name Rw50_s1 --src R_s1 --alpha 0.5 --model mms_300m --score seaspoof_vi_dev radar_vi ...
  python interp_score.py ... --check-only      chỉ kiểm hai checkpoint khớp tên khoá, kích thước (không chấm)

θ_S0: checkpoint post-trained trong cache (như lúc khởi tạo fine-tune); θ_FT: <out>/ckpt/<src>.ckpt (finetune.py).
Hai file là state_dict cùng tên khoá (finetune.py lấy ssl.ckpt của SpeechBrain); khác tên khoá hay kích thước thì
dừng. Tensor không phải số thực (nếu có) lấy theo θ_FT. File θ_α (~1,3 GB) xoá sau khi chấm; thống kê nội suy
(‖θ_FT − θ_S0‖, ‖θ_S0‖) ghi vào <out>/train/<name>/interp.json để collect.py mang về.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_zeroshot as sz  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(path):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "state_dict" in st and isinstance(st["state_dict"], dict):
        st = st["state_dict"]
    return st


def interpolate(a_path, b_path, alpha, out_path=None):
    """θ_α = (1 − α)·A + α·B theo từng tensor; trả về thống kê. out_path=None: chỉ kiểm, không ghi."""
    import torch
    A, B = load_state(a_path), load_state(b_path)
    ka, kb = set(A), set(B)
    if ka != kb:
        raise SystemExit(f"Hai checkpoint khác tên khoá: chỉ ở S0 {sorted(ka - kb)[:5]}, chỉ ở FT {sorted(kb - ka)[:5]}")
    out, n_float, n_other, d2, a2 = {}, 0, 0, 0.0, 0.0
    for k in A:
        a, b = A[k], B[k]
        if tuple(a.shape) != tuple(b.shape):
            raise SystemExit(f"Khác kích thước ở {k}: {tuple(a.shape)} và {tuple(b.shape)}")
        if torch.is_floating_point(a):
            ad, bd = a.double(), b.double()
            if out_path:
                out[k] = ((1 - alpha) * ad + alpha * bd).to(b.dtype)
            d2 += float(((bd - ad) ** 2).sum())
            a2 += float((ad ** 2).sum())
            n_float += a.numel()
        else:
            out[k] = b.clone()
            n_other += 1
    if out_path:
        torch.save(out, out_path + ".part")
        os.replace(out_path + ".part", out_path)
    return {"alpha": alpha, "n_tensors": len(A), "n_params_float": n_float, "n_tensors_nonfloat": n_other,
            "l2_ft_minus_s0": d2 ** 0.5, "l2_s0": a2 ** 0.5, "rel_drift": (d2 / a2) ** 0.5 if a2 else None,
            "l2_alpha_minus_s0": alpha * d2 ** 0.5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-root", required=True, nargs="+")
    ap.add_argument("--out", required=True, help="thư mục kết quả của server (có ckpt/, scores/, train/)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", required=True, help="tên lần chấm, ví dụ Rw50_s1")
    ap.add_argument("--src", required=True, help="lần fine-tune nguồn, ví dụ R_s1 (<out>/ckpt/<src>.ckpt)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--model", default="mms_300m", help="checkpoint post-trained gốc (khoá MODELS)")
    ap.add_argument("--score", nargs="*", default=[])
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    ft = os.path.join(a.out, "ckpt", f"{a.src}.ckpt")
    if not os.path.exists(ft):
        sys.exit(f"Không thấy {ft}")
    s0 = sz.Checkpoints(a.cache, {a.model: 1}, keep=True).get(a.model)
    if a.check_only:
        log(f"Kiểm {s0} và {ft}: {json.dumps(interpolate(s0, ft, a.alpha), ensure_ascii=False)}")
        return
    run_dir = os.path.join(a.out, "train", a.name)
    os.makedirs(run_dir, exist_ok=True)
    out_ckpt = os.path.join(a.out, "ckpt", f"{a.name}.ckpt")
    t0 = time.time()
    stats = interpolate(s0, ft, a.alpha, out_ckpt)
    stats.update({"name": a.name, "src": a.src, "model": a.model, "s0_ckpt": s0, "ft_ckpt": ft,
                  "interp_seconds": round(time.time() - t0)})
    json.dump(stats, open(os.path.join(run_dir, "interp.json"), "w"), indent=1, ensure_ascii=False)
    log(f"[{a.name}] nội suy α = {a.alpha}: ‖θ_FT − θ_S0‖ = {stats['l2_ft_minus_s0']:.3f} "
        f"({100 * stats['rel_drift']:.3f}% của ‖θ_S0‖); {stats['interp_seconds']} s")
    rc = 0
    if a.score:
        cmd = [sys.executable, os.path.join(HERE, "score_zeroshot.py"), "--base", a.base, "--data-root", *a.data_root,
               "--out", os.path.join(a.out, "scores"), "--cache", a.cache, "--gpus", str(a.gpu), "--keep-ckpt",
               "--local", f"{a.name}={out_ckpt},{a.model}", "--jobs", *[f"{d}:{a.name}" for d in a.score]]
        log(f"[{a.name}] chấm: {' '.join(cmd[2:])}")
        rc = subprocess.run(cmd).returncode
    os.remove(out_ckpt)          # ~1,3 GB; tái tạo được từ hai checkpoint gốc và α
    sys.exit(rc)


if __name__ == "__main__":
    main()
