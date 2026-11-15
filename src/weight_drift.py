#!/usr/bin/env python3
"""Độ trôi trọng số của mỗi checkpoint fine-tune so với checkpoint post-trained gốc (chore/GD6-GRPO.md mục 13.5, A2).

  python weight_drift.py --results <ROOT>/results --cache <ROOT>/cache

Chạy trên máy có checkpoint (server hoặc máy chạy bộ), TRƯỚC khi dọn checkpoint. Chỉ cần torch (CPU).

Với mỗi lần chạy có <results>/ckpt/<lần>.ckpt (bỏ lần khởi tạo từ fairseq: P0 không xuất phát từ checkpoint
post-trained):
  Δθ = θ_lần − θ_S0, θ_S0 là checkpoint post-trained của đúng backbone (cột `model` của train_summary.csv).
  Ghi <results>/drift/drift_by_group.csv: lần, backbone, nhóm tham số, số tham số, ‖Δθ‖, ‖θ_S0‖, ‖Δθ‖/‖θ_S0‖.
  Nhóm: cnn (bộ trích đặc trưng tích chập), L00..L23 (từng lớp transformer), ssl_khac (phần còn lại của
  front-end SSL), dau_ra (back-end phân loại), tong.
Cặp lần cùng backbone (mặc định: cùng seed giữa các cấu hình chính; cùng cấu hình giữa seed 1 và 2):
  cos(Δθ_A, Δθ_B) theo nhóm -> <results>/drift/drift_pairs.csv. Gần 1: cùng hướng cập nhật; gần 0: khác hướng.
Hỏi được: mức quên ngoài miền có theo ‖Δθ‖ bất kể SFT hay GRPO không; GRPO đi xa hơn hay đi hướng khác SFT.
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LAYER = re.compile(r"encoder\.layers\.(\d+)\.")
CODES = ("R", "G1", "A0", "G1s", "G1b", "G1lr", "Rlr", "SFTKL", "G1s0", "G2", "G3", "G1A", "G1sA", "G1bA")


def group_of(k):
    m = LAYER.search(k)
    if m:
        return f"L{int(m.group(1)):02d}"
    if "feature_extractor" in k:
        return "cnn"
    return "ssl_khac" if "m_ssl" in k or "ssl" in k.split(".")[0] else "dau_ra"


def load(path):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "state_dict" in st and isinstance(st["state_dict"], dict):
        st = st["state_dict"]
    return {k: v for k, v in st.items() if hasattr(v, "is_floating_point") and v.is_floating_point()}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_pairs(runs):
    out = []
    for s in range(1, 6):
        def has(c):
            return f"{c}_s{s}" in runs
        for a, b in (("R", "G1"), ("R", "A0"), ("G1", "G1s"), ("G1", "G1b"), ("R", "G1s"), ("R", "SFTKL"), ("G1s", "SFTKL"),
                     ("R", "Rlr"), ("G1", "G1lr"), ("G1", "G2"), ("G1", "G3"), ("G1s", "G1s0"), ("A0", "G1A")):
            if has(a) and has(b):
                out.append((f"{a}_s{s}", f"{b}_s{s}"))
    for c in CODES:
        if f"{c}_s1" in runs and f"{c}_s2" in runs:
            out.append((f"{c}_s1", f"{c}_s2"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="<ROOT>/results (có ckpt/, train_summary.csv)")
    ap.add_argument("--cache", required=True, help="<ROOT>/cache (checkpoint post-trained gốc)")
    ap.add_argument("--runs", nargs="*", default=None, help="mặc định: mọi <results>/ckpt/*.ckpt")
    ap.add_argument("--no-pairs", action="store_true")
    a = ap.parse_args()
    import torch
    import score_zeroshot as sz

    info = {r["run"]: r for r in csv.DictReader(open(os.path.join(a.results, "train_summary.csv")))} \
        if os.path.exists(os.path.join(a.results, "train_summary.csv")) else {}
    ckpts = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(a.results, "ckpt", "*.ckpt"))}
    runs = sorted(a.runs or ckpts)
    out_dir = os.path.join(a.results, "drift")
    os.makedirs(out_dir, exist_ok=True)
    base_cache, rows, model_of = {}, [], {}

    def base(model):
        if model not in base_cache:
            base_cache.clear()                     # giữ một backbone trong RAM
            path = os.path.join(a.cache, sz.MODELS[model][2])
            if not os.path.exists(path):
                raise SystemExit(f"Không thấy checkpoint gốc {path}")
            base_cache[model] = load(path)
        return base_cache[model]

    for run in runs:
        r = info.get(run, {})
        if r.get("init") == "fairseq":
            log(f"bỏ {run}: khởi tạo từ fairseq, không cùng điểm xuất phát với checkpoint post-trained")
            continue
        model = r.get("model") or "mms_300m"
        if run not in ckpts:
            log(f"bỏ {run}: không có checkpoint")
            continue
        B, A = base(model), load(ckpts[run])
        if set(A) != set(B):
            log(f"bỏ {run}: khác tên khoá với checkpoint gốc ({len(set(A) ^ set(B))} khoá lệch)")
            continue
        acc = {}
        for k, v in A.items():
            d = v.double() - B[k].double()
            g = group_of(k)
            for gg in (g, "tong"):
                n, d2, b2 = acc.get(gg, (0, 0.0, 0.0))
                acc[gg] = (n + v.numel(), d2 + float((d * d).sum()), b2 + float((B[k].double() ** 2).sum()))
        for g, (n, d2, b2) in sorted(acc.items()):
            rows.append({"run": run, "model": model, "group": g, "n_params": n, "l2_delta": d2 ** 0.5,
                         "l2_base": b2 ** 0.5, "rel": (d2 / b2) ** 0.5 if b2 else ""})
        model_of[run] = model
        log(f"{run} ({model}): ‖Δθ‖ = {acc['tong'][1] ** 0.5:.4f} ({100 * (acc['tong'][1] / acc['tong'][2]) ** 0.5:.4f}% ‖θ_S0‖)")
        del A
    with open(os.path.join(out_dir, "drift_by_group.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "model", "group", "n_params", "l2_delta", "l2_base", "rel"])
        w.writeheader()
        w.writerows(rows)
    log(f"Ghi {out_dir}/drift_by_group.csv ({len(model_of)} lần)")

    if a.no_pairs:
        return
    prs = [(x, y) for x, y in default_pairs(set(model_of)) if model_of[x] == model_of[y]]
    pr_rows = []
    for x, y in prs:
        B = base(model_of[x])
        X, Y = load(ckpts[x]), load(ckpts[y])
        acc = {}
        for k in B:
            dx, dy = X[k].double() - B[k].double(), Y[k].double() - B[k].double()
            for gg in (group_of(k), "tong"):
                xy, xx, yy = acc.get(gg, (0.0, 0.0, 0.0))
                acc[gg] = (xy + float((dx * dy).sum()), xx + float((dx * dx).sum()), yy + float((dy * dy).sum()))
        for g, (xy, xx, yy) in sorted(acc.items()):
            pr_rows.append({"A": x, "B": y, "group": g, "cos": xy / (xx * yy) ** 0.5 if xx and yy else "",
                            "l2_A": xx ** 0.5, "l2_B": yy ** 0.5})
        log(f"cos(Δθ {x}, Δθ {y}) = {acc['tong'][0] / (acc['tong'][1] * acc['tong'][2]) ** 0.5:.3f}")
        del X, Y
    with open(os.path.join(out_dir, "drift_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["A", "B", "group", "cos", "l2_A", "l2_B"])
        w.writeheader()
        w.writerows(pr_rows)
    log(f"Ghi {out_dir}/drift_pairs.csv ({len(prs)} cặp)")
    _ = torch


if __name__ == "__main__":
    main()
