#!/usr/bin/env python
"""Khoảng tin cậy bootstrap cho EER, và cho chênh lệch EER giữa hai mô hình (bootstrap ghép cặp).

Lấy mẫu lại theo câu, có hoàn lại, riêng trong từng lớp (thật / giả) để giữ số câu mỗi lớp.
Mọi mô hình dùng chung một bộ chỉ số ở mỗi lượt, nên chênh lệch A − B là ghép cặp (cùng câu).
EER tính bằng `compute_eer` của `src/evaluate.py` (cùng cách tính với các bảng kết quả).

Ví dụ:
  python src/bootstrap_eer.py --root results/zeroshot --dataset radar_vi \
      --pairs mms_300m:mms_300m_nda w2v_large:mms_300m --n 1000 --out results/zeroshot/bootstrap_radar_vi.csv

Đầu ra: một dòng mỗi mô hình (EER, khoảng 95%), một dòng mỗi cặp (chênh lệch A − B, khoảng 95%,
tỉ lệ lượt mà A − B > 0). Khoảng 95% không chứa 0 → chênh lệch khó là do ngẫu nhiên chọn câu.
Chỉ phản ánh độ bất định do chọn câu, không phản ánh độ bất định do huấn luyện (mỗi checkpoint chỉ có một lần chạy).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_eer, load_scores  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/zeroshot")
    p.add_argument("--dataset", default="radar_vi")
    p.add_argument("--models", nargs="*", help="mặc định: mọi thư mục có <model>/<dataset>/scores.csv")
    p.add_argument("--pairs", nargs="*", default=[], help="A:B, chênh lệch tính là EER(A) − EER(B)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out")
    a = p.parse_args()

    models = a.models or sorted(m for m in os.listdir(a.root)
                                if os.path.exists(os.path.join(a.root, m, a.dataset, "scores.csv")))
    frames = {}
    for m in models:
        path = os.path.join(a.root, m, a.dataset, "scores.csv")
        kw = {"score_col": "Score"}
        df = load_scores(path, **kw).set_index("ID")
        frames[m] = df
    ids = frames[models[0]].index
    for m in models[1:]:
        if not frames[m].index.sort_values().equals(ids.sort_values()):
            sys.exit(f"{m}: tập câu khác {models[0]}")
    labels = frames[models[0]].loc[ids, "label"].to_numpy()
    S = {m: frames[m].loc[ids, "score"].to_numpy(dtype=np.float64) for m in models}
    real_idx, fake_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]

    rng = np.random.default_rng(a.seed)
    boot = {m: np.empty(a.n) for m in models}
    for b in range(a.n):
        r = rng.choice(real_idx, real_idx.size, replace=True)
        f = rng.choice(fake_idx, fake_idx.size, replace=True)
        for m in models:
            boot[m][b] = 100 * compute_eer(S[m][r], S[m][f])[0]

    rows = []
    for m in models:
        point = 100 * compute_eer(S[m][real_idx], S[m][fake_idx])[0]
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        rows.append({"kind": "model", "name": m, "eer_%": point, "ci_low": lo, "ci_high": hi, "p_gt0": np.nan})
    for pair in a.pairs:
        x, y = pair.split(":")
        d = boot[x] - boot[y]
        point = rows[models.index(x)]["eer_%"] - rows[models.index(y)]["eer_%"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"kind": "diff", "name": f"{x} - {y}", "eer_%": point, "ci_low": lo, "ci_high": hi,
                     "p_gt0": float(np.mean(d > 0))})
    out = pd.DataFrame(rows)
    print(f"{a.dataset}: {real_idx.size} thật, {fake_idx.size} giả, {a.n} lượt bootstrap, seed {a.seed}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if a.out:
        out.to_csv(a.out, index=False)
        print("Đã lưu", a.out)


if __name__ == "__main__":
    main()
