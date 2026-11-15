#!/usr/bin/env python
"""Kiểm giả thuyết cơ chế GRPO so với SFT trên điểm từng câu (GĐ6, L9; GĐ5 mục 6.3).

  python src/analyze_mechanism.py --out results/finetune/analysis

Giả thuyết (từ code AntiDeepfake, chuẩn hoá phần thưởng theo batch): kỳ vọng gradient theo log-odds
của nhãn đúng là 1 − p với SFT, p(1 − p)/σ_batch với GRPO (p = xác suất mô hình gán cho nhãn đúng).
Dự đoán: SFT kéo mạnh nhất những câu mô hình đang sai chắc chắn (p → 0); GRPO gần như bỏ qua
chúng và kéo câu ở ranh giới.

Mọi cấu hình MMS-300M Default (R, A0, G1, G2, G3) cùng xuất phát từ S0 (mms_300m dùng thẳng), nên:
  z0 = log-odds của nhãn đúng theo S0 (dương: S0 đúng), z = của mô hình sau fine-tune, Δz = z − z0.
Chia câu theo z0 thành 4 nhóm (|z0| > 3 tương ứng p < 0,047 hoặc p > 0,953):
  "S0 sai chắc" (z0 < −3), "S0 sai" (−3 ≤ z0 < 0), "S0 đúng" (0 ≤ z0 < 3), "S0 đúng chắc" (z0 ≥ 3).
Báo mỗi (tập, lớp, nhóm): số câu, Δz trung vị, % lật nhãn ở ngưỡng 0 (sai -> đúng với nhóm S0 sai;
đúng -> sai với nhóm S0 đúng). Tập dev SEA-Spoof gần với phân bố train nhất (cùng hệ thống, người nói
khác), nên là nơi dự đoán về gradient kiểm được rõ nhất; RADAR VI, ItW, RADAR EN cho biết ngoài miền.
Đây là phân tích mô tả trên câu kiểm tra, không phải đo gradient trên câu train.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_scores  # noqa: E402

BINS = [(-np.inf, -3, "S0 sai chắc"), (-3, 0, "S0 sai"), (0, 3, "S0 đúng"), (3, np.inf, "S0 đúng chắc")]


def signed(df):
    """log-odds của nhãn đúng: score nếu thật, −score nếu giả."""
    return np.where(df.label.values == 1, df.score.values, -df.score.values)


def sets(path):
    """{tên tập: DataFrame (index ID; score, label)}; tách ood_forget thành itw, radar_en."""
    out = {}
    for name in ("seaspoof_vi_dev", "radar_vi", "ood_forget"):
        p = os.path.join(path, name, "scores.csv")
        if not os.path.exists(p):
            continue
        d = load_scores(p, score_col="Score").set_index("ID")
        if name == "ood_forget":
            out["itw"] = d[d.index.str.startswith("ItW-")]
            out["radar_en"] = d[d.index.str.startswith("RADAR2026-")]
        else:
            out["dev" if name == "seaspoof_vi_dev" else name] = d
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zeroshot", default="results/zeroshot")
    p.add_argument("--finetune", default="results/finetune")
    p.add_argument("--base", default="mms_300m", help="mô hình xuất phát (thư mục trong --zeroshot)")
    p.add_argument("--configs", nargs="*", default=None,
                   help="lần chạy cùng điểm xuất phát; mặc định: mọi R_s*, A0_s*, G*_s*")
    p.add_argument("--out", default="results/finetune/analysis")
    a = p.parse_args()
    base = sets(os.path.join(a.zeroshot, a.base))
    runs = a.configs or sorted(r for r in os.listdir(a.finetune)
                               if r.split("_")[0] in ("R", "A0") or r.startswith("G"))
    rows = []
    for run in runs:
        S = sets(os.path.join(a.finetune, run))
        for name, d in S.items():
            if name not in base:
                continue
            j = base[name].join(d[["score"]].rename(columns={"score": "score_ft"}), how="inner")
            z0 = signed(j)
            z = np.where(j.label.values == 1, j.score_ft.values, -j.score_ft.values)
            for lab, lab_name in ((1, "thật"), (0, "giả")):
                for lo, hi, bname in BINS:
                    m = (j.label.values == lab) & (z0 >= lo) & (z0 < hi)
                    if not m.any():
                        continue
                    flip = 100 * np.mean(z[m] >= 0) if hi <= 0 else 100 * np.mean(z[m] < 0)
                    rows.append({"config": run, "set": name, "class": lab_name, "bin": bname, "n": int(m.sum()),
                                 "dz_median": float(np.median(z[m] - z0[m])), "dz_mean": float(np.mean(z[m] - z0[m])),
                                 "flip_%": flip})
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "mechanism_bins.csv"), index=False)
    pd.set_option("display.width", 200)
    print("Δz = log-odds nhãn đúng sau fine-tune − của S0 (dương: tiến về đúng). flip_%: nhóm 'S0 sai' -> "
          "% thành đúng; nhóm 'S0 đúng' -> % thành sai (ngưỡng 0).")
    for name in ("dev", "radar_vi", "itw", "radar_en"):
        sub = t[t.set == name]
        if not len(sub):
            continue
        piv = sub.pivot_table(index=["class", "bin"], columns="config", values=["dz_median", "flip_%"])
        n = sub.groupby(["class", "bin"]).n.first()
        piv.insert(0, ("n", ""), n)
        print(f"\n== {name}")
        print(piv.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
