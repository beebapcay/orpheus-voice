#!/usr/bin/env python
"""Phân tích thêm kết quả zero-shot GĐ3 từ các file điểm đã có (chạy trên máy, không cần GPU).

  python src/analyze_zeroshot.py --root results/zeroshot --protocols protocols/ids \
      --out results/zeroshot/analysis

Ghi ra --out:
  errors_radar_vi.csv     EER, minDCF, actDCF, actDCF − minDCF; tỉ lệ giả bị nhận là thật (FAR) và thật
                          bị coi là giả (FRR) ở ngưỡng log-odds 0 và ở ngưỡng EER
  eer_by_duration.csv     EER theo nhóm độ dài câu trên RADAR tiếng Việt
  error_overlap.csv       với mỗi câu: bao nhiêu mô hình (trong một nhóm) sai ở ngưỡng EER của chính nó;
                          so với mức kỳ vọng nếu các mô hình sai độc lập
  rank_vs_paper.csv       tương quan hạng (Spearman) giữa EER trên RADAR tiếng Việt và EER trên 6 tập của
                          Bảng II bài AntiDeepfake, qua 14 checkpoint
  radar_en_vs_vi.csv      EER trên RADAR tiếng Anh (nửa RADAR của ood_forget) so với RADAR tiếng Việt

Điểm: log-odds = logit thật − logit giả (như src/evaluate.py), cao hơn là thật.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_eer, compute_min_dcf, compute_act_dcf, dcf_beta, load_scores  # noqa: E402

BACKBONES = ["w2v_small", "w2v_large", "mms_300m", "hubert_xl", "mms_1b", "xlsr_1b", "xlsr_2b"]

# Bảng II, bài AntiDeepfake (arXiv v4), EER (%), Default (có RawBoost) / NDA (không RawBoost).
# Chép qua note/Post-Training-SSL.md mục 5.
PAPER = {
    #             ADD2023        DEEP-VOICE     FoR            FoR-norm       ITW           DFE-2024
    "hubert_xl": ((18.91, 35.34), (5.68, 14.84), (2.48, 3.67), (3.18, 15.56), (5.23, 17.99), (34.10, 47.73)),
    "w2v_small": ((13.02, 19.41), (9.79, 16.20), (22.03, 1.06), (17.89, 6.48), (4.23, 4.65), (33.33, 31.97)),
    "w2v_large": ((13.25, 12.67), (4.44, 5.04), (0.67, 0.80), (0.97, 1.45), (1.91, 2.25), (33.36, 30.06)),
    "mms_300m": ((7.93, 11.22), (2.35, 3.06), (1.40, 0.48), (5.92, 2.70), (2.90, 2.00), (32.84, 31.39)),
    "mms_1b": ((9.06, 9.46), (2.47, 2.35), (1.23, 0.88), (1.73, 1.10), (1.82, 1.87), (27.70, 27.54)),
    "xlsr_1b": ((5.39, 6.58), (2.47, 2.59), (5.74, 3.17), (12.14, 10.91), (1.35, 1.36), (26.76, 26.17)),
    "xlsr_2b": ((4.67, 6.84), (2.23, 2.72), (2.61, 1.19), (1.64, 1.73), (1.23, 1.31), (27.76, 25.76)),
}
PAPER_SETS = ["ADD2023", "DEEP-VOICE", "FoR", "FoR-norm", "In-the-Wild", "DFE-2024"]


def eer_of(df):
    r, f = df.score[df.label == 1].values, df.score[df.label == 0].values
    e, thr = compute_eer(r, f)
    return 100 * e, thr


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/zeroshot")
    p.add_argument("--protocols", default="protocols/ids")
    p.add_argument("--out", default="results/zeroshot/analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    beta = dcf_beta()

    proto = pd.read_csv(os.path.join(a.protocols, "radar_vi.tsv"), sep="\t").set_index("ID")
    models = sorted(m for m in os.listdir(a.root)
                    if os.path.exists(os.path.join(a.root, m, "radar_vi", "scores.csv")))
    S = {}
    for m in models:
        kw = {"score_col": "Score"}
        S[m] = load_scores(os.path.join(a.root, m, "radar_vi", "scores.csv"), **kw).set_index("ID").join(proto[["Duration"]])

    # 1. Lỗi ở ngưỡng 0 và ngưỡng EER, chênh hiệu chuẩn
    rows = []
    for m in models:
        d = S[m]
        r, f = d.score[d.label == 1].values, d.score[d.label == 0].values
        e, thr = eer_of(d)
        mind, _ = compute_min_dcf(r, f, beta)
        actd, _ = compute_act_dcf(r, f, beta)
        rows.append({"model": m, "eer_%": e, "minDCF": mind, "actDCF": actd, "act_minus_min": actd - mind,
                     "FAR_at0_%": 100 * np.mean(f > 0), "FRR_at0_%": 100 * np.mean(r <= 0),
                     "eer_threshold": thr, "median_real": np.median(r), "median_fake": np.median(f)})
    err = pd.DataFrame(rows).sort_values("eer_%")
    err.to_csv(os.path.join(a.out, "errors_radar_vi.csv"), index=False)
    print("\n== 1. Lỗi trên RADAR tiếng Việt (FAR: giả bị nhận là thật; FRR: thật bị coi là giả; ngưỡng 0)")
    print(err.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 2. EER theo độ dài câu
    bins, names = [0, 3, 6, 10, 1000], ["<3s", "3-6s", "6-10s", ">10s"]
    rows = []
    for m in models:
        d = S[m].assign(bin=pd.cut(S[m].Duration, bins, labels=names))
        row = {"model": m}
        for n in names:
            g = d[d.bin == n]
            row[n] = eer_of(g)[0]
        rows.append(row)
    dur = pd.DataFrame(rows)
    counts = {n: (int(((pd.cut(proto.Duration, bins, labels=names) == n) & (proto.Label == "real")).sum()),
                  int(((pd.cut(proto.Duration, bins, labels=names) == n) & (proto.Label == "fake")).sum()))
              for n in names}
    dur.to_csv(os.path.join(a.out, "eer_by_duration.csv"), index=False)
    real = proto.Label == "real"
    e_len = 100 * compute_eer(proto.Duration[real].values, proto.Duration[~real].values)[0]
    print("\n== 2. EER theo độ dài câu; số câu (thật, giả):", counts)
    print(f"   EER nếu chỉ dùng độ dài làm điểm: {min(e_len, 100 - e_len):.1f}%")
    print(dur.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # 3. Các mô hình có sai trên cùng câu không
    groups = {"default_7": [m for m in BACKBONES if m in S],
              "nda_7": [m + "_nda" for m in BACKBONES if m + "_nda" in S],
              "top4_1b_2b": [m for m in ["xlsr_2b", "mms_1b", "mms_1b_nda", "xlsr_2b_nda"] if m in S]}
    rows = []
    ids = S[models[0]].index
    lab = S[models[0]].label.reindex(ids).values
    for gname, gm in groups.items():
        wrong = np.zeros((len(ids), len(gm)), dtype=bool)
        rates_f, rates_r = [], []
        for j, m in enumerate(gm):
            d = S[m].reindex(ids)
            _, thr = eer_of(d)
            acc_real = d.score.values > thr
            wrong[:, j] = np.where(lab == 1, ~acc_real, acc_real)
            rates_f.append(wrong[lab == 0, j].mean())
            rates_r.append(wrong[lab == 1, j].mean())
        k = len(gm)
        for cls, mask, rates in (("fake", lab == 0, rates_f), ("real", lab == 1, rates_r)):
            cnt = wrong[mask].sum(axis=1)
            rows.append({"group": gname, "n_models": k, "class": cls, "n": int(mask.sum()),
                         "mean_error_%": 100 * float(np.mean(rates)),
                         "wrong_by_all_%": 100 * float(np.mean(cnt == k)),
                         "expected_if_independent_%": 100 * float(np.prod(rates)),
                         "wrong_by_none_%": 100 * float(np.mean(cnt == 0)),
                         "wrong_by_majority_%": 100 * float(np.mean(cnt > k / 2))})
        if gname == "default_7":
            # Câu mà cả 7 mô hình Default đều sai: tập "khó chung", theo dõi lại sau fine-tune (GĐ5–6)
            cnt_all = wrong.sum(axis=1)
            hard = pd.DataFrame({"ID": ids, "Label": np.where(lab == 1, "real", "fake"),
                                 "n_wrong_of_7": cnt_all,
                                 "Duration": S[models[0]].Duration.reindex(ids).values})
            hard[hard.n_wrong_of_7 == k].to_csv(os.path.join(a.out, "hard_ids_default7.csv"), index=False)
    ov = pd.DataFrame(rows)
    ov.to_csv(os.path.join(a.out, "error_overlap.csv"), index=False)
    print("\n== 3. Sai trùng nhau (ở ngưỡng EER của từng mô hình)")
    print(ov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 4. Tương quan hạng với Bảng II của bài
    ours, paper_cols = [], {s: [] for s in PAPER_SETS}
    for bb in BACKBONES:
        for v, suffix in ((0, ""), (1, "_nda")):
            m = bb + suffix
            if m not in S:
                continue
            ours.append(err.set_index("model").loc[m, "eer_%"])
            for i, s in enumerate(PAPER_SETS):
                paper_cols[s].append(PAPER[bb][i][v])
    rk = pd.DataFrame([{"paper_set": s, "n_checkpoints": len(ours), "spearman_vs_radar_vi": spearman(ours, paper_cols[s])}
                       for s in PAPER_SETS])
    # bỏ HuBERT-XL (ngoại lệ ở mọi tập) để xem tương quan có do một điểm kéo không
    keep = [i for i, bb in enumerate([b for b in BACKBONES for _ in (0, 1)]) if bb != "hubert_xl"]
    rk["spearman_without_hubert"] = [spearman([ours[i] for i in keep], [paper_cols[s][i] for i in keep]) for s in PAPER_SETS]
    rk.to_csv(os.path.join(a.out, "rank_vs_paper.csv"), index=False)
    print("\n== 4. Spearman giữa EER RADAR tiếng Việt và EER trong Bảng II (14 checkpoint)")
    print(rk.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # 5. RADAR tiếng Anh so với tiếng Việt
    rows = []
    for m in models:
        f = os.path.join(a.root, m, "ood_forget", "scores.csv")
        if not os.path.exists(f):
            continue
        kw = {"score_col": "Score"}
        d = load_scores(f, **kw)
        en = d[d.ID.str.startswith("RADAR")]
        itw = d[~d.ID.str.startswith("RADAR")]
        rows.append({"model": m, "radar_en_eer_%": eer_of(en)[0], "radar_vi_eer_%": err.set_index("model").loc[m, "eer_%"],
                     "itw_sub_eer_%": eer_of(itw)[0],
                     "radar_en_FAR_at0_%": 100 * np.mean(en.score[en.label == 0] > 0),
                     "radar_vi_FAR_at0_%": err.set_index("model").loc[m, "FAR_at0_%"]})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(a.out, "radar_en_vs_vi.csv"), index=False)
    print("\n== 5. RADAR tiếng Anh (1.000 + 1.000) so với RADAR tiếng Việt")
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nĐã lưu vào", a.out)


if __name__ == "__main__":
    main()
