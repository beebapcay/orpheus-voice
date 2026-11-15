#!/usr/bin/env python
"""Bảng kết quả GĐ5 (ablation) từ các file điểm đã có (chạy trên máy, không cần GPU).

  python src/analyze_ablation.py --out results/finetune/analysis

Mỗi cấu hình là một nguồn điểm:
  S0 (dùng thẳng)   results/zeroshot/<backbone>/<tập>/scores.csv      (GĐ3; SEA-Spoof chấm ở GĐ5 lần 1)
  Fine-tune         results/finetune/<lần chạy>/<tập>/scores.csv      (R_s1, A0_s1, ...)

Ghi ra --out:
  table_main.csv        mỗi cấu hình một dòng: EER trên test_seen, test_f5, test_minimax, test_cvreal,
                        RADAR tiếng Việt, In-the-Wild, RADAR EN; kèm minDCF, actDCF − minDCF, FAR/FRR ở ngưỡng 0
  table_by_system.csv   EER theo từng hệ thống sinh của SEA-Spoof test (5 đã thấy + 2 để riêng)
  hard_ids.csv          tập 493 câu khó chung của GĐ3 (RADAR tiếng Việt): tỉ lệ còn sai ở ngưỡng EER của
                        chính mô hình, và ở ngưỡng 0
  bootstrap_pairs.csv   chênh lệch EER A − B kèm khoảng tin cậy 95% bootstrap ghép cặp (cùng câu)

Điểm: log-odds = logit thật − logit giả (src/evaluate.py), cao hơn là thật.
EER, minDCF, actDCF, FAR, FRR: thấp hơn là tốt hơn.
FAR: tỉ lệ câu giả bị nhận là thật; FRR: tỉ lệ câu thật bị coi là giả (ngưỡng log-odds 0).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_act_dcf, compute_eer, compute_min_dcf, dcf_beta, load_scores  # noqa: E402

# Tập con của seaspoof_vi_test (cột `group` trong protocols/ids/seaspoof_vi_test.tsv)
SUBSETS = {"test_seen": (["real_vivos"], ["seen"]),
           "test_f5": (["real_vivos"], ["f5_tts"]),
           "test_minimax": (["real_vivos"], ["minimax"]),
           "test_cvreal": (["real_cv"], ["seen"])}
# Cấu hình S0: tên hiển thị -> thư mục trong results/zeroshot
S0 = {"S0_mms_300m": "mms_300m", "S0_mms_300m_nda": "mms_300m_nda", "S0_w2v_large": "w2v_large",
      "S0_w2v_small": "w2v_small", "ref_xlsr_2b": "xlsr_2b"}


def load(path):
    return load_scores(path, score_col="Score").set_index("ID") if os.path.exists(path) else None


def eer(r, f):
    return 100 * compute_eer(np.asarray(r), np.asarray(f))[0]


def metrics(r, f, beta):
    r, f = np.asarray(r), np.asarray(f)
    e, thr = compute_eer(r, f)
    mind, _ = compute_min_dcf(r, f, beta)
    actd, _ = compute_act_dcf(r, f, beta)
    return {"eer": 100 * e, "minDCF": mind, "act_minus_min": actd - mind,
            "far0": 100 * np.mean(f > 0), "frr0": 100 * np.mean(r <= 0), "thr": thr}


def sources(a):
    """{cấu hình: {tập: DataFrame điểm}}; tập: seaspoof_vi_test, radar_vi, ood_forget."""
    out = {}
    for name, d in S0.items():
        out[name] = {ds: load(os.path.join(a.zeroshot, d, ds, "scores.csv"))
                     for ds in ("seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test")}
    for run in sorted(os.listdir(a.finetune)):
        if run.startswith(("pilot", "probe", "analysis")) or not os.path.isdir(os.path.join(a.finetune, run)):
            continue
        out[run] = {ds: load(os.path.join(a.finetune, run, ds, "scores.csv"))
                    for ds in ("seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test")}
    return {k: v for k, v in out.items() if any(x is not None for x in v.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zeroshot", default="results/zeroshot")
    p.add_argument("--finetune", default="results/finetune")
    p.add_argument("--protocols", default="protocols/ids")
    p.add_argument("--hard", default="results/zeroshot/analysis/hard_ids_default7.csv")
    p.add_argument("--pairs", nargs="*",
                   default=["R_s1:S0_mms_300m", "A0_s1:R_s1", "R_s1:ref_xlsr_2b"],
                   help="A:B, chênh lệch EER(A) − EER(B)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/finetune/analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    beta = dcf_beta()
    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.2f}"  # noqa: E731

    proto = pd.read_csv(os.path.join(a.protocols, "seaspoof_vi_test.tsv"), sep="\t").set_index("ID")
    hard = pd.read_csv(a.hard).set_index("ID")
    S = sources(a)

    def views(cfg):
        """{tên tập con: (điểm thật, điểm giả)} của một cấu hình."""
        v = {}
        t = S[cfg].get("seaspoof_vi_test")
        if t is not None:
            t = t.join(proto[["group", "sys"]])
            for name, (rg, fg) in SUBSETS.items():
                v[name] = (t.score[t.group.isin(rg) & (t.label == 1)], t.score[t.group.isin(fg) & (t.label == 0)])
            real = t.score[(t.group == "real_vivos") & (t.label == 1)]
            for s in sorted(t.sys[t.label == 0].unique()):
                v[f"sys:{s}"] = (real, t.score[(t.sys == s) & (t.label == 0)])
        r = S[cfg].get("radar_vi")
        if r is not None:
            v["radar_vi"] = (r.score[r.label == 1], r.score[r.label == 0])
        m = S[cfg].get("mlaad_vi_test")
        if m is not None:
            v["mlaad"] = (m.score[m.label == 1], m.score[m.label == 0])
        o = S[cfg].get("ood_forget")
        if o is not None:
            for pre, name in (("ItW-", "itw"), ("RADAR2026-", "radar_en")):
                m = o.index.str.startswith(pre)
                v[name] = (o.score[m & (o.label == 1)], o.score[m & (o.label == 0)])
        return v

    V = {cfg: views(cfg) for cfg in S}

    # 1. Bảng chính
    rows = []
    for cfg, v in V.items():
        row = {"config": cfg}
        for name in ("test_seen", "test_f5", "test_minimax", "test_cvreal", "mlaad", "radar_vi", "itw", "radar_en"):
            if name in v:
                m = metrics(*v[name], beta)
                row[f"{name}_eer"] = m["eer"]
                if name in ("radar_vi", "test_f5", "test_minimax", "test_cvreal"):
                    row[f"{name}_far0"], row[f"{name}_frr0"] = m["far0"], m["frr0"]
                if name == "radar_vi":
                    row["radar_vi_minDCF"], row["radar_vi_act_minus_min"] = m["minDCF"], m["act_minus_min"]
        rows.append(row)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(os.path.join(a.out, "table_main.csv"), index=False)
    cols = ["config", "test_seen_eer", "test_f5_eer", "test_minimax_eer", "test_cvreal_eer", "mlaad_eer",
            "radar_vi_eer", "itw_eer", "radar_en_eer"]
    print("== 1. EER (%), thấp hơn là tốt hơn")
    print(main_t[[c for c in cols if c in main_t]].to_string(index=False, float_format=fmt))
    cols = ["config", "radar_vi_far0", "radar_vi_frr0", "radar_vi_minDCF", "radar_vi_act_minus_min",
            "test_cvreal_frr0", "test_f5_far0", "test_minimax_far0"]
    print("\n== 1b. Lỗi ở ngưỡng log-odds 0 (%): FAR giả bị nhận là thật, FRR thật bị coi là giả; "
          "actDCF − minDCF (lệch hiệu chuẩn)")
    print(main_t[[c for c in cols if c in main_t]].to_string(index=False, float_format=fmt))

    # 1c. Trung bình ± độ lệch chuẩn qua các seed (cấu hình có từ 2 seed: <mã>_s<seed>)
    import re as _re
    main_t["base"] = main_t.config.map(lambda c: _re.sub(r"_s\d+$", "", c))
    num = [c for c in main_t.columns if c not in ("config", "base")]
    g = main_t[main_t.config.str.match(r".+_s\d+$")].groupby("base")
    multi = [b for b, d in g if len(d) >= 2]
    if multi:
        agg = g[num].agg(["mean", "std"]).loc[multi]
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.insert(0, "n_seed", g.size().loc[multi])
        agg.to_csv(os.path.join(a.out, "table_seeds.csv"))
        show = ["n_seed"] + [f"{c}_{st}" for c in ("test_cvreal_eer", "radar_vi_eer", "itw_eer", "radar_en_eer",
                                                  "radar_vi_frr0") for st in ("mean", "std") if f"{c}_{st}" in agg]
        print("\n== 1c. Trung bình và độ lệch chuẩn (ddof 1) qua các seed")
        print(agg[show].to_string(float_format=fmt))
    main_t = main_t.drop(columns="base")

    # 2. Theo hệ thống sinh (SEA-Spoof test; tiếng thật: real_vivos)
    rows = []
    for cfg, v in V.items():
        row = {"config": cfg}
        for k in sorted(x for x in v if x.startswith("sys:")):
            row[k[4:]] = eer(*v[k])
        if len(row) > 1:
            rows.append(row)
    if rows:
        sys_t = pd.DataFrame(rows)
        sys_t.to_csv(os.path.join(a.out, "table_by_system.csv"), index=False)
        print("\n== 2. EER (%) theo hệ thống sinh, SEA-Spoof test (f5_tts, minimax: để riêng)")
        print(sys_t.to_string(index=False, float_format=fmt))

    # 3. Tập câu khó chung (GĐ3): còn sai bao nhiêu
    rows = []
    for cfg in S:
        r = S[cfg].get("radar_vi")
        if r is None:
            continue
        m = metrics(r.score[r.label == 1], r.score[r.label == 0], beta)
        h = r.join(hard[["n_wrong_of_7"]], how="inner")
        fake, real = h[h.label == 0], h[h.label == 1]
        rows.append({"config": cfg, "n_fake": len(fake), "n_real": len(real),
                     "fake_wrong_at_eer_thr_%": 100 * np.mean(fake.score > m["thr"]),
                     "real_wrong_at_eer_thr_%": 100 * np.mean(real.score <= m["thr"]),
                     "fake_wrong_at0_%": 100 * np.mean(fake.score > 0),
                     "real_wrong_at0_%": 100 * np.mean(real.score <= 0)})
    hard_t = pd.DataFrame(rows)
    hard_t.to_csv(os.path.join(a.out, "hard_ids.csv"), index=False)
    print("\n== 3. Tập 493 câu khó chung của GĐ3 (RADAR tiếng Việt): % còn sai (thấp hơn là tốt hơn)")
    print(hard_t.to_string(index=False, float_format=fmt))

    # 4. Bootstrap ghép cặp cho chênh lệch EER
    rng = np.random.default_rng(a.seed)
    rows = []
    for pair in a.pairs:
        A, B = pair.split(":")
        if A not in V or B not in V:
            continue
        for name in ("test_seen", "test_f5", "test_minimax", "test_cvreal", "mlaad", "radar_vi", "itw", "radar_en"):
            if name not in V[A] or name not in V[B]:
                continue
            (ra, fa), (rb, fb) = V[A][name], V[B][name]
            ri = ra.index.intersection(rb.index)
            fi = fa.index.intersection(fb.index)
            ra, rb, fa, fb = (x.reindex(i).values for x, i in ((ra, ri), (rb, ri), (fa, fi), (fb, fi)))
            d0 = eer(ra, fa) - eer(rb, fb)
            ds = []
            for _ in range(a.n_boot):
                i, j = rng.integers(0, len(ra), len(ra)), rng.integers(0, len(fa), len(fa))
                ds.append(eer(ra[i], fa[j]) - eer(rb[i], fb[j]))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            rows.append({"A": A, "B": B, "set": name, "eer_A": eer(ra, fa), "eer_B": eer(rb, fb),
                         "diff_A_minus_B": d0, "ci95_lo": lo, "ci95_hi": hi,
                         "p_A_gt_B": float(np.mean(np.array(ds) > 0))})
    if rows:
        boot = pd.DataFrame(rows)
        boot.to_csv(os.path.join(a.out, "bootstrap_pairs.csv"), index=False)
        print(f"\n== 4. Chênh lệch EER A − B (điểm %), KTC 95% bootstrap ghép cặp ({a.n_boot} lượt)")
        print(boot.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
