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
    p.add_argument("--pairs", nargs="*", default=GD5_PAIRS + GD6_PAIRS,
                   help="A:B, chênh lệch EER(A) − EER(B). Bootstrap rút ngẫu nhiên tuần tự theo thứ tự cặp, "
                        "nên chỉ nối thêm cặp mới vào cuối để giữ nguyên số của các cặp cũ")
    p.add_argument("--group-pairs", nargs="*",
                   default=["G1:R", "G1:A0", "G2:G1", "G3:G1", "G2:R", "A0:R", "N0:R", "B2:R", "P0:R", "R:S0_mms_300m", "G1:S0_mms_300m", "G2:S0_mms_300m"],
                   help="A:B theo mã cấu hình (<mã>_s<seed>), so qua các seed (mục 5)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/finetune/analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    beta = dcf_beta()
    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.2f}"  # noqa: E731

    proto = pd.read_csv(os.path.join(a.protocols, "seaspoof_vi_test.tsv"), sep="\t").set_index("ID")
    mproto = pd.read_csv(os.path.join(a.protocols, "mlaad_vi_test.tsv"), sep="\t").set_index("ID")
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
            m = m.join(mproto[["sys"]])
            for s_ in sorted(m.sys[m.label == 0].dropna().unique()):
                v[f"mlaad_sys:{s_}"] = (m.score[m.label == 1], m.score[(m.sys == s_) & (m.label == 0)])
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

    # 2b. Theo hệ thống sinh, MLAAD fake/vi (5 TTS tiếng Việt chưa gặp; tiếng thật VIVOS của test)
    rows = []
    for cfg, v in V.items():
        row = {"config": cfg}
        for k in sorted(x for x in v if x.startswith("mlaad_sys:")):
            row[k[10:]] = eer(*v[k])
            row[k[10:] + "_far0"] = 100 * float(np.mean(np.asarray(v[k][1]) > 0))
        if len(row) > 1:
            rows.append(row)
    if rows:
        msys_t = pd.DataFrame(rows)
        msys_t.to_csv(os.path.join(a.out, "table_by_system_mlaad.csv"), index=False)
        print("\n== 2b. EER (%) theo hệ thống sinh, MLAAD fake/vi; _far0: % câu giả bị nhận là thật ở ngưỡng 0")
        print(msys_t.to_string(index=False, float_format=fmt))

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

    # 5. So sánh hai cấu hình qua các seed (độ bất định do seed huấn luyện, GĐ5 phát hiện 8)
    seed_t = group_compare(main_t, a.group_pairs)
    if len(seed_t):
        seed_t.to_csv(os.path.join(a.out, "seed_tests.csv"), index=False)
        print("\n== 5. So sánh qua seed: chênh lệch trung bình A − B (điểm %), KTC 95% và p (t-test hai phía).\n"
              "   Hai bên ≥ 2 seed: Welch. Một bên 1 seed: dùng độ lệch seed của bên kia (giả định cùng\n"
              "   độ lệch), t = (A − B)/(s·√(1 + 1/n)), bậc tự do n − 1.")
        print(seed_t.to_string(index=False, float_format=fmt))


# Cặp bootstrap đã báo ở GĐ5 (đúng thứ tự trong results/finetune/analysis/bootstrap_pairs.csv commit ef4d943),
# rồi cặp của GĐ6 (chỉ nối vào cuối). Cặp chưa có điểm thì bỏ qua.
GD5_PAIRS = ["R_s1:S0_mms_300m", "R_s2:S0_mms_300m", "R_s3:S0_mms_300m", "A0_s1:S0_mms_300m", "A0_s1:R_s1",
             "P0_s1:R_s1", "B2_s1:R_s1", "B2_s1:S0_w2v_large", "B1_s1:B2_s1", "B1_s1:S0_w2v_small",
             "N0_s1:R_s1", "N0_s1:S0_mms_300m_nda", "N0_s1:A0_s1", "R_s2:R_s1", "R_s3:R_s1",
             "R_s1:ref_xlsr_2b", "A0_s1:ref_xlsr_2b", "S0_mms_300m_nda:S0_mms_300m"]
GD6_PAIRS = ["G1_s1:R_s1", "G1_s1:A0_s1", "G1_s1:S0_mms_300m", "G2_s1:G1_s1", "G2_s1:R_s1"]

SEED_METRICS = ("radar_vi_eer", "itw_eer", "radar_en_eer", "radar_vi_frr0", "test_cvreal_eer", "mlaad_eer")


def group_compare(main_t, pairs, metrics_=SEED_METRICS):
    """main_t: bảng chính (cột config = <mã>_s<seed>). pairs: ["G1:R", ...] theo mã cấu hình."""
    import re as _re
    from scipy import stats

    t = main_t[main_t.config.str.match(r".+_s\d+$")].copy()
    t["base"] = t.config.map(lambda c: _re.sub(r"_s\d+$", "", c))
    # Mốc cố định (S0_*, ref_*: dùng thẳng, không huấn luyện nên không có độ lệch seed): so bằng
    # t-test một mẫu, se = s_A/√n_A
    fixed = main_t[main_t.config.str.match(r"^(S0_|ref_)")].copy()
    fixed["base"] = fixed.config
    t = pd.concat([t, fixed], ignore_index=True)
    rows = []
    for pair in pairs:
        A, B = pair.split(":")
        da, db = t[t.base == A], t[t.base == B]
        if da.empty or db.empty:
            continue
        for m in metrics_:
            if m not in t:
                continue
            xa, xb = da[m].dropna().values, db[m].dropna().values
            na, nb = len(xa), len(xb)
            if na == 0 or nb == 0 or (na < 2 and nb < 2):
                continue
            d = xa.mean() - xb.mean()
            if B.startswith(("S0_", "ref_")) and na >= 2:
                se = xa.std(ddof=1) / np.sqrt(na)
                df, kind = na - 1, "một mẫu, B cố định"
            elif na >= 2 and nb >= 2:
                va, vb = xa.var(ddof=1) / na, xb.var(ddof=1) / nb
                se = np.sqrt(va + vb)
                df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if va + vb > 0 else np.nan
                kind = "Welch"
            else:
                x = xa if na >= 2 else xb
                se = x.std(ddof=1) * np.sqrt(1 + 1 / len(x))
                df, kind = len(x) - 1, f"1 seed, s của {A if na >= 2 else B}"
            tq = stats.t.ppf(0.975, df) if df == df else np.nan
            p = 2 * stats.t.sf(abs(d / se), df) if se > 0 else np.nan
            rows.append({"A": A, "B": B, "metric": m, "n_A": na, "n_B": nb, "mean_A": xa.mean(),
                         "sd_A": xa.std(ddof=1) if na >= 2 else np.nan, "mean_B": xb.mean(),
                         "sd_B": xb.std(ddof=1) if nb >= 2 else np.nan, "diff_A_minus_B": d,
                         "ci95_lo": d - tq * se, "ci95_hi": d + tq * se, "df": df, "p": p, "test": kind})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
