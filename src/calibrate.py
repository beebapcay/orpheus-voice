#!/usr/bin/env python
"""Hiệu chuẩn điểm (GĐ6 ngày 3): học phép biến đổi affine s' = a·s + b trên dev, áp lên tập kiểm tra.

  python src/calibrate.py --out results/calibration

Điểm: log-odds = logit thật − logit giả (src/evaluate.py), cao hơn là thật. Hiệu chuẩn bằng hồi quy
logistic có trọng số prior (cách chuẩn cho bộ phát hiện xuất LLR, như BOSARIS/pyllr): với prior hiệu
dụng π = P(thật) của bài toán DCF, cực tiểu
    π · TB_thật[log(1 + e^−(a·s + b + logit π))] + (1 − π) · TB_giả[log(1 + e^(a·s + b + logit π))],
khi đó a·s + b là LLR đã hiệu chuẩn. DCF theo ASVspoof 5 (src/evaluate.py): DCF = β·P_miss + P_fa,
β = 1,9, ngưỡng Bayes t = −ln β ≈ −0,64 (nhận là thật nếu s ≥ t); ngưỡng này tương ứng π = β/(1 + β).
Phép affine với a > 0 không đổi thứ tự điểm, nên EER và minDCF không đổi; chỉ actDCF (lỗi ở ngưỡng cố
định) thay đổi. actDCF − minDCF là phần lỗi do lệch hiệu chuẩn.

Dev: seaspoof_vi_dev (tiếng thật chỉ từ VIVOS, giả từ 5 hệ thống train). Đây là dev hợp lệ duy nhất (tập
kiểm tra không được dùng để học hệ số), nên hiệu chuẩn chỉ sửa được lệch mà dev thấy được.

Ghi ra --out: calibration.csv (mỗi cấu hình × tập một dòng), params.csv (a, b mỗi cấu hình).
minDCF, actDCF, FAR, FRR: thấp hơn là tốt hơn.
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_eer, compute_min_dcf, dcf_beta, load_scores  # noqa: E402

S0 = {"S0_mms_300m": "mms_300m", "S0_mms_300m_nda": "mms_300m_nda", "S0_w2v_large": "w2v_large",
      "S0_w2v_small": "w2v_small"}
SETS = ("seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test")


def fit_affine(r, f, prior, mode="affine"):
    """Hồi quy logistic có trọng số prior: trả về (a, b) sao cho a·s + b là LLR.
    mode: "affine" (học a, b); "shift" (giữ a = 1, chỉ học b: chỉ dời ngưỡng, ổn định khi dev tách được
    hoàn toàn); "platt" (học a, b với nhãn làm mềm kiểu Platt 1999: thật (N+1)/(N+2), giả 1/(N+2), để hệ
    số không phình vô hạn khi dev tách được hoàn toàn)."""
    r, f = np.asarray(r, float), np.asarray(f, float)
    lp = math.log(prior / (1 - prior))
    yr = (len(r) + 1) / (len(r) + 2) if mode == "platt" else 1.0   # nhãn đích của câu thật
    yf = 1 / (len(f) + 2) if mode == "platt" else 0.0              # nhãn đích của câu giả

    def unpack(w):
        return (1.0, w[0]) if mode == "shift" else (w[0], w[1])

    def loss(w):
        a, b = unpack(w)
        zr, zf = a * r + b + lp, a * f + b + lp
        # cross-entropy với nhãn y: y·log(1+e^−z) + (1−y)·log(1+e^z)
        lr = yr * np.logaddexp(0, -zr) + (1 - yr) * np.logaddexp(0, zr)
        lf = yf * np.logaddexp(0, -zf) + (1 - yf) * np.logaddexp(0, zf)
        return prior * np.mean(lr) + (1 - prior) * np.mean(lf)

    def grad(w):
        a, b = unpack(w)
        zr, zf = a * r + b + lp, a * f + b + lp
        sr, sf = 1 / (1 + np.exp(-zr)), 1 / (1 + np.exp(-zf))   # d loss/dz = sigmoid(z) − y
        gr, gf = prior / len(r) * (sr - yr), (1 - prior) / len(f) * (sf - yf)
        gb = np.sum(gr) + np.sum(gf)
        return np.array([gb]) if mode == "shift" else np.array([np.sum(gr * r) + np.sum(gf * f), gb])

    x0 = np.array([0.0]) if mode == "shift" else np.array([1.0, 0.0])
    res = minimize(loss, x0=x0, jac=grad, method="BFGS")
    if not res.success and "precision loss" not in str(res.message):
        raise RuntimeError(f"hiệu chuẩn không hội tụ ({mode}): {res.message}")
    a, b = unpack(res.x)
    return float(a), float(b)


def errs(r, f, t):
    """FRR (thật bị coi là giả, s < t), FAR (giả bị nhận là thật, s ≥ t) ở ngưỡng t, theo %."""
    return 100 * float(np.mean(np.asarray(r) < t)), 100 * float(np.mean(np.asarray(f) >= t))


def load(path):
    return load_scores(path, score_col="Score").set_index("ID") if os.path.exists(path) else None


def views(S, proto):
    """{tập: (điểm thật, điểm giả)}: dev, test_seen, test_cvreal, radar_vi, itw, radar_en, mlaad."""
    v = {}
    d = S.get("seaspoof_vi_dev")
    if d is not None:
        v["dev"] = (d.score[d.label == 1], d.score[d.label == 0])
    t = S.get("seaspoof_vi_test")
    if t is not None:
        t = t.join(proto[["group"]])
        v["test_seen"] = (t.score[(t.group == "real_vivos") & (t.label == 1)], t.score[(t.group == "seen") & (t.label == 0)])
        v["test_cvreal"] = (t.score[(t.group == "real_cv") & (t.label == 1)], t.score[(t.group == "seen") & (t.label == 0)])
    r = S.get("radar_vi")
    if r is not None:
        v["radar_vi"] = (r.score[r.label == 1], r.score[r.label == 0])
    o = S.get("ood_forget")
    if o is not None:
        for pre, name in (("ItW-", "itw"), ("RADAR2026-", "radar_en")):
            m = o.index.str.startswith(pre)
            v[name] = (o.score[m & (o.label == 1)], o.score[m & (o.label == 0)])
    m = S.get("mlaad_vi_test")
    if m is not None:
        v["mlaad"] = (m.score[m.label == 1], m.score[m.label == 0])
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zeroshot", default="results/zeroshot")
    p.add_argument("--finetune", default="results/finetune")
    p.add_argument("--protocols", default="protocols/ids")
    p.add_argument("--out", default="results/calibration")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    beta = dcf_beta()
    t = -math.log(beta)
    prior = beta / (1 + beta)
    proto = pd.read_csv(os.path.join(a.protocols, "seaspoof_vi_test.tsv"), sep="\t").set_index("ID")

    cfgs = {name: {s: load(os.path.join(a.zeroshot, d, s, "scores.csv")) for s in SETS} for name, d in S0.items()}
    for run in sorted(os.listdir(a.finetune)):
        if run.startswith(("pilot", "probe", "analysis")) or not os.path.isdir(os.path.join(a.finetune, run)):
            continue
        cfgs[run] = {s: load(os.path.join(a.finetune, run, s, "scores.csv")) for s in SETS}

    modes = ("affine", "shift", "platt")
    rows, params = [], []
    for cfg, S in cfgs.items():
        v = views(S, proto)
        if "dev" not in v:
            continue
        coef = {m: fit_affine(*v["dev"], prior, m) for m in modes}
        params.append({"config": cfg, **{f"{k}_{m}": c for m, (ca, cb) in coef.items() for k, c in (("a", ca), ("b", cb))},
                       "n_dev_real": len(v["dev"][0]), "n_dev_fake": len(v["dev"][1])})
        for name, (r, f) in v.items():
            r, f = r.values, f.values
            mind, _ = compute_min_dcf(r, f, beta)
            frr0, far0 = errs(r, f, t)
            act0 = beta * frr0 / 100 + far0 / 100
            row = {"config": cfg, "set": name, "eer": 100 * compute_eer(r, f)[0], "minDCF": mind,
                   "actDCF_raw": act0, "frr_raw": frr0, "far_raw": far0}
            for m, (ca, cb) in coef.items():
                frr1, far1 = errs(ca * r + cb, ca * f + cb, t)
                row.update({f"actDCF_{m}": beta * frr1 / 100 + far1 / 100, f"frr_{m}": frr1, f"far_{m}": far1})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "calibration.csv"), index=False)
    pd.DataFrame(params).to_csv(os.path.join(a.out, "params.csv"), index=False)

    pd.set_option("display.width", 250)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print(f"β = {beta:.3f}, ngưỡng Bayes t = {t:.3f}, prior hiệu dụng P(thật) = {prior:.3f}")
    print("Hiệu chuẩn học trên seaspoof_vi_dev. affine: s' = a·s + b; shift: a = 1, chỉ học b; "
          "platt: affine với nhãn làm mềm.")
    print("\n== Hệ số")
    print(pd.DataFrame(params).to_string(index=False, float_format=fmt))
    cols = ["config", "eer", "minDCF", "actDCF_raw"] + [f"actDCF_{m}" for m in modes] + \
           ["frr_raw"] + [f"frr_{m}" for m in modes] + ["far_raw"] + [f"far_{m}" for m in modes]
    for name in ("radar_vi", "itw", "radar_en", "test_cvreal", "mlaad", "test_seen", "dev"):
        sub = out[out.set == name]
        if len(sub):
            print(f"\n== {name}: minDCF; actDCF trước (raw) và sau ba cách hiệu chuẩn; FRR, FAR (%) ở ngưỡng Bayes")
            print(sub[cols].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
