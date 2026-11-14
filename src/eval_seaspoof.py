#!/usr/bin/env python3
"""Độ đo theo tập con cho SEA-Spoof tiếng Việt, từ MỘT file điểm của cả tập test (hoặc dev).

Tập test được chấm một lần (seaspoof_vi_test.csv, cột `group`), rồi tách:
  test_seen     real_vivos + seen          (hệ thống đã thấy khi huấn luyện)
  test_f5       real_vivos + f5_tts        (để riêng, mã nguồn mở, nhái giọng)
  test_minimax  real_vivos + minimax       (để riêng, thương mại)
  test_cvreal   real_cv    + seen          (tiếng thật từ nguồn khác: Common Voice)
  sys:<tên>     real_vivos + một hệ thống đã thấy (EER theo từng hệ thống)
Với dev (không có cột group khác "real_vivos"/"seen") chỉ có dòng gộp và theo hệ thống.

Mỗi dòng: EER, minDCF, actDCF (src/evaluate.py) và hai tỉ lệ lỗi ở ngưỡng log-odds 0
(FAR: giả bị nhận là thật; FRR: thật bị coi là giả), như GĐ3.

  python src/eval_seaspoof.py <scores.csv> --protocol Data/seaspoof/data/seaspoof_vi_test.csv \
      [--out metrics_subsets.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import all_metrics, dcf_beta, load_scores  # noqa: E402

SUBSETS = {"test_seen": (["real_vivos"], ["seen"]),
           "test_f5": (["real_vivos"], ["f5_tts"]),
           "test_minimax": (["real_vivos"], ["minimax"]),
           "test_cvreal": (["real_cv"], ["seen"])}


def row(name, real, fake, beta):
    m = all_metrics(real, fake, beta)
    if m is None:
        return None
    m["far0_%"] = 100.0 * float(np.mean(np.asarray(fake) >= 0))
    m["frr0_%"] = 100.0 * float(np.mean(np.asarray(real) < 0))
    m["subset"] = name
    return m


def report(scores_csv, protocol, beta=None):
    beta = beta or dcf_beta()
    s = load_scores(scores_csv)
    p = pd.read_csv(protocol)[["ID", "Label", "Attack"] + (["group"] if "group" in
                                                           pd.read_csv(protocol, nrows=1) else [])]
    d = s.merge(p, on="ID", how="inner")
    assert len(d) == len(s), f"{len(s) - len(d)} ID trong file điểm không có trong protocol"
    if "group" not in d:
        d["group"] = np.where(d.label == 1, "real_vivos", "seen")
    rows = [row("pooled", d.score[d.label == 1], d.score[d.label == 0], beta)]
    for name, (rg, fg) in SUBSETS.items():
        r, f = d[d.group.isin(rg) & (d.label == 1)], d[d.group.isin(fg) & (d.label == 0)]
        if len(r) and len(f):
            rows.append(row(name, r.score, f.score, beta))
    real = d[(d.group == "real_vivos") & (d.label == 1)].score
    for a in sorted(d[(d.group == "seen") & (d.label == 0)].Attack.unique()):
        rows.append(row(f"sys:{a}", real, d[(d.Attack == a) & (d.label == 0)].score, beta))
    cols = ["subset", "n_real", "n_fake", "eer_%", "minDCF", "actDCF", "far0_%", "frr0_%"]
    return pd.DataFrame([r for r in rows if r])[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    t = report(a.scores, a.protocol)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if a.out:
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
