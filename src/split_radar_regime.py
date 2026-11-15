#!/usr/bin/env python3
"""Chia RADAR VI cho phép kiểm chế độ D1 (chore/GD6-GRPO.md mục 13.5, 14): fine-tune trên một nửa RADAR VI
(miền khó với S0: EER 20%; tiếng thật đa nguồn), chấm nửa còn lại làm tập trong miền. Kiểm chẩn đoán "GRPO không
giúp vì miền fine-tune SEA-Spoof đã bão hoà" bằng cách bỏ bão hoà mà giữ ngôn ngữ, mô hình, code.

  python src/split_radar_regime.py                      ghi protocols/ids/radar_vi_rv_{train,dev,test}.tsv
  python src/split_radar_regime.py --ids-dir ids --make-csv SRC.csv   (trên server) tạo radar_vi_rv_*.csv cạnh SRC.csv

Chia ngẫu nhiên phân tầng theo nhãn (danh sách câu RADAR VI không có người nói hay nguồn: cột Speaker toàn "-"),
seed 2026: nửa A (50%) gồm train (90% của A) và dev (10% của A); nửa B (50%) là test. Vì không chia theo người nói,
tập test trong miền có thể dễ hơn thực tế (cùng người nói, cùng nguồn với train); ghi làm hạn chế khi báo.
"""
import argparse
import csv
import json
import os
import random

IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "protocols", "ids")
SPLITS = ("train", "dev", "test")


def make_split(seed=2026):
    rows = list(csv.DictReader(open(os.path.join(IDS, "radar_vi.tsv")), delimiter="\t"))
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for lab in ("real", "fake"):
        r = sorted((x for x in rows if x["Label"] == lab), key=lambda x: x["ID"])
        rng.shuffle(r)
        half = len(r) // 2
        a, b = r[:half], r[half:]
        n_dev = round(0.1 * len(a))
        out["dev"] += a[:n_dev]
        out["train"] += a[n_dev:]
        out["test"] += b
    for s in SPLITS:
        out[s].sort(key=lambda x: x["ID"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-csv", default=None, help="CSV RADAR VI đầy đủ (data/prep/radar_vi.csv trên server)")
    ap.add_argument("--ids-dir", default=None, help="thư mục danh sách câu (mặc định protocols/ids; server: ids)")
    a = ap.parse_args()
    global IDS
    if a.ids_dir:
        IDS = a.ids_dir
    if a.make_csv:
        src = list(csv.DictReader(open(a.make_csv)))
        fields = list(src[0].keys())
        by_id = {r["ID"]: r for r in src}
        for s in SPLITS:
            ids = [r["ID"] for r in csv.DictReader(open(os.path.join(IDS, f"radar_vi_rv_{s}.tsv")), delimiter="\t")]
            missing = [i for i in ids if i not in by_id]
            if missing:
                raise SystemExit(f"{len(missing)} ID của radar_vi_rv_{s} không có trong {a.make_csv}, ví dụ {missing[:3]}")
            dst = os.path.join(os.path.dirname(a.make_csv), f"radar_vi_rv_{s}.csv")
            with open(dst, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(by_id[i] for i in ids)
            print(f"{dst}: {len(ids)} câu")
        return
    out = make_split()
    stats = {}
    for s in SPLITS:
        with open(os.path.join(IDS, f"radar_vi_rv_{s}.tsv"), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["ID", "Duration", "Label"])
            for r in out[s]:
                w.writerow([r["ID"], r["Duration"], r["Label"]])
        stats[s] = {"n": len(out[s]), "real": sum(r["Label"] == "real" for r in out[s]),
                    "fake": sum(r["Label"] == "fake" for r in out[s]),
                    "hours": round(sum(float(r["Duration"]) for r in out[s]) / 3600, 3)}
    json.dump({"seed": 2026, "source": "radar_vi.tsv", "splits": stats},
              open(os.path.join(IDS, "radar_vi_rv_split.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
