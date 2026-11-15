#!/usr/bin/env python3
"""Kiểm dữ liệu trên server trước khi chạy: mỗi CSV có đúng số câu (và đúng ID) như danh sách đã
commit ở protocols/ids/, và mọi file âm thanh tồn tại. Chỉ dùng thư viện chuẩn.

  python server/check_data.py --root /workspace/orpheus
"""
import argparse
import csv
import os
import sys

# CSV (trong data/<nguồn>/) -> danh sách ID (protocols/ids/, chép vào <root>/ids/)
CHECKS = [
    ("seaspoof", "seaspoof_vi_train", "seaspoof_vi_train.tsv"),
    ("seaspoof", "seaspoof_vi_dev", "seaspoof_vi_dev.tsv"),
    ("seaspoof", "seaspoof_vi_test", "seaspoof_vi_test.tsv"),
    ("seaspoof", "seaspoof_vi_train_n50", None),
    ("seaspoof", "seaspoof_vi_train_n200", None),
    ("seaspoof", "seaspoof_vi_train_n1000", None),
    ("mlaad", "mlaad_vi_test", "mlaad_vi_test.tsv"),
    ("prep", "radar_vi", "radar_vi.tsv"),
    ("prep", "radar_en_sub", "radar_en_sub.tsv"),
    ("prep", "itw_sub", "itw_sub.tsv"),
    ("prep", "ood_forget", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ORPHEUS_ROOT", "/workspace/orpheus"))
    a = ap.parse_args()
    bad = 0
    # $ROOT trong cột Path = <root>/work/Data, nơi finetune.py gắn thư mục con của MỌI data/<nguồn>
    # (ví dụ MLAAD dùng tiếng thật trong seaspoof_vi/), nên tìm file trong mọi nguồn
    srcs = [os.path.join(a.root, "data", s) for s in ("seaspoof", "prep", "mlaad")]

    def exists(path):
        rel = path.replace("$ROOT/", "", 1)
        return any(os.path.exists(os.path.join(s, rel)) for s in srcs)

    for src, name, ids in CHECKS:
        p = os.path.join(a.root, "data", src, f"{name}.csv")
        if not os.path.exists(p):
            print(f"THIẾU {p}")
            bad += 1
            continue
        rows = list(csv.DictReader(open(p)))
        real = sum(r["Label"] == "real" for r in rows)
        missing = [r["Path"] for r in rows if not exists(r["Path"])]
        msg = f"{name}: {len(rows)} câu ({real} thật, {len(rows) - real} giả)"
        if ids:
            ref = {r["ID"] for r in csv.DictReader(open(os.path.join(a.root, "ids", ids)), delimiter="\t")}
            got = {r["ID"] for r in rows}
            if ref != got:
                msg += f"; KHÁC danh sách {ids}: thiếu {len(ref - got)}, thừa {len(got - ref)}"
                bad += 1
            else:
                msg += f"; khớp {ids}"
        if missing:
            msg += f"; THIẾU {len(missing)} file âm thanh, ví dụ {missing[:2]}"
            bad += 1
        print(msg)
    print("DỮ LIỆU OK" if not bad else f"DỮ LIỆU LỖI ({bad} mục)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
