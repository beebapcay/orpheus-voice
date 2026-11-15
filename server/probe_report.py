#!/usr/bin/env python3
"""Đọc các lần thử đo (tầng 0) và chốt số worker đọc dữ liệu. Chỉ dùng thư viện chuẩn; chạy trên máy
sau `vast.sh fetch` (hoặc trên server).

  python3 server/probe_report.py --train results/server/raw/train [--summary results/server/raw/train_summary.csv]

Vì sao không dùng thẳng gpu_util_mean của train_summary.csv: cột đó tính trên cả lần chạy, gồm 1–2 phút
nạp mô hình (GPU 0%) và kiểm tra dev cuối. Lần đo chỉ 200 bước nên phần này kéo trung bình xuống nhiều.
Ở đây chỉ lấy các mẫu monitor.csv (mỗi 5 giây) trong khoảng huấn luyện: từ mẫu đầu tiên bộ nhớ GPU đạt
≥ 60% đỉnh của lần đó đến mẫu cuối cùng như vậy, bỏ thêm 2 mẫu đầu (bước đầu chậm do khởi tạo cuDNN).
CPU là mức dùng của cả máy (các lần chạy song song cùng chia CPU).

Quy tắc chốt (chore/GD6-GRPO.md, server/README.md mục 4): số worker nhỏ nhất cho GPU dùng ≥ 90% ở cả SFT
và GRPO; giữ nguyên cho cả bộ.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<[\d:?]+,\s+([\d.]+)(s/it|it/s), train_loss")
MEM = re.compile(r"MEM_PROBE max_allocated_mib=(\d+) max_reserved_mib=(\d+) total_mib=(\d+)")


def secs(hms):
    s = 0
    for x in hms.split(":"):
        s = 60 * s + int(x)
    return s


def one(d):
    name = os.path.basename(d.rstrip("/"))
    log = open(os.path.join(d, "main.log"), errors="replace").read().replace("\r", "\n") \
        if os.path.exists(os.path.join(d, "main.log")) else ""
    bars = BAR.findall(log)
    mem = MEM.findall(log)
    rows = list(csv.DictReader(open(os.path.join(d, "monitor.csv")))) if os.path.exists(os.path.join(d, "monitor.csv")) else []
    rows = [r for r in rows if r["mem_mib"] not in ("", None)]
    out = {"run": name, "steps": "", "s_per_step_bar": "", "s_per_step_avg": "", "peak_alloc_mib": "",
           "peak_reserved_mib": "", "gpu_util_train": "", "gpu_util_p10": "", "cpu_util_train": "", "n_samples": 0}
    if bars:
        n, _tot, el, rate, unit = bars[-1]
        out["steps"] = int(n)
        out["s_per_step_bar"] = round(float(rate) if unit == "s/it" else 1 / float(rate), 3)
        out["s_per_step_avg"] = round(secs(el) / int(n), 3) if int(n) else ""
    if mem:
        out["peak_alloc_mib"], out["peak_reserved_mib"] = int(mem[-1][0]), int(mem[-1][1])
    if rows:
        peak = max(float(r["mem_mib"]) for r in rows)
        idx = [i for i, r in enumerate(rows) if float(r["mem_mib"]) >= 0.6 * peak]
        win = rows[idx[0] + 2: idx[-1] + 1] if len(idx) > 4 else rows[idx[0]: idx[-1] + 1] if idx else []
        gu = [float(r["gpu_util"]) for r in win if r["gpu_util"] not in ("", None)]
        cu = [float(r["cpu_util"]) for r in win if r["cpu_util"] not in ("", None)]
        if gu:
            out["gpu_util_train"] = round(st.mean(gu), 1)
            out["gpu_util_p10"] = sorted(gu)[max(0, int(0.1 * len(gu)) - 1)]
        if cu:
            out["cpu_util_train"] = round(st.mean(cu), 1)
        out["n_samples"] = len(gu)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="thư mục train/ (mỗi lần một thư mục con)")
    ap.add_argument("--pattern", default="probe_*")
    ap.add_argument("--summary", default=None, help="train_summary.csv: thêm cột gpu_util_mean cả lần chạy để so")
    a = ap.parse_args()
    summ = {r["run"]: r for r in csv.DictReader(open(a.summary))} if a.summary and os.path.exists(a.summary) else {}
    res = [one(d) for d in sorted(glob.glob(os.path.join(a.train, a.pattern))) if os.path.isdir(d)]
    keys = ["run", "steps", "s_per_step_bar", "s_per_step_avg", "peak_alloc_mib", "peak_reserved_mib",
            "gpu_util_train", "gpu_util_p10", "cpu_util_train", "n_samples"]
    if summ:
        keys.append("gpu_util_mean_all")
        for r in res:
            r["gpu_util_mean_all"] = summ.get(r["run"], {}).get("gpu_util_mean", "")
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in res:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    print("\ns_per_step_bar: tốc độ tqdm hiển thị cuối (trung bình trượt); s_per_step_avg: thời gian epoch / số "
          "bước (gồm bước đầu chậm). gpu_util_*: chỉ trong khoảng huấn luyện; p10: phân vị 10 (GPU có lúc "
          "chờ dữ liệu không). Giây mỗi bước, bộ nhớ: thấp hơn là tốt hơn; GPU dùng: cao hơn là tốt hơn.")

    # Khuyến nghị số worker: nhỏ nhất mà GPU dùng ≥ 90% trong khoảng huấn luyện ở cả SFT (R) và GRPO (G1)
    util = {r["run"]: r["gpu_util_train"] for r in res if r["gpu_util_train"] != ""}
    rec = None
    for w in (2, 6):
        u = [util.get(f"probe_{c}_w{w}") for c in ("R", "G1")]
        if all(x is not None for x in u):
            if min(u) >= 90:
                rec = w
                break
            rec = w          # chưa đạt 90% thì lấy số worker lớn nhất đã đo
    if rec is not None:
        print(f"\nWORKERS_RECOMMENDED={rec}  (nhỏ nhất cho GPU dùng ≥ 90% ở cả R và G1; không đạt thì lấy số lớn nhất "
              f"đã đo; GPU dùng: {', '.join(f'{k} {v}%' for k, v in sorted(util.items()))})")

    # Dấu vân tay batch (BATCH_HASH_N, src/mem_probe.py): cùng seed, cùng số worker thì SFT và GRPO có nhận
    # cùng batch (ids) và cùng phép tăng cường RawBoost (wav) không. Quyết định phân tích qua seed có ghép
    # cặp được không (chore/GD6-GRPO.md mục 13)
    fp = {}
    for d in sorted(glob.glob(os.path.join(a.train, a.pattern))):
        p = os.path.join(d, "main.log")
        if os.path.exists(p):
            hits = re.findall(r"BATCH_HASH pid=\d+ ids=(\w+) n=\d+ wav=(\w+)", open(p, errors="replace").read())
            if hits:
                fp[os.path.basename(d)] = hits
    if len(fp) >= 2:
        print("\n| lần A | lần B | batch của A | cùng danh sách câu ở B | cùng cả sóng âm (sau RawBoost) |")
        print("|---|---|---|---|---|")
        names = sorted(fp)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                ids_y, both_y = {h[0] for h in fp[y]}, set(fp[y])
                print(f"| {x} | {y} | {len(fp[x])} | {sum(h[0] in ids_y for h in fp[x])} | "
                      f"{sum(h in both_y for h in fp[x])} |")
        print("\nCùng seed và cùng số worker mà trùng cả sóng âm: các cấu hình cùng đường dữ liệu nhận đúng cùng "
              "batch và cùng phép tăng cường, chỉ khác dropout và phép lấy mẫu của GRPO, nên so qua seed được "
              "ghép cặp theo seed. Khác số worker thì chỉ trùng danh sách câu (sampler dùng seed cố định).")


if __name__ == "__main__":
    main()
