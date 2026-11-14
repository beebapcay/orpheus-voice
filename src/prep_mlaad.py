#!/usr/bin/env python3
"""Tập kiểm tra hệ thống sinh tiếng Việt chưa gặp: tiếng giả `fake/vi` của MLAAD + tiếng thật VIVOS
của tập test SEA-Spoof (nhóm `real_vivos`, 14 người nói, cùng phần thật với test_f5, test_minimax).

Lý do (chore/GD5-Ablation.md phát hiện 2): hai hệ thống để riêng của SEA-Spoof (F5-TTS, MiniMax) đã
gần 0% EER với mô hình dùng thẳng, nên không phân biệt được các cấu hình.

  hf download mueller91/MLAAD --repo-type dataset --include "fake/vi/*" --local-dir Data/mlaad/raw
  python src/prep_mlaad.py build      # chép WAV, ghi CSV định dạng AntiDeepfake + danh sách ID
  python src/prep_mlaad.py check      # kiểm lối tắt bằng một đặc trưng thô

Giữ nguyên file WAV của MLAAD (22,05 kHz); `main.py` của AntiDeepfake tự resample về 16 kHz khi
`SampleRate` khác 16000 (dataio.py, test_audio_pipeline). Tiếng thật tham chiếu tới file FLAC đã có
trong dataset `orpheus-seaspoof-vi` ($ROOT/seaspoof_vi/flac/...), nên dataset MLAAD chỉ cần phần giả.

MLAAD: CC BY-NC 4.0, chỉ dùng nghiên cứu phi thương mại. Không commit âm thanh hay transcript;
dataset Kaggle để riêng tư. Chỉ commit danh sách ID (protocols/ids/mlaad_vi_test.tsv).
"""
import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SUBDIR = "mlaad_vi"
CSV_HEADER = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker", "Proportion",
              "AudioChannel", "AudioEncoding", "AudioBitSample", "Language"]


def log(msg):
    print(msg, flush=True)


def audio_feats(path):
    """Như prep_seaspoof.audio_info, thêm tỉ lệ năng lượng dải 4–8 kHz sau khi đưa về 16 kHz (giả
    và thật ở tập này từ hai kho, tần số lấy mẫu gốc khác nhau: kiểm lối tắt theo băng thông)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from prep_seaspoof import audio_info
    d = audio_info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = np.gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 16000)
    d["hi_band_db"] = round(float(10 * np.log10(spec[f >= 4000].sum() / (spec.sum() + 1e-12) + 1e-12)), 2)
    return d


def cmd_build(a):
    import pandas as pd
    src = os.path.join(a.raw, "fake", "vi")
    out_audio = os.path.join(a.out, SUBDIR)
    rows = []
    for system in sorted(os.listdir(src)):
        meta = os.path.join(src, system, "meta.csv")
        if not os.path.exists(meta):
            continue
        # QUOTE_NONE: transcript có dấu ngoặc kép; đọc mặc định làm gộp dòng (VieNeu-TTS ra 841 / 1.000)
        m = pd.read_csv(meta, sep="|", quoting=csv.QUOTE_NONE)
        for _, r in m.iterrows():
            name = os.path.basename(r["path"])
            stem = os.path.splitext(name)[0]
            rows.append({"ID": f"MLAADVI_{system}_{stem}", "Label": "fake", "sys": system,
                         "group": f"mlaad:{system}", "src": os.path.join(system, name),
                         "architecture": r.get("architecture", ""), "speaker": r.get("reference_speaker", "-")})
    fake = pd.DataFrame(rows)
    assert fake.ID.is_unique, "ID MLAAD trùng"
    missing = [p for p in fake.src if not os.path.exists(os.path.join(src, p))]
    if missing:
        sys.exit(f"Thiếu {len(missing)} file WAV, ví dụ {missing[:3]}: tải chưa xong?")
    log(f"MLAAD fake/vi: {len(fake)} câu, hệ thống: {fake.sys.value_counts().to_dict()}")

    for p in fake.src:
        d = os.path.join(out_audio, p)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not os.path.exists(d):
            shutil.copy(os.path.join(src, p), d)
    with ThreadPoolExecutor(8) as ex:
        info = list(ex.map(lambda p: audio_feats(os.path.join(out_audio, p)), fake.src))
    fake = pd.concat([fake.reset_index(drop=True), pd.DataFrame(info)], axis=1)
    fake["Path"] = "$ROOT/" + SUBDIR + "/" + fake.src

    # Tiếng thật: real_vivos của test SEA-Spoof (đã có trong orpheus-seaspoof-vi)
    test = pd.read_csv(os.path.join(a.seaspoof, "seaspoof_vi_test.csv"))
    real = test[test.group == "real_vivos"].copy()
    flac_root = a.seaspoof
    with ThreadPoolExecutor(8) as ex:
        rinfo = list(ex.map(lambda p: audio_feats(p.replace("$ROOT", flac_root)), real.Path))
    for k in ("rms_db", "peak", "lead_sil", "trail_sil", "hi_band_db"):
        real[k] = [r[k] for r in rinfo]
    real["sys"], real["speaker"], real["architecture"] = "real_vivos", real.Speaker, "-"
    log(f"Tiếng thật real_vivos: {len(real)} câu, {real.Speaker.nunique()} người nói")

    fake = fake.assign(Attack=fake.sys, Speaker="-", Proportion="-", Language="VI")
    cols = CSV_HEADER + ["group"]
    df = pd.concat([real[cols], fake[cols]], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "mlaad_vi_test.csv"), index=False)
    feats = ["ID", "Label", "sys", "Duration", "SampleRate", "rms_db", "peak", "lead_sil", "trail_sil", "hi_band_db"]
    pd.concat([real[feats], fake[feats]]).to_csv(os.path.join(a.out, "mlaad_vi_audio_info.csv"), index=False)
    ids = pd.concat([real[["ID", "Label", "sys", "group"]].assign(src="seaspoof_vi_test"),
                     fake[["ID", "Label", "sys", "group", "src", "architecture"]]])
    ids.to_csv(os.path.join(a.ids_dir, "mlaad_vi_test.tsv"), sep="\t", index=False)
    log(f"Ghi mlaad_vi_test.csv: {len(df)} câu ({len(real)} thật, {len(fake)} giả), "
        f"{df.Duration.sum() / 3600:.2f} giờ (giả {fake.Duration.sum() / 3600:.2f}), dài nhất "
        f"{df.Duration.max():.1f} s, tần số {sorted(df.SampleRate.unique())}")


def cmd_check(a):
    import pandas as pd
    from evaluate import compute_eer
    info = pd.read_csv(os.path.join(a.out, "mlaad_vi_audio_info.csv"))
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)
    p("Thống kê theo nhóm (trung vị)")
    t = info.groupby("sys").agg(n=("ID", "size"), hours=("Duration", lambda x: round(x.sum() / 3600, 2)),
                                dur=("Duration", "median"), rms_db=("rms_db", "median"),
                                lead_sil=("lead_sil", "median"), trail_sil=("trail_sil", "median"),
                                hi_band_db=("hi_band_db", "median"), sr=("SampleRate", "first"))
    p(t.to_string())
    p("\nEER (%) khi chỉ dùng MỘT đặc trưng làm điểm, real_vivos so với từng hệ thống "
      "(50% = không tách được; thấp = có lối tắt)")
    real = info[info.Label == "real"]
    res = []
    for s in ["all"] + sorted(info[info.Label == "fake"].sys.unique()):
        fake = info[info.Label == "fake"] if s == "all" else info[info.sys == s]
        r = {"fake": s, "n_fake": len(fake)}
        for f in ["Duration", "rms_db", "peak", "lead_sil", "trail_sil", "hi_band_db"]:
            e = min(compute_eer(real[f].values, fake[f].values)[0],
                    compute_eer(-real[f].values, -fake[f].values)[0]) * 100
            r[f] = round(e, 1)
        res.append(r)
    p(pd.DataFrame(res).to_string(index=False))
    os.makedirs(a.results, exist_ok=True)
    pd.DataFrame(res).to_csv(os.path.join(a.results, "mlaad_shortcut_eer.csv"), index=False)
    t.to_csv(os.path.join(a.results, "mlaad_stats.csv"))
    open(os.path.join(a.results, "mlaad_check_output.txt"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--raw", default="Data/mlaad/raw")
    ap.add_argument("--out", default="Data/mlaad/data", help="thư mục dữ liệu ($ROOT) của dataset Kaggle")
    ap.add_argument("--seaspoof", default="Data/seaspoof/data", help="$ROOT của SEA-Spoof (đã fetch)")
    ap.add_argument("--ids-dir", default="protocols/ids")
    ap.add_argument("--results", default="results/mlaad")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
