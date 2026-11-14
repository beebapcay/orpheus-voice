#!/usr/bin/env python3
"""Chuẩn bị dữ liệu SEA-Spoof tiếng Việt cho GĐ4–6 (fine-tune).

SEA-Spoof (HF Jack-ppkdczgx/SEA-Spoof, 81,5 GB, 38 file parquet) trộn 7 ngôn ngữ và
nhúng âm thanh FLAC trong cột `audio.bytes`. Mỗi row group chỉ 200 câu, nên lấy được
đúng phần tiếng Việt mà không tải cả bộ.

Giấy phép: phi thương mại, KHÔNG phân phối lại. Âm thanh và metadata đầy đủ (có
transcript) chỉ để ở máy (`Data/`, đã gitignore) hoặc output kernel Kaggle riêng tư.
Chỉ commit danh sách câu (protocols/ids/seaspoof_vi_*.tsv) và thống kê.

Các bước:

  meta    (máy) Đọc các cột nhẹ (mọi cột trừ audio.bytes, ~49 MB trên 81,5 GB) của
          38 file qua HTTP Range, ghi Data/seaspoof/meta.parquet kèm vị trí từng câu
          (file, row group, thứ tự trong row group) và dung lượng âm thanh của từng
          row group (lấy từ footer).
  stats   (máy) Thống kê phần `vi`: theo split, nhãn, hệ thống sinh, người nói, nguồn;
          người nói và văn bản trùng giữa các split có sẵn; giờ ước lượng. Đầu ra không có
          transcript hay ID câu: results/seaspoof/stats_vi.txt.
  select  (máy) Chọn và chia tập theo cách đã chốt, ghi protocols/ids/seaspoof_vi_*.tsv.
  fetch   (máy hoặc Kaggle) Tải các row group chứa câu đã chọn, ghi FLAC và CSV theo
          định dạng protocol của AntiDeepfake.

Revision dataset cố định (REV) để mọi lần chạy đọc đúng một bản.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "Jack-ppkdczgx/SEA-Spoof"
REV = "132f5dca9b6efe39cf1d3b54a858f167f5a421fc"      # commit HF ngày 21/07/2026
LIGHT_COLS = ["row_id", "utterance_id", "audio.path", "text", "language", "label",
              "spoof_type", "category", "split", "text_source", "mapping_source",
              "text_granularity", "is_text_exact", "source_model", "source_dataset",
              "speaker_or_voice", "sampling_rate", "audio_was_resampled"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def base_path():
    return f"datasets/{REPO}@{REV}/data"


# ---------------------------------------------------------------- meta

def _read_rgs(fs, path, rgs, cols):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = []
    for attempt in range(5):
        try:
            with fs.open(path, "rb", block_size=1 << 16, cache_type="readahead") as fh:
                pf = pq.ParquetFile(fh)
                md = pf.metadata
                audio_idx = [i for i in range(md.num_columns)
                             if md.schema.column(i).path == "audio.bytes"][0]
                for rg in rgs:
                    t = pf.read_row_group(rg, columns=cols)
                    n = t.num_rows
                    abytes = md.row_group(rg).column(audio_idx).total_compressed_size
                    t = t.append_column("rg", pa.array([rg] * n, pa.int32()))
                    t = t.append_column("pos", pa.array(range(n), pa.int32()))
                    t = t.append_column("rg_audio_bytes", pa.array([abytes] * n, pa.int64()))
                    out.append(t)
            return out
        except Exception as e:  # noqa: BLE001
            log(f"  lỗi {path} rg {rgs[0]}–{rgs[-1]} (lần {attempt + 1}): {e}")
            out = []
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Không đọc được {path}")


def cmd_meta(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    fs = hf_fs()
    files = sorted(fs.glob(base_path() + "/*/*.parquet"))
    log(f"{len(files)} file parquet ở revision {REV[:10]}")
    tasks = []
    for f in files:
        with fs.open(f, "rb", block_size=1 << 16) as fh:
            n_rg = pq.ParquetFile(fh).metadata.num_row_groups
        for s in range(0, n_rg, args.chunk):
            tasks.append((f, list(range(s, min(s + args.chunk, n_rg)))))
    log(f"{len(tasks)} phần việc, {args.workers} luồng")
    tables, t0, done = [], time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_read_rgs, fs, f, rgs, LIGHT_COLS): f for f, rgs in tasks}
        for fu in as_completed(futs):
            f = futs[fu]
            name = f.split("/data/")[1]
            for t in fu.result():
                tables.append(t.append_column("file", pa.array([name] * t.num_rows)))
            done += 1
            if done % 10 == 0 or done == len(tasks):
                log(f"  {done}/{len(tasks)} phần, {(time.time() - t0) / 60:.1f} phút")
    t = pa.concat_tables(tables)
    df = t.to_pandas()
    # cột audio đọc ra là struct {path}: làm phẳng thành audio_path
    df["audio_path"] = df.pop("audio").map(lambda a: a["path"] if a else None)
    df = df.sort_values(["file", "rg", "pos"]).reset_index(drop=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "meta.parquet")
    df.to_parquet(path, index=False)
    log(f"Ghi {path}: {len(df)} dòng, {os.path.getsize(path) / 1e6:.1f} MB")
    json.dump({"repo": REPO, "revision": REV, "rows": int(len(df)),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(args.out, "meta_info.json"), "w"), indent=1)


# ---------------------------------------------------------------- stats

# Byte FLAC mỗi giây, để ước lượng giờ khi chưa tải âm thanh: hiệu chỉnh sao cho tiếng thật
# VIVOS (12.420 câu) ra ~15 giờ như mô tả của VIVOS. Chỉ là ước lượng (sai ±15–20%); số
# giờ thật đo sau khi tải.
EST_BYTES_PER_SEC = 18.1e3


def system_of(source_model, source_dataset):
    """Tên hệ thống gọn từ source_model (tên gốc có lỗi chính tả: 'vienamese_...')."""
    sm = source_model.lower()
    if sm == "bonafide":
        return "real_vivos" if source_dataset == "vivos" else "real_cv"
    for key, name in [("vits", "vits"), ("edge", "edge_tts"), ("f5", "f5_tts"),
                      ("chatgpt", "chatgpt"), ("eleven", "elevenlabs"), ("heygen", "heygen"),
                      ("minimax", "minimax")]:
        if key in sm:
            return name
    return sm


def load_vi(meta):
    """Phần tiếng Việt, thêm cột: sys (hệ thống), est_sec (giây ước lượng), t (văn bản chuẩn
    hoá), spk (người nói VIVOS, lấy từ utterance_id)."""
    import pandas as pd
    df = pd.read_parquet(meta)
    rg_rows = df.groupby(["file", "rg"]).size().rename("rg_rows")
    vi = df[df.language == "vi"].join(rg_rows, on=["file", "rg"]).copy()
    vi["est_sec"] = vi.rg_audio_bytes / vi.rg_rows / EST_BYTES_PER_SEC
    vi["sys"] = [system_of(a, b) for a, b in zip(vi.source_model, vi.source_dataset)]
    vi["t"] = vi.text.fillna("").str.strip().str.lower()
    vi["spk"] = vi.utterance_id.str.extract(r"^(VIVOS[A-Z]+\d+)_")[0]
    return df, vi


def cmd_stats(args):
    """In thống kê; không in transcript hay ID câu (để có thể commit đầu ra)."""
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    df, vi = load_vi(args.meta)
    print(f"Revision {REV}. Tổng {len(df)} dòng")
    print(df.groupby("language").size().to_string())
    print(f"\n== Tiếng Việt: {len(vi)} câu")
    for c in ["split", "label", "spoof_type", "category", "source_model", "source_dataset",
              "mapping_source", "text_granularity", "is_text_exact", "sampling_rate",
              "audio_was_resampled", "text_source"]:
        print(f"\n-- {c}\n{vi[c].value_counts(dropna=False).to_string()}")
    print("\n-- hệ thống × split (số câu)")
    print(vi.groupby(["sys", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\n-- giờ ước lượng ({EST_BYTES_PER_SEC / 1e3:.1f} KB/s, sai ±15–20%) hệ thống × split")
    print((vi.groupby(["sys", "split"]).est_sec.sum().unstack(fill_value=0) / 3600)
          .round(1).to_string())
    print(f"tổng ~{vi.est_sec.sum() / 3600:.0f} giờ")
    keys = vi[["file", "rg"]].drop_duplicates()
    n_lang = df.groupby(["file", "rg"]).language.nunique()
    mixed = sum(n_lang.get((f, r), 1) > 1 for f, r in keys.itertuples(index=False))
    gb = vi.drop_duplicates(["file", "rg"]).rg_audio_bytes.sum() / 1e9
    print(f"\n-- vị trí: {len(keys)} row group chứa câu vi ({mixed} row group trộn ngôn ngữ), "
          f"tổng âm thanh {gb:.2f} GB")
    print(vi.groupby("file").size().to_string())

    print("\n-- speaker_or_voice: số giá trị khác nhau theo hệ thống × split (0 hoặc 1 = trống)")
    print(vi.groupby(["sys", "split"]).speaker_or_voice.nunique().unstack(fill_value=0).to_string())
    for s in ["elevenlabs", "heygen", "minimax"]:
        g = vi[vi.sys == s].groupby("split").speaker_or_voice.apply(set)
        print(f"   {s}: giọng train∩evaluation = {len(g['train'] & g['evaluation'])}, "
              f"train∩validation = {len(g['train'] & g['validation'])}; "
              f"số câu theo giọng (lớn nhất) {vi[vi.sys == s].speaker_or_voice.value_counts().max()}")

    rv = vi[vi.sys == "real_vivos"]
    by = rv.groupby("split").spk.apply(set)
    print(f"\n-- VIVOS thật: {rv.spk.nunique()} người nói "
          f"({rv.spk.str.extract(r'VIVOS([A-Z]+)')[0].value_counts().to_dict()}); "
          f"theo split {({k: len(v) for k, v in by.items()})}; "
          f"trùng train∩evaluation {len(by['train'] & by['evaluation'])}, "
          f"train∩validation {len(by['train'] & by['validation'])}")
    print("   câu mỗi người nói (min / trung vị / max):",
          rv.groupby("spk").size().describe()[["min", "50%", "max"]].tolist())
    ts = vi.groupby("split").t.apply(set)
    print(f"\n-- văn bản: {vi.t.nunique()} câu khác nhau trên {len(vi)} dòng; "
          f"trùng train∩evaluation {len(ts['train'] & ts['evaluation'])}, "
          f"train∩validation {len(ts['train'] & ts['validation'])}")
    vivos_t, cv_t = set(rv.t), set(vi[vi.sys == "real_cv"].t)
    print("   tỉ lệ câu giả có văn bản trùng câu thật VIVOS / Common Voice:")
    for s, g in vi[vi.label == "spoof"].groupby("sys"):
        print(f"   {s:<11} VIVOS {g.t.isin(vivos_t).mean():6.1%}   CV {g.t.isin(cv_t).mean():6.1%}")
    cv = vi[vi.sys == "real_cv"]
    print(f"\n-- Common Voice thật: {len(cv)} clip, {cv.t.nunique()} câu khác nhau; "
          f"số clip (min/max) {cv.utterance_id.str.extract(r'(\d+)$')[0].astype(int).agg(['min', 'max']).tolist()}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("meta")
    m.add_argument("--out", default="Data/seaspoof")
    m.add_argument("--workers", type=int, default=24)
    m.add_argument("--chunk", type=int, default=15, help="số row group mỗi phần việc")
    s = sub.add_parser("stats")
    s.add_argument("--meta", default="Data/seaspoof/meta.parquet")
    args = ap.parse_args()
    {"meta": cmd_meta, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
