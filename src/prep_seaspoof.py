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


# ---------------------------------------------------------------- select
#
# Cách chia đã chốt ngày 14/11 (chore/GD4-Du-Lieu-Va-Pipeline.md mục 3):
#   D1  tiếng thật Common Voice không vào train/dev; 2.000 clip làm tập kiểm tra riêng
#   D2  để riêng F5-TTS và MiniMax (không có trong train/dev)
#   D3  train ~20 giờ (toàn bộ tiếng thật VIVOS của người nói train + cùng số câu giả),
#       kèm tập con lồng nhau: 10 giờ, và N = 1000/200/50 câu mỗi lớp (GĐ6)
# Không dùng split có sẵn của SEA-Spoof (trùng cả 65 người nói VIVOS).

SEED = 2026
SEEN = ["vits", "edge_tts", "chatgpt", "elevenlabs", "heygen"]
HELDOUT = ["f5_tts", "minimax"]
FRAC = {"test": 0.20, "dev": 0.10}          # phần còn lại là train
N_CVREAL = 2000
SHOTS = [1000, 200, 50]


def _assign_groups(sizes, seed):
    """sizes: Series (nhóm -> số câu). Xếp ngẫu nhiên rồi gán theo tỉ lệ số câu cộng dồn:
    20% đầu -> test, 10% tiếp -> dev, còn lại -> train."""
    import numpy as np
    order = sizes.sort_index().sample(frac=1, random_state=seed)
    cum = order.cumsum() / order.sum()
    t, d = FRAC["test"], FRAC["test"] + FRAC["dev"]
    return dict(zip(order.index, np.where(cum <= t, "test", np.where(cum <= d, "dev", "train"))))


def _sample_balanced(pool, n, systems, seed):
    """Lấy n câu giả chia đều cho các hệ thống; hệ thống thiếu câu thì lấy hết và chia
    phần thiếu cho các hệ thống còn lại."""
    import pandas as pd
    quota = {s: 0 for s in systems}
    avail = {s: int((pool.sys == s).sum()) for s in systems}
    left, open_ = n, [s for s in systems if avail[s] > 0]
    while left > 0 and open_:
        share = max(left // len(open_), 1)
        for s in list(open_):
            take = min(share, avail[s] - quota[s], left)
            quota[s] += take
            left -= take
            if quota[s] >= avail[s]:
                open_.remove(s)
            if left == 0:
                break
    parts = [pool[pool.sys == s].sort_values("ID").sample(n=quota[s], random_state=seed)
             for s in systems if quota[s]]
    return pd.concat(parts), quota


def cmd_select(args):
    import pandas as pd
    _, vi = load_vi(args.meta)
    vi = vi.rename(columns={"utterance_id": "ID"})
    assert vi.ID.is_unique
    vi["Label"] = vi.label.map({"bonafide": "real", "spoof": "fake"})
    is_vivos_text = vi.source_dataset.eq("vivos")
    is_cv_text = (vi.sys.isin(["vits", "edge_tts", "f5_tts"]) & vi.source_model.str.lower()
                  .str.contains("common")) | vi.sys.isin(["chatgpt", "real_cv"])
    vi["text_src"] = "other"
    vi.loc[is_vivos_text, "text_src"] = "vivos"
    vi.loc[is_cv_text, "text_src"] = "cv"

    # 1. Người nói VIVOS -> tập
    rv = vi[vi.sys == "real_vivos"]
    spk_set = _assign_groups(rv.groupby("spk").size(), SEED)
    # 2. Văn bản VIVOS -> tập của người nói đọc nó (đọc bởi người nói ở >1 tập: bỏ)
    rv_set = rv.assign(s=rv.spk.map(spk_set)).groupby("t").s
    vivos_text_set = rv_set.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "drop")
    # 3. Văn bản Common Voice -> tập, theo nhóm văn bản (tính trên số câu dùng văn bản đó)
    cv_text_set = _assign_groups(vi[vi.text_src == "cv"].groupby("t").size(), SEED + 1)
    # 4. Giọng ElevenLabs, video HeyGen -> tập
    el_set = _assign_groups(vi[vi.sys == "elevenlabs"].groupby("speaker_or_voice").size(), SEED + 2)
    hg_set = _assign_groups(vi[vi.sys == "heygen"].groupby("speaker_or_voice").size(), SEED + 3)

    def where(r):
        if r.sys == "minimax":
            return "test"
        if r.sys == "real_vivos":
            return spk_set[r.spk]
        if r.sys == "elevenlabs":
            return el_set[r.speaker_or_voice]
        if r.sys == "heygen":
            return hg_set[r.speaker_or_voice]
        if r.text_src == "vivos":
            return vivos_text_set.get(r.t, "drop")
        if r.text_src == "cv":
            return cv_text_set[r.t]
        return "drop"

    vi["set"] = [where(r) for r in vi.itertuples()]
    # Không để câu ở dev/test trùng văn bản với câu KHÁC nhãn ở tập trước đó (train, rồi dev).
    # Trùng còn lại sau bước này là cùng nhãn: 12 giọng ElevenLabs đọc chung một bộ câu,
    # câu ngắn phổ biến do ASR nhận dạng (HeyGen, MiniMax), câu nhắc VIVOS nhiều người đọc.
    usable = vi.sys.isin(SEEN + ["real_vivos"])

    def texts(sets, label):
        return set(vi.t[vi.set.isin(sets) & vi.Label.eq(label) & usable])
    dropped = {}
    for s, before in [("dev", ["train"]), ("test", ["train", "dev"])]:
        bad = (vi.set == s) & ((vi.Label.eq("real") & vi.t.isin(texts(before, "fake"))) |
                               (vi.Label.eq("fake") & vi.t.isin(texts(before, "real"))))
        dropped[s] = vi[bad].groupby("sys").size().to_dict()
        vi.loc[bad, "set"] = "drop"
    log(f"Bỏ vì trùng văn bản khác nhãn với tập trước: {dropped}")
    # Người nói (hoặc giọng) để ghi vào CSV: VIVOS theo văn bản cho câu giả đọc văn bản VIVOS
    text_spk = rv.groupby("t").spk.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "-")
    vi["speaker"] = vi.spk
    m = vi.speaker.isna() & vi.text_src.eq("vivos")
    vi.loc[m, "speaker"] = vi.loc[m, "t"].map(text_spk)
    m = vi.sys.isin(["elevenlabs", "heygen", "minimax"])
    vi.loc[m, "speaker"] = vi.loc[m, "speaker_or_voice"]
    vi["speaker"] = vi.speaker.fillna("-")

    cols = ["ID", "Label", "sys", "speaker", "text_src", "file", "rg", "pos", "est_sec"]
    out = {}
    for s in ["train", "dev", "test"]:
        real = vi[(vi.sys == "real_vivos") & (vi.set == s)]
        pool = vi[vi.sys.isin(SEEN) & (vi.set == s)]
        fake, quota = _sample_balanced(pool, len(real), SEEN, SEED + 10)
        log(f"{s}: {len(real)} thật, giả theo hệ thống {quota}")
        part = pd.concat([real, fake])
        part["group"] = part.sys.where(part.Label == "fake", "real_vivos")
        part.loc[part.Label == "fake", "group"] = "seen"
        if s == "test":
            held = vi[vi.sys.isin(HELDOUT) & (vi.set == "test")].assign(group=lambda d: d.sys)
            cvr = vi[(vi.sys == "real_cv") & (vi.set == "test")].sort_values("ID")
            cvr = cvr.sample(n=min(N_CVREAL, len(cvr)), random_state=SEED + 20).assign(group="real_cv")
            part = pd.concat([part, held, cvr])
        out[s] = part.sort_values("ID")[cols + ["group"]].reset_index(drop=True)

    # Tập con lồng nhau của train: 10 giờ (một nửa, giữ tỉ lệ theo người nói / hệ thống), và
    # N câu mỗi lớp (giả chia đều 5 hệ thống); N=50 ⊂ 200 ⊂ 1000 ⊂ train_10h ⊂ train
    tr = out["train"]
    key = tr.speaker.where(tr.Label == "real", tr.sys)
    tr["in_10h"] = False
    half = tr.groupby(key, group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=SEED + 30))
    tr.loc[half.index, "in_10h"] = True
    tr["shot"] = 0
    sub = tr[tr.in_10h]
    real_rank = sub[sub.Label == "real"].sample(frac=1, random_state=SEED + 40).index
    fake = sub[sub.Label == "fake"]
    fake_rank = {s: fake[fake.sys == s].sample(frac=1, random_state=SEED + 41).index for s in SEEN}
    for n in SHOTS:                       # lớn trước, nhỏ ghi đè sau: shot = N nhỏ nhất chứa câu
        tr.loc[real_rank[:n], "shot"] = n
        for s in SEEN:
            tr.loc[fake_rank[s][:n // len(SEEN)], "shot"] = n
    out["train"] = tr

    os.makedirs(args.ids_dir, exist_ok=True)
    summary = {"repo": REPO, "revision": REV, "seed": SEED, "seen": SEEN, "heldout": HELDOUT,
               "frac": FRAC, "decisions": "chore/GD4-Du-Lieu-Va-Pipeline.md mục 3 (D1–D3, 14/11)",
               "speakers": {s: sorted(k for k, v in spk_set.items() if v == s)
                            for s in ["train", "dev", "test"]},
               "elevenlabs_voices": {s: sorted(k for k, v in el_set.items() if v == s)
                                     for s in ["train", "dev", "test"]},
               "heygen_videos": {s: sum(v == s for v in hg_set.values()) for s in ["train", "dev", "test"]},
               "cv_texts": {s: sum(v == s for v in cv_text_set.values()) for s in ["train", "dev", "test"]},
               "vivos_texts_dropped": int((vivos_text_set == "drop").sum()),
               "dropped_cross_label_text": dropped,
               "sets": {}}
    for s, df in out.items():
        path = os.path.join(args.ids_dir, f"seaspoof_vi_{s}.tsv")
        df.to_csv(path, sep="\t", index=False)
        st = {"n": int(len(df)), "real": int((df.Label == "real").sum()),
              "fake": int((df.Label == "fake").sum()),
              "est_hours": round(float(df.est_sec.sum()) / 3600, 1),
              "by_group_sys": {f"{g}/{y}": int(c) for (g, y), c in
                               df.groupby(["group", "sys"]).size().items()}}
        if s == "train":
            t10 = df[df.in_10h]
            st["train_10h"] = {"n": int(len(t10)), "real": int((t10.Label == "real").sum()),
                               "est_hours": round(float(t10.est_sec.sum()) / 3600, 1)}
            st["shots"] = {n: {"real": int(((df.shot > 0) & (df.shot <= n) & (df.Label == "real")).sum()),
                               "fake": int(((df.shot > 0) & (df.shot <= n) & (df.Label == "fake")).sum())}
                           for n in SHOTS}
        summary["sets"][s] = st
        log(f"Ghi {path}: {st['n']} câu ({st['real']} thật, {st['fake']} giả), ~{st['est_hours']} giờ")
    # Kiểm: không trùng người nói VIVOS và không trùng văn bản giữa các tập
    spk_by = {s: set(df.speaker[df.speaker.str.startswith("VIVOS")]) for s, df in out.items()}
    ids = pd.concat(out.values()).ID
    allv = vi.set_index("ID")
    txt_by = {s: set(allv.loc[df.ID, "t"]) for s, df in out.items()}

    def txt(s, label):
        df = out[s]
        return set(allv.loc[df.ID[df.Label == label], "t"])

    def voices(s, sys_):
        df = out[s]
        return set(df.speaker[df.sys == sys_])
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    check = {"speaker_overlap_vivos": {f"{a}&{b}": len(spk_by[a] & spk_by[b]) for a, b in pairs},
             "voice_overlap_elevenlabs": {f"{a}&{b}": len(voices(a, "elevenlabs") & voices(b, "elevenlabs"))
                                          for a, b in pairs},
             "video_overlap_heygen": {f"{a}&{b}": len(voices(a, "heygen") & voices(b, "heygen"))
                                      for a, b in pairs},
             "text_overlap_real_vs_fake": {f"{a}&{b}": len(txt(a, "real") & txt(b, "fake")) +
                                           len(txt(a, "fake") & txt(b, "real")) for a, b in pairs},
             "text_overlap_any": {f"{a}&{b}": len(txt_by[a] & txt_by[b]) for a, b in pairs},
             "id_unique": bool(ids.is_unique)}
    summary["check"] = check
    log(f"Kiểm tra: {json.dumps(check)}")
    json.dump(summary, open(os.path.join(args.ids_dir, "seaspoof_vi_split.json"), "w"),
              indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- fetch

CSV_HEADER = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample", "Language"]
SUBDIR = "seaspoof_vi/flac"


def read_ids(ids_dir):
    import pandas as pd
    return {s: pd.read_csv(os.path.join(ids_dir, f"seaspoof_vi_{s}.tsv"), sep="\t")
            for s in ["train", "dev", "test"]}


def _fetch_rg(fs, path, rg, want, flac_dir):
    """Đọc một row group (utterance_id + audio.bytes), ghi các câu cần. want: {pos: ID}."""
    import pyarrow.parquet as pq
    for attempt in range(5):
        try:
            with fs.open(path, "rb", block_size=1 << 22, cache_type="readahead") as fh:
                t = pq.ParquetFile(fh).read_row_group(rg, columns=["utterance_id", "audio"])
            ids = t.column("utterance_id").to_pylist()
            audio = t.column("audio").combine_chunks().field("bytes")
            n = 0
            for pos, uid in want.items():
                assert ids[pos] == uid, f"{path} rg {rg} pos {pos}: {ids[pos]} != {uid}"
                dest = os.path.join(flac_dir, f"{uid}.flac")
                with open(dest + ".part", "wb") as f:
                    f.write(audio[pos].as_py())
                os.replace(dest + ".part", dest)
                n += 1
            return n, t.nbytes
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"  lỗi {path} rg {rg} (lần {attempt + 1}): {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Không tải được {path} rg {rg}")


def cmd_fetch(args):
    import pandas as pd
    sets = read_ids(args.ids_dir)
    allrows = pd.concat(sets.values()).drop_duplicates("ID")
    flac_dir = os.path.join(args.out, SUBDIR)
    os.makedirs(flac_dir, exist_ok=True)
    have = {f[:-5] for f in os.listdir(flac_dir) if f.endswith(".flac")}
    todo = allrows[~allrows.ID.isin(have)]
    groups = {k: dict(zip(g.pos, g.ID)) for k, g in todo.groupby(["file", "rg"])}
    log(f"{len(allrows)} câu cần, đã có {len(allrows) - len(todo)}; "
        f"cần đọc {len(groups)} row group")
    fs, base = hf_fs(), base_path()
    t0, done, nb = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_fetch_rg, fs, f"{base}/{f}", int(rg), want, flac_dir)
                for (f, rg), want in groups.items()]
        for fu in as_completed(futs):
            n, b = fu.result()
            done += 1
            nb += b
            if done % 20 == 0 or done == len(futs):
                el = time.time() - t0
                log(f"  {done}/{len(futs)} row group, {nb / 1e9:.2f} GB, "
                    f"{nb / 1e6 / max(el, 1):.1f} MB/s, {el / 60:.1f} phút")
    write_csvs(args.out, sets, flac_dir)


def audio_info(path):
    """Thông tin file và vài đặc trưng thô để kiểm lối tắt (độ dài, âm lượng, khoảng lặng)."""
    import numpy as np
    import soundfile as sf
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    frame = int(0.02 * sr)
    n = len(x) // frame
    e = 20 * np.log10(np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9)
    voiced = np.where(e > e.max() - 40)[0] if n else np.array([0])
    return {"Duration": round(info.frames / info.samplerate, 3), "SampleRate": info.samplerate,
            "AudioChannel": info.channels, "AudioEncoding": info.format,
            "AudioBitSample": info.subtype, "rms_db": round(20 * np.log10(rms), 2),
            "peak": round(float(np.abs(x).max()), 4),
            "lead_sil": round(voiced[0] * frame / sr, 3),
            "trail_sil": round((n - 1 - voiced[-1]) * frame / sr, 3)}


def write_csvs(out, sets, flac_dir, workers=8):
    """CSV theo định dạng protocol của AntiDeepfake (Path dùng tiền tố $ROOT/ như GĐ3)."""
    import pandas as pd
    allrows = pd.concat(sets.values()).drop_duplicates("ID")
    log(f"Đọc thông tin {len(allrows)} file âm thanh")
    with ThreadPoolExecutor(workers) as ex:
        infos = list(ex.map(lambda i: audio_info(os.path.join(flac_dir, f"{i}.flac")), allrows.ID))
    info = pd.DataFrame(infos, index=allrows.ID.values)
    info.to_csv(os.path.join(out, "seaspoof_vi_audio_info.csv"), index_label="ID")

    def to_csv(df, name):
        d = df.join(info, on="ID")
        d = d.assign(Path="$ROOT/" + SUBDIR + "/" + d.ID + ".flac", Attack=d.sys,
                     Speaker=d.speaker, Proportion="-", Language="VI")
        d[CSV_HEADER + ["group"] if "group" in d else CSV_HEADER].to_csv(
            os.path.join(out, f"{name}.csv"), index=False)
        log(f"Ghi {name}.csv: {len(d)} câu ({(d.Label == 'real').sum()} thật, "
            f"{(d.Label == 'fake').sum()} giả), {d.Duration.sum() / 3600:.2f} giờ, "
            f"dài nhất {d.Duration.max():.1f} s, tần số {sorted(d.SampleRate.unique())}, "
            f"kênh {sorted(d.AudioChannel.unique())}")
    tr = sets["train"]
    to_csv(tr, "seaspoof_vi_train")
    to_csv(tr[tr.in_10h], "seaspoof_vi_train_10h")
    for n in SHOTS:
        to_csv(tr[(tr.shot > 0) & (tr.shot <= n)], f"seaspoof_vi_train_n{n}")
    to_csv(sets["dev"], "seaspoof_vi_dev")
    to_csv(sets["test"], "seaspoof_vi_test")


# ---------------------------------------------------------------- check

def cmd_check(args):
    """Kiểm lối tắt: một đặc trưng thô (độ dài, âm lượng, khoảng lặng) tách được thật/giả
    tới mức nào. Ghi bảng giờ thật và EER của từng đặc trưng (cả hai chiều, lấy chiều tốt)."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import compute_eer
    info = pd.read_csv(os.path.join(args.data, "seaspoof_vi_audio_info.csv")).set_index("ID")
    sets = read_ids(args.ids_dir)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)
    p("Giờ âm thanh đo thật (từ file FLAC đã tải), theo tập × nhóm/hệ thống")
    rows = []
    for s, df in sets.items():
        d = df.join(info, on="ID")
        for (g, y), x in d.groupby(["group", "sys"]):
            rows.append({"set": s, "group": g, "sys": y, "n": len(x),
                         "hours": round(x.Duration.sum() / 3600, 2),
                         "mean_s": round(x.Duration.mean(), 2), "max_s": round(x.Duration.max(), 1),
                         "rms_db": round(x.rms_db.median(), 1),
                         "lead_sil": round(x.lead_sil.median(), 2),
                         "trail_sil": round(x.trail_sil.median(), 2)})
        p(f"  {s}: {len(d)} câu, {d.Duration.sum() / 3600:.2f} giờ "
          f"(thật {d[d.Label == 'real'].Duration.sum() / 3600:.2f}, "
          f"giả {d[d.Label == 'fake'].Duration.sum() / 3600:.2f}), "
          f"câu > 13 s: {(d.Duration > 13).sum()}")
    tab = pd.DataFrame(rows)
    p(tab.to_string(index=False))
    tr = sets["train"]
    p(f"  train_10h: {tr[tr.in_10h].join(info, on='ID').Duration.sum() / 3600:.2f} giờ")

    p("\nEER (%) khi chỉ dùng MỘT đặc trưng làm điểm (50% = không tách được; thấp = có lối tắt)")
    subsets = {"train": ("train", None), "dev": ("dev", None),
               "test_seen": ("test", ["real_vivos", "seen"]),
               "test_f5": ("test", ["real_vivos", "f5_tts"]),
               "test_minimax": ("test", ["real_vivos", "minimax"]),
               "test_cvreal": ("test", ["real_cv", "seen"])}
    feats = ["Duration", "rms_db", "peak", "lead_sil", "trail_sil"]
    res = []
    for name, (s, groups) in subsets.items():
        d = sets[s].join(info, on="ID")
        if groups:
            d = d[d.group.isin(groups)]
        r = {"subset": name, "n": len(d)}
        for f in feats:
            real, fake = d[d.Label == "real"][f].values, d[d.Label == "fake"][f].values
            e = min(compute_eer(real, fake)[0], compute_eer(-real, -fake)[0]) * 100
            r[f] = round(e, 1)
        res.append(r)
    p(pd.DataFrame(res).to_string(index=False))
    os.makedirs(args.results, exist_ok=True)
    tab.to_csv(os.path.join(args.results, "hours_by_set.csv"), index=False)
    pd.DataFrame(res).to_csv(os.path.join(args.results, "shortcut_eer.csv"), index=False)
    open(os.path.join(args.results, "check_output.txt"), "w").write("\n".join(lines) + "\n")


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
    c = sub.add_parser("select")
    c.add_argument("--meta", default="Data/seaspoof/meta.parquet")
    c.add_argument("--ids-dir", default="protocols/ids")
    f = sub.add_parser("fetch")
    f.add_argument("--ids-dir", default="protocols/ids")
    f.add_argument("--out", default="Data/seaspoof/data", help="thư mục dữ liệu ($ROOT)")
    f.add_argument("--workers", type=int, default=8)
    k = sub.add_parser("check")
    k.add_argument("--ids-dir", default="protocols/ids")
    k.add_argument("--data", default="Data/seaspoof/data")
    k.add_argument("--results", default="results/seaspoof")
    args = ap.parse_args()
    {"meta": cmd_meta, "stats": cmd_stats, "select": cmd_select, "fetch": cmd_fetch,
     "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
