#!/usr/bin/env python3
"""Chuẩn bị dữ liệu kiểm tra cho GĐ3 (zero-shot) và GĐ5 (đo mức quên).

Hai bước, chạy ở hai nơi:

  select  (chạy trên máy, chỉ cần mạng + numpy/pandas)
          Tải nhãn RADAR 2026 và meta.csv của In-the-Wild (đọc thẳng trong file zip
          trên Hugging Face bằng HTTP Range, không tải 8 GB), chọn danh sách câu cho
          từng tập, ghi vào protocols/ids/*.tsv. Danh sách này được commit, nên mọi
          lần chạy sau dùng đúng cùng một tập.

  fetch   (chạy trên Kaggle/Colab)
          Tải âm thanh cho các câu trong danh sách và tạo CSV theo định dạng
          protocol của AntiDeepfake (cột Path dùng tiền tố $ROOT/ như repo gốc).
          - RADAR: đọc luồng file RADAR2026-eval.tar.gz (8,75 GB) từ Zenodo và chỉ
            ghi ra các file cần, nên không cần chỗ chứa cả file nén.
          - In-the-Wild: nếu cần cả bộ (itw_full) thì tải zip 8,16 GB rồi giải nén;
            nếu chỉ cần tập con (itw_sub) thì đọc từng file trong zip qua HTTP Range.

Các tập:
  radar_vi      toàn bộ tiếng Việt trong RADAR eval: 7.000 thật, 6.438 giả
  radar_en_sub  RADAR eval tiếng Anh, chọn ngẫu nhiên 1.000 thật + 1.000 giả
  itw_full      toàn bộ In-the-Wild (31.779 câu)
  itw_sub       In-the-Wild, chọn ngẫu nhiên 1.000 thật + 1.000 giả
  ood_forget    = itw_sub + radar_en_sub (tạo khi có đủ hai tập), dùng đo mức quên

Không dùng RADAR dev: tiếng thật của dev lấy từ LibriTTS dev/test, mà LibriTTS
(và LibriTTS-R, LibriTTS-Vocoded) nằm trong dữ liệu post-training của AntiDeepfake.

Cách dùng:
  python src/prep_data.py select --ids-dir protocols/ids
  python src/prep_data.py fetch --ids-dir <ids> --out <data_root> --tmp <tmp> \
      --sets radar_vi,radar_en_sub,itw_sub,itw_full

Chạy lại được: file âm thanh đã có thì bỏ qua.
"""
import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile

SEED = 2026
N_PER_CLASS = 1000

RADAR_REC = "https://zenodo.org/api/records/22110815/files"
RADAR_TAR = f"{RADAR_REC}/RADAR2026-eval.tar.gz/content"
ITW_ZIP = ("https://huggingface.co/datasets/mueller91/In-The-Wild/"
           "resolve/main/release_in_the_wild.zip")

CSV_HEADER = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",
              "Language"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get(url, dest, retries=5):
    """Tải về dest, có tải tiếp khi đứt (HTTP Range)."""
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:      # máy chủ không hỗ trợ Range: tải lại
                    f.seek(0); f.truncate()
                shutil.copyfileobj(r, f, length=1 << 22)
            os.replace(part, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"  tải lỗi (lần {attempt}): {e}")
            time.sleep(10 * attempt)
    raise RuntimeError(f"Không tải được {url}")


class HTTPRangeFile(io.RawIOBase):
    """File chỉ đọc trên HTTP, đọc theo đoạn bằng Range. Đủ cho zipfile."""

    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url            # URL sau khi chuyển hướng (CDN)
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("HTTP Range thất bại")
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 20))


# ---------------------------------------------------------------- select

def read_radar_labels(tmp):
    import pandas as pd
    files = {}
    for name in ["label_RADAR2026-eval.txt", "lang_RADAR2026-eval.txt"]:
        dest = os.path.join(tmp, name)
        if not os.path.exists(dest):
            http_get(f"{RADAR_REC}/{name}/content", dest)
        files[name] = dest
    rows = [line.split() for line in open(files["label_RADAR2026-eval.txt"])]
    lab = pd.DataFrame({"ID": [r[0] for r in rows],
                        "Duration": [float(r[1]) for r in rows],
                        "Label": ["real" if r[2] == "bonafide" else "fake" for r in rows]})
    lang = pd.read_csv(files["lang_RADAR2026-eval.txt"], sep=r"\s+", header=None,
                       names=["ID", "Language"])
    assert (lab.ID.values == lang.ID.values).all(), "label và lang lệch dòng"
    lab["Language"] = lang.Language.values
    lab["Speaker"] = "-"
    return lab


def sample_balanced(df, n, seed):
    import pandas as pd
    parts = []
    for i, label in enumerate(["real", "fake"]):
        sub = df[df.Label == label]
        assert len(sub) >= n, f"không đủ {n} câu {label}"
        parts.append(sub.sample(n=n, random_state=seed + i))
    return pd.concat(parts).sort_values("ID").reset_index(drop=True)


def cmd_select(args):
    import pandas as pd
    os.makedirs(args.ids_dir, exist_ok=True)
    tmp = args.tmp or args.ids_dir
    os.makedirs(tmp, exist_ok=True)

    log("RADAR: tải nhãn eval")
    radar = read_radar_labels(tmp)
    vi = radar[radar.Language == "VI"].reset_index(drop=True)
    en = radar[radar.Language == "EN"].reset_index(drop=True)
    en_sub = sample_balanced(en, N_PER_CLASS, SEED)

    log("In-the-Wild: đọc meta.csv trong zip qua HTTP Range")
    z = open_remote_zip(ITW_ZIP)
    meta_name = next(n for n in z.namelist() if n.endswith("meta.csv"))
    meta = pd.read_csv(io.BytesIO(z.read(meta_name)))
    sizes = {os.path.basename(i.filename): i.file_size for i in z.infolist()}
    itw = pd.DataFrame({
        "ID": "ItW-" + meta.file.str.replace(".wav", "", regex=False),
        "File": meta.file,
        "Label": meta.label.map({"bona-fide": "real", "spoof": "fake"}),
        "Speaker": meta.speaker,
        "Language": "EN",
    })
    assert itw.Label.notna().all(), "nhãn In-the-Wild lạ"
    itw["Bytes"] = itw.File.map(sizes)
    itw_sub = sample_balanced(itw, N_PER_CLASS, SEED)

    out = {"radar_vi": vi, "radar_en_sub": en_sub, "itw_full": itw, "itw_sub": itw_sub}
    stats = {}
    for name, df in out.items():
        path = os.path.join(args.ids_dir, f"{name}.tsv")
        df.to_csv(path, sep="\t", index=False)
        s = {"n": int(len(df)), "real": int((df.Label == "real").sum()),
             "fake": int((df.Label == "fake").sum())}
        if "Duration" in df:
            s["hours"] = round(float(df.Duration.sum()) / 3600, 2)
            s["max_sec"] = round(float(df.Duration.max()), 2)
        stats[name] = s
        log(f"{name}: {s}")
    stats["seed"] = SEED
    stats["note"] = ("RADAR: Zenodo 22110815, eval set. In-the-Wild: HF mueller91/In-The-Wild. "
                     "Chọn ngẫu nhiên theo lớp, pandas.sample(random_state=seed+i), i=0 thật, 1 giả.")
    json.dump(stats, open(os.path.join(args.ids_dir, "stats.json"), "w"), indent=1,
              ensure_ascii=False)


# ---------------------------------------------------------------- fetch

def read_ids(ids_dir, name):
    import pandas as pd
    return pd.read_csv(os.path.join(ids_dir, f"{name}.tsv"), sep="\t")


def audio_row(path_on_disk, rel_path, row, language):
    import soundfile as sf
    info = sf.info(path_on_disk)
    return {
        "ID": row.ID, "Label": row.Label,
        "Duration": round(info.frames / info.samplerate, 2),
        "SampleRate": info.samplerate,
        "Path": "$ROOT/" + rel_path,
        "Attack": "-", "Speaker": row.Speaker if isinstance(row.Speaker, str) else "-",
        "Proportion": "-", "AudioChannel": info.channels,
        "AudioEncoding": info.format, "AudioBitSample": info.subtype,
        "Language": language,
    }


def fetch_radar(args, names):
    import pandas as pd
    want = {}
    for name in names:
        df = read_ids(args.ids_dir, name)
        for r in df.itertuples():
            want[r.ID] = r
    flac_dir = os.path.join(args.out, "radar", "flac")
    os.makedirs(flac_dir, exist_ok=True)
    missing = {i for i in want if not os.path.exists(os.path.join(flac_dir, f"{i}.flac"))}
    log(f"RADAR: cần {len(want)} file, còn thiếu {len(missing)}")

    for attempt in range(1, 4):
        if not missing:
            break
        log(f"RADAR: đọc luồng RADAR2026-eval.tar.gz (8,75 GB), lần {attempt}")
        t0 = time.time()
        try:
            with urllib.request.urlopen(RADAR_TAR, timeout=120) as r, \
                    tarfile.open(fileobj=r, mode="r|gz") as tar:
                seen = 0
                for m in tar:
                    seen += 1
                    if seen % 10000 == 0:
                        log(f"  đã qua {seen} mục, còn thiếu {len(missing)}, "
                            f"{(time.time() - t0) / 60:.1f} phút")
                    if not m.isfile():
                        continue
                    uid = os.path.splitext(os.path.basename(m.name))[0]
                    if uid not in missing:
                        continue
                    src = tar.extractfile(m)
                    dest = os.path.join(flac_dir, f"{uid}.flac")
                    with open(dest + ".part", "wb") as f:
                        shutil.copyfileobj(src, f)
                    os.replace(dest + ".part", dest)
                    missing.discard(uid)
                    if not missing:
                        log("  đủ file, dừng đọc luồng")
                        break
        except Exception as e:  # noqa: BLE001
            log(f"  luồng bị lỗi: {e}")
    if missing:
        raise RuntimeError(f"RADAR còn thiếu {len(missing)} file, ví dụ {sorted(missing)[:3]}")

    for name in names:
        df = read_ids(args.ids_dir, name)
        rows = [audio_row(os.path.join(flac_dir, f"{r.ID}.flac"), f"radar/flac/{r.ID}.flac",
                          r, r.Language) for r in df.itertuples()]
        write_csv(args.out, name, rows)


def fetch_itw(args, names):
    wav_dir = os.path.join(args.out, "itw", "wav")
    os.makedirs(wav_dir, exist_ok=True)
    frames = {n: read_ids(args.ids_dir, n) for n in names}
    want = set()
    for df in frames.values():
        want.update(df.File)
    missing = {f for f in want if not os.path.exists(os.path.join(wav_dir, f))}
    log(f"In-the-Wild: cần {len(want)} file, còn thiếu {len(missing)}")

    mode = args.itw_mode
    if mode == "auto":
        # zip (8,16 GB) + file giải nén (8,7 GB) nếu tmp và out cùng ổ
        free = shutil.disk_usage(args.tmp).free
        mode = "zip" if free > 18e9 else "range"
        log(f"  chỗ trống ở {args.tmp}: {free / 1e9:.1f} GB -> chế độ {mode}")
    if missing and "itw_full" in names and mode == "zip":
        zpath = os.path.join(args.tmp, "release_in_the_wild.zip")
        if not os.path.exists(zpath):
            log("  tải zip 8,16 GB")
            t0 = time.time()
            http_get(ITW_ZIP, zpath)
            log(f"  tải xong sau {(time.time() - t0) / 60:.1f} phút")
        z = zipfile.ZipFile(zpath)
    elif missing:
        log("  đọc từng file trong zip qua HTTP Range (không tải cả zip)")
        z = open_remote_zip(ITW_ZIP)
    else:
        z = None

    if z is not None:
        members = {os.path.basename(i.filename): i for i in z.infolist()}
        t0 = time.time()
        for k, f in enumerate(sorted(missing), 1):
            dest = os.path.join(wav_dir, f)
            with z.open(members[f]) as src, open(dest + ".part", "wb") as out:
                shutil.copyfileobj(src, out)
            os.replace(dest + ".part", dest)
            if k % 2000 == 0:
                log(f"  {k}/{len(missing)} file, {(time.time() - t0) / 60:.1f} phút")
        z.close()
        zpath = os.path.join(args.tmp, "release_in_the_wild.zip")
        if os.path.exists(zpath) and not args.keep_zip:
            os.remove(zpath)

    for name, df in frames.items():
        rows = [audio_row(os.path.join(wav_dir, r.File), f"itw/wav/{r.File}", r, "EN")
                for r in df.itertuples()]
        write_csv(args.out, name, rows)


def write_csv(out, name, rows):
    import pandas as pd
    df = pd.DataFrame(rows)[CSV_HEADER]
    path = os.path.join(out, f"{name}.csv")
    df.to_csv(path, index=False)
    log(f"Ghi {path}: {len(df)} câu ({(df.Label == 'real').sum()} thật, "
        f"{(df.Label == 'fake').sum()} giả), {df.Duration.sum() / 3600:.2f} giờ, "
        f"dài nhất {df.Duration.max():.1f} s, tần số lấy mẫu {sorted(df.SampleRate.unique())}, "
        f"số kênh {sorted(df.AudioChannel.unique())}")


def cmd_fetch(args):
    import pandas as pd
    sets = [s for s in args.sets.split(",") if s]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    for p in sorted({args.out, args.tmp}):
        u = shutil.disk_usage(p)
        log(f"Ổ chứa {p}: trống {u.free / 1e9:.1f} / {u.total / 1e9:.1f} GB")

    radar = [s for s in sets if s.startswith("radar")]
    itw = [s for s in sets if s.startswith("itw")]
    if radar:
        fetch_radar(args, radar)
    if itw:
        fetch_itw(args, itw)

    a, b = (os.path.join(args.out, f"{n}.csv") for n in ["itw_sub", "radar_en_sub"])
    if os.path.exists(a) and os.path.exists(b):
        df = pd.concat([pd.read_csv(a), pd.read_csv(b)], ignore_index=True)
        df.to_csv(os.path.join(args.out, "ood_forget.csv"), index=False)
        log(f"Ghi ood_forget.csv: {len(df)} câu")

    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": {}}
    for f in sorted(os.listdir(args.out)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(args.out, f))
            manifest["sets"][f[:-4]] = {
                "n": int(len(df)), "real": int((df.Label == "real").sum()),
                "fake": int((df.Label == "fake").sum()),
                "hours": round(float(df.Duration.sum()) / 3600, 3),
                "max_sec": float(df.Duration.max())}
    total = 0
    for root, _, files in os.walk(args.out):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    manifest["bytes"] = total
    json.dump(manifest, open(os.path.join(args.out, "orpheus_data_manifest.json"), "w"),
              indent=1, ensure_ascii=False)
    log(f"Xong. Tổng dung lượng {args.out}: {total / 1e9:.2f} GB")
    log(json.dumps(manifest["sets"], ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--ids-dir", default="protocols/ids")
    s.add_argument("--tmp", default=None, help="nơi để file nhãn tải về (mặc định: ids-dir)")
    f = sub.add_parser("fetch")
    f.add_argument("--ids-dir", required=True)
    f.add_argument("--out", required=True, help="thư mục dữ liệu (sẽ là $ROOT)")
    f.add_argument("--tmp", required=True, help="nơi để file tạm (zip In-the-Wild)")
    f.add_argument("--sets", default="radar_vi,radar_en_sub,itw_sub,itw_full")
    f.add_argument("--keep-zip", action="store_true")
    f.add_argument("--itw-mode", choices=["auto", "zip", "range"], default="auto",
                   help="zip: tải cả file 8,16 GB; range: đọc từng file qua HTTP (chậm, không tốn ổ)")
    args = ap.parse_args()
    {"select": cmd_select, "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
