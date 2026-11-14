#!/usr/bin/env bash
# Điều khiển Kaggle từ máy (Claude chạy). Cần Kaggle CLI (>= 2.x) đã đăng nhập bằng `kaggle auth login`.
#
#   bash kaggle/push.sh src                 tạo/cập nhật dataset riêng tư <user>/orpheus-src
#                                           (src/*.py, src/*.sh, protocols/ids/*, VERSION)
#   bash kaggle/push.sh data                tạo/cập nhật dataset riêng tư <user>/orpheus-seaspoof-vi
#                                           từ Data/seaspoof/data (FLAC + CSV, prep_seaspoof.py fetch).
#                                           SEA-Spoof không được phân phối lại: dataset PHẢI riêng tư
#   bash kaggle/push.sh push prep|zeroshot|finetune|finetune-b  đẩy kernel và bắt đầu chạy
#                                           (finetune-b: kernel thứ hai, cùng run.py, để chạy song song)
#   bash kaggle/push.sh status <kernel>
#   bash kaggle/push.sh logs <kernel> [-f]  xem log (không tải output)
#   bash kaggle/push.sh quota               quota GPU còn lại
#   bash kaggle/push.sh output zeroshot N   tải output version N về results/kaggle-raw/,
#                                           lưu bằng chứng vào results/runs/, kết quả vào results/zeroshot/
#   bash kaggle/push.sh output finetune N   như trên, trừ checkpoint và embedding (ở lại Kaggle);
#                                           kết quả vào results/finetune/
#   bash kaggle/push.sh archive-log prep N  chỉ lưu log version N vào results/runs/ (không tải output)
#
# Mỗi lần push lưu run.py + metadata vào results/runs/orpheus-<kernel>-v<N>/; output và
# archive-log bổ sung log, môi trường, log từng shard và RUN.md. Thư mục này được commit.
#
# Dataset và kernel đều riêng tư (is_private mặc định của Kaggle).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build/kaggle"
CMD="${1:-help}"
# Gọi CLI qua kaggle_ipv4.py (chỉ dùng IPv4): mạng của máy không đi được IPv6 tới
# Google, làm lệnh tải file lên bị treo. Dùng đúng Python mà Kaggle CLI được cài vào.
# Hàm thay vì biến, vì đường dẫn repo có dấu cách.
KPY=$(head -1 "$(command -v kaggle)" | sed 's/^#!//')
kg() { "$KPY" "$ROOT/kaggle/kaggle_ipv4.py" "$@"; }

user() {
  # Tên người dùng: biến KAGGLE_USERNAME nếu có, không thì lấy từ `kaggle config view`
  # (không đọc token).
  local u="${KAGGLE_USERNAME:-}"
  [ -n "$u" ] || u=$(kg config view 2>/dev/null | sed -n 's/^- username: //p' | head -1)
  [ -n "$u" ] || { echo "Không lấy được username: chạy 'kaggle auth login' hoặc đặt KAGGLE_USERNAME" >&2; exit 1; }
  echo "$u"
}

stage_src() {
  local d="$BUILD/orpheus-src" u="$1"
  rm -rf "$d" && mkdir -p "$d"
  cp "$ROOT"/src/*.py "$ROOT"/src/*.sh "$d"/
  cp "$ROOT"/protocols/ids/* "$d"/
  {
    echo "git $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet -- src protocols || echo ' (có thay đổi chưa commit)')"
    echo "staged $(date '+%Y-%m-%d %H:%M:%S')"
  } > "$d/VERSION"
  cat > "$d/dataset-metadata.json" <<EOF
{"title": "orpheus-src", "id": "$u/orpheus-src", "licenses": [{"name": "other"}]}
EOF
  echo "$d"
}

case "$CMD" in
  src)
    U=$(user); D=$(stage_src "$U")
    if kg datasets status "$U/orpheus-src" >/dev/null 2>&1; then
      kg datasets version -p "$D" -m "$(git -C "$ROOT" log -1 --format=%s)"
    else
      kg datasets create -p "$D"
    fi
    for _ in $(seq 1 30); do
      s=$(kg datasets status "$U/orpheus-src" 2>/dev/null || true)
      echo "trạng thái dataset: $s"
      [ "$s" = "ready" ] && break
      sleep 10
    done
    ;;
  data)
    # Dataset riêng tư (mặc định của CLI; KHÔNG dùng --public). Thư mục seaspoof_vi/ đưa lên
    # dạng tar không nén (FLAC đã nén sẵn).
    U=$(user); D="$ROOT/Data/seaspoof/data"
    [ -f "$D/seaspoof_vi_train.csv" ] || { echo "Chưa có $D: chạy prep_seaspoof.py fetch trước"; exit 2; }
    cat > "$D/dataset-metadata.json" <<EOF
{"title": "orpheus-seaspoof-vi", "id": "$U/orpheus-seaspoof-vi", "licenses": [{"name": "other"}]}
EOF
    git -C "$ROOT" rev-parse --short HEAD > "$D/VERSION"
    if kg datasets status "$U/orpheus-seaspoof-vi" >/dev/null 2>&1; then
      kg datasets version -p "$D" -r tar -m "$(git -C "$ROOT" log -1 --format=%s)"
    else
      kg datasets create -p "$D" -r tar
    fi
    for _ in $(seq 1 60); do
      s=$(kg datasets status "$U/orpheus-seaspoof-vi" 2>/dev/null || true)
      echo "trạng thái dataset: $s"
      [ "$s" = "ready" ] && break
      sleep 20
    done
    ;;
  data-mlaad)
    # Tập kiểm tra MLAAD fake/vi (src/prep_mlaad.py build): dataset RIÊNG TƯ <user>/orpheus-mlaad-vi.
    # MLAAD là CC BY-NC 4.0; giữ riêng tư như SEA-Spoof.
    U=$(user); D="$ROOT/Data/mlaad/data"
    [ -f "$D/mlaad_vi_test.csv" ] || { echo "Chưa có $D: chạy prep_mlaad.py build trước"; exit 2; }
    cat > "$D/dataset-metadata.json" <<EOF
{"title": "orpheus-mlaad-vi", "id": "$U/orpheus-mlaad-vi", "licenses": [{"name": "other"}]}
EOF
    git -C "$ROOT" rev-parse --short HEAD > "$D/VERSION"
    if kg datasets status "$U/orpheus-mlaad-vi" >/dev/null 2>&1; then
      kg datasets version -p "$D" -r tar -m "$(git -C "$ROOT" log -1 --format=%s)"
    else
      kg datasets create -p "$D" -r tar
    fi
    for _ in $(seq 1 60); do
      s=$(kg datasets status "$U/orpheus-mlaad-vi" 2>/dev/null || true)
      echo "trạng thái dataset: $s"
      [ "$s" = "ready" ] && break
      sleep 20
    done
    ;;
  push)
    K="${2:?prep, zeroshot hoặc finetune}"; U=$(user)
    d="$BUILD/orpheus-$K"; rm -rf "$d" && mkdir -p "$d"
    cp "$ROOT/kaggle/$K/run.py" "$d/"
    sed "s/KAGGLE_USER/$U/g" "$ROOT/kaggle/$K/kernel-metadata.json" > "$d/kernel-metadata.json"
    out=$(kg kernels push -p "$d" 2>&1); echo "$out"
    # Lưu bản đã đẩy vào results/runs/orpheus-<K>-v<N>/ (bằng chứng, để tái hiện)
    V=$(echo "$out" | sed -n 's/.*Kernel version \([0-9][0-9]*\) successfully pushed.*/\1/p')
    if [ -n "$V" ]; then
      python3 -I "$ROOT/kaggle/archive_run.py" pushed --run-dir "$ROOT/results/runs/orpheus-$K-v$V" \
        --build-dir "$d" --kernel "orpheus-$K" --version "$V"
    else
      echo "CẢNH BÁO: không đọc được số version, chưa lưu bản đẩy" >&2
    fi
    ;;
  status)
    K="${2:?prep hoặc zeroshot}"; U=$(user)
    kg kernels status "$U/orpheus-$K"
    ;;
  logs)
    # Xem log mà không tải output (output của prep ~10 GB)
    K="${2:?prep hoặc zeroshot}"; U=$(user)
    kg kernels logs "$U/orpheus-$K" "${@:3}"
    ;;
  quota)
    kg quota
    ;;
  archive-log)
    # Chỉ lưu log và bằng chứng (không tải output), dùng cho prep (~10 GB, >45.000 file)
    K="${2:?prep hoặc zeroshot}"; V="${3:?số version}"; U=$(user)
    run="$ROOT/results/runs/orpheus-$K-v$V"; mkdir -p "$run"
    kg kernels logs "$U/orpheus-$K/$V" > "$run/.kernel.log.raw"
    q=$(mktemp); kg quota > "$q" 2>&1 || true
    python3 -I "$ROOT/kaggle/archive_run.py" collect --run-dir "$run" --raw "$run" \
      --log "$run/.kernel.log.raw" --kernel "orpheus-$K" --version "$V" --quota "$q"
    rm -f "$run/.kernel.log.raw" "$q"
    ;;
  output)
    # bash kaggle/push.sh output zeroshot|finetune <version>
    K="${2:?zeroshot hoặc finetune}"; V="${3:?số version (xem results/runs/ hoặc dòng 'Kernel version N' lúc đẩy)}"; U=$(user)
    case "$K" in zeroshot|finetune|finetune-b) ;; *) echo "Chỉ tải output của zeroshot, finetune, finetune-b (prep: dùng archive-log)"; exit 2;; esac
    # Nhãn version trong results/finetune/*.csv: "3" cho orpheus-finetune v3, "b1" cho orpheus-finetune-b v1
    VTAG="$V"; [ "$K" = "finetune-b" ] && VTAG="b$V"
    # Mỗi version tải vào thư mục riêng, để không mất embedding của lần chạy trước
    raw="$ROOT/results/kaggle-raw/orpheus-$K-v$V"
    mkdir -p "$raw"
    if [ "${K%-b}" = "finetune" ]; then
      # Không tải checkpoint (~1,3 GB mỗi lần huấn luyện) và embedding (emb.npy, *_emb.pkl: ~98%
      # dung lượng còn lại, tải ~0,5 MB/s ở GĐ5 lần 1); chúng ở lại output trên Kaggle.
      # Cần embedding thì: kg kernels output <user>/orpheus-finetune/<N> -p <thư mục> --file-pattern 'emb'
      kg kernels output "$U/orpheus-$K/$V" -p "$raw" --page-size 200 \
        --file-pattern '^(?!.*(\.ckpt|emb\.npy|_emb\.pkl)$)'
    else
      kg kernels output "$U/orpheus-$K/$V" -p "$raw" --page-size 200
    fi
    q=$(mktemp); kg quota > "$q" 2>&1 || true
    python3 -I "$ROOT/kaggle/archive_run.py" collect --run-dir "$ROOT/results/runs/orpheus-$K-v$V" \
      --raw "$raw" --kernel "orpheus-$K" --version "$V" --quota "$q"
    rm -f "$q"
    if [ "${K%-b}" = "finetune" ]; then
      # Lưu vào git: tóm tắt huấn luyện, log, điểm và độ đo. Không lưu embedding, shard, checkpoint.
      dst="$ROOT/results/finetune"; mkdir -p "$dst"
      python3 -I "$ROOT/kaggle/archive_run.py" merge-finetune --raw "$raw" --dst "$dst" --version "$VTAG"
      echo "Đã chép vào results/finetune/. Checkpoint ở output Kaggle orpheus-$K/$V (ckpt/)"
      ls "$raw"
      exit 0
    fi
    # Lưu vào git: tóm tắt, độ đo, file điểm, log cài đặt. Không lưu embedding và shard.
    dst="$ROOT/results/zeroshot"; mkdir -p "$dst"
    # Gộp summary.csv: dòng mới thay dòng cũ cùng (dataset, model), giữ các dòng cũ còn lại;
    # gộp theo tên cột (từ GĐ5 bảng có thêm cột source), không theo vị trí
    merged=$(mktemp)
    [ -f "$dst/summary.csv" ] && cp "$dst/summary.csv" "$merged"
    python3 -I "$ROOT/kaggle/archive_run.py" merge-csv --src "$raw/results/summary.csv" --dst "$merged" \
      --key dataset,model --tag "source=orpheus-zeroshot-v$V"
    rsync -a --exclude 'emb.npy' --exclude 'shards/' "$raw/results/" "$dst/"
    mv "$merged" "$dst/summary.csv"
    echo "Đã chép vào results/zeroshot/ (summary.csv đã gộp). Embedding ở $raw/results/*/*/emb.npy"
    ls "$raw"
    ;;
  *)
    sed -n '2,24p' "$0"
    ;;
esac
