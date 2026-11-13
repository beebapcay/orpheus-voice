#!/usr/bin/env bash
# Điều khiển Kaggle từ máy (Claude chạy). Cần Kaggle CLI (>= 2.x) đã đăng nhập bằng `kaggle auth login`.
#
#   bash kaggle/push.sh src                 tạo/cập nhật dataset riêng tư <user>/orpheus-src
#                                           (src/*.py, src/*.sh, protocols/ids/*, VERSION)
#   bash kaggle/push.sh push prep|zeroshot  đẩy kernel và bắt đầu chạy
#   bash kaggle/push.sh status prep|zeroshot
#   bash kaggle/push.sh logs prep|zeroshot [-f]   xem log (không tải output)
#   bash kaggle/push.sh quota               quota GPU còn lại
#   bash kaggle/push.sh output zeroshot     tải kết quả về results/kaggle-raw/, chép phần
#                                           cần lưu vào results/zeroshot/
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
  push)
    K="${2:?prep hoặc zeroshot}"; U=$(user)
    d="$BUILD/orpheus-$K"; rm -rf "$d" && mkdir -p "$d"
    cp "$ROOT/kaggle/$K/run.py" "$d/"
    sed "s/KAGGLE_USER/$U/g" "$ROOT/kaggle/$K/kernel-metadata.json" > "$d/kernel-metadata.json"
    kg kernels push -p "$d"
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
  output)
    K="${2:?zeroshot}"; U=$(user)
    [ "$K" = "zeroshot" ] || { echo "Chỉ tải output của zeroshot (prep có ~10 GB âm thanh)"; exit 2; }
    raw="$ROOT/results/kaggle-raw/orpheus-$K"
    rm -rf "$raw" && mkdir -p "$raw"
    kg kernels output "$U/orpheus-$K" -p "$raw"
    # Lưu vào git: tóm tắt, độ đo, file điểm, log cài đặt. Không lưu embedding và shard.
    mkdir -p "$ROOT/results/zeroshot"
    rsync -a --exclude 'emb.npy' --exclude 'shards/' "$raw/results/" "$ROOT/results/zeroshot/"
    echo "Đã chép vào results/zeroshot/. Embedding ở $raw/results/*/*/emb.npy"
    ls "$raw"
    ;;
  *)
    sed -n '2,12p' "$0"
    ;;
esac
