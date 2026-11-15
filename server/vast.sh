#!/usr/bin/env bash
# Điều khiển server thuê (vast.ai) từ máy (Claude chạy). Cần SSH vào được server bằng khoá SSH sẵn có
# trên máy (thêm khoá công khai vào tài khoản vast.ai trước khi thuê). Claude không đọc khoá.
#
#   bash server/vast.sh init <host> <port> [user]   ghi server/host.env (không commit)
#   bash server/vast.sh check                       SSH được không, GPU, CPU, RAM, ổ đĩa
#   bash server/vast.sh code                        chép src/, protocols/ids/, server/ lên (kèm VERSION)
#   bash server/vast.sh data                        chép SEA-Spoof vi và MLAAD vi (Data/*/data) lên
#   bash server/vast.sh bootstrap                   chạy server/bootstrap.sh trên server (chạy nền)
#   bash server/vast.sh bootlog                     xem tiến độ dựng server
#   bash server/vast.sh start [tham số run.py]      chạy nền server/run.py run ... (ví dụ --tiers 0)
#   bash server/vast.sh status                      results/status.txt + cuối driver.log
#   bash server/vast.sh log <tên lần> [số dòng]     log của một lần chạy
#   bash server/vast.sh fetch                       tải kết quả về results/server/raw/ (không checkpoint)
#   bash server/vast.sh collect <nhãn>              gộp vào results/server/ và lưu bằng chứng results/runs/server-<nhãn>/
#   bash server/vast.sh stop --yes                  dừng điều phối và mọi lần đang chạy
#   bash server/vast.sh cleanup --yes               xoá SEA-Spoof, MLAAD, checkpoint trên server (trước khi trả máy)
#   bash server/vast.sh ssh                         mở shell trên server
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
HOSTENV="$HERE/host.env"
CMD="${1:-help}"

load() {
  [ -f "$HOSTENV" ] || { echo "Chưa có $HOSTENV: chạy 'bash server/vast.sh init <host> <port>'" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$HOSTENV"
  R="${VAST_ROOT:-/workspace/orpheus}"
  SSHO=(-p "$VAST_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
  DPY="$R/work/mamba/envs/orpheus-driver/bin/python"
}
remote() { ssh "${SSHO[@]}" "$VAST_USER@$VAST_HOST" "$@"; }
# rsync phải có ở cả hai đầu; ảnh máy của vast.ai có thể chưa có (bootstrap.sh cài sau bước chép)
need_rsync() { remote "command -v rsync >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync)"; }
rs() { rsync -az --partial -e "ssh ${SSHO[*]}" "$@"; }

case "$CMD" in
  init)
    H="${2:?host}"; P="${3:?port}"; U="${4:-root}"
    printf 'VAST_HOST=%s\nVAST_PORT=%s\nVAST_USER=%s\nVAST_ROOT=/workspace/orpheus\n' "$H" "$P" "$U" > "$HOSTENV"
    echo "Đã ghi $HOSTENV"
    ;;
  check)
    load
    remote "echo SSH OK; hostname; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; \
            nproc; free -g | head -2; df -h /workspace 2>/dev/null || df -h ~"
    ;;
  code)
    load
    v="git $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet -- src server protocols || echo ' (có thay đổi chưa commit)')"
    need_rsync
    remote "mkdir -p $R/code $R/ids $R/server $R/results"
    rs --delete --exclude __pycache__ "$REPO"/src/*.py "$REPO"/src/*.sh "$VAST_USER@$VAST_HOST:$R/code/"
    rs --delete "$REPO"/protocols/ids/ "$VAST_USER@$VAST_HOST:$R/ids/"
    rs --exclude host.env --exclude __pycache__ "$REPO"/server/*.py "$REPO"/server/*.sh "$REPO"/server/*.md \
       "$VAST_USER@$VAST_HOST:$R/server/"
    remote "printf '%s\nstaged %s\n' '$v' '$(date '+%Y-%m-%d %H:%M:%S')' > $R/code/VERSION; cat $R/code/VERSION"
    ;;
  data)
    # SEA-Spoof không được phân phối lại: chỉ chép lên server riêng đã thuê, xoá khi trả máy (cleanup)
    load
    need_rsync
    remote "mkdir -p $R/data/seaspoof $R/data/mlaad"
    rs --stats --exclude dataset-metadata.json "$REPO/Data/seaspoof/data/" "$VAST_USER@$VAST_HOST:$R/data/seaspoof/"
    rs --stats --exclude dataset-metadata.json "$REPO/Data/mlaad/data/" "$VAST_USER@$VAST_HOST:$R/data/mlaad/"
    remote "du -sh $R/data/*"
    ;;
  bootstrap)
    load
    remote "cd $R && mkdir -p results/env && (ORPHEUS_ROOT=$R setsid nohup bash server/bootstrap.sh \
            > results/env/bootstrap.out 2>&1 < /dev/null &) ; echo 'đã chạy nền bootstrap.sh'"
    ;;
  bootlog)
    load
    remote "tail -n ${2:-25} $R/results/env/bootstrap.log 2>/dev/null; echo ---; tail -n 5 $R/results/env/bootstrap.out 2>/dev/null; \
            pgrep -f 'server/bootstrap.sh' >/dev/null && echo 'bootstrap ĐANG CHẠY' || echo 'bootstrap không chạy'"
    ;;
  start)
    load; shift
    if remote "pgrep -f 'server/run.py run' >/dev/null"; then echo "server/run.py run đang chạy; dừng trước (stop --yes)"; exit 1; fi
    remote "cd $R && (ORPHEUS_ROOT=$R setsid nohup $DPY server/run.py run $* >> results/driver.log 2>&1 < /dev/null &); \
            sleep 5; tail -n 5 results/driver.log"
    ;;
  status)
    load
    remote "cat $R/results/status.txt 2>/dev/null; echo ---; tail -n ${2:-8} $R/results/driver.log 2>/dev/null; \
            pgrep -f 'server/run.py run' >/dev/null && echo 'điều phối ĐANG CHẠY' || echo 'điều phối KHÔNG chạy'"
    ;;
  log)
    load; N="${2:?tên lần chạy}"
    remote "tail -n ${3:-30} $R/results/jobs/$N.log; echo '--- main.log'; tail -c 3000 $R/results/train/$N/main.log 2>/dev/null | tr '\r' '\n' | tail -n 8"
    ;;
  fetch)
    load
    dst="$REPO/results/server/raw"; mkdir -p "$dst/_code"
    rs --stats --exclude 'ckpt/' --exclude 'exp/' --exclude '*.ckpt' --exclude 'emb.npy' \
       --exclude '*_emb.pkl' --exclude 'shards/' --exclude '*.safetensors' --exclude '*.lock' \
       "$VAST_USER@$VAST_HOST:$R/results/" "$dst/"
    rs "$VAST_USER@$VAST_HOST:$R/code/VERSION" "$dst/_code/"
    rs --exclude __pycache__ "$VAST_USER@$VAST_HOST:$R/server/" "$dst/_code/server/"
    du -sh "$dst"
    ;;
  collect)
    TAG="${2:?nhãn, ví dụ gd6-20261116}"
    python3 -I "$HERE/collect.py" --raw "$REPO/results/server/raw" --dst "$REPO/results/server" \
      --runs "$REPO/results/runs/server-$TAG" --tag "$TAG"
    ;;
  stop)
    load; [ "${2:-}" = "--yes" ] || { echo "Dừng mọi lần đang chạy? Thêm --yes"; exit 2; }
    remote "pkill -f 'server/run.py run' || true; pkill -f 'code/finetune.py' || true; pkill -f 'code/score_zeroshot.py' || true; \
            sleep 2; pkill -f 'python main.py' || true; sleep 3; nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    ;;
  cleanup)
    load; [ "${2:-}" = "--yes" ] || { echo "Xoá dữ liệu SEA-Spoof, MLAAD và checkpoint trên server? Thêm --yes"; exit 2; }
    remote "rm -rf $R/data/seaspoof $R/data/mlaad $R/results/ckpt $R/cache $R/work/Data; du -sh $R 2>/dev/null"
    ;;
  ssh)
    load; ssh "${SSHO[@]}" "$VAST_USER@$VAST_HOST"
    ;;
  *)
    sed -n '2,20p' "$0"
    ;;
esac
