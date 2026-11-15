#!/usr/bin/env bash
# Chạy bộ thí nghiệm trực tiếp trên MỘT máy Linux có GPU NVIDIA (người chạy gõ lệnh trên chính máy đó; không cần
# vast.ai hay SSH từ máy khác). Cùng cấu trúc thư mục và cùng script với server/vast.sh. Hướng dẫn cho người chạy:
# server/HUONG-DAN-CHAY.md
#
#   bash server/local.sh check                        GPU, CPU, RAM, ổ trống
#   bash server/local.sh setup                        tạo thư mục chạy $ORPHEUS_ROOT (mặc định ~/orpheus-run), chép code
#   bash server/local.sh env                          dựng môi trường, tải dữ liệu kiểm tra (RADAR, In-the-Wild)
#   HF_TOKEN=hf_xxx bash server/local.sh fetch-data   tự tạo SEA-Spoof vi, MLAAD vi từ Hugging Face
#   bash server/local.sh copy-data <seaspoof> <mlaad> hoặc: chép dữ liệu có sẵn (thư mục Data/seaspoof/data, Data/mlaad/data)
#   bash server/local.sh bootstrap                    kiểm dữ liệu, tải trước checkpoint (chạy lại được)
#   bash server/local.sh probe                        thử đo (tầng 0, ~30–40 phút), in số worker khuyến nghị
#   bash server/local.sh start [N] [--gpus 0,1]       chạy cả bộ ở chế độ nền (N worker đọc dữ liệu; mặc định theo probe)
#   bash server/local.sh status | log <tên lần>       theo dõi
#   bash server/local.sh pack                         đóng gói kết quả (không checkpoint, không âm thanh) để gửi lại
#   bash server/local.sh drift                        đo độ trôi trọng số trên mọi checkpoint (TRƯỚC cleanup)
#   bash server/local.sh stop --yes                   dừng điều phối và mọi lần đang chạy
#   bash server/local.sh cleanup --yes                xoá SEA-Spoof, MLAAD, checkpoint, cache (sau khi đã pack và drift)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
export ORPHEUS_ROOT="${ORPHEUS_ROOT:-$HOME/orpheus-run}"
R="$ORPHEUS_ROOT"
DPY="$R/work/mamba/envs/orpheus-driver/bin/python"
CMD="${1:-help}"
need_driver() { [ -x "$DPY" ] || { echo "Chưa có môi trường điều phối: chạy 'bash server/local.sh env' trước" >&2; exit 2; }; }

case "$CMD" in
  check)
    echo "ORPHEUS_ROOT=$R"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo "CPU: $(nproc) luồng"; free -g | head -2; mkdir -p "$R"; df -h "$R"
    ;;
  setup)
    mkdir -p "$R"/{code,ids,server,results,data/seaspoof,data/mlaad}
    rm -f "$R"/code/*.py "$R"/code/*.sh "$R"/ids/*
    cp "$REPO"/src/*.py "$REPO"/src/*.sh "$R/code/"
    cp "$REPO"/protocols/ids/* "$R/ids/"
    cp "$REPO"/server/*.py "$REPO"/server/*.sh "$REPO"/server/*.md "$R/server/"
    v="git $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
    printf '%s\nstaged %s\n' "$v" "$(date '+%Y-%m-%d %H:%M:%S')" > "$R/code/VERSION"
    echo "Đã chép code ($v) vào $R. Chạy lại lệnh này sau 'git pull' để cập nhật (không đụng results/, data/)"
    ;;
  env)
    cd "$R" && SKIP_DATA=1 bash server/bootstrap.sh
    ;;
  fetch-data)
    need_driver
    [ -n "${HF_TOKEN:-}" ] || { echo "Cần HF_TOKEN (token Hugging Face của tài khoản đã được duyệt quyền SEA-Spoof)" >&2; exit 2; }
    cd "$R"
    echo "== SEA-Spoof vi (đọc đúng các câu trong ids/ từ Jack-ppkdczgx/SEA-Spoof, bản cố định)"
    "$DPY" code/prep_seaspoof.py fetch --ids-dir ids --out data/seaspoof
    echo "== MLAAD fake/vi (mueller91/MLAAD)"
    "$DPY" -c "from huggingface_hub import snapshot_download; snapshot_download('mueller91/MLAAD', repo_type='dataset', allow_patterns=['fake/vi/*'], local_dir='data/mlaad_raw')"
    mkdir -p tmp/mlaad_ids results/mlaad_build
    "$DPY" code/prep_mlaad.py build --raw data/mlaad_raw --out data/mlaad --seaspoof data/seaspoof \
      --ids-dir tmp/mlaad_ids --results results/mlaad_build
    rm -rf data/mlaad_raw
    du -sh data/seaspoof data/mlaad
    echo "Xong. Chạy 'bash server/local.sh bootstrap' để kiểm dữ liệu khớp danh sách câu đã commit"
    ;;
  copy-data)
    S="${2:?thư mục dữ liệu SEA-Spoof (chứa seaspoof_vi_*.csv và seaspoof_vi/)}"; M="${3:?thư mục dữ liệu MLAAD (chứa mlaad_vi_test.csv)}"
    mkdir -p "$R/data/seaspoof" "$R/data/mlaad"
    cp -r "$S"/. "$R/data/seaspoof/"; cp -r "$M"/. "$R/data/mlaad/"
    rm -f "$R"/data/*/dataset-metadata.json; du -sh "$R"/data/*
    ;;
  bootstrap)
    cd "$R" && bash server/bootstrap.sh
    ;;
  probe)
    need_driver; cd "$R"
    "$DPY" server/run.py run --tiers 0 2>&1 | tee -a results/driver_probe.log
    "$DPY" server/probe_report.py --train results/train --summary results/train_summary.csv | tee results/probe_report.txt
    n=$(grep -o "WORKERS_RECOMMENDED=[0-9]*" results/probe_report.txt | cut -d= -f2 || true)
    [ -n "$n" ] && echo "$n" > results/workers.txt && echo "Số worker khuyến nghị: $n (ghi results/workers.txt)"
    ;;
  start)
    need_driver; cd "$R"; shift
    N="${1:-}"; [ -n "$N" ] && [[ "$N" =~ ^[0-9]+$ ]] && shift || N="$(cat results/workers.txt 2>/dev/null || echo 2)"
    if pgrep -f 'server/run.py run' >/dev/null; then echo "server/run.py run đang chạy; dừng trước (stop --yes)"; exit 1; fi
    (setsid nohup "$DPY" server/run.py run --workers "$N" "$@" >> results/driver.log 2>&1 < /dev/null &)
    sleep 5; tail -n 5 results/driver.log; echo "Đang chạy nền với $N worker. Theo dõi: bash server/local.sh status"
    ;;
  status)
    cat "$R/results/status.txt" 2>/dev/null || echo "chưa có status.txt"; echo ---; tail -n "${2:-8}" "$R/results/driver.log" 2>/dev/null || true
    pgrep -f 'server/run.py run' >/dev/null && echo 'điều phối ĐANG CHẠY' || echo 'điều phối KHÔNG chạy'
    ;;
  log)
    N="${2:?tên lần chạy}"; tail -n "${3:-30}" "$R/results/jobs/$N.log"; echo '--- main.log'
    tail -c 3000 "$R/results/train/$N/main.log" 2>/dev/null | tr '\r' '\n' | tail -n 8 || true
    ;;
  pack)
    out="$HOME/orpheus-results-$(date +%Y%m%d-%H%M).tar.gz"
    tar -czf "$out" -C "$R" --exclude='results/ckpt' --exclude='exp' --exclude='*.ckpt' --exclude='emb.npy' \
      --exclude='*_emb.pkl' --exclude='shards' --exclude='*.safetensors' --exclude='*.lock' --exclude='mlaad_build' \
      results code/VERSION server
    ls -lh "$out"; echo "Gửi file này cho chủ đồ án (không chứa âm thanh, checkpoint)"
    ;;
  drift)
    need_driver; cd "$R"
    "$DPY" code/weight_drift.py --results results --cache cache
    ;;
  stop)
    [ "${2:-}" = "--yes" ] || { echo "Dừng mọi lần đang chạy? Thêm --yes"; exit 2; }
    pkill -f 'server/run.py run' || true; pkill -f 'code/finetune.py' || true; pkill -f 'code/score_zeroshot.py' || true
    pkill -f 'code/interp_score.py' || true; sleep 2; pkill -f 'python main.py' || true; pkill -f 'mem_probe.py' || true
    sleep 3; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    ;;
  cleanup)
    [ "${2:-}" = "--yes" ] || { echo "Xoá SEA-Spoof, MLAAD, checkpoint, cache? (đã pack và drift chưa?) Thêm --yes"; exit 2; }
    rm -rf "$R/data/seaspoof" "$R/data/mlaad" "$R/results/ckpt" "$R/cache" "$R/work/Data"; du -sh "$R" 2>/dev/null || true
    ;;
  *)
    sed -n '2,19p' "$0"
    ;;
esac
