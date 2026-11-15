#!/usr/bin/env bash
# Dựng server một lần (chạy TRÊN server, sau khi `server/vast.sh code` và `server/vast.sh data` đã
# chép code và SEA-Spoof, MLAAD lên). Chạy lại được: bước nào xong thì bỏ qua.
#
#   ORPHEUS_ROOT=/workspace/orpheus bash server/bootstrap.sh
#
# Các bước:
#   1. gói hệ thống (git, curl, wget, bzip2, trình biên dịch cho fairseq, rsync, tmux)
#   2. ảnh chụp máy -> results/env/machine.txt
#   3. môi trường AntiDeepfake như Kaggle (src/setup_antideepfake.sh: Python 3.9, torch 2.6.0 cu118,
#      fairseq 862efab, SpeechBrain 1.0.2, AntiDeepfake 86098091) + toy example (EER phải 6,11%)
#   4. môi trường điều phối orpheus-driver (Python 3.11, pandas, numpy, scipy, soundfile,
#      safetensors, torch CPU) cho server/run.py, src/finetune.py, src/score_zeroshot.py
#   5. dữ liệu kiểm tra RADAR VI, RADAR EN (tập con), In-the-Wild (tập con), tải thẳng từ Zenodo và
#      Hugging Face bằng src/prep_data.py, đúng danh sách câu đã commit (protocols/ids/)
#   3b. kiểm TF32 bằng số (server/tf32_check.py): NVIDIA_TF32_OVERRIDE=0 phải cho kết quả fp32
#   5b. tạo radar_vi_rv_{train,dev,test}.csv cho kiểm chế độ D1 (src/split_radar_regime.py)
#   6. kiểm dữ liệu (server/check_data.py) và tải trước checkpoint (server/run.py prefetch)
set -euo pipefail
ROOT="${ORPHEUS_ROOT:-/workspace/orpheus}"
cd "$ROOT"
mkdir -p results/env cache work data tmp
LOG="$ROOT/results/env/bootstrap.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
trap 'log "LỖI ở dòng $LINENO: $BASH_COMMAND"' ERR
log "Bắt đầu dựng server, ROOT=$ROOT, code $(head -1 code/VERSION 2>/dev/null || echo '?')"

# 1. Gói hệ thống
need=0
for c in git curl wget bzip2 gcc g++ make rsync tmux; do command -v "$c" >/dev/null || need=1; done
if [ "$need" = 1 ]; then
  log "Cài gói hệ thống"
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
  $SUDO apt-get update -qq >>"$LOG" 2>&1
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq git curl wget bzip2 build-essential rsync tmux >>"$LOG" 2>&1
fi

# 2. Ảnh chụp máy
{
  date; echo; nvidia-smi; echo
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv; echo
  lscpu; echo; free -g; echo; df -h "$ROOT"; echo; cat /etc/os-release
} > results/env/machine.txt 2>&1 || true
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"

# 3. Môi trường AntiDeepfake + toy example
export CACHE_DIR="$ROOT/cache"
bash code/setup_antideepfake.sh "$ROOT/work" env
if ! grep -q "eer" results/env/toy_check.txt 2>/dev/null; then
  bash code/setup_antideepfake.sh "$ROOT/work" toy > results/env/toy_check.txt 2>&1 || true
fi
cp work/setup.log work/versions.txt work/pip-freeze.txt results/env/ 2>/dev/null || true
cp work/toy_inference.log results/env/ 2>/dev/null || true
if grep -qE "0\.0611|6\.11" results/env/toy_check.txt; then
  log "Toy example: EER 6,11% (khớp README), môi trường đúng"
else
  log "CẢNH BÁO: toy example không ra 6,11%; xem results/env/toy_check.txt trước khi chạy tiếp"
fi
if grep -qE "is not in the model|Wrong para. length" results/env/toy_check.txt; then
  log "CẢNH BÁO: có cảnh báo nạp trọng số trong toy example"
fi

# 3b. TF32: SpeechBrain 1.0.2 tự bật TF32 khi import; NVIDIA_TF32_OVERRIDE=0 phải ghi đè được (server/tf32_check.py)
if ! grep -q "NVIDIA_TF32_OVERRIDE=0 .*-> fp32" results/env/tf32_check.txt 2>/dev/null; then
  export MAMBA_ROOT_PREFIX="$ROOT/work/mamba"
  (cd / && NVIDIA_TF32_OVERRIDE=0 "$ROOT/work/bin/micromamba" run -n antideepfake python "$ROOT/server/tf32_check.py") > results/env/tf32_check.txt 2>&1 || true
  (cd / && env -u NVIDIA_TF32_OVERRIDE "$ROOT/work/bin/micromamba" run -n antideepfake python "$ROOT/server/tf32_check.py") >> results/env/tf32_check.txt 2>&1 || true
fi
grep "TF32_CHECK" results/env/tf32_check.txt | tee -a "$LOG" || true
if grep -q "NVIDIA_TF32_OVERRIDE=0 .*-> fp32" results/env/tf32_check.txt; then
  log "TF32: NVIDIA_TF32_OVERRIDE=0 cho kết quả fp32 (như T4)"
else
  log "CẢNH BÁO: NVIDIA_TF32_OVERRIDE=0 không tắt được TF32; xem results/env/tf32_check.txt trước khi chạy tiếp"
fi

# 4. Môi trường điều phối
export MAMBA_ROOT_PREFIX="$ROOT/work/mamba"
MM="$ROOT/work/bin/micromamba"
if ! "$MM" env list | grep -q orpheus-driver; then
  log "Tạo môi trường orpheus-driver"
  "$MM" create -y -n orpheus-driver -c conda-forge python=3.11 pandas numpy scipy pysoundfile safetensors pip >>"$LOG" 2>&1
  "$MM" run -n orpheus-driver pip install -q torch --index-url https://download.pytorch.org/whl/cpu >>"$LOG" 2>&1
fi
DPY="$MAMBA_ROOT_PREFIX/envs/orpheus-driver/bin/python"
"$DPY" -c "import pandas, numpy, scipy, soundfile, safetensors, torch; print('driver', pandas.__version__, numpy.__version__, torch.__version__)" | tee -a "$LOG"
"$MM" run -n orpheus-driver pip freeze > results/env/driver-pip-freeze.txt

# 5. Dữ liệu kiểm tra (RADAR đọc luồng file tar 8,75 GB từ Zenodo; In-the-Wild tập con đọc từng file
#    trong zip qua HTTP Range)
if [ ! -f data/prep/orpheus_data_manifest.json ]; then
  log "Tải RADAR VI, RADAR EN (tập con), In-the-Wild (tập con)"
  "$DPY" code/prep_data.py fetch --ids-dir ids --out data/prep --tmp tmp/prep \
    --sets radar_vi,radar_en_sub,itw_sub >>"$LOG" 2>&1
fi
cp data/prep/orpheus_data_manifest.json results/env/ 2>/dev/null || true
# 5b. Kiểm chế độ D1: chia RADAR VI theo danh sách câu đã commit (protocols/ids/radar_vi_rv_*.tsv)
if [ ! -f data/prep/radar_vi_rv_test.csv ]; then
  "$DPY" code/split_radar_regime.py --ids-dir ids --make-csv data/prep/radar_vi.csv | tee -a "$LOG"
fi

# 6. Kiểm dữ liệu, tải trước checkpoint
"$DPY" server/check_data.py --root "$ROOT" | tee results/env/check_data.txt
"$DPY" server/run.py prefetch >>"$LOG" 2>&1
log "Dựng xong. Chạy thử đo: bash server/vast.sh start --tiers 0 (từ máy)"
