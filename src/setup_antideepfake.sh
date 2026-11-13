#!/usr/bin/env bash
# Dựng môi trường AntiDeepfake trên Colab/Kaggle (Linux + GPU NVIDIA) và chạy toy example.
# CHƯA CHẠY THỬ: viết theo install.sh và README của repo, cần kiểm chứng ở GĐ2 ngày 2.
#
# Cách dùng (thường gọi từ colab/GD2_toy_example.ipynb):
#   bash src/setup_antideepfake.sh <BASE> [all|env|toy]
#   CACHE_DIR=/content/drive/MyDrive/orpheus/cache bash src/setup_antideepfake.sh /content/work toy
#
# BASE      : thư mục làm việc trên máy Colab (mất khi hết phiên).
# CACHE_DIR : nơi giữ file tải về (checkpoint, dữ liệu toy). Đặt trên Drive để
#             phiên sau không phải tải lại. Mặc định: $BASE/downloads.
#
# Cấu trúc tạo ra (giống mục "Working directory structure" của README AntiDeepfake):
#   $BASE/Data/toy_example/   dữ liệu
#   $BASE/fairseq/            fairseq commit cố định
#   $BASE/AntiDeepfake/       code, commit cố định
#   $BASE/Log/exps/           kết quả
#   $BASE/setup.log, versions.txt, pip-freeze.txt
#
# Chạy lại được: bước nào đã xong thì bỏ qua.

set -euo pipefail

BASE="${1:-$PWD/work}"
STEP="${2:-all}"   # all | env | toy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$BASE/downloads}"

# Commit cố định để chạy lại cho ra cùng kết quả.
# AntiDeepfake: HEAD của main ngày 2026-11-12. fairseq: commit install.sh yêu cầu.
ANTIDEEPFAKE_COMMIT=86098091c52c5d0e38c44396c95608f93b0f648d
FAIRSEQ_COMMIT=862efab86f649c04ea31545ce28d13c59560113d

TOY_URL=https://zenodo.org/records/7497769/files/project-04-toy_example.tar.gz
CKPT_URL=https://zenodo.org/records/15580543/files/mms_300m.ckpt

mkdir -p "$BASE"/{Data,Log/exps,Log/ssl-weights} "$CACHE_DIR"
cd "$BASE"
export MAMBA_ROOT_PREFIX="$BASE/mamba"
MM="$BASE/bin/micromamba"
RUN="$MM run -n antideepfake"
# Kiểm tra gói đã cài chưa. Chạy từ "/" để Python không nhầm thư mục
# $BASE/fairseq (repo vừa clone) với gói fairseq đã cài.
has() { (cd / && $RUN python -c "import $1" >/dev/null 2>&1); }
# Gói đã được pip cài chưa (không cần import được). Dùng cho fairseq: fairseq
# chỉ import được SAU khi numpy được hạ về 1.21.2 ở bước 4 (fairseq dùng np.float,
# đã bị xoá từ numpy 1.24).
installed() { $RUN pip show "$1" >/dev/null 2>&1; }
LOG="$BASE/setup.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# Lệnh nào lỗi thì in ra dòng và lệnh đó trước khi dừng (set -E để áp dụng cả trong hàm).
set -E
trap 'log "LỖI ở dòng $LINENO: $BASH_COMMAND. Chi tiết: tail -n 50 $LOG"' ERR

# Tải về CACHE_DIR nếu chưa có. Tải vào file tạm rồi đổi tên, để lần tải dở
# không để lại file hỏng trông như đã xong.
fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    log "Đã có $(basename "$dest") trong cache, bỏ qua tải"
    return
  fi
  log "Tải $(basename "$dest")"
  wget -q --show-progress -O "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

setup_env() {
  local t0=$SECONDS

  # 1. micromamba: Colab/Kaggle dùng Python 3.10+, repo cần Python 3.9.
  if [ ! -x "$MM" ]; then
    log "Tải micromamba"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  fi
  if ! $MM env list | grep -q antideepfake; then
    log "Tạo env Python 3.9, pip 24.0"
    # pip 24.0: pip mới hơn từ chối metadata cũ của omegaconf/hydra mà fairseq cần.
    $MM create -y -n antideepfake -c conda-forge python=3.9 pip=24.0 >>"$LOG" 2>&1
  fi

  # 2. PyTorch 2.6.0 bản CUDA 11.8 (chạy được trên T4 và P100).
  if ! has torch; then
    log "Cài torch 2.6.0 + torchaudio 2.6.0 (cu118)"
    $RUN pip install torch==2.6.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu118 >>"$LOG" 2>&1
  fi

  # 3. fairseq ở commit cố định. Đặt ngoài thư mục AntiDeepfake theo README.
  if [ ! -d "$BASE/fairseq" ]; then
    log "Clone fairseq"
    git clone -q https://github.com/pytorch/fairseq "$BASE/fairseq"
  fi
  local fairseq_new=0
  if ! installed fairseq; then
    log "Cài fairseq $FAIRSEQ_COMMIT (biên dịch, mất vài phút)"
    (cd "$BASE/fairseq" && git checkout -q "$FAIRSEQ_COMMIT")
    # Cách của install.sh. Nếu setuptools mới không build được (môi trường build
    # tách biệt thiếu numpy), thử lại với cython/numpy cài sẵn và tắt cô lập.
    if ! (cd "$BASE/fairseq" && $RUN pip install --editable . >>"$LOG" 2>&1); then
      log "Build fairseq lần 1 lỗi, thử lại với --no-build-isolation"
      $RUN pip install cython "numpy<2" >>"$LOG" 2>&1
      (cd "$BASE/fairseq" && $RUN pip install --no-build-isolation --editable . >>"$LOG" 2>&1)
    fi
    installed fairseq || { log "LỖI: pip không cài được fairseq, xem $LOG"; exit 1; }
    fairseq_new=1
  fi

  # 4. SpeechBrain và các gói còn lại, đúng phiên bản trong install.sh.
  # Phải chạy SAU fairseq (như install.sh) để các phiên bản cố định là bản cuối
  # cùng được cài; vừa cài fairseq thì cài lại.
  if ! has speechbrain || [ "$fairseq_new" = 1 ]; then
    log "Cài speechbrain 1.0.2 và các gói phụ"
    $RUN pip install speechbrain==1.0.2 >>"$LOG" 2>&1
    $RUN pip install tensorboard tensorboardX soundfile pandarallel scikit-learn \
      numpy==1.21.2 pandas==1.4.3 scipy==1.7.2 >>"$LOG" 2>&1
  fi

  # Giờ mới import thử fairseq (sau khi numpy đã về 1.21.2). Lỗi thì in cả
  # traceback ra màn hình và vào log.
  if ! has fairseq.models.wav2vec; then
    log "LỖI: không import được fairseq.models.wav2vec. Traceback:"
    (cd / && $RUN python -c "import numpy; print('numpy', numpy.__version__); import fairseq.models.wav2vec") 2>&1 | tail -n 25 | tee -a "$LOG" || true
    exit 1
  fi

  # 5. Code AntiDeepfake ở commit cố định.
  if [ ! -d "$BASE/AntiDeepfake" ]; then
    log "Clone AntiDeepfake"
    git clone -q https://github.com/nii-yamagishilab/AntiDeepfake.git "$BASE/AntiDeepfake"
  fi
  (cd "$BASE/AntiDeepfake" && git checkout -q "$ANTIDEEPFAKE_COMMIT")

  log "Kiểm tra môi trường"
  (cd / && $RUN python -) <<'EOF' | tee "$BASE/versions.txt" | tee -a "$LOG"
import sys, torch, torchaudio, fairseq, speechbrain, numpy, pandas
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("torchaudio  ", torchaudio.__version__)
print("fairseq     ", fairseq.__version__)
print("speechbrain ", speechbrain.__version__)
print("numpy/pandas", numpy.__version__, pandas.__version__)
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
print("fairseq Wav2Vec2Model import: OK")
EOF
  echo "AntiDeepfake $ANTIDEEPFAKE_COMMIT" >> "$BASE/versions.txt"
  echo "fairseq      $FAIRSEQ_COMMIT" >> "$BASE/versions.txt"
  $RUN pip freeze > "$BASE/pip-freeze.txt"
  log "Dựng môi trường mất $(( (SECONDS - t0) / 60 )) phút"
}

run_toy() {
  # Dữ liệu toy: 150 câu thuộc ASVspoof 2019 LA (16 thật, 134 giả).
  # AntiDeepfake đã dùng ASVspoof 2019 LA để huấn luyện, nên đây chỉ là
  # kiểm tra code chạy đúng, không phải kết quả ngoài miền.
  if [ ! -x "$MM" ] || ! has fairseq.models.wav2vec; then
    log "LỖI: chưa có môi trường. Chạy bước env trước."
    exit 1
  fi

  if [ ! -d "$BASE/Data/toy_example/eval" ]; then
    fetch "$TOY_URL" "$CACHE_DIR/toy_example.tar.gz"
    tar -xzf "$CACHE_DIR/toy_example.tar.gz" -C "$BASE/Data"
  fi
  if [ ! -d "$BASE/Data/toy_example/eval" ]; then
    log "LỖI: không thấy $BASE/Data/toy_example/eval sau khi giải nén. Nội dung Data/:"
    ls -la "$BASE/Data" | tee -a "$LOG"
    exit 1
  fi

  CKPT="$CACHE_DIR/mms_300m.ckpt"
  fetch "$CKPT_URL" "$CKPT"

  # exp_name mới: nếu thư mục exp đã có checkpoint, inference sẽ nạp checkpoint đó
  # thay vì --ckpt_path.
  EXP=eval_toy_mms_300m
  OUT="$BASE/Log/exps/exp_mms_300m_$EXP"
  rm -rf "$OUT"

  log "Inference toy example với mms_300m"
  local t0=$SECONDS
  (cd "$BASE/AntiDeepfake" && $RUN python main.py inference hparams/mms_300m.yaml \
      --base_path "$BASE" \
      --exp_name "$EXP" \
      --test_csv protocols/toy_example_test.csv \
      --ckpt_path "$CKPT" 2>&1 | tee "$BASE/toy_inference.log")
  log "Inference mất $(( SECONDS - t0 )) giây (gồm cả nạp mô hình)"

  # load_weights() chỉ in cảnh báo rồi chạy tiếp khi trọng số không khớp.
  # Có dòng nào ở đây nghĩa là mô hình đang chạy với trọng số ngẫu nhiên một phần.
  if grep -E "is not in the model|Wrong para. length" "$BASE/toy_inference.log" >/dev/null; then
    log "CẢNH BÁO: có trọng số không nạp được:"
    grep -E "is not in the model|Wrong para. length" "$BASE/toy_inference.log" | head -20 | tee -a "$LOG"
  else
    log "Nạp trọng số: không có cảnh báo"
  fi

  log "evaluation.py của repo (mong đợi eer = 0.0611)"
  (cd "$BASE/AntiDeepfake" && $RUN python evaluation.py "$OUT/evaluation_score.csv" | tee -a "$LOG")

  log "evaluate.py của nhóm (EER phải trùng evaluation.py)"
  $RUN python "$SCRIPT_DIR/evaluate.py" "$OUT/evaluation_score.csv" \
    --protocol "$BASE/AntiDeepfake/protocols/toy_example_test.csv" --group-col Attack \
    --out "$OUT/metrics.csv" | tee -a "$LOG"
}

case "$STEP" in
  env) setup_env ;;
  toy) run_toy ;;
  all) setup_env; run_toy ;;
  *) echo "STEP phải là all | env | toy"; exit 2 ;;
esac
log "Xong bước $STEP. Log đầy đủ: $LOG"
