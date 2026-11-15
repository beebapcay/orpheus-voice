#!/usr/bin/env bash
# Phân tích bộ server trên máy (chạy sau `vast.sh fetch` + `vast.sh collect <nhãn>`), chỉ trên cây
# results/server/ (không trộn kết quả Kaggle). Cần .venv có pandas, numpy, scipy:
#   uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install pandas numpy scipy
#
#   bash server/analyze.sh
#
# Ghi ra:
#   results/server/analysis/     table_main.csv, table_seeds.csv, seed_tests.csv (so qua seed, t-test),
#                                bootstrap_pairs.csv, table_by_system*.csv, hard_ids.csv, mechanism_bins.csv,
#                                ablation_output.txt, mechanism_output.txt
#   results/server/calibration/  calibration.csv, params.csv, calibration_output.txt
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
S="${SERVER_RESULTS:-$REPO/results/server}"   # ghi đè để thử trên cây giả lập
[ -d "$S/finetune" ] || { echo "Chưa có $S/finetune: chạy vast.sh fetch rồi collect trước" >&2; exit 2; }
mkdir -p "$S/analysis" "$S/calibration"
cd "$REPO"

# So qua seed (mục 5 của analyze_ablation.py): A:B theo mã cấu hình. Cặp mặc định của script, rồi các
# cặp của hướng "chẩn đoán và sửa" (chore/GD6-GRPO.md mục 12): G1s:G1lr tách "đổi cơ chế" khỏi "giảm
# cường độ"; G1s:G1b chọn cách sửa tốt hơn (bản ghép tầng 4); G1sA, G1bA so với A0 (cùng không RawBoost).
PAIRS=(G1:R G1:A0 G2:G1 G3:G1 G2:R A0:R N0:R B2:R P0:R R:S0_mms_300m G1:S0_mms_300m G2:S0_mms_300m
       B1:B2 B2:S0_w2v_large N0:S0_mms_300m_nda B1:S0_w2v_small
       G1s:R G1b:R G1lr:R G1s:G1 G1b:G1 G1lr:G1 G1s:G1lr G1b:G1lr G1s:G1b
       G1s:S0_mms_300m G1b:S0_mms_300m G1lr:S0_mms_300m
       G1sA:A0 G1bA:A0 G1sA:G1s G1bA:G1b A0:S0_mms_300m G1sA:S0_mms_300m G1bA:S0_mms_300m
       G1n50:SFTn50 G1n200:SFTn200 G1n1000:SFTn1000
       SFTn50:R SFTn200:R SFTn1000:R G1n50:G1 G1n200:G1 G1n1000:G1
       Rlr:R G1lr:Rlr Rlr:S0_mms_300m G1lr:S0_mms_300m
       SFTKL:R SFTKL:G1s SFTKL:G1b G1s0:G1s G1s0:G3 G1s0:R
       G1A:A0 G1A:G1 G1sA:G1A G1bA:G1A
       Rw50:R Rw50:S0_mms_300m A0w50:A0 A0w50:S0_mms_300m G1w50:G1 G1w50:S0_mms_300m
       G1sw50:G1s G1bw50:G1b Rw50:A0 G1rv:Rrv)

echo "== analyze_ablation.py"
"$PY" src/analyze_ablation.py --finetune "$S/finetune" --zeroshot "$S/zeroshot" --out "$S/analysis" \
  --group-pairs "${PAIRS[@]}" > "$S/analysis/ablation_output.txt"
echo "== calibrate.py"
"$PY" src/calibrate.py --finetune "$S/finetune" --zeroshot "$S/zeroshot" --out "$S/calibration" \
  > "$S/calibration/calibration_output.txt"

# Cơ chế: chỉ các lần cùng điểm xuất phát với S0 MMS-300M (post-trained): R, Rlr, A0, SFTKL, G*, SFTn*, WiSE-FT
# (Rw*, A0w*, G1*w*) (không P0: khởi tạo fairseq; không B1, B2, N0: backbone khác)
CFGS=()
while IFS= read -r r; do CFGS+=("$r"); done < <(ls "$S/finetune" | grep -E '^(R|Rlr|A0|SFTKL|G[0-9A-Za-z]*|SFTn[0-9]+|(R|A0)w[0-9]+)_s[0-9]+$' || true)
if [ "${#CFGS[@]}" -gt 0 ]; then
  echo "== analyze_mechanism.py (${#CFGS[@]} lần)"
  "$PY" src/analyze_mechanism.py --finetune "$S/finetune" --zeroshot "$S/zeroshot" --out "$S/analysis" \
    --configs "${CFGS[@]}" > "$S/analysis/mechanism_output.txt"
fi
ls "$S/analysis" "$S/calibration"
