"""Bộ thí nghiệm chạy trên server thuê (vast.ai): một bộ hoàn chỉnh, mọi cấu hình đem so với nhau
đều chạy trên cùng một loại máy. Nguồn duy nhất cho server/run.py. Giải thích: server/README.md,
chore/GD6-GRPO.md mục 9, 13, 14.

server/run.py ĐỌC LẠI file này mỗi lần một GPU nhận việc: thêm, bớt lần chạy ở điểm kiểm (sửa file, `vast.sh code`)
có hiệu lực ngay, không dừng lần đang chạy. Lần chạy đã xong (results/jobs/<tên>.json) không chạy lại.

Vì sao chạy lại cả GĐ5: kết quả từ hai phần cứng khác nhau (T4 trên Kaggle, GPU 24 GB trên server)
không nên đặt vào cùng một phép so sánh (khác kernel số học, khác tốc độ đọc dữ liệu). Bộ Kaggle (R × 5, G1, G2,
A0, P0, B2, B1, N0 một seed) giữ làm lần lặp độc lập trên phần cứng khác.

Công thức huấn luyện giữ NGUYÊN như Kaggle (GĐ5 mục 4.1, lịch E1'; GĐ6 mục 3, 5):
  - SFT: bộ ghi đè SFT; GRPO: thêm file YAML grpo_v1/v2, expandable_segments, --oom-retry 0.
  - Mọi lần: --oom-retry 0 (không tự giảm batch), --probe-mem (ghi đỉnh bộ nhớ thật).
  - Môi trường: NVIDIA_TF32_OVERRIDE=0 (SpeechBrain 1.0.2 tự bật TF32 khi import; biến này ghi đè ở mức
    cuBLAS/cuDNN: tính fp32 như T4).

Người dùng chốt 15/11 (mục 14): làm ĐẦY ĐỦ, chia giai đoạn có điểm kiểm. Tầng = thứ tự ưu tiên; mọi tầng đưa vào
hàng đợi ngay, GPU lấy lần chưa xong có tầng nhỏ nhất mà các lần nó cần (needs) đã xong:
  0  (A) thử đo: tốc độ, bộ nhớ, số worker (2 hoặc 6), dấu vân tay batch; thử nhanh SFT + KL
  1  (B) lõi: S0 (dùng thẳng, 4 backbone), R, G1, A0 × 3 seed
  2  (C) chẩn đoán và sửa × 3 seed: G1s, G1b (bản sửa); G1lr, Rlr (cường độ); G3 (vai trò KL);
         SFTKL, G1s0 (đối chứng quy kết); G2 (thành phần GRPO của plan, chạy cuối tầng)
  3  (D) WiSE-FT: nội suy S0 ↔ bản fine-tune (chỉ chấm): α 0,5 cả 3 seed, α 0,25 và 0,75 seed 1; R, A0, G1, G1s, G1b
  4  (D) không RawBoost: G1sA, G1bA (bản sửa ghép), G1A (G1 gốc) × 3 seed
  5  (D) seed 4, 5: R, G1, G1s, G1b
  6  (E) các yếu tố GĐ5: P0, B2, N0, B1 × 3 seed
  7  (E) kiểm chế độ D1: fine-tune trên nửa RADAR VI (src/split_radar_regime.py): R, G1 × 3 seed; S0 trên hai tập chia
  8  (E) ít mẫu: SFT, G1 với N = 50/200/1000 mỗi lớp × 3 seed (cùng 4.563 bước); bản sửa tốt nhất thêm ở điểm kiểm C
  9  (E) mốc XLS-R-2B dùng thẳng trên RADAR VI
"""
import math

# Số worker đọc dữ liệu của mỗi lần huấn luyện (Kaggle: 2, vì 4 nhân CPU cho 2 GPU). Chốt sau lần thử đo (tầng 0)
# và GIỮ NGUYÊN cho mọi lần của bộ server. Ghi đè bằng `run.py --workers N`.
NUM_WORKERS = 2

STEPS = 4563          # số bước optimizer của bản đủ: 3 epoch × 1.521 batch (R trên Kaggle)
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test"]
S0_MODELS = ["mms_300m", "mms_300m_nda", "w2v_large", "w2v_small"]
GRPO_ENV = ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
COMMON_ARGS = ["--oom-retry", "0", "--probe-mem"]
# Lần thử đo: in dấu vân tay 30 batch đầu mỗi worker (src/mem_probe.py), để kiểm cùng seed thì SFT và GRPO
# có nhận cùng batch, cùng phép tăng cường RawBoost không (quyết định cách phân tích qua seed, mục 13)
PROBE_ENV = ["BATCH_HASH_N=30"]

# Batch mỗi epoch ở max_batch_length 50: mô phỏng DynamicBatchSampler của SpeechBrain 1.0.2 (cùng tham số YAML;
# kiểm: seaspoof_vi_train ra 1.521, đúng số đo của R)
FEW_SHOT_BPE = {"n50": 12, "n200": 42, "n1000": 178}
RV_BPE = 652          # radar_vi_rv_train (6.047 câu, 7,73 giờ): 4.563 bước = 7,0 epoch

# Kiểm chế độ D1: train/dev là nửa A của RADAR VI, test là nửa B (trong miền); các tập còn lại ngoài miền
RV_TRAIN, RV_DEV, RV_TEST = "radar_vi_rv_train", "radar_vi_rv_dev", "radar_vi_rv_test"
RV_SCORE = [RV_DEV, RV_TEST, "ood_forget", "seaspoof_vi_test", "mlaad_vi_test"]

# WiSE-FT: cấu hình nguồn và α
WISE_SRC = ("R", "A0", "G1", "G1s", "G1b")


def sft_set(workers, seed, epochs=3, step_limit=4600):
    """Bộ ghi đè SFT của GĐ5 (lịch E1'). optimizer_step_limit là tham số chạy của SpeechBrain: chỉ
    dừng huấn luyện, không vào lịch learning rate (GĐ5 phát hiện 1)."""
    return ["max_batch_length=50", f"number_of_epochs={epochs}", f"optimizer_step_limit={step_limit}",
            "warmup=460", "lr=1e-6", "valid_step=1000000000", f"num_workers={workers}", f"seed={seed}"]


def train_job(name, tier, model="mms_300m", method="sft", init="post", seed=1, extra_set=(), train="seaspoof_vi_train",
              epochs=3, step_limit=4600, workers=NUM_WORKERS, args=(), score=tuple(SCORE), valid="seaspoof_vi_dev",
              env=()):
    # finetune.py --env là nargs="*": gộp mọi biến vào MỘT cờ (viết hai lần thì lần sau ghi đè lần trước)
    e = list(env) + (list(GRPO_ENV) if method.startswith("grpo") else [])
    a = list(COMMON_ARGS) + list(args) + (["--env", *e] if e else [])
    return {"name": name, "tier": tier, "kind": "train", "model": model, "method": method, "init": init,
            "train": train, "valid": valid, "set": sft_set(workers, seed, epochs, step_limit) + list(extra_set),
            "args": a, "score": list(score)}


def zeroshot_job(name, tier, models, sets):
    return {"name": name, "tier": tier, "kind": "zeroshot",
            "jobs": [f"{s}:{','.join(models)}" for s in sets]}


def interp_job(name, tier, src, alpha, model="mms_300m", score=tuple(SCORE)):
    """WiSE-FT: nội suy S0 ↔ <src>.ckpt rồi chấm (src/interp_score.py). Chờ lần nguồn xong (needs)."""
    return {"name": name, "tier": tier, "kind": "interp", "src": src, "alpha": alpha, "model": model,
            "score": list(score), "needs": [src]}


# Cấu hình: mã -> tham số (như GĐ5 mục 1 và GĐ6 mục 1, 12, 13)
CONFIGS = {
    "R": dict(),
    "A0": dict(extra_set=["use_da=False"]),
    "G1": dict(method="grpo_v1"),
    "G2": dict(method="grpo_v2"),
    "G3": dict(method="grpo_v1", extra_set=["rl_config={beta: 0.0}"]),
    # Sửa GRPO (GĐ6 mục 11–12): khi miền fine-tune đã bão hoà, chia phần thưởng cho độ lệch chuẩn của
    # batch (rất nhỏ) khuếch đại cập nhật, GRPO tiếp tục làm sắc câu đã đúng (~30 lần SFT ở p = 0,999).
    # std_floor 1.0: mẫu số σ + 1 gần như hằng, tức gần như bỏ chia cho độ lệch chuẩn; kỳ vọng gradient
    # theo log-odds nhãn đúng thành p(1 − p) (bằng SFT ở p gần 1, nhẹ tay với câu sai chắc). Lưu ý: cũng
    # làm bước cập nhật nhỏ hơn SFT ở câu ranh giới (0,25 so với 0,5), nên so thêm với G1lr.
    "G1s": dict(method="grpo_v1", extra_set=["rl_config={std_floor: 1.0}"]),
    "G1b": dict(method="grpo_v1", extra_set=["rl_config={beta: 1.0}"]),        # phạt KL mạnh (núm của bài báo)
    "G1lr": dict(method="grpo_v1", extra_set=["lr=1e-7"]),                    # đối chứng: chỉ giảm cường độ
    "Rlr": dict(extra_set=["lr=1e-7"]),                                       # cặp SFT của G1lr (mục 13.5 B1)
    # G1s còn làm phạt KL nặng lên tương đối khi bão hoà (biên độ lợi thế co về ~0,1, KL giữ nguyên;
    # chore/GD6-GRPO.md mục 12.1). G1s0 = G1s với β = 0: cùng G1, G3, G1s thành bảng 2 × 2
    # {chia σ_batch hay không} × {β 0,04 hay 0}, tách "bỏ khuếch đại" khỏi "KL nặng lên"
    "G1s0": dict(method="grpo_v1", extra_set=["rl_config={std_floor: 1.0, beta: 0.0}"]),
    # SFT + KL (mục 13.3, 13.5 B2): cross-entropy như R + đúng số hạng KL k3 của GRPO (β 0,04 của YAML grpo_v2),
    # qua src/loss_variants.py (không sửa code gốc). Cùng điểm cân bằng với G1s (Δz* ≈ 3,2 so với 3,1)
    "SFTKL": dict(method="grpo_v2", args=["--loss-variant", "sft_kl"]),
    "G1sA": dict(method="grpo_v1", extra_set=["rl_config={std_floor: 1.0}", "use_da=False"]),
    "G1bA": dict(method="grpo_v1", extra_set=["rl_config={beta: 1.0}", "use_da=False"]),
    "G1A": dict(method="grpo_v1", extra_set=["use_da=False"]),                # D3: G1 gốc không RawBoost
    "P0": dict(init="fairseq"),
    "B2": dict(model="w2v_large"),
    "N0": dict(model="mms_300m_nda"),
    "B1": dict(model="w2v_small"),
    # D1 kiểm chế độ: cùng công thức, cùng 4.563 bước; dừng bằng optimizer_step_limit (epoch dư dừng ngay ở batch đầu)
    "Rrv": dict(train=RV_TRAIN, valid=RV_DEV, score=RV_SCORE, epochs=math.ceil(STEPS / RV_BPE) + 2, step_limit=STEPS),
    "G1rv": dict(method="grpo_v1", train=RV_TRAIN, valid=RV_DEV, score=RV_SCORE,
                 epochs=math.ceil(STEPS / RV_BPE) + 2, step_limit=STEPS),
}


def build(workers=NUM_WORKERS):
    J = []

    def cfg(code, seed, tier, **over):
        kw = dict(CONFIGS[code])
        kw["extra_set"] = list(kw.get("extra_set", [])) + list(over.pop("extra_set", []))
        kw["args"] = list(over.pop("args", [])) + list(kw.get("args", []))
        kw.setdefault("workers", workers)
        kw.update(over)
        name = kw.pop("name", f"{code}_s{seed}")
        return train_job(name, tier, seed=seed, **kw)

    # Tầng 0 (A): thử đo (không giữ checkpoint, không chấm); 2 và 6 worker cho SFT và G1; SFT + KL thử nhanh
    for code in ("R", "G1"):
        for w in (2, 6):
            J.append(cfg(code, 1, 0, name=f"probe_{code}_w{w}", step_limit=200, epochs=1, workers=w,
                         args=["--discard-ckpt"], score=(), valid="seaspoof_vi_train_n50", env=PROBE_ENV))
    J.append(cfg("SFTKL", 1, 0, name="probe_SFTKL_w2", step_limit=100, epochs=1, workers=2,
                 args=["--discard-ckpt"], score=(), valid="seaspoof_vi_train_n50", env=PROBE_ENV))

    # Tầng 1 (B): S0 (mốc chung của mọi bảng), R, G1, A0; xen kẽ theo seed để có seed 1 của mọi cấu hình sớm
    for m in S0_MODELS:   # mỗi backbone một lần chấm, để chia ra nhiều GPU
        J.append(zeroshot_job(f"S0_{m}", 1, [m], SCORE))
    for s in (1, 2, 3):
        for code in ("R", "G1", "A0"):
            J.append(cfg(code, s, 1))
    # Tầng 2 (C): bản sửa trước, rồi cường độ và KL, rồi đối chứng quy kết, G2 cuối
    for group in (("G1s", "G1b"), ("G1lr", "Rlr", "G3"), ("SFTKL", "G1s0"), ("G2",)):
        for s in (1, 2, 3):
            for code in group:
                J.append(cfg(code, s, 2))
    # Tầng 3 (D): WiSE-FT, α 0,5 cả 3 seed; α 0,25 và 0,75 ở seed 1
    for code in WISE_SRC:
        for s in (1, 2, 3):
            J.append(interp_job(f"{code}w50_s{s}", 3, f"{code}_s{s}", 0.5))
        for al in (0.25, 0.75):
            J.append(interp_job(f"{code}w{int(al * 100)}_s1", 3, f"{code}_s1", al))
    # Tầng 4 (D): không RawBoost
    for s in (1, 2, 3):
        for code in ("G1sA", "G1bA", "G1A"):
            J.append(cfg(code, s, 4))
    # Tầng 5 (D): seed 4, 5
    for s in (4, 5):
        for code in ("R", "G1", "G1s", "G1b"):
            J.append(cfg(code, s, 5))
    # Tầng 6 (E): các yếu tố của GĐ5
    for s in (1, 2, 3):
        for code in ("P0", "B2", "N0", "B1"):
            J.append(cfg(code, s, 6))
    # Tầng 7 (E): kiểm chế độ D1
    J.append(zeroshot_job("S0rv_mms_300m", 7, ["mms_300m"], [RV_DEV, RV_TEST]))
    for s in (1, 2, 3):
        for code in ("Rrv", "G1rv"):
            J.append(cfg(code, s, 7))
    # Tầng 8 (E): ít mẫu. Cùng STEPS bước và lịch E1' như bản đủ: dừng bằng optimizer_step_limit, số epoch
    # đủ lớn để giới hạn số bước là thứ dừng trước (dư 2 epoch; epoch dư dừng ngay ở batch đầu)
    for s in (1, 2, 3):
        for n, bpe in FEW_SHOT_BPE.items():
            ep = math.ceil(STEPS / bpe) + 2
            for code, method in (("SFT", "sft"), ("G1", "grpo_v1")):
                J.append(train_job(f"{code}{n}_s{s}", 8, method=method, seed=s, train=f"seaspoof_vi_train_{n}",
                                   epochs=ep, step_limit=STEPS, workers=workers))
    # Tầng 9 (E): mốc XLS-R-2B (chỉ chấm)
    J.append(zeroshot_job("ref_xlsr_2b", 9, ["xlsr_2b"], ["radar_vi"]))

    names = [j["name"] for j in J]
    assert len(names) == len(set(names)), "trùng tên lần chạy"
    known = set(names)
    for j in J:
        for n in j.get("needs", []):
            assert n in known, f"{j['name']} cần {n} nhưng không có lần chạy này"
    return J


def prefetch_list(jobs):
    """(mô hình, cách khởi tạo) cần tải trước, để các GPU không cùng tải một file."""
    out = set()
    for j in jobs:
        if j["kind"] == "train":
            out.add((j["model"], j["init"]))
        elif j["kind"] == "interp":
            out.add((j["model"], "post"))
        else:
            for spec in j["jobs"]:
                for m in spec.split(":", 1)[1].split(","):
                    out.add((m, "post"))
    return sorted(out)


if __name__ == "__main__":
    jobs = build()
    by = {}
    for j in jobs:
        by.setdefault(j["tier"], []).append(j["name"])
    for t in sorted(by):
        print(f"tầng {t}: {len(by[t])} lần: {' '.join(by[t])}")
    print(f"tổng {len(jobs)} lần; tải trước:", prefetch_list(jobs))
