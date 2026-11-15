"""Bộ thí nghiệm chạy trên server thuê (vast.ai): một bộ hoàn chỉnh, mọi cấu hình đem so với nhau
đều chạy trên cùng một loại máy. Nguồn duy nhất cho server/run.py. Giải thích: server/README.md,
chore/GD6-GRPO.md mục 9.

Vì sao chạy lại cả GĐ5: kết quả từ hai phần cứng khác nhau (T4 trên Kaggle, GPU 24 GB trên server)
không nên đặt vào cùng một phép so sánh (khác kernel số học, khác tốc độ đọc dữ liệu nên khác chuỗi
ngẫu nhiên của RawBoost). Bộ Kaggle (R × 5, G1 × 5, A0, P0, B2, B1, N0, G2, G3 một seed) giữ làm
lần lặp độc lập trên phần cứng khác.

Công thức huấn luyện giữ NGUYÊN như Kaggle (GĐ5 mục 4.1, lịch E1'; GĐ6 mục 3, 5):
  - SFT: bộ ghi đè SFT; GRPO: thêm file YAML grpo_v1/v2, expandable_segments, --oom-retry 0.
  - Mọi lần: --oom-retry 0 (không tự giảm batch), --probe-mem (ghi đỉnh bộ nhớ thật).
  - Môi trường: NVIDIA_TF32_OVERRIDE=0 (tắt TF32 trên GPU Ampere/Ada: tính fp32 như T4).
Mỗi cấu hình 3 seed (GĐ5 phát hiện 8, GĐ6 mục 7.1); R và G1 thêm seed 4, 5 ở tầng cuối.

Tầng (chạy theo thứ tự; tầng sau chỉ bắt đầu khi GPU rảnh và tầng trước đã được xếp hết). Câu hỏi 3 theo
hướng "chẩn đoán và sửa" (chốt 15/11, chore/GD6-GRPO.md mục 11, 12):
  0  thử đo: tốc độ, bộ nhớ, số worker đọc dữ liệu (chạy riêng, trước tất cả)
  1  lõi: S0 (dùng thẳng, 4 backbone), R, G1, A0 × 3 seed
  2  sửa GRPO và kiểm cơ chế × 3 seed: G1s (bỏ khuếch đại: std_floor 1.0), G1b (beta 1.0),
     G1lr (lr 1e-7, đối chứng cường độ); G2, G3 (thành phần GRPO của plan)
  3  các yếu tố GĐ5: P0, B2, N0, B1 × 3 seed
  4  ghép bản sửa với công thức tốt nhất của GĐ5 (không RawBoost khi fine-tune): G1sA, G1bA × 3 seed.
     Chỉ cần bản ghép của cách sửa tốt hơn ở tầng 2; bỏ bản kia bằng `vast.sh skip <tên>...`
  5  ít mẫu (ngày 2 GĐ6): SFT và G1 với N = 50/200/1000 mỗi lớp × 3 seed (cùng 4.563 bước như bản đủ)
  6  thêm seed 4, 5 cho R và G1; mốc tham chiếu XLS-R-2B dùng thẳng trên RADAR VI
"""
import math

# Số worker đọc dữ liệu của mỗi lần huấn luyện (Kaggle: 2, vì 4 nhân CPU cho 2 GPU). Trên server
# GPU nhanh hơn nên 2 worker có thể là nút thắt; chốt sau lần thử đo (tầng 0) và GIỮ NGUYÊN cho mọi
# lần của bộ server. Ghi đè bằng `run.py --workers N`.
NUM_WORKERS = 2

STEPS = 4563          # số bước optimizer của bản đủ: 3 epoch × 1.521 batch (R trên Kaggle)
SCORE = ["seaspoof_vi_dev", "seaspoof_vi_test", "radar_vi", "ood_forget", "mlaad_vi_test"]
S0_MODELS = ["mms_300m", "mms_300m_nda", "w2v_large", "w2v_small"]
GRPO_ARGS = ["--env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
COMMON_ARGS = ["--oom-retry", "0", "--probe-mem"]

# Batch mỗi epoch ở max_batch_length 50 (mô phỏng DynamicBatchSampler, GĐ6 mục 8; n50 khớp pilot GĐ4)
FEW_SHOT_BPE = {"n50": 12, "n200": 42, "n1000": 178}


def sft_set(workers, seed, epochs=3, step_limit=4600):
    """Bộ ghi đè SFT của GĐ5 (lịch E1'). optimizer_step_limit là tham số chạy của SpeechBrain: chỉ
    dừng huấn luyện, không vào lịch learning rate (GĐ5 phát hiện 1)."""
    return ["max_batch_length=50", f"number_of_epochs={epochs}", f"optimizer_step_limit={step_limit}",
            "warmup=460", "lr=1e-6", "valid_step=1000000000", f"num_workers={workers}", f"seed={seed}"]


def train_job(name, tier, model="mms_300m", method="sft", init="post", seed=1, extra_set=(), train="seaspoof_vi_train",
              epochs=3, step_limit=4600, workers=NUM_WORKERS, args=(), score=tuple(SCORE), valid="seaspoof_vi_dev"):
    a = list(COMMON_ARGS) + list(args) + (list(GRPO_ARGS) if method.startswith("grpo") else [])
    return {"name": name, "tier": tier, "kind": "train", "model": model, "method": method, "init": init,
            "train": train, "valid": valid, "set": sft_set(workers, seed, epochs, step_limit) + list(extra_set),
            "args": a, "score": list(score)}


def zeroshot_job(name, tier, models, sets):
    return {"name": name, "tier": tier, "kind": "zeroshot",
            "jobs": [f"{s}:{','.join(models)}" for s in sets]}


# Cấu hình: mã -> tham số (như GĐ5 mục 1 và GĐ6 mục 1)
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
    "G1sA": dict(method="grpo_v1", extra_set=["rl_config={std_floor: 1.0}", "use_da=False"]),
    "G1bA": dict(method="grpo_v1", extra_set=["rl_config={beta: 1.0}", "use_da=False"]),
    "P0": dict(init="fairseq"),
    "B2": dict(model="w2v_large"),
    "N0": dict(model="mms_300m_nda"),
    "B1": dict(model="w2v_small"),
}


def build(workers=NUM_WORKERS):
    J = []

    def cfg(code, seed, tier, **over):
        kw = dict(CONFIGS[code])
        kw["extra_set"] = list(kw.get("extra_set", [])) + list(over.pop("extra_set", []))
        kw.setdefault("workers", workers)
        kw.update(over)
        name = kw.pop("name", f"{code}_s{seed}")
        return train_job(name, tier, seed=seed, **kw)

    # Tầng 0: thử đo (không giữ checkpoint, không chấm); 2 và 6 worker cho SFT và G1
    for code in ("R", "G1"):
        for w in (2, 6):
            j = cfg(code, 1, 0, name=f"probe_{code}_w{w}", step_limit=200, epochs=1, workers=w,
                    args=["--discard-ckpt"], score=(), valid="seaspoof_vi_train_n50")
            J.append(j)

    # Tầng 1: S0 (mốc chung của mọi bảng), R, G1, A0; xen kẽ theo seed để có seed 1 của mọi cấu hình sớm
    for m in S0_MODELS:   # mỗi backbone một lần chấm, để chia ra nhiều GPU
        J.append(zeroshot_job(f"S0_{m}", 1, [m], SCORE))
    for s in (1, 2, 3):
        for code in ("R", "G1", "A0"):
            J.append(cfg(code, s, 1))
    # Tầng 2: sửa GRPO, kiểm cơ chế, thành phần GRPO
    for s in (1, 2, 3):
        for code in ("G1s", "G1b"):
            J.append(cfg(code, s, 2))
    for s in (1, 2, 3):
        for code in ("G1lr", "G2", "G3"):
            J.append(cfg(code, s, 2))
    # Tầng 3: các yếu tố của GĐ5
    for s in (1, 2, 3):
        for code in ("P0", "B2", "N0", "B1"):
            J.append(cfg(code, s, 3))
    # Tầng 4: bản sửa ghép không RawBoost
    for s in (1, 2, 3):
        for code in ("G1sA", "G1bA"):
            J.append(cfg(code, s, 4))
    # Tầng 5: ít mẫu. Cùng STEPS bước và lịch E1' như bản đủ: dừng bằng optimizer_step_limit, số epoch
    # đủ lớn để giới hạn số bước là thứ dừng trước (dư 2 epoch; epoch dư dừng ngay ở batch đầu)
    for s in (1, 2, 3):
        for n, bpe in FEW_SHOT_BPE.items():
            ep = math.ceil(STEPS / bpe) + 2
            for code, method in (("SFT", "sft"), ("G1", "grpo_v1")):
                J.append(train_job(f"{code}{n}_s{s}", 5, method=method, seed=s, train=f"seaspoof_vi_train_{n}",
                                   epochs=ep, step_limit=STEPS, workers=workers))
    # Tầng 6: thêm seed; mốc XLS-R-2B (chỉ chấm)
    for s in (4, 5):
        for code in ("R", "G1"):
            J.append(cfg(code, s, 6))
    J.append(zeroshot_job("ref_xlsr_2b", 6, ["xlsr_2b"], ["radar_vi"]))

    names = [j["name"] for j in J]
    assert len(names) == len(set(names)), "trùng tên lần chạy"
    return J


def prefetch_list(jobs):
    """(mô hình, cách khởi tạo) cần tải trước, để các GPU không cùng tải một file."""
    out = set()
    for j in jobs:
        if j["kind"] == "train":
            out.add((j["model"], j["init"]))
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
    print("tải trước:", prefetch_list(jobs))
