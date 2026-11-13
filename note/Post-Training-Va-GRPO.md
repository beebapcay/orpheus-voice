# Post-training và fine-tune bằng học tăng cường (GRPO)

**Phạm vi:** quy trình huấn luyện nhiều giai đoạn cho bộ phát hiện dùng mô hình SSL, vấn đề "quên" khi fine-tune, và cách dùng học tăng cường GRPO thay cho fine-tune có giám sát. Bài nền: *Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection?* (Interspeech 2026). Mọi bảng số liệu là **EER (%), thấp hơn là tốt hơn**.

## 1. Ba giai đoạn huấn luyện

```mermaid
flowchart LR
    A["Pre-training<br/>~500K giờ tiếng thật<br/>tự giám sát, không nhãn"] --> B["Post-training<br/>~75K giờ thật + giả + vocoded<br/>có nhãn thật/giả"]
    B --> C["Fine-tune<br/>vài chục giờ<br/>dữ liệu của miền đích"]
```

| Giai đoạn | Học gì | Ai làm |
|---|---|---|
| Pre-training | Biểu diễn chung của tiếng nói (wav2vec 2.0, XLS-R, MMS) | Nhóm lớn có nhiều GPU |
| Post-training | Chỉnh mô hình SSL thành bộ phát hiện "đa năng" trên dữ liệu thật/giả rất lớn, nhiều miền | Nhóm phát hành mô hình (ví dụ AntiDeepfake) |
| Fine-tune | Thích nghi với một miền cụ thể (một ngôn ngữ, một nguồn dữ liệu) | Người dùng, với GPU nhỏ |

Quy trình này giống LLM: pre-training → mid-training → fine-tune.

## 2. Vấn đề: fine-tune làm mô hình "quên"

Fine-tune trên miền mới thì kết quả **trên miền đó tốt lên**, nhưng kết quả **trên các miền khác có thể xấu đi**. Hiện tượng này gọi là **catastrophic forgetting** (quên thảm khốc).

Ví dụ thật, với bản post-trained của XLS-R-2B fine-tune trên Deepfake-Eval-2024 (khoảng 50 giờ dữ liệu mạng xã hội năm 2024):

| | Trong miền (DFE24) | In-the-Wild (ngoài miền) |
|---|---|---|
| Chưa fine-tune | 22,64 | **1,24** |
| Fine-tune có giám sát (SFT) | 10,26 | **6,35** |

Lỗi trong miền giảm một nửa, nhưng lỗi trên In-the-Wild tăng gấp 5 lần.

## 3. SFT và học tăng cường khác nhau ở đâu

**SFT (fine-tune có giám sát):** đưa đáp án đúng, ép mô hình tăng xác suất của đáp án đó. Loss là $-\log p(\text{nhãn đúng})$. Câu nào mô hình đang sai càng nặng thì bị kéo càng mạnh.

**Học tăng cường (RL):** mô hình **tự thử trả lời**, được chấm điểm, rồi tăng xác suất của những câu trả lời được điểm cao và giảm những câu bị điểm thấp. Mô hình không bị "cầm tay" theo đáp án, mà học từ chính các lần thử của nó.

## 4. GRPO từng bước

Cho một câu âm thanh có nhãn đúng là **thật**. Mô hình hiện đang cho $p(\text{thật}) = 0{,}7$.

| Bước | Làm gì | Ví dụ |
|---|---|---|
| 1. Lấy mẫu | Rút ngẫu nhiên $G = 64$ câu trả lời theo xác suất của mô hình | Khoảng 45 lần "thật", 19 lần "giả" |
| 2. Chấm điểm | Phần thưởng $r = 1$ nếu trả lời đúng nhãn, $r = 0$ nếu sai | 45 mẫu được 1, 19 mẫu được 0 |
| 3. Chuẩn hoá trong nhóm | Lợi thế $A = (r - \bar r)/\sigma$, với $\bar r$, $\sigma$ là trung bình và độ lệch chuẩn của 64 phần thưởng | $\bar r = 0{,}7$, $\sigma \approx 0{,}46$. Mẫu đúng: $A \approx +0{,}65$. Mẫu sai: $A \approx -1{,}53$ |
| 4. Cập nhật | Tăng xác suất các câu trả lời có $A > 0$, giảm các câu có $A < 0$ | Đẩy $p(\text{thật})$ lên |
| 5. Giới hạn bước đi | Tỉ số $p_\text{mới}/p_\text{cũ}$ bị kẹp trong $[1-\epsilon, 1+\epsilon]$, với $\epsilon = 0{,}2$ | Mỗi bước không được đổi quá 20% so với mô hình cũ |
| 6. Phạt đi xa | Trừ $\beta \cdot \text{KL}$, đo độ lệch so với **mô hình tham chiếu** (bản post-trained, giữ băng), $\beta = 0{,}04$ | Kéo mô hình về gần điểm xuất phát |

Công thức tổng quát:

$$
\mathcal{L}_\text{RL}(\Theta) = -\mathbb{E}\left[ A \cdot \frac{p_\Theta(\tilde y \mid x)}{p_{\hat\Theta}(\tilde y \mid x)} - \beta \, \mathcal{L}_p(\dot\Theta, \Theta) \right]
$$

| Ký hiệu | Nghĩa |
|---|---|
| $\Theta$ | Mô hình đang học |
| $\hat\Theta$ | Mô hình "cũ", bản chụp từ trước, cập nhật thưa hơn |
| $\dot\Theta$ | Mô hình tham chiếu, giữ băng |
| $\tilde y$ | Câu trả lời được lấy mẫu |

GRPO cần giữ **ba bản mô hình** cùng lúc, nên tốn bộ nhớ hơn SFT. Hai bản phụ chỉ chạy suy luận, không cần gradient.

**Biến thể đơn giản (GRPOs):** bỏ mô hình cũ và bỏ bước kẹp. Kết quả gần như GRPO đầy đủ.

**Lưu ý khi đọc code:** công thức trong bài chuẩn hoá phần thưởng **trong nhóm 64 mẫu của từng câu**. Có những cài đặt chuẩn hoá **trên toàn bộ batch** (mọi câu, mọi mẫu). Hai cách cho lợi thế khác nhau, nên phải kiểm tra code đang dùng cách nào.

## 5. Kết quả

Fine-tune trên Deepfake-Eval-2024. Ngoài miền là trung bình của 4 tập ADD23, FoR, DEEP-VOICE, In-the-Wild. Mỗi số là trung bình 3 lần chạy.

| Mô hình | Cách fine-tune | Trong miền | Ngoài miền (TB) | In-the-Wild |
|---|---|---|---|---|
| XLS-R-2B post-trained | Không fine-tune | 22,64 | 2,69 | 1,24 |
| | SFT | 10,26 | 6,28 | 6,35 |
| | **GRPO** | **9,93** | **2,69** | 2,19 |
| | SFT rồi GRPO | 9,81 | 5,37 | 5,89 |
| XLS-R-2B **chỉ pre-train** | SFT | 10,45 | 19,34 | 13,49 |
| | GRPO | 11,15 | 21,86 | 19,00 |
| MMS-1B post-trained | Không / SFT / GRPO | 23,87 / 9,88 / 10,17 | 3,64 / 4,36 / 2,96 | 1,82 / 6,59 / 2,98 |
| MMS-300M post-trained | Không / SFT / GRPO | 29,23 / 12,23 / 11,94 | 3,68 / 7,74 / 3,78 | 2,91 / 14,17 / 5,91 |

**Bốn kết luận của bài:**
1. **GRPO giữ được ngoài miền, SFT thì không.** Trong miền hai cách ngang nhau (khoảng 10%).
2. **Chỉ dùng GRPO tốt hơn SFT rồi GRPO.** Đã SFT trước thì GRPO không cứu lại được.
3. **GRPO chỉ có tác dụng trên mô hình đã post-train.** Từ mô hình chỉ pre-train, GRPO còn kém SFT.
4. **Bỏ kẹp và mô hình cũ không ảnh hưởng nhiều** (GRPOs gần bằng GRPO).

## 6. Ablation (chỉ làm với XLS-R-2B)

| Biến thể | Trong miền | Ngoài miền (TB) | Ý nghĩa |
|---|---|---|---|
| GRPO mặc định ($\beta = 0{,}04$) | 9,93 | 2,69 | |
| Bỏ phạt đi xa ($\beta = 0$) | 10,21 | 3,53 | DEEP-VOICE tăng lên 6,50%. Phạt đi xa **có** vai trò, khác với kết luận ở LLM |
| Phạt rất mạnh ($\beta = 1$) | 15,83 | **2,04** | Ngoài miền tốt nhất nhưng trong miền kém hẳn: mô hình gần như không học miền mới |
| Bỏ phần thưởng âm | 11,41 | 3,44 | Kém hơn ở mọi tập. Bài cho rằng **phần thưởng âm** là khác biệt then chốt giữa GRPO và SFT |

"Bỏ phần thưởng âm" nghĩa là $A = r$: mẫu đúng được 1, mẫu sai được 0 thay vì bị phạt. Biến thể này về bản chất gần với SFT cộng thêm phạt đi xa.

**Phân tích độ trôi:** bài đo khoảng cách Wasserstein giữa phân bố embedding của tập kiểm tra và tập tham chiếu. SFT kéo các tập ngoài miền (DEEP-VOICE, In-the-Wild) **ra xa**, còn GRPO thì không.

## 7. Đọc kỹ các con số

**"GRPO cải thiện ngoài miền" là so với SFT, không phải so với không fine-tune.** So với mô hình chưa fine-tune:
- XLS-R-2B: trung bình ngoài miền bằng nhau (2,69 và 2,69). Xét từng tập, GRPO kém hơn ở ADD23 (5,34 so với 4,67), DEEP-VOICE (2,76 so với 2,23), In-the-Wild (2,19 so với 1,24), và chỉ tốt hơn ở FoR.
- MMS-300M: hơi kém hơn (3,78 so với 3,68). In-the-Wild vẫn tăng từ 2,91% lên 5,91%.

Nói chính xác: GRPO **giảm mức quên**, chưa **loại bỏ** được việc quên.

**$\beta = 1$ là một sự đánh đổi, không chỉ là "underfit".** Ngoài miền tốt nhất trong bảng (2,04), đổi lại trong miền kém (15,83). Chọn $\beta$ là chọn điểm cân bằng giữa học cái mới và giữ cái cũ.

**Phạm vi hẹp:** chỉ một bộ dữ liệu fine-tune, ablation chỉ trên XLS-R-2B, cơ chế để ngỏ. Thí nghiệm chạy trên GPU H100.

## 8. Phân tích thêm: vì sao GRPO có thể ít làm quên hơn

**Phần này là phân tích của note, không có trong bài.** Giả sử chuẩn hoá theo nhóm như công thức (4), lấy kỳ vọng theo phép lấy mẫu, và bỏ qua kẹp và phạt KL. Gọi $p$ là xác suất mô hình gán cho nhãn đúng, $z$ là logit.

| | Lực kéo lên logit của nhãn đúng |
|---|---|
| SFT | $(1 - p)$ |
| GRPO | $\sqrt{p(1-p)}$ |

| $p$ (mô hình đang...) | SFT | GRPO |
|---|---|---|
| 0,01 (sai rất tự tin) | **0,99** | 0,10 |
| 0,5 (phân vân) | 0,50 | 0,50 |
| 0,99 (đúng rất tự tin) | 0,01 | 0,10 |

SFT kéo mạnh nhất vào những câu mô hình **đang sai nặng**. Đó thường là những câu đặc thù của miền mới, nên mô hình bị đổi nhiều để chiều theo chúng. GRPO kéo đều hơn và nhẹ tay với những câu sai nặng. Đây là một giả thuyết hợp lý cho việc GRPO giữ kiến thức cũ tốt hơn, nhưng **chưa được kiểm chứng**.

Phân tích này cũng gợi ý vì sao GRPO kém trên mô hình chỉ pre-train: khi mô hình ban đầu đoán gần như ngẫu nhiên, các lần thử ít mang thông tin, nên tín hiệu học yếu.

## 9. Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Giữ khả năng ngoài miền tốt hơn SFT, trong miền ngang SFT | Chưa giữ được hoàn toàn: vẫn có tập kém hơn so với không fine-tune |
| Không cần dữ liệu mới, chỉ đổi hàm loss | Tốn bộ nhớ hơn: 3 bản mô hình, 64 mẫu mỗi câu |
| Phần thưởng đơn giản (đúng = 1, sai = 0) | Chỉ có tác dụng khi điểm xuất phát đã tốt (đã post-train) |
| Có ablation tách vai trò của phạt KL và phần thưởng âm | Ablation chỉ trên một backbone, một bộ dữ liệu; cơ chế chưa giải thích |

## 10. Câu hỏi tự kiểm tra

1. Post-training nằm ở đâu trong quy trình, và khác fine-tune ở điểm nào?
2. "Quên" khi fine-tune là gì? Lấy một ví dụ bằng số.
3. Kể 6 bước của GRPO. Bước nào giới hạn mức thay đổi mỗi lần, bước nào kéo mô hình về điểm xuất phát?
4. Với $p(\text{nhãn đúng}) = 0{,}7$, lợi thế của mẫu đúng và mẫu sai là bao nhiêu?
5. Vì sao nói "GRPO cải thiện ngoài miền" cần nói rõ là so với cái gì?
6. $\beta = 0$ và $\beta = 1$ cho kết quả thế nào? Rút ra điều gì?
