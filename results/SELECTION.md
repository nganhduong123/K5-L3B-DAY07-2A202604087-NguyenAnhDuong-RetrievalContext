# Chốt cấu hình trước khi chạy test

Ghi lúc 2026-11-16, **trước** lần chạy test đầu tiên (chưa chạy retrieval hay agent trên split `test`).

## Tiêu chí chọn (theo thứ tự ưu tiên, chỉ dùng dev)

1. Chất lượng câu trả lời trên dev: số câu có đáp án được trả lời có citation trong context (`answered_answerable`), sau đó kiểm tra thủ công.
2. Abstention trên câu thiếu evidence (không được kém baseline).
3. Context Evidence Rate của context cuối gửi model.
4. Hòa thì chọn cấu hình có median latency và chi phí thấp hơn.

## Kết quả dev dùng để chọn (agent thật, openai/gpt-4o-mini trừ khi ghi khác)

| Config | Ctx Evidence | Trả lời câu có đáp án | Abstain đúng | Median latency | Chi phí |
|---|---|---|---|---|---|
| agent_dense_b800 (baseline) | 8/20 | 10/20 | 3/3 | 4.20 s | $0.00588 |
| agent_dense_b2400 | 15/20 | 18/20 | 3/3 | 3.12 s | $0.00584 |
| agent_rerank_b2400 | 19/20 | 20/20 | 2/3 | 3.74 s | $0.00499 |
| agent_rerank_b2400_v2 | 19/20 | 20/20 | 2/3 | 3.58 s | $0.00591 |
| **agent_rerank_b2400_v2_seed** | **20/20** | **20/20** | 2/3 | **2.41 s** | **$0.00413** |
| agent_rerank_b2400_v2_seed_41mini (gpt-4.1-mini) | 20/20 | 18/20 | 2/3 | 2.26 s | $0.01059 |

## Quyết định

- **Cấu hình chọn:** `agent_rerank_b2400_v2_seed` — hybrid RRF (dense Day 6 + BM25, 20 candidate mỗi bên, k=60) → cross-encoder `ms-marco-MiniLM-L-6-v2` → top-3 mỗi lượt; context budget tích lũy 2400 ký tự, dedup; tối đa 3 lượt search; lượt đầu dùng nguyên câu hỏi; prompt v2; `gpt-4o-mini`, temperature 0, max_tokens 800.
- **Baseline so sánh trên test:** `agent_dense_b800` (dense Day 6, top-3, budget 800, 4 lượt, prompt v1).
- Đánh đổi chấp nhận: abstain 2/3 (kém baseline 3/3 do u02); baseline đạt 3/3 chủ yếu vì nó từ chối gần như mọi thứ (10/20 câu có đáp án bị từ chối). Rerank thêm ~0.5 s/lượt CPU nhưng tổng latency vẫn thấp hơn nhờ ít lượt search hơn.
- Sau khi chạy test sẽ **không** chỉnh thêm cấu hình. Nếu cần chỉnh, phải tạo test mới.
