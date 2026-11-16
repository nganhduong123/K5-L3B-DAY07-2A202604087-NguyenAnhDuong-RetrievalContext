# Báo cáo Day 7 — Nguyễn Anh Dương (2A202604087)

> Code và kết quả được làm cùng trợ lý AI (Claude Code). Mọi số liệu dưới đây lấy từ file kết quả trong `results/`. Nhãn chấm câu trả lời và context được tạo bằng cách đọc từng câu trả lời và từng chunk được cite (không dùng verifier model); người học đã/cần đọc lại để xác nhận.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** *Writing Guide with Handbook* (OpenStax, Rice University), giấy phép **CC BY-NC-SA 4.0** (ghi ở trang 4 PDF), https://openstax.org/details/books/writing-guide. Phạm vi giống Day 6: Chương 1 (PDF tr. 23–50) và Chương 2 (PDF tr. 53–75) = **51 trang nội dung**, 156.889 ký tự. Dùng index **strategy B** của Day 6 (chunk theo đoạn, tối đa 800 ký tự) = **263 chunk** (median 639 ký tự). Chunk được xuất ra `data/corpus/chunks.json` bằng `python -m src.export_day6` (chia sẻ được theo CC BY-NC-SA, ghi nguồn ở trên).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker, repo Day 6, bảng `chunks`, `strategy='B'`), cosine distance, exact search. Embedding `intfloat/multilingual-e5-small`, **384 chiều**, tiền tố `query: ` / `passage: `, chuẩn hóa L2, sentence-transformers 6.3.0 / torch 2.15.0. Query embedding Day 7 dùng đúng model và tiền tố này; code kiểm tra số chiều = 384 trước khi truy vấn. Không tạo index mới. Người chấm không có DB Day 6 có thể dựng lại index từ `data/corpus/chunks.json` bằng `python -m src.build_index` (cùng model/prefix/schema; đã kiểm chứng top-3 dense trùng 36/36 câu với index Day 6 gốc). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU).
- **Agent model, cách chạy và phân chia dev/test:** `openai/gpt-4o-mini` qua OpenRouter, temperature 0, max_tokens 800, tool calling (`search_documents`). Thử thêm `openai/gpt-4.1-mini` cho 1 cấu hình dev. Python 3.13.0. Lệnh chạy: [src/README.md](../../src/README.md).
  - **Dev (23 câu):** 20 câu có đáp án = toàn bộ 20 câu Day 6 (q01–q20) + 3 câu thiếu evidence (u01–u03). Câu test Day 6 (q16–q20) đã bị xem kết quả ở Day 6 nên được chuyển vào dev, không dùng làm test.
  - **Test (13 câu):** 10 câu có đáp án **mới** (t01–t10) viết từ các trang chưa có câu Day 6 nào (p26, 30, 40, 41, 42, 46, 56) + 3 câu thiếu evidence (u04–u06). Test được viết trước mọi lần chạy Day 7 và chỉ chạy **một lần** sau khi chốt cấu hình ([results/SELECTION.md](../../results/SELECTION.md)).
  - Loại câu: direct, paraphrase, condition, metadata (4 câu có filter `chapter` do người dùng chọn, ví dụ "In chapter 1, …").
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án có `gold = {source_id, page, evidence}`; `build_benchmark.py` dùng hàm `validate` của starter để bảo đảm evidence nằm trọn trong một chunk đúng nguồn và trang. Câu thiếu evidence là câu **gần chủ đề**, kiểm tra bằng grep toàn bộ 263 chunk (không chỉ top-k); lần kiểm tra đầu phát hiện "prize" có ở p57, đọc lại thì đó là giải của **một cuốn sách khác** của Cathy Park Hong → giữ u03 làm câu bẫy và ghi lý do trong `cases.json`. Hit = chunk đúng nguồn + trang và chứa nguyên văn evidence (định nghĩa của starter). Câu thiếu evidence không vào mẫu số retrieval metrics. Gold không bao giờ được đưa vào agent hay retriever; chỉ dùng sau khi agent chạy xong để tính Context Evidence Rate.

## Retrieval và thử nghiệm

Bốn phương pháp (đều trên cùng 263 chunk, cùng câu hỏi gốc, cùng nhãn):

1. **Dense** — E5 query embedding → pgvector Day 6, filter `chapter` bằng SQL tham số hóa.
2. **BM25** — `rank_bm25` BM25Okapi, tách từ `\w+` + bỏ stopword tiếng Anh, filter trong Python.
3. **Hybrid RRF** — dense top-20 + BM25 top-20, hợp nhất bằng Reciprocal Rank Fusion `Σ 1/(60 + rank)`. Dùng rank thay vì cộng score vì cosine và BM25 khác thang đo.
4. **Hybrid + rerank** — 20 candidate từ hybrid, chấm lại bằng cross-encoder, lấy top-k.

Context budget đo bằng **ký tự** (gồm header `[id | source | page]`), ≈ ký tự/4 token. Hit@1/Hit@3 luôn tính trên ranking top-3. Retrieval-only không gọi LLM nên chi phí $0; thời gian là median ms/query trên CPU (laptop), không tính lần tải model.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense Day 6, top-3, filter chapter nếu có, budget 800 ký tự (mặc định starter) | — | dev | 13/20 (65%) | 19/20 (95%) | 12/20 (60%) | 32 ms / $0 |
| Phương pháp 2 | BM25, top-3, 800 | Lexical bắt được từ khóa hiếm (tên, năm, thuật ngữ) | dev | 16/20 (80%) | 19/20 (95%) | 12/20 (60%) | 0.7 ms / $0 |
| Phương pháp 3 | Hybrid RRF (20+20, k=60), top-3, 800 | Kết hợp ngữ nghĩa + từ khóa sẽ ổn định hơn từng cái | dev | 14/20 (70%) | 19/20 (95%) | 11/20 (55%) | 29 ms / $0 |
| Cải thiện 1 (= phương pháp 4) | Hybrid RRF → cross-encoder rerank, top-3, 800 | Cross-encoder đọc cặp (câu hỏi, đoạn) nên xếp hạng top-1 tốt hơn | dev | **18/20 (90%)** | **20/20 (100%)** | 13/20 (65%) | 525 ms / $0 |
| Cải thiện 2 | Dense, top-3, **budget 2400** | Chunk median 639 ký tự + header → budget 800 chỉ chứa ~1 chunk; tăng budget để top-3 vào context | dev | 13/20 (65%) | 19/20 (95%) | 19/20 (95%) | 29 ms / $0 |
| Cải thiện 1+2 | Hybrid RRF + rerank, top-3, budget 2400 | Kết hợp hai cải thiện | dev | 18/20 (90%) | 20/20 (100%) | **20/20 (100%)** | 581 ms / $0 |
| (thêm) | Hybrid RRF, top-3, 2400 | | dev | 14/20 (70%) | 19/20 (95%) | 19/20 (95%) | 28 ms / $0 |
| (thêm) | Hybrid + rerank, top-5, 4000 | Context rộng hơn có thêm lợi ích? | dev | 18/20 (90%) | 20/20 (100%) | 20/20 (100%) | 520 ms / $0 — không tăng, gấp 1.7× ký tự → không chọn |
| **Cấu hình đã chọn** | Hybrid RRF + rerank, top-3, 2400 | Chốt trên dev | test | **8/10 (80%)** | **10/10 (100%)** | **10/10 (100%)** | 510 ms / $0 |
| Baseline | Dense, top-3, 800 | | test | 7/10 (70%) | 10/10 (100%) | 4/10 (40%) | 26 ms / $0 |
| (tham khảo) | BM25, top-3, 800 | | test | 6/10 (60%) | 7/10 (70%) | 3/10 (30%) | 0.5 ms / $0 |
| (tham khảo) | Hybrid RRF, top-3, 800 | | test | 7/10 (70%) | 8/10 (80%) | 4/10 (40%) | 25 ms / $0 |

**Nhận xét retrieval.** (1) Trên dev BM25 có Hit@1 cao nhất trong 3 phương pháp không rerank, nhưng trên test lại kém nhất (Hit@3 7/10, trượt t02, t08, t10; chunk gold vẫn ở rank 6, 7, 5 trong top-10). Nguyên nhân khi xem điểm: ở t02/t08 các từ chung của câu hỏi ("key terms", "list", "define/defined") khớp mạnh với chunk khác (ví dụ đoạn mở đầu danh sách thuật ngữ p30:B:1) hơn là từ đặc thù ("authority", "meme"); ở t10 chunk gold không chứa "verdict"/"jury" trong khi p41:B:1 chứa cả hai → BM25 nhạy với cách đặt câu hỏi. (2) RRF không tốt hơn dense: khi BM25 xếp sai, RRF kéo theo (dev q08, test t08, t10 trượt top-3 dù dense tìm được). (3) Rerank là cải thiện retrieval duy nhất ổn định trên cả hai split, đổi lại ~0.5 s CPU/lượt. (4) **Nút thắt lớn nhất là context, không phải retrieval:** dense có Hit@3 95–100% nhưng Context Evidence Rate với budget 800 chỉ 60% (dev) và 40% (test); mọi phương pháp ở budget 800 đều chỉ đạt 30–65%.

**Hai lỗi dev, nguyên nhân, cách thử:**

1. **q03 (baseline agent, budget 800) – từ chối dù retriever tìm đúng.** Trace: cả 4 lượt search đều trả `p25:B:3` (chunk gold) nhưng `skipped_budget` vì chunk + header > 800 ký tự; context cuối chỉ có `p66:B:3` không liên quan → agent trả `INSUFFICIENT_EVIDENCE`. Đây là lỗi chọn context. Thử: budget 2400 (cải thiện 2). Kết quả agent dev: Context Evidence 8/20 → 15/20, trả lời 10/20 → 18/20, q03 đúng. **Lưu ý gây nhiễu:** cấu hình agent này đổi đồng thời max_searches 4 → 3 (để giới hạn chi phí khi budget lớn hơn), nên mức tăng ở tầng agent không thể quy hết cho budget. Bằng chứng tách riêng: (a) ở tầng retrieval, chỉ đổi budget (cùng top-3, một lượt) đã tăng Context Evidence 12/20 → 19/20; (b) trong trace baseline dev, 8/10 câu từ chối nhầm (q03, q06, q07, q10, q11, q14, q17, q18) có chunk gold được retriever trả về nhưng bị `skipped_budget`; 2 câu còn lại (q19, q20) không bao giờ lấy được chunk gold (q20 chạm giới hạn 4 lượt). Giới hạn 3 lượt có thể đã làm hại 2 câu chạm giới hạn ở `agent_dense_b2400` (q07, q11); chưa chạy cấu hình chỉ đổi budget giữ 4 lượt.
2. **q19 (agent rerank) – đáp án đúng nhưng citation không hỗ trợ.** Agent tự viết lại query thành "writing inclusively impact of language good intentions" → không lấy được `p60:B:4` (chunk gold); agent vẫn trả lời "impact matters more" và cite `p60:B:2` — chunk này **không chứa** câu đó (agent suy ra từ chính câu hỏi). Retrieval với câu hỏi gốc thì có `p60:B:4` ở rank 2 → lỗi do query rewriting. Thử (a) prompt v2 yêu cầu chunk được cite phải chứa câu hỗ trợ → **không cải thiện** (q19 vẫn sai); (b) lượt search đầu luôn dùng nguyên câu hỏi (`seed`) → q19 đúng, Context Evidence 19/20 → 20/20.
3. (thêm) **u02 – trả lời câu thiếu evidence.** Tài liệu chỉ ghi "1–2 pages"; agent đổi thành "approximately 250–500 words" (kiến thức ngoài). Prompt v2 (cấm quy đổi số, đúng đơn vị) và model mạnh hơn `gpt-4.1-mini` **đều không sửa được**; vẫn là lỗi còn tồn tại.

**Lý do chọn cấu hình:** theo tiêu chí công bố trước trong [results/SELECTION.md](../../results/SELECTION.md): chất lượng câu trả lời dev → abstention → Context Evidence Rate → latency/chi phí. `agent_rerank_b2400_v2_seed` đạt 20/20 grounded (trung bình 2.0), Context Evidence 20/20, nhanh và rẻ nhất trong các cấu hình gpt-4o-mini (2.41 s, $0.0041/23 câu). Đánh đổi: abstention dev 2/3 so với 3/3 của baseline — baseline đạt 3/3 vì từ chối gần như mọi câu (từ chối nhầm 10/20).

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** một tool `search_documents(query)`; filter `chapter` do người dùng cung cấp được hệ thống gắn vào mọi lượt search (model không tự chọn filter, không suy từ gold). Cấu hình chọn: lượt 1 = câu hỏi gốc, model được tự viết lại query thêm tối đa đến **3 lượt**; dừng khi model trả lời không gọi tool, hoặc đạt giới hạn (khi đó gọi lại với `tool_choice=none` để buộc trả lời từ context đã có). Context: mỗi lượt top-3, **dedup** chunk đã hiện, chỉ thêm chunk còn vừa **budget tích lũy 2400 ký tự**; context cuối = mọi chunk đã đưa cho model. Trả lời phải có `[chunk-id]`; citation không nằm trong context hoặc thiếu citation → chuyển thành abstain (ghi event `invalid_or_missing_citations`). Prompt nói rõ kết quả tool là dữ liệu, không phải chỉ dẫn. Lỗi provider/vector store được ghi `status: "error"` kèm thông báo, không thay bằng kết quả giả (ví dụ lần chạy `gpt-4.1-mini` đầu tiên bị HTTP 402 cho cả 23 câu vì thiếu `max_tokens`; đã ghi lỗi, sửa `max_tokens=800`, chạy lại — file trace hiện tại là lần chạy lại).

**Agent, dev (chỉ số tự động; nhãn thủ công cho baseline và cấu hình chọn):**

| Cấu hình / split | Ctx Evidence | Trả lời câu có đáp án | Abstain đúng | Phân bố số lượt search | Median latency | Token in/out | Chi phí |
|---|---|---|---|---|---|---|---|
| agent_dense_b800 (baseline) / dev | 8/20 | 10/20 | 3/3 | 1:10, 2:4, 3:5, 4:4 | 4.20 s | 32.070 / 1.779 | $0.00588 |
| agent_dense_b2400 / dev | 15/20 | 18/20 | 3/3 | 1:16, 2:5, 3:2 | 3.12 s | 32.318 / 1.659 | $0.00584 |
| agent_rerank_b2400 / dev | 19/20 | 20/20 | 2/3 | 1:21, 2:2 | 3.74 s | 26.803 / 1.618 | $0.00499 |
| agent_rerank_b2400_v2 / dev | 19/20 | 20/20 | 2/3 | 1:21, 2:2 | 3.58 s | 33.005 / 1.599 | $0.00591 |
| **agent_rerank_b2400_v2_seed / dev** | **20/20** | **20/20** | 2/3 | 1:21, 2:2 | **2.41 s** | 22.994 / 1.135 | **$0.00413** |
| agent_rerank_b2400_v2_seed, gpt-4.1-mini / dev | 20/20 | 18/20 (q07, q13 thiếu `[id]` → abstain) | 2/3 | 1:23 | 2.26 s | 20.950 / 1.382 | $0.01059 |

**Nhãn thủ công (tổng hợp bằng `starter/agent_metrics.py`):**

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline / dev | 23/23 | 1:10, 2:4, 3:5, 4:4 | 10 / 0 / 10 — 1.00 | 10/10 | 10/10 | 3/3 | 10/20 |
| Chọn / dev | 23/23 | 1:21, 2:2 | 0 / 0 / 20 — **2.00** | 21/22 | 21/21 | 2/3 | 0/20 |
| Baseline / test | 13/13 | 1:3, 2:6, 3:2, 4:2 | 7 / 0 / 3 — 0.60 | 3/3 | 3/3 | 3/3 | 7/10 |
| **Chọn / test** | 13/13 | 1:10, 2:3 | 0 / 0 / 10 — **2.00** | 10/10 | 10/10 | **3/3** | **0/10** |

Citation không hợp lệ duy nhất (dev, cấu hình chọn): u02, `p35:B:1` chỉ hỗ trợ "1–2 pages", không hỗ trợ "250–500 words".

**Chấm context test (đọc thủ công cả 10 câu có đáp án; 0 = không có evidence, 1 = có nhưng thiếu điều kiện/nguồn, 2 = đủ):**

| Cấu hình | Mức 2 | Mức 1 | Mức 0 |
|---|---|---|---|
| Baseline (dense, 800) | 3 (t01, t03, t08) | 2 (t06: có `p56:B:1` nêu Michelle Obama nhưng không có tên sách; t10: `p41:B:1` nhắc "the verdict" nhưng không nêu "not guilty") | 5 (t02, t04, t05, t07, t09) |
| Chọn (rerank, 2400, seed) | 10 | 0 | 0 |

Ở t02, t04, t05, t07, t10 baseline **có** tìm thấy chunk gold nhưng bỏ vì vượt budget (`skipped_budget` trong trace) — cùng cơ chế với lỗi q03 trên dev.

**Ví dụ trace (test, cấu hình chọn trừ khi ghi khác):**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t07 | có đáp án | search("At approximately what mental age did Goddard estimate Jean Gianini?") → p42:B:0, p41:B:0, p40:B:6 | "Goddard estimated Jean Gianini's mentality at approximately ten years of age [p42:B:0]" | 2 — chunk chứa nguyên văn "estimated his mentality at approximately ten years of age" |
| t10 | có đáp án (paraphrase) | search(câu gốc) → p41:B:1, p40:B:6, p41:B:0 | "…a verdict of "not guilty as charged" due to claims that Gianini was legally insane… [p41:B:0]" | 2 — verdict đúng, có trong p41:B:0; "due to" là suy luận nhân quả nhẹ (chunk nói luật sư biện hộ cho rằng bị cáo mất trí) |
| t06 | có đáp án (paraphrase) | search(câu gốc) → p56:B:2, p56:B:1, p57:B:2 | "…in her book "Becoming" (2018) [p56:B:2]" | 2 — tên sách + năm có trong p56:B:2; tên Michelle Obama nằm ở p56:B:1 (không được cite, câu hỏi không hỏi tên) |
| u05 | thiếu evidence | 2 lượt: câu gốc → p29:B:2 (Dawkins, meme), rồi query viết lại → không có "Selfish Gene" | "INSUFFICIENT_EVIDENCE: The publication year … is not mentioned…" | Abstain đúng — model không dùng kiến thức ngoài (1976) |
| u04 | thiếu evidence | 2 lượt, context có p40–41 (vụ Gianini, verdict) | "INSUFFICIENT_EVIDENCE: … institution … after his trial" | Abstain đúng — không nhầm verdict thành nơi giam giữ |
| t02 (baseline) | có đáp án | 4 lượt; `p30:B:2` (gold) bị `skipped_budget`, context chỉ có p24:B:1 | "INSUFFICIENT_EVIDENCE…" | 0 — từ chối nhầm do budget 800 |
| u02 (dev) | thiếu evidence | search(câu gốc) → p35:B:0, p23:B:0, p35:B:1 | "…1–2 pages, which typically translates to approximately 250–500 words… [p35:B:1]" | Sai — lẽ ra phải từ chối; quy đổi bằng kiến thức ngoài; citation không hỗ trợ |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Retrieval: `results/retrieval-{dev,test}/<config>.traces.json` và `summary.json`.
  - Agent: `results/agent-{dev,test}/<config>/traces.json` (mỗi tool call: query, filter, kết quả + rank + score, chunk được thêm/bỏ vì trùng/vì budget; context cuối; câu trả lời; citation; token; chi phí; thời gian; lỗi), `summary.json`; nhãn `reviews.json` (kèm lý do từng câu) và `agent_metrics.json`.
  - Chốt cấu hình: `results/SELECTION.md`. Benchmark: `data/benchmark/questions.json`, `data/benchmark/cases.json`.
  - Lệnh: [src/README.md](../../src/README.md). Tổng chi phí LLM toàn bộ thí nghiệm (theo `usage.cost` của OpenRouter): dev $0.0373, test $0.0059 ≈ **$0.043**. Retrieval, embedding và rerank chạy local ($0).
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - Test (chỉ chạy một lần sau khi chốt): grounded answer trung bình **0.60 → 2.00**, từ chối nhầm 7/10 → 0/10, Context Evidence 3/10 → 10/10, abstention giữ 3/3, median latency 4.9 s → 2.3 s, chi phí $0.0034 → $0.0025 cho 13 câu. Cấu hình chọn rẻ và nhanh hơn dù có rerank, vì context đủ ngay từ lượt đầu nên agent ít phải tìm lại.
  - Không phải cải thiện nào cũng có tác dụng: RRF không hơn dense; prompt v2 không đổi kết quả; `gpt-4.1-mini` tốn gấp 2.5 lần mà có 2 lỗi định dạng citation và vẫn sai u02; top-5/budget 4000 không tăng chỉ số nhưng tốn gấp 1.7 lần token.
  - **Giới hạn:** (1) Benchmark nhỏ (20 dev, 10 test có đáp án; 3+3 câu thiếu evidence) nên 1 câu = 5–10 điểm %; test 100% không có nghĩa hệ thống không lỗi — u02 trên dev cho thấy agent vẫn có thể bịa số khi tài liệu có thông tin gần giống. (2) Câu hỏi đều một-hop, evidence trong một chunk; chưa đánh giá multi-hop. (3) Định nghĩa hit theo nguyên văn bỏ sót evidence thay thế (baseline dev q01, q09 cite chunk không phải gold nhưng vẫn hỗ trợ đúng — chấm thủ công là 2). (4) Câu hỏi test do cùng người làm hệ thống viết, có thể gần cách diễn đạt của tài liệu. (5) Nhãn do một người chấm (có AI hỗ trợ đọc), chưa có người chấm thứ hai. (6) LLM qua API có thể cho kết quả hơi khác giữa các lần chạy dù temperature 0.

## Thử thách nâng cao (nếu thực hiện)

> **Kết quả: 27/30 = 90.0% trên test** — đạt đúng ngưỡng, **không có biên dư**. Câu at17 là ca giáp ranh (xem bên dưới); nếu người chấm đánh trượt at17, kết quả là 26/30 (86.7%) và không đạt ngưỡng. Nhãn do người làm bài chấm (có AI hỗ trợ đọc) bằng cách đối chiếu trang/ảnh gốc; cần người chấm xác nhận.

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:** *Business Ethics*, OpenStax (Rice University), Byars & Stanberry; PDF bản web, metadata `CreationDate 2026-04-20`, **355 trang**, SHA-256 `9d0a7b3b…5034c01`. Giấy phép **CC BY-NC-SA 4.0** (trang 4 PDF). Tải lại tại https://openstax.org/details/books/business-ethics (file gốc 51 MB không commit, đặt vào `data/advanced/source/business-ethics.pdf`). Crop ảnh (`data/advanced/crops/`, 12 MB) được chia sẻ theo cùng giấy phép, ghi nguồn OpenStax; một số ảnh có giấy phép riêng ghi trong chú thích.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý** (`src/advanced/`):
  - `parse.py` (PyMuPDF 1.26.5): index **trang 17–352 (336 trang)** = toàn bộ 11 chương + phụ lục A–C + đáp án; bỏ tr. 1–16 (bìa, bản quyền, mục lục, lời nói đầu) và tr. 353–355 (bảng tra từ). Ra **1.650 đơn vị**: 1.547 đoạn text (≤ 900 ký tự/đơn vị, bỏ header/footer và chú thích để không index trùng), **4 bảng** (`find_tables` → markdown + tiêu đề + chú thích "Table x.y"), **99 hình** (mọi ảnh raster của trang nội dung, gộp theo chú thích "Figure x.y" in ngay dưới ảnh; crop JPEG 200 dpi từ vùng trang). Mỗi đơn vị giữ trang, chương, mục, bbox (điểm PDF) → citation chỉ ra trang + vùng.
  - Lỗi parse tìm thấy khi kiểm tra và đã sửa: chú thích phụ lục dạng "Figure A1" không khớp regex; sơ đồ dạng dải mảnh p323 (1100×28 px) bị lọc nhầm là icon → giờ index cả ảnh không chú thích (ngữ cảnh = đoạn văn ngay trên). Quét vector drawing: không có sơ đồ vector nào ngoài các khung chữ.
  - **Chưa xử lý:** Table 11.1 (p322) là bảng bài tập để trống, không có dữ liệu → không index. Biểu đồ là ảnh raster không có số liệu nhúng → phụ thuộc vision model.
  - `describe.py`: mỗi hình được vision model mô tả một lần (cache JSON): loại hình, chữ trong hình, trục/đơn vị/chú giải, giá trị từng chuỗi (số đọc từ trục ghi "≈"), quan hệ trong sơ đồ. 45 hình bằng `gemini-3.5-flash`, 54 hình bằng `gemini-3.5-flash-lite` (free tier chỉ cho 20 request/ngày với flash — xem Lỗi hệ thống). 3 hình (4.5, 6.11, 7.4) bị bộ lọc **RECITATION** của Gemini chặn khi yêu cầu chép nguyên văn → prompt dự phòng (diễn đạt lại câu dài, giữ nguyên nhãn và số); ghi `prompt_variant` trong từng file.
  - `index.py`: text + bảng + (chú thích + mô tả hình) → E5-small 384d trên pgvector (`strategy='ADV'`, 1.650 dòng, 123.5 s embed CPU) và BM25.
  - **Kiểm tra mô tả với bản gốc** (17 hình dùng trong benchmark, so từng giá trị với crop): 16/17 đúng (ví dụ Fig 4.5 đúng cả 25 ô; Fig 6.6, 8.2, 8.3, 8.4, 10.2 đúng mọi nhãn; Fig 6.10 đúng 28 bang). **Fig 6.9 sai**: mô tả xếp WA vào $9.50–10.24 (đúng là $10.25–11.50), xếp nhầm ME, WV, HI và còn lẫn lời "suy nghĩ" của model. Không sửa riêng vì là hình của test; đây là lý do agent có `view_figure`.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - `data/advanced/benchmark.json` (`src/advanced/benchmark.py`), viết từ **trang/crop gốc** trước mọi lần chạy retrieval/agent và không dựa vào mô tả do model tạo. **Dev 18** (bảng 3, ảnh 3, biểu đồ 4, multi-hop 2, text 3, thiếu evidence 3); **test 30** (bảng 6, ảnh 5, biểu đồ 5, multi-hop 4, text 4, thiếu evidence 6). Dev và test dùng ô/hình/đoạn khác nhau. Fig 4.5 là bảng dạng ảnh nên tính vào nhóm Bảng.
  - Schema mỗi câu có đáp án: `id, split, group, question, expected_answer, evidence[]` với mỗi evidence `{unit, page, kind, locator (hàng/cột, vùng hình, giá trị đọc được), check, crop}`; multi-hop liệt kê **mọi** đơn vị bắt buộc. Script kiểm tra chuỗi `check` có trong đơn vị (bảng/text). Câu thiếu evidence: `absence_reason` + grep từ khóa toàn bộ 1.650 đơn vị, thêm kiểm tra đồng xuất hiện (ví dụ "Japan"+"telework" chỉ trúng p302 về *teletherapy*; "Google"+"Black" trúng p226 nhưng đó là 2.4% tổng nhân viên, không phải lãnh đạo — bẫy có chủ đích).
  - Quy tắc chấm (ghi trong benchmark trước khi chạy): đạt khi đúng + đủ (đơn vị/phép tính), **cite mọi đơn vị evidence bắt buộc** và nguồn được cite hỗ trợ toàn bộ câu trả lời; câu thiếu evidence đạt khi từ chối và không thêm khẳng định sai.
  - Agent (`agent.py`): `gemini-3.5-flash-lite`, temperature 0, 2 tool — `search_document` (hybrid RRF dense+BM25 → cross-encoder, top-5, budget tích lũy 9.000 ký tự, dedup) và `view_figure(unit_id, question)` gửi **crop gốc** của hình đã có trong context cho vision model; tối đa 4 search + 3 view; lượt đầu dùng câu hỏi gốc. Trả lời phải cite `[unit-id]` có trong context, nếu không → từ chối.
  - Dev (chấm thủ công, cùng quy tắc): `adv_dense_noview` 17/18 (ad11 thêm "clients" không có trong Fig 3.4), `adv_rerank_noview` 17/18 (ad08 không cite Fig 8.2 bắt buộc — ban đầu chấm đạt vì cite đoạn text cùng số liệu, đã chấm lại cho nhất quán sau khi chấm test; không đổi lựa chọn), **`adv_rerank_view` 18/18** → chốt ([results/advanced/SELECTION.md](../../results/advanced/SELECTION.md), ghi trước khi chạy test). Lỗi tìm thấy trên dev và đã sửa: parser citation chỉ nhận một id mỗi ngoặc (model viết `[id1, id2]`) → sửa và chấm lại trên câu trả lời đã lưu (không gọi lại model); Gemini 3 từ chối function call chèn tay thiếu *thought signature* → đưa kết quả search đầu vào tin nhắn user.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | Gồm 3 câu trên bảng dạng ảnh Fig 4.5 (agent dùng `view_figure` xác nhận) và 1 phép trừ (2.00) |
| Ảnh, sơ đồ | 5/5 | 100% | at10: mô tả index của Fig 6.9 sai nhưng `view_figure` đọc crop gốc cho đúng dải lương của WA |
| Biểu đồ | 5/5 | 100% | Số đọc từ trục (≈$11.50 năm 1968, ≈$730K năm 2007) nằm trong dung sai ghi sẵn |
| Multi-hop | 2/4 | 50% | at18: đúng số nhưng không cite đoạn text bắt buộc p175:t2; at19: đúng kết quả nhưng khẳng định sai "China, India có trong Table 6.2" |
| Text | 4/4 | 100% | Khẳng định phụ ở at21, at22 đều có trong nguồn được cite (đã đọc đối chiếu) |
| Thiếu bằng chứng | 5/6 | 83% | atu3: từ chối đúng nhưng thêm "biểu đồ chỉ có dữ liệu tới 2010" — sai (đường kéo tới ~2017). atu2, atu4: parser ghi "answered" vì có citation, nhưng nội dung nói rõ không có dữ liệu và số liệu kèm theo đều đúng → chấm đạt |
| **Toàn bộ** | **27/30** | **90.0%** | Ngưỡng cần 27/30; lỗi hệ thống 0 |

- **Ca giáp ranh at17** (multi-hop, chấm đạt): "Không" đúng và cite đủ hai bản đồ; nhưng vision chỉ liệt kê WA, CA, DC trong dải cao nhất và bỏ sót MA. Câu hỏi không yêu cầu liệt kê và MA cũng không có luật right-to-work trên Fig 6.10, nên evidence được cite hỗ trợ toàn bộ câu trả lời; tuy vậy chuỗi kiểm tra của agent chưa đầy đủ.
- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:** trace `results/advanced/test/adv_rerank_view/traces.json` (mỗi search: query, kết quả + rank + score + loại đơn vị, đơn vị thêm/bỏ; mỗi `view_figure`: unit, crop, bbox, câu hỏi, nội dung đọc được; câu trả lời, citation, token, thời gian), nhãn + lý do từng câu `labels.json`, tổng hợp `grade.json`; dev tương tự trong `results/advanced/dev/`. Test: median 5.5 s/câu, 63 lời gọi LLM + 17 lời gọi vision, 160.596 token vào / 4.573 token ra; dev (3 cấu hình): 78 LLM + 13 vision. Mô tả hình: 99 lời gọi (+ lần thử lại), tổng ~16 phút. **Chi phí $0** (Gemini API free tier); retrieval/embedding/rerank chạy local. Lỗi hệ thống khi chạy benchmark: 0. Lỗi trong lúc mô tả hình (đã xử lý, không ảnh hưởng kết quả): mạng mất tạm thời (DNS), 429 theo phút khi chạy 2 tiến trình song song, **hết quota 20 request/ngày của `gemini-3.5-flash`** → chuyển sang flash-lite, mô tả rỗng do RECITATION → prompt dự phòng; mô tả rỗng không còn được lưu vào cache.
- **Giới hạn, cách tái lập và đường dẫn benchmark:** (1) Biên an toàn bằng 0 (27/30) và có một ca giáp ranh; test 30 câu nên mỗi câu = 3.3 điểm %. (2) Multi-hop là điểm yếu (2/4): agent tìm đủ evidence (context đủ 24/24 câu) nhưng diễn đạt kèm khẳng định sai hoặc không cite đủ nguồn. (3) Mô tả hình do hai model khác nhau tạo (do quota), có ít nhất một mô tả sai (Fig 6.9); chất lượng câu trả lời về hình phụ thuộc `view_figure`. (4) Chưa chạy baseline trên test để tiết kiệm quota free tier; so sánh cấu hình chỉ có trên dev. (5) Chỉ có một người chấm. Tái lập: tải PDF → `python -m src.advanced.parse` → `describe` → `index` → `benchmark` → `run --split dev/test` → `rescore` (nếu cần) → `grade` (chi tiết: [src/README.md](../../src/README.md)); cần `GEMINI_API_KEY` trong `.env` và pgvector đang chạy. Benchmark: `data/advanced/benchmark.json`.
