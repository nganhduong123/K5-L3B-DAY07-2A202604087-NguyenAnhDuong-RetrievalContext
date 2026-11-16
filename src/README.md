# Code Day 7 — Agentic RAG trên vector store Day 6

| File | Vai trò |
|---|---|
| `config.py` | Đọc `.env`, cấu hình model, đường dẫn. Không chứa secret. |
| `export_day6.py` | Xuất 263 chunk strategy B từ pgvector Day 6 → `data/corpus/chunks.json` |
| `build_index.py` | Dựng lại index pgvector từ `chunks.json` (khi không có repo Day 6) |
| `build_benchmark.py` | Tạo `data/benchmark/questions.json` (câu có đáp án + gold) và `cases.json` (gồm câu thiếu evidence); kiểm tra evidence nằm trong chunk đúng trang và grep toàn corpus cho câu thiếu evidence |
| `retrievers.py` | 4 phương pháp: `dense` (pgvector Day 6), `bm25`, `hybrid_rrf`, `hybrid_rerank` |
| `eval_retrieval.py` | Hit@1, Evidence Hit@3, Context Evidence Rate, latency theo cấu hình (không gọi LLM) |
| `llm.py` | Client OpenRouter (tool calling, stdlib) |
| `agent.py` | Agent tool-calling: search → dedup + context budget → trả lời có `[chunk-id]` hoặc `INSUFFICIENT_EVIDENCE` |
| `run_agent.py` | Chạy agent thật trên một split, lưu trace + summary (chi phí, token, latency) |
| `make_reviews.py` | Nhãn chấm thủ công (kèm lý do) → `reviews.json` cho `starter/agent_metrics.py` |

## Cách chạy lại (từ thư mục gốc repo)

```powershell
# 0. Day 6 DB phải chạy (repo Day 6, thư mục starter/)
docker compose -p day6_student_starter up -d --wait

# 1. Môi trường + secret
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# tạo .env (không commit):
#   DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab
#   OPENROUTER_API_KEY=sk-or-...
$env:PYTHONUTF8 = "1"

# 2. Dữ liệu + benchmark
python -m src.export_day6        # có repo/DB Day 6: xuất chunk (đã commit sẵn data/corpus/chunks.json)
# Không có DB Day 6? Dựng lại index từ chunks.json vào pgvector bất kỳ (cùng model/prefix/schema;
# đã kiểm chứng top-3 dense trùng 36/36 câu với index Day 6 gốc):
#   python -m src.build_index
python -m src.build_benchmark    # đọc data/day6/questions_day6.json (bản sao benchmark Day 6)

# 3. Retrieval benchmark
python -m src.eval_retrieval --split dev
# 4. Agent trên dev
python -m src.run_agent --split dev --configs agent_dense_b800 agent_dense_b2400 agent_rerank_b2400 agent_rerank_b2400_v2 agent_rerank_b2400_v2_seed agent_rerank_b2400_v2_seed_41mini
# 5. Chốt cấu hình (results/SELECTION.md) rồi mới chạy test
python -m src.eval_retrieval --split test --configs dense_k3_b800 bm25_k3_b800 hybrid_rrf_k3_b800 hybrid_rerank_k3_b2400
python -m src.run_agent --split test --configs agent_dense_b800 agent_rerank_b2400_v2_seed
# 6. Nhãn + tổng hợp
python -m src.make_reviews
python starter/agent_metrics.py --cases data/benchmark/cases.json --traces results/agent-test/agent_rerank_b2400_v2_seed/traces.json --reviews results/agent-test/agent_rerank_b2400_v2_seed/reviews.json --split test --output results/agent-test/agent_rerank_b2400_v2_seed/agent_metrics.json

# Unit test (mock, không phải benchmark)
python -m unittest discover -s tests -v
```

Lần chạy đầu tải `intfloat/multilingual-e5-small` và `cross-encoder/ms-marco-MiniLM-L-6-v2` từ Hugging Face. LLM ở nhiệt độ 0 nhưng kết quả có thể lệch nhẹ giữa các lần gọi API.

## Thử thách nâng cao (`src/advanced/`)

| File | Vai trò |
|---|---|
| `parse.py` | PDF → 1.650 đơn vị text/bảng/hình (trang, chương, mục, bbox, crop JPEG) |
| `gemini.py` | Client Gemini (OpenAI-compatible endpoint): text, ảnh, tool calling, retry |
| `describe.py` | Mô tả từng hình bằng vision model (cache, fallback khi bị RECITATION) |
| `index.py` | Index E5 + pgvector (`strategy='ADV'`) và file cho BM25 |
| `benchmark.py` | Benchmark dev/test viết từ bản gốc + kiểm tra evidence/absence |
| `agent.py` | Agent 2 tool: `search_document`, `view_figure` (đọc crop gốc); `parse_answer` |
| `run.py`, `rescore.py`, `grade.py` | Chạy cấu hình, áp lại parser lên trace đã lưu, tổng hợp nhãn đạt/không đạt |

```powershell
# cần GEMINI_API_KEY trong .env, pgvector đang chạy, PDF tại data/advanced/source/business-ethics.pdf
pip install pymupdf==1.26.5
python -m src.advanced.parse
python -m src.advanced.describe                               # gemini-3.5-flash (20 req/ngày free)
python -m src.advanced.describe --model gemini-3.5-flash-lite  # phần còn lại (đã cache sẽ bỏ qua)
python -m src.advanced.index
python -m src.advanced.benchmark
python -m src.advanced.run --split dev --configs adv_dense_noview adv_rerank_noview adv_rerank_view
python -m src.advanced.rescore --split dev --configs adv_dense_noview adv_rerank_noview adv_rerank_view
# chốt (results/advanced/SELECTION.md) rồi:
python -m src.advanced.run --split test --configs adv_rerank_view
python -m src.advanced.grade --split test --config adv_rerank_view    # cần labels.json (chấm thủ công)
```
