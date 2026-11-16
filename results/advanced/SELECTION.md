# Thử thách nâng cao — chốt pipeline trước khi chạy test

Ghi lúc 2026-11-16, **trước** lần chạy test đầu tiên của phần nâng cao (chưa chạy agent trên split `test`).

## Tiêu chí (chỉ dùng dev, theo thứ tự)

1. Tỷ lệ đạt dev chấm thủ công (đúng + đủ + citation hỗ trợ).
2. Số câu dev cite **đúng vùng evidence bắt buộc** (`cites_all_required`) — vì test chấm cả citation trang/vùng.
3. Không có lỗi hệ thống; sau đó latency và số request (free tier giới hạn request/ngày).

## Kết quả dev (18 câu: 15 có đáp án + 3 thiếu evidence), parser citation v2

| Config | Đạt | Bảng | Ảnh | Biểu đồ | Multi-hop | Text | Thiếu evidence | cites_all_required | Median latency | LLM + vision calls |
|---|---|---|---|---|---|---|---|---|---|---|
| adv_dense_noview (baseline) | 17/18 | 3/3 | 3/3 | 4/4 | 1/2 | 3/3 | 3/3 | 15/15 | 1.94 s | 23 + 0 |
| adv_rerank_noview | 18/18 | 3/3 | 3/3 | 4/4 | 2/2 | 3/3 | 3/3 | 14/15 | 2.68 s | 21 + 0 |
| **adv_rerank_view** | **18/18** | 3/3 | 3/3 | 4/4 | 2/2 | 3/3 | 3/3 | **15/15** | 5.33 s | 34 + 13 |

## Quyết định

- **Chọn `adv_rerank_view`**: hybrid RRF (dense E5 trên pgvector `ADV` + BM25) → cross-encoder rerank, top-5 mỗi lượt, budget 9000 ký tự tích lũy, tối đa 4 search + 3 `view_figure`, lượt đầu dùng câu hỏi gốc, `gemini-3.5-flash-lite` cho cả agent và đọc ảnh, temperature 0.
- Lý do: hoà 18/18 với `adv_rerank_noview` ở tiêu chí 1; thắng ở tiêu chí 2 (15/15 so với 14/15). Ngoài ra (lý do thiết kế, không dựa vào test): mô tả hình là do model sinh ra, đề yêu cầu không coi đó là đúng mặc định; `view_figure` đọc lại ảnh crop gốc trước khi trả lời. Đánh đổi: chậm hơn ~2×, nhiều request hơn.
- Baseline so sánh trên test (nếu còn quota): `adv_dense_noview`.
- Thay đổi trong quá trình dev (ghi lại): sửa parser citation (một ngoặc chứa nhiều id) — áp dụng lại lên câu trả lời đã lưu, không gọi lại model; seed search chuyển sang tin nhắn user (Gemini 3 từ chối function call chèn tay không có thought signature).
- Sau khi chạy test sẽ không chỉnh pipeline. Lỗi hệ thống trên test tính là không đạt.

## Ghi chú thêm sau khi chấm test (không thay đổi pipeline)

Khi chấm test theo quy tắc "cite mọi đơn vị evidence bắt buộc", dev `adv_rerank_noview` ad08 (chỉ cite đoạn text p225:t1, không cite Fig 8.2 bắt buộc) được chấm lại thành **không đạt** cho nhất quán → `adv_rerank_noview` = 17/18. Lựa chọn `adv_rerank_view` (18/18) không đổi. Kết quả test: 27/30 (xem `test/adv_rerank_view/grade.json`).
