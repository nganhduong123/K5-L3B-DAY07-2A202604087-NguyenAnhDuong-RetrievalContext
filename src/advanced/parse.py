"""Advanced step 1 — parse the PDF into indexable units with provenance.

Run: python -m src.advanced.parse

Unit types (all keep pdf page, chapter, section, bbox in PDF points, 612x792 page):
- text:   paragraph blocks of a page, packed to <= MAX_CHARS (headers/footers, captions and
          table cells removed so they are not indexed twice)
- table:  PyMuPDF `find_tables` cells -> markdown rows + title (line above) + "Table x.y" caption
- figure: every raster image >= MIN_PX, grouped by the "Figure x.y" caption printed below it;
          crop rendered from the page region (JPEG) so the visual can be checked/re-read.
          Visual content is described in step 2 (describe.py); here only caption + crop.
"""
import json
import re
from pathlib import Path

import pymupdf

from ..config import ROOT

PDF = ROOT / "data" / "advanced" / "source" / "business-ethics.pdf"
OUT = ROOT / "data" / "advanced" / "units.json"
CROPS = ROOT / "data" / "advanced" / "crops"
SOURCE_ID = "openstax-business-ethics"
MAX_CHARS = 900
MIN_PX = 150                     # images small in BOTH dimensions are icons/decoration
HEADER_Y, FOOTER_Y = 40, 750
FIG_RE = re.compile(r"^Figure (\d+\.\d+|[A-Z]\d+)\b")
TAB_RE = re.compile(r"^Table (\d+\.\d+)\b")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("­", "")).strip()


def toc_lookup(doc):
    toc = doc.get_toc()
    first_content = next(p for lvl, t, p in toc if t.startswith("Chapter 1"))
    last_content = next(p for lvl, t, p in toc if lvl == 1 and t.strip().lower() == "index") - 1

    def where(page):
        chapter = section = None
        for lvl, title, p in toc:
            if p > page:
                break
            if lvl == 1:
                m = re.match(r"Chapter (\d+)\s+(.*)", title)
                chapter, section = (int(m.group(1)), title) if m else (None, title)
            elif lvl == 2:
                section = title
        return chapter, section
    return first_content, last_content, where


def intersects(a, b, pad=2):
    return not (a[2] < b[0] - pad or a[0] > b[2] + pad or a[3] < b[1] - pad or a[1] > b[3] + pad)


def main() -> None:
    doc = pymupdf.open(PDF)
    first, last, where = toc_lookup(doc)
    CROPS.mkdir(parents=True, exist_ok=True)
    units, skipped = [], []
    for pno in range(first, last + 1):
        page = doc[pno - 1]
        chapter, section = where(pno)
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and HEADER_Y < b[1] and b[3] < FOOTER_Y]
        used = set()

        # tables
        for ti, tab in enumerate(page.find_tables().tables):
            rows = [[clean(c or "") for c in r] for r in tab.extract()]
            bbox = tuple(tab.bbox)
            title = next((clean(b[4]) for b in reversed(blocks) if b[3] <= bbox[1] + 2 and bbox[1] - b[3] < 30
                          and not TAB_RE.match(clean(b[4]))), "")
            cap_i = next((i for i, b in enumerate(blocks) if b[1] >= bbox[3] - 2 and TAB_RE.match(clean(b[4]))), None)
            caption = clean(blocks[cap_i][4]) if cap_i is not None else ""
            for i, b in enumerate(blocks):
                if intersects(b[:4], bbox) or i == cap_i or (title and clean(b[4]) == title):
                    used.add(i)
            md = "\n".join("| " + " | ".join(r) + " |" for r in rows)
            tid = TAB_RE.match(caption).group(1) if caption else f"p{pno}-{ti}"
            units.append({"id": f"{SOURCE_ID}:p{pno}:table{tid}", "type": "table", "page": pno, "chapter": chapter,
                          "section": section, "bbox": [round(x, 1) for x in bbox], "table_id": tid,
                          "title": title, "caption": caption, "rows": rows,
                          "text": f"Table {tid}: {title}\n{md}\n{caption}".strip()})

        # figures: group images by the caption printed below them
        images = [i for i in page.get_image_info() if i["width"] >= MIN_PX or i["height"] >= MIN_PX]
        groups: dict[str, dict] = {}
        for img in images:
            bb = img["bbox"]
            cap_i = None
            for i, b in sorted(enumerate(blocks), key=lambda x: x[1][1]):
                if b[1] >= bb[3] - 5 and FIG_RE.match(clean(b[4])):
                    cap_i = i
                    break
            if cap_i is None:
                # uncaptioned image (e.g. inline diagram strip): keep it, context = paragraph just above
                above = [b for b in blocks if b[3] <= bb[1] + 2]
                caption = ("[Uncaptioned image] Context above: " + clean(above[-1][4])) if above else "[Uncaptioned image]"
                fid = f"p{pno}-img{len(groups)}"
                skipped.append({"page": pno, "bbox": [round(x) for x in bb], "note": "no Figure caption; indexed as " + fid})
            else:
                caption = clean(blocks[cap_i][4])
                fid = FIG_RE.match(caption).group(1)
                used.add(cap_i)
            g = groups.setdefault(fid, {"caption": caption, "bbox": list(bb)})
            g["bbox"] = [min(g["bbox"][0], bb[0]), min(g["bbox"][1], bb[1]), max(g["bbox"][2], bb[2]), max(g["bbox"][3], bb[3])]
        for fid, g in groups.items():
            crop = CROPS / f"fig{fid}_p{pno}.jpg"
            page.get_pixmap(clip=pymupdf.Rect(g["bbox"]), dpi=200).save(crop, jpg_quality=88)
            units.append({"id": f"{SOURCE_ID}:p{pno}:fig{fid}", "type": "figure", "page": pno, "chapter": chapter,
                          "section": section, "bbox": [round(x, 1) for x in g["bbox"]], "figure_id": fid,
                          "caption": g["caption"], "crop": str(crop.relative_to(ROOT)).replace("\\", "/"),
                          "text": g["caption"]})          # description appended in step 2

        # text: remaining blocks, packed per page
        buf, k = "", 0
        for i, b in enumerate(blocks):
            if i in used:
                continue
            t = clean(b[4])
            if not t:
                continue
            if buf and len(buf) + 1 + len(t) > MAX_CHARS:
                units.append({"id": f"{SOURCE_ID}:p{pno}:t{k}", "type": "text", "page": pno, "chapter": chapter,
                              "section": section, "text": buf})
                buf, k = "", k + 1
            buf = f"{buf} {t}".strip()
        if buf:
            units.append({"id": f"{SOURCE_ID}:p{pno}:t{k}", "type": "text", "page": pno, "chapter": chapter,
                          "section": section, "text": buf})

    OUT.write_text(json.dumps(units, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    stats = {"pdf_pages": doc.page_count, "indexed_pages": f"{first}-{last}", "n_indexed_pages": last - first + 1,
             "units": len(units), "by_type": {t: sum(u["type"] == t for u in units) for t in ("text", "table", "figure")},
             "skipped_images": skipped,
             "excluded": {f"1-{first - 1}": "cover, copyright, contents, preface",
                          f"{last + 1}-{doc.page_count}": "back-of-book index"}}
    (OUT.parent / "parse_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "skipped_images"}, ensure_ascii=False),
          "| uncaptioned images:", len(skipped))


if __name__ == "__main__":
    main()
