# rag_ingest.py
# ------------------------------------------------------------
# pip install pymupdf faiss-cpu sentence-transformers numpy tqdm
# (optional) pip install pandas
#
# What this does:
# 1) Walk __test_data__/ for PDFs
# 2) Extract per-page blocks (text/image/table), per-line IDs + bboxes (via fitz)
# 3) Save per-PDF JSON (same basename) under artifacts/extracted_json/
# 4) Build chunks from those JSONs, embed with SentenceTransformers,
#    and write a FAISS index + sidecar metadata for RAG.
# ------------------------------------------------------------

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable, Set
from tqdm import tqdm

import numpy as np
import fitz  # PyMuPDF

import faiss
from sentence_transformers import SentenceTransformer


# ------------------------ CONFIG ------------------------

INPUT_DIR = Path("__test_data__")             # your research PDFs
OUT_JSON_DIR = Path("artifacts/extracted_json")
ARTIFACTS_DIR = Path("artifacts")
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"  # text + metadata per chunk
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
DOCSTORE_PATH = ARTIFACTS_DIR / "docstore.jsonl"  # aligns vector IDs to chunks

# embedding model: small + fast
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# chunking
MAX_CHARS = 900
OVERLAP_CHARS = 200

# layout heuristics
MIN_RIGHT_RATIO_FOR_TWO_COL = 0.20  # if < 20% words fall in right half, assume single column

from dataclasses import dataclass

LINE_Y_TOL = 3.0  # y-proximity for grouping words into lines

@dataclass
class _W:
    x0: float; y0: float; x1: float; y1: float; text: str

def _words_in_rect(page: fitz.Page, rect: Tuple[float, float, float, float]) -> List[_W]:
    """Return words (as _W) clipped to a block bbox."""
    r = fitz.Rect(*rect)
    words = page.get_text("words", clip=r)  # (x0,y0,x1,y1, text, block_no, line_no, word_no)
    return [_W(w[0], w[1], w[2], w[3], w[4]) for w in words]

def _group_words_into_lines(words: List[_W]) -> List[Dict[str, Any]]:
    """Group words into lines by Y with tolerance; return [{'text','bbox','y'}...]"""
    if not words:
        return []
    # sort by y then x
    ws = sorted(words, key=lambda w: (w.y0, w.x0))
    buckets: List[List[_W]] = [[ws[0]]]
    cur_y = ws[0].y0
    for w in ws[1:]:
        if abs(w.y0 - cur_y) <= LINE_Y_TOL:
            buckets[-1].append(w)
        else:
            buckets.append([w])
            cur_y = w.y0
    lines = []
    for b in buckets:
        b.sort(key=lambda w: w.x0)
        text = " ".join(w.text for w in b if w.text)
        if not text.strip():
            continue
        x0 = min(w.x0 for w in b); y0 = min(w.y0 for w in b)
        x1 = max(w.x1 for w in b); y1 = max(w.y1 for w in b)
        lines.append({"text": text, "bbox": [float(x0), float(y0), float(x1), float(y1)], "y": float(y0)})
    return lines
# ------------------------ COLUMN / TABLE UTILS ------------------------

def _is_two_col(words: List[Tuple[float, float, float, float, str, int, int, int]], page_width: float) -> bool:
    """Heuristic: if ≥20% of words are in the right half → two columns."""
    if not words:
        return False
    mid = page_width / 2.0
    right = sum(1 for w in words if ((w[0] + w[2]) / 2.0) > mid)
    return (right / max(1, len(words))) >= MIN_RIGHT_RATIO_FOR_TWO_COL


def _col_index_from_bbox(bbox: Tuple[float, float, float, float], page_width: float, two_col: bool) -> int:
    if not two_col:
        return 1
    x_center = (bbox[0] + bbox[2]) / 2.0
    return 1 if x_center <= page_width / 2.0 else 2


def _detect_table_candidates(page: fitz.Page) -> List[Tuple[float, float, float, float]]:
    """
    Lightweight table candidate detector:
    - scans vector drawings for straight horizontal/vertical segments
    - clusters dense regions into rectangles
    """
    drawings = page.get_drawings()
    horiz, vert = [], []

    for d in drawings:
        for path in d["items"]:
            if path[0] != "l":  # straight segment
                continue
            (x0, y0), (x1, y1) = path[1], path[2]
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            if dx >= 4 and dy < 0.5:
                horiz.append((min(x0, x1), y0, max(x0, x1), y1))
            if dy >= 4 and dx < 0.5:
                vert.append((x0, min(y0, y1), x1, max(y0, y1)))

    if not horiz or not vert:
        return []

    # bucket lines into a coarse grid and find dense clusters
    buckets = {}
    def _key(x0, y0, x1, y1):
        return (int(x0 // 25), int(y0 // 25), int(x1 // 25), int(y1 // 25))

    for (x0, y0, x1, y1) in horiz + vert:
        k = _key(x0, y0, x1, y1)
        buckets[k] = buckets.get(k, 0) + 1

    dense = {k for k, v in buckets.items() if v >= 2}
    if not dense:
        return []

    visited: Set[Tuple[int, int, int, int]] = set()
    clusters: List[List[Tuple[int, int, int, int]]] = []

    for k in dense:
        if k in visited:
            continue
        stack, comp = [k], []
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in dense:
                continue
            visited.add(cur)
            comp.append(cur)
            cx0, cy0, cx1, cy1 = cur
            for nx in range(cx0 - 1, cx0 + 2):
                for ny in range(cy0 - 1, cy0 + 2):
                    for nx1 in range(cx1 - 1, cx1 + 2):
                        for ny1 in range(cy1 - 1, cy1 + 2):
                            nk = (nx, ny, nx1, ny1)
                            if nk in dense and nk not in visited:
                                stack.append(nk)
        clusters.append(comp)

    tables: List[Tuple[float, float, float, float]] = []
    for comp in clusters:
        xs, ys = [], []
        for (gx0, gy0, gx1, gy1) in comp:
            xs.extend([gx0 * 25, gx1 * 25 + 25])
            ys.extend([gy0 * 25, gy1 * 25 + 25])
        if xs and ys:
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            if (x1 - x0) > 60 and (y1 - y0) > 40:
                tables.append((float(x0), float(y0), float(x1), float(y1)))

    return tables


# ------------------------ PDF → JSON (BLOCKS + LINES) ------------------------

def _extract_page_blocks(page: fitz.Page, file_name: str, page_number: int) -> Dict[str, Any]:
    pw = page.rect.width
    words = page.get_text("words")
    two_col = _is_two_col(words, pw)

    raw = page.get_text("rawdict")
    blocks_out: List[Dict[str, Any]] = []
    block_counter = 0
    image_counter = 0
    table_counter = 0

    def mk_block_id(kind: str, idx: int) -> str:
        return f"{file_name}::p{page_number}::{kind}{idx}"

    def mk_line_id(bid: str, idx: int) -> str:
        return f"{bid}::l{idx}"

    # TEXT & IMAGES
    for b in raw.get("blocks", []):
        btype = b.get("type", 0)
        bbox = tuple(float(v) for v in b.get("bbox", (0, 0, 0, 0)))
        column = _col_index_from_bbox(bbox, pw, two_col)

        if btype == 0:  # text block
            block_counter += 1
            block_id = mk_block_id("b", block_counter)
            lines_out, line_ids = [], []

            # 1) Try span-based extraction from rawdict
            span_lines: List[Dict[str, Any]] = []
            for ln in b.get("lines", []):
                text = " ".join(
                    (s.get("text") or "").strip()
                    for s in ln.get("spans", [])
                    if s.get("text") is not None
                ).strip()
                if not text:
                    continue
                lbbox = tuple(float(v) for v in ln.get("bbox", (0, 0, 0, 0)))
                span_lines.append({"text": text, "bbox": [*lbbox], "y": float(lbbox[1])})

            # 2) Fallback: build lines from words inside this block bbox if spans were empty
            if not span_lines:
                word_lines = _group_words_into_lines(_words_in_rect(page, bbox))
                span_lines = word_lines  # use the same structure

            # 3) Write lines with ids
            for li, ln in enumerate(span_lines, start=1):
                line_id = mk_line_id(block_id, li)
                line_ids.append(line_id)
                lines_out.append({
                    "line_id": line_id,
                    "text": ln["text"],
                    "bbox": ln["bbox"],
                    "y": ln["y"],
                    "column": column,
                    "block_id": block_id
                })

            blocks_out.append({
                "block_id": block_id,
                "type": "text",
                "bbox": [*bbox],
                "column": column,
                "line_ids": line_ids,
                "lines": lines_out
            })

            blocks_out.append({
                "block_id": block_id,
                "type": "text",
                "bbox": [*bbox],
                "column": column,
                "line_ids": line_ids,
                "lines": lines_out
            })

        elif btype == 1:  # image block
            image_counter += 1
            block_id = mk_block_id("img", image_counter)

            # Some PDFs have inline image bytes in b["image"]; keep only JSON-safe bits
            raw_img = b.get("image", None)
            if isinstance(raw_img, (int, float, str)):
                xref_val = raw_img
            else:
                xref_val = None  # ignore inline bytes

            blocks_out.append({
                "block_id": block_id,
                "type": "image",
                "bbox": [*bbox],
                "column": column,
                "meta": {
                    "xref": xref_val,
                    "width": b.get("width"),
                    "height": b.get("height"),
                    "colorspace": b.get("cs-name"),
                }
            })

    # TABLE CANDIDATES (vector drawings)
    for tb in _detect_table_candidates(page):
        table_counter += 1
        column = _col_index_from_bbox(tb, pw, two_col)
        block_id = mk_block_id("tbl", table_counter)
        blocks_out.append({
            "block_id": block_id,
            "type": "table",
            "bbox": [*tb],
            "column": column,
            "cells": None  # placeholder for future cell extraction
        })

    # sort for stable read order: by column, then top-y
    blocks_out.sort(key=lambda blk: (blk["column"], blk["bbox"][1] if blk.get("bbox") else 0.0))

    return {
        "page_number": page_number,
        "two_columns": two_col,
        "blocks": blocks_out
    }


def extract_pdf_to_struct_with_blocks(pdf_path: Path) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    out = {"file_name": pdf_path.name, "num_pages": doc.page_count, "pages": []}
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        out["pages"].append(_extract_page_blocks(page, pdf_path.name, pno + 1))
    doc.close()
    return out

def _json_default_fallback(o):
    # Avoid dumping huge base64; just summarize non-JSON types
    if isinstance(o, (bytes, bytearray)):
        return f"<bytes:{len(o)}>"
    # Last-resort stringification for anything else odd
    return str(o)

def save_pdf_json(struct: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(struct["file_name"]).with_suffix(".json").name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(struct, f, ensure_ascii=False, indent=2)


def extract_all_pdfs(input_dir: Path = INPUT_DIR, out_dir: Path = OUT_JSON_DIR) -> None:
    pdfs = sorted(list(input_dir.glob("**/*.pdf")))
    if not pdfs:
        print(f"[warn] no PDFs found under {input_dir.resolve()}")
        return
    for pdf in tqdm(pdfs, desc="Extracting PDFs"):
        struct = extract_pdf_to_struct_with_blocks(pdf)
        save_pdf_json(struct, out_dir)


# ------------------------ CHUNKING (USES LINE & BLOCK IDS) ------------------------

def iter_text_lines(obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield text-line records in reading order:
    {page, column, block_id, line_id, text, bbox}
    """
    for page in obj["pages"]:
        for blk in page["blocks"]:
            if blk.get("type") != "text":
                continue
            for ln in blk.get("lines", []):
                yield {
                    "page": page["page_number"],
                    "column": ln["column"],
                    "block_id": ln["block_id"],
                    "line_id": ln["line_id"],
                    "text": ln["text"],
                    "bbox": ln["bbox"]
                }


def make_chunks_from_json_dir(json_dir: Path,
                              max_chars: int = MAX_CHARS,
                              overlap: int = OVERLAP_CHARS) -> None:
    """
    Turn per-line JSON into overlapping text chunks with rich metadata:
      - keep all line_ids included in the chunk
      - keep the set of block_ids spanned
      - preserve first/last page/column + line span counts
    Writes CHUNKS_PATH as jsonl.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = 0

    with open(CHUNKS_PATH, "w", encoding="utf-8") as out_f:
        for jf in sorted(json_dir.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                obj = json.load(f)

            buffer = ""
            lines_in_chunk: List[Dict[str, Any]] = []

            def flush():
                nonlocal buffer, lines_in_chunk, n_chunks
                if not buffer:
                    return
                first = lines_in_chunk[0]
                last = lines_in_chunk[-1]
                block_ids = sorted({ln["block_id"] for ln in lines_in_chunk})
                line_ids = [ln["line_id"] for ln in lines_in_chunk]

                chunk_id = (
                    f"{obj['file_name']}::"
                    f"p{first['page']}-p{last['page']}::"
                    f"c{first['column']}-c{last['column']}::"
                    f"{n_chunks}"
                )

                record = {
                    "id": chunk_id,
                    "text": buffer.strip(),
                    "meta": {
                        "file": obj["file_name"],
                        "page_start": first["page"],
                        "page_end": last["page"],
                        "column_start": first["column"],
                        "column_end": last["column"],
                        "block_ids": block_ids,
                        "line_ids": line_ids,
                        # optional quick pointers
                        "first_line_bbox": first["bbox"],
                        "last_line_bbox": last["bbox"]
                    }
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1

                # start new buffer with character overlap (not line aware)
                tail = buffer[-overlap:] if overlap > 0 else ""
                buffer = tail
                lines_in_chunk = []

            # accumulate lines and flush on size
            for ln in iter_text_lines(obj):
                proposed = (buffer + ("\n" if buffer else "") + ln["text"]).strip()
                if len(proposed) > max_chars and buffer:
                    flush()
                buffer = (buffer + ("\n" if buffer else "") + ln["text"]).strip()
                lines_in_chunk.append(ln)

            # flush remainder
            if buffer:
                flush()

    print(f"[ok] wrote {n_chunks} chunks → {CHUNKS_PATH}")


# ------------------------ EMBEDDINGS + FAISS ------------------------

def load_chunks_jsonl(path: Path) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    texts, metas, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append(rec["meta"])
            ids.append(rec["id"])
    return texts, metas, ids


def build_faiss_index(texts: List[str], ids: List[str], model_name: str = EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)  # vector ids are 0..N-1

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            f.write(json.dumps({"vec_id": i, "chunk_id": cid}) + "\n")

    print(f"[ok] FAISS index saved → {FAISS_INDEX_PATH} (vectors: {len(texts)})")
    print(f"[ok] Docstore saved → {DOCSTORE_PATH}")
    return index


# ------------------------ MAIN ------------------------
# ------------------------ PATHS FOR PER-DOC INDICES ------------------------

INDICES_DIR = ARTIFACTS_DIR / "indices"   # artifacts/indices/<docstem>/...

def _doc_stem(file_name: str) -> str:
    return Path(file_name).with_suffix("").name

def _doc_dir(doc_stem: str) -> Path:
    return INDICES_DIR / doc_stem

def _doc_paths(doc_stem: str):
    d = _doc_dir(doc_stem)
    return {
        "dir": d,
        "chunks": d / "chunks.jsonl",
        "index": d / "index.faiss",
        "docstore": d / "docstore.jsonl"
    }


# ------------------------ CHUNKING (PER DOCUMENT) ------------------------

def iter_text_lines(obj: Dict[str, Any]):
    """Yield text-line records in reading order for a single parsed JSON object."""
    for page in obj["pages"]:
        for blk in page["blocks"]:
            if blk.get("type") != "text":
                continue
            for ln in blk.get("lines", []):
                yield {
                    "page": page["page_number"],
                    "column": ln["column"],
                    "block_id": ln["block_id"],
                    "line_id": ln["line_id"],
                    "text": ln["text"],
                    "bbox": ln["bbox"]
                }

def _write_chunks_for_obj(obj: Dict[str, Any],
                          max_chars: int,
                          overlap: int) -> Path:
    """
    Create chunks.jsonl for ONE document (obj is the parsed per-PDF JSON).
    Returns the path to that document's chunks.jsonl.
    """
    docstem = _doc_stem(obj["file_name"])
    paths = _doc_paths(docstem)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    n_chunks = 0
    buffer = ""
    lines_in_chunk: List[Dict[str, Any]] = []

    with open(paths["chunks"], "w", encoding="utf-8") as out_f:

        def flush():
            nonlocal buffer, lines_in_chunk, n_chunks
            if not buffer:
                return
            first = lines_in_chunk[0]
            last = lines_in_chunk[-1]
            block_ids = sorted({ln["block_id"] for ln in lines_in_chunk})
            line_ids = [ln["line_id"] for ln in lines_in_chunk]

            chunk_id = (
                f"{obj['file_name']}::"
                f"p{first['page']}-p{last['page']}::"
                f"c{first['column']}-c{last['column']}::"
                f"{n_chunks}"
            )

            record = {
                "id": chunk_id,
                "text": buffer.strip(),
                "meta": {
                    "file": obj["file_name"],
                    "docstem": docstem,
                    "page_start": first["page"],
                    "page_end": last["page"],
                    "column_start": first["column"],
                    "column_end": last["column"],
                    "block_ids": block_ids,
                    "line_ids": line_ids,
                    "first_line_bbox": first["bbox"],
                    "last_line_bbox": last["bbox"],
                }
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_chunks += 1

            # start new buffer with character overlap
            tail = buffer[-overlap:] if overlap > 0 else ""
            buffer = tail
            lines_in_chunk = []

        for ln in iter_text_lines(obj):
            proposed = (buffer + ("\n" if buffer else "") + ln["text"]).strip()
            if len(proposed) > max_chars and buffer:
                flush()
            buffer = (buffer + ("\n" if buffer else "") + ln["text"]).strip()
            lines_in_chunk.append(ln)

        if buffer:
            flush()

    print(f"[ok] wrote {n_chunks} chunks → {paths['chunks']}")
    return paths["chunks"]


def make_chunks_per_document(json_dir: Path,
                             max_chars: int = MAX_CHARS,
                             overlap: int = OVERLAP_CHARS) -> List[Tuple[str, Path]]:
    """
    For every extracted JSON in json_dir, write a per-document chunks.jsonl.
    Returns list of (docstem, chunks_path).
    """
    results: List[Tuple[str, Path]] = []
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    for jf in sorted(json_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            obj = json.load(f)
        chunks_path = _write_chunks_for_obj(obj, max_chars, overlap)
        results.append((_doc_stem(obj["file_name"]), chunks_path))

    return results


# ------------------------ FAISS (PER DOCUMENT) ------------------------

def _load_chunks_jsonl(path: Path) -> Tuple[List[str], List[str]]:
    texts, ids = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            ids.append(rec["id"])
    return texts, ids

def build_faiss_index_for_doc(docstem: str,
                              model: SentenceTransformer) -> None:
    """
    Build FAISS index for one document using its chunks.jsonl.
    Writes index.faiss + docstore.jsonl under artifacts/indices/<docstem>/.
    """
    paths = _doc_paths(docstem)
    texts, ids = _load_chunks_jsonl(paths["chunks"])
    if not texts:
        print(f"[warn] no chunks for {docstem}")
        return

    emb = model.encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)  # vector ids 0..N-1

    faiss.write_index(index, str(paths["index"]))
    with open(paths["docstore"], "w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            f.write(json.dumps({"vec_id": i, "chunk_id": cid}) + "\n")

    print(f"[ok] [{docstem}] index → {paths['index']}  (vectors: {len(texts)})")


def build_all_doc_indices(docstems: List[str],
                          model_name: str = EMBED_MODEL_NAME) -> None:
    """
    Build per-doc indices with a single embed model instance (faster).
    """
    model = SentenceTransformer(model_name)
    for ds in docstems:
        build_faiss_index_for_doc(ds, model)


# ------------------------ QUERY ACROSS SELECTED DOCS ------------------------

def _load_doc_index(docstem: str):
    """Load one per-doc FAISS index and its docstore + chunk map."""
    paths = _doc_paths(docstem)
    if not paths["index"].exists() or not paths["docstore"].exists() or not paths["chunks"].exists():
        raise FileNotFoundError(f"Missing index/docstore/chunks for doc '{docstem}'")

    index = faiss.read_index(str(paths["index"]))

    vecid_to_chunkid = {}
    with open(paths["docstore"], "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            vecid_to_chunkid[j["vec_id"]] = j["chunk_id"]

    chunk_by_id = {}
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_by_id[rec["id"]] = rec

    return index, vecid_to_chunkid, chunk_by_id


def list_available_docs() -> List[str]:
    """Return available document stems that have a chunks.jsonl (and likely an index)."""
    if not INDICES_DIR.exists():
        return []
    stems = []
    for d in INDICES_DIR.iterdir():
        if not d.is_dir():
            continue
        if (d / "chunks.jsonl").exists():
            stems.append(d.name)
    return sorted(stems)


def multi_index_query(query: str,
                      include_docs: List[str],
                      top_k: int = 5,
                      model_name: str = EMBED_MODEL_NAME) -> List[Dict[str, Any]]:
    """
    Search ONLY the specified docs. We:
    - encode the query once
    - search each selected per-doc index for top_k
    - merge + re-sort by score
    """
    if not include_docs:
        return []

    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    merged: List[Dict[str, Any]] = []
    for ds in include_docs:
        try:
            index, v2c, chunks = _load_doc_index(ds)
        except FileNotFoundError:
            print(f"[warn] skipping '{ds}' (no index)")
            continue

        sims, idxs = index.search(q, top_k)
        for rank_local, (vec_id, score) in enumerate(zip(idxs[0], sims[0]), start=1):
            cid = v2c.get(int(vec_id))
            rec = chunks.get(cid)
            if not rec:
                continue
            meta = rec["meta"]
            merged.append({
                "doc": ds,
                "rank_local": rank_local,
                "score": float(score),
                "id": cid,
                "file": meta["file"],
                "pages": f"{meta['page_start']}–{meta['page_end']}",
                "columns": f"{meta['column_start']}–{meta['column_end']}",
                "blocks": meta["block_ids"],
                "lines": (meta["line_ids"][0], meta["line_ids"][-1]),
                "preview": rec["text"][:240].replace("\n", " ") + ("…" if len(rec["text"]) > 240 else "")
            })

    # global sort by score desc
    merged.sort(key=lambda r: r["score"], reverse=True)
    # return top_k overall (optional: set higher to see more)
    return merged[:top_k]


# ------------------------ MAIN ------------------------

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Extract PDFs → JSON (blocks + lines + bboxes)
    print("[1/4] extracting PDFs → JSON")
    extract_all_pdfs(INPUT_DIR, OUT_JSON_DIR)

    # 2) Build per-document chunks
    print("[2/4] building per-document chunks")
    doc_chunks = make_chunks_per_document(OUT_JSON_DIR, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS)
    docstems = [ds for ds, _ in doc_chunks]
    print(f"[ok] documents prepared: {docstems}")

    # 3) Build FAISS index per document
    print("[3/4] building FAISS indices (per document)")
    build_all_doc_indices(docstems, model_name=EMBED_MODEL_NAME)

    # 4) (Optional) Smoke test against selected docs
    #    e.g., include only the first two docs
    # include = docstems[:2]
    # hits = multi_index_query("temperature scaling and label smoothing", include_docs=include, top_k=5)
    # for h in hits:
    #     print(h)


if __name__ == "__main__":
    main()
