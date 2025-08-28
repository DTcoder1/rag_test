# rag_reference.py
# ------------------------------------------------------------
# pip install sentence-transformers faiss-cpu python-dotenv openai
# (optional) pip install rank-bm25
# (optional) pip install cross-encoder
#
# CLI:
#   python rag_reference.py --text "Your paragraph here." --docs paperA paperB --deployment gpt-4o
#   cat story.txt | python rag_reference.py --docs paperA paperB --deployment gpt-4o
#
# Output JSON keys:
#   - annotated_paragraph: str  (original text with [1,2] style citations)
#   - sentence_citations: [{sentence_index, source_ids:[int,...]}]
#   - sources: [{
#        source_id: int,
#        doc: str, file: str, page: str, block_id: str, line_ids: [str,...],
#        address: str,          # file::page::block_id
#        snippet: str,          # short supporting text
#        why_used: str          # brief rationale
#     }]
#   - notes: str
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Optional deps ---
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _HAS_XENC = True
except Exception:
    _HAS_XENC = False

# --- Azure OpenAI SDK ---
try:
    from openai import AzureOpenAI  # type: ignore
    _HAS_AZURE_OPENAI = True
except Exception:
    _HAS_AZURE_OPENAI = False

ARTIFACTS_DIR = Path("artifacts")
INDICES_DIR = ARTIFACTS_DIR / "indices"
DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -------------------- IO helpers --------------------

def _doc_paths(docstem: str):
    d = INDICES_DIR / docstem
    return {"dir": d, "chunks": d / "chunks.jsonl", "index": d / "index.faiss", "docstore": d / "docstore.jsonl"}

def _load_doc_index(docstem: str):
    paths = _doc_paths(docstem)
    if not (paths["index"].exists() and paths["docstore"].exists() and paths["chunks"].exists()):
        raise FileNotFoundError(f"Missing index/docstore/chunks for '{docstem}'")
    index = faiss.read_index(str(paths["index"]))

    vecid_to_chunkid: Dict[int, str] = {}
    with open(paths["docstore"], "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            vecid_to_chunkid[int(j["vec_id"])] = j["chunk_id"]

    chunk_by_id: Dict[str, Dict[str, Any]] = {}
    texts_for_bm25: List[str] = []
    ids_for_bm25: List[str] = []
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cid = rec["id"]
            chunk_by_id[cid] = rec
            texts_for_bm25.append(rec["text"])
            ids_for_bm25.append(cid)

    return index, vecid_to_chunkid, chunk_by_id, texts_for_bm25, ids_for_bm25

# -------------------- small text utils --------------------

_SENT_SPLIT = re.compile(r'(?<!\b[A-Z])[.!?](?=\s+|$)')  # rough sentence splitter

def _split_sentences(paragraph: str) -> List[str]:
    txt = paragraph.strip()
    if not txt:
        return []
    parts = _SENT_SPLIT.split(txt)
    delims = _SENT_SPLIT.findall(txt)
    sents = []
    for i, p in enumerate(parts):
        if not p.strip():
            continue
        tail = delims[i] if i < len(delims) else ""
        sents.append((p.strip(), tail))
    # Join text+delimiter to preserve punctuation
    return [p + d for p, d in sents]

def _has_digits(s: str) -> bool:
    return any(c.isdigit() for c in s)

def _needs_citation(sent: str) -> bool:
    """
    Lightweight heuristic: cite if the sentence looks like a factual/quantitative claim.
    (You can add an LLM-based decider later if desired.)
    """
    s = sent.lower()
    triggers = [
        "according to", "study", "studies", "paper", "report", "dataset",
        "increase", "decrease", "improve", "outperform", "first", "largest",
        "state-of-the-art", "sota", "achieves", "shows", "demonstrates",
        "accuracy", "precision", "recall", "auc", "f1", "benchmark", "estimate",
        "approximately", "around ", "more than", "less than"
    ]
    if _has_digits(s):
        return True
    return any(t in s for t in triggers)

def _unwrap_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith("```") and t.endswith("```"):
        parts = t.split("```")
        for part in parts:
            p = part.strip()
            if not p:
                continue
            if p.lower().startswith("json"):
                p = p[4:].strip()
            return p
    return t

def _json_loads_relaxed(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        if "```" in s:
            for part in s.split("```"):
                p = part.strip()
                if p.lower().startswith("json"):
                    p = p[4:].strip()
                try:
                    return json.loads(p)
                except Exception:
                    pass
    return None

# -------------------- data models --------------------

@dataclass
class Hit:
    text: str
    doc: str
    id: str
    score: float
    citation: Dict[str, Any]

@dataclass
class RetrievedInternal:
    doc: str
    id: str
    text: str
    score: float
    file: str
    page_start: int
    page_end: int
    column_start: int
    column_end: int
    block_ids: List[str]
    line_ids_all: List[str]
    line_text_map: Dict[str, str]
    line_bbox_map: Dict[str, Optional[List[float]]]

# -------------------- provider-agnostic chat wrapper --------------------

class ChatLLM:
    def __init__(
        self,
        *,
        provider: str = "azure",
        deployment: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        provider = (provider or "azure").lower()
        if provider != "azure":
            raise RuntimeError(f"Unsupported provider '{provider}' (only 'azure' implemented)")
        if not _HAS_AZURE_OPENAI:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.azure_endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_API_TARGET_URI") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

        if not (self.deployment and self.azure_endpoint and self.api_key):
            raise RuntimeError("Missing Azure config. Need AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_TARGET_URI, and a deployment name.")

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.deployment,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

# -------------------- RAG Engine --------------------

class RAGEngine:
    def __init__(self, embed_model_name: str = DEFAULT_EMBED, use_cross_encoder: bool = False, cross_encoder_name: str = DEFAULT_RERANKER):
        self.embed = SentenceTransformer(embed_model_name)
        self.use_xenc = use_cross_encoder and _HAS_XENC
        self.xenc = CrossEncoder(cross_encoder_name) if self.use_xenc else None

    def list_docs(self) -> List[str]:
        if not INDICES_DIR.exists():
            return []
        return sorted([d.name for d in INDICES_DIR.iterdir() if (d / "chunks.jsonl").exists()])

    def _hybrid_candidates(
        self, query: str, include_docs: List[str], bm25_weight: float = 0.35, faiss_topk_per_doc: int = 12
    ) -> List[RetrievedInternal]:
        q_vec = self.embed.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        all_cands: List[RetrievedInternal] = []

        for ds in include_docs:
            index, v2c, chunks, bm25_texts, bm25_ids = _load_doc_index(ds)

            # semantic (FAISS)
            sims, idxs = index.search(np.array([q_vec]), faiss_topk_per_doc)
            sem_pairs: List[Tuple[str, float]] = []
            for vec_id, score in zip(idxs[0], sims[0]):
                cid = v2c.get(int(vec_id))
                if cid and cid in chunks:
                    sem_pairs.append((cid, float(score)))

            # lexical (BM25)
            lex_pairs: List[Tuple[str, float]] = []
            if _HAS_BM25 and bm25_texts:
                tok = [t.split() for t in bm25_texts]
                bm = BM25Okapi(tok)
                scrs = bm.get_scores(query.split())
                if len(scrs) > 0:
                    top_idx = np.argsort(scrs)[::-1][:faiss_topk_per_doc]
                    lex_pairs = [(bm25_ids[i], float(scrs[i])) for i in top_idx]

            def _znorm(vals: List[float]) -> List[float]:
                if not vals:
                    return []
                arr = np.array(vals, dtype=np.float32)
                mu, sd = arr.mean(), arr.std()
                if sd < 1e-6: return [0.0] * len(vals)
                return ((arr - mu) / sd).tolist()

            sem_ids, sem_scores = zip(*sem_pairs) if sem_pairs else ([], [])
            lex_ids, lex_scores = zip(*lex_pairs) if lex_pairs else ([], [])
            sem_norm = _znorm(list(sem_scores)) if sem_pairs else []
            lex_norm = _znorm(list(lex_scores)) if lex_pairs else []

            sem_map = {i: s for i, s in zip(sem_ids, sem_norm)}
            lex_map = {i: s for i, s in zip(lex_ids, lex_norm)}

            fused_ids = set(sem_map) | set(lex_map) if _HAS_BM25 else set(sem_map)
            for cid in fused_ids:
                sem_s = sem_map.get(cid, 0.0)
                lex_s = lex_map.get(cid, 0.0) if _HAS_BM25 else 0.0
                fused = (1 - bm25_weight) * sem_s + bm25_weight * lex_s

                rec = chunks[cid]
                meta = rec["meta"]

                block_ids: List[str] = meta.get("block_ids", [])
                line_ids_all: List[str] = meta.get("line_ids", []) or []
                line_text_map: Dict[str, str] = meta.get("line_text_map") or {}
                line_bbox_map: Dict[str, Optional[List[float]]] = meta.get("line_bbox_map") or {}

                if not line_text_map and line_ids_all:
                    lines = rec["text"].split("\n")
                    if len(lines) == len(line_ids_all):
                        line_text_map = {lid: ln for lid, ln in zip(line_ids_all, lines)}
                    else:
                        m = min(len(lines), len(line_ids_all))
                        line_text_map = {lid: ln for lid, ln in zip(line_ids_all[:m], lines[:m])}

                for lid in line_ids_all:
                    if lid not in line_bbox_map:
                        line_bbox_map[lid] = None

                all_cands.append(RetrievedInternal(
                    doc=ds, id=cid, text=rec["text"], score=float(fused),
                    file=meta["file"], page_start=int(meta.get("page_start", 0)), page_end=int(meta.get("page_end", 0)),
                    column_start=int(meta.get("column_start", 0)), column_end=int(meta.get("column_end", 0)),
                    block_ids=block_ids, line_ids_all=line_ids_all, line_text_map=line_text_map, line_bbox_map=line_bbox_map
                ))

        all_cands.sort(key=lambda r: r.score, reverse=True)
        return all_cands

    def retrieve(self, query: str, include_docs: List[str], top_k_after_mmr: int = 6, bm25_weight: float = 0.35, faiss_topk_per_doc: int = 12) -> List[Hit]:
        cands = self._hybrid_candidates(query, include_docs=include_docs, bm25_weight=bm25_weight, faiss_topk_per_doc=faiss_topk_per_doc)
        if not cands:
            return []

        texts = [c.text for c in cands]
        mat = self.embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        qvec = self.embed.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # MMR
        if mat.shape[0] == 0:
            return []
        sim_to_q = mat @ qvec
        selected: List[int] = []
        candidates = list(range(mat.shape[0]))
        while candidates and len(selected) < min(top_k_after_mmr, len(cands)):
            if not selected:
                i = int(np.argmax(sim_to_q[candidates]))
                selected.append(candidates.pop(i))
                continue
            div = np.max(mat[candidates] @ mat[selected].T, axis=1)
            mmr_scores = 0.7 * sim_to_q[candidates] - 0.3 * div
            i = int(np.argmax(mmr_scores))
            selected.append(candidates.pop(i))

        mmr_cands = [cands[i] for i in selected]

        # optional cross-encoder re-rank
        if self.use_xenc and len(mmr_cands) > 1 and self.xenc is not None:
            pairs = [(query, c.text) for c in mmr_cands]
            scores = self.xenc.predict(pairs)
            order = np.argsort(scores)[::-1]
            mmr_cands = [mmr_cands[i] for i in order]

        hits: List[Hit] = []
        for c in mmr_cands:
            hits.append(Hit(
                text=c.text,
                doc=c.doc,
                id=c.id,
                score=float(c.score),
                citation={
                    "file": c.file,
                    "pages": f"{c.page_start}–{c.page_end}",
                    "columns": f"{c.column_start}–{c.column_end}",
                    "block_ids": c.block_ids,
                    "line_ids_all": c.line_ids_all,
                    "line_text_map": c.line_text_map,
                    "line_bbox_map": c.line_bbox_map,
                },
            ))
        return hits

# -------------------- Query expansion --------------------

def generate_abstract_queries(
    seed: str,
    *,
    n: int = 5,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    extra_instructions: Optional[str] = None,
) -> List[str]:
    """LLM-based; falls back to simple variants if LLM not available."""
    if not _HAS_AZURE_OPENAI and not deployment:
        # fallback: simple variants
        base = seed.strip()
        words = [w for w in re.split(r'\W+', base) if w]
        short = " ".join(words[:min(10, len(words))])
        return list(dict.fromkeys([
            base,
            short,
            f"{short} evidence",
            f"{short} results",
            f"{short} benchmark"
        ]))[:n]

    llm = ChatLLM(deployment=deployment, api_version=api_version)
    sys_msg = (
        "You create high-recall verification queries for fact checking a single claim. "
        f"Produce EXACTLY {n} diverse, precise queries (≤16 words each). "
        'Return ONLY JSON: {"queries": ["...","...","...","...","..."]}.'
    )
    if extra_instructions:
        sys_msg += f" Extra: {extra_instructions.strip()}"

    content = llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": seed.strip()},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    obj = _json_loads_relaxed(content) or {}
    queries = [str(x).strip() for x in obj.get("queries", []) if str(x).strip()]
    if not queries:
        return [seed] * n
    # de-dup and cap
    out, seen = [], set()
    for q in queries:
        if q not in seen:
            out.append(q); seen.add(q)
        if len(out) >= n: break
    while len(out) < n:
        out.append(seed)
    return out[:n]

# -------------------- Evidence selection & validation --------------------

def _evidence_lines_from_hit(hit: Hit, max_blocks: int = 1, max_lines: int = 6) -> Tuple[List[str], List[str], str]:
    """
    Return (line_ids, line_texts, block_id_used)
    """
    block_ids = hit.citation.get("block_ids") or []
    line_ids_all: List[str] = hit.citation.get("line_ids_all") or []
    line_text_map: Dict[str, str] = hit.citation.get("line_text_map") or {}

    chosen_block = block_ids[0] if block_ids else ""
    # Take a slice of consecutive lines as snippet
    line_ids = line_ids_all[:max_lines]
    texts = [line_text_map.get(lid, "") for lid in line_ids]
    return line_ids, texts, chosen_block

def validate_candidates_with_llm(
    claim: str,
    candidates: List[Dict[str, Any]],
    *,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
) -> List[Dict[str, Any]]:
    """
    Ask LLM to pick which candidate snippets support the claim.
    Returns list of items: {"idx": int, "why": str}
    """
    if not _HAS_AZURE_OPENAI and not deployment:
        # Fallback: naive keep top-2
        out = []
        for c in candidates[:2]:
            out.append({"idx": c["idx"], "why": "Top-ranked lexical/semantic match to the claim."})
        return out

    llm = ChatLLM(deployment=deployment, api_version=api_version)
    sys_msg = (
        "You are verifying which snippets support a claim. "
        "Choose ONLY those that directly support the key factual content (not tangential). "
        'Return STRICT JSON: {"supported":[{"idx": int, "why": str}, ...]} (no markdown).'
    )
    user_obj = {"claim": claim, "candidates": [{"idx": c["idx"], "text": c["text"]} for c in candidates]}
    content = llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        temperature=0.0, top_p=0.9, max_tokens=500, response_format={"type": "json_object"},
    )
    obj = _json_loads_relaxed(content) or {}
    supported = obj.get("supported", [])
    out = []
    for it in supported:
        try:
            out.append({"idx": int(it["idx"]), "why": str(it.get("why","")).strip()})
        except Exception:
            pass
    return out

# -------------------- Main annotate pipeline --------------------

def annotate_paragraph(
    paragraph: str,
    include_docs: List[str],
    *,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    per_sentence_queries: int = 5,
    per_query_topk: int = 6,
    candidate_consider: int = 6,
    max_sources_per_sentence: int = 3,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "annotated_paragraph": "... [1,2] ...",
        "sentence_citations": [{"sentence_index": 0, "source_ids": [1,2]} ...],
        "sources": [{"source_id":1, "doc":..., "file":..., "page":..., "block_id":..., "line_ids":[...], "address":..., "snippet":..., "why_used":...}],
        "notes": "..."
      }
    """
    engine = RAGEngine()
    if not include_docs:
        include_docs = engine.list_docs()

    sentences = _split_sentences(paragraph)
    if not sentences:
        return {"annotated_paragraph": paragraph, "sentence_citations": [], "sources": [], "notes": "No sentences found."}

    # Global source registry (dedupe across sentences)
    sources: List[Dict[str, Any]] = []
    source_key_to_id: Dict[str, int] = {}  # key: f"{doc}::{id}::{first_line}"
    next_id = 1

    sentence_citations: List[Dict[str, Any]] = []
    annotated_parts: List[str] = []

    for sidx, sent in enumerate(sentences):
        needs = _needs_citation(sent)
        if not needs:
            annotated_parts.append(sent)
            continue

        # 1) Generate abstract queries for the sentence (claim)
        queries = generate_abstract_queries(
            sent,
            n=per_sentence_queries,
            deployment=deployment,
            api_version=api_version,
            extra_instructions="The goal is to verify the claim precisely; include likely terminology and synonyms.",
        )

        # 2) Pool hits across queries (keep best unique by (doc,id))
        pooled: Dict[Tuple[str, str], Hit] = {}
        for q in queries:
            hits = engine.retrieve(q, include_docs=include_docs, top_k_after_mmr=per_query_topk)
            for h in hits:
                key = (h.doc, h.id)
                if key not in pooled or h.score > pooled[key].score:
                    pooled[key] = h

        cand_hits = sorted(pooled.values(), key=lambda x: x.score, reverse=True)[:max(candidate_consider, max_sources_per_sentence)]
        if not cand_hits:
            annotated_parts.append(_append_refs(sent, []))
            continue

        # 3) Prepare candidates for LLM validation
        candidates: List[Dict[str, Any]] = []
        for idx, h in enumerate(cand_hits):
            line_ids, line_texts, blk = _evidence_lines_from_hit(h, max_blocks=1, max_lines=4)
            snippet = " ".join([t for t in line_texts if t]).strip()
            if not snippet:
                snippet = h.text.replace("\n", " ")[:400]
            candidates.append({
                "idx": idx,
                "hit": h,
                "text": snippet,
                "block_id": blk,
                "line_ids": line_ids
            })

        # 4) Ask LLM which candidates support the claim
        supported = validate_candidates_with_llm(sent, candidates, deployment=deployment, api_version=api_version)
        # keep top-N in original candidate order
        supported_idxs = {x["idx"]: (x.get("why","").strip() or "Supports the claim.") for x in supported}
        chosen: List[int] = []
        for c in candidates:
            if c["idx"] in supported_idxs:
                chosen.append(c["idx"])
            if len(chosen) >= max_sources_per_sentence:
                break

        # if LLM returns nothing, fallback to top-1/2
        if not chosen:
            chosen = list(range(min(max_sources_per_sentence, len(candidates))))
            for i in chosen:
                supported_idxs.setdefault(i, "Top-ranked match used as fallback.")

        # 5) Assign source_ids and build mapping for this sentence
        this_sentence_source_ids: List[int] = []
        for i in chosen:
            c = candidates[i]
            h = c["hit"]
            # use first line id (if any) to pinpoint address
            first_line = c["line_ids"][0] if c["line_ids"] else ""
            skey = f"{h.doc}::{h.id}::{first_line}"
            if skey not in source_key_to_id:
                sid = next_id
                source_key_to_id[skey] = sid
                next_id += 1
                src = {
                    "source_id": sid,
                    "doc": h.doc,
                    "file": h.citation.get("file",""),
                    "page": h.citation.get("pages",""),
                    "block_id": c["block_id"],
                    "line_ids": c["line_ids"],
                    "address": f"{h.citation.get('file','')}::{h.citation.get('pages','')}::{c['block_id']}",
                    "snippet": c["text"][:500],
                    "why_used": supported_idxs.get(i, ""),
                }
                sources.append(src)
            this_sentence_source_ids.append(source_key_to_id[skey])

        # 6) Insert citations into the sentence
        this_sentence_source_ids = sorted(set(this_sentence_source_ids))
        annotated_parts.append(_append_refs(sent, this_sentence_source_ids))
        sentence_citations.append({"sentence_index": sidx, "source_ids": this_sentence_source_ids})

    annotated_paragraph = " ".join(_fix_spacing(annotated_parts))

    return {
        "annotated_paragraph": annotated_paragraph,
        "sentence_citations": sentence_citations,
        "sources": sorted(sources, key=lambda s: s["source_id"]),
        "notes": "Citations added based on heuristic claim detection and LLM validation over RAG results."
    }

def _append_refs(sentence: str, ids: List[int]) -> str:
    if not ids:
        return sentence
    ids_str = ", ".join(str(i) for i in ids)
    # place before trailing punctuation if present
    m = re.search(r'([.!?])\s*$', sentence)
    if m:
        start = m.start(1)
        return sentence[:start] + f" [{ids_str}]" + sentence[start:]
    else:
        return sentence.rstrip() + f" [{ids_str}]"

def _fix_spacing(parts: List[str]) -> List[str]:
    # Clean up spaces after joining sentences split earlier
    out: List[str] = []
    for p in parts:
        out.append(re.sub(r'\s+', ' ', p).strip())
    return out

# -------------------- CLI --------------------

def _read_stdin_if_needed() -> Optional[str]:
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data and data.strip():
            return data
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser(description="RAG-Reference: auto-insert citations into a paragraph using local indices + LLM validation.")
    ap.add_argument("--text", type=str, default=None, help="Paragraph text. If omitted, reads stdin.")
    ap.add_argument("--docs", nargs="*", default=None, help="Docstems to search (default: all available).")
    ap.add_argument("--deployment", type=str, default=None, help="Azure OpenAI deployment (e.g., gpt-4o).")
    ap.add_argument("--api-version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version.")
    ap.add_argument("--per-sentence-queries", type=int, default=5, help="Abstract queries per sentence.")
    ap.add_argument("--per-query-topk", type=int, default=6, help="Top-k per query before pooling.")
    ap.add_argument("--candidate-consider", type=int, default=6, help="How many pooled candidates to validate per sentence.")
    ap.add_argument("--max-sources-per-sentence", type=int, default=3, help="Upper bound on citations per sentence.")
    args = ap.parse_args()

    text = args.text or _read_stdin_if_needed()
    if not text:
        print("No input text. Use --text or pipe stdin.", file=sys.stderr)
        sys.exit(2)

    try:
        out = annotate_paragraph(
            text,
            include_docs=args.docs or RAGEngine().list_docs(),
            deployment=args.deployment,
            api_version=args.api_version,
            per_sentence_queries=args.per_sentence_queries,
            per_query_topk=args.per_query_topk,
            candidate_consider=args.candidate_consider,
            max_sources_per_sentence=args.max_sources_per_sentence,
        )
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
