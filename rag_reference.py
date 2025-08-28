# rag_reference.py
# ------------------------------------------------------------
# pip install sentence-transformers faiss-cpu python-dotenv openai
# (optional) pip install rank-bm25
# (optional) pip install cross-encoder
#
# CLI:
#   python rag_reference.py --text "Your paragraph here." --docs paperA paperB --deployment gpt-4o --trace
#   cat story.txt | python rag_reference.py --docs paperA paperB --deployment gpt-4o --trace
#
# Output JSON keys (block-first):
#   - annotated_paragraph: str
#   - block_citations: [{
#        block_id: str,                        # inferred from supporting hit
#        neighbor_block_ids: [prev, next],     # may include "" if not present
#        source_ids: [int,...],
#        query_transforms: [str,...]
#     }]
#   - query_transforms_by_block: [{block_id, transforms:[...]}]
#   - sources: [ CHOSEN sources only, with neighbors ]
#   - retrieved_sources: [ ALL retrieved (pre-validation) with neighbors ]
#   - sentence_citations: [{sentence_index, source_ids:[int,...]}]   # legacy/back-compat
#   - citation_decisions: [{sentence_index, needs: bool, reason: str, category: str}]
#   - notes: str
#   - trace: [ ... ]                         # when --trace is set
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

# -------------------- Tracing --------------------

class Tracer:
    def __init__(self, enabled: bool = False, to_stderr: bool = True):
        self.enabled = enabled
        self.to_stderr = to_stderr
        self.events: List[Dict[str, Any]] = []

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        rec = {"event": event, **fields}
        self.events.append(rec)
        if self.to_stderr:
            try:
                print(f"[TRACE] {event}: {json.dumps(fields, ensure_ascii=False)[:1200]}", file=sys.stderr)
            except Exception:
                pass

    def dump(self) -> List[Dict[str, Any]]:
        return list(self.events)

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

_SENT_SPLIT = re.compile(r'(?<!\b[A-Z])[.!?](?=\s+|$)')

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
    return [p + d for p, d in sents]

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
        tracer: Optional[Tracer] = None,
        provider: str = "azure",
        deployment: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        self.tracer = tracer or Tracer(False)
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

    def chat(self, messages: List[Dict[str, str]], name: str = "llm_call", **kwargs) -> str:
        self.tracer.log(
            "llm_request",
            name=name,
            deployment=self.deployment,
            api_version=self.api_version,
            messages=messages,
            params={k: v for k, v in kwargs.items() if k != "api_key"}
        )
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.deployment,
            **kwargs,
        )
        content = (resp.choices[0].message.content or "").strip()
        self.tracer.log("llm_response", name=name, content_preview=content[:2000], raw_length=len(content))
        return content

# -------------------- RAG Engine --------------------

class RAGEngine:
    def __init__(self, tracer: Optional[Tracer] = None, embed_model_name: str = DEFAULT_EMBED, use_cross_encoder: bool = False, cross_encoder_name: str = DEFAULT_RERANKER):
        self.tracer = tracer or Tracer(False)
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

            sims, idxs = index.search(np.array([q_vec]), faiss_topk_per_doc)
            sem_pairs: List[Tuple[str, float]] = []
            for vec_id, score in zip(idxs[0], sims[0]):
                cid = v2c.get(int(vec_id))
                if cid and cid in chunks:
                    sem_pairs.append((cid, float(score)))

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

                block_ids: List[str] = meta.get("block_ids", []) or []
                line_ids_all: List[str] = meta.get("line_ids", []) or []
                line_text_map: Dict[str, str] = meta.get("line_text_map") or {}
                line_bbox_map: Dict[str, Optional[List[float]]] = meta.get("line_bbox_map") or {}

                if not line_text_map and line_ids_all:
                    lines = rec["text"].split("\n")
                    m = min(len(lines), len(line_ids_all))
                    line_text_map = {lid: ln for lid, ln in zip(line_ids_all[:m], lines[:m])}

                for lid in line_ids_all:
                    if lid not in line_bbox_map:
                        line_bbox_map[lid] = None

                all_cands.append(RetrievedInternal(
                    doc=ds, id=cid, text=rec["text"], score=float(fused),
                    file=meta.get("file",""), page_start=int(meta.get("page_start", 0)), page_end=int(meta.get("page_end", 0)),
                    column_start=int(meta.get("column_start", 0)), column_end=int(meta.get("column_end", 0)),
                    block_ids=block_ids, line_ids_all=line_ids_all, line_text_map=line_text_map, line_bbox_map=line_bbox_map
                ))

        all_cands.sort(key=lambda r: r.score, reverse=True)
        return all_cands

    def retrieve(self, query: str, include_docs: List[str], top_k_after_mmr: int = 6, bm25_weight: float = 0.35, faiss_topk_per_doc: int = 12) -> List[Hit]:
        self.tracer.log("retrieve_start", query=query, docs=include_docs, top_k_after_mmr=top_k_after_mmr)
        cands = self._hybrid_candidates(query, include_docs=include_docs, bm25_weight=bm25_weight, faiss_topk_per_doc=faiss_topk_per_doc)
        if not cands:
            self.tracer.log("retrieve_end", query=query, hits=[])
            return []

        texts = [c.text for c in cands]
        mat = self.embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        qvec = self.embed.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        if mat.shape[0] == 0:
            self.tracer.log("retrieve_end", query=query, hits=[])
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
        self.tracer.log("retrieve_end", query=query, hits=[{"doc": h.doc, "chunk_id": h.id, "score": h.score, "file": h.citation.get("file",""), "pages": h.citation.get("pages","")} for h in hits])
        return hits

# -------------------- LLM: Research-Supervisor Citation Decision --------------------

_SUPERVISOR_POLICY = (
    "You are a research supervisor enforcing academic citation standards. "
    "For each sentence, decide if an external citation is REQUIRED. "
    "Require citation for: empirical or quantitative claims; statements about benchmarks, SOTA, or performance; "
    "specific methods/datasets/results; stats, counts, comparisons, timelines; factual assertions not common knowledge; "
    "claims about impact or prevalence; domain-specific assertions that would normally be referenced. "
    "Do NOT require citation for: author opinions, writing glue, obvious/common-knowledge facts, task intent, or purely stylistic content. "
    'Return STRICT JSON: {"decisions":[{"index":int,"needs":true|false,"reason":str,"category":str}], "notes":str}'
)

def supervisor_decide_needs_citation(
    sentences: List[str],
    *,
    tracer: Tracer,
    deployment: Optional[str],
    api_version: str = "2024-12-01-preview",
) -> List[Dict[str, Any]]:
    """Batch classify sentences with Azure OpenAI; fallback: cite if sentence is long or looks claim-y."""
    if not _HAS_AZURE_OPENAI or not deployment:
        # conservative fallback: flag most content-rich sentences
        dec = []
        for i, s in enumerate(sentences):
            tok = len(re.findall(r"\w+", s))
            has_num = bool(re.search(r"\d|%|20\d{2}", s))
            looks_claimy = any(t in s.lower() for t in ["improve", "increase", "decrease", "outperform", "accuracy", "benchmark", "study", "report", "dataset"])
            need = has_num or tok >= 16 or looks_claimy
            dec.append({"index": i, "needs": bool(need), "reason": "fallback_heuristic", "category": "fallback"})
        tracer.log("supervisor_decision_fallback", decisions=dec[:20])
        return dec

    llm = ChatLLM(tracer=tracer, deployment=deployment, api_version=api_version)
    rows = [{"index": i, "text": s} for i, s in enumerate(sentences)]
    sys_msg = _SUPERVISOR_POLICY
    user_msg = json.dumps({"sentences": rows}, ensure_ascii=False)

    content = llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        name="needs_citation_supervisor",
        temperature=0.0,
        top_p=0.9,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    obj = _json_loads_relaxed(content) or {}
    decisions = obj.get("decisions", [])
    out: List[Dict[str, Any]] = []
    # sanitize and ensure coverage
    index_set = {d.get("index") for d in decisions if isinstance(d, dict)}
    for i, s in enumerate(sentences):
        if i in index_set:
            d = [x for x in decisions if x.get("index") == i][0]
            out.append({
                "index": i,
                "needs": bool(d.get("needs", False)),
                "reason": str(d.get("reason", "") or ""),
                "category": str(d.get("category", "") or "")
            })
        else:
            out.append({"index": i, "needs": False, "reason": "not_listed_by_llm", "category": ""})
    tracer.log("supervisor_decision_result", sample=out[:20])
    return out

# -------------------- Query transform --------------------

def query_transform(
    claim_with_context: str,
    *,
    tracer: Tracer,
    n: int = 5,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    extra_instructions: Optional[str] = None,
) -> List[str]:
    if not _HAS_AZURE_OPENAI or not deployment:
        base = claim_with_context.strip()
        words = [w for w in re.split(r'\W+', base) if w]
        short = " ".join(words[:min(12, len(words))])
        queries = list(dict.fromkeys([
            short,
            f"{short} evidence",
            f"{short} results",
            f"{short} benchmark",
            f"{short} dataset"
        ]))[:n]
        tracer.log("query_transform_fallback", claim=claim_with_context, queries=queries)
        return queries

    llm = ChatLLM(tracer=tracer, deployment=deployment, api_version=api_version)
    sys_msg = (
        "You act as a meticulous research supervisor creating high-recall verification queries for fact-checking one claim WITH its local context. "
        f"Produce EXACTLY {n} diverse, precise queries (≤16 words each). "
        "Expand key terms with synonyms, dataset/task names, metrics, and likely section headers. "
        'Return ONLY JSON: {"queries": ["...","...","...","...","..."]}.'
    )
    if extra_instructions:
        sys_msg += f" Extra: {extra_instructions.strip()}"

    content = llm.chat(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": claim_with_context.strip()},
        ],
        name="query_transform",
        temperature=0.2,
        top_p=0.95,
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    obj = _json_loads_relaxed(content) or {}
    queries = [str(x).strip() for x in obj.get("queries", []) if str(x).strip()]
    tracer.log("query_transform_result", claim=claim_with_context, queries=queries)
    if not queries:
        return [claim_with_context] * n
    out, seen = [], set()
    for q in queries:
        if q not in seen:
            out.append(q); seen.add(q)
        if len(out) >= n: break
    while len(out) < n:
        out.append(out[-1])
    return out[:n]

# -------------------- Evidence utilities --------------------

def _evidence_lines_from_hit(hit: Hit, max_lines: int = 6) -> Tuple[List[str], List[str], str, List[str]]:
    block_ids = hit.citation.get("block_ids") or []
    chosen_block = block_ids[0] if block_ids else ""
    prev_blk, next_blk = "", ""
    if block_ids:
        idx = 0
        try:
            idx = block_ids.index(chosen_block)
        except Exception:
            idx = 0
        if idx - 1 >= 0:
            prev_blk = block_ids[idx - 1]
        if idx + 1 < len(block_ids):
            next_blk = block_ids[idx + 1]

    line_ids_all: List[str] = hit.citation.get("line_ids_all") or []
    line_text_map: Dict[str, str] = hit.citation.get("line_text_map") or {}
    line_ids = line_ids_all[:max_lines]
    texts = [line_text_map.get(lid, "") for lid in line_ids]
    return line_ids, texts, chosen_block, [prev_blk, next_blk]

def _register_source(
    sources: List[Dict[str, Any]],
    source_key_to_id: Dict[str, int],
    next_id: int,
    *,
    hit: Hit,
    snippet_text: str,
    block_id: str,
    neighbor_block_ids: List[str],
    line_ids: List[str],
    why_used: str,
) -> Tuple[int, int]:
    first_line = line_ids[0] if line_ids else ""
    skey = f"{hit.doc}::{hit.id}::{block_id}::{first_line}"
    if skey not in source_key_to_id:
        sid = next_id
        source_key_to_id[skey] = sid
        next_id += 1
        src = {
            "source_id": sid,
            "doc": hit.doc,
            "chunk_id": hit.id,
            "file": hit.citation.get("file",""),
            "page": hit.citation.get("pages",""),
            "block_id": block_id,
            "neighbor_block_ids": neighbor_block_ids,  # [prev, next]
            "line_ids": line_ids,
            "address": f"{hit.citation.get('file','')}::{hit.citation.get('pages','')}::{block_id}",
            "snippet": (snippet_text or hit.text.replace("\n"," "))[:600],
            "why_used": why_used,
        }
        sources.append(src)
    return source_key_to_id[skey], next_id

def validate_candidates_with_llm(
    claim: str,
    candidates: List[Dict[str, Any]],
    *,
    tracer: Tracer,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
) -> List[Dict[str, Any]]:
    if not _HAS_AZURE_OPENAI or not deployment:
        out = []
        for c in candidates[:2]:
            out.append({"idx": c["idx"], "why": "Top-ranked lexical/semantic match to the claim."})
        tracer.log("validation_fallback", claim=claim, chosen=[x["idx"] for x in out])
        return out

    llm = ChatLLM(tracer=tracer, deployment=deployment, api_version=api_version)
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
        name="evidence_validation",
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
    tracer.log("validation_result", claim=claim, chosen=[x["idx"] for x in out])
    return out

# -------------------- Annotator (block-first) --------------------

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
    trace: bool = False,
) -> Dict[str, Any]:
    tracer = Tracer(enabled=trace, to_stderr=True)
    engine = RAGEngine(tracer=tracer)
    if not include_docs:
        include_docs = engine.list_docs()

    sentences = _split_sentences(paragraph)
    tracer.log("sentence_split", count=len(sentences), sentences=sentences[:20])

    if not sentences:
        out = {"annotated_paragraph": paragraph, "sentence_citations": [], "block_citations": [], "sources": [], "retrieved_sources": [], "citation_decisions": [], "notes": "No sentences found."}
        if trace:
            out["trace"] = tracer.dump()
        return out

    # --- NEW: ask Azure (research supervisor) which sentences need citations ---
    decisions = supervisor_decide_needs_citation(
        sentences, tracer=tracer, deployment=deployment, api_version=api_version
    )
    needs_map: Dict[int, Dict[str, Any]] = {d["index"]: d for d in decisions}

    # Outputs
    sources: List[Dict[str, Any]] = []                 # chosen only
    retrieved_sources: List[Dict[str, Any]] = []        # ALL retrieved candidates (pre-validation)
    source_key_to_id: Dict[str, int] = {}
    next_id = 1

    sentence_citations: List[Dict[str, Any]] = []       # legacy
    block_citations: List[Dict[str, Any]] = []          # canonical
    query_transforms_by_block: List[Dict[str, Any]] = []

    annotated_parts: List[str] = []

    def _retrieved_key(doc: str, chunk_id: str, block_id: str) -> str:
        return f"{doc}::{chunk_id}::{block_id}"

    seen_retrieved: set[str] = set()

    for sidx, sent in enumerate(sentences):
        need = bool(needs_map.get(sidx, {}).get("needs", False))
        tracer.log("needs_citation_decision", sentence_index=sidx, sentence=sent, needs=need, reason=needs_map.get(sidx, {}).get("reason",""), category=needs_map.get(sidx, {}).get("category",""))
        if not need:
            annotated_parts.append(sent)
            continue

        # Build context block: sentence + prev + next
        prev_s = sentences[sidx - 1] if sidx - 1 >= 0 else ""
        next_s = sentences[sidx + 1] if sidx + 1 < len(sentences) else ""
        claim_with_context = f"CLAIM: {sent.strip()}\nCONTEXT:\nPrev: {prev_s.strip()}\nNext: {next_s.strip()}"

        # 1) Query transform (supervisor style)
        queries = query_transform(
            claim_with_context,
            tracer=tracer,
            n=per_sentence_queries,
            deployment=deployment,
            api_version=api_version,
            extra_instructions="Leverage the claim + context; prefer including dataset/metric/task names; avoid overly long queries.",
        )

        # 2) Retrieve per query (pool)
        pooled: Dict[Tuple[str, str], Hit] = {}
        for q in queries:
            hits = engine.retrieve(q, include_docs=include_docs, top_k_after_mmr=per_query_topk)
            for h in hits:
                key = (h.doc, h.id)
                if key not in pooled or h.score > pooled[key].score:
                    pooled[key] = h

                # register ALL retrieved (pre-validation) with neighbors
                line_ids, line_texts, blk, neighbors = _evidence_lines_from_hit(h, max_lines=4)
                snippet = " ".join([t for t in line_texts if t]).strip() or h.text.replace("\n"," ")[:400]
                if blk:
                    rk = _retrieved_key(h.doc, h.id, blk)
                    if rk not in seen_retrieved:
                        seen_retrieved.add(rk)
                        retrieved_sources.append({
                            "doc": h.doc,
                            "chunk_id": h.id,
                            "file": h.citation.get("file",""),
                            "page": h.citation.get("pages",""),
                            "block_id": blk,
                            "neighbor_block_ids": neighbors,
                            "line_ids": line_ids,
                            "address": f"{h.citation.get('file','')}::{h.citation.get('pages','')}::{blk}",
                            "snippet": snippet[:600],
                            "why_used": "Retrieved candidate (pre-validation)."
                        })

        cand_hits = sorted(pooled.values(), key=lambda x: x.score, reverse=True)[:max(candidate_consider, max_sources_per_sentence)]
        tracer.log("pooled_hits", sentence_index=sidx, queries=queries, candidate_count=len(cand_hits),
                   hits=[{"doc": h.doc, "chunk_id": h.id, "score": h.score} for h in cand_hits])

        if not cand_hits:
            annotated_parts.append(_append_refs(sent, []))
            block_citations.append({"block_id": "", "neighbor_block_ids": ["",""], "source_ids": [], "query_transforms": queries})
            query_transforms_by_block.append({"block_id": "", "transforms": queries})
            continue

        # 3) Build candidate snippets
        candidates: List[Dict[str, Any]] = []
        for idx, h in enumerate(cand_hits):
            line_ids, line_texts, blk, neighbors = _evidence_lines_from_hit(h, max_lines=4)
            snippet = " ".join([t for t in line_texts if t]).strip() or h.text.replace("\n"," ")[:400]
            candidates.append({
                "idx": idx,
                "hit": h,
                "text": snippet,
                "block_id": blk,
                "neighbor_block_ids": neighbors,
                "line_ids": line_ids
            })
        tracer.log("candidates_built", sentence_index=sidx, candidates=[{"idx": c["idx"], "doc": c["hit"].doc, "chunk_id": c["hit"].id, "block_id": c["block_id"]} for c in candidates])

        # 4) Validate with LLM
        supported = validate_candidates_with_llm(sent, candidates, tracer=tracer, deployment=deployment, api_version=api_version)
        supported_map = {x["idx"]: (x.get("why","").strip() or "Supports the claim.") for x in supported}
        chosen: List[int] = []
        for c in candidates:
            if c["idx"] in supported_map:
                chosen.append(c["idx"])
            if len(chosen) >= max_sources_per_sentence:
                break
        if not chosen:
            chosen = list(range(min(max_sources_per_sentence, len(candidates))))
            for i in chosen:
                supported_map.setdefault(i, "Top-ranked match used as fallback.")
        tracer.log("chosen_candidates", sentence_index=sidx, chosen=chosen)

        # 5) Assign source IDs & register (chosen)
        this_sentence_source_ids: List[int] = []
        unit_block_id = ""
        unit_neighbors = ["",""]
        for i in chosen:
            c = candidates[i]
            h = c["hit"]
            sid, next_id = _register_source(
                sources, source_key_to_id, next_id,
                hit=h,
                snippet_text=c["text"],
                block_id=c["block_id"],
                neighbor_block_ids=c["neighbor_block_ids"],
                line_ids=c["line_ids"],
                why_used=supported_map.get(i, ""),
            )
            this_sentence_source_ids.append(sid)
            if not unit_block_id and c["block_id"]:
                unit_block_id = c["block_id"]
                unit_neighbors = c["neighbor_block_ids"]

        this_sentence_source_ids = sorted(set(this_sentence_source_ids))
        annotated_parts.append(_append_refs(sent, this_sentence_source_ids))
        sentence_citations.append({"sentence_index": sidx, "source_ids": this_sentence_source_ids})

        # Record block-first structure
        block_citations.append({
            "block_id": unit_block_id,
            "neighbor_block_ids": unit_neighbors,
            "source_ids": this_sentence_source_ids,
            "query_transforms": queries
        })
        query_transforms_by_block.append({"block_id": unit_block_id, "transforms": queries})

    annotated_paragraph = " ".join(_fix_spacing(annotated_parts))
    out: Dict[str, Any] = {
        "annotated_paragraph": annotated_paragraph,
        "block_citations": block_citations,
        "query_transforms_by_block": query_transforms_by_block,
        "sentence_citations": sentence_citations,  # legacy/back-compat
        "sources": sorted(sources, key=lambda s: s["source_id"]),
        "retrieved_sources": retrieved_sources,     # exhaustive list of what was looked at
        "citation_decisions": [{"sentence_index": d["index"], "needs": d["needs"], "reason": d.get("reason",""), "category": d.get("category","")} for d in decisions],
        "notes": "Supervisor-LLM decision for citation need + query transform + hybrid retrieval + LLM validation. Outputs are block-first with neighbor context.",
    }
    if trace:
        out["trace"] = tracer.dump()
    return out

# -------------------- formatting helpers --------------------

def _append_refs(sentence: str, ids: List[int]) -> str:
    if not ids:
        return sentence
    ids_str = ", ".join(str(i) for i in ids)
    m = re.search(r'([.!?])\s*$', sentence)
    if m:
        start = m.start(1)
        return sentence[:start] + f" [{ids_str}]" + sentence[start:]
    else:
        return sentence.rstrip() + f" [{ids_str}]"

def _fix_spacing(parts: List[str]) -> List[str]:
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
    ap = argparse.ArgumentParser(description="RAG-Reference (block-first): Azure-supervised citation decisions + query transform + retrieval + validation.")
    ap.add_argument("--text", type=str, default=None, help="Paragraph text. If omitted, reads stdin.")
    ap.add_argument("--docs", nargs="*", default=None, help="Docstems to search (default: all available).")
    ap.add_argument("--deployment", type=str, default=None, help="Azure OpenAI deployment (e.g., gpt-4o).")
    ap.add_argument("--api-version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version.")
    ap.add_argument("--per-sentence-queries", type=int, default=5, help="Abstract queries per sentence needing citation.")
    ap.add_argument("--per-query-topk", type=int, default=6, help="Top-k per query before pooling.")
    ap.add_argument("--candidate-consider", type=int, default=6, help="How many pooled candidates to validate per sentence.")
    ap.add_argument("--max-sources-per-sentence", type=int, default=3, help="Upper bound on citations per sentence.")
    ap.add_argument("--trace", action="store_true", help="Include detailed trace (plus stderr prints).")
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
            trace=args.trace,
        )
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
