# rag_lookup.py
# ------------------------------------------------------------
# pip install sentence-transformers faiss-cpu python-dotenv openai
# (optional) pip install rank-bm25
# (optional) pip install cross-encoder
#
# CLI examples:
#   python rag_lookup.py "What is temperature scaling?" --docs paperA paperB --topk 6
#   python rag_lookup.py "label smoothing vs temperature scaling" --expand --deployment gpt-4o
#
# Programmatic:
#   from rag_lookup import RAGLookup, generate_abstract_queries
#   eng = RAGLookup(use_cross_encoder=False)
#   docs = eng.list_docs()[:2]
#   queries, per_query_hits, fused = eng.retrieve_with_llm_queries(
#       "label smoothing vs temperature scaling", include_docs=docs
#   )
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from typing import Iterable
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Optional dependencies (safe import) ---
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

# --- LLM SDK (Azure) ---
try:
    # openai>=1.0
    from openai import AzureOpenAI  # type: ignore
    _HAS_AZURE_OPENAI = True
except Exception:
    _HAS_AZURE_OPENAI = False

ARTIFACTS_DIR = Path("artifacts")
INDICES_DIR = ARTIFACTS_DIR / "indices"
DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small & fast

# -------------------- IO helpers --------------------

def _doc_paths(docstem: str):
    d = INDICES_DIR / docstem
    return {
        "dir": d,
        "chunks": d / "chunks.jsonl",
        "index": d / "index.faiss",
        "docstore": d / "docstore.jsonl",
    }

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

def _evidence_from_hit(
    hit: Hit,
    *,
    max_blocks: int = 3,
    max_lines_per_block: int = 12
) -> List[EvidenceItem]:
    """
    Select up to `max_blocks` distinct block_ids from a Hit and include their lines.
    Falls back to line_ids_all order if block grouping can't be inferred.
    """
    block_ids = hit.citation.get("block_ids") or []
    line_ids_all: List[str] = hit.citation.get("line_ids_all") or []
    line_text_map: Dict[str, str] = hit.citation.get("line_text_map") or {}
    line_bbox_map: Dict[str, Optional[List[float]]] = hit.citation.get("line_bbox_map") or {}

    # If we have block_ids, try to keep lines grouped by block sequence
    chosen_blocks = block_ids[:max_blocks] if block_ids else []
    ev: List[EvidenceItem] = []

    def _add_line(lid: str, blk: str = ""):
        ev.append(
            EvidenceItem(
                doc=hit.doc,
                id=hit.id,
                file=str(hit.citation.get("file", "")),
                page=str(hit.citation.get("pages", "")),
                block_id=blk,
                line_id=lid,
                line_text=line_text_map.get(lid, "")
            )
        )

    if chosen_blocks:
        # naive mapping: assume line_ids_all are in reading order; slice evenly by blocks
        # (Your ingest could later attach a real line->block map; this stays robust.)
        per_block = max(1, len(line_ids_all) // max(1, len(chosen_blocks)))
        cursor = 0
        for b in chosen_blocks:
            lines = line_ids_all[cursor:cursor+per_block]
            for lid in lines[:max_lines_per_block]:
                _add_line(lid, blk=b)
            cursor += per_block
    else:
        # no block info; just take first few lines
        for lid in line_ids_all[: max_blocks * max_lines_per_block]:
            _add_line(lid, blk="")

    return ev

def _chunked(it: Iterable[Any], size: int) -> Iterable[List[Any]]:
    buf: List[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
        
def answer_question_with_batches(
    question: str,
    hits: List[Hit],
    *,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    max_chunks_per_call: int = 5,
    max_blocks_per_hit: int = 3,
    max_lines_per_block: int = 12,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_tokens_partial: int = 700,
    max_tokens_final: int = 900,
) -> Dict[str, Any]:
    """
    1) Split selected hits into batches (≤ max_chunks_per_call per request).
    2) For each batch: send evidence (capped at ≤3 blocks/hit) and get a *JSON* partial answer with
       per-line 'why_used' reasoning.
    3) Synthesize all partials into a unified *JSON* final answer.
    Returns a JSON-able dict.
    """
    llm = ChatLLM(provider="azure", deployment=deployment, api_version=api_version)

    partials: List[PartialAnswer] = []
    # -------- per-batch requests --------
    for batch_idx, batch in enumerate(_chunked(hits, max_chunks_per_call), start=1):
        # Build evidence payload for this batch
        batch_evidence: List[Dict[str, Any]] = []
        for h in batch:
            ev_items = _evidence_from_hit(h, max_blocks=max_blocks_per_hit, max_lines_per_block=max_lines_per_block)
            # Pack into compact JSON entries for the model
            for it in ev_items:
                batch_evidence.append({
                    "doc": it.doc,
                    "id": it.id,
                    "file": it.file,
                    "page": it.page,
                    "block_id": it.block_id,
                    "line_id": it.line_id,
                    "line_text": it.line_text,
                })

        sys_msg = (
            "You are a careful analyst. Using ONLY the provided evidence lines, answer the user's question.\n"
            "Return STRICT JSON with keys: {\"partial_answer\": str, \"evidence\": [{\"doc\": str, \"id\": str, "
            "\"file\": str, \"page\": str, \"block_id\": str, \"line_id\": str, \"why_used\": str}] }.\n"
            "For each evidence entry you include, attach a brief 'why_used' explanation (1-2 sentences) explaining "
            "how that specific line supports the answer. Do not invent citations. Do not reference any outside info."
        )

        user_payload = {
            "question": question,
            "evidence_lines": batch_evidence
        }

        content = llm.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens_partial,
        )

        # Parse model JSON; be robust to code fences
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            if "```" in content:
                for part in content.split("```"):
                    s = part.strip()
                    if s.lower().startswith("json"):
                        s = s[4:].strip()
                    try:
                        parsed = json.loads(s)
                        break
                    except Exception:
                        pass
        if not parsed or "partial_answer" not in parsed:
            parsed = {"partial_answer": str(content).strip(), "evidence": []}

        ev_with_why: List[EvidenceItem] = []
        for e in parsed.get("evidence", []):
            ev_with_why.append(
                EvidenceItem(
                    doc=e.get("doc",""),
                    id=e.get("id",""),
                    file=e.get("file",""),
                    page=e.get("page",""),
                    block_id=e.get("block_id",""),
                    line_id=e.get("line_id",""),
                    line_text="",  # keep payload small in the final merge; lines already known above
                    why_used=e.get("why_used","")
                )
            )

        partials.append(PartialAnswer(
            batch_index=batch_idx,
            partial_answer=parsed.get("partial_answer", ""),
            evidence=ev_with_why
        ))

    # -------- final synthesis --------
    synthesis_sys = (
        "You are synthesizing several partial answers. Merge them into a single, precise final answer.\n"
        "Return STRICT JSON with keys:\n"
        "{\n"
        "  \"final_answer\": str,\n"
        "  \"merged_evidence\": [ {\"doc\": str, \"id\": str, \"file\": str, \"page\": str, "
        "\"block_id\": str, \"line_id\": str, \"why_used\": str} ],\n"
        "  \"notes\": str\n"
        "}\n"
        "Deduplicate overlapping evidence by (doc,id,line_id). Keep the best 'why_used'. "
        "Do not add new facts beyond the partials."
    )

    synthesis_user = {
        "question": question,
        "partials": [
            {
                "batch_index": p.batch_index,
                "partial_answer": p.partial_answer,
                "evidence": [e.__dict__ for e in p.evidence],
            } for p in partials
        ]
    }

    final_content = llm.chat(
        [
            {"role": "system", "content": synthesis_sys},
            {"role": "user", "content": json.dumps(synthesis_user, ensure_ascii=False)},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_final,
    )

    final_parsed = None
    try:
        final_parsed = json.loads(final_content)
    except Exception:
        if "```" in final_content:
            for part in final_content.split("```"):
                s = part.strip()
                if s.lower().startswith("json"):
                    s = s[4:].strip()
                try:
                    final_parsed = json.loads(s)
                    break
                except Exception:
                    pass
    if not final_parsed or "final_answer" not in final_parsed:
        final_parsed = {
            "final_answer": str(final_content).strip(),
            "merged_evidence": [],
            "notes": "Model returned non-JSON or malformed JSON; captured raw text."
        }

    # Return a compact, capture-everything envelope
    return {
        "question": question,
        "batches": [
            {
                "batch_index": p.batch_index,
                "partial_answer": p.partial_answer,
                "evidence": [e.__dict__ for e in p.evidence]
            } for p in partials
        ],
        "final": final_parsed
    }

def _prepare_answer_paths(answer_out: Optional[str]) -> Tuple[Path, Path]:
    """
    Returns (primary_path, session_path).
    - primary_path: args.answer_out or out/answer.json
    - session_path: out/YYYYMMDD_HHMMSS/<filename>
    Ensures both parent dirs exist.
    """
    base_path = Path(answer_out) if answer_out else Path("out") / "answer.json"

    # ensure base out directory exists
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # session folder with timestamp to the second
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_path.parent / stamp
    session_dir.mkdir(parents=True, exist_ok=True)

    session_path = session_dir / base_path.name
    return base_path, session_path
# -------------------- Provider-agnostic chat wrapper --------------------

class ChatLLM:
    """
    Minimal provider-agnostic chat wrapper.
    Currently implements Azure OpenAI via AzureOpenAI SDK.
    Extend later with OpenAI(), Anthropic(), etc. while keeping the same interface.
    """
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

        # Resolve from env if not passed
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.azure_endpoint = (azure_endpoint or os.environ.get("AZURE_OPENAI_API_TARGET_URI") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview"

        if not (self.deployment and self.azure_endpoint and self.api_key):
            raise RuntimeError(
                "Missing Azure config. Need AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_TARGET_URI, "
                "and a deployment name (arg or AZURE_OPENAI_DEPLOYMENT)."
            )

        # Initialize SDK client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send a chat completion request; returns assistant content string.
        kwargs are passed to client.chat.completions.create (e.g., temperature, max_tokens).
        """
        resp = self.client.chat.completions.create(
            messages=messages,
            model=self.deployment,   # Azure uses 'model' to pass the *deployment name*
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()

# -------------------- LLM query expansion --------------------

def generate_abstract_queries(
    seed_query: str,
    *,
    n: int = 5,
    deployment: Optional[str] = None,          # Azure *deployment name* (e.g., "gpt-4o")
    api_version: str = "2024-12-01-preview",   # default to current preview per Azure docs
    extra_instructions: Optional[str] = None,  # domain hints, doc types, etc.
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 600,
) -> List[str]:
    """
    Use Azure OpenAI (via SDK) to turn a single query into N higher-recall variations.
    Returns a list with exactly up to N non-empty, deduped queries (backfilled by seed if short).
    """
    # Build wrapper (reads env if args not provided)
    llm = ChatLLM(
        provider="azure",
        deployment=deployment,
        api_version=api_version,
    )

    sys_msg = (
        "You are an expert search query reformulator for scholarly/technical retrieval. "
        f"Produce EXACTLY {n} diverse, precise queries that improve recall. "
        "Cover alternate phrasings, abbreviations, synonyms, adjacent concepts, and likely section headers. "
        "Avoid near-duplicates. Each under 16 words. "
        'Return ONLY JSON: {"queries": ["...","...","...","...","..."]}.'
    )
    if extra_instructions:
        sys_msg += f" Additional context: {extra_instructions.strip()}"

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Seed query: {seed_query.strip()}"},
    ]

    try:
        content = llm.chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Bubble up; caller prints a clear Azure error
        raise

    # Parse JSON result safely
    queries: List[str] = []
    try:
        obj = json.loads(content)
        q = obj.get("queries", [])
        if isinstance(q, list):
            queries = [str(s).strip() for s in q if str(s).strip()]
    except Exception:
        # try code fences
        if "```" in content:
            for part in content.split("```"):
                s = part.strip()
                if s.lower().startswith("json"):
                    s = s[4:].strip()
                try:
                    obj = json.loads(s)
                    q = obj.get("queries", [])
                    if isinstance(q, list):
                        queries = [str(x).strip() for x in q if str(x).strip()]
                        break
                except Exception:
                    pass
        if not queries:
            # last-resort: line-based
            lines = [ln.strip("-•\t ") for ln in content.splitlines() if ln.strip()]
            queries = lines[:n]

    # Dedupe and ensure length n
    seen, uniq = set(), []
    for q in queries:
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
        if len(uniq) >= n:
            break
    while len(uniq) < n:
        uniq.append(seed_query)
    return uniq[:n]

# -------------------- Data models --------------------

@dataclass
class Hit:
    text: str
    doc: str
    id: str
    score: float
    citation: Dict[str, Any]  # includes: file/pages/columns/block_ids + line_ids_all + line_text_map + line_bbox_map

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
    
    

@dataclass
class EvidenceItem:
    doc: str
    id: str
    file: str
    page: str
    block_id: str
    line_id: str
    line_text: str
    why_used: Optional[str] = None

@dataclass
class PartialAnswer:
    batch_index: int
    partial_answer: str
    evidence: List[EvidenceItem]

# -------------------- Core RAG engine --------------------

class RAGLookup:
    def __init__(
        self,
        embed_model_name: str = DEFAULT_EMBED,
        use_cross_encoder: bool = False,
        cross_encoder_name: str = DEFAULT_RERANKER,
    ):
        self.embed = SentenceTransformer(embed_model_name)
        self.use_xenc = use_cross_encoder and _HAS_XENC
        self.xenc = CrossEncoder(cross_encoder_name) if self.use_xenc else None
        self._chunk_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def list_docs(self) -> List[str]:
        if not INDICES_DIR.exists():
            return []
        return sorted([d.name for d in INDICES_DIR.iterdir() if (d / "chunks.jsonl").exists()])

    def retrieve(
        self,
        query: str,
        include_docs: List[str],
        top_k_after_mmr: int = 6,
        bm25_weight: float = 0.35,
        faiss_topk_per_doc: int = 12,
    ) -> List[Hit]:
        cands = self._hybrid_candidates(
            query,
            include_docs=include_docs,
            bm25_weight=bm25_weight,
            faiss_topk_per_doc=faiss_topk_per_doc,
        )
        if not cands:
            return []

        texts = [c.text for c in cands]
        mat = self.embed.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        qvec = self.embed.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        sel = self._mmr(qvec, mat, lambda_mult=0.7, k=min(top_k_after_mmr, len(cands)))
        mmr_cands = [cands[i] for i in sel]

        if self.use_xenc and len(mmr_cands) > 1 and self.xenc is not None:
            pairs = [(query, c.text) for c in mmr_cands]
            scores = self.xenc.predict(pairs)
            order = np.argsort(scores)[::-1]
            mmr_cands = [mmr_cands[i] for i in order]

        hits: List[Hit] = []
        for c in mmr_cands:
            hits.append(
                Hit(
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
                )
            )
        return hits

    def retrieve_with_llm_queries(
        self,
        seed_query: str,
        include_docs: List[str],
        *,
        n_queries: int = 5,
        deployment: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        extra_instructions: Optional[str] = None,
        per_query_topk: int = 6,
        final_topk: int = 8,
        bm25_weight: float = 0.35,
        faiss_topk_per_doc: int = 12,
    ) -> Tuple[List[str], List[List[Hit]], List[Hit]]:
        """
        1) Expand seed_query into n_queries via Azure OpenAI (SDK).
        2) For each query: retrieve top per_query_topk.
        3) Fuse all results by (doc,id) keeping max score. Return:
           (queries, per_query_hits, fused_top_hits)
        """
        queries: List[str] = [seed_query]
        per_query_hits: List[List[Hit]] = []

        try:
            queries = generate_abstract_queries(
                seed_query,
                n=n_queries,
                deployment=deployment,
                api_version=api_version,
                extra_instructions=extra_instructions,
            )
        except Exception as e:
            print(f"[Azure Error] {type(e).__name__}: {e}")

        per_query_topk = max(1, int(per_query_topk))

        pooled: Dict[Tuple[str, str], Hit] = {}
        for q in queries:
            sub_hits = self.retrieve(
                q,
                include_docs=include_docs,
                top_k_after_mmr=per_query_topk,
                bm25_weight=bm25_weight,
                faiss_topk_per_doc=faiss_topk_per_doc,
            )
            per_query_hits.append(sub_hits)
            for h in sub_hits:
                key = (h.doc, h.id)
                if key not in pooled or h.score > pooled[key].score:
                    pooled[key] = h

        fused = sorted(pooled.values(), key=lambda x: x.score, reverse=True)
        return queries, per_query_hits, fused[:final_topk]

    # -------------------- internals --------------------
    def _hybrid_candidates(
        self,
        query: str,
        include_docs: List[str],
        bm25_weight: float,
        faiss_topk_per_doc: int,
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

            # lexical (BM25) — optional
            lex_pairs: List[Tuple[str, float]] = []
            if _HAS_BM25 and bm25_texts:
                tok = [t.split() for t in bm25_texts]
                bm = BM25Okapi(tok)
                scrs = bm.get_scores(query.split())
                if len(scrs) > 0:
                    top_idx = np.argsort(scrs)[::-1][:faiss_topk_per_doc]
                    lex_pairs = [(bm25_ids[i], float(scrs[i])) for i in top_idx]

            # z-norm per source
            def _znorm(vals: List[float]) -> List[float]:
                if not vals:
                    return []
                arr = np.array(vals, dtype=np.float32)
                mu, sd = arr.mean(), arr.std()
                if sd < 1e-6:
                    return [0.0] * len(vals)
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

                all_cands.append(
                    RetrievedInternal(
                        doc=ds,
                        id=cid,
                        text=rec["text"],
                        score=float(fused),
                        file=meta["file"],
                        page_start=int(meta.get("page_start", 0)),
                        page_end=int(meta.get("page_end", 0)),
                        column_start=int(meta.get("column_start", 0)),
                        column_end=int(meta.get("column_end", 0)),
                        block_ids=block_ids,
                        line_ids_all=line_ids_all,
                        line_text_map=line_text_map,
                        line_bbox_map=line_bbox_map,
                    )
                )

        all_cands.sort(key=lambda r: r.score, reverse=True)
        return all_cands

    @staticmethod
    def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_mult: float, k: int) -> List[int]:
        if cand_vecs.shape[0] == 0:
            return []
        sim_to_q = cand_vecs @ query_vec
        selected: List[int] = []
        candidates = list(range(cand_vecs.shape[0]))

        while candidates and len(selected) < k:
            if not selected:
                i = int(np.argmax(sim_to_q[candidates]))
                selected.append(candidates.pop(i))
                continue
            div = np.max(cand_vecs[candidates] @ cand_vecs[selected].T, axis=1)  # diversity penalty
            mmr_scores = lambda_mult * sim_to_q[candidates] - (1 - lambda_mult) * div
            i = int(np.argmax(mmr_scores))
            selected.append(candidates.pop(i))

        return selected

    # ---- neighbors (prev/next chunks) ----
    def _load_chunks_cache(self, docstem: str) -> None:
        if docstem in self._chunk_cache:
            return
        paths = _doc_paths(docstem)
        by_id: Dict[str, Dict[str, Any]] = {}
        order: Dict[int, str] = {}
        with open(paths["chunks"], "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cid = rec["id"]
                by_id[cid] = rec
                try:
                    n = int(cid.rsplit("::", 1)[1])
                    order[n] = cid
                except Exception:
                    pass
        self._chunk_cache[docstem] = {"by_id": by_id, "order": order}

    def get_neighbors(self, hit: Hit) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        self._load_chunks_cache(hit.doc)
        cache = self._chunk_cache[hit.doc]
        by_id = cache["by_id"]

        try:
            prefix, num_str = hit.id.rsplit("::", 1)
            cur = int(num_str)
        except Exception:
            return None, None

        prev_id = f"{prefix}::{cur-1}"
        next_id = f"{prefix}::{cur+1}"

        def _mk(rec: Optional[Dict[str, Any]]):
            if not rec:
                return None
            m = rec["meta"]
            text = rec["text"]
            line_ids_all: List[str] = m.get("line_ids", []) or []
            line_text_map: Dict[str, str] = m.get("line_text_map") or {}
            line_bbox_map: Dict[str, Optional[List[float]]] = m.get("line_bbox_map") or {}
            if not line_text_map and line_ids_all:
                lines = text.split("\n")
                if len(lines) == len(line_ids_all):
                    line_text_map = {lid: ln for lid, ln in zip(line_ids_all, lines)}
                else:
                    k = min(len(lines), len(line_ids_all))
                    line_text_map = {lid: ln for lid, ln in zip(line_ids_all[:k], lines[:k])}
            for lid in line_ids_all:
                if lid not in line_bbox_map:
                    line_bbox_map[lid] = None

            return {
                "text": text,
                "citation": {
                    "file": m["file"],
                    "pages": f"{m.get('page_start', 0)}–{m.get('page_end', 0)}",
                    "columns": f"{m.get('column_start', 0)}–{m.get('column_end', 0)}",
                    "block_ids": m.get("block_ids", []),
                    "line_ids_all": line_ids_all,
                    "line_text_map": line_text_map,
                    "line_bbox_map": line_bbox_map,
                },
            }

        return _mk(by_id.get(prev_id)), _mk(by_id.get(next_id))

# -------------------- CLI --------------------

if __name__ == "__main__":
    import argparse, sys, textwrap

    def _print_query_results(q: str, hits: List[Hit], preview_chars: int, limit_k: int) -> None:
        print(f"\n--- Query: {q}")
        if not hits:
            print("   (no results)")
            return
        for i, h in enumerate(hits[:limit_k], 1):
            cite = f"{h.citation['file']} {h.citation['pages']} (cols {h.citation['columns']})"
            txt = h.text.replace("\n", " ")
            snippet = textwrap.shorten(txt, width=preview_chars, placeholder=" …") if preview_chars > 0 else txt
            print(f"  [{i}] score={h.score:+.3f}  {cite}\n      {snippet}")
            print(f"      Blocks: {', '.join(h.citation['block_ids'])}")
            all_ids: List[str] = h.citation.get("line_ids_all") or []
            if all_ids:
                print(f"      Line IDs: {', '.join(all_ids)}")

    ap = argparse.ArgumentParser(
        description="Simple per-document RAG lookup",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("question", type=str, help="Your query")
    ap.add_argument("--docs", nargs="*", default=None, help="Docstems to search (default: all available)")
    ap.add_argument("--topk", type=int, default=6, help="Top-k after MMR")
    ap.add_argument("--bm25", type=float, default=0.35, help="BM25 fusion weight (0..1); ignored if rank-bm25 not installed")
    ap.add_argument("--faiss-per-doc", type=int, default=12, help="FAISS topk per document before re-ranking")
    ap.add_argument("--cross", action="store_true", help="Use cross-encoder re-ranking (requires cross-encoder)")
    ap.add_argument("--preview", type=int, default=220, help="Preview chars in console (0 = full text)")

    # LLM expansion controls
    ap.add_argument("--expand", action="store_true", help="Use Azure OpenAI to expand the query into multiple sub-queries")
    ap.add_argument("--n-queries", type=int, default=5, help="Number of LLM-generated queries for expansion")
    ap.add_argument("--deployment", type=str, default=None, help="Azure OpenAI deployment name (e.g., gpt-4o)")
    ap.add_argument("--api-version", type=str, default="2024-12-01-preview", help="Azure OpenAI API version for chat completions")
    ap.add_argument("--extra", type=str, default=None, help="Extra instructions for the query reformulator")
    ap.add_argument("--final-topk", type=int, default=8, help="Final number of results when using --expand")
    ap.add_argument("--per-query-topk", type=int, default=6, help="Per-subquery top-k after MMR when using --expand")
    ap.add_argument("--per-query-print-k", type=int, default=5, help="How many results to print per subquery")
    
    ap.add_argument("--answer", action="store_true", help="Answer the question by batching chunks to the LLM and return unified JSON")
    ap.add_argument("--answer-out", type=str, default=None, help="If set, write the unified JSON to this file")


    args = ap.parse_args()

    eng = RAGLookup(use_cross_encoder=args.cross)
    docs = args.docs or eng.list_docs()
    if not docs:
        print("No documents available under artifacts/indices/. Run your ingest first.", file=sys.stderr)
        sys.exit(2)

    if args.expand:
        queries, per_query_hits, fused_hits = eng.retrieve_with_llm_queries(
            args.question,
            include_docs=docs,
            n_queries=args.n_queries,
            deployment=args.deployment,
            api_version=args.api_version,
            extra_instructions=args.extra,
            per_query_topk=max(args.per_query_topk, args.per_query_print_k),
            final_topk=args.final_topk,
            bm25_weight=args.bm25 if _HAS_BM25 else 0.0,
            faiss_topk_per_doc=args.faiss_per_doc,
        )

        print("\n=== LLM Expanded Queries ===")
        for idx, q in enumerate(queries, 1):
            print(f"{idx}. {q}")

        print("\n=== Per-Query Results ===")
        for q, hits in zip(queries, per_query_hits):
            _print_query_results(q, hits, args.preview, args.per_query_print_k)

        hits = fused_hits
    else:
        hits = eng.retrieve(
            args.question,
            include_docs=docs,
            top_k_after_mmr=args.topk,
            bm25_weight=args.bm25 if _HAS_BM25 else 0.0,
            faiss_topk_per_doc=args.faiss_per_doc,
        )

    if not hits:
        print("\nNo hits.")
        sys.exit(0)
        
    answer_json = None
    if args.answer and hits:
        answer_json = answer_question_with_batches(
            args.question,
            hits,
            deployment=args.deployment,
            api_version=args.api_version,
            max_chunks_per_call=5,
            max_blocks_per_hit=3,
            max_lines_per_block=12,
        )
        
        
        # Always prepare paths (even if --answer-out omitted)
        primary_path, session_path = _prepare_answer_paths(args.answer_out)

        # Write primary ONLY if it doesn't already exist
        if not primary_path.exists():
            with open(primary_path, "w", encoding="utf-8") as f:
                json.dump(answer_json, f, ensure_ascii=False, indent=2)
            print(f"(Primary JSON written to {primary_path})")
        else:
            print(f"(Primary exists, not overwritten: {primary_path})")

        # Always write a session copy
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(answer_json, f, ensure_ascii=False, indent=2)
        print(f"(Session copy written to {session_path})")
                
    if answer_json:
        print("\n=== Unified JSON Answer (summary) ===")
        print(f"Primary path: {primary_path}")
        print(f"Session path: {session_path}")
        from textwrap import shorten
        print("Final answer preview:", shorten(answer_json["final"].get("final_answer",""), width=200, placeholder=" …"))
        print("Merged evidence items:", len(answer_json["final"].get("merged_evidence", [])))
        if args.answer_out:
            print(f"(Full JSON written to {args.answer_out})")

    print("\n=== Final Top Hits ===")
    for i, h in enumerate(hits, 1):
        cite = f"{h.citation['file']} {h.citation['pages']} (cols {h.citation['columns']})"
        txt = h.text.replace("\n", " ")
        from textwrap import shorten
        snippet = shorten(txt, width=args.preview, placeholder=" …") if args.preview > 0 else txt
        print(f"[{i}] score={h.score:+.3f}  {cite}\n    {snippet}")
        print(f"    Blocks: {', '.join(h.citation['block_ids'])}")
        all_ids: List[str] = h.citation.get("line_ids_all") or []
        print(f"    Line IDs: {', '.join(all_ids)}")

        prev, nxt = eng.get_neighbors(h)
        if prev:
            p_cite = f"{prev['citation']['file']} {prev['citation']['pages']} (cols {prev['citation']['columns']})"
            p_txt = prev["text"].replace("\n", " ")
            print(f"    Prev ▸ {p_cite}\n      { (p_txt if args.preview==0 else shorten(p_txt, width=args.preview, placeholder=' …')) }")
        if nxt:
            n_cite = f"{nxt['citation']['file']} {nxt['citation']['pages']} (cols {nxt['citation']['columns']})"
            n_txt = nxt["text"].replace("\n", " ")
            print(f"    Next ▸ {n_cite}\n      { (n_txt if args.preview==0 else shorten(n_txt, width=args.preview, placeholder=' …')) }")
