"""
backend/rag/agent.py
=====================
Real-time satellite intelligence reasoning agent.

Architecture
------------
                    ┌─────────────────────────────────────────┐
  analysis_metrics  │                                         │
  detections     ──►│  generate_intelligence_briefing()       │──► BriefingResult
  coordinates       │                                         │
                    └───────────┬─────────────────┬───────────┘
                                │                 │
                    ┌───────────▼──────┐  ┌───────▼────────────┐
                    │  ChromaDB (RAG)  │  │  IBM Granite LLM   │
                    │  local vector    │  │  (WatsonxLLM via   │
                    │  store seeded    │  │  langchain_ibm)    │
                    │  from KB docs    │  │  or fallback sim   │
                    └──────────────────┘  └────────────────────┘

Components
----------
SatelliteVectorStore
    Wraps Chroma + HuggingFaceEmbeddings.  Builds the in-process collection
    from KNOWLEDGE_DOCUMENTS on first call; returns a LangChain Retriever.

GraniteLLMClient
    Thin factory: returns a real WatsonxLLM when WATSONX_APIKEY is present,
    otherwise returns a GraniteFallbackAgent instance that generates a
    structured local briefing without any network call.

GraniteFallbackAgent
    Local simulator activated when WATSONX_APIKEY is missing.  Produces a
    deterministic but fully structured operational briefing from the supplied
    metrics — allows local development and CI to proceed uninterrupted.

SatelliteIntelligenceAgent
    Orchestrates retrieval + prompt construction + LLM call.
    Exposes generate_intelligence_briefing() as the single public method.

BriefingResult
    Typed dataclass returned by generate_intelligence_briefing().

Entry point
-----------
    from backend.rag.agent import build_agent, BriefingResult
    agent = build_agent()
    result = agent.generate_intelligence_briefing(metrics, detections, coords)

Environment variables
---------------------
    WATSONX_APIKEY       — IBM Cloud API key (required for live Granite)
    WATSONX_PROJECT_ID   — watsonx.ai project ID (required for live Granite)
    WATSONX_URL          — watsonx.ai endpoint URL
                           (default: https://us-south.ml.cloud.ibm.com)
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module is always importable
# ---------------------------------------------------------------------------

try:
    from langchain_ibm import WatsonxLLM  # type: ignore
    _LANGCHAIN_IBM_AVAILABLE = True
except BaseException:
    # Must catch BaseException, not Exception:
    # ibm_watsonx_ai triggers a KeyboardInterrupt (not ImportError) when its
    # pandas/Cython C-extensions fail to initialise on Python 3.13/3.14.
    # KeyboardInterrupt is a BaseException subclass and bypasses except Exception.
    _LANGCHAIN_IBM_AVAILABLE = False

try:
    from langchain_community.vectorstores import Chroma  # type: ignore
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    _LANGCHAIN_COMMUNITY_AVAILABLE = True
except BaseException:
    # Same reason — sentence-transformers / chromadb pull pandas Cython at import.
    _LANGCHAIN_COMMUNITY_AVAILABLE = False

from backend.rag.knowledge_base import KNOWLEDGE_DOCUMENTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WATSONX_DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
_GRANITE_MODEL_ID    = "ibm/granite-3-8b-instruct"
_EMBED_MODEL_ID      = "sentence-transformers/all-MiniLM-L6-v2"

# Number of RAG documents to retrieve per query
_RAG_TOP_K = 4

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BriefingResult:
    """Structured output of generate_intelligence_briefing().

    Attributes
    ----------
    briefing_text : str
        Full natural-language operational intelligence briefing.
    retrieved_context_ids : list[str]
        Document IDs of the knowledge-base entries retrieved by ChromaDB.
    used_fallback : bool
        True when GraniteFallbackAgent was used instead of live Granite.
    model_id : str
        The model identifier that produced the briefing.
    analysis_metrics : dict
        The raw metrics that were fed into the prompt (for audit logging).
    """
    briefing_text: str
    retrieved_context_ids: List[str]
    used_fallback: bool
    model_id: str
    analysis_metrics: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "briefing_text":          self.briefing_text,
            "retrieved_context_ids":  self.retrieved_context_ids,
            "used_fallback":          self.used_fallback,
            "model_id":               self.model_id,
            "analysis_metrics":       self.analysis_metrics,
        }


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------

class SatelliteVectorStore:
    """Builds and queries a local in-process ChromaDB collection.

    Uses ``sentence-transformers/all-MiniLM-L6-v2`` for embeddings (runs
    entirely locally — no external API required).  The collection is built
    once from ``KNOWLEDGE_DOCUMENTS`` and held in memory for the lifetime
    of the process.

    Falls back to a keyword-match retriever when langchain_community is not
    installed, so the pipeline never hard-fails due to a missing vector store.
    """

    def __init__(self) -> None:
        self._retriever = None
        self._doc_map: Dict[str, str] = {
            d["id"]: d["text"] for d in KNOWLEDGE_DOCUMENTS
        }
        self._build()

    def _build(self) -> None:
        if not _LANGCHAIN_COMMUNITY_AVAILABLE:
            logger.warning(
                "langchain_community not available; using keyword retriever fallback."
            )
            return

        try:
            embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL_ID)
            texts    = [d["text"]     for d in KNOWLEDGE_DOCUMENTS]
            metadatas = [d["metadata"] for d in KNOWLEDGE_DOCUMENTS]
            ids      = [d["id"]       for d in KNOWLEDGE_DOCUMENTS]

            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                ids=ids,
                collection_name="satellite_knowledge",
            )
            self._retriever = vectorstore.as_retriever(
                search_kwargs={"k": _RAG_TOP_K}
            )
            logger.info("ChromaDB vector store built (%d documents).", len(texts))
        except Exception as exc:  # pragma: no cover
            logger.warning("ChromaDB build failed (%s); using keyword fallback.", exc)

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """Return up to _RAG_TOP_K relevant documents for *query*.

        Returns a list of dicts with keys ``"id"`` and ``"text"``.
        Falls back to simple keyword overlap when ChromaDB is unavailable.
        """
        if self._retriever is not None:
            try:
                docs = self._retriever.invoke(query)
                results = []
                for doc in docs:
                    doc_id = doc.metadata.get("id", "unknown")
                    results.append({"id": doc_id, "text": doc.page_content})
                return results
            except Exception as exc:  # pragma: no cover
                logger.warning("ChromaDB retrieval failed (%s); using keyword fallback.", exc)

        # Keyword fallback — score by word overlap
        query_tokens = set(query.lower().split())
        scored = []
        for doc in KNOWLEDGE_DOCUMENTS:
            overlap = len(query_tokens & set(doc["text"].lower().split()))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": d["id"], "text": d["text"]}
            for _, d in scored[:_RAG_TOP_K]
        ]


# ---------------------------------------------------------------------------
# Fallback simulator
# ---------------------------------------------------------------------------

class GraniteFallbackAgent:
    """Local briefing generator — activated when WATSONX_APIKEY is absent.

    Produces a fully structured operational briefing from supplied metrics
    using deterministic string formatting.  No network calls are made.
    This allows the full backend pipeline to run in offline / CI mode.
    """

    model_id: str = "fallback/local-template-agent"

    def invoke(self, prompt: str) -> str:  # mirrors LangChain LLM interface
        """Extract key stats embedded in *prompt* and format a structured briefing."""
        # The prompt contains a clearly delimited METRICS block — parse it back
        # so the fallback output contains real numbers, not placeholders.
        metrics = _parse_metrics_from_prompt(prompt)
        return _render_fallback_briefing(metrics)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

class GraniteLLMClient:
    """Returns either a live WatsonxLLM or a GraniteFallbackAgent.

    The decision is made once at construction based on environment variables.
    Callers always interact through a single `.invoke(prompt) -> str` method.
    """

    def __init__(self) -> None:
        api_key    = os.getenv("WATSONX_APIKEY", "").strip()
        project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
        url        = os.getenv("WATSONX_URL", _WATSONX_DEFAULT_URL).strip()

        if api_key and project_id and _LANGCHAIN_IBM_AVAILABLE:
            logger.info("Initialising live WatsonxLLM (%s).", _GRANITE_MODEL_ID)
            self._llm = WatsonxLLM(
                model_id=_GRANITE_MODEL_ID,
                url=url,
                apikey=api_key,
                project_id=project_id,
                params={
                    "decoding_method": "greedy",
                    "max_new_tokens":  1024,
                    "min_new_tokens":  200,
                    "repetition_penalty": 1.1,
                },
            )
            self.model_id    = _GRANITE_MODEL_ID
            self.used_fallback = False
        else:
            if not api_key:
                logger.warning(
                    "WATSONX_APIKEY not set — using GraniteFallbackAgent. "
                    "Set WATSONX_APIKEY and WATSONX_PROJECT_ID to enable live Granite."
                )
            self._llm = GraniteFallbackAgent()
            self.model_id    = GraniteFallbackAgent.model_id
            self.used_fallback = True

    def invoke(self, prompt: str) -> str:
        """Call the underlying LLM (live or fallback) with *prompt*."""
        return self._llm.invoke(prompt)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    analysis_metrics: Dict[str, Any],
    detections: List[Any],
    retrieved_docs: List[Dict[str, str]],
    coordinates: Optional[str],
) -> str:
    """Assemble the final prompt string for Granite.

    The prompt is structured in four sections:
    1. SYSTEM ROLE  — analyst persona and output format contract.
    2. CONTEXT      — retrieved RAG knowledge passages.
    3. METRICS      — quantitative scene statistics and detection table.
    4. INSTRUCTION  — explicit briefing sections to produce.
    """
    # ── Section 1: System role ──────────────────────────────────────────────
    system_role = textwrap.dedent("""\
        You are an expert satellite imagery intelligence analyst specialising in
        Landsat thermal band analysis and neural RGB reconstruction with Monte Carlo
        Dropout uncertainty quantification.

        Your task is to produce a structured operational intelligence briefing in
        three clearly labelled sections:

        1. VISUAL RECONSTRUCTION INTEGRITY
           Evaluate the structural fidelity of the thermal-to-RGB reconstruction
           based on the reported mean and maximum aleatoric uncertainty values and
           image resolution.

        2. TACTICAL OBJECT ASSESSMENT
           Analyse each detected entity. Highlight any detections whose adjusted
           confidence is meaningfully lower than the raw confidence, explain why
           the uncertainty penalisation occurred, and assess operational reliability.

        3. OPERATIONAL GUIDANCE
           Explicitly identify any high-uncertainty spatial zones (mean_uncertainty > 0.4).
           Warn analysts that RGB visual features in those zones are speculative outputs
           of the reconstruction model. Recommend whether each detection can be acted
           upon, requires secondary verification, or should be disregarded.

        Be precise, concise, and quantitative. Reference the provided metrics directly.
        Do not hallucinate class names, confidence values, or coordinates.
    """)

    # ── Section 2: RAG context ──────────────────────────────────────────────
    context_block = "RETRIEVED KNOWLEDGE CONTEXT:\n"
    for doc in retrieved_docs:
        context_block += f"[{doc['id']}]\n{doc['text']}\n\n"

    # ── Section 3: Metrics block ────────────────────────────────────────────
    res_h = analysis_metrics.get("resolution_h", "unknown")
    res_w = analysis_metrics.get("resolution_w", "unknown")
    mean_unc = analysis_metrics.get("mean_uncertainty", 0.0)
    max_unc  = analysis_metrics.get("max_uncertainty",  0.0)
    mc_passes = analysis_metrics.get("mc_passes", "unknown")
    coord_str = f"Coordinates: {coordinates}" if coordinates else "Coordinates: not provided"

    metrics_block = textwrap.dedent(f"""\
        SCENE METRICS:
        {coord_str}
        Image resolution      : {res_h} × {res_w} pixels
        MC-Dropout passes     : {mc_passes}
        Mean aleatoric uncertainty (scene-wide) : {mean_unc:.4f}
        Max  aleatoric uncertainty (scene-wide) : {max_unc:.4f}
        Number of detections  : {len(detections)}

        DETECTION TABLE:
        | # | Class       | Raw Conf | Mean Unc | Adj Conf | BBox (x1,y1,x2,y2)        |
        |---|-------------|----------|----------|----------|---------------------------|
    """)

    for idx, det in enumerate(detections, start=1):
        if hasattr(det, "class_name"):
            # DetectionResult dataclass
            cls   = det.class_name
            raw   = det.raw_confidence
            unc   = det.mean_uncertainty
            adj   = det.adjusted_confidence
            bbox  = det.bbox
        else:
            # Plain dict (e.g. from to_dict())
            cls   = det.get("class_name", "unknown")
            raw   = det.get("raw_confidence", 0.0)
            unc   = det.get("mean_uncertainty", 0.0)
            adj   = det.get("adjusted_confidence", 0.0)
            b     = det.get("bbox", {})
            bbox  = (b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0))

        x1, y1, x2, y2 = bbox
        metrics_block += (
            f"        | {idx:<2d}| {cls:<12s}| {raw:>8.4f}| {unc:>8.4f}| "
            f"{adj:>8.4f}| ({x1:>5d},{y1:>5d},{x2:>5d},{y2:>5d})|\n"
        )

    # Embed a parseable summary for the fallback agent
    metrics_block += textwrap.dedent(f"""
        ##METRICS_JSON_START##
        mean_uncertainty={mean_unc:.6f}
        max_uncertainty={max_unc:.6f}
        resolution={res_h}x{res_w}
        mc_passes={mc_passes}
        num_detections={len(detections)}
        ##METRICS_JSON_END##
    """)

    # ── Section 4: Instruction ──────────────────────────────────────────────
    instruction = (
        "Based on the knowledge context and the scene metrics above, produce the "
        "three-section operational intelligence briefing. Write in plain English. "
        "Use the exact section headings: VISUAL RECONSTRUCTION INTEGRITY, "
        "TACTICAL OBJECT ASSESSMENT, and OPERATIONAL GUIDANCE."
    )

    return "\n\n".join([system_role, context_block, metrics_block, instruction])


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _parse_metrics_from_prompt(prompt: str) -> Dict[str, Any]:
    """Extract the embedded metrics block from the prompt string."""
    metrics: Dict[str, Any] = {}
    in_block = False
    for line in prompt.splitlines():
        if "##METRICS_JSON_START##" in line:
            in_block = True
            continue
        if "##METRICS_JSON_END##" in line:
            break
        if in_block and "=" in line:
            key, _, val = line.strip().partition("=")
            metrics[key.strip()] = val.strip()
    return metrics


def _render_fallback_briefing(metrics: Dict[str, Any]) -> str:
    """Render a structured local briefing from parsed metrics."""
    mean_unc     = float(metrics.get("mean_uncertainty", 0.0))
    max_unc      = float(metrics.get("max_uncertainty",  0.0))
    resolution   = metrics.get("resolution", "unknown")
    mc_passes    = metrics.get("mc_passes",  "unknown")
    num_det      = int(metrics.get("num_detections", 0))

    # Qualitative thresholds
    if mean_unc < 0.2:
        fidelity = "HIGH — reconstruction is spatially stable across all MC-Dropout passes"
    elif mean_unc < 0.4:
        fidelity = "MODERATE — minor stochastic variation; spatial boundaries may shift by 1–2 pixels"
    elif mean_unc < 0.6:
        fidelity = "ELEVATED — colour assignments in portions of the scene are speculative"
    else:
        fidelity = "LOW — significant uncertainty throughout; treat reconstruction as indicative only"

    adj_guidance = (
        "proceed with standard verification protocols"
        if mean_unc < 0.4 else
        "apply additional cross-source verification before operational use"
    )

    return textwrap.dedent(f"""\
        ─────────────────────────────────────────────────────────────────────
        OPERATIONAL INTELLIGENCE BRIEFING  [LOCAL FALLBACK MODE]
        ─────────────────────────────────────────────────────────────────────

        VISUAL RECONSTRUCTION INTEGRITY
        --------------------------------
        Scene resolution     : {resolution} pixels
        MC-Dropout passes    : {mc_passes}
        Mean aleatoric uncertainty : {mean_unc:.4f}
        Max  aleatoric uncertainty : {max_unc:.4f}
        Reconstruction fidelity    : {fidelity}

        The thermal-to-RGB reconstruction pipeline completed {mc_passes} stochastic
        forward passes. The scene-wide mean uncertainty of {mean_unc:.4f} indicates
        {fidelity.lower()}. Regions approaching the maximum uncertainty value of
        {max_unc:.4f} should be treated as unreliable for detailed feature extraction.

        TACTICAL OBJECT ASSESSMENT
        ---------------------------
        Total detections : {num_det}

        {"No objects were detected in this scene." if num_det == 0 else
         f"{num_det} object(s) were detected. Detections with mean_uncertainty > 0.4 "
         f"in their bounding-box region have been uncertainty-penalised. Adjusted "
         f"confidence scores reflect the product of raw YOLO confidence and the "
         f"complement of regional mean uncertainty. Detections in high-uncertainty "
         f"zones may have their class boundaries incorrectly localised due to "
         f"speculative RGB texture generation in those regions."}

        OPERATIONAL GUIDANCE
        ---------------------
        {"No high-uncertainty zones detected (mean_uncertainty < 0.4). Standard "
         "operational confidence applies." if mean_unc < 0.4 else
         f"WARNING: Scene mean uncertainty ({mean_unc:.4f}) exceeds the 0.4 threshold. "
         f"High-uncertainty spatial zones are present. Visual RGB features in these "
         f"zones are speculative outputs of the MC-Dropout reconstruction model and "
         f"should NOT be treated as confirmed visual observations."}

        Analyst recommendation: {adj_guidance}.
        All detections with adjusted_confidence < 0.3 require secondary verification
        against raw thermal DN values, QA_PIXEL cloud masks, or co-registered optical
        imagery before operational use.

        ─────────────────────────────────────────────────────────────────────
        NOTE: This briefing was generated by GraniteFallbackAgent (local mode).
        Set WATSONX_APIKEY and WATSONX_PROJECT_ID to enable IBM Granite reasoning.
        ─────────────────────────────────────────────────────────────────────
    """)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class SatelliteIntelligenceAgent:
    """Orchestrates RAG retrieval + prompt construction + LLM generation.

    Instantiate via :func:`build_agent` rather than directly.

    Parameters
    ----------
    vector_store : SatelliteVectorStore
        Pre-built local ChromaDB retriever.
    llm_client : GraniteLLMClient
        Live WatsonxLLM or GraniteFallbackAgent wrapper.
    """

    def __init__(
        self,
        vector_store: SatelliteVectorStore,
        llm_client: GraniteLLMClient,
    ) -> None:
        self._vs  = vector_store
        self._llm = llm_client

    # ------------------------------------------------------------------
    def generate_intelligence_briefing(
        self,
        analysis_metrics: Dict[str, Any],
        detections: List[Any],
        coordinates: Optional[str] = None,
    ) -> BriefingResult:
        """Generate a structured operational intelligence briefing.

        Parameters
        ----------
        analysis_metrics:
            Dictionary of quantitative scene statistics.  Expected keys:

            ==================  ============================================
            Key                 Description
            ==================  ============================================
            ``resolution_h``    Image height in pixels (int)
            ``resolution_w``    Image width  in pixels (int)
            ``mean_uncertainty``Scene-wide mean of uncertainty_map (float)
            ``max_uncertainty`` Scene-wide max  of uncertainty_map (float)
            ``mc_passes``       Number of MC-Dropout passes used   (int)
            ==================  ============================================

        detections:
            List of :class:`~backend.models.detector.DetectionResult` instances
            (or their ``to_dict()`` equivalents).  May be empty.

        coordinates:
            Optional human-readable location string, e.g.
            ``"37.77°N 122.41°W — San Francisco, CA"`` or WRS-2 path/row.

        Returns
        -------
        BriefingResult
            Contains the full briefing text, retrieved doc IDs, fallback flag,
            model ID, and the input metrics (for audit logging).

        Raises
        ------
        ValueError
            If ``analysis_metrics`` is missing required keys.
        """
        _validate_metrics(analysis_metrics)

        # ── 1. Formulate RAG query from the scene context ─────────────────────
        rag_query = _build_rag_query(analysis_metrics, detections, coordinates)
        logger.debug("RAG query: %s", rag_query)

        # ── 2. Retrieve relevant knowledge passages ───────────────────────────
        retrieved = self._vs.retrieve(rag_query)
        retrieved_ids = [d["id"] for d in retrieved]
        logger.debug("Retrieved docs: %s", retrieved_ids)

        # ── 3. Build the full prompt ──────────────────────────────────────────
        prompt = _build_prompt(
            analysis_metrics=analysis_metrics,
            detections=detections,
            retrieved_docs=retrieved,
            coordinates=coordinates,
        )

        # ── 4. Call LLM ───────────────────────────────────────────────────────
        logger.info(
            "Calling LLM (%s) for briefing generation...", self._llm.model_id
        )
        briefing_text = self._llm.invoke(prompt)

        return BriefingResult(
            briefing_text=briefing_text,
            retrieved_context_ids=retrieved_ids,
            used_fallback=self._llm.used_fallback,
            model_id=self._llm.model_id,
            analysis_metrics=analysis_metrics,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_metrics(metrics: Dict[str, Any]) -> None:
    required = {"resolution_h", "resolution_w", "mean_uncertainty",
                "max_uncertainty", "mc_passes"}
    missing = required - set(metrics.keys())
    if missing:
        raise ValueError(
            f"analysis_metrics is missing required keys: {sorted(missing)}"
        )


def _build_rag_query(
    metrics: Dict[str, Any],
    detections: List[Any],
    coordinates: Optional[str],
) -> str:
    """Build a natural-language query for the vector store retriever."""
    mean_unc = metrics.get("mean_uncertainty", 0.0)

    # Collect class names from detections
    class_names: List[str] = []
    for det in detections:
        if hasattr(det, "class_name"):
            class_names.append(det.class_name)
        elif isinstance(det, dict):
            class_names.append(det.get("class_name", ""))

    unc_term = (
        "high uncertainty zones operational guidance"
        if mean_unc > 0.4 else
        "MC-Dropout uncertainty interpretation low uncertainty"
    )

    classes_term = (
        f"detection of {', '.join(set(class_names))}"
        if class_names else
        "no object detections empty scene"
    )

    location_term = f"near {coordinates}" if coordinates else ""

    return (
        f"Landsat thermal analysis {classes_term} {unc_term} "
        f"adjusted confidence {location_term}"
    ).strip()


# ---------------------------------------------------------------------------
# Public factory — entry point for backend/app.py
# ---------------------------------------------------------------------------

def build_agent() -> SatelliteIntelligenceAgent:
    """Construct and return a ready-to-use SatelliteIntelligenceAgent.

    Reads environment variables, builds ChromaDB, and selects live Granite
    or the fallback simulator automatically.

    Example
    -------
    >>> from backend.rag.agent import build_agent
    >>> agent = build_agent()
    >>> result = agent.generate_intelligence_briefing(
    ...     analysis_metrics={
    ...         "resolution_h":     256,
    ...         "resolution_w":     256,
    ...         "mean_uncertainty": 0.31,
    ...         "max_uncertainty":  0.78,
    ...         "mc_passes":        20,
    ...     },
    ...     detections=detections,       # list[DetectionResult]
    ...     coordinates="51.50°N 0.12°W — London, UK",
    ... )
    >>> print(result.briefing_text)
    >>> print("Fallback used:", result.used_fallback)
    """
    vector_store = SatelliteVectorStore()
    llm_client   = GraniteLLMClient()
    return SatelliteIntelligenceAgent(
        vector_store=vector_store,
        llm_client=llm_client,
    )
