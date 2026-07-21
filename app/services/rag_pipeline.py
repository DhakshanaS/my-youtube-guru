"""RAG pipeline (Module 5) — the core retrieval-augmented question answering.

This is where everything comes together and where hallucination is controlled.
Flow for a user question:

    embed question
      → similarity search in ChromaDB (top-k)
      → is the best hit relevant enough? (similarity >= threshold)
            NO  → don't touch YouTube or the LLM's knowledge; return an honest
                  "not in your watch history — answer from general knowledge?"
                  prompt and wait for explicit user confirmation.
            YES → for those hits only, LAZILY fetch transcripts (cached back to
                  ChromaDB so it's a one-time cost), refine each category from
                  the real content, build a STRICTLY GROUNDED prompt containing
                  only those transcripts, and have the LLM answer from them.

The grounding system prompt (below) is the heart of the anti-hallucination
design: the model is told, in strong and explicit terms, to answer only from
the supplied transcripts and to admit when the answer isn't there.

Dependencies (vector store, embedder, LLM, transcript service) are injected
with singleton defaults so the whole flow is unit-testable without a network,
an API key, or real YouTube access.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from app.config import get_settings
from app.services.embeddings import build_embedding_text, embedding_model
from app.services.grounding_log import grounding_log
from app.services.llm_service import llm_service
from app.services.transcripts import transcript_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


# ── Prompts ─────────────────────────────────────────────────────────────────
# Kept as module constants so the grounding rules are easy to find, review and
# tweak — this is the most important prompt in the project.

GROUNDING_SYSTEM_PROMPT = """You are "My YouTube Guru", an assistant that answers \
the user's questions using ONLY the transcripts of videos they have watched on \
YouTube, supplied below as VIDEO CONTEXT.

Grounding rules — follow exactly:
1. Ground every statement in the VIDEO CONTEXT. It is your only permitted source \
of information. Do not use any outside or prior knowledge to add facts.
2. If the VIDEO CONTEXT does not fully answer the question, answer what it does \
cover and clearly state what it doesn't. Never fill gaps with invented details, \
figures, names, or quotes.
3. Cite sources inline by number and title, like "[1] <title>", right next to \
the claims they support, so the user can verify each point.
4. Some entries may have only a title and no transcript. You may note what such \
a video appears to be about from its title, but make clear you are inferring \
from the title, not its content.

Style — write like a knowledgeable expert giving a genuinely helpful, thorough \
answer:
- Open with a direct answer to the question, then expand with the supporting \
detail.
- Be comprehensive and in-depth: fully develop each relevant idea, walk through \
any steps, frameworks, or examples the videos give, and explain the reasoning \
behind them rather than just naming them.
- When several videos speak to the question, synthesise across them and note \
where they agree or differ; when a video doesn't address it, say so briefly.
- Format for easy reading using Markdown: short paragraphs, **bold** for key \
terms, and bulleted or numbered lists for steps and enumerations. Add a short \
heading only when the answer is long enough to need sections.
- Never pad with repetition or filler, and never introduce anything not \
grounded in the VIDEO CONTEXT. Your depth must come from the transcripts, not \
from your own knowledge."""

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. \
The user's YouTube watch history did not contain anything relevant to their \
question, and they have explicitly asked you to answer from your own general \
knowledge instead.

Begin by making clear this comes from general knowledge and not from their \
watched videos. Then give a thorough, well-organised answer: develop the key \
points in depth, format it with Markdown (short paragraphs, **bold**, and lists) \
for readability, and include concrete examples where they help."""

# Message shown when nothing relevant is found (matches the project spec).
NO_MATCH_MESSAGE = (
    "I could not find this in your YouTube knowledge base. "
    "Would you like me to answer from my own knowledge instead?"
)


# ── Response types ───────────────────────────────────────────────────────────

@dataclass
class Source:
    """A video used to support an answer (surfaced in the UI for verification)."""
    video_id: str
    title: str
    url: str
    similarity: float
    transcript_used: bool           # True if real transcript content was used
    note: str | None = None         # e.g. "no transcript — inferred from title"


@dataclass
class RAGResponse:
    """Everything the API/UI needs, including data for grounding inspection."""
    answer: str
    grounded: bool = False              # answered from watch-history transcripts
    from_general_knowledge: bool = False
    needs_confirmation: bool = False    # nothing relevant → asking to use general knowledge
    question: str = ""
    sources: list[Source] = field(default_factory=list)
    # Debug/eval trail (Module 9): what was retrieved and with what score.
    retrieval: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self, *, store=vector_store, embedder=embedding_model,
                 llm=llm_service, transcripts=transcript_service,
                 evaluator=grounding_log) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._transcripts = transcripts
        self._evaluator = evaluator

    # ── main entry point ─────────────────────────────────────────────────
    def answer_question(self, question: str, *, history: list | None = None,
                        top_k: int | None = None, on_event=None) -> RAGResponse:
        """Answer a question grounded in the user's watched-video transcripts.

        `history` is the recent conversation (list of {role, content}); it lets
        follow-up questions work (see `_condense_query`). `on_event(kind, data)`
        reports progress for the live thinking trace / streaming UI.
        """
        emit = on_event or (lambda *a, **k: None)
        settings = get_settings()
        max_sources = top_k or settings.top_k_results
        threshold = settings.similarity_threshold

        # 1. Retrieve candidates. For a follow-up in an ongoing chat, first
        #    rewrite the question into a standalone search query using the
        #    conversation, so references ("the second one", "it") resolve to the
        #    right videos instead of retrieving on the literal follow-up text.
        emit("status", {"stage": "searching",
                        "message": "Searching your watch history"})
        search_query = self._condense_query(question, history, emit)
        q_vec = self._embedder.embed_text(search_query)
        candidates = self._store.query(q_vec, top_k=settings.retrieval_candidates)

        # 2. Adaptive selection + relevance gate. Keep EVERY candidate above the
        #    threshold (up to the cap) — so the number of sources scales with how
        #    many videos are actually relevant, not a fixed count. If none clear
        #    the threshold, stop here and ask before using general knowledge.
        relevant = [h for h in candidates if h.similarity >= threshold]
        if not relevant:
            near = [{"video_id": h.video_id, "title": h.metadata.get("title", ""),
                     "similarity": round(h.similarity, 4)} for h in candidates[:8]]
            best = candidates[0].similarity if candidates else 0.0
            logger.info("No relevant video for %r (best=%.3f < %.3f)",
                        search_query, best, threshold)
            emit("retrieved", {"message": "No closely matching videos", "videos": near})
            emit("status", {"stage": "no_match",
                            "message": "Nothing relevant in your watch history"})
            resp = RAGResponse(
                answer=NO_MATCH_MESSAGE, grounded=False, needs_confirmation=True,
                question=question, sources=[], retrieval=near,
            )
            self._record(resp)
            return resp

        hits = relevant[:max_sources]
        retrieval_trail = [
            {"video_id": h.video_id, "title": h.metadata.get("title", ""),
             "similarity": round(h.similarity, 4)}
            for h in hits
        ]
        emit("retrieved", {
            "message": f"Found {len(hits)} related video{'s' if len(hits) != 1 else ''}",
            "videos": retrieval_trail,
        })

        # 3. Build grounded context from the relevant hits (lazy transcripts).
        context_blocks: list[str] = []
        sources: list[Source] = []
        budget = settings.transcript_char_budget

        for i, hit in enumerate(hits, start=1):
            title = hit.metadata.get("title", "(untitled)")
            url = hit.metadata.get("url", "")
            text, used, note = self._ensure_transcript(hit, emit)

            if used and text:
                excerpt = text[:budget]
                truncated = " …[transcript truncated]" if len(text) > budget else ""
                context_blocks.append(
                    f"[{i}] TITLE: {title}\nURL: {url}\nTRANSCRIPT: {excerpt}{truncated}"
                )
            else:
                # No transcript: give the model only the title, clearly flagged.
                context_blocks.append(
                    f"[{i}] TITLE: {title}\nURL: {url}\n"
                    f"TRANSCRIPT: (unavailable — {note}. Only the title is known.)"
                )

            sources.append(Source(
                video_id=hit.video_id, title=title, url=url,
                similarity=round(hit.similarity, 4),
                transcript_used=bool(used and text),
                note=None if (used and text) else f"no transcript — {note}",
            ))

        # 4. Ask the LLM, strictly grounded on the assembled context. Stream
        #    the answer token-by-token so the UI can render it as it's written.
        #    Recent conversation is included so follow-ups keep continuity, but
        #    the model is told to draw FACTS only from the video transcripts.
        emit("status", {"stage": "generating",
                        "message": "Reading transcripts and writing a grounded answer"})
        context = "\n\n".join(context_blocks)
        convo = self._format_history(history)
        convo_block = (f"CONVERSATION SO FAR (for context and resolving references "
                       f"only — not a source of facts):\n{convo}\n\n") if convo else ""
        user_prompt = (
            f"{convo_block}"
            f"VIDEO CONTEXT (your only source of facts):\n\n{context}\n\n"
            f"----\nQUESTION: {question}\n\n"
            "Answer the question using only the VIDEO CONTEXT for facts; use the "
            "conversation above only to understand what the question refers to. "
            "Cite the sources you used."
        )
        answer = self._llm.chat_stream(
            system=GROUNDING_SYSTEM_PROMPT, user=user_prompt,
            temperature=0.35, max_tokens=3072,
            on_delta=lambda piece: emit("answer_delta", {"text": piece}),
        ).strip()
        if not answer:
            answer = ("I found relevant videos in your history (listed below), but "
                      "the model didn't return any answer text this time. Please "
                      "try asking again or rephrasing the question.")
            emit("answer_delta", {"text": answer})

        resp = RAGResponse(
            answer=answer, grounded=True, question=question,
            sources=sources, retrieval=retrieval_trail,
        )
        self._record(resp)
        return resp

    # ── explicit general-knowledge fallback (after user confirmation) ────
    def answer_from_general_knowledge(self, question: str, *, history: list | None = None,
                                      on_event=None) -> RAGResponse:
        """Answer from the LLM's own knowledge — ONLY called after the user
        confirms via the /confirm endpoint. Clearly flagged as not grounded."""
        emit = on_event or (lambda *a, **k: None)
        emit("status", {"stage": "generating",
                        "message": "Answering from general knowledge"})
        convo = self._format_history(history)
        user = (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "") + f"QUESTION: {question}"
        answer = self._llm.chat_stream(
            system=GENERAL_KNOWLEDGE_SYSTEM_PROMPT, user=user,
            temperature=0.5, max_tokens=3072,
            on_delta=lambda piece: emit("answer_delta", {"text": piece}),
        ).strip()
        if not answer:
            answer = ("The model didn't return any text this time. Please try "
                      "asking again.")
            emit("answer_delta", {"text": answer})
        resp = RAGResponse(
            answer=answer, grounded=False, from_general_knowledge=True,
            question=question, sources=[],
        )
        self._record(resp)
        return resp

    # ── conversation helpers (follow-up support) ─────────────────────────
    def _condense_query(self, question: str, history: list | None, emit=None) -> str:
        """Rewrite a follow-up into a standalone search query using the chat
        history, so retrieval finds the right videos for questions like "tell me
        more about the second one". Returns the original question unchanged when
        there's no history or on any failure — so a first question is never
        slowed by an extra LLM call."""
        if not history:
            return question
        recent = self._format_history(history, max_messages=6, per_message_chars=500)
        if not recent:
            return question
        emit = emit or (lambda *a, **k: None)
        try:
            system = ("You rewrite a user's follow-up question into a single, "
                      "self-contained search query for a video search engine, "
                      "using the conversation for context. Resolve references "
                      "like 'it' or 'the second one' into explicit terms. Output "
                      "ONLY the rewritten query — no quotes, no preamble.")
            user = (f"CONVERSATION:\n{recent}\n\nFOLLOW-UP QUESTION: {question}\n\n"
                    "Standalone search query:")
            rewritten = self._llm.chat(system=system, user=user,
                                       temperature=0.0, max_tokens=80).strip()
            if rewritten and len(rewritten) <= 400:
                logger.info("Condensed follow-up %r -> %r", question, rewritten)
                emit("status", {"stage": "searching",
                                "message": f"Interpreting your follow-up as: “{rewritten}”"})
                return rewritten
        except Exception:  # noqa: BLE001 — condensation is best-effort
            logger.debug("Query condensation failed; using raw question", exc_info=True)
        return question

    @staticmethod
    def _format_history(history: list | None, max_messages: int = 6,
                        per_message_chars: int = 800) -> str:
        """Render recent turns as plain text for prompts (newest kept, capped)."""
        if not history:
            return ""
        lines = []
        for m in history[-max_messages:]:
            role = "User" if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user" else "Assistant"
            raw = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            content = " ".join((raw or "").split())[:per_message_chars]
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _record(self, response) -> None:
        """Log this answer's grounding to the evaluation audit trail (Module 9)."""
        try:
            self._evaluator.record(response)
        except Exception:  # noqa: BLE001 — never let logging break answering
            logger.debug("Grounding log write failed", exc_info=True)

    # ── transcript resolution (lazy load + cache) ────────────────────────
    def _ensure_transcript(self, hit, emit=None) -> tuple[str | None, bool, str | None]:
        """Return (transcript_text, used, note) for a retrieved video.

        Uses the cached transcript if we already have one; otherwise fetches it
        once, caches it back into ChromaDB (with a refreshed embedding and
        category), and records unavailability so we never re-fetch a video that
        has no captions. Emits progress events for the (slow) fetch step.
        """
        emit = emit or (lambda *a, **k: None)
        meta = hit.metadata
        title = meta.get("title", "")

        # Already resolved on a previous query?
        if meta.get("transcript_fetched"):
            cached = meta.get("transcript")
            if cached:
                emit("transcript", {"video_id": hit.video_id, "title": title,
                                    "status": "cached",
                                    "message": "Using saved transcript"})
                return cached, True, None
            # Previously found to have no transcript.
            note = meta.get("transcript_note", "no transcript available")
            emit("transcript", {"video_id": hit.video_id, "title": title,
                                "status": "unavailable", "message": note})
            return None, False, note

        # First time: fetch from YouTube (with retry/backoff inside the service).
        emit("transcript", {"video_id": hit.video_id, "title": title,
                            "status": "fetching",
                            "message": "Fetching transcript from YouTube"})
        result = self._transcripts.fetch(hit.video_id)
        channel = meta.get("channel") or None

        if not result.available or not result.text:
            # Remember the miss so future queries skip the fetch.
            reason = result.reason or "no transcript available"
            emit("transcript", {"video_id": hit.video_id, "title": title,
                                "status": "unavailable", "message": reason})
            try:
                self._store.mark_transcript_unavailable(hit.video_id, reason)
            except Exception:  # noqa: BLE001 — caching is best-effort
                logger.debug("Could not mark transcript unavailable", exc_info=True)
            return None, False, reason

        # Cache the transcript back, and use it to refine the vector + category
        # now that we have the real content (spec: refine category from transcript).
        transcript = result.text
        emit("transcript", {"video_id": hit.video_id, "title": title,
                            "status": "ready", "chars": len(transcript),
                            "message": f"Transcript ready ({len(transcript):,} chars)"})
        try:
            new_doc = build_embedding_text(title, channel, transcript)
            new_vec = self._embedder.embed_text(new_doc)
            refined = None
            if getattr(self._llm, "is_configured", lambda: False)():
                refined = self._llm.categorize_one(
                    title=title, channel=channel, transcript_excerpt=transcript,
                )
            self._store.update_transcript(
                hit.video_id, transcript,
                category=refined, embedding=new_vec, document=new_doc,
            )
        except Exception:  # noqa: BLE001 — never fail the answer over caching
            logger.debug("Transcript cache/refine failed for %s", hit.video_id,
                         exc_info=True)

        return transcript, True, None


# Process-wide singleton.
rag_pipeline = RAGPipeline()
