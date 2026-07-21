/* ============================================================================
   My YouTube Guru — frontend logic (vanilla JS, no framework)
   Talks to the FastAPI backend under /api/*. All data rendering is driven by
   the exact response shapes defined in app/models/schemas.py.
   ========================================================================== */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* Small fetch helper: JSON in/out, throws a useful Error on non-2xx so the
   UI can show the backend's `detail` message. */
async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, headers: {} };
  if (form) {
    opts.body = form; // FormData — let the browser set the multipart boundary
  } else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`/api${path}`, opts);
  let data = null;
  try { data = await res.json(); } catch { /* some errors have no body */ }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || `Request failed (${res.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function toast(message, ms = 2600) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.hidden = true), ms);
}

/* ── Tab navigation ─────────────────────────────────────────────────────── */
function showTab(name) {
  $$(".nav__item").forEach((b) => b.classList.toggle("is-active", b.dataset.tab === name));
  $$(".view").forEach((v) => v.classList.toggle("is-active", v.id === `view-${name}`));
  if (name === "knowledge") loadKnowledgeBase();
  if (name === "settings") loadSettings();
  if (name === "chat") loadSessions();
  if (name === "grounding") loadEvaluation();
}
$$(".nav__item").forEach((btn) => btn.addEventListener("click", () => showTab(btn.dataset.tab)));

/* Switch to the Ask view without re-fetching sessions (used when opening or
   starting a chat from the rail, which is visible on every tab). */
function activateChatTab() {
  $$(".nav__item").forEach((b) => b.classList.toggle("is-active", b.dataset.tab === "chat"));
  $$(".view").forEach((v) => v.classList.toggle("is-active", v.id === "view-chat"));
}

/* ── Startup: health + setup hint ───────────────────────────────────────── */
let hasKey = false;
let kbCount = 0;

async function refreshStatus() {
  const dot = $("#statusDot");
  const text = $("#statusText");
  try {
    await api("/health");
    const s = await api("/settings/llm");
    hasKey = !!s.configured;
    const stats = await api("/knowledge-base/stats");
    kbCount = stats.total_videos;

    if (!hasKey) { dot.className = "dot is-warn"; text.textContent = "no API key"; }
    else if (kbCount === 0) { dot.className = "dot is-warn"; text.textContent = "no videos yet"; }
    else { dot.className = "dot is-ok"; text.textContent = `${kbCount} videos`; }

    renderSetupHint();
  } catch (err) {
    dot.className = "dot is-err";
    text.textContent = "backend offline";
  }
}

function renderSetupHint() {
  const hint = $("#setupHint");
  if (!hasKey) {
    hint.hidden = false;
    hint.innerHTML = `Add your provider API key in <b>Settings</b> to enable categorisation and answers.`;
  } else if (kbCount === 0) {
    hint.hidden = false;
    hint.innerHTML = `Your knowledge base is empty — import your history from <b>Add data</b> to begin.`;
  } else {
    hint.hidden = true;
  }
}

/* ── Settings ───────────────────────────────────────────────────────────── */
async function loadSettings() {
  try {
    const s = await api("/settings/llm");
    hasKey = !!s.configured;
    const badge = $("#keyBadge");
    badge.textContent = s.configured ? "key set" : "no key";
    badge.className = "badge " + (s.configured ? "is-ok" : "is-off");
    $("#provText").textContent = `${s.provider} · ${s.model}`;
    $("#provider").value = s.provider || "";
    $("#model").value = s.model || "";
    $("#baseUrl").value = s.base_url || "";
  } catch (err) {
    toast(err.message);
  }
}

$("#settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#settingsMsg");
  const payload = {};
  const key = $("#apiKey").value.trim();
  const prov = $("#provider").value.trim();
  const model = $("#model").value.trim();
  const base = $("#baseUrl").value.trim();
  if (key) payload.api_key = key;
  if (prov) payload.provider = prov;
  if (model) payload.model = model;
  if (base) payload.base_url = base;
  if (Object.keys(payload).length === 0) {
    msg.textContent = "Enter a key (or change a field) first.";
    msg.className = "set__msg is-err";
    return;
  }
  try {
    await api("/settings/llm", { method: "POST", body: payload });
    $("#apiKey").value = "";
    msg.textContent = "Saved.";
    msg.className = "set__msg";
    await loadSettings();
    await refreshStatus();
  } catch (err) {
    msg.textContent = err.message;
    msg.className = "set__msg is-err";
  }
});

/* ── Upload ─────────────────────────────────────────────────────────────── */
let chosenFile = null;

function setFile(file) {
  chosenFile = file || null;
  $("#uploadBtn").disabled = !chosenFile;
  $("#fileName").textContent = chosenFile ? chosenFile.name : "";
  $("#dropTitle").textContent = chosenFile ? "File ready" : "Choose a Takeout .zip";
}

$("#fileInput").addEventListener("change", (e) => setFile(e.target.files[0]));

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-drag"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("is-drag"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

$("#uploadBtn").addEventListener("click", async () => {
  if (!chosenFile) return;
  const btn = $("#uploadBtn");
  const progress = $("#progress");
  const fill = $("#progressFill");
  const ptext = $("#progressText");
  const result = $("#uploadResult");

  btn.disabled = true;
  result.hidden = true;
  progress.hidden = false;
  fill.style.width = "0%";
  ptext.textContent = "Uploading…";

  const form = new FormData();
  form.append("file", chosenFile);

  try {
    const start = await api("/upload/takeout", { method: "POST", form });
    if (!start.categorization_enabled) {
      toast("No API key set — videos will import without categories.");
    }
    ptext.textContent = `Parsed ${start.unique_videos} videos. Ingesting…`;
    await pollJob(start.job_id, fill, ptext, result);
  } catch (err) {
    progress.hidden = true;
    result.hidden = false;
    result.innerHTML = `<div class="hint">${escapeHtml(err.message)}</div>`;
    btn.disabled = false;
  }
});

async function pollJob(jobId, fill, ptext, result) {
  while (true) {
    await sleep(1000);
    const st = await api(`/upload/status/${jobId}`);
    const pct = st.total > 0 ? Math.round((st.done / st.total) * 100) : 0;
    fill.style.width = `${pct}%`;
    ptext.textContent = st.status === "running"
      ? `${st.phase} — ${st.done}/${st.total} (${pct}%)`
      : st.status;

    if (st.status === "done") {
      renderUploadResult(result, st.result);
      $("#uploadBtn").disabled = false;
      setFile(null);
      $("#fileInput").value = "";
      await refreshStatus();
      return;
    }
    if (st.status === "error") {
      result.hidden = false;
      result.innerHTML = `<div class="hint">Import failed: ${escapeHtml(st.error || "unknown error")}</div>`;
      $("#uploadBtn").disabled = false;
      return;
    }
  }
}

function renderUploadResult(el, result) {
  const ing = result.ingest_stats || {};
  const parse = result.parse_stats || {};
  const cell = (num, label) =>
    `<div class="rstat"><div class="rstat__num">${num}</div><div class="rstat__label">${label}</div></div>`;
  el.hidden = false;
  el.innerHTML = `
    <div class="card__title">Import complete</div>
    <div class="result__grid">
      ${cell(fmt(parse.unique_videos ?? "—"), "parsed (unique)")}
      ${cell(fmt(ing.added ?? 0), "added to index")}
      ${cell(fmt(ing.already_present ?? 0), "already had (skipped)")}
      ${cell(fmt(ing.categorised ?? 0), "categorised")}
    </div>`;
}

/* ── Chat / RAG ─────────────────────────────────────────────────────────── */
const chatLog = $("#chatLog");
let activeSessionId = null; // null = a fresh, not-yet-saved chat
let convo = [];             // running {role, content} turns, sent as follow-up context

const CHAT_EMPTY_HTML = `
  <div class="empty" id="chatEmpty">
    <p class="empty__title">Nothing asked yet.</p>
    <p class="empty__body">Try something you know you've watched about.</p>
    <div class="examples">
      <button class="chip">What have I learned about vector databases?</button>
      <button class="chip">Summarize what I've watched on productivity.</button>
      <button class="chip">Which videos covered prompt engineering?</button>
    </div>
  </div>`;

function wireChips() {
  $$("#chatLog .chip").forEach((c) =>
    c.addEventListener("click", () => { $("#askInput").value = c.textContent; submitQuestion(); }));
}
wireChips();

$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); submitQuestion(); });
$("#newChat").addEventListener("click", newChat);

/* ── Sessions sidebar (persisted conversation history) ──────────────────── */
async function loadSessions() {
  try {
    const { sessions } = await api("/sessions");
    renderSessionsList(sessions);
  } catch { /* backend offline is surfaced by the status dot */ }
}

function renderSessionsList(sessions) {
  const list = $("#sessionsList");
  list.innerHTML = "";
  if (!sessions.length) {
    list.innerHTML = `<div class="sessions__empty">No chats yet.<br>Ask something to start one.</div>`;
    return;
  }
  sessions.forEach((s) => list.appendChild(sessionRow(s)));
}

function sessionRow(s) {
  const row = document.createElement("div");
  row.className = "srow" + (s.id === activeSessionId ? " is-active" : "");
  row.dataset.id = s.id;

  const title = document.createElement("span");
  title.className = "srow__title";
  title.textContent = s.title;

  const rename = document.createElement("button");
  rename.className = "srow__btn"; rename.title = "Rename"; rename.textContent = "✎";
  rename.addEventListener("click", (e) => { e.stopPropagation(); beginRename(row, s); });

  const del = document.createElement("button");
  del.className = "srow__btn is-danger"; del.title = "Delete"; del.textContent = "🗑";
  del.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(s); });

  row.append(title, rename, del);
  row.addEventListener("click", () => openSession(s.id));
  return row;
}

function beginRename(row, s) {
  const input = document.createElement("input");
  input.className = "srow__edit";
  input.value = s.title;
  row.replaceChild(input, row.querySelector(".srow__title"));
  input.focus();
  input.select();
  let done = false;
  const commit = async (save) => {
    if (done) return;
    done = true;
    const next = input.value.trim();
    if (save && next && next !== s.title) {
      try { await api(`/sessions/${s.id}`, { method: "PATCH", body: { title: next } }); }
      catch (err) { toast(err.message); }
    }
    await loadSessions();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(true); }
    else if (e.key === "Escape") commit(false);
  });
  input.addEventListener("blur", () => commit(true));
}

async function deleteSession(s) {
  if (!confirm(`Delete "${s.title}"? This can't be undone.`)) return;
  try {
    await api(`/sessions/${s.id}`, { method: "DELETE" });
    if (s.id === activeSessionId) newChat();
    await loadSessions();
  } catch (err) { toast(err.message); }
}

function newChat() {
  activeSessionId = null;
  convo = [];
  activateChatTab();
  chatLog.innerHTML = CHAT_EMPTY_HTML;
  wireChips();
  $$("#sessionsList .srow").forEach((r) => r.classList.remove("is-active"));
  $("#askInput").focus();
}

async function openSession(id) {
  try {
    const s = await api(`/sessions/${id}`);
    activeSessionId = id;
    convo = (s.messages || []).map((m) => ({ role: m.role, content: (m.content || "").slice(0, 1500) }));
    activateChatTab();
    chatLog.innerHTML = "";
    if (s.messages && s.messages.length) s.messages.forEach(renderStoredMessage);
    else { chatLog.innerHTML = CHAT_EMPTY_HTML; wireChips(); }
    $$("#sessionsList .srow").forEach((r) => r.classList.toggle("is-active", r.dataset.id === id));
    scrollChat();
  } catch (err) { toast(err.message); }
}

/* Re-render a stored turn. Assistant turns are rebuilt from their saved data
   (grounded flag + sources), but without the live thinking panel or the
   interactive confirm buttons — it's history, shown as it happened. */
function renderStoredMessage(m) {
  if (m.role === "user") { addUserMessage(m.content); return; }
  const bot = document.createElement("div");
  bot.className = "msg msg--bot";
  chatLog.appendChild(bot);
  if (m.data) {
    appendAssistantAnswer(bot, m.data);
  } else {
    const b = document.createElement("div");
    b.className = "msg__bubble";
    b.textContent = m.content;
    bot.appendChild(b);
  }
}

/* Create the session on first message so empty chats never clutter the list. */
async function ensureSession() {
  if (activeSessionId) return activeSessionId;
  const s = await api("/sessions", { method: "POST", body: {} });
  activeSessionId = s.id;
  return s.id;
}

async function persistMessage(role, content, data = null) {
  if (!activeSessionId) return;
  try {
    await api(`/sessions/${activeSessionId}/messages`, {
      method: "POST", body: { role, content, data },
    });
  } catch { /* history-save is non-fatal; the chat itself still works */ }
}

function addUserMessage(text) {
  const empty = $("#chatEmpty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "msg msg--user";
  el.innerHTML = `<div class="msg__bubble"></div>`;
  el.querySelector(".msg__bubble").textContent = text;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

/* Read a Server-Sent Events stream from a POST endpoint, invoking onEvent for
   each {kind, data} frame. Uses fetch + ReadableStream because EventSource is
   GET-only and we send a JSON body. */
async function streamSSE(path, body, onEvent) {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let detail = `Request failed (${res.status})`;
    try { const d = await res.json(); detail = d.detail || detail; } catch { /* no body */ }
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 2);
      if (frame.startsWith("data:")) {
        const payload = frame.slice(5).trim();
        if (payload) onEvent(JSON.parse(payload));
      }
    }
  }
}

/* The live "thinking" panel inside a bot message. While the pipeline works it
   shows a spinner + the current step and streams detailed sub-steps; when done
   it collapses into a "Thought for Ns" summary the user can reopen. */
function createThinking(botEl) {
  const wrap = document.createElement("div");
  wrap.className = "think";
  wrap.dataset.open = "true";
  wrap.dataset.state = "working";
  wrap.innerHTML =
    `<button class="think__head" type="button">
       <span class="think__spin"></span>
       <span class="think__label">Thinking…</span>
       <span class="think__chev">▾</span>
     </button>
     <div class="think__body"></div>`;
  botEl.appendChild(wrap);
  const label = wrap.querySelector(".think__label");
  const bodyEl = wrap.querySelector(".think__body");
  wrap.querySelector(".think__head").addEventListener("click", () => {
    wrap.dataset.open = wrap.dataset.open === "true" ? "false" : "true";
  });
  const vids = {}; // video_id -> step element, so fetching→ready updates in place

  function addStep(icon, textStr, cls = "") {
    const s = document.createElement("div");
    s.className = "tstep " + cls;
    s.innerHTML = `<span class="tstep__icon">${icon}</span><span class="tstep__text"></span>`;
    s.querySelector(".tstep__text").textContent = textStr;
    bodyEl.appendChild(s);
    scrollChat();
    return s;
  }

  return {
    setLabel(t) { label.textContent = t; },
    step(t) { addStep("·", t); },
    videos(list, message) {
      const s = addStep("⑃", message);
      const box = document.createElement("div");
      box.className = "tvids";
      (list || []).forEach((v) => {
        const li = document.createElement("div");
        li.className = "tvid";
        li.innerHTML = `<span class="tvid__title"></span><span class="tvid__sim mono">sim ${Number(v.similarity).toFixed(2)}</span>`;
        li.querySelector(".tvid__title").textContent = v.title || v.video_id;
        box.appendChild(li);
      });
      s.appendChild(box);
      scrollChat();
    },
    transcript(d) {
      const icons = { fetching: "◌", ready: "✓", cached: "✓", unavailable: "—" };
      const line = `${d.title || d.video_id} — ${d.message}`;
      let el = vids[d.video_id];
      if (!el) { el = addStep(icons[d.status] || "·", line, "tstep--tx"); vids[d.video_id] = el; }
      else {
        el.querySelector(".tstep__icon").textContent = icons[d.status] || "·";
        el.querySelector(".tstep__text").textContent = line;
      }
      el.dataset.status = d.status;
    },
    finish(elapsedS, failed = false) {
      wrap.dataset.state = failed ? "error" : "done";
      wrap.dataset.open = "false"; // collapse but stay reopenable
      label.textContent = failed ? "Stopped" : `Thought for ${elapsedS}s`;
    },
  };
}

function scrollChat() { chatLog.scrollTop = chatLog.scrollHeight; }

/* Render Markdown answers to sanitised HTML so they look like a proper
   assistant reply (headings, bold, lists, quotes) — like ChatGPT/Claude.
   Self-contained (no external libraries): every character is HTML-escaped
   first, then only known-safe tags are added, so nothing a transcript or the
   model produces can inject live HTML. */
function renderMarkdown(el, mdText) {
  try { el.innerHTML = mdToHtml(mdText || ""); }
  catch { el.textContent = mdText || ""; }
}

function mdToHtml(src) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // Pull out fenced ``` code blocks first (no inline processing inside them).
  const blocks = [];
  src = src.replace(/```[^\n]*\n([\s\S]*?)```/g, (_, code) => {
    blocks.push(esc(code.replace(/\n$/, "")));
    return `\u0000CB${blocks.length - 1}\u0000`;
  });

  // Inline spans: escape, then bold / italic / links / `code`.
  const inline = (line) => {
    let t = esc(line);
    const codes = [];
    t = t.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0001${codes.length - 1}\u0001`; });
    t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, txt, url) =>
      `<a href="${/^(https?:|mailto:|\/)/i.test(url) ? url : "#"}" target="_blank" rel="noopener">${txt}</a>`);
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/__([^_]+)__/g, "<strong>$1</strong>");
    t = t.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>").replace(/(^|[^_\w])_([^_\n]+)_/g, "$1<em>$2</em>");
    t = t.replace(/\u0001(\d+)\u0001/g, (_, n) => `<code>${codes[n]}</code>`);
    return t;
  };

  const lines = src.split(/\r?\n/);
  const out = [];
  const isBlock = (l) => /^\s*$/.test(l) || /^(#{1,6})\s+/.test(l) || /^\s*>/.test(l)
    || /^\s*[-*+]\s+/.test(l) || /^\s*\d+\.\s+/.test(l)
    || /^\s*([-*_])\1{2,}\s*$/.test(l) || /^\u0000CB\d+\u0000$/.test(l);

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = line.match(/^\u0000CB(\d+)\u0000$/))) { out.push(`<pre><code>${blocks[+m[1]]}</code></pre>`); i++; continue; }
    if (/^\s*$/.test(line)) { i++; continue; }
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) { out.push(`<h${m[1].length}>${inline(m[2].trim())}</h${m[1].length}>`); i++; continue; }
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
    if (/^\s*>/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
      out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`); continue;
    }
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) { items.push(`<li>${inline(lines[i].replace(/^\s*[-*+]\s+/, ""))}</li>`); i++; }
      out.push(`<ul>${items.join("")}</ul>`); continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push(`<li>${inline(lines[i].replace(/^\s*\d+\.\s+/, ""))}</li>`); i++; }
      out.push(`<ol>${items.join("")}</ol>`); continue;
    }
    const para = [];
    while (i < lines.length && !isBlock(lines[i])) { para.push(lines[i]); i++; }
    out.push(`<p>${para.map(inline).join("<br>")}</p>`);
  }
  return out.join("\n");
}

async function submitQuestion() {
  const input = $("#askInput");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  addUserMessage(q);
  $("#askBtn").disabled = true;
  const history = convo.slice(-6);       // prior turns, before this question
  convo.push({ role: "user", content: q });
  await ensureSession();
  await persistMessage("user", q);
  await loadSessions();          // reflect the auto-title and move chat to top
  await runAnswer("/chat/ask/stream", { question: q, history }, "grounded");
  $("#askBtn").disabled = false;
}

/* Shared driver for grounded asks and general-knowledge confirms: opens a bot
   message with a thinking panel, streams the pipeline's real events into it —
   including the answer token-by-token — then finalises and persists it.
   `answerKind` picks the tag shown as the answer starts streaming. */
async function runAnswer(path, body, answerKind = "grounded") {
  const bot = document.createElement("div");
  bot.className = "msg msg--bot";
  chatLog.appendChild(bot);
  const think = createThinking(bot);
  const started = performance.now();
  const elapsed = () => ((performance.now() - started) / 1000).toFixed(1);

  let finalData = null;
  let buffer = "";
  let bubble = null;
  let mdEl = null;
  let thinkingDone = false;
  let renderQueued = false;

  const doneThinking = (failed = false) => {
    if (!thinkingDone) { think.finish(elapsed(), failed); thinkingDone = true; }
  };

  // Lazily create the answer bubble when the first token arrives (this is when
  // "thinking" ends and the answer begins — just like ChatGPT/Claude).
  const ensureBubble = () => {
    if (bubble) return;
    doneThinking();
    bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    const tag = document.createElement("span");
    if (answerKind === "general") { tag.className = "tag tag--general"; tag.textContent = "general knowledge · not your videos"; }
    else { tag.className = "tag tag--grounded"; tag.textContent = "grounded in your videos"; }
    bubble.appendChild(tag);
    mdEl = document.createElement("div");
    mdEl.className = "md";
    bubble.appendChild(mdEl);
    bot.appendChild(bubble);
  };

  // Re-render the accumulated markdown at most ~16x/sec so long answers don't
  // thrash the DOM while tokens stream in.
  const scheduleRender = () => {
    if (renderQueued) return;
    renderQueued = true;
    setTimeout(() => {
      renderQueued = false;
      if (mdEl) { renderMarkdown(mdEl, buffer); scrollChat(); }
    }, 60);
  };

  try {
    await streamSSE(path, body, (evt) => {
      const d = evt.data || {};
      if (evt.kind === "status") { think.setLabel(d.message); think.step(d.message); }
      else if (evt.kind === "retrieved") { think.setLabel(d.message); think.videos(d.videos, d.message); }
      else if (evt.kind === "transcript") { think.setLabel("Fetching transcripts…"); think.transcript(d); }
      else if (evt.kind === "answer_delta") { ensureBubble(); buffer += (d.text || ""); scheduleRender(); }
      else if (evt.kind === "final") {
        finalData = d;
        doneThinking();
        if (d.needs_confirmation) {
          renderConfirmBubble(bot, d);
        } else {
          ensureBubble();
          // Final authoritative render of the complete answer, then sources.
          renderMarkdown(mdEl, (d.answer || "").trim() || "(No answer text was returned. Try asking again.)");
          const sb = sourcesBox(d);
          if (sb) bot.appendChild(sb);
        }
      }
      else if (evt.kind === "error") { doneThinking(true); renderError(bot, d); }
    });
  } catch (err) {
    doneThinking(true);
    renderError(bot, { detail: err.message });
  }

  if (finalData) {
    if (!finalData.needs_confirmation) {
      convo.push({ role: "assistant", content: (finalData.answer || "").slice(0, 1500) });
    }
    await persistMessage("assistant", finalData.answer, finalData);
    await loadSessions();
  }
  scrollChat();
}

function renderConfirmBubble(botEl, resp) {
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.appendChild(text(resp.answer));
  const row = document.createElement("div");
  row.className = "confirm";
  const yes = button("Use general knowledge", "btn btn--primary btn--sm");
  const no = button("No thanks", "btn btn--ghost btn--sm");
  yes.addEventListener("click", () => { row.remove(); runAnswer("/chat/confirm/stream", { question: resp.question, history: convo.slice(-6) }, "general"); });
  no.addEventListener("click", () => { row.remove(); bubble.appendChild(text(" — okay, staying grounded.")); });
  row.append(yes, no);
  bubble.appendChild(row);
  botEl.appendChild(bubble);
}

function sourcesBox(resp) {
  if (!resp.sources || !resp.sources.length) return null;
  const box = document.createElement("div");
  box.className = "sources";
  const head = document.createElement("div");
  head.className = "sources__head";
  head.textContent = "sources";
  box.appendChild(head);
  resp.sources.forEach((s, i) => box.appendChild(sourceRow(s, i + 1)));
  return box;
}

function renderError(botEl, d) {
  const box = document.createElement("div");
  box.className = "msg__bubble";
  box.innerHTML = `<span class="tag tag--general">error</span><br>${escapeHtml(d.detail || "Something went wrong")}`;
  if (d.code === "no_key" || /api key/i.test(d.detail || "")) {
    box.innerHTML += `<br><br>Set your key in <b>Settings</b>, then ask again.`;
  }
  botEl.appendChild(box);
}

/* Renders a full assistant answer at once — used only when re-loading saved
   history (live answers stream in via runAnswer). Shows the grounded/general
   tag, the markdown answer, and cited sources; no interactive buttons. */
function appendAssistantAnswer(botEl, resp) {
  if (resp.needs_confirmation) {
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.appendChild(text(resp.answer));
    botEl.appendChild(bubble);
    return;
  }

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  const tag = document.createElement("span");
  if (resp.from_general_knowledge) { tag.className = "tag tag--general"; tag.textContent = "general knowledge · not your videos"; }
  else { tag.className = "tag tag--grounded"; tag.textContent = "grounded in your videos"; }
  bubble.appendChild(tag);
  const md = document.createElement("div");
  md.className = "md";
  const answerText = (resp.answer || "").trim();
  if (answerText) renderMarkdown(md, answerText);
  else md.innerHTML = '<em style="color:var(--muted)">(No answer text was returned.)</em>';
  bubble.appendChild(md);
  botEl.appendChild(bubble);

  const sb = sourcesBox(resp);
  if (sb) botEl.appendChild(sb);
}

function sourceRow(s, idx) {
  const a = document.createElement("a");
  a.className = "source";
  a.href = s.url;
  a.target = "_blank";
  a.rel = "noopener";
  const flag = s.transcript_used
    ? `sim ${s.similarity.toFixed(2)} · transcript`
    : `sim ${s.similarity.toFixed(2)} · <span class="no-tx">title only</span>`;
  a.innerHTML =
    `<span class="source__idx">[${idx}]</span>` +
    `<span class="source__title"></span>` +
    `<span class="source__meta mono">${flag}</span>`;
  a.querySelector(".source__title").textContent = s.title;
  return a;
}

/* ── Knowledge base ─────────────────────────────────────────────────────── */
let kbChart = null;
const CAT_COLORS = [
  "#e2362b", "#2f6f93", "#e0a13b", "#5b8c5a", "#8e5ea2", "#c2607f",
  "#4b9b9b", "#d2743b", "#6b7fb3", "#a0883b", "#7a9e5c", "#9b5c5c",
  "#b0546e", "#5f9ea0", "#c98a3b", "#7d7d7d",
];
let currentFilter = null;

async function loadKnowledgeBase() {
  try {
    const [stats, cats] = await Promise.all([
      api("/knowledge-base/stats"),
      api("/knowledge-base/categories"),
    ]);
    kbCount = stats.total_videos;
    $("#kbTotal").textContent = fmt(stats.total_videos);
    $("#kbCats").textContent = fmt(cats.categories.length);

    const empty = $("#kbEmpty");
    const body = $(".kb__body");
    if (stats.total_videos === 0) {
      empty.hidden = false; body.style.display = "none"; return;
    }
    empty.hidden = true; body.style.display = "";

    renderChart(cats.categories);
    loadVideos(null);
  } catch (err) {
    toast(err.message);
  }
}

function renderChart(categories) {
  const labels = categories.map((c) => c.category);
  const data = categories.map((c) => c.count);
  const colors = labels.map((_, i) => CAT_COLORS[i % CAT_COLORS.length]);

  if (kbChart) kbChart.destroy();
  const ctx = $("#kbChart").getContext("2d");
  kbChart = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: "#fff", borderWidth: 2 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "58%",
      plugins: {
        legend: { position: "right", labels: { font: { family: "IBM Plex Sans", size: 12 }, boxWidth: 12, padding: 8 } },
        tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw}` } },
      },
      onClick: (_evt, els) => {
        if (!els.length) return;
        const label = labels[els[0].index];
        loadVideos(currentFilter === label ? null : label);
      },
    },
  });
}

async function loadVideos(category) {
  currentFilter = category;
  const title = $("#kbListTitle");
  const clear = $("#kbClearFilter");
  title.textContent = category ? category : "All videos";
  clear.hidden = !category;
  try {
    const q = category ? `?category=${encodeURIComponent(category)}&limit=300` : "?limit=300";
    const res = await api(`/knowledge-base/videos${q}`);
    const list = $("#kbList");
    list.innerHTML = "";
    if (!res.videos.length) {
      list.innerHTML = `<div class="empty" style="padding:24px">No videos here.</div>`;
      return;
    }
    res.videos
      .sort((a, b) => (b.watch_count || 1) - (a.watch_count || 1))
      .forEach((v) => list.appendChild(videoRow(v)));
  } catch (err) {
    toast(err.message);
  }
}

function videoRow(v) {
  const row = document.createElement("div");
  row.className = "vrow";
  const watched = v.watch_count > 1 ? ` · watched ${v.watch_count}×` : "";
  const main = document.createElement("div");
  main.className = "vrow__main";
  main.innerHTML =
    `<div class="vrow__title"><a href="${v.url}" target="_blank" rel="noopener"></a></div>` +
    `<div class="vrow__sub mono"></div>`;
  main.querySelector("a").textContent = v.title;
  main.querySelector(".vrow__sub").textContent = `${v.channel || "unknown channel"}${watched}`;
  const cat = document.createElement("span");
  cat.className = "vrow__cat";
  cat.textContent = v.category || "Uncategorized";
  row.append(main, cat);
  return row;
}

$("#kbRefresh").addEventListener("click", loadKnowledgeBase);
$("#kbClearFilter").addEventListener("click", () => loadVideos(null));

/* ── Tiny DOM/util helpers ──────────────────────────────────────────────── */
function text(str) { return document.createTextNode(str); }
function button(label, cls) { const b = document.createElement("button"); b.className = cls; b.textContent = label; return b; }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function fmt(n) { return typeof n === "number" ? n.toLocaleString() : n; }
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

/* ── Grounding evaluation (Module 9) ────────────────────────────────────── */
const MODE_META = {
  grounded: { label: "grounded", color: "var(--good)" },
  general_knowledge: { label: "general", color: "var(--amber)" },
  no_match: { label: "no match", color: "var(--muted)" },
};

async function loadEvaluation() {
  try {
    const [m, log] = await Promise.all([
      api("/evaluation/metrics"),
      api("/evaluation/log?limit=50"),
    ]);
    const empty = $("#evalEmpty");
    const body = $(".eval__body");
    const stats = $("#evalStats");

    if (m.total_questions === 0) {
      empty.hidden = false; body.style.display = "none"; stats.innerHTML = ""; return;
    }
    empty.hidden = true; body.style.display = "";

    const cell = (num, label) =>
      `<div class="rstat"><div class="rstat__num">${fmt(num)}</div><div class="rstat__label">${label}</div></div>`;
    stats.innerHTML =
      cell(m.total_questions, "questions asked") +
      cell(m.grounded_pct + "%", "grounded in your videos") +
      cell(m.general_knowledge_pct + "%", "general knowledge") +
      cell(m.no_match_pct + "%", "no match found") +
      cell(m.avg_sources, "avg sources / answer") +
      cell(m.transcript_coverage_pct + "%", "sources w/ transcript") +
      cell(m.avg_best_similarity, "avg top similarity");

    const logEl = $("#evalLog");
    logEl.innerHTML = "";
    log.entries.forEach((e) => logEl.appendChild(evalRow(e)));

    const topEl = $("#evalTop");
    topEl.innerHTML = "";
    if (!m.top_sources.length) topEl.innerHTML = `<div class="sessions__empty">—</div>`;
    m.top_sources.forEach((t) => topEl.appendChild(topRow(t)));
  } catch (err) {
    toast(err.message);
  }
}

function evalRow(e) {
  const row = document.createElement("div");
  row.className = "vrow";
  const meta = MODE_META[e.mode] || { label: e.mode, color: "var(--ink-2)" };
  const main = document.createElement("div");
  main.className = "vrow__main";
  main.innerHTML = `<div class="vrow__title"></div><div class="vrow__sub mono"></div>`;
  main.querySelector(".vrow__title").textContent = e.question;
  const sim = e.best_similarity != null ? ` · top sim ${e.best_similarity.toFixed(2)}` : "";
  const when = e.ts ? new Date(e.ts).toLocaleString() : "";
  main.querySelector(".vrow__sub").textContent =
    `${e.num_sources} source${e.num_sources !== 1 ? "s" : ""}${sim} · ${when}`;
  const badge = document.createElement("span");
  badge.className = "vrow__cat";
  badge.textContent = meta.label;
  badge.style.color = meta.color;
  row.append(main, badge);
  return row;
}

function topRow(t) {
  const row = document.createElement("div");
  row.className = "vrow";
  const main = document.createElement("div");
  main.className = "vrow__main";
  main.innerHTML = `<div class="vrow__title"></div>`;
  main.querySelector(".vrow__title").textContent = t.title;
  const badge = document.createElement("span");
  badge.className = "vrow__cat";
  badge.textContent = `${t.count}×`;
  row.append(main, badge);
  return row;
}

$("#evalRefresh").addEventListener("click", loadEvaluation);

/* ── Boot ───────────────────────────────────────────────────────────────── */
refreshStatus();
loadSessions();   // Ask is the default tab; populate its sidebar immediately
