/* SURPTech coaching dashboard — frontend */
"use strict";

const SHAPES = ["box", "cylinder", "sphere", "wooden_corners"];
const SHAPE_LABEL = { box: "Box", cylinder: "Cylinder", sphere: "Sphere", wooden_corners: "Wooden corners" };
const EXPERIENCED = "Experienced (inferred)";
const C = {
  pine: "#1e2b25", soft: "#4a5a52", faint: "#7d8b84", rule: "#d9ddd6", pencil: "#b3261e",
  chalk: "#3355c4", moss: "#4f7a43", amber: "#a8681a", birch: "#cdb98d",
};
const TASK_COLORS = { box: "#1e2b25", cylinder: "#3355c4", sphere: "#a8681a", wooden_corners: "#4f7a43" };

const state = {
  view: "coach", pid: null, shape: "box", filter: "all",
  status: null, participants: [], detail: {}, focus: null, selectedMoment: null,
  cohortShape: "box", cohortMetric: "force_variability", generating: null,
};

/* ---------------- utilities ---------------- */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok && !(body && body.error)) throw new Error(`Request failed (${res.status}) for ${path}`);
  return body;
}
function fmtTime(s) {
  if (s == null || !isFinite(s)) return "?";
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  return `${m}:${(s - 60 * m).toFixed(1).padStart(4, "0")}`;
}
function plural(n, one, many) { return `${n} ${n === 1 ? one : many || one + "s"}`; }
function statusClass(s) { return s === EXPERIENCED ? "exp" : s ? "nov" : ""; }
function statusWord(s) { return s === EXPERIENCED ? "experienced" : s ? "novice" : "no label"; }
function strengthPct(z) { return Math.max(8, Math.min(100, ((z - 3.5) / 6.5) * 100 + 20)); }
function plotlyReady() { return typeof window.Plotly !== "undefined"; }
const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };
const baseLayout = () => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "IBM Plex Sans, Segoe UI, system-ui, sans-serif", size: 12, color: C.pine },
  margin: { l: 64, r: 16, t: 10, b: 44 },
  hoverlabel: { bgcolor: "#fbfbf9", bordercolor: C.rule, font: { color: C.pine } },
});

/* ---------------- routing ---------------- */
function setHash() {
  const parts = [state.view, state.pid || ""];
  if (state.view === "session") parts.push(state.shape);
  history.replaceState(null, "", "#/" + parts.join("/"));
}
function readHash() {
  const [view, pid, shape] = location.hash.replace(/^#\/?/, "").split("/");
  if (["coach", "session", "cohort", "method"].includes(view)) state.view = view;
  if (pid && /^\d{2}$/.test(pid)) state.pid = pid;
  if (shape && SHAPES.includes(shape)) state.shape = shape;
}
function go(view, extra = {}) {
  if (view !== state.view) window.scrollTo(0, 0);
  Object.assign(state, extra, { view });
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === view)));
  setHash();
  render();
}

/* ---------------- boot ---------------- */
document.addEventListener("DOMContentLoaded", async () => {
  readHash();
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => go(t.dataset.view)));
  document.querySelectorAll(".filter button").forEach((b) => b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    document.querySelectorAll(".filter button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderRail();
  }));
  await refreshStatus();
  if (state.status && state.status.ready) {
    await loadParticipants();
  }
  syncTabs();
  render();
});

function syncTabs() {
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === state.view)));
}

window.addEventListener("hashchange", () => {
  const before = `${state.view}/${state.pid}/${state.shape}`;
  readHash();
  if (`${state.view}/${state.pid}/${state.shape}` === before) return;
  syncTabs();
  renderRail();
  window.scrollTo(0, 0);
  render();
});

async function refreshStatus() {
  try { state.status = await api("/api/status"); } catch (e) { state.status = null; }
  const el = $("#pipeline-status");
  const st = state.status;
  if (!st) { el.className = "status-dot off"; el.textContent = "Server not reachable"; return; }
  if (st.pipeline.running) { el.className = "status-dot busy"; el.textContent = "Analysis running"; }
  else if (st.ready) { el.className = "status-dot"; el.textContent = `${st.recordings} recordings analysed`; }
  else { el.className = "status-dot off"; el.textContent = "Analysis not run yet"; }
}

async function loadParticipants() {
  state.participants = await api("/api/participants");
  if (!state.pid) {
    const first = state.participants.find((p) => p.has_data && !p.is_expert && p.n_moments) || state.participants.find((p) => p.has_data) || state.participants[0];
    state.pid = first ? first.pid : "01";
  }
  renderRail();
}

/* ---------------- rail ---------------- */
function renderRail() {
  const ul = $("#plist");
  const list = state.participants.filter((p) =>
    state.filter === "all" ? true : state.filter === "expert" ? p.is_expert : !p.is_expert);
  ul.innerHTML = list.map((p) => {
    const sub = !p.has_data ? '<span class="nodata">no recordings</span>'
      : p.is_expert ? "Experienced on all tasks"
      : `Novice on ${plural(p.n_novice_shapes, "task")}`;
    const marks = Array.from({ length: Math.min(p.n_moments, 8) }, (_, i) => `<i style="height:${6 + (i % 3) * 4}px"></i>`).join("");
    return `<li><button class="prow" data-pid="${p.pid}" aria-current="${p.pid === state.pid}">
      <span class="id">${p.pid}</span>
      <span class="sub">${sub}<br>${p.has_data ? plural(p.n_moments, "moment") + " flagged" : ""}</span>
      <span class="marks" aria-hidden="true">${marks}</span></button></li>`;
  }).join("") || '<li class="empty">No participants match this filter.</li>';
  const sel = $("#pselect");
  sel.innerHTML = list.map((p) => `<option value="${p.pid}" ${p.pid === state.pid ? "selected" : ""}>Participant ${p.pid}, ${p.is_expert ? "experienced" : "novice"}, ${plural(p.n_moments, "moment")}</option>`).join("");
  sel.onchange = () => { state.pid = sel.value; state.selectedMoment = null; state.focus = null; renderRail(); setHash(); render(); };
  ul.querySelectorAll(".prow").forEach((b) => b.addEventListener("click", () => {
    state.pid = b.dataset.pid; state.selectedMoment = null; state.focus = null;
    renderRail(); setHash(); render();
  }));
}

/* ---------------- render dispatch ---------------- */
async function render() {
  const main = $("#main");
  if (!state.status || !state.status.ready) {
    if (state.view === "method") return renderMethod(main);
    return renderSetup(main);
  }
  if (state.view === "coach") return renderCoach(main);
  if (state.view === "session") return renderSession(main);
  if (state.view === "cohort") return renderCohort(main);
  return renderMethod(main);
}

async function getDetail(pid, force = false) {
  if (!force && state.detail[pid]) return state.detail[pid];
  state.detail[pid] = await api(`/api/participant/${pid}`);
  return state.detail[pid];
}

/* ---------------- setup (no outputs yet) ---------------- */
function renderSetup(main) {
  const st = state.status;
  $("#plist").innerHTML = '<li class="empty" style="padding:6px 4px;font-size:13px">Participants appear here after the analysis has run.</li>';
  if (!st) {
    main.innerHTML = `<div class="setup"><h1>The dashboard server is not responding</h1><p>Start it with <b>python app.py</b> in the project folder, then reload this page.</p></div>`;
    return;
  }
  const running = st.pipeline.running;
  const failed = st.pipeline.returncode != null && st.pipeline.returncode !== 0;
  main.innerHTML = `<div class="setup">
    <h1>Analyse the recordings to begin</h1>
    <p>The coaching view needs the analysis outputs. Put each participant's four recordings in
    <b>data/01</b> to <b>data/21</b> (box.pts, cylinder.pts, sphere.pts, wooden_corners.pts), then run the analysis.</p>
    <div class="surface">
      <p style="margin-top:0">Found <b>${st.participant_folders}</b> of 21 participant folders and <b>${st.recordings}</b> of 84 recordings in <code>${esc(st.data_dir)}</code>.</p>
      ${st.missing.length ? `<p>Missing outputs: ${st.missing.map(esc).join(", ")}</p>` : ""}
      <button class="btn" id="run-pipeline" ${running || !st.recordings ? "disabled" : ""}>
        ${running ? '<span class="spinner"></span> Running analysis' : "Run analysis"}</button>
      ${!st.recordings ? '<div class="callout info">Add the data folders first; the button turns on when recordings are found.</div>' : ""}
      ${failed ? '<div class="callout err">The last analysis run stopped with an error. The log below shows which step failed.</div>' : ""}
      ${st.pipeline.log ? `<pre class="log" id="log">${esc(st.pipeline.log)}</pre>` : ""}
    </div></div>`;
  const btn = $("#run-pipeline");
  if (btn) btn.addEventListener("click", startPipeline);
  if (running) pollPipeline();
}

async function startPipeline() {
  await api("/api/pipeline/run", { method: "POST" });
  await refreshStatus();
  render();
  pollPipeline();
}

let pollTimer = null;
function pollPipeline() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    await refreshStatus();
    if (state.status && state.status.pipeline.running) {
      if (!state.status.ready && state.view !== "method") renderSetup($("#main"));
      const log = $("#log");
      if (log) { log.textContent = state.status.pipeline.log; log.scrollTop = log.scrollHeight; }
      return pollPipeline();
    }
    state.detail = {};
    if (state.status && state.status.ready) await loadParticipants();
    render();
  }, 2000);
}

/* ---------------- coaching view ---------------- */
async function renderCoach(main) {
  const pid = state.pid;
  main.innerHTML = '<div class="loading">Loading participant…</div>';
  let d;
  try { d = await getDetail(pid); } catch (e) { main.innerHTML = `<div class="callout err">${esc(e.message)}</div>`; return; }
  if (pid !== state.pid || state.view !== "coach") return;

  const novShapes = d.shapes.filter((s) => s.status && s.status !== EXPERIENCED);
  const haveData = d.shapes.some((s) => s.n_windows);
  let summary;
  if (!haveData) summary = "No analysed recordings were found for this participant.";
  else if (d.is_expert) summary = `Grouped with the experienced participants on all four tasks. ${d.moments.length ? `${plural(d.moments.length, "moment")} still stood out from the others.` : "No moment stood out strongly."}`;
  else summary = `Grouped with the novices on ${novShapes.map((s) => s.label.toLowerCase()).join(", ") || "no task"}. ${d.moments.length ? `The sensors point to ${plural(d.moments.length, "moment")} worth reviewing, each with the time it happened.` : "No single moment stood out strongly from experienced participants."}`;

  const prob = d.prediction && d.prediction.probability_expert;
  main.innerHTML = `
    <section class="phead">
      <div>
        <h1>Participant ${esc(pid)}</h1>
        <p>${esc(summary)}</p>
        <div class="chips">${d.shapes.map((s) => `<span class="chip ${statusClass(s.status)}">${esc(s.label)}: ${statusWord(s.status)}</span>`).join("")}</div>
      </div>
      ${prob != null ? `<div class="gauge"><div class="lbl">Model's estimate of experienced-like</div>
        <div class="val num">${Math.round(prob * 100)}%</div><div class="bar"><i style="width:${Math.round(prob * 100)}%"></i></div></div>` : ""}
    </section>

    <section class="block">
      <header><div><h2>Where in each task</h2>
        <p class="note">Each bar is one recording, marked in seconds from its start. Red marks span the periods where the sensors differed most from experienced participants on the same task; the dot is the sharpest point. Select a mark to read it.</p></div></header>
      <div class="surface rulers" id="rulers"></div>
    </section>

    <section class="block coach">
      <div>
        <header style="margin-bottom:12px"><h2>Moments to review</h2></header>
        <div class="surface"><ul class="moments" id="moment-list"></ul></div>
      </div>
      <div class="surface feedback" id="feedback"></div>
    </section>`;

  renderRulers($("#rulers"), d);
  renderMomentList($("#moment-list"), d.moments, { showTask: true, limit: 6 });
  renderFeedback($("#feedback"), d);
}

function tickSpacing(duration) {
  if (duration <= 40) return [1, 5, 10];
  if (duration <= 120) return [5, 10, 30];
  if (duration <= 400) return [10, 30, 60];
  return [30, 60, 300];
}

function renderRulers(el, d) {
  const rows = d.shapes.map((s) => {
    const dur = s.duration_s;
    if (!dur) return `<div class="ruler-row"><div class="ruler-name">${esc(s.label)}</div><div class="ruler-empty">No analysed recording</div></div>`;
    const [small, mid, big] = tickSpacing(dur);
    let ticks = "";
    for (let t = 0; t <= dur + 1e-6; t += small) {
      const cls = Math.abs(t % big) < 1e-6 ? "l" : Math.abs(t % mid) < 1e-6 ? "m" : "s";
      const left = (t / dur) * 100;
      ticks += `<i class="tick ${cls}" style="left:${left}%"></i>`;
      if (cls !== "s" && t > 0 && left < 97) ticks += `<span class="tick-label" style="left:${left}%">${fmtTime(t).replace(/\.0$/, "")}</span>`;
    }
    const ms = d.moments.filter((m) => m.object_type === s.shape);
    const bands = ms.filter(isThroughout).length
      ? `<span class="band" style="left:${(Math.min(...ms.filter(isThroughout).map((m) => m.start_s)) / dur) * 100}%;right:${100 - (Math.max(...ms.filter(isThroughout).map((m) => m.end_s)) / dur) * 100}%" title="Pattern present through most of the task"></span>` : "";
    const marks = bands + ms.map((m) => {
      const [a, b] = focusRange(m);
      const left = (a / dur) * 100;
      const width = Math.max(0.6, ((b - a) / dur) * 100);
      const pin = m.peak_s != null ? ((m.peak_s - a) / Math.max(b - a, 1e-6)) * 100 : 50;
      const key = momentKey(m);
      return `<button class="mark" data-key="${key}" aria-pressed="${state.selectedMoment === key}"
        style="left:${left}%;width:${width}%" title="${esc(m.metric_label)}: ${esc(whenText(m))}"
        aria-label="${esc(m.metric_label)}, ${esc(whenText(m))}"><span class="pin" style="left:${Math.min(100, Math.max(0, pin))}%"></span></button>`;
    }).join("");
    return `<div class="ruler-row">
      <div class="ruler-name">${esc(s.label)}<small>${fmtTime(dur)} long, ${statusWord(s.status)}${ms.length ? `, ${plural(ms.length, "mark")}` : ""}</small></div>
      <div class="ruler">${ticks}${marks}</div></div>`;
  }).join("");
  el.innerHTML = rows + `<div class="ruler-legend"><span><i class="k1"></i>period that stood out</span><span><i class="k2"></i>sharpest point</span><span><i class="k3"></i>pattern through most of the task</span></div>`;
  el.querySelectorAll(".mark").forEach((b) => b.addEventListener("click", () => selectMoment(b.dataset.key, true)));
}

function isThroughout(m) { return m.pattern === "throughout"; }
function focusRange(m) {
  if (isThroughout(m) && distinctWorst(m)) return [m.worst_start_s, m.worst_end_s];
  if (isThroughout(m) && m.peak_s != null) return [Math.max(m.start_s, m.peak_s - 1), Math.min(m.end_s, m.peak_s + 1)];
  return [m.start_s, m.end_s];
}
function distinctWorst(m) {
  return m.worst_start_s != null && (m.worst_end_s - m.worst_start_s) < 0.6 * (m.end_s - m.start_s);
}
function whenText(m) {
  if (!isThroughout(m)) return `${m.start_label}–${m.end_label}`;
  return distinctWorst(m) ? `Throughout the task, worst ${m.worst_start_label}–${m.worst_end_label}` : "Throughout the task";
}
function momentKey(m) { return `${m.object_type}|${m.metric}|${m.start_s}`; }

function selectMoment(key, scroll) {
  state.selectedMoment = key;
  const list = $("#moment-list");
  if (list && !list.querySelector(`.moment[data-key="${CSS.escape(key)}"]`) && state.detail[state.pid]) {
    renderMomentList(list, state.detail[state.pid].moments, { showTask: true, limit: 6 });
  }
  document.querySelectorAll(".mark").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
  document.querySelectorAll(".moment").forEach((li) => {
    const on = li.dataset.key === key;
    li.setAttribute("aria-current", String(on));
    if (on && scroll) li.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
}

function renderMomentList(el, moments, { showTask, limit }) {
  const all = moments;
  const sel = state.selectedMoment;
  if (limit && moments.length > limit && !el.dataset.expanded) {
    const top = moments.slice(0, limit);
    const chosen = moments.find((m) => momentKey(m) === sel);
    if (chosen && !top.includes(chosen)) top.push(chosen);
    moments = top;
  }
  if (!moments.length) {
    el.innerHTML = `<li class="empty" style="padding:18px">No period stood out strongly from experienced participants${showTask ? "" : " in this task"}. The general advice still applies.</li>`;
    return;
  }
  el.innerHTML = moments.map((m) => `
    <li class="moment" data-key="${momentKey(m)}" aria-current="${state.selectedMoment === momentKey(m)}">
      <div class="when num">${m.peak_label !== "?" ? m.peak_label : m.start_label}<small>${showTask ? esc(m.task) : "sharpest"}</small></div>
      <div>
        <h3>${esc(m.metric_label)}</h3>
        <p>${esc(capitalise(m.description))}.</p>
        <p class="tip">${esc(m.tip)}</p>
        <div class="meta num">${esc(whenText(m))}, strength <span class="strength"><i style="width:${strengthPct(m.max_abs_z)}%"></i></span>
          &nbsp; <button data-open="${momentKey(m)}">Show in timeline</button></div>
      </div></li>`).join("");
  if (limit && all.length > limit) {
    const more = document.createElement("li");
    more.className = "more";
    more.innerHTML = el.dataset.expanded
      ? `<button>Show the ${limit} strongest only</button>`
      : `<button>Show all ${all.length} moments</button>`;
    more.querySelector("button").addEventListener("click", () => {
      if (el.dataset.expanded) delete el.dataset.expanded; else el.dataset.expanded = "1";
      renderMomentList(el, all, { showTask, limit });
    });
    el.appendChild(more);
  }
  el.querySelectorAll(".moment").forEach((li) => li.addEventListener("click", (ev) => {
    if (ev.target.closest("button")) return;
    selectMoment(li.dataset.key, false);
    if (state.view === "session") zoomToMoment(moments.find((m) => momentKey(m) === li.dataset.key));
  }));
  el.querySelectorAll("button[data-open]").forEach((b) => b.addEventListener("click", () => {
    const m = moments.find((x) => momentKey(x) === b.dataset.open);
    openInTimeline(m);
  }));
}

function capitalise(s) { s = String(s || ""); return s.charAt(0).toUpperCase() + s.slice(1); }

function openInTimeline(m) {
  state.selectedMoment = momentKey(m);
  go("session", { shape: m.object_type, focus: m });
}

/* ---------------- feedback ---------------- */
function renderFeedback(el, d) {
  const fb = d.feedback || {};
  const meta = fb.meta || {};
  const hasTiming = (meta.feedback_version || 1) >= 2;
  const busy = state.generating === d.pid;
  let badge = "";
  if (fb.text) badge = hasTiming ? (meta.includes_timing ? '<span class="badge good">Includes timing</span>' : '<span class="badge">No moments to time</span>') : '<span class="badge warn">Older feedback without timing</span>';
  const unknown = meta.unknown_times || [];
  el.innerHTML = `
    <div class="feedback-head"><h2>Coaching feedback</h2>${badge}</div>
    ${fb.text ? `<div class="prose" id="fb-text">${renderMarkdown(fb.text, d.moments, unknown)}</div>` : `<p class="empty">No feedback has been written for this participant yet.</p>`}
    ${unknown.length ? `<div class="callout err">The model wrote ${plural(unknown.length, "time")} that ${unknown.length === 1 ? "is" : "are"} not in the measured moments (${unknown.map(esc).join(", ")}). Trust the moments list for timing.</div>` : ""}
    <div style="margin-top:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <button class="btn ${fb.text && hasTiming ? "ghost" : ""}" id="gen" ${busy ? "disabled" : ""}>
        ${busy ? '<span class="spinner"></span> Writing feedback' : fb.text ? (hasTiming ? "Write new feedback" : "Rewrite with timing") : "Write feedback"}</button>
      <span style="font-size:13px;color:var(--pine-faint)">${busy ? "Uses gemma3:4b in Ollama; usually under two minutes." : meta.generated_at ? `Written ${new Date(meta.generated_at).toLocaleString()}` : "Runs ollama_feedback.py with your local Ollama model."}</span>
    </div>
    <div id="gen-error"></div>`;
  $("#gen", el).addEventListener("click", () => generateFeedback(d.pid));
  el.querySelectorAll(".time-chip[data-key]").forEach((b) => b.addEventListener("click", () => {
    selectMoment(b.dataset.key, true);
  }));
}

async function generateFeedback(pid) {
  state.generating = pid;
  const d = state.detail[pid];
  if (state.view === "coach" && state.pid === pid) renderFeedback($("#feedback"), d);
  const res = await api(`/api/participant/${pid}/feedback`, { method: "POST" }).catch((e) => ({ ok: false, error: e.message }));
  state.generating = null;
  if (res && res.ok) {
    d.feedback = { text: res.text, meta: res.meta };
    const p = state.participants.find((x) => x.pid === pid);
    if (p) { p.has_feedback = true; p.feedback_has_timing = true; }
  }
  if (state.view === "coach" && state.pid === pid) {
    renderFeedback($("#feedback"), d);
    if (!res || !res.ok) $("#gen-error").innerHTML = `<div class="callout err">Feedback was not written: ${esc((res && res.error) || "unknown error")}. Check that Ollama is running and that gemma3:4b is installed (ollama pull gemma3:4b).</div>`;
  }
}

function renderMarkdown(text, moments, unknown) {
  const lookup = new Map();
  for (const m of moments) {
    for (const k of ["peak_label", "worst_start_label", "worst_end_label", "start_label", "end_label"]) if (m[k] && !lookup.has(m[k])) lookup.set(m[k], momentKey(m));
  }
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>")
    .replace(/\b(\d{1,3}:\d{2}(?:\.\d)?)\b/g, (t) => {
      const key = lookup.get(t) || lookup.get(t + ".0");
      if (key) return `<button class="time-chip" data-key="${key}" title="Show this moment">${t}</button>`;
      return `<span class="time-chip unknown" title="Not one of the measured moment times">${t}</span>`;
    });
  const lines = text.split(/\r?\n/);
  let html = "", inList = false;
  const close = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { close(); continue; }
    const plain = line.replace(/\*\*/g, "").replace(/^#+\s*/, "");
    if (/^#+\s/.test(line) || /^\d\.\s+[A-Z][A-Z ?'’]+:?$/.test(plain)) {
      close();
      const title = plain.replace(/^\d\.\s+/, "").replace(/:$/, "");
      html += `<h4>${esc(title.charAt(0) + title.slice(1).toLowerCase())}</h4>`;
      continue;
    }
    const bullet = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (bullet) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(bullet[1])}</li>`;
      continue;
    }
    close();
    html += `<p>${inline(line)}</p>`;
  }
  close();
  return html;
}

/* ---------------- session timeline ---------------- */
async function renderSession(main) {
  const pid = state.pid, shape = state.shape;
  let d;
  try { d = await getDetail(pid); } catch (e) { main.innerHTML = `<div class="callout err">${esc(e.message)}</div>`; return; }
  main.innerHTML = `
    <section class="phead"><div><h1>Participant ${esc(pid)}</h1>
      <p>Session timeline: what the sensors recorded, second by second, with the moments marked.</p></div></section>
    <div class="seg" role="group" aria-label="Task">${d.shapes.map((s) => `<button data-shape="${s.shape}" aria-pressed="${s.shape === shape}">${esc(s.label)}${s.n_moments ? `<span class="n">${s.n_moments}</span>` : ""}</button>`).join("")}</div>
    <div class="session-grid">
      <div class="surface plot"><div id="session-plot" style="height:760px"><div class="loading" style="padding:30px">Loading recording…</div></div></div>
      <div><div class="surface side-list"><ul class="moments" id="session-moments"></ul></div>
        <p class="note" style="font-size:13px;color:var(--pine-soft);margin-top:12px" id="session-note"></p></div>
    </div>`;
  main.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    state.focus = null; state.selectedMoment = null; go("session", { shape: b.dataset.shape });
  }));

  let s;
  try { s = await api(`/api/participant/${pid}/session/${shape}`); } catch (e) { $("#session-plot").innerHTML = `<div class="callout err">${esc(e.message)}</div>`; return; }
  if (pid !== state.pid || shape !== state.shape || state.view !== "session") return;
  renderMomentList($("#session-moments"), s.moments, { showTask: false });
  const ref = Object.values(s.reference || {})[0];
  $("#session-note").textContent = ref
    ? `Compared with ${ref.group} (${ref.participants} people, this participant excluded). Deviation colours: red is higher than experienced, blue is lower; colour starts at the flag threshold.`
    : "";
  drawSession(s);
  if (state.focus && state.focus.object_type === shape) zoomToMoment(state.focus);
}

function drawSession(s) {
  const el = $("#session-plot");
  if (!plotlyReady()) { el.innerHTML = '<div class="callout err">The chart library did not load. Reinstall the plotly Python package and reload.</div>'; return; }
  el.innerHTML = "";
  const traces = [];
  const metrics = Object.keys(s.metrics || {}).filter((m) => s.windows[m]);
  const zthr = s.z_threshold || 3.5;
  if (metrics.length && s.windows.mid_s) {
    const z = metrics.map((m) => s.windows[`z_${m}`].map((v) => (v == null ? null : Math.max(-8, Math.min(8, v)))));
    const text = metrics.map((m) => s.windows.mid_s.map((_, i) =>
      `${s.metrics[m].label}<br>${fmtTime(s.windows.start_s[i])}–${fmtTime(s.windows.end_s[i])}<br>` +
      (s.windows[`z_${m}`][i] == null ? "no reference" : `${s.windows[`z_${m}`][i] > 0 ? "above" : "below"} experienced by ${Math.abs(s.windows[`z_${m}`][i]).toFixed(1)} spreads`)));
    const t = zthr / 8 / 2;
    traces.push({
      type: "heatmap", x: s.windows.mid_s, y: metrics.map((m) => s.metrics[m].label), z, text, hoverinfo: "text",
      zmin: -8, zmax: 8, xaxis: "x", yaxis: "y",
      colorscale: [[0, C.chalk], [0.5 - t, "#c9d3f0"], [0.5 - t + 0.0001, "#f4f5f2"], [0.5 + t - 0.0001, "#f4f5f2"], [0.5 + t, "#efc4bf"], [1, C.pencil]],
      showscale: false, xgap: 0, ygap: 2,
    });
  }
  const sig = s.signals;
  if (sig) {
    const lane = (key, axis, name, color) => traces.push({
      type: "scattergl", mode: "lines", x: sig.elapsed, y: sig[key], xaxis: "x", yaxis: axis, name,
      line: { color, width: 1.1 }, hovertemplate: `%{x:.1f} s<br>${name}: %{y:.3g}<extra></extra>`,
    });
    lane("force", "y2", "Pressure |F|", C.pine);
    lane("torque", "y3", "Twisting |T|", C.soft);
    lane("speed", "y4", "Speed", C.amber);
  }
  const shapes = [];
  const annotations = [];
  const labelled = [];
  const minGap = Math.max(0.8, (s.duration_s || 60) * 0.025);
  for (const m of s.moments) {
    const [a, b] = focusRange(m);
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: a, x1: b, y0: 0, y1: 0.745, fillcolor: "rgba(179,38,30,0.09)", line: { width: 0 }, layer: "below" });
    if (m.peak_s != null) shapes.push({ type: "line", xref: "x", yref: "paper", x0: m.peak_s, x1: m.peak_s, y0: 0, y1: 0.745, line: { color: C.pencil, width: 1.2, dash: "dot" } });
    const at = m.peak_s ?? m.start_s;
    if (!labelled.some((x) => Math.abs(x - at) < minGap)) {
      labelled.push(at);
      annotations.push({ x: at, y: 0.755, xref: "x", yref: "paper", text: m.peak_label, showarrow: false, font: { size: 11, color: C.pencil }, yanchor: "bottom" });
    }
  }
  const forceRef = s.reference && s.reference.force_level;
  if (forceRef && forceRef.q25 != null && sig) {
    shapes.push({ type: "rect", xref: "paper", yref: "y2", x0: 0, x1: 1, y0: forceRef.q25, y1: forceRef.q75, fillcolor: "rgba(51,85,196,0.10)", line: { width: 0 }, layer: "below" });
    annotations.push({ xref: "paper", yref: "paper", x: 0, y: 0.742, text: '<span style="color:#3355c4">■</span> blue band: typical pressure of experienced participants', showarrow: false, xanchor: "left", yanchor: "top", font: { size: 10.5, color: C.soft }, bgcolor: "rgba(251,251,249,.85)" });
  }
  const yax = (domain, title) => ({ domain, title: { text: title, font: { size: 11.5, color: C.soft } }, gridcolor: "#eceee9", zeroline: false, fixedrange: true });
  const layout = Object.assign(baseLayout(), {
    showlegend: false, height: 760, margin: { l: 150, r: 16, t: 14, b: 48 },
    xaxis: { title: { text: "Seconds from the start of the recording", font: { size: 12, color: C.soft } }, gridcolor: "#eceee9", zeroline: false, anchor: "y4", range: s.duration_s ? [0, s.duration_s] : undefined },
    yaxis: { domain: [0.80, 1], automargin: true, tickfont: { size: 11 }, fixedrange: true },
    yaxis2: yax([0.54, 0.74], "Pressure"),
    yaxis3: yax([0.28, 0.48], "Twisting"),
    yaxis4: yax([0, 0.22], "Speed"),
    shapes, annotations,
  });
  if (!sig) annotations.push({ xref: "paper", yref: "paper", x: 0.5, y: 0.35, text: s.error || "Raw recording not available", showarrow: false, font: { color: C.faint } });
  Plotly.newPlot(el, traces, layout, plotConfig);
}

function zoomToMoment(m) {
  if (!m || !plotlyReady()) return;
  const el = $("#session-plot");
  if (!el || !el.data) return;
  const [a, b] = focusRange(m);
  const pad = Math.max(4, (b - a) * 0.8);
  Plotly.relayout(el, { "xaxis.range": [Math.max(0, a - pad), b + pad] });
  selectMoment(momentKey(m), true);
}

/* ---------------- cohort ---------------- */
async function renderCohort(main) {
  main.innerHTML = '<div class="loading">Loading cohort…</div>';
  const c = await api(`/api/cohort?shape=${state.cohortShape}&metric=${state.cohortMetric}`);
  if (state.view !== "cohort") return;
  const metricOptions = Object.entries(c.metrics).map(([k, v]) => `<option value="${k}" ${k === state.cohortMetric ? "selected" : ""}>${esc(v.label)}</option>`).join("");
  main.innerHTML = `
    <section class="phead"><div><h1>Cohort</h1>
      <p>How participant ${esc(state.pid)} compares with everyone else on the same task. The cohort is this study only, not a clinical norm.</p></div></section>
    <div class="controls">
      <label>Task<select id="c-shape">${SHAPES.map((s) => `<option value="${s}" ${s === state.cohortShape ? "selected" : ""}>${SHAPE_LABEL[s]}</option>`).join("")}</select></label>
      <label>Behaviour<select id="c-metric">${metricOptions}</select></label>
    </div>
    <section class="block"><header><div><h2>${esc((c.metrics[state.cohortMetric] || {}).label || "")} on the ${esc(SHAPE_LABEL[state.cohortShape].toLowerCase())} task</h2>
      <p class="note">Dot: the participant's median over all 2-second windows. Line: their middle half of windows. Dashed line: median of the experienced group. Units: ${esc((c.metrics[state.cohortMetric] || {}).unit || "")}.</p></div></header>
      <div class="surface plot"><div id="c-dots" style="height:380px"></div></div></section>
    <section class="block two">
      <div><header><h2>Moments flagged per participant</h2></header><div class="surface plot"><div id="c-moments" style="height:360px"></div></div></div>
      <div><header><h2>What the model weighs most</h2></header><div class="surface plot"><div id="c-shap" style="height:360px"></div></div></div>
    </section>
    ${c.classifiers.length ? `<section class="block"><header><div><h2>Classifier comparison</h2><p class="note">Cross-validated scores for reproducing the inferred labels (not verified expertise).</p></div></header>
      <div class="surface" style="padding:6px 8px"><table class="data"><thead><tr><th>Model</th><th>Balanced accuracy</th><th>Macro F1</th><th>Accuracy</th></tr></thead><tbody>
      ${c.classifiers.map((r) => `<tr><td>${esc(r.model)}</td><td>${fmtPM(r.balanced_accuracy_mean, r.balanced_accuracy_std)}</td><td>${fmtPM(r.macro_f1_mean, r.macro_f1_std)}</td><td>${fmtPM(r.accuracy_mean, r.accuracy_std)}</td></tr>`).join("")}
      </tbody></table></div></section>` : ""}`;
  $("#c-shape").addEventListener("change", (e) => { state.cohortShape = e.target.value; renderCohort(main); });
  $("#c-metric").addEventListener("change", (e) => { state.cohortMetric = e.target.value; renderCohort(main); });
  if (!plotlyReady()) return;

  const rows = [...c.participants].sort((a, b) => a.pid.localeCompare(b.pid));
  const color = (r) => (r.status === EXPERIENCED ? C.moss : r.status ? C.amber : C.faint);
  const sel = rows.map((r) => r.pid === state.pid);
  Plotly.newPlot("c-dots", [{
    type: "scatter", mode: "markers", x: rows.map((r) => r.pid), y: rows.map((r) => r.median),
    error_y: { type: "data", symmetric: false, array: rows.map((r) => r.q75 - r.median), arrayminus: rows.map((r) => r.median - r.q25), color: "#b9c0b6", thickness: 1.5, width: 0 },
    marker: { size: sel.map((s) => (s ? 15 : 10)), color: rows.map(color), line: { color: sel.map((s) => (s ? C.pine : "#fff")), width: sel.map((s) => (s ? 2.5 : 1)) } },
    customdata: rows.map((r) => [statusWord(r.status), r.n_windows]),
    hovertemplate: "Participant %{x}<br>Median %{y:.3g}<br>%{customdata[0]}, %{customdata[1]} windows<extra></extra>",
  }], Object.assign(baseLayout(), {
    xaxis: { type: "category", title: { text: "Participant" }, gridcolor: "#eceee9" },
    yaxis: { gridcolor: "#eceee9", zeroline: false },
    shapes: c.experienced_median != null ? [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: c.experienced_median, y1: c.experienced_median, line: { color: C.chalk, dash: "dash", width: 1.5 } }] : [],
    annotations: [
      ...(c.experienced_median != null ? [{ xref: "paper", x: 1, y: c.experienced_median, text: "experienced median", showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: C.chalk, size: 11 } }] : []),
      { xref: "paper", yref: "paper", x: 0, y: 1.06, showarrow: false, xanchor: "left", text: '<span style="color:#4f7a43">●</span> experienced on this task   <span style="color:#a8681a">●</span> novice on this task', font: { size: 11.5, color: C.soft } },
    ],
    margin: { l: 56, r: 16, t: 30, b: 44 },
  }), plotConfig);

  Plotly.newPlot("c-moments", SHAPES.map((s) => ({
    type: "bar", name: SHAPE_LABEL[s], x: c.moment_counts.map((r) => r.pid), y: c.moment_counts.map((r) => r[s]),
    marker: { color: TASK_COLORS[s] }, hovertemplate: `Participant %{x}<br>${SHAPE_LABEL[s]}: %{y}<extra></extra>`,
  })), Object.assign(baseLayout(), {
    barmode: "stack", xaxis: { type: "category" }, yaxis: { gridcolor: "#eceee9", title: { text: "Moments" }, dtick: 2 },
    legend: { orientation: "h", y: 1.12, x: 0 }, margin: { l: 56, r: 10, t: 30, b: 40 },
  }), plotConfig);

  const imp = [...c.importance].reverse();
  Plotly.newPlot("c-shap", [{
    type: "bar", orientation: "h", x: imp.map((r) => r.mean_absolute_shap), y: imp.map((r) => r.feature),
    marker: { color: C.pine }, hovertemplate: "%{y}<br>mean |SHAP| %{x:.3g}<extra></extra>",
  }], Object.assign(baseLayout(), {
    xaxis: { title: { text: "Average influence on the model (mean |SHAP|)" }, gridcolor: "#eceee9" },
    yaxis: { automargin: true, tickfont: { size: 11 } }, margin: { l: 10, r: 10, t: 10, b: 44 },
  }), plotConfig);
}
function fmtPM(m, s) { return m == null ? "—" : `${m.toFixed(3)} ± ${(s || 0).toFixed(3)}`; }

/* ---------------- method & data ---------------- */
async function renderMethod(main) {
  await refreshStatus();
  const st = state.status || { pipeline: {}, outputs: {} };
  const ts = st.timing_summary || {};
  let pre = [];
  if (st.ready) { try { pre = (await api(`/api/cohort?shape=box&metric=force_level`)).preprocessing; } catch (_) { /* ignore */ } }
  main.innerHTML = `
    <section class="phead"><div><h1>Method and data</h1>
      <p>How the moments and their times are found, what they can and cannot tell you, and the state of the analysis.</p></div></section>
    <div class="prose-wide">
      <h3>How a moment gets its time</h3>
      <dl class="steps">
        <dt>1</dt><dd>Each recording is cleaned and cut into 2-second windows that overlap by 1 second, never across a gap in the timestamps (the same preprocessing as the feature pipeline).</dd>
        <dt>2</dt><dd>For every window seven behaviours are measured from the raw sensors: pressure level, pressure steadiness, sudden pressure changes, twisting steadiness, movement speed, smoothness of motion, and how much the piece is turned.</dd>
        <dt>3</dt><dd>Each window is compared with all windows of the same task from participants grouped as experienced on that task, always leaving out the person being analysed.</dd>
        <dt>4</dt><dd>A window is flagged when it is at least ${ts.z_threshold ?? 3.5} robust spreads away from the experienced median and at least ${Math.round((ts.min_relative_difference ?? 0.25) * 100)}% different. Neighbouring flagged windows join into one moment.</dd>
        <dt>5</dt><dd>The moment's start and end are the first and last flagged window. The sharpest point is the exact sample, inside the worst window, where that behaviour was most extreme.</dd>
        <dt>6</dt><dd>The language model receives these moments with their times, is told to copy the times exactly, and every time it writes is checked against the list.</dd>
      </dl>
      <h3>Read the results with care</h3>
      <ul>
        <li>Times are seconds from the first valid sample of each recording.</li>
        <li>"Experienced" and "novice" are inferred by clustering sensor variability, not verified skill labels. Moments show when someone's sensors looked least like the experienced group, which is a strong hint about when technique slipped, not proof of a mistake.</li>
        <li>Force, torque and position units are whatever the sensor recorded; comparisons are relative to other participants on the same task.</li>
        <li>Because novices were grouped by higher variability, they will naturally show more variability moments.</li>
      </ul>
      <h3>Analysis status</h3>
      <p>${st.participant_folders ?? 0} of 21 participant folders, ${st.recordings ?? 0} of 84 recordings found in <code>${esc(st.data_dir || "data")}</code>. ${st.ready ? `${ts.windows ?? "?"} windows and ${ts.moments ?? "?"} moments in the current results.` : "The analysis has not produced all outputs yet."}</p>
      <button class="btn ghost" id="rerun" ${st.pipeline.running ? "disabled" : ""}>${st.pipeline.running ? '<span class="spinner" style="border-color:rgba(30,43,37,.25);border-top-color:#1e2b25"></span> Running analysis' : "Run the analysis again"}</button>
      ${st.pipeline.log ? `<pre class="log" id="log" style="margin-top:14px">${esc(st.pipeline.log)}</pre>` : ""}
      ${pre.length ? `<h3>Preprocessing summary</h3><div class="surface" style="padding:6px 8px"><table class="data"><thead><tr><th>Recording</th><th>Clean samples</th><th>Continuous segments</th><th>Windows</th></tr></thead><tbody>
        ${pre.map((r) => `<tr><td>${esc(r.recording_id)}</td><td>${(r.cleaned_samples ?? 0).toLocaleString()}</td><td>${r.continuous_segments}</td><td>${r.windows}</td></tr>`).join("")}</tbody></table></div>` : ""}
    </div>`;
  $("#rerun").addEventListener("click", async () => { await startPipeline(); go("method"); });
  if (st.pipeline.running) pollPipeline();
}
