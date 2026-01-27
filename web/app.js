/* global Plotly, d3 */

const DATA_ROOT = "./data";

const els = {
  problemSelect: document.getElementById("problemSelect"),
  exampleSelect: document.getElementById("exampleSelect"),
  metricSelect: document.getElementById("metricSelect"),
  highlightTag: document.getElementById("highlightTag"),
  nodeThreshold: document.getElementById("nodeThreshold"),
  topKEdges: document.getElementById("topKEdges"),

  instruction: document.getElementById("instruction"),
  fullcotDetails: document.getElementById("fullcotDetails"),
  fullcot: document.getElementById("fullcot"),
  chunks: document.getElementById("chunks"),
  plot: document.getElementById("plot"),
  graph: document.getElementById("graph"),
  selected: document.getElementById("selected"),
};

let indexData = null;
let currentProblem = null;
let selectedChunkIdx = null;
let selectedRolloutIdx = null;

function getProblemGroupById(problemId) {
  return indexData?.problems?.find((p) => String(p.problem_id) === String(problemId)) || null;
}

function getExampleById(problemId, exampleId) {
  const group = getProblemGroupById(problemId);
  if (!group) return null;
  return (group.examples || []).find((ex) => String(ex.id) === String(exampleId)) || null;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

function setQueryParam(name, value) {
  const url = new URL(window.location.href);
  url.searchParams.set(name, value);
  window.history.replaceState({}, "", url.toString());
}

async function fetchJson(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error(`Failed to fetch ${path}: ${resp.status}`);
  return await resp.json();
}

function metricValue(chunk, metric) {
  const v = chunk?.metrics?.[metric];
  if (v === null || v === undefined) return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

function chunkHasTag(chunk, tag) {
  const target = String(tag || "").trim().toLowerCase();
  if (!target) return false;
  return (chunk.function_tags || []).some((t) => String(t).toLowerCase() === target);
}

// -- Rendering Logic --

function renderProblemHeader() {
  const instr = currentProblem?.title || currentProblem?.instruction || "";
  const ci = currentProblem?.ci || "";
  const va = (typeof currentProblem?.va_chunks === "number" && Number.isFinite(currentProblem.va_chunks))
    ? currentProblem.va_chunks
    : (currentProblem?.va_chunks ?? "n/a");
  const edgeCount = Array.isArray(currentProblem?.edges) ? currentProblem.edges.length : 0;
  const chunkCount = Array.isArray(currentProblem?.chunks) ? currentProblem.chunks.length : 0;
  const meta = `${currentProblem?.id || ""}${ci ? `  (${ci})` : ""}  VA=${va}  chunks=${chunkCount}  edges=${edgeCount}`.trim();

  els.instruction.textContent = meta || (instr || "(No instruction found)");
  els.instruction.title = instr ? `${instr}\n\n${meta}` : meta;

  const cot = currentProblem?.full_cot || "";
  if (cot) {
    els.fullcot.textContent = cot;
    els.fullcotDetails.style.display = "block";
  } else {
    els.fullcot.textContent = "";
    els.fullcotDetails.style.display = "none";
  }
}

function renderSelectedChunk() {
  if (!currentProblem || selectedChunkIdx === null) {
    els.selected.innerHTML = `
        <div class="empty-state">
            <div class="icon">👈</div>
            <p>Select a node in the graph to view details.</p>
        </div>`;
    return;
  }
  const chunk = currentProblem.chunks.find((c) => String(c.idx) === String(selectedChunkIdx));
  if (!chunk) {
    els.selected.innerHTML = "<div class=\"empty-state\">Selection not found.</div>";
    return;
  }

  const metric = els.metricSelect.value;
  const v = metricValue(chunk, metric);
  const tags = (chunk.function_tags || []).join(", ");

  const rollouts = Array.isArray(chunk.rollouts) ? chunk.rollouts : [];
  let selectedRollout = null;
  if (rollouts.length > 0) {
    if (selectedRolloutIdx === null || !rollouts.some((r) => String(r.i) === String(selectedRolloutIdx))) {
      selectedRolloutIdx = rollouts[0].i;
    }
    selectedRollout = rollouts.find((r) => String(r.i) === String(selectedRolloutIdx)) || null;
  }

  // Highlight the chunk text based on tags if needed, for now just show text
  let summaryHtml = "";
  if (chunk.summary) {
    summaryHtml = `
        <div class="section-title">Summary</div>
        <div class="chunk-box">${escapeHtml(chunk.summary)}</div>
      `;
  }

  els.selected.innerHTML = `
    <div class="chunk-card">
        <div class="chunk-header">
            <div class="chunk-id">CHUNK ${escapeHtml(chunk.idx)}</div>
            <h2>${tags ? tags : "Reasoning Step"}</h2>
        </div>
        
        <div class="chunk-metrics">
            <div class="detail-metric">
                <span class="detail-label">${escapeHtml(metric)}</span>
                <span class="detail-value">${v === null ? "n/a" : (v.toFixed ? v.toFixed(4) : v)}</span>
            </div>
            <div class="detail-metric">
                <span class="detail-label">Tags</span>
                <span class="detail-value">${tags ? escapeHtml(tags) : '<span style="color:#d1d5db">None</span>'}</span>
            </div>
        </div>

        ${rollouts.length > 0 ? `
        <div class="chunk-body">
             <div class="section-title">Trajectory</div>
             <select id="rolloutSelect" class="select-input" style="width:100%; margin-top:8px;">
               ${rollouts.map((r) => {
                 const rs = (typeof r.rubric_score === "number" && Number.isFinite(r.rubric_score)) ? r.rubric_score.toFixed(3) : "n/a";
                 const lab = `#${r.i}  score=${rs}`;
                 const sel = String(r.i) === String(selectedRolloutIdx) ? "selected" : "";
                 return `<option value="${escapeHtml(r.i)}" ${sel}>${escapeHtml(lab)}</option>`;
               }).join("")}
             </select>

             <div style="margin-top:12px;">
               <div class="detail-metric">
                   <span class="detail-label">rubric_score</span>
                   <span class="detail-value">${selectedRollout && typeof selectedRollout.rubric_score === "number" && Number.isFinite(selectedRollout.rubric_score) ? selectedRollout.rubric_score.toFixed(4) : "n/a"}</span>
               </div>
               <div class="detail-metric">
                   <span class="detail-label">rubric_grade</span>
                   <span class="detail-value">${selectedRollout && selectedRollout.rubric_grade ? escapeHtml(selectedRollout.rubric_grade) : "n/a"}</span>
               </div>
             </div>

             <div style="margin-top:16px;">
               <div class="section-title">Rollout Reasoning</div>
               <div class="chunk-box" style="font-family:var(--font-mono); font-size:12px;">${escapeHtml(selectedRollout?.rollout_reasoning_text || "")}</div>
             </div>
             <div style="margin-top:16px;">
               <div class="section-title">Rollout Final</div>
               <div class="chunk-box" style="font-family:var(--font-mono); font-size:12px;">${escapeHtml(selectedRollout?.rollout_final_text || "")}</div>
             </div>
        </div>
        ` : ""}

        <div class="chunk-body">
             <div class="section-title">Content</div>
             <div class="chunk-box" style="font-family:var(--font-mono); font-size:12px;">${escapeHtml(chunk.chunk || "")}</div>
             ${summaryHtml}
        </div>
    </div>
  `;

  if (rollouts.length > 0) {
    const sel = document.getElementById("rolloutSelect");
    if (sel) {
      sel.addEventListener("change", (e) => {
        selectedRolloutIdx = e.target.value;
        renderSelectedChunk();
      });
    }
  }
}

function renderChunks() {
  const metric = els.metricSelect.value;
  const highlight = els.highlightTag.value;

  els.chunks.innerHTML = "";
  if (!currentProblem || !currentProblem.chunks || currentProblem.chunks.length === 0) {
    els.chunks.innerHTML = "<div style='padding:12px; color:#9ca3af'>No chunks.</div>";
    return;
  }

  // Map chunks to bar items
  const maxVal = 1.0; // Assume normalized 0-1 usually, or find max

  for (const chunk of currentProblem.chunks) {
    const v = metricValue(chunk, metric);
    const val = v === null ? 0 : Math.max(0, Math.min(1, v)); // clamp 0-1

    const active = String(chunk.idx) === String(selectedChunkIdx);
    const hit = chunkHasTag(chunk, highlight);

    const div = document.createElement("div");
    div.className = `chunk-bar-item ${active ? "active" : ""} ${hit ? "hit" : ""}`;
    div.title = `Chunk ${chunk.idx}: ${v !== null ? v.toFixed(3) : "n/a"}`;

    const fillHeight = Math.max(5, val * 100); // at least 5% so visible

    div.innerHTML = `
        <div class="fill" style="height: ${fillHeight}%; opacity: ${v === null ? 0.3 : 1}"></div>
      `;

    div.addEventListener("click", () => {
      selectedChunkIdx = chunk.idx;
      selectedRolloutIdx = null;
      renderChunks();
      renderSelectedChunk();
      renderPlot();
      renderGraph(); // update highlighting
    });

    els.chunks.appendChild(div);
  }
}

function renderPlot() {
  const metric = els.metricSelect.value;
  if (!currentProblem || !currentProblem.chunks || currentProblem.chunks.length === 0) {
    Plotly.react(els.plot, [], { title: "No data" }, { displayModeBar: false });
    return;
  }

  const xs = [];
  const ys = [];
  const hover = [];
  const colors = [];
  const highlight = els.highlightTag.value;

  for (const c of currentProblem.chunks) {
    const y = metricValue(c, metric);
    xs.push(c.idx);
    ys.push(y === null ? NaN : y);
    hover.push(
      `Chunk ${c.idx}<br>${metric}: ${y !== null ? y.toFixed(4) : "n/a"}`
    );

    const isSel = String(c.idx) === String(selectedChunkIdx);
    const isHit = chunkHasTag(c, highlight);
    // Use CSS var colors if possible, but Plotly needs hex
    // Primary: #0f172a, Accent: #f97316, Selection: #0ea5e9
    colors.push(isSel ? "#0ea5e9" : isHit ? "#f97316" : "#cbd5e1");
  }

  const trace = {
    x: xs,
    y: ys,
    mode: "lines+markers",
    hoverinfo: "text",
    text: hover,
    marker: {
      size: xs.map((x) => (String(x) === String(selectedChunkIdx) ? 10 : 6)),
      color: colors,
      line: { width: 1, color: "white" },
      opacity: 1
    },
    line: { width: 2, color: "#94a3b8", shape: 'spline' }, // Smooth line
  };

  const layout = {
    margin: { l: 40, r: 20, t: 10, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: {
      title: "", // Hide title for clean look
      zeroline: false,
      gridcolor: "#f1f5f9",
      showticklabels: true
    },
    yaxis: {
      title: "",
      zeroline: false,
      gridcolor: "#f1f5f9",
      range: [0, 1.1] // slightly explicit range
    },
    autosize: true
  };

  const config = { displayModeBar: false, responsive: true };

  Plotly.react(els.plot, [trace], layout, config);

  if (els.plot.removeAllListeners) els.plot.removeAllListeners("plotly_click");
  els.plot.on("plotly_click", (ev) => {
    if (!ev || !ev.points || !ev.points[0]) return;
    const idx = ev.points[0].x;
    selectedChunkIdx = idx;
    renderChunks();
    renderSelectedChunk();
    renderPlot();
    renderGraph();
  });
}

function tagToColor(tag) {
  const t = String(tag || "unknown").toLowerCase();
  // Premium palette mapping
  const map = {
    plan_generation: "#3b82f6",     // blue-500
    fact_retrieval: "#10b981",      // emerald-500
    uncertainty_management: "#8b5cf6", // violet-500
    result_consolidation: "#06b6d4", // cyan-500
    self_checking: "#ef4444",       // red-500
    problem_setup: "#64748b",       // slate-500
    final_answer_emission: "#f59e0b", // amber-500
    verbalized_evaluation_awareness: "#f97316", // orange-500
    refusal: "#e11d48", // rose-600
    unknown: "#9ca3af",
  };
  return map[t] || "#9ca3af";
}

function renderGraph() {
  if (!currentProblem || !currentProblem.chunks || currentProblem.chunks.length === 0) {
    els.graph.innerHTML = "<div style='padding:20px; color:#9ca3af'>No graph data.</div>";
    return;
  }
  if (!window.d3) {
    els.graph.innerHTML = "<div style='padding:20px; color:red'>D3 failed to load.</div>";
    return;
  }

  const metric = els.metricSelect.value;
  const highlightTag = els.highlightTag.value;
  const thresholdPct = Number(els.nodeThreshold?.value || 0);
  const topK = Number(els.topKEdges?.value || 0);

  const chunks = currentProblem.chunks.slice().sort((a, b) => Number(a.idx) - Number(b.idx));
  const values = chunks.map((c) => metricValue(c, metric)).filter((v) => v !== null && Number.isFinite(v));
  const sortedVals = values.slice().sort((a, b) => a - b);
  const qIndex = Math.floor((thresholdPct / 100) * Math.max(0, sortedVals.length - 1));
  const cutoff = sortedVals.length ? sortedVals[qIndex] : -Infinity;

  const visible = chunks.filter((c) => {
    const v = metricValue(c, metric);
    if (v === null) return cutoff <= 0;
    return v >= cutoff;
  });
  const visibleIds = new Set(visible.map((c) => Number(c.idx)));

  els.graph.innerHTML = "";
  const containerWidth = els.graph.clientWidth;
  const containerHeight = els.graph.clientHeight;
  const width = containerWidth || 800;
  const height = containerHeight || 600;

  // D3 Setup with Zoom
  const svg = d3.select(els.graph)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`);

  // Create a group for zooming
  const g = svg.append("g");

  const zoom = d3.zoom()
    .scaleExtent([0.5, 4])
    .on("zoom", (e) => g.attr("transform", e.transform));

  svg.call(zoom);

  // Layout: Circular for now, but improved
  const r = Math.min(width, height) * 0.4; // 80% fit
  const cx = width / 2;
  const cy = height / 2;

  // Compute positions
  const nodes = visible.map((c, i) => {
    const n = visible.length;
    // Spiral or Circle? Circle is cleaner for linear flow usually
    const angle = (2 * Math.PI * i) / Math.max(1, n) - Math.PI / 2;
    return {
      id: Number(c.idx),
      idx: c.idx,
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      value: metricValue(c, metric),
      tags: c.function_tags || [],
      primary: (c.function_tags || [])[0] || "unknown",
      chunk: c
    };
  });

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  // Edges
  const edges = [];
  const rawEdges = currentProblem.edges || [];

  for (const e of rawEdges) {
    const s = Number(e.source);
    const t = Number(e.target);
    if (!visibleIds.has(s) || !visibleIds.has(t)) continue;
    edges.push({ ...e, source: s, target: t, weight: Number(e.weight || 1) });
  }

  const causal = edges.filter(e => e.type === "causal").sort((a, b) => b.weight - a.weight);
  const causalKept = topK > 0 ? causal.slice(0, topK) : [];
  const sequential = edges.filter(e => e.type === "sequential");

  const drawLine = (e, color, w, opacity) => {
    const a = nodeById.get(e.source);
    const b = nodeById.get(e.target);
    if (!a || !b) return;
    g.append("line")
      .attr("x1", a.x).attr("y1", a.y)
      .attr("x2", b.x).attr("y2", b.y)
      .attr("stroke", color)
      .attr("stroke-width", w)
      .attr("stroke-opacity", opacity);
  };

  // Draw Edges
  sequential.forEach(e => drawLine(e, "#e5e7eb", 1.5, 1));
  causalKept.forEach(e => drawLine(e, "#0f172a", 2, 0.35));

  // Draw Nodes
  // Add drop shadow
  //   const defs = svg.append("defs");
  //   const dropShadowFilter = defs.append("filter").attr("id", "dropShadow").attr("height", "130%");
  // ... svg filters are verbose, skip for now.

  const nodeGroups = g.selectAll(".node")
    .data(nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${d.x},${d.y})`)
    .style("cursor", "pointer")
    .on("click", (evt, d) => {
      evt.stopPropagation(); // prevent zoom drag interfering click
      selectedChunkIdx = d.id;
      renderChunks();
      renderSelectedChunk();
      renderPlot();
      renderGraph();
    });

  // Highlight Selection Ring
  nodeGroups.append("circle")
    .attr("r", d => d.id == selectedChunkIdx ? 14 : 0)
    .attr("fill", "transparent")
    .attr("stroke", "#0ea5e9")
    .attr("stroke-width", 2)
    .attr("opacity", 0.6);

  // Main Node Circle
  nodeGroups.append("circle")
    .attr("r", d => 6 + (d.value || 0) * 6) // Scale size by value
    .attr("fill", d => tagToColor(d.primary))
    .attr("stroke", "white")
    .attr("stroke-width", 2)
    .attr("opacity", d => chunkHasTag(d.chunk, highlightTag) ? 1 : 0.4);

  // Node ID Label
  nodeGroups.append("text")
    .attr("dy", 3)
    .attr("text-anchor", "middle")
    .attr("fill", "white")
    .attr("font-size", 8)
    .attr("font-family", "monospace")
    .style("pointer-events", "none")
    .text(d => d.id);

}


// -- Init --

async function loadProblemById(id) {
  const problemId = els.problemSelect?.value;
  const item = getExampleById(problemId, id);
  if (!item) throw new Error(`Unknown example id: ${id}`);

  const data = await fetchJson(`${DATA_ROOT}/${item.path}`);
  currentProblem = data;
  selectedChunkIdx = (currentProblem.chunks?.[0]?.idx ?? null);
  selectedRolloutIdx = null;

  setQueryParam("p", problemId);
  setQueryParam("e", id);
  renderProblemHeader();
  renderChunks();
  renderSelectedChunk();
  renderPlot();
  renderGraph();
}

function populateProblemSelect() {
  els.problemSelect.innerHTML = "";
  for (const p of indexData.problems) {
    const opt = document.createElement("option");
    opt.value = p.problem_id;
    const title = String(p.title || p.problem_id);
    const shortTitle = title.length > 70 ? title.substring(0, 70) + "..." : title;
    opt.textContent = `${p.problem_id}: ${shortTitle}`;
    els.problemSelect.appendChild(opt);
  }
}

function populateExampleSelect(problemId) {
  const group = getProblemGroupById(problemId);
  const examples = group?.examples || [];
  els.exampleSelect.innerHTML = "";
  for (const ex of examples) {
    const opt = document.createElement("option");
    opt.value = ex.id;
    const va = Number.isFinite(ex.va_chunks) ? ex.va_chunks : (ex.va_chunks ?? 0);
    const score = (typeof ex.base_rubric_score === "number" && Number.isFinite(ex.base_rubric_score))
      ? ex.base_rubric_score.toFixed(3)
      : "n/a";
    const label = `VA=${va}  score=${score}  ${ex.ci || ""}`.trim();
    opt.textContent = label;
    els.exampleSelect.appendChild(opt);
  }
}

async function init() {
  try {
    indexData = await fetchJson(`${DATA_ROOT}/index.json`);
  } catch (e) {
    els.instruction.textContent = "Error loading index.json. Ensure server is running.";
    console.error(e);
    return;
  }

  populateProblemSelect();

  const requestedProblem = getQueryParam("p");
  const firstProblem = indexData?.problems?.[0]?.problem_id;
  const initialProblem = requestedProblem && indexData.problems.some((p) => String(p.problem_id) === String(requestedProblem))
    ? requestedProblem
    : firstProblem;
  els.problemSelect.value = initialProblem;

  populateExampleSelect(initialProblem);

  const requestedExample = getQueryParam("e");
  const group = getProblemGroupById(initialProblem);
  const firstExample = group?.examples?.[0]?.id;
  const initialExample = requestedExample && group?.examples?.some((ex) => String(ex.id) === String(requestedExample))
    ? requestedExample
    : firstExample;
  els.exampleSelect.value = initialExample;

  // Event Listeners
  els.problemSelect.addEventListener("change", async () => {
    const pid = els.problemSelect.value;
    populateExampleSelect(pid);
    const g = getProblemGroupById(pid);
    const ex0 = g?.examples?.[0]?.id;
    if (ex0) {
      els.exampleSelect.value = ex0;
      await loadProblemById(ex0);
    }
  });

  els.exampleSelect.addEventListener("change", async () => {
    await loadProblemById(els.exampleSelect.value);
  });

  els.metricSelect.addEventListener("change", () => {
    renderChunks();
    renderSelectedChunk();
    renderPlot();
    renderGraph();
  });

  els.highlightTag.addEventListener("input", () => { // Changed to input for realtime
    renderChunks();
    renderGraph();
    renderPlot();
  });

  els.nodeThreshold?.addEventListener("input", renderGraph);
  els.topKEdges?.addEventListener("input", renderGraph);

  // Resize observer for graph
  new ResizeObserver(() => {
    if (currentProblem) renderGraph();
    if (currentProblem) Plotly.Plots.resize(els.plot);
  }).observe(els.graph.parentElement);

  new ResizeObserver(() => {
    if (currentProblem) Plotly.Plots.resize(els.plot);
  }).observe(els.plot.parentElement);


  if (initialExample) await loadProblemById(initialExample);
}

// Start
init();
