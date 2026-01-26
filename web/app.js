/* global Plotly, d3 */

const DATA_ROOT = "./data";

const els = {
  problemSelect: document.getElementById("problemSelect"),
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

function computeBarWidth(chunk, metric) {
  const v = metricValue(chunk, metric);
  if (v === null) return 0;
  if (v >= 0 && v <= 1) return Math.round(v * 100);
  return Math.round(Math.max(0, Math.min(1, v / 10)) * 100);
}

function renderProblemHeader() {
  const instr = currentProblem?.instruction || "";
  els.instruction.textContent = instr || "(No instruction found in exported data.)";

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
    els.selected.innerHTML = "<div class=\"muted\">No chunk selected.</div>";
    return;
  }
  const chunk = currentProblem.chunks.find((c) => String(c.idx) === String(selectedChunkIdx));
  if (!chunk) {
    els.selected.innerHTML = "<div class=\"muted\">No chunk selected.</div>";
    return;
  }
  const metric = els.metricSelect.value;
  const v = metricValue(chunk, metric);
  const tags = (chunk.function_tags || []).map(escapeHtml).join(" · ");

  els.selected.innerHTML = `
    <h3>Chunk ${escapeHtml(chunk.idx)}</h3>
    <div class="meta">
      <span><b>Metric:</b> ${escapeHtml(metric)}</span>
      <span> · <b>Value:</b> ${v === null ? "n/a" : escapeHtml(v.toFixed ? v.toFixed(4) : v)}</span>
      <span> · <b>Tags:</b> ${tags || "none"}</span>
    </div>
    ${chunk.summary ? `<div class="box"><b>Summary</b>\n\n${escapeHtml(chunk.summary)}</div>` : ""}
    <div style="height: 10px"></div>
    <div class="box"><b>Chunk</b>\n\n${escapeHtml(chunk.chunk || "")}</div>
  `;
}

function renderChunks() {
  const metric = els.metricSelect.value;
  const highlight = els.highlightTag.value;

  els.chunks.innerHTML = "";
  if (!currentProblem || !currentProblem.chunks || currentProblem.chunks.length === 0) {
    els.chunks.innerHTML = "<div class=\"chunk\"><div class=\"muted\">No chunks available.</div></div>";
    return;
  }

  for (const chunk of currentProblem.chunks) {
    const width = computeBarWidth(chunk, metric);
    const active = String(chunk.idx) === String(selectedChunkIdx);
    const tags = chunk.function_tags || [];
    const hit = chunkHasTag(chunk, highlight);

    const tagHtml = tags
      .slice(0, 6)
      .map((t) => `<span class="tag ${String(t).toLowerCase() === String(highlight).toLowerCase() ? "hit" : ""}">${escapeHtml(t)}</span>`)
      .join("");

    const div = document.createElement("div");
    div.className = `chunk${active ? " active" : ""}`;
    div.innerHTML = `
      <div class="chunk-row">
        <div class="chunk-idx">#${escapeHtml(chunk.idx)}</div>
        <div class="bar" title="${escapeHtml(metric)}">
          <span style="width:${width}%;"></span>
        </div>
        <div class="chunk-idx" title="highlight">${hit ? "hit" : ""}</div>
      </div>
      <div class="tags">${tagHtml || "<span class=\"tag\">none</span>"}</div>
    `;
    div.addEventListener("click", () => {
      selectedChunkIdx = chunk.idx;
      renderChunks();
      renderSelectedChunk();
      renderPlot();
      renderGraph();
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
      `<b>Chunk ${escapeHtml(c.idx)}</b><br>` +
        `<b>Tags:</b> ${escapeHtml((c.function_tags || []).join(", ") || "none")}<br>` +
        `<b>Summary:</b> ${escapeHtml(c.summary || "")}<br>` +
        `<span style="opacity:.7">${escapeHtml((c.chunk || "").slice(0, 240))}${(c.chunk || "").length > 240 ? "…" : ""}</span>`
    );
    const isSel = String(c.idx) === String(selectedChunkIdx);
    const isHit = chunkHasTag(c, highlight);
    colors.push(isSel ? "#0c4a6e" : isHit ? "#c36a2d" : "rgba(25,26,24,0.35)");
  }

  const trace = {
    x: xs,
    y: ys,
    mode: "lines+markers",
    hoverinfo: "text",
    text: hover,
    marker: {
      size: xs.map((x) => (String(x) === String(selectedChunkIdx) ? 10 : 7)),
      color: colors,
      line: { width: 1, color: "rgba(0,0,0,0.35)" },
    },
    line: { width: 2, color: "rgba(12,74,110,0.35)" },
  };

  const layout = {
    margin: { l: 50, r: 20, t: 20, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: "chunk_idx", zeroline: false, gridcolor: "rgba(0,0,0,0.06)" },
    yaxis: { title: metric, zeroline: false, gridcolor: "rgba(0,0,0,0.06)" },
  };

  Plotly.react(els.plot, [trace], layout, { displayModeBar: false, responsive: true });
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
  const map = {
    plan_generation: "#0c4a6e",
    fact_retrieval: "#b45309",
    uncertainty_management: "#7c3aed",
    result_consolidation: "#0f766e",
    self_checking: "#9a3412",
    problem_setup: "#1d4ed8",
    final_answer_emission: "#6b7280",
    verbalized_evaluation_awareness: "#c36a2d",
    unknown: "rgba(25,26,24,0.35)",
  };
  return map[t] || "rgba(25,26,24,0.35)";
}

function renderGraph() {
  if (!currentProblem || !currentProblem.chunks || currentProblem.chunks.length === 0) {
    els.graph.innerHTML = "<div class=\"muted\" style=\"padding:14px 16px\">No graph data.</div>";
    return;
  }
  if (!window.d3) {
    els.graph.innerHTML = "<div class=\"muted\" style=\"padding:14px 16px\">D3 failed to load.</div>";
    return;
  }

  const metric = els.metricSelect.value;
  const highlightTag = els.highlightTag.value;
  const thresholdPct = Number(els.nodeThreshold?.value || 0);
  const topK = Number(els.topKEdges?.value || 0);

  const chunks = currentProblem.chunks.slice().sort((a, b) => Number(a.idx) - Number(b.idx));

  const values = chunks
    .map((c) => metricValue(c, metric))
    .filter((v) => v !== null && Number.isFinite(v));
  const sortedVals = values.slice().sort((a, b) => a - b);
  const qIndex = Math.floor((thresholdPct / 100) * Math.max(0, sortedVals.length - 1));
  const cutoff = sortedVals.length ? sortedVals[qIndex] : -Infinity;

  const visible = chunks.filter((c) => {
    const v = metricValue(c, metric);
    if (v === null) return cutoff <= 0;
    return v >= cutoff;
  });
  const visibleIds = new Set(visible.map((c) => Number(c.idx)));

  const width = els.graph.clientWidth || 600;
  const height = els.graph.clientHeight || 520;
  const r = Math.min(width, height) * 0.36;
  const cx = width / 2;
  const cy = height / 2;

  els.graph.innerHTML = "";
  const svg = d3
    .select(els.graph)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`);

  const tooltip = d3
    .select(els.graph)
    .append("div")
    .style("position", "absolute")
    .style("pointer-events", "none")
    .style("opacity", 0)
    .style("background", "rgba(255,255,255,0.95)")
    .style("border", "1px solid rgba(0,0,0,0.12)")
    .style("border-radius", "12px")
    .style("padding", "10px 12px")
    .style("box-shadow", "0 10px 20px rgba(0,0,0,0.08)")
    .style("font-family", "ui-sans-serif, system-ui")
    .style("font-size", "12px")
    .style("max-width", "340px");

  const vmin = Math.min(...values, 0);
  const vmax = Math.max(...values, 1);
  const sizeScale = (v) => {
    if (v === null || !Number.isFinite(v)) return 6;
    if (vmax === vmin) return 10;
    const t = (v - vmin) / (vmax - vmin);
    return 6 + 10 * Math.max(0, Math.min(1, t));
  };

  // Build nodes in a circle
  const nodes = visible.map((c, i) => {
    const n = visible.length;
    const angle = (2 * Math.PI * i) / Math.max(1, n) - Math.PI / 2;
    const v = metricValue(c, metric);
    const tags = c.function_tags || [];
    const primary = tags.find((t) => String(t).trim()) || "unknown";
    return {
      id: Number(c.idx),
      idx: c.idx,
      angle,
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      value: v,
      tags,
      primary,
      summary: c.summary || "",
      chunk: c.chunk || "",
    };
  });

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  // Edges: sequential + depends_on
  const rawEdges = (currentProblem.edges || []).slice();
  const edges = [];
  for (const e of rawEdges) {
    const s = Number(e.source);
    const t = Number(e.target);
    if (!visibleIds.has(s) || !visibleIds.has(t)) continue;
    const type = e.type || "causal";
    const weight = Number.isFinite(e.weight) ? Number(e.weight) : 1;
    edges.push({ source: s, target: t, type, weight });
  }

  const causal = edges.filter((e) => e.type === "causal");
  causal.sort((a, b) => b.weight - a.weight);
  const causalKept = topK > 0 ? causal.slice(0, topK) : [];
  const sequential = edges.filter((e) => e.type === "sequential");

  const drawEdge = (e, style) => {
    const a = nodeById.get(e.source);
    const b = nodeById.get(e.target);
    if (!a || !b) return;
    svg
      .append("line")
      .attr("x1", a.x)
      .attr("y1", a.y)
      .attr("x2", b.x)
      .attr("y2", b.y)
      .attr("stroke", style.stroke)
      .attr("stroke-opacity", style.opacity)
      .attr("stroke-width", style.width);
  };

  // Draw sequential perimeter lightly
  for (const e of sequential) {
    drawEdge(e, { stroke: "rgba(0,0,0,0.18)", opacity: 1, width: 1 });
  }
  // Draw causal chords
  for (const e of causalKept) {
    drawEdge(e, { stroke: "rgba(12,74,110,0.45)", opacity: 1, width: 1.6 });
  }

  // Nodes
  const nodeSel = svg
    .selectAll("circle.node")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("class", "node")
    .attr("cx", (d) => d.x)
    .attr("cy", (d) => d.y)
    .attr("r", (d) => sizeScale(d.value))
    .attr("fill", (d) => tagToColor(d.primary))
    .attr("stroke", (d) =>
      String(d.id) === String(selectedChunkIdx) ? "rgba(12,74,110,1)" : "rgba(0,0,0,0.25)"
    )
    .attr("stroke-width", (d) => (String(d.id) === String(selectedChunkIdx) ? 2.2 : 1))
    .attr("opacity", (d) => (chunkHasTag({ function_tags: d.tags }, highlightTag) ? 1 : 0.85))
    .style("cursor", "pointer")
    .on("mouseenter", (event, d) => {
      const val = d.value === null || !Number.isFinite(d.value) ? "n/a" : d.value.toFixed(4);
      tooltip
        .style("opacity", 1)
        .html(
          `<div style="font-weight:700">Chunk ${escapeHtml(d.idx)}</div>` +
            `<div style="color:rgba(0,0,0,0.65);margin-top:4px"><b>${escapeHtml(metric)}:</b> ${escapeHtml(val)}</div>` +
            `<div style="margin-top:6px"><b>Tags:</b> ${escapeHtml((d.tags || []).join(", ") || "none")}</div>` +
            (d.summary ? `<div style="margin-top:6px"><b>Summary:</b> ${escapeHtml(d.summary)}</div>` : "")
        );
      const rect = els.graph.getBoundingClientRect();
      tooltip
        .style("left", `${event.clientX - rect.left + 14}px`)
        .style("top", `${event.clientY - rect.top + 14}px`);
    })
    .on("mousemove", (event) => {
      const rect = els.graph.getBoundingClientRect();
      tooltip
        .style("left", `${event.clientX - rect.left + 14}px`)
        .style("top", `${event.clientY - rect.top + 14}px`);
    })
    .on("mouseleave", () => {
      tooltip.style("opacity", 0);
    })
    .on("click", (_event, d) => {
      selectedChunkIdx = d.id;
      renderChunks();
      renderSelectedChunk();
      renderPlot();
      renderGraph();
    });

  // Index labels (small)
  svg
    .selectAll("text.lbl")
    .data(nodes)
    .enter()
    .append("text")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y + 3)
    .attr("text-anchor", "middle")
    .attr("font-size", 9)
    .attr("fill", "rgba(0,0,0,0.7)")
    .attr("font-family", "ui-monospace, monospace")
    .text((d) => d.id);
}

async function loadProblemById(id) {
  const item = indexData.problems.find((p) => p.id === id);
  if (!item) throw new Error(`Unknown problem id: ${id}`);
  const data = await fetchJson(`${DATA_ROOT}/${item.path}`);
  currentProblem = data;
  selectedChunkIdx = (currentProblem.chunks?.[0]?.idx ?? null);
  setQueryParam("p", id);
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
    opt.value = p.id;
    opt.textContent = `${p.id} · ${p.title}`;
    els.problemSelect.appendChild(opt);
  }
}

async function init() {
  try {
    indexData = await fetchJson(`${DATA_ROOT}/index.json`);
  } catch (e) {
    els.instruction.textContent =
      "Failed to load web/data/index.json. Run scripts/build_web_data.py and serve web/ with a local server.";
    els.chunks.innerHTML = `<div class="chunk"><div class="muted">${escapeHtml(e.message)}</div></div>`;
    return;
  }

  populateProblemSelect();

  const requested = getQueryParam("p");
  const first = indexData?.problems?.[0]?.id;
  const initial = requested && indexData.problems.some((p) => p.id === requested) ? requested : first;
  els.problemSelect.value = initial;

  els.problemSelect.addEventListener("change", async () => {
    await loadProblemById(els.problemSelect.value);
  });
  els.metricSelect.addEventListener("change", () => {
    renderChunks();
    renderSelectedChunk();
    renderPlot();
    renderGraph();
  });
  els.highlightTag.addEventListener("change", () => {
    renderChunks();
    renderPlot();
    renderGraph();
  });

  els.nodeThreshold?.addEventListener("input", () => {
    renderGraph();
  });
  els.topKEdges?.addEventListener("input", () => {
    renderGraph();
  });

  if (initial) await loadProblemById(initial);
}

init();
