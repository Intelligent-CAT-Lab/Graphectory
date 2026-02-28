/* browser.js – sidebar list + on-demand graph loading */

let allGraphs = [];
let activeId  = null;

// ── Bootstrap ──────────────────────────────────────────────
async function init() {
    await loadGraphList();
    wireSearch();
    wireToggle();
}

// ── Graph list ─────────────────────────────────────────────
async function loadGraphList() {
    try {
        const res = await fetch('/api/graphs');
        allGraphs  = await res.json();
        renderStats(allGraphs);
        renderList(allGraphs);
    } catch (err) {
        document.getElementById('graphList').innerHTML =
            '<div class="placeholder" style="position:relative">Failed to load graphs</div>';
    }
}

function renderStats(graphs) {
    const total      = graphs.length;
    const resolved   = graphs.filter(g => g.status === 'resolved').length;
    const unresolved = graphs.filter(g => g.status === 'unresolved').length;
    document.getElementById('stats').textContent =
        `${total} total · ${resolved} resolved · ${unresolved} unresolved`;
}

function renderList(graphs) {
    const el = document.getElementById('graphList');

    if (!graphs.length) {
        el.innerHTML = '<div class="placeholder" style="position:relative;padding:30px">No results</div>';
        return;
    }

    el.innerHTML = graphs.map(g => {
        // Normalise status so we always have a valid badge class
        const status     = g.status || 'unknown';
        const badgeClass = ['resolved','unresolved','unsubmitted'].includes(status)
                           ? status : 'unknown';
        const steps      = g.step_count != null ? `${g.step_count} steps` : '';
        const diff       = g.difficulty && g.difficulty !== 'unknown' ? escHtml(g.difficulty) : '';
        const metaParts  = [steps, diff].filter(Boolean).join(' · ');

        return `
        <div class="graph-item${g.instance_id === activeId ? ' active' : ''}"
             data-id="${escHtml(g.instance_id)}"
             onclick="selectGraph('${escHtml(g.instance_id)}')">
            <div class="item-title" title="${escHtml(g.instance_id)}">${escHtml(g.instance_id)}</div>
            <div class="item-meta">
                <span class="badge badge-${badgeClass}">${escHtml(status)}</span>
                ${metaParts ? `<span>${metaParts}</span>` : ''}
            </div>
        </div>`;
    }).join('');
}

// ── Search ─────────────────────────────────────────────────
function wireSearch() {
    document.getElementById('searchInput').addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        renderList(allGraphs.filter(g => g.instance_id.toLowerCase().includes(q)));
    });
}

// ── CD toggle + new toggles ────────────────────────────────
function wireToggle() {
    ['filterCdToggle', 'thoughtQuotesToggle', 'nodeVerbosityToggle', 'observationToggle', 'uniqueThinkToggle'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            if (activeId) loadGraph(activeId);
        });
    });
}

function filterCd() {
    return document.getElementById('filterCdToggle').checked;
}

function thoughtQuotes() {
    return document.getElementById('thoughtQuotesToggle').checked;
}

function nodeVerbosity() {
    return document.getElementById('nodeVerbosityToggle').checked;
}

function showObservation() {
    return document.getElementById('observationToggle').checked;
}

function uniqueThink() {
    return document.getElementById('uniqueThinkToggle').checked;
}

// ── Graph loading ──────────────────────────────────────────
function selectGraph(instanceId) {
    activeId = instanceId;

    // Update active highlight
    document.querySelectorAll('.graph-item').forEach(el =>
        el.classList.toggle('active', el.dataset.id === instanceId)
    );

    loadGraph(instanceId);
}

function showLoading() {
    const pane = document.getElementById('graphPane');
    // Remove any existing iframe so the spinner is visible
    const iframe = pane.querySelector('iframe');
    if (iframe) iframe.remove();

    let ph = pane.querySelector('.placeholder');
    if (!ph) {
        ph = document.createElement('div');
        ph.className = 'placeholder';
        pane.appendChild(ph);
    }
    ph.innerHTML = '<div class="spinner"></div><span>Rendering graph…</span>';
}

async function loadGraph(instanceId) {
    showLoading();

    const params = new URLSearchParams({
        id:            instanceId,
        filter_cd:     filterCd(),
        thought_quotes: thoughtQuotes(),
        node_verbosity: nodeVerbosity(),
        show_observation: showObservation(),
        unique_think:  uniqueThink(),
    });
    const url = `/api/graph?${params.toString()}`;

    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const html = await res.text();
        injectGraph(html);
    } catch (err) {
        const pane = document.getElementById('graphPane');
        const ph = pane.querySelector('.placeholder');
        if (ph) ph.innerHTML = `<span style="color:#e74c3c">⚠ ${escHtml(err.message)}</span>`;
    }
}

function injectGraph(html) {
    const pane = document.getElementById('graphPane');

    // Remove placeholder
    const ph = pane.querySelector('.placeholder');
    if (ph) ph.remove();

    // Reuse or create iframe
    let iframe = pane.querySelector('iframe');
    if (!iframe) {
        iframe = document.createElement('iframe');
        iframe.sandbox = 'allow-scripts allow-same-origin';
        pane.appendChild(iframe);
    }

    // Write HTML into iframe
    const doc = iframe.contentDocument || iframe.contentWindow.document;
    doc.open();
    doc.write(html);
    doc.close();
}

// ── Utility ────────────────────────────────────────────────
function escHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Start ──────────────────────────────────────────────────
init();