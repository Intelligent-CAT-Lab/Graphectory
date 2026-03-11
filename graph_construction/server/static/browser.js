/* browser.js — sidebar list + on-demand graph loading */

'use strict';

let allGraphs          = [];
let activeId           = null;
let dataSourceExpanded = false;

// ── Bootstrap ─────────────────────────────────────────────────────────────
async function init() {
    await loadConfig();   // populate path inputs and conditionally load graphs
    wireSearch();
    wireToggles();
    wireEnterKey();
}

// ── Data source configuration ─────────────────────────────────────────────

async function loadConfig() {
    try {
        const res  = await fetch('/api/config');
        const data = await res.json();

        if (data.trajs) {
            document.getElementById('trajsInput').value  = data.trajs;
            document.getElementById('reportInput').value = data.eval_report;
            updateDataSourceLabel(data.trajs);
            // Paths were supplied via CLI — load the graph list immediately.
            await loadGraphList();
        } else {
            // No paths configured yet — expand the panel so the user notices it.
            setDataSourceExpanded(true);
        }
    } catch (_) {
        setDataSourceExpanded(true);
    }
}

function toggleDataSource() {
    setDataSourceExpanded(!dataSourceExpanded);
}

function setDataSourceExpanded(open) {
    dataSourceExpanded = open;
    document.getElementById('dataSourceBody').style.display = open ? 'flex' : 'none';
    document.getElementById('dsChevron').textContent = open ? '▴' : '▾';
}

function updateDataSourceLabel(trajsPath) {
    // Show only the final path component so the header stays compact.
    const parts = trajsPath.replace(/\\/g, '/').split('/').filter(Boolean);
    const short = parts.length ? parts[parts.length - 1] : trajsPath;
    document.getElementById('dataSourceLabel').textContent = short || 'Data source';
}

async function applyDataSource() {
    const trajs      = document.getElementById('trajsInput').value.trim();
    const evalReport = document.getElementById('reportInput').value.trim();
    const errEl      = document.getElementById('dsError');
    const btn        = document.getElementById('dsApplyBtn');

    errEl.style.display = 'none';

    if (!trajs || !evalReport) {
        showDsError('Both paths are required.');
        return;
    }

    btn.disabled        = true;
    btn.textContent     = 'Loading…';

    try {
        const res  = await fetch('/api/config', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ trajs, eval_report: evalReport }),
        });
        const data = await res.json();

        if (!res.ok) {
            showDsError(data.error || `Server error (HTTP ${res.status})`);
            return;
        }

        // Success — update UI and reload the graph list.
        updateDataSourceLabel(data.trajs);
        activeId = null;
        clearGraphPane();
        setDataSourceExpanded(false);
        await loadGraphList();

    } catch (err) {
        showDsError(`Request failed: ${err.message}`);
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Load';
    }
}

function showDsError(msg) {
    const el = document.getElementById('dsError');
    el.textContent     = msg;
    el.style.display   = 'block';
}

// Allow pressing Enter in either input to trigger Load.
function wireEnterKey() {
    ['trajsInput', 'reportInput'].forEach(id => {
        document.getElementById(id).addEventListener('keydown', e => {
            if (e.key === 'Enter') applyDataSource();
        });
    });
}

// ── Graph list ────────────────────────────────────────────────────────────

async function loadGraphList() {
    document.getElementById('graphList').innerHTML =
        '<div class="placeholder" style="position:relative">'
        + '<div class="spinner"></div></div>';
    document.getElementById('stats').textContent = 'Loading…';

    try {
        const res = await fetch('/api/graphs');
        allGraphs = await res.json();
        renderStats(allGraphs);
        renderList(allGraphs);
        // Re-apply search filter if the user had typed something.
        const q = document.getElementById('searchInput').value.toLowerCase();
        if (q) renderList(allGraphs.filter(g => g.instance_id.toLowerCase().includes(q)));
    } catch (_) {
        document.getElementById('graphList').innerHTML =
            '<div class="placeholder" style="position:relative">Failed to load graphs.</div>';
        document.getElementById('stats').textContent = '—';
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
        el.innerHTML =
            '<div class="placeholder" style="position:relative;padding:30px">No results</div>';
        return;
    }

    el.innerHTML = graphs.map(g => {
        const status     = g.status || 'unknown';
        const badgeClass = ['resolved', 'unresolved', 'unsubmitted'].includes(status)
                           ? status : 'unknown';
        const steps      = g.step_count != null ? `${g.step_count} steps` : '';
        const diff       = g.difficulty && g.difficulty !== 'unknown'
                           ? escHtml(g.difficulty) : '';
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

// ── Search ────────────────────────────────────────────────────────────────

function wireSearch() {
    document.getElementById('searchInput').addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        renderList(allGraphs.filter(g => g.instance_id.toLowerCase().includes(q)));
    });
}

// ── View toggles ──────────────────────────────────────────────────────────

function wireToggles() {
    ['filterCdToggle', 'thoughtQuotesToggle', 'nodeVerbosityToggle',
     'observationToggle', 'uniqueThinkToggle'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            if (activeId) loadGraph(activeId);
        });
    });
}

const filterCd        = () => document.getElementById('filterCdToggle').checked;
const thoughtQuotes   = () => document.getElementById('thoughtQuotesToggle').checked;
const nodeVerbosity   = () => document.getElementById('nodeVerbosityToggle').checked;
const showObservation = () => document.getElementById('observationToggle').checked;
const uniqueThink     = () => document.getElementById('uniqueThinkToggle').checked;

// ── Graph loading ─────────────────────────────────────────────────────────

function selectGraph(instanceId) {
    activeId = instanceId;
    document.querySelectorAll('.graph-item').forEach(el =>
        el.classList.toggle('active', el.dataset.id === instanceId)
    );
    loadGraph(instanceId);
}

function showLoading() {
    const pane   = document.getElementById('graphPane');
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

function clearGraphPane() {
    const pane = document.getElementById('graphPane');
    pane.innerHTML =
        '<div class="placeholder">'
        + '<span class="placeholder-icon">📊</span>'
        + '<p>Select a graph from the list</p>'
        + '</div>';
}

async function loadGraph(instanceId) {
    showLoading();

    const params = new URLSearchParams({
        id:               instanceId,
        filter_cd:        filterCd(),
        thought_quotes:   thoughtQuotes(),
        node_verbosity:   nodeVerbosity(),
        show_observation: showObservation(),
        unique_think:     uniqueThink(),
    });

    try {
        const res = await fetch(`/api/graph?${params}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        injectGraph(await res.text());
    } catch (err) {
        const ph = document.getElementById('graphPane').querySelector('.placeholder');
        if (ph) ph.innerHTML = `<span style="color:#e74c3c">⚠ ${escHtml(err.message)}</span>`;
    }
}

function injectGraph(html) {
    const pane = document.getElementById('graphPane');
    const ph   = pane.querySelector('.placeholder');
    if (ph) ph.remove();

    let iframe = pane.querySelector('iframe');
    if (!iframe) {
        iframe         = document.createElement('iframe');
        iframe.sandbox = 'allow-scripts allow-same-origin';
        pane.appendChild(iframe);
    }

    const doc = iframe.contentDocument || iframe.contentWindow.document;
    doc.open();
    doc.write(html);
    doc.close();
}

// ── Utility ───────────────────────────────────────────────────────────────

function escHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Start ─────────────────────────────────────────────────────────────────
init();