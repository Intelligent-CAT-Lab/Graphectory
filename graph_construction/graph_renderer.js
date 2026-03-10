// graph_renderer.js — SVG graph rendering with Dagre layout.

// ==================== Layout and Coordinate Normalization ====================
function layoutGraph() {
    // Create dagre graph
    const g = new dagre.graphlib.Graph({ multigraph: true });
    g.setGraph({
        rankdir: 'LR',
        ranksep: 120,
        nodesep: 60,
        edgesep: 40,
        marginx: 40,
        marginy: 40
    });
    g.setDefaultEdgeLabel(() => ({}));
    
    // Add nodes with sizing based on label content.
    // Apply verbosity setting to choose which label to use for sizing.
    nodesData.forEach(node => {
        const label = settings.nodeVerbosity ? node.label : node.label_minimal;
        node.displayLabel = label;  // Store for rendering
        
        const lines = label.split('\\n');
        const line1Len = lines[0] ? Math.min(lines[0].length, 30) : 0;
        const restMax  = lines.slice(1).reduce((m, l) => Math.max(m, Math.min(l.length, 35)), 0);
        const widthFromL1   = line1Len  * 7.5 + 24;
        const widthFromRest = restMax   * 5.5 + 24;
        const width  = Math.max(100, widthFromL1, widthFromRest);
        const height = Math.max(40, lines.length * 16 + 12);
        g.setNode(node.id, { width, height, ...node });
    });
    
    // Add edges - use unique names for multigraph support
    edgesData.forEach((edge, idx) => {
        g.setEdge(edge.from, edge.to, edge, `edge-${idx}`);
    });
    
    // Layout
    dagre.layout(g);
    
    // Calculate bounding box
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    g.nodes().forEach(nodeId => {
        const node = g.node(nodeId);
        const left = node.x - node.width / 2;
        const right = node.x + node.width / 2;
        const top = node.y - node.height / 2;
        const bottom = node.y + node.height / 2;
        
        minX = Math.min(minX, left);
        maxX = Math.max(maxX, right);
        minY = Math.min(minY, top);
        maxY = Math.max(maxY, bottom);
    });
    
    // Add padding
    const padding = 40;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;
    
    // Normalize coordinates to start at (0, 0)
    const offsetX = -minX;
    const offsetY = -minY;
    
    // Update node positions
    g.nodes().forEach(nodeId => {
        const node = g.node(nodeId);
        node.x += offsetX;
        node.y += offsetY;
    });
    
    // Update edge points
    g.edges().forEach(e => {
        const edge = g.edge(e);
        if (edge.points) {
            edge.points.forEach(point => {
                point.x += offsetX;
                point.y += offsetY;
            });
        }
    });
    
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    
    return { g, graphWidth, graphHeight };
}

// ==================== SVG Creation ====================
function createSVG(graphWidth, graphHeight) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${graphWidth} ${graphHeight}`);
    svg.setAttribute('width', graphWidth);
    svg.setAttribute('height', graphHeight);
    svg.style.overflow = 'visible';
    return svg;
}

// ==================== Markers ====================
function createMarkers(svg) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    
    // Exec arrow (regular)
    const markerExec = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerExec.setAttribute('id', 'arrowhead-exec');
    markerExec.setAttribute('markerWidth', '10');
    markerExec.setAttribute('markerHeight', '10');
    markerExec.setAttribute('refX', '9');
    markerExec.setAttribute('refY', '3');
    markerExec.setAttribute('orient', 'auto');
    const pathExec = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathExec.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathExec.setAttribute('class', 'arrowhead');
    markerExec.appendChild(pathExec);
    defs.appendChild(markerExec);
    
    // Exec arrow for multi-node steps (blue)
    const markerExecMulti = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerExecMulti.setAttribute('id', 'arrowhead-exec-multi');
    markerExecMulti.setAttribute('markerWidth', '10');
    markerExecMulti.setAttribute('markerHeight', '10');
    markerExecMulti.setAttribute('refX', '9');
    markerExecMulti.setAttribute('refY', '3');
    markerExecMulti.setAttribute('orient', 'auto');
    const pathExecMulti = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathExecMulti.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathExecMulti.setAttribute('fill', '#3498db');
    markerExecMulti.appendChild(pathExecMulti);
    defs.appendChild(markerExecMulti);

    // Exec arrow for thought-continuation edges (red)
    const markerThoughtCont = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerThoughtCont.setAttribute('id', 'arrowhead-thought-cont');
    markerThoughtCont.setAttribute('markerWidth', '10');
    markerThoughtCont.setAttribute('markerHeight', '10');
    markerThoughtCont.setAttribute('refX', '9');
    markerThoughtCont.setAttribute('refY', '3');
    markerThoughtCont.setAttribute('orient', 'auto');
    const pathThoughtCont = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathThoughtCont.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathThoughtCont.setAttribute('fill', '#e74c3c');
    markerThoughtCont.appendChild(pathThoughtCont);
    defs.appendChild(markerThoughtCont);
    
    // Hier arrow
    const markerHier = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    markerHier.setAttribute('id', 'arrowhead-hier');
    markerHier.setAttribute('markerWidth', '10');
    markerHier.setAttribute('markerHeight', '10');
    markerHier.setAttribute('refX', '9');
    markerHier.setAttribute('refY', '3');
    markerHier.setAttribute('orient', 'auto');
    const pathHier = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathHier.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    pathHier.setAttribute('class', 'arrowhead hier');
    markerHier.appendChild(pathHier);
    defs.appendChild(markerHier);
    
    svg.appendChild(defs);
    return defs;
}

// ==================== Edge Rendering ====================

/**
 * Map a thought_length to a stroke width.
 * Uses settings.thoughtQuotes to choose raw or clean length.
 */
function getThoughtLength(edge) {
    return settings.thoughtQuotes ? edge.thought_length_clean : edge.thought_length_raw;
}

function thoughtToWidth(thoughtLength) {
    if (thoughtLength <= 0) return 1;
    const capped = Math.min(thoughtLength, 5000);
    if (capped <= 200)  return 6 + (capped / 200) * 6;
    if (capped <= 800)  return 12   + ((capped - 200) / 600) * 12;
    return 24 + ((capped - 800) / 700) * 12;
}

function calculateEdgeStyle(edge) {
    if (edge.type === 'hier') {
        return {
            strokeWidth:    1.5,
            strokeDasharray: '6,4',
            stroke:          '#27ae60',
            markerEnd:       'url(#arrowhead-hier)',
            opacity:         0.75,
        };
    }

    if (edge.type === 'exec') {
        // Thought-continuation: model reused/extended prior step's thought verbatim
        if (edge.is_thought_continuation) {
            return {
                strokeWidth:    2,
                strokeDasharray: '',
                stroke:          '#e74c3c',
                markerEnd:       'url(#arrowhead-thought-cont)',
                opacity:         0.9,
            };
        }

        // Intra-step edges after the first (&&-chained commands)
        if (edge.is_multi_node_step) {
            return {
                strokeWidth:    1,
                strokeDasharray: '4,4',
                stroke:          '#3498db',
                markerEnd:       'url(#arrowhead-exec-multi)',
                opacity:         0.9,
            };
        }
        const tlen = getThoughtLength(edge);
        if (tlen === 0) {
            return {
                strokeWidth:    1.5,
                strokeDasharray: '4,4',
                stroke:          '#95a5a6',
                markerEnd:       'url(#arrowhead-exec)',
                opacity:         0.75,
            };
        }
        // Body stays thin; arrowhead marker scales with thought length.
        const w = thoughtToWidth(tlen);
        return {
            strokeWidth:    1.5,
            strokeDasharray: '',
            stroke:          '#7f8c8d',
            markerEnd:       `url(#arrowhead-exec-w${Math.round(w)})`,
            opacity:         1,
        };
    }

    return { strokeWidth: 1, strokeDasharray: '', stroke: '#bbb',
             markerEnd: 'url(#arrowhead-exec)', opacity: 1 };
}

/**
 * Build a smooth cubic-bezier path string from dagre waypoints.
 * Dagre returns 3+ collinear-ish points; we turn them into a smooth spline.
 */
function pointsToPath(points, offsetY) {
    if (!points || points.length === 0) return '';
    if (points.length === 1) {
        return `M ${points[0].x} ${points[0].y + offsetY}`;
    }
    // Move to first point
    let d = `M ${points[0].x} ${points[0].y + offsetY}`;
    if (points.length === 2) {
        d += ` L ${points[1].x} ${points[1].y + offsetY}`;
        return d;
    }
    // For 3+ points use cubic bezier with control points at 1/3 & 2/3 between segments
    for (let i = 1; i < points.length - 1; i++) {
        const x0 = points[i - 1].x, y0 = points[i - 1].y + offsetY;
        const x1 = points[i].x,     y1 = points[i].y + offsetY;
        const x2 = points[i + 1].x, y2 = points[i + 1].y + offsetY;
        const cpx1 = x0 + (x1 - x0) * 0.67;
        const cpy1 = y0 + (y1 - y0) * 0.67;
        const cpx2 = x1 - (x2 - x1) * 0.33;
        const cpy2 = y1 - (y2 - y1) * 0.33;
        d += ` C ${cpx1} ${cpy1} ${cpx2} ${cpy2} ${x1} ${y1}`;
    }
    // Final segment to last point
    const last = points[points.length - 1];
    d += ` L ${last.x} ${last.y + offsetY}`;
    return d;
}

/**
 * Map observation length to a square half-size (radius) in SVG px.
 * Uses a power curve for more visual variation across the range.
 * Range: ~5px (tiny) → ~28px (very long, ≥100000 chars).
 */
function obsLengthToSize(obsLength) {
    if (!obsLength || obsLength <= 0) return 0;
    const capped = Math.min(obsLength, 100000);
    // Power curve: small differences at the low end are magnified
    const t    = capped / 8000;                  // 0‥1
    const half = 5 + Math.pow(t, 0.55) * 23;    // 5px → 28px
    return half;
}

/**
 * Return the colour for an observation square given its outcome.
 */
function obsOutcomeColor(outcome) {
    if (outcome === 'success') return '#4ade80';
    if (outcome === 'failure') return '#ff8080';
    return '#8899cc';   // neutral
}

/**
 * Return {x, y} at fraction t (0–1) of the polyline's total arc length.
 */
function interpOnPath(points, t, offsetY) {
    if (!points || points.length <= 1) {
        const p = points && points[0] ? points[0] : { x: 0, y: 0 };
        return { x: p.x, y: p.y + offsetY };
    }
    const segs = [];
    let total = 0;
    for (let i = 1; i < points.length; i++) {
        const dx = points[i].x - points[i-1].x;
        const dy = points[i].y - points[i-1].y;
        const len = Math.sqrt(dx*dx + dy*dy);
        segs.push(len);
        total += len;
    }
    const target = t * total;
    let acc = 0;
    for (let i = 0; i < segs.length; i++) {
        if (acc + segs[i] >= target) {
            const frac = segs[i] > 0 ? (target - acc) / segs[i] : 0;
            return {
                x: points[i].x + frac * (points[i+1].x - points[i].x),
                y: (points[i].y + offsetY) + frac * ((points[i+1].y + offsetY) - (points[i].y + offsetY)),
            };
        }
        acc += segs[i];
    }
    const last = points[points.length - 1];
    return { x: last.x, y: last.y + offsetY };
}

function renderEdges(svg, g, defs) {
    // Pre-compute edge counts per (from,to) pair for multi-edge offsetting
    const edgesByPair = {};
    edgesData.forEach((edge, idx) => {
        const key = `${edge.from}-${edge.to}`;
        if (!edgesByPair[key]) edgesByPair[key] = [];
        edgesByPair[key].push({ ...edge, idx });
    });

    // Create per-width arrowhead markers for thought-length scaling.
    function makeArrowMarker(id, w, color) {
        const m = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        m.setAttribute('id', id);
        m.setAttribute('markerWidth',  String(6 + w));
        m.setAttribute('markerHeight', String(6 + w));
        m.setAttribute('refX', String(5 + w));
        m.setAttribute('refY', String((4 + w) / 2));
        m.setAttribute('orient', 'auto');
        const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        p.setAttribute('d', `M0,0 L0,${4 + w} L${5 + w},${(4 + w) / 2} z`);
        p.setAttribute('fill', color);
        m.appendChild(p);
        defs.appendChild(m);
    }

    // Pre-create per-width arrowhead markers for thought length scaling.
    const thoughtWidthsSeen = new Set();
    edgesData.forEach(edge => {
        if (edge.type === 'exec' && !edge.is_multi_node_step && !edge.is_thought_continuation) {
            const tlen = getThoughtLength(edge);
            if (tlen > 0) thoughtWidthsSeen.add(Math.round(thoughtToWidth(tlen)));
        }
    });
    thoughtWidthsSeen.forEach(w => makeArrowMarker(`arrowhead-exec-w${w}`, w, '#7f8c8d'));

    g.edges().forEach(e => {
        const edge        = g.edge(e);
        const edgeKey     = `${e.v}-${e.w}`;
        const edgesInPair = edgesByPair[edgeKey] || [];
        const edgeIndex   = edgesInPair.findIndex(
            ed => ed.type === edge.type && ed.label === edge.label
        );
        const totalEdges  = edgesInPair.length;

        const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        edgeGroup.setAttribute('class', `edge ${edge.type}`);

        const style  = calculateEdgeStyle(edge);
        const points = edge.points;

        let offsetY = 0;
        if (totalEdges > 1) {
            offsetY = (edgeIndex - (totalEdges - 1) / 2) * 14;
        }

        // ── Edge path (single, uniform thin body + scaled arrowhead) ────────
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d',            pointsToPath(points, offsetY));
        path.setAttribute('fill',         'none');
        path.setAttribute('stroke',       style.stroke);
        path.setAttribute('stroke-width', String(style.strokeWidth));
        path.setAttribute('opacity',      String(style.opacity));
        if (style.strokeDasharray) {
            path.setAttribute('stroke-dasharray', style.strokeDasharray);
        }
        path.setAttribute('marker-end', style.markerEnd);
        edgeGroup.appendChild(path);

        // ── Observation square ───────────────────────────────────────────────
        // Drawn on top of the edge at ~25% arc length.
        // Only for first-in-step exec edges when showObservation is on.
        const showObsSquare = (
            settings.showObservation &&
            edge.type === 'exec' &&
            edge.is_first_in_step &&
            !edge.is_multi_node_step &&
            !edge.is_thought_continuation &&
            edge.obs_length > 0 &&
            points && points.length >= 2
        );
        if (showObsSquare) {
            const sqPt   = interpOnPath(points, 0.25, offsetY);
            const half   = obsLengthToSize(edge.obs_length);
            const color  = obsOutcomeColor(edge.obs_outcome);
            const sq     = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            sq.setAttribute('x',            String(sqPt.x - half));
            sq.setAttribute('y',            String(sqPt.y - half));
            sq.setAttribute('width',        String(half * 2));
            sq.setAttribute('height',       String(half * 2));
            sq.setAttribute('rx',           String(half * 0.4));   // rounded corners
            sq.setAttribute('ry',           String(half * 0.4));
            sq.setAttribute('fill',         color);
            sq.setAttribute('opacity',      '0.85');
            sq.setAttribute('stroke',       '#1a1f2e');
            sq.setAttribute('stroke-width', '1');
            edgeGroup.appendChild(sq);
        }

        // ── Step-number label (exec edges only) ──────────────────────────────
        if (edge.label && edge.type === 'exec') {
            const midIdx   = Math.floor(points.length / 2);
            const midPoint = points[midIdx];
            const text     = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x',           midPoint.x);
            text.setAttribute('y',           midPoint.y + offsetY - 5);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size',   '10');
            text.setAttribute('fill',        '#7f8c8d');
            text.textContent = edge.label;
            edgeGroup.appendChild(text);
        }

        svg.appendChild(edgeGroup);
    });
}

// ==================== Node Rendering ====================
function makeNodeRect(node, fillAttr) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x',      node.x - node.width  / 2);
    rect.setAttribute('y',      node.y - node.height / 2);
    rect.setAttribute('width',  node.width);
    rect.setAttribute('height', node.height);
    rect.setAttribute('rx',     '5');
    rect.setAttribute('ry',     '5');
    rect.setAttribute('fill',   fillAttr);
    rect.setAttribute('stroke',       '#2c3e50');
    rect.setAttribute('stroke-width', '1.5');
    return rect;
}

function renderNodes(svg, g, defs) {
    g.nodes().forEach(nodeId => {
        const node      = g.node(nodeId);
        const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodeGroup.setAttribute('class',        'node');
        nodeGroup.setAttribute('data-id',      nodeId);
        nodeGroup.setAttribute('data-tooltip', node.tooltip);

        // Left-click opens the detail sidebar for this node
        nodeGroup.addEventListener('click', (e) => {
            e.stopPropagation();
            openSidebar(node);
        });

        if (node.colors && node.colors.length > 1) {
            const gradId = `grad-${nodeId.replace(/[^a-zA-Z0-9]/g, '_')}`;
            const grad   = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            grad.setAttribute('id', gradId);
            grad.setAttribute('x1', '0%'); grad.setAttribute('y1', '0%');
            grad.setAttribute('x2', '100%'); grad.setAttribute('y2', '0%');
            node.colors.forEach((color, i) => {
                const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                s1.setAttribute('offset',     `${i / node.colors.length * 100}%`);
                s1.setAttribute('stop-color', color);
                grad.appendChild(s1);
                const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
                s2.setAttribute('offset',     `${(i + 1) / node.colors.length * 100}%`);
                s2.setAttribute('stop-color', color);
                grad.appendChild(s2);
            });
            defs.appendChild(grad);
            nodeGroup.appendChild(makeNodeRect(node, `url(#${gradId})`));
        } else {
            nodeGroup.appendChild(makeNodeRect(node, node.color || '#CFE0F6'));
        }
        
        // Add triangular "hat" for nodes that had cd command stripped
        if (node.has_cd) {
            const hatSize = 12;
            const topY = node.y - node.height / 2;
            const centerX = node.x;
            
            const triangle = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            const d = `M ${centerX} ${topY - hatSize} ` +
                     `L ${centerX - hatSize} ${topY} ` +
                     `L ${centerX + hatSize} ${topY} Z`;
            triangle.setAttribute('d', d);
            triangle.setAttribute('fill', '#f39c12');
            triangle.setAttribute('stroke', '#e67e22');
            triangle.setAttribute('stroke-width', '1.5');
            
            nodeGroup.appendChild(triangle);
        }

        
        const lines = node.displayLabel.split('\\n');
        const lineHeight = 16;
        const totalTextHeight = lines.length * lineHeight;
        const startY = node.y - totalTextHeight / 2 + lineHeight / 2;
        
        lines.forEach((line, i) => {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', node.x);
            text.setAttribute('y', startY + i * lineHeight);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            
            if (i === 0) {
                // Line 1: action title — bold, dark, readable
                text.setAttribute('font-weight', 'bold');
                text.setAttribute('font-size', '12');
                text.setAttribute('fill', '#1a1a2e');
            } else if (i === 1) {
                // Line 2: step index — medium, slightly muted
                text.setAttribute('font-size', '10');
                text.setAttribute('fill', '#555');
            } else {
                // Lines 3+: path / view-range — small, light grey
                text.setAttribute('font-size', '9');
                text.setAttribute('fill', '#666');
            }
            
            text.textContent = line;
            nodeGroup.appendChild(text);
        });
        
        svg.appendChild(nodeGroup);
    });
}

// ==================== Tooltip ====================
function setupTooltips() {
    const tooltip = document.getElementById('tooltip');
    document.querySelectorAll('.node').forEach(node => {
        node.addEventListener('mouseenter', (e) => {
            const tooltipContent = e.currentTarget.getAttribute('data-tooltip');
            tooltip.innerHTML = tooltipContent;
            tooltip.style.display = 'block';
        });
        
        node.addEventListener('mousemove', (e) => {
            tooltip.style.left = (e.pageX + 15) + 'px';
            tooltip.style.top  = (e.pageY + 15) + 'px';
        });
        
        node.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
    });
}

// ==================== Detail Sidebar ====================
let sidebarNodeId = null;
let sidebarStepIdx = 0;        // which step visit is being shown (0-based within step_data)

/**
 * Open (or refresh) the sidebar for the given node data object.
 * Called from the click handler set up in renderNodes.
 */
function openSidebar(node) {
    sidebarNodeId = node.id;
    sidebarStepIdx = 0;         // reset to first visit on each new node

    const sidebar  = document.getElementById('detailSidebar');
    const title    = document.getElementById('sidebarTitle');
    const stepTabs = document.getElementById('stepTabs');

    // Derive a human-readable title from the node label
    const labelLine = (node.displayLabel || node.label || node.id).split('\\n')[0];
    title.textContent = labelLine;
    title.title       = labelLine;

    // Build step-picker tabs only when there are multiple visits
    const steps = node.step_data || [];
    stepTabs.innerHTML = '';
    if (steps.length > 1) {
        stepTabs.style.display = 'flex';
        steps.forEach((sd, i) => {
            const btn = document.createElement('button');
            btn.className   = 'step-tab' + (i === 0 ? ' active' : '');
            btn.textContent = `Step ${sd.step_idx}`;
            btn.addEventListener('click', () => {
                sidebarStepIdx = i;
                // Re-render content and update active tab
                document.querySelectorAll('.step-tab').forEach((b, j) =>
                    b.classList.toggle('active', j === i)
                );
                renderSidebarContent(node, i);
            });
            stepTabs.appendChild(btn);
        });
    } else {
        stepTabs.style.display = 'none';
    }

    renderSidebarContent(node, 0);

    sidebar.classList.add('open');
}

function closeSidebar() {
    const sidebar = document.getElementById('detailSidebar');
    sidebar.classList.remove('open');
    sidebarNodeId = null;
    // Clear content after the CSS transition so the DOM collapse never races
    // with the width animation and causes a page-height flash.
    setTimeout(() => {
        if (!sidebar.classList.contains('open')) {
            const tabs    = document.getElementById('stepTabs');
            const content = document.getElementById('sidebarContent');
            if (tabs)    tabs.innerHTML    = '';
            if (content) content.innerHTML = '';
        }
    }, 250);  // matches the 0.22s transition + small buffer
}

/**
 * Render thought / action / observation for visit index `visitIdx` of `node`.
 */
function renderSidebarContent(node, visitIdx) {
    const steps = node.step_data || [];
    const sd    = steps[visitIdx] || {};

    const thought     = sd.thought     || '';
    const action      = sd.action      || '';
    const observation = sd.observation || '';

    const container = document.getElementById('sidebarContent');
    container.innerHTML = '';

    container.appendChild(makeSidebarSection('Thought',     'thought',     thought));
    container.appendChild(makeSidebarSection('Action',      'action',      action));
    container.appendChild(makeSidebarSection('Observation', 'observation', observation));
}

/**
 * Build a collapsible section element with a sticky header.
 * Sections for empty text are shown as "(empty)" and start collapsed.
 */
function makeSidebarSection(title, cssClass, text) {
    const section = document.createElement('div');
    section.className = 'sidebar-section';

    const isEmpty  = !text || !text.trim();
    let collapsed  = isEmpty;                 // start collapsed when empty

    const header = document.createElement('div');
    header.className = 'sidebar-section-header';

    const label = document.createElement('span');
    label.className = `section-label ${cssClass}`;
    label.textContent = title;

    const lenSpan = document.createElement('span');
    lenSpan.className = 'section-len';
    lenSpan.textContent = isEmpty ? '' : `${text.length} chars`;

    const toggle = document.createElement('span');
    toggle.className = 'section-toggle' + (collapsed ? ' collapsed' : '');
    toggle.textContent = '▾';

    header.appendChild(label);
    header.appendChild(lenSpan);
    header.appendChild(toggle);

    const body = document.createElement('div');
    body.className = 'sidebar-section-body' + (isEmpty ? ' empty' : '');
    body.textContent = isEmpty ? '(empty)' : text;
    if (collapsed) body.style.display = 'none';

    header.addEventListener('click', () => {
        collapsed = !collapsed;
        body.style.display = collapsed ? 'none' : '';
        toggle.classList.toggle('collapsed', collapsed);
    });

    section.appendChild(header);
    section.appendChild(body);
    return section;
}

// ==================== Fullscreen ====================
// Fullscreen the entire document so the browser owns the whole viewport.
// A CSS class is toggled on .graph-container so it paints edge-to-edge while
// fullscreen, then removed on exit so layout returns to its original state.
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen().catch(() => {});
    }
}

// Handles Esc key, button click, and any other exit path uniformly.
document.addEventListener('fullscreenchange', _onFullscreenChange);
document.addEventListener('webkitfullscreenchange', _onFullscreenChange);

function _onFullscreenChange() {
    const container = document.querySelector('.graph-container');
    if (document.fullscreenElement) {
        container.classList.add('fullscreen-active');
    } else {
        container.classList.remove('fullscreen-active');
    }
    // Double rAF: first frame finishes the DOM update, second gets real dimensions.
    requestAnimationFrame(() => requestAnimationFrame(fitToScreen));
}

// ==================== Zoom and Pan Controls ====================
let currentScale = 1;
let currentX = 0;
let currentY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;
let graphEl;
let svg;
let graphWidth;
let graphHeight;

function updateTransform() {
    svg.style.transform = `translate(${currentX}px, ${currentY}px) scale(${currentScale})`;
    svg.style.transformOrigin = '0 0';
}

function fitToScreen() {
    // Use the #graph div's own dimensions — these are correct both in normal
    // layout (CSS height: 900px) and when fullscreen-active forces it to fill
    // the viewport via position:fixed.
    const w = graphEl.clientWidth  || graphEl.offsetWidth;
    const h = graphEl.clientHeight || graphEl.offsetHeight;

    const scaleX = w / graphWidth;
    const scaleY = h / graphHeight;
    currentScale = Math.min(scaleX, scaleY, 1) * 0.95;

    const scaledWidth  = graphWidth  * currentScale;
    const scaledHeight = graphHeight * currentScale;
    currentX = (w - scaledWidth)  / 2;
    currentY = (h - scaledHeight) / 2;

    updateTransform();
}

function zoomIn() {
    const centerX = graphEl.clientWidth  / 2;
    const centerY = graphEl.clientHeight / 2;
    
    const oldScale = currentScale;
    currentScale = currentScale * 1.2;
    const scaleRatio = currentScale / oldScale;
    
    currentX = centerX - (centerX - currentX) * scaleRatio;
    currentY = centerY - (centerY - currentY) * scaleRatio;
    
    updateTransform();
}

function zoomOut() {
    const centerX = graphEl.clientWidth  / 2;
    const centerY = graphEl.clientHeight / 2;
    
    const oldScale = currentScale;
    currentScale = currentScale / 1.2;
    const scaleRatio = currentScale / oldScale;
    
    currentX = centerX - (centerX - currentX) * scaleRatio;
    currentY = centerY - (centerY - currentY) * scaleRatio;
    
    updateTransform();
}

// ==================== Mouse Wheel Zoom ====================
function setupWheelZoom() {
    graphEl.addEventListener('wheel', (e) => {
        e.preventDefault();
        
        const rect = graphEl.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const oldScale = currentScale;
        const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1;
        currentScale = currentScale * zoomDelta;
        
        const scaleRatio = currentScale / oldScale;
        currentX = mouseX - (mouseX - currentX) * scaleRatio;
        currentY = mouseY - (mouseY - currentY) * scaleRatio;
        
        updateTransform();
    }, { passive: false });
}

// ==================== Pan with Drag ====================
function setupPanning() {
    let dragMoved = false;   // true once the pointer moves >4px after mousedown

    graphEl.addEventListener('mousedown', (e) => {
        if (e.target.closest('.node')) return;
        if (e.target.closest('.detail-sidebar')) return;

        isDragging = true;
        dragMoved  = false;
        startX = e.clientX - currentX;
        startY = e.clientY - currentY;
        graphEl.style.cursor = 'grabbing';
        // Do NOT call e.preventDefault() here — that would suppress the
        // subsequent 'click' event and break the X button on the sidebar.
    });

    graphEl.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - (startX + currentX);
        const dy = e.clientY - (startY + currentY);
        if (!dragMoved && Math.sqrt(dx*dx + dy*dy) > 4) dragMoved = true;
        currentX = e.clientX - startX;
        currentY = e.clientY - startY;
        updateTransform();
        // Suppress text selection only while actually dragging
        e.preventDefault();
    });

    graphEl.addEventListener('mouseup', (e) => {
        if (isDragging && !dragMoved && !e.target.closest('.node')) {
            // Genuine click on whitespace → close sidebar
            closeSidebar();
        }
        isDragging = false;
        dragMoved  = false;
        graphEl.style.cursor = 'grab';
    });

    graphEl.addEventListener('mouseleave', () => {
        isDragging = false;
        dragMoved  = false;
        graphEl.style.cursor = 'grab';
    });
}

// ==================== Sidebar Resize ====================
function setupSidebarResize() {
    const resizer = document.getElementById('sidebarResizer');
    const sidebar = document.getElementById('detailSidebar');
    if (!resizer || !sidebar) return;

    let isResizing = false;
    let startX     = 0;
    let startWidth = 0;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX     = e.clientX;
        startWidth = sidebar.offsetWidth;
        document.body.style.cursor    = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        // Dragging left (towards graph) increases width; right decreases it.
        const delta    = startX - e.clientX;
        const newWidth = Math.max(260, Math.min(900, startWidth + delta));
        sidebar.style.width = newWidth + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (!isResizing) return;
        isResizing = false;
        document.body.style.cursor    = '';
        document.body.style.userSelect = '';
    });
}

// ==================== Initialization ====================

function initializeGraph() {
    // Dagre must be present before we attempt layout.  The guard below catches
    // the case where the script tag failed to load (CDN timeout, etc.) and
    // renders a clear error rather than a blank canvas.
    if (typeof dagre === 'undefined' || !dagre.graphlib || !dagre.layout) {
        const graphEl = document.getElementById('graph');
        if (graphEl) {
            graphEl.innerHTML =
                '<div style="padding:32px;font-family:monospace;color:#e74c3c;">'
                + '<strong>⚠ dagre failed to load</strong><br><br>'
                + 'The graph layout library did not initialise correctly. '
                + 'Check the browser console (F12) for details, then reload the page.'
                + '</div>';
        }
        console.error('[graph] dagre is not defined — cannot render graph.');
        return;
    }

    try {
        graphEl = document.getElementById('graph');

        const { g, graphWidth: gw, graphHeight: gh } = layoutGraph();
        graphWidth  = gw;
        graphHeight = gh;

        svg = createSVG(graphWidth, graphHeight);
        const defs = createMarkers(svg);

        renderEdges(svg, g, defs);
        renderNodes(svg, g, defs);

        graphEl.appendChild(svg);

        setupTooltips();
        setupWheelZoom();
        setupPanning();
        setupSidebarResize();

        setTimeout(fitToScreen, 150);
    } catch (err) {
        console.error('[graph] initializeGraph failed:', err);
        throw err;  // Propagate to the window error handler for UI display.
    }
}

// Dagre is loaded via a synchronous <script> tag injected by the Python
// renderer, so it will always be available by the time this script runs.
// The _tryInit poll exists as a safety net for any async-load edge case.
function _tryInit(retriesLeft) {
    if (typeof dagre !== 'undefined' && dagre.graphlib && dagre.layout) {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => { initializeGraph(); _wireSidebarClose(); });
        } else {
            initializeGraph();
            _wireSidebarClose();
        }
    } else if (retriesLeft > 0) {
        setTimeout(() => _tryInit(retriesLeft - 1), 100);
    } else {
        // All retries exhausted — call initializeGraph so the error UI is shown.
        console.error('[graph] dagre did not become available after maximum retries.');
        initializeGraph();
        _wireSidebarClose();
    }
}

_tryInit(30);  // Up to 3 s of polling before surfacing the error UI.

function _wireSidebarClose() {
    const btn = document.getElementById('sidebarCloseBtn');
    if (!btn) return;
    // Belt-and-suspenders: both click and mouseup so the event fires even if
    // something upstream cancelled the synthetic click (e.g. drag-end logic).
    btn.addEventListener('click',   closeSidebar);
    btn.addEventListener('mouseup', (e) => { e.stopPropagation(); closeSidebar(); });
}