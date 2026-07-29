// graph_renderer.js — SVG graph rendering with Dagre layout.

// ==================== Label Sizing ====================
const NODE_LABEL_MIN_WIDTH = 100;
const NODE_LABEL_PADDING_X = 24;
const LARGE_LAYOUT_NODE_LIMIT = 250;
const LARGE_LAYOUT_EDGE_LIMIT = 500;
const REPEATED_BLUE_EDGE_THRESHOLD = 10;
const REPEATED_COMMAND_COLORS = [
    '#f39c12', '#8e44ad', '#16a085', '#e74c3c',
    '#2980b9', '#d35400', '#27ae60', '#c0392b',
];
const SUBGRAPH_NODE_THRESHOLD = 100;
const SUBGRAPH_MIN_NODES = 50;
const SUBGRAPH_MAX_NODES = 150;

let activeNodesData = nodesData;
let activeEdgesData = edgesData;
let subgraphSegments = [];
let activeSubgraphIndex = -1;
let subgraphsPrepared = false;

function activeNodes() {
    return activeNodesData || nodesData;
}

function activeEdges() {
    return activeEdgesData || edgesData;
}

function nodeFirstStep(node, fallback) {
    const steps = Array.isArray(node.step_indices) ? node.step_indices : [];
    const numericSteps = steps
        .map(Number)
        .filter(step => Number.isFinite(step));
    return numericSteps.length ? Math.min(...numericSteps) : fallback;
}

function nodeLastStep(node, fallback) {
    const steps = Array.isArray(node.step_indices) ? node.step_indices : [];
    const numericSteps = steps
        .map(Number)
        .filter(step => Number.isFinite(step));
    return numericSteps.length ? Math.max(...numericSteps) : fallback;
}

function nodeIsPlan(node) {
    return (Array.isArray(node.phases)
        && node.phases.some(phase => String(phase).toLowerCase() === 'plan'))
        || String(node?.color || '').toLowerCase() === '#f4d06f';
}

function buildNodeOrder() {
    return nodesData
        .map((node, index) => ({ node, index }))
        .sort((left, right) =>
            nodeFirstStep(left.node, left.index) - nodeFirstStep(right.node, right.index)
            || left.index - right.index,
        )
        .map(entry => entry.node);
}

function buildNodeDegrees() {
    const degrees = new Map(nodesData.map(node => [node.id, 0]));
    edgesData.forEach(edge => {
        if (degrees.has(edge.from)) degrees.set(edge.from, degrees.get(edge.from) + 1);
        if (degrees.has(edge.to) && edge.to !== edge.from) {
            degrees.set(edge.to, degrees.get(edge.to) + 1);
        }
    });
    return degrees;
}

function chooseBoundaryPlan(planPositions, start, minBoundary, maxBoundary, target) {
    const candidates = planPositions.filter(position =>
        position >= minBoundary && position <= maxBoundary,
    );
    if (!candidates.length) return null;
    return candidates.reduce((best, position) =>
        Math.abs(position - target) < Math.abs(best - target) ? position : best,
    );
}

function chooseBoundaryFallback(nodeOrder, degrees, start, minBoundary, maxBoundary, target) {
    const candidates = [];
    for (let position = minBoundary; position <= maxBoundary; position += 1) {
        const node = nodeOrder[position];
        candidates.push({
            position,
            degree: degrees.get(node.id) ?? Number.MAX_SAFE_INTEGER,
        });
    }
    candidates.sort((left, right) =>
        left.degree - right.degree
        || Math.abs(left.position - target) - Math.abs(right.position - target)
        || left.position - right.position,
    );
    return candidates[0]?.position ?? Math.min(maxBoundary, start + 99);
}

function buildSegmentGraph(segmentNodes, segmentNumber, isFirstSegment, isLastSegment) {
    const nodeIds = new Set(segmentNodes.map(node => node.id));
    const contextNodes = [];
    const contextByDirection = new Map();
    const contextEdgeKeys = new Set();

    const getContextNode = (direction) => {
        if (contextByDirection.has(direction)) return contextByDirection.get(direction);

        const boundaryNode = direction === 'incoming'
            ? segmentNodes[0]
            : segmentNodes[segmentNodes.length - 1];
        const boundaryStep = nodeFirstStep(boundaryNode, 0);
        const contextNode = {
            id: `__segment_${segmentNumber}_${direction}`,
            label: direction === 'incoming'
                ? 'previous segment\\ncontext'
                : 'next segment\\ncontext',
            tooltip: direction === 'incoming'
                ? 'Context anchor for edges entering from the previous segment.'
                : 'Context anchor for edges continuing into the next segment.',
            color: '#d8e2e7',
            colors: ['#d8e2e7'],
            phases: ['general'],
            is_context: true,
            command: '',
            observation_length: 0,
            tool: '',
            subcommand: '',
            step_indices: [direction === 'incoming' ? boundaryStep - 0.5 : boundaryStep + 0.5],
            step_data: [],
        };
        contextByDirection.set(direction, contextNode);
        contextNodes.push(contextNode);
        return contextNode;
    };

    const segmentEdges = [];
    edgesData.forEach(edge => {
        const fromInside = nodeIds.has(edge.from);
        const toInside = nodeIds.has(edge.to);
        if (fromInside && toInside) {
            segmentEdges.push({ ...edge });
            return;
        }

        if (toInside && !fromInside) {
            if (isFirstSegment) return;
            const contextNode = getContextNode('incoming');
            const key = `${contextNode.id}|${edge.to}|${edge.type}`;
            if (contextEdgeKeys.has(key)) return;
            contextEdgeKeys.add(key);
            segmentEdges.push({
                ...edge,
                from: contextNode.id,
                is_context: true,
                label: '',
            });
            return;
        }

        if (fromInside && !toInside) {
            if (isLastSegment) return;
            const contextNode = getContextNode('outgoing');
            const key = `${edge.from}|${contextNode.id}|${edge.type}`;
            if (contextEdgeKeys.has(key)) return;
            contextEdgeKeys.add(key);
            segmentEdges.push({
                ...edge,
                to: contextNode.id,
                is_context: true,
                label: '',
            });
        }
    });

    return { contextNodes, edges: segmentEdges };
}

function buildSubgraphSegments() {
    const nodeOrder = buildNodeOrder();
    if (nodeOrder.length <= SUBGRAPH_NODE_THRESHOLD) {
        return [{
            nodeIds: nodeOrder.map(node => node.id),
            edges: edgesData.map(edge => ({ ...edge })),
            label: 'Full trajectory',
        }];
    }

    const planPositions = nodeOrder
        .map((node, index) => nodeIsPlan(node) ? index : null)
        .filter(index => index !== null);
    const degrees = buildNodeDegrees();
    const segments = [];
    let start = 0;

    while (start < nodeOrder.length) {
        const remaining = nodeOrder.length - start;
        let boundary;

        if (remaining <= SUBGRAPH_MAX_NODES) {
            // Keep the tail intact. It is already within the requested range;
            // a missing final plan node is unavoidable when the run ends in a
            // command rather than an explicit plan action.
            boundary = nodeOrder.length - 1;
        } else {
            const minBoundary = start + SUBGRAPH_MIN_NODES - 1;
            // Leave at least 50 nodes for the next overlapping segment.
            const maxBoundary = Math.min(
                start + SUBGRAPH_MAX_NODES - 1,
                nodeOrder.length - SUBGRAPH_MIN_NODES,
            );
            const target = Math.min(start + 99, maxBoundary);
            boundary = chooseBoundaryPlan(
                planPositions, start, minBoundary, maxBoundary, target,
            );
            if (boundary === null) {
                boundary = chooseBoundaryFallback(
                    nodeOrder, degrees, start, minBoundary, maxBoundary, target,
                );
            }
        }

        if (boundary <= start && start < nodeOrder.length - 1) {
            boundary = Math.min(start + SUBGRAPH_MAX_NODES - 1, nodeOrder.length - 1);
        }

        const segmentNodes = nodeOrder.slice(start, boundary + 1);
        const number = segments.length + 1;
        const segmentGraph = buildSegmentGraph(
            segmentNodes,
            number,
            start === 0,
            boundary >= nodeOrder.length - 1,
        );
        segments.push({
            nodeIds: segmentNodes.map(node => node.id),
            contextNodes: segmentGraph.contextNodes,
            edges: segmentGraph.edges,
            label: `Subgraph ${number}`,
        });

        if (boundary >= nodeOrder.length - 1) break;
        // Reuse the boundary node so adjacent subgraphs retain a visible
        // anchor and their plan-to-plan transition remains inspectable.
        start = boundary;
    }

    return segments;
}

function updateSubgraphPicker() {
    const picker = document.getElementById('subgraphPicker');
    const select = document.getElementById('subgraphSelect');
    if (!picker || !select) return;

    const segmented = nodesData.length > SUBGRAPH_NODE_THRESHOLD;
    picker.hidden = !segmented;
    if (!segmented) return;

    select.replaceChildren();
    const fullOption = document.createElement('option');
    fullOption.value = 'full';
    fullOption.textContent = `Full graph (${nodesData.length} nodes)`;
    select.appendChild(fullOption);
    subgraphSegments.forEach((segment, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${segment.label} (${segment.nodeIds.length} nodes)`;
        select.appendChild(option);
    });
    select.value = activeSubgraphIndex < 0 ? 'full' : String(activeSubgraphIndex);
}

function setActiveSubgraph(index, rerender = true) {
    if (!subgraphSegments.length) return;
    const fullGraph = String(index) === 'full';
    const nextIndex = fullGraph
        ? -1
        : Math.max(0, Math.min(Number(index) || 0, subgraphSegments.length - 1));
    activeSubgraphIndex = nextIndex;
    if (fullGraph) {
        activeNodesData = buildNodeOrder();
        activeEdgesData = edgesData.map(edge => ({ ...edge }));
    } else {
        const segment = subgraphSegments[nextIndex];
        const nodesById = new Map(nodesData.map(node => [node.id, node]));
        activeNodesData = segment.nodeIds
            .map(nodeId => nodesById.get(nodeId))
            .filter(Boolean)
            .concat(segment.contextNodes || []);
        activeEdgesData = segment.edges.map(edge => ({ ...edge }));
    }
    updateSubgraphPicker();
    if (rerender && graphEl) {
        closeSidebar();
        renderActiveSubgraph();
    }
}

function prepareSubgraphs() {
    if (subgraphsPrepared) return;
    subgraphSegments = buildSubgraphSegments();
    setActiveSubgraph(nodesData.length < 200 ? 'full' : 0, false);
    subgraphsPrepared = true;
}

function parseEdgeStep(label) {
    const match = String(label ?? '').trim().match(/^\d+$/);
    return match ? Number(match[0]) : null;
}

function consecutiveEdgeKey(edge) {
    return [
        edge.from,
        edge.to,
        edge.type,
        edge.is_multi_node_step,
        edge.is_thought_continuation,
        edge.is_first_in_step,
    ].join('|');
}

function collapseConsecutiveEdges(rawEdges) {
    const collapsed = [];
    const lastGroups = new Map();

    rawEdges.forEach(edge => {
        // Only chronological execution edges have step ranges. Structural
        // edges and semantic continuation edges retain their individual form.
        const step = (
            edge.type === 'exec' && !edge.is_thought_continuation
        ) ? parseEdgeStep(edge.label) : null;
        if (step === null) {
            collapsed.push({ ...edge });
            return;
        }

        const key = consecutiveEdgeKey(edge);
        const previous = lastGroups.get(key);
        if (previous && step === previous.lastStep + 1) {
            const merged = collapsed[previous.outputIndex];
            merged.label = `${previous.firstStep}-${step}`;
            merged.collapsed_count = (merged.collapsed_count || 1) + 1;
            merged.thought_length_raw = Math.max(
                merged.thought_length_raw || 0,
                edge.thought_length_raw || 0,
            );
            merged.thought_length_clean = Math.max(
                merged.thought_length_clean || 0,
                edge.thought_length_clean || 0,
            );
            merged.obs_length = Math.max(merged.obs_length || 0, edge.obs_length || 0);
            previous.lastStep = step;
            return;
        }

        const outputIndex = collapsed.length;
        collapsed.push({ ...edge });
        lastGroups.set(key, { outputIndex, firstStep: step, lastStep: step });
    });

    return collapsed;
}

function nodeActionName(node) {
    const parts = [node.command, node.subcommand, node.tool]
        .map(value => String(value || '').trim())
        .filter(Boolean);
    return parts[0] || String(node.label || node.id || 'action').split('\\n')[0];
}

function repeatedCommandColor(node) {
    const key = nodeActionName(node).toLowerCase();
    let hash = 0;
    for (let i = 0; i < key.length; i += 1) {
        hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
    }
    return REPEATED_COMMAND_COLORS[Math.abs(hash) % REPEATED_COMMAND_COLORS.length];
}

function summarizeRepeatedEdgeLabels(edges) {
    const labels = edges
        .map(edge => String(edge.label ?? '').trim())
        .filter(Boolean);
    if (!labels.length) return '';

    const steps = labels
        .map(parseEdgeStep)
        .filter(step => step !== null)
        .sort((left, right) => left - right);
    const uniqueSteps = [...new Set(steps)];
    if (uniqueSteps.length === labels.length && uniqueSteps.length > 1) {
        const consecutive = uniqueSteps.every(
            (step, index) => index === 0 || step === uniqueSteps[index - 1] + 1,
        );
        const lastStep = uniqueSteps[uniqueSteps.length - 1];
        if (consecutive) return `${uniqueSteps[0]}-${lastStep}`;
        return `${uniqueSteps[0]}-${lastStep} (x${edges.length})`;
    }

    return labels.length > 1
        ? `${labels[0]} (x${edges.length})`
        : labels[0];
}

function collapseRepeatedBlueEdges(rawEdges, collapsedSources) {
    const output = [];
    const groupedEdges = new Map();
    const groupedLabels = new Map();

    rawEdges.forEach(edge => {
        const isCollapsedBlueEdge = (
            collapsedSources.has(edge.from)
            && edge.type === 'exec'
            && edge.is_multi_node_step
        );
        if (!isCollapsedBlueEdge) {
            output.push({ ...edge });
            return;
        }

        // Keep one edge for every surviving neighbor. Removing every parallel
        // edge disconnects targets from the source and leaves orphaned nodes.
        const key = `${edge.from}|${edge.to}`;
        const existing = groupedEdges.get(key);
        if (!existing) {
            const representative = { ...edge, repeated_edge_count: 1 };
            groupedEdges.set(key, representative);
            groupedLabels.set(key, [edge]);
            output.push(representative);
            return;
        }

        existing.repeated_edge_count += 1;
        groupedLabels.get(key).push(edge);
        existing.thought_length_raw = Math.max(
            existing.thought_length_raw || 0,
            edge.thought_length_raw || 0,
        );
        existing.thought_length_clean = Math.max(
            existing.thought_length_clean || 0,
            edge.thought_length_clean || 0,
        );
        existing.obs_length = Math.max(existing.obs_length || 0, edge.obs_length || 0);
    });

    // The first occurrence has not been summarized yet. Do this after all
    // groups are complete so non-consecutive repetitions are represented too.
    groupedEdges.forEach((representative, key) => {
        if (representative.repeated_edge_count <= 1) return;
        representative.label = summarizeRepeatedEdgeLabels(groupedLabels.get(key));
    });

    return output;
}

function displayEdgesWithRepeatedFilter(rawEdges) {
    const copiedEdges = rawEdges.map(edge => ({ ...edge }));
    const visibleNodes = activeNodes();
    const nodeById = new Map(visibleNodes.map(node => [node.id, node]));

    // Reset decorations each time the display set is rebuilt. The underlying
    // graph remains unchanged, so switching the option off restores all edges.
    nodesData.forEach(node => { node.repeatedHatData = []; });
    if (!settings.filterRepeated) return collapseConsecutiveEdges(copiedEdges);

    const blueOutgoingCounts = new Map();
    copiedEdges.forEach(edge => {
        if (edge.type === 'exec' && edge.is_multi_node_step) {
            blueOutgoingCounts.set(
                edge.from,
                (blueOutgoingCounts.get(edge.from) || 0) + 1,
            );
        }
    });

    const collapsedSources = new Set(
        [...blueOutgoingCounts.entries()]
            .filter(([, count]) => count >= REPEATED_BLUE_EDGE_THRESHOLD)
            .map(([nodeId]) => nodeId),
    );
    if (!collapsedSources.size) return collapseConsecutiveEdges(copiedEdges);

    copiedEdges.forEach(edge => {
        if (!collapsedSources.has(edge.from)
            || edge.type !== 'exec'
            || !edge.is_multi_node_step) {
            return;
        }
        const source = nodeById.get(edge.from);
        const target = nodeById.get(edge.to);
        if (!source || !target) return;

        const color = repeatedCommandColor(source);
        const label = nodeActionName(source);
        if (!target.repeatedHatData.some(item => item.color === color)) {
            target.repeatedHatData.push({ color, label });
        }
    });

    return collapseConsecutiveEdges(
        collapseRepeatedBlueEdges(copiedEdges, collapsedSources),
    );
}

function estimateNodeTextWidth(line, lineIndex) {
    if (!line) return 0;
    if (lineIndex === 0) return line.length * 7.8;
    if (lineIndex === 1) return line.length * 6.2;
    return line.length * 6.4;
}

function containedNodeSize(lines) {
    const contentWidth = lines.reduce(
        (m, line, i) => Math.max(m, estimateNodeTextWidth(line, i)),
        0
    );
    return {
        width: Math.max(NODE_LABEL_MIN_WIDTH, contentWidth + NODE_LABEL_PADDING_X),
        height: Math.max(40, lines.length * 16 + 12),
    };
}

function isLargeGraph(nodeCount, edgeCount) {
    return (
        nodeCount >= LARGE_LAYOUT_NODE_LIMIT
        || edgeCount >= LARGE_LAYOUT_EDGE_LIMIT
    );
}

/**
 * Provide the small subset of the Dagre graph API used by the SVG renderer.
 *
 * Large trajectories use a deterministic layout and do not need Dagre's
 * graphlib. Avoiding graphlib here also prevents hundreds of nodes and edges
 * from being copied into a second general-purpose graph representation.
 */
function createLightweightGraph(displayEdges) {
    const nodeMap = new Map();
    activeNodes().forEach(node => {
        const label = node.label || node.id;
        node.displayLabel = label;
        const { width, height } = containedNodeSize(label.split('\\n'));
        nodeMap.set(node.id, { width, height, ...node });
    });

    const edgeRecords = displayEdges.map((edge, index) => ({
        id: { v: edge.from, w: edge.to, name: `edge-${index}` },
        value: edge,
    }));
    const edgeMap = new Map(edgeRecords.map(record => [record.id.name, record.value]));

    return {
        nodes: () => [...nodeMap.keys()],
        node: nodeId => nodeMap.get(nodeId),
        edges: () => edgeRecords.map(record => record.id),
        edge: edgeId => edgeMap.get(edgeId.name),
        setGraph: () => {},
    };
}

/**
 * Place very large graphs using trajectory order instead of Dagre.
 *
 * Dagre is useful for small and medium graphs, but its general-purpose
 * network-simplex layout becomes increasingly expensive for cyclic,
 * multi-edge trajectories. Large graphs still retain all of their nodes and
 * edges here; only the placement strategy changes. Nodes are grouped by
 * their first occurrence, which keeps the left-to-right reading order that
 * matters most for trajectory analysis.
 */
function layoutLargeGraph(g) {
    const padding = 50;
    const columnGap = 120;
    const rowGap = 60;
    const nodeIds = g.nodes();
    const nodeOrder = new Map(nodeIds.map((id, index) => [id, index]));

    const firstStep = (nodeId) => {
        const node = g.node(nodeId);
        const steps = Array.isArray(node.step_indices) ? node.step_indices : [];
        const step = Number(steps[0]);
        return Number.isFinite(step) ? step : nodeOrder.get(nodeId);
    };

    const columns = new Map();
    nodeIds
        .slice()
        .sort((left, right) => firstStep(left) - firstStep(right)
            || nodeOrder.get(left) - nodeOrder.get(right))
        .forEach(nodeId => {
            const step = firstStep(nodeId);
            if (!columns.has(step)) columns.set(step, []);
            columns.get(step).push(nodeId);
        });

    const orderedColumns = [...columns.entries()];
    const maxNodeHeight = Math.max(...nodeIds.map(id => g.node(id).height), 40);
    const rowPitch = maxNodeHeight + rowGap;
    const columnWidths = orderedColumns.map(([, ids]) =>
        Math.max(...ids.map(id => g.node(id).width), NODE_LABEL_MIN_WIDTH)
    );
    const columnX = [];
    let cursorX = padding;
    orderedColumns.forEach(([, ids], index) => {
        columnX[index] = cursorX;
        cursorX += columnWidths[index] + columnGap;
        ids.forEach((nodeId, row) => {
            const node = g.node(nodeId);
            node.x = columnX[index] + columnWidths[index] / 2;
            node.y = padding + node.height / 2 + row * rowPitch;
        });
    });

    const columnIndex = new Map();
    orderedColumns.forEach(([, ids], index) => {
        ids.forEach(nodeId => columnIndex.set(nodeId, index));
    });

    const maxRows = Math.max(...orderedColumns.map(([, ids]) => ids.length), 1);
    const graphWidth = Math.max(cursorX - columnGap + padding, 1);
    const graphHeight = Math.max(
        padding * 2 + maxRows * rowPitch,
        maxNodeHeight + padding * 2,
    );
    const backwardRouteY = graphHeight - padding / 2;

    g.edges().forEach(edgeId => {
        const edge = g.edge(edgeId);
        const source = g.node(edgeId.v);
        const target = g.node(edgeId.w);
        const sourceRight = source.x + source.width / 2;
        const targetLeft = target.x - target.width / 2;
        const targetRight = target.x + target.width / 2;
        const sourceColumn = columnIndex.get(edgeId.v);
        const targetColumn = columnIndex.get(edgeId.w);

        if (targetColumn > sourceColumn) {
            const midX = (sourceRight + targetLeft) / 2;
            edge.points = [
                { x: sourceRight, y: source.y },
                { x: midX, y: source.y },
                { x: midX, y: target.y },
                { x: targetLeft, y: target.y },
            ];
        } else if (edgeId.v === edgeId.w) {
            const loopX = sourceRight + 42;
            edge.points = [
                { x: sourceRight, y: source.y },
                { x: loopX, y: source.y - 34 },
                { x: loopX, y: source.y + 34 },
                { x: sourceRight, y: source.y },
            ];
        } else {
            // Back edges and same-column edges route around the right/bottom
            // side instead of crossing through a node.
            const routeX = Math.max(sourceRight, targetRight) + 42;
            edge.points = [
                { x: sourceRight, y: source.y },
                { x: routeX, y: source.y },
                { x: routeX, y: backwardRouteY },
                { x: targetRight, y: backwardRouteY },
                { x: targetRight, y: target.y },
            ];
        }
    });

    g.setGraph({
        rankdir: 'LR',
        width: graphWidth,
        height: graphHeight,
    });
    return { graphWidth, graphHeight };
}

// ==================== Layout and Coordinate Normalization ====================
function layoutGraph() {
    // Collapse consecutive same-endpoint execution edges before layout.
    const displayEdges = displayEdgesWithRepeatedFilter(activeEdges());

    // Large cyclic multigraphs are the expensive case for Dagre. They use a
    // compact graph adapter and deterministic placement without waiting for
    // the optional Dagre dependency.
    if (isLargeGraph(activeNodes().length, Math.max(displayEdges.length, activeEdges().length))) {
        const g = createLightweightGraph(displayEdges);
        return { g, ...layoutLargeGraph(g) };
    }

    if (typeof dagre === 'undefined' || !dagre.graphlib || !dagre.layout) {
        throw new Error('Dagre failed to load for this graph.');
    }

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
    
    // Keep each label inside its node so every graph has a consistent visual grammar.
    activeNodes().forEach(node => {
        const label = node.label || node.id;
        node.displayLabel = label;  // Store for rendering
        
        const lines = label.split('\\n');
        const { width, height } = containedNodeSize(lines);
        g.setNode(node.id, { width, height, ...node });
    });
    
    displayEdges.forEach((edge, idx) => {
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
    if (useViewportNavigation) {
        // Keep the rendered surface at viewport size. Moving an enormous SVG
        // element forces the browser to repaint its full bounds on every drag.
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.classList.add('large-graph-svg');
        svg.style.overflow = 'hidden';
    } else {
        svg.setAttribute('width', graphWidth);
        svg.setAttribute('height', graphHeight);
        svg.style.overflow = 'visible';
    }
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
 * Map a thought_length to an arrowhead width.
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
    if (edge.is_context) {
        return {
            strokeWidth:    1.2,
            strokeDasharray: '5,4',
            stroke:          '#95a5a6',
            markerEnd:       'url(#arrowhead-exec)',
            opacity:         0.75,
        };
    }

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
    const displayEdges = displayEdgesWithRepeatedFilter(activeEdges());

    // Pre-compute edge counts per (from,to) pair for multi-edge offsetting
    const edgesByPair = {};
    displayEdges.forEach((edge, idx) => {
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
    displayEdges.forEach(edge => {
        if (edge.type === 'exec' && !edge.is_multi_node_step && !edge.is_thought_continuation) {
            const tlen = getThoughtLength(edge);
            if (tlen > 0) thoughtWidthsSeen.add(Math.round(thoughtToWidth(tlen)));
        }
    });
    thoughtWidthsSeen.forEach(w => makeArrowMarker(`arrowhead-exec-w${w}`, w, '#7f8c8d'));

    const pairOffsets = Object.create(null);
    const edgeFragment = document.createDocumentFragment();
    g.edges().forEach(e => {
        const edge        = g.edge(e);
        const edgeKey     = `${e.v}-${e.w}`;
        const edgesInPair = edgesByPair[edgeKey] || [];
        const edgeIndex   = pairOffsets[edgeKey] || 0;
        pairOffsets[edgeKey] = edgeIndex + 1;
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
            !edge.is_context &&
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

        edgeFragment.appendChild(edgeGroup);
    });
    svg.appendChild(edgeFragment);
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

function addRepeatedCommandEars(nodeGroup, node) {
    const earData = Array.isArray(node.repeatedHatData)
        ? node.repeatedHatData.slice(0, 3)
        : [];
    if (!earData.length) return;

    const topY = node.y - node.height / 2;
    const pairSpacing = 25;
    const earWidth = 9;
    const earHeight = 11;

    earData.forEach((item, index) => {
        const centerX = node.x + (index - (earData.length - 1) / 2) * pairSpacing;
        const leftEar = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        leftEar.setAttribute('d',
            `M ${centerX - earWidth} ${topY + 1} `
            + `L ${centerX - earWidth + 3} ${topY - earHeight} `
            + `L ${centerX - 1} ${topY + 1} Z`,
        );
        leftEar.setAttribute('fill', item.color);
        leftEar.setAttribute('stroke', '#2c3e50');
        leftEar.setAttribute('stroke-width', '1.2');

        const rightEar = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        rightEar.setAttribute('d',
            `M ${centerX + 1} ${topY + 1} `
            + `L ${centerX + earWidth - 3} ${topY - earHeight} `
            + `L ${centerX + earWidth} ${topY + 1} Z`,
        );
        rightEar.setAttribute('fill', item.color);
        rightEar.setAttribute('stroke', '#2c3e50');
        rightEar.setAttribute('stroke-width', '1.2');

        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `Collapsed repeated ${item.label} commands`;
        leftEar.appendChild(title);
        rightEar.appendChild(title.cloneNode(true));
        nodeGroup.appendChild(leftEar);
        nodeGroup.appendChild(rightEar);
    });
}

function boundaryBadgeForNode(nodeId) {
    const visibleNodes = activeNodes().filter(node => !node.is_context);
    const firstId = visibleNodes[0]?.id;
    const visibleIds = new Set(visibleNodes.map(node => node.id));
    let latestExecution = null;

    // A deduplicated node can represent an early action and the final action
    // in the same trajectory. Follow the latest raw execution edge instead of
    // using the node's first-occurrence order for the LAST marker.
    activeEdges().forEach((edge, index) => {
        if (edge.type !== 'exec' || edge.is_context || edge.is_thought_continuation) return;
        if (!visibleIds.has(edge.to)) return;
        const step = parseEdgeStep(edge.label);
        if (step === null) return;
        if (!latestExecution
            || step > latestExecution.step
            || (step === latestExecution.step && index > latestExecution.index)) {
            latestExecution = { nodeId: edge.to, step, index };
        }
    });

    const fallbackLast = visibleNodes.reduce((best, node, index) => {
        const candidate = {
            id: node.id,
            step: nodeLastStep(node, index),
            index,
        };
        return !best
            || candidate.step > best.step
            || (candidate.step === best.step && candidate.index > best.index)
            ? candidate
            : best;
    }, null)?.id;
    const lastId = latestExecution?.nodeId || fallbackLast;
    const isFirst = nodeId === firstId;
    const isLast = nodeId === lastId;
    if (isFirst && isLast) return 'FIRST / LAST';
    if (isFirst) return 'FIRST';
    if (isLast) return 'LAST';
    return '';
}

function addBoundaryBadge(nodeGroup, node, label) {
    if (!label) return;

    const isLastOnly = label === 'LAST';
    const width = label.length * 7 + 24;
    const height = 22;
    const x = node.x - width / 2;
    const y = isLastOnly
        ? node.y + node.height / 2 + 9
        : node.y - node.height / 2 - height - 9;

    const badge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    badge.setAttribute('class', 'boundary-badge');
    badge.setAttribute('aria-label', label);

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    rect.setAttribute('width', width);
    rect.setAttribute('height', height);
    rect.setAttribute('rx', '7');
    rect.setAttribute('fill', isLastOnly ? '#e5f3ee' : '#fff3cb');
    rect.setAttribute('stroke', isLastOnly ? '#0a817b' : '#c89327');
    rect.setAttribute('stroke-width', '2');

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', node.x);
    text.setAttribute('y', y + height / 2 + 0.5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('font-size', '10');
    text.setAttribute('font-weight', '800');
    text.setAttribute('fill', isLastOnly ? '#075e5a' : '#84631e');
    text.textContent = label;

    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${label} node in the displayed graph view`;
    badge.append(rect, text, title);
    nodeGroup.appendChild(badge);
}

function renderNodes(svg, g, defs) {
    const nodeFragment = document.createDocumentFragment();
    g.nodes().forEach(nodeId => {
        const node      = g.node(nodeId);
        const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodeGroup.setAttribute('class',        'node');
        nodeGroup.setAttribute('data-id',      nodeId);
        if (node.is_context) nodeGroup.classList.add('context-node');
        // Keep the tooltip off the DOM attribute: large trajectories can
        // otherwise duplicate a substantial string for every rendered node.
        nodeGroup._tooltipContent = node.tooltip;

        // Left-click opens the detail sidebar for this node
        nodeGroup.addEventListener('click', (e) => {
            e.stopPropagation();
            openSidebar(node);
            if (shouldAutoOpenFileFootprint()) revealFileFootprintForNode(node.id);
        });

        let nodeFillAttr = node.color || '#CFE0F6';
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
            nodeFillAttr = `url(#${gradId})`;
        }
        const boundaryLabel = boundaryBadgeForNode(nodeId);
        const nodeRect = makeNodeRect(node, nodeFillAttr);
        if (boundaryLabel) {
            if (boundaryLabel === 'FIRST' || boundaryLabel === 'FIRST / LAST') {
                nodeGroup.classList.add('trajectory-first');
            }
            if (boundaryLabel === 'LAST' || boundaryLabel === 'FIRST / LAST') {
                nodeGroup.classList.add('trajectory-last');
            }
            nodeRect.classList.add('boundary-node');
        }
        nodeGroup.appendChild(nodeRect);
        
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

        addRepeatedCommandEars(nodeGroup, node);
        addBoundaryBadge(nodeGroup, node, boundaryLabel);

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
        
        nodeFragment.appendChild(nodeGroup);
    });
    svg.appendChild(nodeFragment);
}

// ==================== Tooltip ====================
function setupTooltips() {
    const tooltip = document.getElementById('tooltip');
    document.querySelectorAll('.node').forEach(node => {
        node.addEventListener('mouseenter', (e) => {
            const tooltipContent = e.currentTarget._tooltipContent
                || e.currentTarget.getAttribute('data-tooltip');
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
let sidebarCustomWidth = null;  // remembers user-dragged width across open/close cycles

/**
 * Open (or refresh) the sidebar for the given node data object.
 * Called from the click handler set up in renderNodes.
 */
function openSidebar(node) {
    const sidebar  = document.getElementById('detailSidebar');
    // Restore custom width if the user previously dragged the resizer
    sidebar.style.width = sidebarCustomWidth ? sidebarCustomWidth + 'px' : '';
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
    sidebar.style.width = '';   // clear inline width so CSS transition to 0 takes effect
    sidebar.classList.remove('open');
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

function toggleTrajectoryName(forceExpanded) {
    const identity = document.querySelector('.trajectory-identity');
    const heading = document.getElementById('trajectoryName');
    const button = document.getElementById('trajectoryNameToggle');
    if (!identity || !heading || !button) return;

    const expanded = typeof forceExpanded === 'boolean'
        ? forceExpanded
        : !identity.classList.contains('expanded');
    identity.classList.toggle('expanded', expanded);
    const hud = identity.closest('.graph-hud');
    if (hud) hud.classList.toggle('trajectory-title-expanded', expanded);
    button.textContent = expanded ? '-' : '+';
    button.setAttribute('aria-expanded', String(expanded));
    button.setAttribute(
        'aria-label',
        expanded ? 'Collapse full trajectory name' : 'Expand full trajectory name',
    );
    button.title = expanded ? 'Collapse full trajectory name' : 'Expand full trajectory name';
}

function prepareTrajectoryName() {
    const identity = document.querySelector('.trajectory-identity');
    const heading = document.getElementById('trajectoryName');
    if (!identity || !heading) return;

    const fullName = heading.dataset.fullName || heading.textContent || '';
    const characters = Array.from(fullName);
    const maxCharacters = 1000;
    const displayName = characters.length > maxCharacters
        ? `${characters.slice(0, maxCharacters).join('')}…`
        : fullName;

    heading.dataset.fullName = fullName;
    heading.textContent = displayName;
    heading.title = fullName;
    identity.classList.toggle('long', characters.length > 80);
    identity.classList.toggle('very-long', characters.length > 180);
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
let useViewportNavigation = false;
let transformFrame = null;
let navigationEndTimer = null;
let graphResizeObserver = null;

function applyTransform() {
    if (!svg || !svg.isConnected) return;
    if (useViewportNavigation) {
        const width = graphEl.clientWidth || graphEl.offsetWidth;
        const height = graphEl.clientHeight || graphEl.offsetHeight;
        if (width <= 0 || height <= 0) return;
        const safeScale = Math.max(currentScale, 0.0001);
        svg.setAttribute('viewBox', [
            -currentX / safeScale,
            -currentY / safeScale,
            width / safeScale,
            height / safeScale,
        ].map(value => value.toFixed(3)).join(' '));
        return;
    }

    svg.style.transform = `translate3d(${currentX}px, ${currentY}px, 0) scale(${currentScale})`;
    svg.style.transformOrigin = '0 0';
}

function updateTransform(immediate = false) {
    if (immediate) {
        if (transformFrame !== null) cancelAnimationFrame(transformFrame);
        transformFrame = null;
        applyTransform();
        return;
    }
    if (transformFrame !== null) return;
    transformFrame = requestAnimationFrame(() => {
        transformFrame = null;
        applyTransform();
    });
}

function setNavigationActive(active) {
    if (!graphEl) return;
    graphEl.classList.toggle('is-navigating', active);
}

function scheduleNavigationEnd() {
    if (navigationEndTimer !== null) clearTimeout(navigationEndTimer);
    navigationEndTimer = window.setTimeout(() => {
        navigationEndTimer = null;
        setNavigationActive(false);
    }, 120);
}

function setupViewportResize() {
    if (!useViewportNavigation || typeof ResizeObserver === 'undefined') return;
    if (graphResizeObserver) graphResizeObserver.disconnect();
    graphResizeObserver = new ResizeObserver(() => updateTransform());
    graphResizeObserver.observe(graphEl);
}

function fitToScreen() {
    if (!graphEl || !svg || !graphWidth || !graphHeight) return;
    // Use the #graph div's own dimensions — these are correct both in normal
    // layout (CSS height: 900px) and when fullscreen-active forces it to fill
    // the viewport via position:fixed.
    const w = graphEl.clientWidth  || graphEl.offsetWidth;
    const h = graphEl.clientHeight || graphEl.offsetHeight;
    if (w <= 0 || h <= 0) return;

    const scaleX = w / graphWidth;
    const scaleY = h / graphHeight;
    const minimumScale = useViewportNavigation ? 0.005 : 0.05;
    currentScale = Math.max(minimumScale, Math.min(scaleX, scaleY, 1) * 0.95);

    const scaledWidth  = graphWidth  * currentScale;
    const scaledHeight = graphHeight * currentScale;
    currentX = (w - scaledWidth)  / 2;
    currentY = (h - scaledHeight) / 2;

    updateTransform(true);
}

function zoomIn() {
    if (!graphEl || !svg) return;
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
    if (!graphEl || !svg) return;
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
        setNavigationActive(true);
        scheduleNavigationEnd();
        
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
        if (e.target.closest('.file-footprint')) return;

        isDragging = true;
        dragMoved  = false;
        startX = e.clientX - currentX;
        startY = e.clientY - currentY;
        graphEl.style.cursor = 'grabbing';
        setNavigationActive(true);
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
        setNavigationActive(false);
    });

    graphEl.addEventListener('mouseleave', () => {
        isDragging = false;
        dragMoved  = false;
        graphEl.style.cursor = 'grab';
        setNavigationActive(false);
    });
}

function togglePhaseLegend(forceOpen) {
    const legend = document.getElementById('phaseLegend');
    if (!legend) return;
    const shouldOpen = typeof forceOpen === 'boolean'
        ? forceOpen
        : !legend.classList.contains('open');
    legend.classList.toggle('open', shouldOpen);
    legend.setAttribute('aria-hidden', String(!shouldOpen));
}

function openParentViewOptions() {
    try {
        if (window.parent !== window && typeof window.parent.openViewOptions === 'function') {
            window.parent.openViewOptions();
        }
    } catch (_) {
        // A standalone exported graph has no browser-shell settings to open.
    }
}

function shouldAutoOpenFileFootprint() {
    try {
        if (window.parent !== window && typeof window.parent.shouldAutoOpenFileFootprint === 'function') {
            return window.parent.shouldAutoOpenFileFootprint();
        }
    } catch (_) {
        // Standalone exports retain the default node-click behavior.
    }
    return true;
}

// ==================== File Footprint ====================
// The backend provides a compact activity index keyed by parsed file/path
// arguments. This panel turns that index into a per-file step timeline.
let activeFileFootprintPath = null;
const collapsedFileFootprintDirs = new Set();

function activityName(type) {
    return ({ seen: 'referenced', view: 'viewed', edit: 'edited' })[type] || 'referenced';
}

function clearFileFootprintHighlight() {
    document.querySelectorAll('.node.file-footprint-highlight, .node.file-footprint-muted')
        .forEach(node => node.classList.remove('file-footprint-highlight', 'file-footprint-muted'));
    activeFileFootprintPath = null;
}

function highlightFileFootprintNodes(nodeIds) {
    const activeIds = new Set(nodeIds || []);
    document.querySelectorAll('.node').forEach(node => {
        const isMatch = activeIds.has(node.getAttribute('data-id'));
        node.classList.toggle('file-footprint-highlight', isMatch);
        node.classList.toggle('file-footprint-muted', activeIds.size > 0 && !isMatch);
    });
}

function selectFileFootprintRow(activity, row, allowToggle = true) {
    const wasSelected = activeFileFootprintPath === activity.path;
    document.querySelectorAll('.file-footprint-row.selected')
        .forEach(item => item.classList.remove('selected'));

    if (wasSelected && allowToggle) {
        clearFileFootprintHighlight();
        return;
    }

    activeFileFootprintPath = activity.path;
    row.classList.add('selected');
    highlightFileFootprintNodes(activity.node_ids);
}

function revealFileFootprintForNode(nodeId) {
    const matches = (fileActivityData || [])
        .filter(activity => (activity.node_ids || []).includes(nodeId))
        .sort((left, right) => {
            const leftScore = left.edit_count * 4 + left.view_count * 2 + left.seen_count;
            const rightScore = right.edit_count * 4 + right.view_count * 2 + right.seen_count;
            return rightScore - leftScore || left.path.localeCompare(right.path);
        });
    const activity = matches[0];
    if (!activity) return;

    const filter = document.getElementById('fileFootprintFilter');
    if (filter) filter.value = '';
    // Reopen each parent so a previously collapsed branch never hides the match.
    let directoryKey = '';
    footprintPathParts(activity.path).forEach(part => {
        directoryKey = directoryKey ? `${directoryKey}/${part}` : part;
        collapsedFileFootprintDirs.delete(directoryKey);
    });
    openFileFootprint();
    renderFileFootprint();

    const row = [...document.querySelectorAll('.file-footprint-row')]
        .find(item => item.dataset.footprintPath === activity.path);
    if (!row) return;

    selectFileFootprintRow(activity, row, false);
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function makeFootprintMark(event, maxStep) {
    const mark = document.createElement('span');
    const position = maxStep > 0 ? Math.max(1, Math.min(99, (event.step / maxStep) * 100)) : 50;
    mark.className = `file-footprint-mark ${event.type}`;
    mark.style.left = `${position}%`;
    mark.title = `Step ${event.step}: ${activityName(event.type)}`;
    mark.setAttribute('aria-label', mark.title);
    return mark;
}

function footprintPathParts(path) {
    const normalized = String(path || '').replace(/\\/g, '/').replace(/^\.\//, '');
    const parts = normalized.split('/').filter(Boolean);
    return parts.length ? parts : [normalized || 'unknown path'];
}

function buildFootprintTree(activities) {
    const root = { key: '', dirs: new Map(), files: [], directory_activities: [] };

    activities.forEach(activity => {
        const parts = footprintPathParts(activity.path);
        const isDirectoryActivity = activity.kind === 'path' && parts.length > 0;
        const leaf = parts.pop();
        let parent = root;

        parts.forEach(part => {
            const key = parent.key ? `${parent.key}/${part}` : part;
            if (!parent.dirs.has(part)) {
                parent.dirs.set(part, {
                    key,
                    name: part,
                    dirs: new Map(),
                    files: [],
                    directory_activities: [],
                });
            }
            parent = parent.dirs.get(part);
        });

        if (isDirectoryActivity) {
            const key = parent.key ? `${parent.key}/${leaf}` : leaf;
            if (!parent.dirs.has(leaf)) {
                parent.dirs.set(leaf, {
                    key,
                    name: leaf,
                    dirs: new Map(),
                    files: [],
                    directory_activities: [],
                });
            }
            parent.dirs.get(leaf).directory_activities.push(activity);
        } else {
            parent.files.push({ ...activity, leaf_name: leaf });
        }
    });

    return root;
}

function footprintTreeStats(node) {
    const stats = node.files.reduce((total, activity) => ({
        files: total.files + 1,
        views: total.views + activity.view_count,
        edits: total.edits + activity.edit_count,
    }), { files: 0, views: 0, edits: 0 });

    node.dirs.forEach(child => {
        const childStats = footprintTreeStats(child);
        stats.files += childStats.files;
        stats.views += childStats.views;
        stats.edits += childStats.edits;
    });
    node.stats = stats;
    return stats;
}

function isFootprintDirectoryOpen(directory, forceOpen) {
    return forceOpen || !collapsedFileFootprintDirs.has(directory.key);
}

function toggleFootprintDirectory(directory, forceOpen) {
    if (forceOpen) return;
    if (isFootprintDirectoryOpen(directory, false)) {
        collapsedFileFootprintDirs.add(directory.key);
    } else {
        collapsedFileFootprintDirs.delete(directory.key);
    }
    renderFileFootprint(document.getElementById('fileFootprintFilter')?.value || '');
}

function makeFootprintFileRow(activity, depth, maxStep) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'file-footprint-row';
    if (activity.kind === 'plan') row.classList.add('plan-activity');
    row.dataset.footprintPath = activity.path;
    row.style.setProperty('--file-indent', `${depth * 16}px`);
    row.title = `${activity.path}\nFirst seen: step ${activity.first_step}; last seen: step ${activity.last_step}`;
    if (activeFileFootprintPath === activity.path) row.classList.add('selected');

    const pathLine = document.createElement('div');
    pathLine.className = 'file-footprint-path';
    const pathLabel = document.createElement('span');
    pathLabel.textContent = activity.leaf_name;
    pathLabel.title = activity.path;
    const kind = document.createElement('span');
    kind.className = 'file-footprint-kind';
    kind.textContent = activity.kind;
    pathLine.append(pathLabel, kind);

    const timeline = document.createElement('div');
    timeline.className = 'file-footprint-timeline';
    const track = document.createElement('span');
    track.className = 'file-footprint-track';
    timeline.appendChild(track);
    activity.events.forEach(event => timeline.appendChild(makeFootprintMark(event, maxStep)));

    row.append(pathLine, timeline);
    row.addEventListener('click', () => selectFileFootprintRow(activity, row));
    return row;
}

function makeFootprintDirectoryActivityRow(activity, depth, maxStep) {
    const row = makeFootprintFileRow({ ...activity, leaf_name: '(folder activity)' }, depth, maxStep);
    row.classList.add('directory-activity');
    return row;
}

function renderFootprintTree(directory, container, depth, maxStep, forceOpen) {
    directory.directory_activities
        .sort((left, right) => left.path.localeCompare(right.path))
        .forEach(activity => container.appendChild(makeFootprintDirectoryActivityRow(activity, depth, maxStep)));

    [...directory.dirs.values()]
        .sort((left, right) => left.name.localeCompare(right.name))
        .forEach(child => {
            const isOpen = isFootprintDirectoryOpen(child, forceOpen);
            const folder = document.createElement('button');
            folder.type = 'button';
            folder.className = 'file-footprint-folder';
            folder.style.setProperty('--file-indent', `${depth * 16}px`);
            folder.setAttribute('aria-expanded', String(isOpen));
            folder.title = `${child.stats.files} tracked ${child.stats.files === 1 ? 'path' : 'paths'} in ${child.key}`;

            const chevron = document.createElement('span');
            chevron.className = 'file-footprint-chevron';
            chevron.setAttribute('aria-hidden', 'true');
            chevron.textContent = '>';
            const name = document.createElement('span');
            name.className = 'file-footprint-folder-name';
            name.textContent = child.name;
            const count = document.createElement('span');
            count.className = 'file-footprint-folder-count';
            count.textContent = `${child.stats.files} ${child.stats.files === 1 ? 'file' : 'files'}`;
            folder.append(chevron, name, count);
            folder.addEventListener('click', () => toggleFootprintDirectory(child, forceOpen));
            container.appendChild(folder);

            if (isOpen) {
                const children = document.createElement('div');
                children.className = 'file-footprint-children';
                renderFootprintTree(child, children, depth + 1, maxStep, forceOpen);
                container.appendChild(children);
            }
        });

    directory.files
        .sort((left, right) => left.leaf_name.localeCompare(right.leaf_name))
        .forEach(activity => container.appendChild(makeFootprintFileRow(activity, depth, maxStep)));
}

function renderFileFootprint(query = '') {
    const rows = document.getElementById('fileFootprintRows');
    const summary = document.getElementById('fileFootprintSummary');
    if (!rows || !summary) return;

    const normalizedQuery = query.trim().toLowerCase();
    const activities = (fileActivityData || []).filter(activity =>
        !normalizedQuery || activity.path.toLowerCase().includes(normalizedQuery)
    );
    const totalEdits = activities.reduce((total, activity) => total + activity.edit_count, 0);
    const totalViews = activities.reduce((total, activity) => total + activity.view_count, 0);
    summary.textContent = `${activities.length} tracked ${activities.length === 1 ? 'item' : 'items'} | ${totalViews} views | ${totalEdits} edits`;
    rows.replaceChildren();

    if (!activities.length) {
        const empty = document.createElement('div');
        empty.className = 'file-footprint-empty';
        empty.textContent = normalizedQuery
            ? 'No parsed files, paths, or plans match this filter.'
            : 'No files, paths, or plans were detected in this trajectory.';
        rows.appendChild(empty);
        return;
    }

    const tree = buildFootprintTree(activities);
    footprintTreeStats(tree);
    const maxStep = Math.max(1, ...activities.map(activity => activity.last_step));
    renderFootprintTree(tree, rows, 0, maxStep, Boolean(normalizedQuery));
}

function toggleFileFootprint() {
    const panel = document.getElementById('fileFootprint');
    if (!panel) return;
    panel.classList.contains('open') ? closeFileFootprint() : openFileFootprint();
}

function openFileFootprint() {
    const panel = document.getElementById('fileFootprint');
    const toggle = document.getElementById('fileFootprintToggle');
    if (!panel) return;
    panel.classList.add('open');
    if (toggle) toggle.classList.add('active');
}

function closeFileFootprint() {
    const panel = document.getElementById('fileFootprint');
    const toggle = document.getElementById('fileFootprintToggle');
    if (panel) panel.classList.remove('open');
    if (toggle) toggle.classList.remove('active');
    clearFileFootprintHighlight();
    renderFileFootprint(document.getElementById('fileFootprintFilter')?.value || '');
}

function setupFileFootprint() {
    const filter = document.getElementById('fileFootprintFilter');
    if (filter) {
        filter.addEventListener('input', () => {
            clearFileFootprintHighlight();
            renderFileFootprint(filter.value);
        });
    }
    renderFileFootprint();
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
        // Persist the dragged width so it survives close/reopen
        sidebarCustomWidth = sidebar.offsetWidth;
    });
}

// ==================== Initialization ====================

function renderActiveSubgraph() {
    if (!graphEl) return;

    const { g, graphWidth: gw, graphHeight: gh } = layoutGraph();
    graphWidth  = gw;
    graphHeight = gh;
    useViewportNavigation = isLargeGraph(activeNodes().length, activeEdges().length);

    const nextSvg = createSVG(graphWidth, graphHeight);
    const defs = createMarkers(nextSvg);
    renderEdges(nextSvg, g, defs);
    renderNodes(nextSvg, g, defs);

    if (svg && svg !== nextSvg) svg.remove();
    svg = nextSvg;
    graphEl.replaceChildren(svg);
    updateSubgraphPicker();
    setupTooltips();
    setTimeout(fitToScreen, 150);
}

function initializeGraph() {
    prepareSubgraphs();

    // Small and medium graphs require Dagre. Large graphs use the deterministic
    // local layout and can render even when the CDN dependency is unavailable.
    if (!isLargeGraph(activeNodes().length, activeEdges().length)
        && (typeof dagre === 'undefined' || !dagre.graphlib || !dagre.layout)) {
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
        prepareTrajectoryName();
        graphEl = document.getElementById('graph');
        renderActiveSubgraph();
        setupWheelZoom();
        setupPanning();
        setupViewportResize();
        setupSidebarResize();
        setupFileFootprint();

        setTimeout(fitToScreen, 150);
    } catch (err) {
        console.error('[graph] initializeGraph failed:', err);
        throw err;  // Propagate to the window error handler for UI display.
    }
}

// Dagre loads asynchronously so a slow CDN cannot block the large-graph path.
// Small graphs wait for it, while large graphs initialize immediately.
function _tryInit(retriesLeft) {
    prepareSubgraphs();
    const largeGraph = isLargeGraph(activeNodes().length, activeEdges().length);
    if (largeGraph
        || (typeof dagre !== 'undefined' && dagre.graphlib && dagre.layout)) {
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

_tryInit(3000);  // Up to five minutes before surfacing the dependency error.

function _wireSidebarClose() {
    const btn = document.getElementById('sidebarCloseBtn');
    if (!btn) return;
    // Belt-and-suspenders: both click and mouseup so the event fires even if
    // something upstream cancelled the synthetic click (e.g. drag-end logic).
    btn.addEventListener('click',   closeSidebar);
    btn.addEventListener('mouseup', (e) => { e.stopPropagation(); closeSidebar(); });
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') togglePhaseLegend(false);
    });
}
