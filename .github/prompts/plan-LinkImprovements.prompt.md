# Link Processing Architectural Improvements

**Document Version**: 2.0  
**Date**: November 23, 2025  
**Status**: Architectural Proposal for Implementation

---

## Summary

The Notion Recursive Scraper's link processing pipeline requires two architectural improvements:

1. **Back Edge Handling**: Capture and propagate edge classification information from the discovery phase to the execution phase, enabling proper handling of cyclic page references (e.g., parent pages referenced by children).

2. **Internal Page References**: Preserve and rewrite Notion block-level anchors (`data-block-id`) to support section-level navigation in offline mode.

Both improvements require lightweight component additions to the existing pipeline with no changes to core discovery or conflict resolution logic.

---

## Problem 1: Back Edge Handling

### Analysis

The BFS discovery process naturally encounters back edges—links pointing to already-visited pages in the hierarchy. Currently:

- `GlobalQueueManager`: Detects back edges via `visitedUrls` set
- `ConflictResolver`: Groups duplicate URL references into single canonical pages
- `LinkRewriter`: **Has no knowledge** of edge directionality

When a child page links back to its parent (e.g., "Back to Home"), the system can rewrite the link correctly by path, but loses the semantic information that this is a back edge.

### Example

```
JBC090_Language_AI (root)
  └─ Syllabus (child)
      └─ JBC090_Language_AI (back-edge to parent)
```

In the HTML, Syllabus contains a link to JBC090_Language_AI. The current system:
1. Detects both pages during discovery ✓
2. Merges duplicates in conflict resolution ✓
3. Rewrites the link as `../../index.html` ✓
4. **Never captures that this is a back edge** ✗

### Improvement

Add a lightweight **edge classification** step that:
1. Identifies edge types during discovery (FORWARD: child→parent, BACK: parent→child, CROSS: siblings)
2. Stores edge metadata in a `PageGraph` structure
3. Passes this metadata to `LinkRewriter` for context-aware processing

---

## Problem 2: Internal Page References

### Analysis

Notion pages support section-level links via anchors that reference specific blocks:

```
URL: /page-name-29d979eeca9f8102a85be4dd9007f020?pvs=25#29d979eeca9f81f7b82fe4b983834212
      └─ Page ID (raw)                                    └─ Block ID (raw anchor)
      
HTML: <div data-block-id="29d979ee-ca9f-81f7-b82f-e4b983834212">Section</div>
           └─ Block ID (formatted with dashes)
```

Currently:
- `LinkExtractor`: Discards hash fragments entirely
- `LinkRewriter`: Cannot map raw block IDs (URL format) to formatted IDs (HTML format)

Result: Section-level links break offline because the anchor doesn't match any `data-block-id` in the saved HTML.

### Example

Original link: `https://notion.so/page#29d979eeca9f81f7b82fe4b983834212`
Saved HTML has: `<div data-block-id="29d979ee-ca9f-81f7-b82f-e4b983834212">`
Offline link rewrite: `page/index.html#29d979eeca9f81f7b82fe4b983834212` (broken—ID format mismatch)

### Improvement

Add block ID mapping that:
1. Extracts `data-block-id` values during page download
2. Maintains raw ↔ formatted ID mappings (stored per page)
3. Rewrites anchors with correct formatted IDs during link rewriting

---

## Proposed Improvements

### 1. Edge Classification

**Component: `EdgeClassifier`**

```javascript
/**
 * Classifies edges in the discovered page graph based on depth relationships.
 * BFS guarantees all pages visited, so edge type is deterministic.
 * 
 * Edge types:
 * - FORWARD: source.depth < target.depth (deeper in hierarchy)
 * - BACK: source.depth >= target.depth (returning or crossing)
 */
class EdgeClassifier {
  /**
   * Classify an edge between two discovered pages
   * @param {PageContext} sourceContext - Source page (where link originates)
   * @param {PageContext} targetContext - Target page (where link points)
   * @returns {Object} Classification object
   * @returns {string} .type - 'FORWARD' or 'BACK'
   * @returns {number} .depthDelta - Absolute difference in depths
   * @returns {boolean} .isAncestor - True if target is ancestor of source
   */
  classifyEdge(sourceContext, targetContext) {
    const depthDelta = Math.abs(sourceContext.depth - targetContext.depth);
    
    if (targetContext.depth > sourceContext.depth) {
      return { type: 'FORWARD', depthDelta, isAncestor: false };
    }
    
    const isAncestor = this._isAncestor(sourceContext, targetContext);
    return { type: 'BACK', depthDelta, isAncestor };
  }
  
  /**
   * Check if target is an ancestor of source in the page hierarchy
   * @private
   */
  _isAncestor(sourceContext, targetContext) {
    let current = sourceContext;
    while (current.parentId) {
      if (current.parentId === targetContext.id) return true;
      current = this.contextMap.get(current.parentId);
      if (!current) break;
    }
    return false;
  }
}
```

**Component: `PageGraph`**

```javascript
/**
 * Maintains the discovered page hierarchy with edge classification metadata.
 * Built during discovery phase, passed to execution phase.
 */
class PageGraph {
  constructor() {
    this.nodes = new Map();        // pageId → PageContext
    this.edges = new Map();        // pageId → Set of target pageIds
    this.edgeMetadata = new Map(); // "sourceId-targetId" → EdgeClassification
  }
  
  /**
   * Add a discovered link with edge classification
   * @param {string} sourceId - Source page ID
   * @param {string} targetId - Target page ID
   * @param {Object} classification - EdgeClassifier result
   */
  addEdge(sourceId, targetId, classification) {
    if (!this.edges.has(sourceId)) {
      this.edges.set(sourceId, new Set());
    }
    this.edges.get(sourceId).add(targetId);
    this.edgeMetadata.set(`${sourceId}-${targetId}`, classification);
  }
  
  /**
   * Get classification for an edge
   * @param {string} sourceId
   * @param {string} targetId
   * @returns {Object} EdgeClassification or null if not found
   */
  getEdgeClassification(sourceId, targetId) {
    return this.edgeMetadata.get(`${sourceId}-${targetId}`) || null;
  }
}
```

### 2. Block ID Mapping

**Component: `BlockIDExtractor`**

```javascript
/**
 * Extracts block IDs from downloaded page HTML.
 * Maps raw block IDs (URL format) to formatted IDs (HTML format).
 * 
 * Block ID format conversion:
 *   Raw:       29d979eeca9f81f7b82fe4b983834212 (32 hex chars)
 *   Formatted: 29d979ee-ca9f-81f7-b82f-e4b983834212 (with dashes at [8,13,18,23])
 */
class BlockIDExtractor {
  /**
   * Extract block ID mapping from saved HTML
   * @param {Document} document - Parsed HTML DOM
   * @returns {Map<string, string>} raw block ID → formatted block ID
   */
  extractBlockIDs(document) {
    const blockMap = new Map();
    
    const blocks = document.querySelectorAll('[data-block-id]');
    for (const block of blocks) {
      const formattedId = block.getAttribute('data-block-id');
      const rawId = this._formatToRaw(formattedId);
      blockMap.set(rawId, formattedId);
    }
    
    return blockMap;
  }
  
  /**
   * Convert UUID format to raw hex
   * @private
   */
  _formatToRaw(formattedId) {
    return formattedId.replace(/-/g, '').toLowerCase();
  }
  
  /**
   * Convert raw hex to UUID format
   * @private
   */
  _rawToFormatted(rawId) {
    const hex = rawId.toLowerCase();
    return [
      hex.substring(0, 8),
      hex.substring(8, 12),
      hex.substring(12, 16),
      hex.substring(16, 20),
      hex.substring(20, 32)
    ].join('-');
  }
}
```

**Component: `BlockIDMapper`**

```javascript
/**
 * Persists and retrieves block ID mappings for offline link rewriting.
 * Stores mappings in `.block-ids.json` alongside saved HTML.
 */
class BlockIDMapper {
  /**
   * Save block ID mapping to disk
   * @param {string} pageId - Page identifier
   * @param {string} saveDir - Directory where page HTML is saved
   * @param {Map<string, string>} blockMap - raw ID → formatted ID mapping
   */
  async saveBlockMap(pageId, saveDir, blockMap) {
    const mapFile = path.join(saveDir, '.block-ids.json');
    const mapObj = Object.fromEntries(blockMap);
    await fs.writeFile(mapFile, JSON.stringify(mapObj, null, 2));
  }
  
  /**
   * Load block ID mapping from disk
   * @param {string} saveDir - Directory where page HTML is saved
   * @returns {Map<string, string>} raw ID → formatted ID, or empty Map
   */
  async loadBlockMap(saveDir) {
    const mapFile = path.join(saveDir, '.block-ids.json');
    try {
      const content = await fs.readFile(mapFile, 'utf-8');
      return new Map(Object.entries(JSON.parse(content)));
    } catch {
      return new Map();
    }
  }
  
  /**
   * Get formatted ID for a raw block ID
   * @param {string} rawId - Raw block ID from URL
   * @param {Map} blockMap - Loaded block ID mapping
   * @returns {string} Formatted ID for use as HTML anchor
   */
  getFormattedId(rawId, blockMap) {
    return blockMap.get(rawId) || this._fallbackFormat(rawId);
  }
  
  /**
   * Fallback formatting if block not found in map
   * @private
   */
  _fallbackFormat(rawId) {
    const hex = rawId.toLowerCase();
    return [
      hex.substring(0, 8),
      hex.substring(8, 12),
      hex.substring(12, 16),
      hex.substring(16, 20),
      hex.substring(20, 32)
    ].join('-');
  }
}
```

### 3. Component Updates

**Enhanced `LinkExtractor`** — Preserve block IDs in extracted links

```javascript
/**
 * MODIFICATION: extractLinks() now returns links with optional blockId
 * 
 * Before:
 * {
 *   url: "https://notion.so/page",
 *   title: "Link text"
 * }
 * 
 * After:
 * {
 *   url: "https://notion.so/page",
 *   title: "Link text",
 *   blockId: "29d979eeca9f81f7b82fe4b983834212"  // NEW: if anchor present
 * }
 */

// In extractLinks() evaluation:
const parts = absoluteUrl.split('#');
const urlWithoutHash = parts[0].split('?')[0];
const blockIdRaw = parts.length > 1 ? parts[1] : null;

return {
  url: urlWithoutHash,
  title: linkTitle,
  blockId: blockIdRaw,  // NEW: preserve raw block ID from URL
  // ... other fields ...
};
```

**Enhanced `DownloadHandler`** — Extract and save block IDs

```javascript
/**
 * MODIFICATION: After page is scraped, extract and save block IDs
 * 
 * Added logic:
 * 1. Create BlockIDExtractor instance
 * 2. Extract block IDs from saved HTML
 * 3. Save mapping via BlockIDMapper
 */

async handle(payload) {
  // ... existing scraping logic ...
  
  // NEW: Extract block IDs after download
  const blockExtractor = new BlockIDExtractor();
  const blockMap = blockExtractor.extractBlockIDs(document);
  
  const blockMapper = new BlockIDMapper();
  const saveDir = path.dirname(payload.savePath);
  await blockMapper.saveBlockMap(payload.pageId, saveDir, blockMap);
  
  return {
    success: true,
    pageId: payload.pageId,
    // ... existing fields ...
    blockMapSaved: true
  };
}
```

**Enhanced `LinkRewriter`** — Use edge context and rewrite block anchors

```javascript
/**
 * MODIFICATION: Accept edge graph and block maps, use them for context-aware rewriting
 * 
 * Parameters:
 * - pageGraph: PageGraph with edge classifications
 * - blockMapCache: Pre-loaded block ID mappings for all pages
 */

async rewriteLinksInFile(pageContext, urlToContextMap, pageGraph, blockMapCache) {
  // ... existing DOM parsing ...
  
  for (const link of links) {
    const href = link.getAttribute('href');
    const { urlPart, blockIdRaw } = this._parseHref(href);
    
    const targetContext = urlToContextMap.get(urlPart);
    if (!targetContext) continue;
    
    // Build base path
    let newHref = pageContext === targetContext
      ? 'index.html'
      : pageContext.getRelativePathTo(targetContext);
    
    // NEW: Rewrite block anchor if present
    if (blockIdRaw && blockMapCache.has(targetContext.id)) {
      const blockMap = blockMapCache.get(targetContext.id);
      const mapper = new BlockIDMapper();
      const formattedId = mapper.getFormattedId(blockIdRaw, blockMap);
      newHref += '#' + formattedId;
    }
    
    link.setAttribute('href', newHref);
  }
  
  // ... save and return ...
}

/**
 * Parse href to separate URL and block ID
 * @private
 */
_parseHref(href) {
  const parts = href.split('#');
  const urlPart = parts[0].split('?')[0];
  const blockIdRaw = parts.length > 1 ? parts[1] : null;
  return { urlPart, blockIdRaw };
}
```

**Enhanced `GlobalQueueManager`** — Build PageGraph during discovery

```javascript
/**
 * MODIFICATION: Build PageGraph as pages are discovered
 * 
 * In completeDiscovery():
 * - After processing discovered links, classify edges
 * - Add to PageGraph
 */

completeDiscovery(pageId, discoveredLinks, metadata, resolvedTitle) {
  // ... existing logic ...
  
  const parentContext = this.allContexts.get(pageId);
  const edgeClassifier = new EdgeClassifier(this.allContexts);
  
  for (const link of discoveredLinks) {
    const childContext = // ... create context ...
    parentContext.addChild(childContext);
    
    // NEW: Classify edge and add to graph
    const classification = edgeClassifier.classifyEdge(
      parentContext,
      childContext
    );
    this.pageGraph.addEdge(pageId, childContext.id, classification);
  }
  
  return newContexts;
}
```

---

## Implementation Roadmap

### Phase 1: Component Creation

Create new components with clean interfaces:
- `EdgeClassifier`: Classify edges by depth relationships
- `PageGraph`: Store hierarchy with edge metadata
- `BlockIDExtractor`: Extract block IDs from HTML
- `BlockIDMapper`: Persist and retrieve block mappings

**Deliverables**:
- 4 new component files
- Unit tests for each component
- No changes to existing components

### Phase 2: Discovery Integration

Wire edge classification into discovery pipeline:
- `GlobalQueueManager` builds `PageGraph` as pages discovered
- `DiscoveryHandler` passes `PageGraph` to orchestrator
- Enhanced `LinkExtractor` captures block IDs in extracted links

**Deliverables**:
- `GlobalQueueManager` enhanced with PageGraph building
- `LinkExtractor` enhanced to preserve block IDs
- Integration tests validating discovery produces complete graph

### Phase 3: Execution Integration

Integrate edge context and block mapping into download and rewriting:
- `DownloadHandler` extracts and saves block ID maps
- `BlockIDMapper` loaded and cached before link rewriting
- `LinkRewriter` uses edge graph for context, block maps for anchors

**Deliverables**:
- `DownloadHandler` enhanced with block extraction
- `LinkRewriter` enhanced with edge awareness and anchor rewriting
- Pre-download phase to load all block maps into cache
- Integration tests validating offline links work

### Phase 4: Validation and Testing

End-to-end testing with real Notion data:
- Test pages with back edge references
- Test pages with section-level links
- Verify offline navigation works correctly
- Performance benchmarking

**Deliverables**:
- Comprehensive integration test suite
- Test results with example Notion sites
- Performance report

---

## Summary of Changes by File

| File | Change | Purpose |
|------|--------|---------|
| `src/orchestration/analysis/EdgeClassifier.js` | NEW | Classify edge directionality |
| `src/orchestration/PageGraph.js` | NEW | Store hierarchy with edge metadata |
| `src/extraction/BlockIDExtractor.js` | NEW | Extract block IDs from HTML |
| `src/processing/BlockIDMapper.js` | NEW | Manage block ID mappings |
| `src/extraction/LinkExtractor.js` | ENHANCED | Preserve block IDs in links |
| `src/worker/handlers/DownloadHandler.js` | ENHANCED | Extract and save block maps |
| `src/processing/LinkRewriter.js` | ENHANCED | Use edge context, rewrite anchors |
| `src/orchestration/GlobalQueueManager.js` | ENHANCED | Build PageGraph during discovery |

---

## Data Flow Summary

**Discovery Phase**:
```
BFS Traversal
  ↓
LinkExtractor (extracts URLs + block IDs)
  ↓
EdgeClassifier (classifies edges by depth)
  ↓
PageGraph (stores hierarchy with edge metadata)
  ↓
GlobalQueueManager (maintains complete graph)
```

**Execution Phase**:
```
DownloadHandler
  ↓
BlockIDExtractor (from saved HTML)
  ↓
BlockIDMapper (save to .block-ids.json)
  ↓
LinkRewriter (use PageGraph for context, BlockIDMapper for anchors)
  ↓
Offline HTML with correct relative paths and working anchors
```

---

## Benefits

1. **Back Edge Awareness**: System understands edge directionality, enabling future optimizations and navigation analysis

2. **Section Navigation**: Users can jump to specific sections offline via proper anchor rewriting

3. **Clean Decomposition**: Four focused components, each handling one concern

4. **Non-invasive**: Improvements layer on top of existing logic without major refactoring

5. **Extensible**: Edge metadata can support future features (navigation flow analysis, circular pattern detection, etc.)
