/**
 * @fileoverview Unified tree rendering service
 * @module orchestration/renderers/PageTreeRenderer
 * @description Encapsulates all logic for rendering site hierarchy trees.
 * 
 * **Single Responsibility**: Renders page trees in ASCII art format with cycle detection,
 * capturing output for both console display and structured logging.
 * 
 * **Used by**: UserConfirmationPhase (user review) and CompletionPhase (final reporting)
 */

const SystemEventBus = require('../core/SystemEventBus');
const Logger = require('../core/Logger');

/**
 * @class PageTreeRenderer
 * @classdesc Renders the discovered page hierarchy as an ASCII tree with cycle detection.
 * 
 * Provides a single source of truth for tree rendering logic, ensuring consistency
 * between user confirmation and completion phases. Captures output for logging.
 */
class PageTreeRenderer {
  constructor() {
    this.eventBus = SystemEventBus.getInstance();
    this.logger = Logger.getInstance();
  }

  /**
   * Render the page tree to console and logger
   * 
   * @param {PageContext} rootContext - Root page context
   * @param {Object} titleRegistry - ID-to-title mapping
   * @param {number} maxBfsDepth - Maximum depth of BFS expansion
   * @returns {string} The rendered tree as a string (for logging)
   */
  renderToConsoleAndLog(rootContext, titleRegistry, maxBfsDepth) {
    // Capture tree output as array of lines
    const treeLines = this._renderTree(rootContext, titleRegistry, maxBfsDepth);
    const treeOutput = treeLines.join('\n');

    // Display to console
    this.logger.separator('Page Tree');
    console.log('.');
    console.log(treeOutput);
    this.logger.separator();

    // Log to file
    this._logTreeToFile(treeOutput);

    return treeOutput;
  }

  /**
   * Render the page tree and return as string (for testing/logging)
   * 
   * @param {PageContext} rootContext - Root page context
   * @param {Object} titleRegistry - ID-to-title mapping
   * @param {number} maxBfsDepth - Maximum depth of BFS expansion
   * @returns {string} The rendered tree as a string
   */
  renderToString(rootContext, titleRegistry, maxBfsDepth) {
    const treeLines = this._renderTree(rootContext, titleRegistry, maxBfsDepth);
    return treeLines.join('\n');
  }

  /**
   * @private
   * Internal rendering logic
   * 
   * @param {PageContext} rootContext - Root page context
   * @param {Object} titleRegistry - ID-to-title mapping
   * @param {number} maxBfsDepth - Maximum depth of BFS expansion
   * @returns {Array<string>} Array of tree lines
   */
  _renderTree(rootContext, titleRegistry, maxBfsDepth) {
    const lines = [];

    const rootLabel = titleRegistry[rootContext.id] || rootContext.title || '(root)';
    lines.push(`└─ ${rootLabel}`);

    rootContext.children.forEach((child, index) => {
      const isLast = index === rootContext.children.length - 1;
      this._printTreeNode(
        lines,
        child,
        '   ',
        isLast,
        titleRegistry,
        new Set([rootContext.id]),
        maxBfsDepth
      );
    });

    return lines;
  }

  /**
   * @private
   * Recursively render tree nodes
   * 
   * @param {Array<string>} lines - Accumulator for output lines
   * @param {PageContext} context - Current page context
   * @param {string} prefix - Line prefix for indentation
   * @param {boolean} isLast - Is this the last child of its parent?
   * @param {Object} titleRegistry - ID-to-title mapping
   * @param {Set<string>} pathVisited - Visited nodes in current recursion path (cycle detection)
   * @param {number} maxBfsDepth - Maximum BFS depth (leaf level)
   */
  _printTreeNode(
    lines,
    context,
    prefix,
    isLast,
    titleRegistry,
    pathVisited = new Set(),
    maxBfsDepth = Infinity
  ) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = titleRegistry[context.id] || context.title || 'Untitled';

    // Detect cycles within the path (visited in current recursion)
    if (pathVisited.has(context.id)) {
      lines.push(`${prefix}${connector}${title} ↺ (Cycle)`);
      return;
    }

    // Filter to only show discovered children (have titles in registry)
    const exploredChildren = context.children.filter(child => titleRegistry[child.id]);
    const internalRefs = context.children.length - exploredChildren.length;
    const label =
      internalRefs > 0
        ? `${title} [${internalRefs} internal ref${internalRefs > 1 ? 's' : ''}]`
        : title;

    lines.push(`${prefix}${connector}${label}`);

    // Stop recursion at BFS depth to limit tree representation
    // This shows the full BFS expansion plus edges creating cycles at leaves
    if (context.depth >= maxBfsDepth) {
      // We're at or past the BFS leaf level - don't recurse deeper
      return;
    }

    // Recurse into children (we're within BFS expansion depth)
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    const nextVisited = new Set(pathVisited);
    nextVisited.add(context.id);

    exploredChildren.forEach((child, index) => {
      const childIsLast = index === exploredChildren.length - 1;

      // Only recurse if this is a tree edge (child's parent is current context)
      if (child.parentContext === context) {
        this._printTreeNode(
          lines,
          child,
          childPrefix,
          childIsLast,
          titleRegistry,
          nextVisited,
          maxBfsDepth
        );
      } else {
        // Non-tree edge (Cycle/Cross-link) - Print leaf and stop
        const connector = childIsLast ? '└─ ' : '├─ ';
        const title = titleRegistry[child.id] || child.title || 'Untitled';
        lines.push(`${childPrefix}${connector}${title} ↺ (Cycle)`);
      }
    });
  }

  /**
   * @private
   * Log tree output to structured logs
   * 
   * @param {string} treeOutput - The rendered tree string
   */
  _logTreeToFile(treeOutput) {
    // Split tree into lines for structured logging
    const lines = treeOutput.split('\n');

    // Log each line with PAGE_TREE context
    lines.forEach(line => {
      if (line.trim()) {
        this.logger.info('PAGE_TREE', line);
      }
    });
  }
}

module.exports = PageTreeRenderer;
