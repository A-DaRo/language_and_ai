/**
 * Renders the discovered page hierarchy as an ASCII tree for user review.
 * 
 * @classdesc Provides visualization of the discovered page structure before execution.
 * Generates an ASCII tree representation showing the hierarchical relationships between
 * pages, with special handling for already-visited nodes to prevent infinite recursion
 * in cyclic graphs.
 * 
 * @see NotionScraper
 * @see PageContext
 * @see RecursiveScraper
 */
class PlanDisplayer {
  /**
   * @param {Logger} logger - Logger instance for informational messages.
   */
  constructor(logger) {
    this.logger = logger;
  }

  /**
   * @summary Display the discovered page hierarchy as an ASCII tree.
   * 
   * Renders the complete page structure as an indented tree, showing parent-child
   * relationships. Detects and marks already-visited nodes to prevent infinite loops
   * in graphs with back edges or cross edges.
   * 
   * @param {PageContext} rootContext - The root page context to display.
   * 
   * @example
   * displayer.displayTree(rootContext);
   * // Output:
   * // .
   * // └─ Main Page
   * //    ├─ Getting Started
   * //    │  └─ Installation
   * //    └─ Documentation
   */
  displayTree(rootContext) {
    if (!rootContext) {
      this.logger.warn('PLAN', 'No root context to display.');
      return;
    }

    console.log('[PLAN] Discovery complete. The following site structure will be scraped:');
    console.log('.');
    
    const rootLabel = rootContext.displayTitle || rootContext.title || '(root)';
    const rootChildren = rootContext.children || [];
    
    if (rootChildren.length === 0) {
      console.log(`└─ ${rootLabel}`);
      return;
    }
    
    console.log(`└─ ${rootLabel}`);
    const visited = new Set();
    if (rootContext.url) visited.add(rootContext.url);
    
    rootChildren.forEach((child, index) => {
      const isLast = index === rootChildren.length - 1;
      this._printTreeNode(child, '   ', isLast, visited);
    });
  }

  /**
   * @summary Recursively print a tree node with proper indentation.
   * 
   * Internal helper that handles the recursive tree traversal and formatting.
   * Tracks visited URLs to detect and mark cycles in the graph.
   * 
   * @param {PageContext} context - The page context node to print.
   * @param {string} prefix - Current indentation prefix.
   * @param {boolean} isLast - Whether this is the last child of its parent.
   * @param {Set<string>} visited - Set of already-visited URLs to detect cycles.
   * @private
   */
  _printTreeNode(context, prefix, isLast, visited) {
    if (context.url && visited.has(context.url)) {
      const connector = isLast ? '└─ ' : '├─ ';
      const title = context.displayTitle || context.title || 'Untitled';
      console.log(`${prefix}${connector} ${title} [already visited]`);
      return;
    }
    
    if (context.url) visited.add(context.url);
    
    const connector = isLast ? '└─ ' : '├─ ';
    const title = context.displayTitle || context.title || 'Untitled';
    console.log(`${prefix}${connector} ${title}`);
    
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    context.children.forEach((child, index) => {
      const childIsLast = index === context.children.length - 1;
      this._printTreeNode(child, childPrefix, childIsLast, visited);
    });
  }
}

module.exports = PlanDisplayer;
