const PageContext = require('../domain/PageContext');
const GraphAnalyzer = require('./analysis/GraphAnalyzer');

/**
 * @classdesc Orchestrates recursive scraping of Notion pages using strict BFS traversal.
 * 
 * Implements a two-phase approach:
 * 1. **Discovery Phase**: Builds the PageContext tree using strict breadth-first search.
 *    Processes all nodes at level N before any node at level N+1, ensuring proper
 *    depth hierarchy and edge classification.
 * 2. **Execution Phase**: Traverses the discovered tree, performing full scraping
 *    operations (downloads, parsing, link rewriting) following tree edges only.
 * 
 * Edge classification during discovery helps identify graph structure:
 * - Tree edges: Parent-child relationships (first discovery)
 * - Back edges: Cycles to ancestors or self-loops
 * - Forward edges: Shortcuts to descendants
 * - Cross edges: Connections between branches
 * 
 * @see GraphAnalyzer
 * @see PageProcessor
 * @see LinkRewriter
 * @see PageContext
 */
class RecursiveScraper {
  /**
   * @param {Config} config - Configuration object.
   * @param {Logger} logger - Logger instance.
   * @param {PageProcessor} pageProcessor - Page processor instance.
   * @param {LinkRewriter} linkRewriter - Link rewriter instance.
   */
  constructor(config, logger, pageProcessor, linkRewriter) {
    this.config = config;
    this.logger = logger;
    this.pageProcessor = pageProcessor;
    this.linkRewriter = linkRewriter;
    this.allContexts = [];
    this.graphAnalyzer = new GraphAnalyzer(logger);
  }
  
  /**
   * @summary Discovery phase: build the PageContext tree without heavy downloads.
   * 
   * @description Implements strict BFS traversal:
   * - Processes all nodes at level N before any node at level N+1
   * - Enforces breadth hierarchy: deeper nodes are ONLY explored after ALL shallower nodes
   * - Classifies edges into tree/back/forward/cross types using GraphAnalyzer
   * - Registers all contexts for later scraping execution
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {string} rootUrl - The starting URL.
   * @param {number} maxDepth - Maximum recursion depth.
   * @returns {Promise<{rootContext: PageContext, allContexts: PageContext[]}>} Discovery results.
   * 
   * @throws {Error} If page navigation fails or required page info cannot be extracted.
   * 
   * @see GraphAnalyzer#classifyEdge
   * @see PageProcessor#discoverPageInfo
   */
  async discover(page, rootUrl, maxDepth) {
    this.logger.separator('Starting Recursive Discovery (Strict BFS with Edge Classification)');
    this.allContexts = [];
    this.pageProcessor.resetContextMap();
    this.graphAnalyzer.resetEdgeClassification();
    
    const rootContext = new PageContext(rootUrl, 'Main_Page', 0, null);
    this.pageProcessor.registerPageContext(rootUrl, rootContext);
    this.allContexts.push(rootContext);
    
    // BFS state tracking
    const discovered = new Map(); // url -> { context, depth, discoveryOrder }
    discovered.set(rootUrl, { context: rootContext, depth: 0, discoveryOrder: 0 });
    
    let currentLevel = [{ context: rootContext, isFirstPage: true }];
    let currentDepth = 0;
    let discoveryOrder = 1;
    
    // STRICT BFS: Complete entire level before proceeding
    while (currentLevel.length > 0 && currentDepth < maxDepth) {
      this.logger.info('DISCOVERY', `=== BFS Level ${currentDepth}: ${currentLevel.length} node(s) ===`);
      
      // Collect ALL links from ALL nodes at current level
      const allNewChildren = await this._processLevelNodes(
        page, 
        currentLevel, 
        discovered, 
        discoveryOrder
      );
      
      // Log edge classification statistics for this level
      this.graphAnalyzer.logEdgeStatistics(currentDepth);
      
      // Move to next level ONLY after ALL nodes at current level are processed
      currentLevel = allNewChildren;
      currentDepth++;
    }
    
    if (currentLevel.length > 0) {
      this.logger.info('DISCOVERY', `Depth limit (${maxDepth}) reached. ${currentLevel.length} node(s) at level ${currentDepth} not expanded.`);
    }
    
    this.graphAnalyzer.logFinalEdgeStatistics();
    this.logger.success('DISCOVERY', `Discovery complete. Total pages: ${this.allContexts.length}`);
    return { rootContext, allContexts: this.allContexts };
  }

  /**
   * @summary Process all nodes at the current BFS level and collect new children.
   * 
   * @description For each node in the current level:
   * - Discover page info and links
   * - Process each link to classify edges or create new child contexts
   * - Register new contexts and add them to the discovery queue
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {Array<{context: PageContext, isFirstPage: boolean}>} currentLevel - Nodes to process.
   * @param {Map<string, {context: PageContext, depth: number, discoveryOrder: number}>} discovered - Tracking map.
   * @param {number} discoveryOrder - Current discovery order counter.
   * @returns {Promise<Array<{context: PageContext, isFirstPage: boolean}>>} New children for next level.
   * @private
   */
  async _processLevelNodes(page, currentLevel, discovered, discoveryOrder) {
    const allNewChildren = [];
    
    for (const { context, isFirstPage } of currentLevel) {
      this.logger.info('DISCOVERY', `  Discovering links in: ${context.displayTitle || context.title} (depth: ${context.depth})`);
      
      const pageInfo = await this.pageProcessor.discoverPageInfo(page, context.url, isFirstPage);
      if (pageInfo && pageInfo.title) {
        context.setDisplayTitle(pageInfo.title);
      }
      const links = (pageInfo && pageInfo.links) || [];
      
      // Process each discovered link
      for (const linkInfo of links) {
        const childData = this._processLink(linkInfo, context, discovered, discoveryOrder);
        if (childData) {
          allNewChildren.push(childData);
        }
      }
    }
    
    return allNewChildren;
  }

  /**
   * @summary Process a single link during discovery phase.
   * 
   * @description Determines if the link is a new discovery (tree edge) or a revisit
   * (back/forward/cross edge), then handles appropriately by either creating a new
   * child context or classifying the edge type.
   * 
   * @param {Object} linkInfo - Link information containing url, title, section, subsection.
   * @param {PageContext} context - Current page context.
   * @param {Map<string, {context: PageContext, depth: number, discoveryOrder: number}>} discovered - Tracking map.
   * @param {number} discoveryOrder - Current discovery order counter (mutated).
   * @returns {{context: PageContext, isFirstPage: boolean}|null} Child data for next level, or null.
   * @private
   */
  _processLink(linkInfo, context, discovered, discoveryOrder) {
    if (!linkInfo.url) {
      return null;
    }
    
    const targetUrl = linkInfo.url;
    const existingNode = discovered.get(targetUrl);
    
    if (existingNode) {
      // URL already discovered - classify the edge type
      this.graphAnalyzer.classifyEdge(context, existingNode.context, targetUrl, linkInfo.title);
      return null;
    }
    
    // NEW DISCOVERY: This is a tree edge
    const childDepth = context.depth + 1;
    const childContext = new PageContext(
      targetUrl,
      linkInfo.title,
      childDepth,
      context
    );
    childContext.isNestedUnderParent = true;
    childContext.setDisplayTitle(linkInfo.title);
    if (linkInfo.section) childContext.setSection(linkInfo.section);
    if (linkInfo.subsection) childContext.setSubsection(linkInfo.subsection);
    
    // Register immediately to prevent duplicate discoveries in same level
    discovered.set(targetUrl, { 
      context: childContext, 
      depth: childDepth, 
      discoveryOrder: discoveryOrder++ 
    });
    this.pageProcessor.registerPageContext(targetUrl, childContext);
    this.allContexts.push(childContext);
    context.addChild(childContext);
    
    // Mark as tree edge
    this.graphAnalyzer.addTreeEdge(context.url, targetUrl, linkInfo.title);
    
    this.logger.debug('DISCOVERY', `    [TREE EDGE] ${context.title} -> ${linkInfo.title}`);
    
    // Return child data for next level exploration
    return { context: childContext, isFirstPage: false };
  }

  
  /**
   * @summary Execution phase: traverse planned tree and run full scraping routine.
   * 
   * @description Implements strict BFS traversal following tree edges only:
   * - Processes all nodes at level N before any node at level N+1
   * - Ignores back/forward/cross edges discovered during the discovery phase
   * - Performs full page scraping (downloads, parsing, link extraction)
   * - Rewrites internal links after all pages are scraped
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {PageContext} rootContext - The root context from discovery phase.
   * @returns {Promise<{rootContext: PageContext, totalLinksRewritten: number, allContexts: PageContext[]}>} Execution results.
   * 
   * @throws {Error} If rootContext is not provided or scraping operations fail.
   * 
   * @see PageProcessor#scrapePage
   * @see LinkRewriter#rewriteLinksInFile
   */
  async execute(page, rootContext) {
    if (!rootContext) {
      throw new Error('Execution requires a previously discovered root context');
    }
    if (!this.allContexts || this.allContexts.length === 0) {
      this.allContexts = this._collectContexts(rootContext);
    }
    
    this.logger.separator('Starting Recursive Scraping (Strict BFS - Tree Edges Only)');
    this.pageProcessor.resetVisited();
    
    // Use level-by-level processing for strict BFS
    let currentLevel = [{ context: rootContext, isFirstPage: true }];
    let currentDepth = 0;
    
    while (currentLevel.length > 0) {
      this.logger.info('RECURSION', `=== Scraping Level ${currentDepth}: ${currentLevel.length} page(s) ===`);
      const nextLevel = [];
      
      // Process ALL nodes at current level before moving to next level
      for (const { context, isFirstPage } of currentLevel) {
        if (context.depth >= this.config.MAX_RECURSION_DEPTH) {
          this.logger.warn('RECURSION', `Skipping ${context.displayTitle || context.title} - max depth reached`);
          continue;
        }
        
        this.logger.info('RECURSION', `  Scraping: ${context.displayTitle || context.title} (depth: ${context.depth})`);
        await this.pageProcessor.scrapePage(page, context, isFirstPage);
        
        // Collect children for next level (only tree edges, children are already in context.children)
        for (const child of context.children) {
          nextLevel.push({ context: child, isFirstPage: false });
        }
      }
      
      // Move to next level only after ALL nodes at current level are scraped
      currentLevel = nextLevel;
      currentDepth++;
    }
    
    this.logger.separator('Scraping Complete - Starting Link Rewriting');
    
    let totalLinksRewritten = 0;
    const contextMap = this.pageProcessor.getContextMap();
    
    for (const ctx of this.allContexts) {
      const count = await this.linkRewriter.rewriteLinksInFile(ctx, contextMap);
      totalLinksRewritten += count;
    }
    
    this.logger.success('LINK-REWRITE', `Total internal links rewritten: ${totalLinksRewritten}`);
    this.logger.separator('All Operations Complete');
    
    return { rootContext, totalLinksRewritten, allContexts: this.allContexts };
  }
  
  /**
   * @summary Collect all contexts from the tree using depth-first traversal.
   * 
   * @param {PageContext} rootContext - The root context.
   * @returns {PageContext[]} List of all contexts in the tree.
   * @private
   */
  _collectContexts(rootContext) {
    const results = [];
    const stack = [rootContext];
    while (stack.length > 0) {
      const ctx = stack.pop();
      results.push(ctx);
      for (const child of ctx.children) {
        stack.push(child);
      }
    }
    return results;
  }
}

module.exports = RecursiveScraper;
