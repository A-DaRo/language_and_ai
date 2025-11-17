const PageContext = require('./PageContext');

/**
 * Orchestrates recursive scraping of Notion pages
 */
class RecursiveScraper {
  constructor(config, logger, pageScraper) {
    this.config = config;
    this.logger = logger;
    this.pageScraper = pageScraper;
    this.allContexts = [];
  }
  
  /**
   * Discovery phase: build the PageContext tree without heavy downloads
   * STRICT BFS: Process all nodes at level N before any node at level N+1
   */
  async discover(page, rootUrl, maxDepth) {
    this.logger.separator('Starting Recursive Discovery (Strict BFS)');
    this.allContexts = [];
    this.pageScraper.resetContextMap();
    
    const rootContext = new PageContext(rootUrl, 'Main_Page', 0, null);
    this.pageScraper.registerPageContext(rootUrl, rootContext);
    this.allContexts.push(rootContext);
    
    // Use level-by-level processing for strict BFS
    let currentLevel = [{ context: rootContext, isFirstPage: true }];
    const visited = new Set([rootUrl]);
    let currentDepth = 0;
    
    while (currentLevel.length > 0 && currentDepth < maxDepth) {
      this.logger.info('DISCOVERY', `Processing level ${currentDepth} with ${currentLevel.length} node(s)`);
      const nextLevel = [];
      
      // Process ALL nodes at current level before moving to next level
      for (const { context, isFirstPage } of currentLevel) {
        this.logger.info('DISCOVERY', `  Expanding: ${context.displayTitle || context.title} (depth: ${context.depth})`);
        
        const pageInfo = await this.pageScraper.discoverPageInfo(page, context.url, isFirstPage);
        if (pageInfo && pageInfo.title) {
          context.setDisplayTitle(pageInfo.title);
        }
        const links = (pageInfo && pageInfo.links) || [];
        
        // Collect children for next level
        for (const linkInfo of links) {
          if (!linkInfo.url) {
            continue;
          }
          
          // CRITICAL: Check if URL is already registered FIRST
          // This prevents breadcrumb/navigation links from creating spurious parent-child relationships
          // A URL already in the context map means it's already been placed in the tree at its correct depth
          if (this.pageScraper.isUrlRegistered(linkInfo.url)) {
            this.logger.debug('DISCOVERY', `  Skipping already registered URL: ${linkInfo.title} (breadcrumb/navigation link)`);
            continue;
          }
          
          // Check visited set (redundant safety check)
          if (visited.has(linkInfo.url)) {
            continue;
          }
          
          visited.add(linkInfo.url);
          
          const childContext = new PageContext(
            linkInfo.url,
            linkInfo.title,
            context.depth + 1,
            context
          );
          childContext.isNestedUnderParent = true;
          childContext.setDisplayTitle(linkInfo.title);
          if (linkInfo.section) childContext.setSection(linkInfo.section);
          if (linkInfo.subsection) childContext.setSubsection(linkInfo.subsection);
          
          context.addChild(childContext);
          this.pageScraper.registerPageContext(linkInfo.url, childContext);
          this.allContexts.push(childContext);
          
          // Add to next level (not current queue)
          nextLevel.push({ context: childContext, isFirstPage: false });
        }
      }
      
      // Move to next level only after ALL nodes at current level are processed
      currentLevel = nextLevel;
      currentDepth++;
    }
    
    if (currentLevel.length > 0) {
      this.logger.info('DISCOVERY', `Depth limit (${maxDepth}) reached. ${currentLevel.length} node(s) at level ${currentDepth} not expanded.`);
    }
    
    this.logger.success('DISCOVERY', `Discovery complete. Total pages found: ${this.allContexts.length}`);
    return { rootContext, allContexts: this.allContexts };
  }
  
  /**
   * Execution phase: traverse planned tree and run full scraping routine
   * STRICT BFS: Process all nodes at level N before any node at level N+1
   */
  async execute(page, rootContext) {
    if (!rootContext) {
      throw new Error('Execution requires a previously discovered root context');
    }
    if (!this.allContexts || this.allContexts.length === 0) {
      this.allContexts = this._collectContexts(rootContext);
    }
    
    this.logger.separator('Starting Recursive Scraping (Strict BFS)');
    this.pageScraper.resetVisited();
    
    // Use level-by-level processing for strict BFS
    let currentLevel = [{ context: rootContext, isFirstPage: true }];
    let currentDepth = 0;
    
    while (currentLevel.length > 0) {
      this.logger.info('RECURSION', `Scraping level ${currentDepth} with ${currentLevel.length} page(s)`);
      const nextLevel = [];
      
      // Process ALL nodes at current level before moving to next level
      for (const { context, isFirstPage } of currentLevel) {
        if (context.depth >= this.config.MAX_RECURSION_DEPTH) {
          this.logger.warn('RECURSION', `Skipping ${context.displayTitle || context.title} - max depth reached`);
          continue;
        }
        
        this.logger.info('RECURSION', `  Scraping: ${context.displayTitle || context.title} (depth: ${context.depth})`);
        await this.pageScraper.scrapePage(page, context, isFirstPage);
        
        // Collect children for next level
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
    for (const ctx of this.allContexts) {
      const count = await this.pageScraper.rewriteLinksInFile(ctx);
      totalLinksRewritten += count;
    }
    
    this.logger.success('LINK-REWRITE', `Total internal links rewritten: ${totalLinksRewritten}`);
    this.logger.separator('All Operations Complete');
    
    return { rootContext, totalLinksRewritten, allContexts: this.allContexts };
  }
  
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
