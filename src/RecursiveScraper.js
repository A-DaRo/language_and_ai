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
   */
  async discover(page, rootUrl, maxDepth) {
    this.logger.separator('Starting Recursive Discovery');
    this.allContexts = [];
    this.pageScraper.resetContextMap();
    
    const rootContext = new PageContext(rootUrl, 'Main_Page', 0, null);
    this.pageScraper.registerPageContext(rootUrl, rootContext);
    this.allContexts.push(rootContext);
    
    const queue = [{ context: rootContext, isFirstPage: true }];
    const visited = new Set([rootUrl]);
    
    while (queue.length > 0) {
      const { context, isFirstPage } = queue.shift();
      
      if (context.depth >= maxDepth) {
        this.logger.info('DISCOVERY', `Depth limit reached for ${context.displayTitle || context.title}`);
        continue;
      }
      
      const pageInfo = await this.pageScraper.discoverPageInfo(page, context.url, isFirstPage);
      if (pageInfo && pageInfo.title) {
        context.setDisplayTitle(pageInfo.title);
      }
      const links = (pageInfo && pageInfo.links) || [];
      
      for (const linkInfo of links) {
        if (!linkInfo.url || visited.has(linkInfo.url)) {
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
        
        queue.push({ context: childContext, isFirstPage: false });
      }
    }
    
    return { rootContext, allContexts: this.allContexts };
  }
  
  /**
   * Execution phase: traverse planned tree and run full scraping routine
   */
  async execute(page, rootContext) {
    if (!rootContext) {
      throw new Error('Execution requires a previously discovered root context');
    }
    if (!this.allContexts || this.allContexts.length === 0) {
      this.allContexts = this._collectContexts(rootContext);
    }
    
    this.logger.separator('Starting Recursive Scraping');
    this.pageScraper.resetVisited();
    const scrapeQueue = [{ context: rootContext, isFirstPage: true }];
    
    while (scrapeQueue.length > 0) {
      const { context, isFirstPage } = scrapeQueue.shift();
      
      if (context.depth >= this.config.MAX_RECURSION_DEPTH) {
        this.logger.warn('RECURSION', `Skipping ${context.displayTitle || context.title} - max depth reached`);
        continue;
      }
      
      await this.pageScraper.scrapePage(page, context, isFirstPage);
      
      for (const child of context.children) {
        scrapeQueue.push({ context: child, isFirstPage: false });
      }
      
      this.logger.info('RECURSION', `Queue size: ${scrapeQueue.length} pages remaining`);
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
