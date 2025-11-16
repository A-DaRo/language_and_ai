const PageContext = require('./PageContext');

/**
 * Orchestrates recursive scraping of Notion pages
 */
class RecursiveScraper {
  constructor(config, logger, pageScraper) {
    this.config = config;
    this.logger = logger;
    this.pageScraper = pageScraper;
    this.scrapeQueue = [];
    this.allContexts = []; // Track all contexts for link rewriting
  }
  
  /**
   * Start recursive scraping from the root page
   */
  async scrape(page, rootUrl) {
    this.logger.separator('Starting Recursive Scraping');
    this.scrapeQueue = [];
    this.allContexts = [];
    
    // Create root context
    const rootContext = new PageContext(rootUrl, 'Main_Page', 0, null);
    
    // Register the root context
    this.pageScraper.registerPageContext(rootUrl, rootContext);
    this.allContexts.push(rootContext);
    
    // Add to queue
    this.scrapeQueue.push({ context: rootContext, isFirstPage: true });
    
    // Process queue
    while (this.scrapeQueue.length > 0) {
      const { context, isFirstPage } = this.scrapeQueue.shift();
      
      // Check depth limit
      if (context.depth >= this.config.MAX_RECURSION_DEPTH) {
        this.logger.warn('RECURSION', `Skipping ${context.title} - max depth reached`);
        continue;
      }
      
      // Scrape the page
      const links = await this.pageScraper.scrapePage(page, context, isFirstPage);
      
      // Add child pages to queue
      for (const linkInfo of links) {
        // Skip if already visited
        if (this.pageScraper.isVisited(linkInfo.url)) {
          continue;
        }
        
        // Create child context - now as a nested child under parent
        const childContext = new PageContext(
          linkInfo.url,
          linkInfo.title,
          context.depth + 1,
          context // This creates the nested structure
        );
        
        childContext.isNestedUnderParent = true;
        
        // Add to parent's children
        context.addChild(childContext);
        
        // Register context for link rewriting
        this.pageScraper.registerPageContext(linkInfo.url, childContext);
        this.allContexts.push(childContext);
        
        // Add to queue
        this.scrapeQueue.push({ context: childContext, isFirstPage: false });
      }
      
      // Log progress
      this.logger.info('RECURSION', `Queue size: ${this.scrapeQueue.length} pages remaining`);
    }
    
    this.logger.separator('Scraping Complete - Starting Link Rewriting');
    
    // Phase 2: Rewrite all internal links in saved HTML files
    let totalLinksRewritten = 0;
    for (const ctx of this.allContexts) {
      const count = await this.pageScraper.rewriteLinksInFile(ctx);
      totalLinksRewritten += count;
    }
    
    this.logger.success('LINK-REWRITE', `Total internal links rewritten: ${totalLinksRewritten}`);
    this.logger.separator('All Operations Complete');
    
    return { rootContext, totalLinksRewritten, allContexts: this.allContexts };
  }
}

module.exports = RecursiveScraper;
