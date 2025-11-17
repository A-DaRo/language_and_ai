const fs = require('fs/promises');
const path = require('path');
const { JSDOM } = require('jsdom');

/**
 * Scrapes individual Notion pages and rewrites links for offline browsing
 */
class PageScraper {
  constructor(
    config,
    logger,
    cookieHandler,
    contentExpander,
    linkExtractor,
    assetDownloader,
    cssDownloader,
    fileDownloader
  ) {
    this.config = config;
    this.logger = logger;
    this.cookieHandler = cookieHandler;
    this.contentExpander = contentExpander;
    this.linkExtractor = linkExtractor;
    this.assetDownloader = assetDownloader;
    this.cssDownloader = cssDownloader;
    this.fileDownloader = fileDownloader;
    this.visitedUrls = new Set();
    this.urlToContextMap = new Map(); // Map of URL -> PageContext for link rewriting
  }
  
  /**
   * Reset visited state before a fresh execution phase
   */
  resetVisited() {
    this.visitedUrls.clear();
  }
  
  /**
   * Reset the page context map prior to discovery
   */
  resetContextMap() {
    this.urlToContextMap.clear();
  }
  
  /**
   * Register a page context for link rewriting
   */
  registerPageContext(url, context) {
    this.urlToContextMap.set(url, context);
  }
  
  /**
   * Scrape a single page and return its links
   */
  async scrapePage(page, pageContext, isFirstPage = false) {
    const url = pageContext.url;
    
    // Check if already visited
    if (this.visitedUrls.has(url)) {
      this.logger.info('SCRAPE', `Skipping already visited page: ${pageContext.title}`);
      return [];
    }
    
    this.logger.info('SCRAPE', `Scraping page: ${pageContext.title} (depth: ${pageContext.depth})`);
    this.visitedUrls.add(url);
    
    try {
      // Navigate to the page
      await page.goto(url, {
        waitUntil: 'networkidle0',
        timeout: this.config.TIMEOUT_PAGE_LOAD
      });

      const pageTitle = await page.title();
      this.logger.info('SCRAPE', `Page loaded: "${pageTitle}"`);

      // Handle cookie banner only on first page
      if (isFirstPage) {
        await this.cookieHandler.handle(page);
      }
      
      // Expand all content
      await this.contentExpander.expandAll(page);
      
      // Extract links for recursion
      const links = await this.linkExtractor.extractLinks(page, url);
      
      // Get the output directory - CRITICAL: use the single source of truth
      const outputDir = pageContext.getDirectoryPath(this.config.OUTPUT_DIR);
      
      // CRITICAL: Ensure the deeply nested directory exists
      await fs.mkdir(outputDir, { recursive: true });
      this.logger.info('SCRAPE', `Created directory: ${outputDir}`);
      
      // Download images
      await this.assetDownloader.downloadAndRewriteImages(page, outputDir);
      
      // Download embedded files (PDFs, code files, etc.)
      await this.fileDownloader.downloadAndRewriteFiles(page, outputDir);
      
      // Save the HTML with UTF-8 encoding
      await this._savePageHtml(page, pageContext);
      
      return links;
      
    } catch (error) {
      this.logger.error('SCRAPE', `Error scraping page ${pageContext.title}`, error);
      return [];
    }
  }

  /**
   * Lightweight metadata fetch used during discovery
   */
  async discoverPageInfo(page, url, isFirstPage = false) {
    try {
      this.logger.info('DISCOVERY', `Scanning page metadata: ${url}`);
      await page.goto(url, {
        waitUntil: 'domcontentloaded',
        timeout: this.config.TIMEOUT_PAGE_LOAD
      });

      if (isFirstPage) {
        await this.cookieHandler.handle(page);
      }

      let title = null;
      try {
        await page.waitForSelector('.notion-frame .notion-page-title-icon-and-text', {
          timeout: this.config.TIMEOUT_NAVIGATION
        });
        title = await page.$eval('.notion-frame .notion-page-title-icon-and-text', el => el.innerText.trim());  
      } catch (err) {
        this.logger.debug('DISCOVERY', `Fallback title strategy for ${url}: ${err.message}`);
      }

      if (!title) {
        title = (await page.title()) || 'Untitled';
      }

      const links = await this.linkExtractor.extractLinks(page, url);
      return { title, links };
    } catch (error) {
      this.logger.error('DISCOVERY', `Failed to discover page info for ${url}`, error);
      return { title: 'Unavailable', links: [] };
    }
  }
  
  /**
   * Save the page HTML to disk with full preservation of CSS and JavaScript
   * CRITICAL: Uses the single source of truth for file path
   */
  async _savePageHtml(page, pageContext) {
    // Get the complete HTML with all styles and scripts
    const html = await page.content();
    
    // CRITICAL: Use the context's method to get the exact file path
    const htmlFilePath = pageContext.getFilePath(this.config.OUTPUT_DIR);
    
    // Ensure the directory exists (defensive programming)
    const directory = path.dirname(htmlFilePath);
    await fs.mkdir(directory, { recursive: true });
    
    // Save with UTF-8 encoding to preserve emojis and special characters
    await fs.writeFile(htmlFilePath, html, { encoding: 'utf-8' });
    this.logger.success('SCRAPE', `Saved HTML to: ${htmlFilePath}`);
    
    // Store the file path for link rewriting later
    pageContext.htmlFilePath = htmlFilePath;
  }
  
  /**
   * Rewrite all internal links in saved HTML files to point to local paths
   * This is called after all pages have been scraped
   */
  async rewriteLinksInFile(pageContext) {
    try {
      const htmlFilePath = pageContext.htmlFilePath;
      if (!htmlFilePath) {
        this.logger.warn('LINK-REWRITE', `No HTML file path for ${pageContext.title}`);
        return 0;
      }
      
      // Read the HTML file
      const html = await fs.readFile(htmlFilePath, 'utf-8');
      
      // Parse with JSDOM
      const dom = new JSDOM(html);
      const document = dom.window.document;
      
      let rewriteCount = 0;
      let modified = false;
      const pageDir = path.dirname(htmlFilePath);
      
      // Localize external stylesheets and inline CSS assets
      if (this.cssDownloader) {
        const cssResult = await this.cssDownloader.downloadAndRewriteCss(dom, pageDir, pageContext.url);
        if (cssResult.modified) {
          modified = true;
          this.logger.success(
            'CSS',
            `Localized ${cssResult.stylesheets} stylesheet(s), ${cssResult.assets} dependent asset(s), ${cssResult.inlineStyles} inline style block(s) in ${pageContext.title}`
          );
        }
      }
      
      // Find all anchor tags
      const links = document.querySelectorAll('a[href]');
      
      for (const link of links) {
        const href = link.getAttribute('href');
        
        if (!href) continue;
        
        // Build absolute URL
        let absoluteUrl;
        try {
          if (href.startsWith('/')) {
            absoluteUrl = this.config.getBaseUrl() + href;
          } else if (href.startsWith('http')) {
            absoluteUrl = href;
          } else {
            continue; // Skip relative links, anchors, etc.
          }
          
          // Remove query parameters and fragments for matching
          absoluteUrl = absoluteUrl.split('?')[0].split('#')[0];
          
          // Check if this URL was scraped
          const targetContext = this.urlToContextMap.get(absoluteUrl);
          
          if (targetContext) {
            // This is an internal link - rewrite it
            const relativePath = pageContext.getRelativePathTo(targetContext);
            link.setAttribute('href', relativePath);
            rewriteCount++;
            this.logger.debug('LINK-REWRITE', `${pageContext.title}: ${href} -> ${relativePath}`);
          }
        } catch (error) {
          // Invalid URL, skip
          continue;
        }
      }
      
      // Save the modified HTML
      if (rewriteCount > 0 || modified) {
        const modifiedHtml = dom.serialize();
        await fs.writeFile(htmlFilePath, modifiedHtml, { encoding: 'utf-8' });
        if (rewriteCount > 0) {
          this.logger.success('LINK-REWRITE', `Rewrote ${rewriteCount} links in ${pageContext.title}`);
        }
      }
      
      return rewriteCount;
      
    } catch (error) {
      this.logger.error('LINK-REWRITE', `Error rewriting links in ${pageContext.title}`, error);
      return 0;
    }
  }
  
  /**
   * Check if a URL has been visited
   */
  isVisited(url) {
    return this.visitedUrls.has(url);
  }
  
  /**
   * Get statistics
   */
  getStats() {
    return {
      pagesScraped: this.visitedUrls.size
    };
  }
}

module.exports = PageScraper;
