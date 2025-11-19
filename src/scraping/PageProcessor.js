const fs = require('fs/promises');
const path = require('path');

/**
 * @classdesc Handles the scraping of individual Notion pages.
 * 
 * Coordinates the complete page scraping workflow:
 * - Navigation and page loading via Puppeteer
 * - Cookie consent handling
 * - Content expansion (toggles, databases)
 * - Link extraction for discovery
 * - Asset downloads (images, files)
 * - HTML persistence with proper directory structure
 * 
 * Maintains visited URL tracking to prevent duplicate processing and manages
 * the PageContext registry for link rewriting operations.
 * 
 * @see PageContext
 * @see CookieHandler
 * @see ContentExpander
 * @see LinkExtractor
 * @see AssetDownloader
 * @see FileDownloader
 */
class PageProcessor {
  /**
   * @param {Config} config - Configuration object.
   * @param {Logger} logger - Logger instance.
   * @param {CookieHandler} cookieHandler - Cookie handler instance.
   * @param {ContentExpander} contentExpander - Content expander instance.
   * @param {LinkExtractor} linkExtractor - Link extractor instance.
   * @param {AssetDownloader} assetDownloader - Asset downloader instance.
   * @param {FileDownloader} fileDownloader - File downloader instance.
   */
  constructor(
    config,
    logger,
    cookieHandler,
    contentExpander,
    linkExtractor,
    assetDownloader,
    fileDownloader
  ) {
    this.config = config;
    this.logger = logger;
    this.cookieHandler = cookieHandler;
    this.contentExpander = contentExpander;
    this.linkExtractor = linkExtractor;
    this.assetDownloader = assetDownloader;
    this.fileDownloader = fileDownloader;
    this.visitedUrls = new Set();
    this.urlToContextMap = new Map(); // Map of URL -> PageContext for link rewriting
  }
  
  /**
   * @summary Reset visited state before a fresh execution phase.
   * 
   * @description Clears the visitedUrls Set to allow pages to be re-scraped in a new
   * execution phase. Should be called before starting the execution phase after discovery.
   */
  resetVisited() {
    this.visitedUrls.clear();
  }
  
  /**
   * @summary Reset the page context map prior to discovery.
   * 
   * @description Clears the URL-to-PageContext mapping to prepare for a fresh discovery phase.
   * Should be called before starting a new discovery run.
   */
  resetContextMap() {
    this.urlToContextMap.clear();
  }
  
  /**
   * @summary Register a page context for link rewriting.
   * 
   * @description Stores a PageContext in the registry, allowing the LinkRewriter to
   * resolve URLs to local file paths during the link rewriting phase.
   * 
   * @param {string} url - The URL of the page.
   * @param {PageContext} context - The page context object.
   * 
   * @see LinkRewriter#rewriteLinksInFile
   */
  registerPageContext(url, context) {
    this.urlToContextMap.set(url, context);
  }
  
  /**
   * @summary Check if a URL is already registered in the context map.
   * 
   * @param {string} url - The URL to check.
   * @returns {boolean} True if registered, false otherwise.
   */
  isUrlRegistered(url) {
    return this.urlToContextMap.has(url);
  }
  
  /**
   * @summary Get the map of registered page contexts.
   * 
   * @description Returns the complete URL-to-PageContext mapping for use in link rewriting.
   * 
   * @returns {Map<string, PageContext>} The context map keyed by URL.
   */
  getContextMap() {
    return this.urlToContextMap;
  }

  /**
   * @summary Scrape a single page and return its links.
   * 
   * @description Executes the complete page scraping workflow:
   * 1. Checks visited state to prevent duplicates
   * 2. Navigates to the page URL
   * 3. Handles cookie consent (if first page)
   * 4. Expands all collapsible content
   * 5. Extracts internal links
   * 6. Downloads assets (images, files)
   * 7. Saves HTML to disk
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {PageContext} pageContext - Context of the page to scrape.
   * @param {boolean} [isFirstPage=false] - Whether this is the first page (for cookie handling).
   * @returns {Promise<Array<Object>>} List of discovered link objects with url, title, section, subsection.
   * 
   * @throws {Error} If page navigation fails or timeout is exceeded.
   * 
   * @see PageContext#getDirectoryPath
   * @see ContentExpander#expandAll
   * @see LinkExtractor#extractLinks
   */
  async scrapePage(page, pageContext, isFirstPage = false) {
    const url = pageContext.url;
    
    if (this.visitedUrls.has(url)) {
      this.logger.info('SCRAPE', `Skipping already visited page: ${pageContext.title}`);
      return [];
    }
    
    this.logger.info('SCRAPE', `Scraping page: ${pageContext.title} (depth: ${pageContext.depth})`);
    this.visitedUrls.add(url);
    
    try {
      await page.goto(url, {
        waitUntil: 'networkidle0',
        timeout: this.config.TIMEOUT_PAGE_LOAD
      });

      const pageTitle = await page.title();
      this.logger.info('SCRAPE', `Page loaded: "${pageTitle}"`);

      if (isFirstPage) {
        await this.cookieHandler.handle(page);
      }
      
      await this.contentExpander.expandAll(page);
      
      const links = await this.linkExtractor.extractLinks(page, url);
      
      const outputDir = pageContext.getDirectoryPath(this.config.OUTPUT_DIR);
      await fs.mkdir(outputDir, { recursive: true });
      this.logger.info('SCRAPE', `Created directory: ${outputDir}`);
      
      await this.assetDownloader.downloadAndRewriteImages(page, outputDir);
      await this.fileDownloader.downloadAndRewriteFiles(page, outputDir);
      
      await this._savePageHtml(page, pageContext);
      
      return links;
      
    } catch (error) {
      this.logger.error('SCRAPE', `Error scraping page ${pageContext.title}`, error);
      return [];
    }
  }

  /**
   * @summary Lightweight metadata fetch used during discovery phase.
   * 
   * @description Performs minimal page interaction to extract page title and links
   * without heavy asset downloads. Uses 'domcontentloaded' wait strategy for speed.
   * Attempts to extract Notion-specific title, falls back to page title.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {string} url - URL to discover.
   * @param {boolean} [isFirstPage=false] - Whether this is the first page.
   * @returns {Promise<{title: string, links: Array<Object>}>} Object containing page title and link array.
   * 
   * @throws {Error} If page navigation fails.
   * 
   * @see RecursiveScraper#discover
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

      await this.contentExpander.expandAll(page);

      const links = await this.linkExtractor.extractLinks(page, url);
      return { title, links };
    } catch (error) {
      this.logger.error('DISCOVERY', `Failed to discover page info for ${url}`, error);
      return { title: 'Unavailable', links: [] };
    }
  }
  
  /**
   * Save the page HTML to disk.
   * @param {Page} page - Puppeteer page instance.
   * @param {PageContext} pageContext - Context of the page.
   */
  async _savePageHtml(page, pageContext) {
    const html = await page.content();
    const htmlFilePath = pageContext.getFilePath(this.config.OUTPUT_DIR);
    const directory = path.dirname(htmlFilePath);
    await fs.mkdir(directory, { recursive: true });
    await fs.writeFile(htmlFilePath, html, { encoding: 'utf-8' });
    this.logger.success('SCRAPE', `Saved HTML to: ${htmlFilePath}`);
    pageContext.htmlFilePath = htmlFilePath;
  }
  
  /**
   * Get statistics.
   * @returns {Object} Statistics object.
   */
  getStats() {
    return {
      pagesScraped: this.visitedUrls.size
    };
  }
}

module.exports = PageProcessor;
