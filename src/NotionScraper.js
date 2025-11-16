const puppeteer = require('puppeteer');
const path = require('path');
const Config = require('./Config');
const Logger = require('./Logger');
const CookieHandler = require('./CookieHandler');
const ContentExpander = require('./ContentExpander');
const LinkExtractor = require('./LinkExtractor');
const AssetDownloader = require('./AssetDownloader');
const CssDownloader = require('./CssDownloader');
const FileDownloader = require('./FileDownloader');
const PageScraper = require('./PageScraper');
const RecursiveScraper = require('./RecursiveScraper');
const IntegrityAuditor = require('./IntegrityAuditor');

/**
 * Main orchestrator for Notion scraping
 */
class NotionScraper {
  constructor(config = null) {
    this.config = config || new Config();
    this.logger = new Logger();
    this.browser = null;
    this.page = null;
    
    // Initialize components
    this.cookieHandler = new CookieHandler(this.config, this.logger);
    this.contentExpander = new ContentExpander(this.config, this.logger);
    this.linkExtractor = new LinkExtractor(this.config, this.logger);
    this.assetDownloader = new AssetDownloader(this.config, this.logger);
    this.cssDownloader = new CssDownloader(this.config, this.logger);
    this.fileDownloader = new FileDownloader(this.config, this.logger);
    this.pageScraper = new PageScraper(
      this.config,
      this.logger,
      this.cookieHandler,
      this.contentExpander,
      this.linkExtractor,
      this.assetDownloader,
      this.cssDownloader,
      this.fileDownloader
    );
    this.recursiveScraper = new RecursiveScraper(
      this.config,
      this.logger,
      this.pageScraper
    );
    this.integrityAuditor = new IntegrityAuditor(this.config, this.logger);
  }
  
  /**
   * Initialize browser and page
   */
  async initialize() {
    this.logger.separator('Initializing Notion Scraper');
    this.logger.info('MAIN', `Target URL: ${this.config.NOTION_PAGE_URL}`);
    this.logger.info('MAIN', `Output directory: ${this.config.OUTPUT_DIR}`);
    this.logger.info('MAIN', `Max recursion depth: ${this.config.MAX_RECURSION_DEPTH}`);
    
    this.browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    this.page = await this.browser.newPage();
    
    // Set a reasonable viewport
    await this.page.setViewport({ width: 1920, height: 1080 });
    
    this.logger.success('MAIN', 'Browser initialized successfully');
  }
  
  /**
   * Run the complete scraping process
   */
  async run() {
    try {
      await this.initialize();
      
      // Start recursive scraping and link rewriting
      const result = await this.recursiveScraper.scrape(
        this.page,
        this.config.NOTION_PAGE_URL
      );

      const auditSummary = await this.integrityAuditor.audit(result.allContexts);
      
      // Print statistics
      this._printStatistics(result.rootContext, result.totalLinksRewritten, auditSummary);
      
      this.logger.success('MAIN', 'All operations completed successfully');
      
    } catch (error) {
      this.logger.error('MAIN', 'Fatal error during scraping', error);
      throw error;
    } finally {
      await this.cleanup();
    }
  }
  
  /**
   * Print scraping statistics
   */
  _printStatistics(rootContext, totalLinksRewritten, auditSummary) {
    this.logger.separator('Scraping Statistics');
    
    const scraperStats = this.pageScraper.getStats();
    const assetStats = this.assetDownloader.getStats();
    const fileStats = this.fileDownloader.getStats();
    const cssStats = this.cssDownloader.getStats();
    
    this.logger.info('STATS', `Total pages scraped: ${scraperStats.pagesScraped}`);
    this.logger.info('STATS', `Total images downloaded: ${assetStats.totalAssets}`);
    this.logger.info('STATS', `Total stylesheets downloaded: ${cssStats.totalCss}`);
    this.logger.info('STATS', `Total CSS assets localized: ${cssStats.totalCssAssets}`);
    this.logger.info('STATS', `Total files downloaded: ${fileStats.totalFiles}`);
    this.logger.info('STATS', `Total internal links rewritten: ${totalLinksRewritten}`);
    this.logger.info('STATS', `Total time elapsed: ${this.logger.getElapsedTime()}`);
    this.logger.info('STATS', '');
    this.logger.info('STATS', 'SUCCESS: The downloaded site is now fully browsable offline!');
    this.logger.info('STATS', `Open ${path.join(this.config.OUTPUT_DIR, 'Main_Page', 'index.html')} in your browser.`);
    this.logger.info('STATS', '');

    if (auditSummary) {
      this.logger.info(
        'STATS',
        `Integrity audit → missing HTML: ${auditSummary.missingFiles}, residual Notion URLs: ${auditSummary.residualNotionLinks}, external stylesheets: ${auditSummary.externalStylesheets}`
      );
      if (auditSummary.issuesFound === 0) {
        this.logger.success('STATS', 'Integrity audit passed with no outstanding issues.');
      } else {
        this.logger.warn('STATS', `Integrity audit flagged ${auditSummary.issuesFound} potential issue(s). See audit log for details.`);
      }
      this.logger.info('STATS', '');
    }
    
    // Print hierarchy
    this.logger.info('STATS', 'Page hierarchy:');
    this._printHierarchy(rootContext, 0);
    
    this.logger.separator();
  }
  
  /**
   * Print page hierarchy recursively
   */
  _printHierarchy(context, indent) {
    const prefix = '  '.repeat(indent) + '├─ ';
    const path = context.getRelativePath() || 'root';
    console.log(`${prefix}${context.title} (${path})`);
    
    for (const child of context.children) {
      this._printHierarchy(child, indent + 1);
    }
  }
  
  /**
   * Cleanup resources
   */
  async cleanup() {
    this.logger.info('MAIN', 'Cleaning up resources...');
    
    if (this.browser) {
      await this.browser.close();
      this.logger.success('MAIN', 'Browser closed');
    }
  }
}

module.exports = NotionScraper;
