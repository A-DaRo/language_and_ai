const puppeteer = require('puppeteer');
const path = require('path');
const readline = require('readline');
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
    this.currentPlan = null;
    
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
  async run(options = {}) {
    try {
      await this.initialize();

      const runOptions = {
        dryRunOnly: false,
        autoConfirm: false,
        initialMaxDepth: this.config.MAX_RECURSION_DEPTH,
        ...options
      };

      let currentDepth = Math.min(
        runOptions.initialMaxDepth || this.config.MAX_RECURSION_DEPTH,
        this.config.MAX_RECURSION_DEPTH
      );
      if (currentDepth < 1) {
        currentDepth = 1;
      }

      while (true) {
        this.currentPlan = await this.plan(currentDepth);
        this.displayTree(this.currentPlan.rootContext);

        if (runOptions.dryRunOnly) {
          this.logger.info('MAIN', 'Dry run flag detected. Skipping execution phase.');
          return;
        }

        if (runOptions.autoConfirm) {
          this.logger.info('MAIN', '--yes flag provided. Skipping confirmation prompt.');
          break;
        }

        const userChoice = await this._promptForPlanDecision();
        if (userChoice === 'yes') {
          break;
        }
        if (userChoice === 'no') {
          this.logger.warn('MAIN', 'Aborted by user after discovery phase.');
          return;
        }
        if (userChoice === 'deeper') {
          if (currentDepth >= this.config.MAX_RECURSION_DEPTH) {
            this.logger.warn('MAIN', 'Already at configured MAX_RECURSION_DEPTH; cannot go deeper.');
          } else {
            currentDepth += 1;
            this.logger.info('MAIN', `Increasing discovery depth to ${currentDepth} and re-running plan...`);
            continue;
          }
        }
      }

      this.logger.info('MAIN', 'User confirmed. Starting Execution Phase...');
      console.log('[MAIN] User confirmed. Starting Execution Phase...');
      const result = await this.recursiveScraper.execute(this.page, this.currentPlan.rootContext);

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
   * Run discovery-only planning step
   */
  async plan(maxDepth) {
    this.logger.separator('Discovery Phase');
    const plan = await this.recursiveScraper.discover(
      this.page,
      this.config.NOTION_PAGE_URL,
      maxDepth
    );
    this.logger.success('PLAN', `Discovery complete with ${plan.allContexts.length} page(s).`);
    return plan;
  }

  /**
   * Render the discovered hierarchy as ASCII tree
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
    rootChildren.forEach((child, index) => {
      const isLast = index === rootChildren.length - 1;
      this._printTreeNode(child, '   ', isLast);
    });
  }

  _printTreeNode(context, prefix, isLast) {
    const connector = isLast ? '└─ ' : '├─ ';
    const title = context.displayTitle || context.title || 'Untitled';
    console.log(`${prefix}${connector} ${title}`);
    const childPrefix = prefix + (isLast ? '   ' : '│  ');
    context.children.forEach((child, index) => {
      const childIsLast = index === context.children.length - 1;
      this._printTreeNode(child, childPrefix, childIsLast);
    });
  }

  async _promptForPlanDecision() {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const promptText = '[PROMPT] Do you want to proceed with scraping this structure?\n> Enter (Y)es to continue, (n)o to abort, or (d)eeper to expand the search by one level: ';
    const answer = await new Promise(resolve => {
      rl.question(promptText, response => {
        rl.close();
        resolve(response);
      });
    });

    const normalized = (answer || '').trim().toLowerCase();
    if (normalized === 'n' || normalized === 'no') {
      return 'no';
    }
    if (normalized === 'd' || normalized === 'deeper') {
      return 'deeper';
    }
    return 'yes';
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
    const title = context.displayTitle || context.title;
    console.log(`${prefix}${title} (${path})`);
    
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
