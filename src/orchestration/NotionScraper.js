const puppeteer = require('puppeteer');
const path = require('path');

const Config = require('../core/Config');
const Logger = require('../core/Logger');
const StateManager = require('../core/StateManager');

const PageProcessor = require('../scraping/PageProcessor');
const LinkRewriter = require('../processing/LinkRewriter');
const RecursiveScraper = require('./RecursiveScraper');

// UI Components
const PlanDisplayer = require('./ui/PlanDisplayer');
const StatisticsDisplayer = require('./ui/StatisticsDisplayer');
const UserPrompt = require('./ui/UserPrompt');

// Helpers
const CookieHandler = require('../processing/CookieHandler');
const ContentExpander = require('../processing/ContentExpander');
const LinkExtractor = require('../extraction/LinkExtractor');
const AssetDownloader = require('../download/AssetDownloader');
const CssDownloader = require('../download/CssDownloader');
const FileDownloader = require('../download/FileDownloader');
const IntegrityAuditor = require('../utils/IntegrityAuditor');

/**
 * Facade orchestrating the entire Notion scraping workflow including discovery, planning, execution, and reporting.
 * 
 * @classdesc Main entry point for the Notion scraping system. Coordinates initialization,
 * user interaction, recursive scraping, and final reporting. Follows the Facade pattern,
 * delegating actual scraping logic to RecursiveScraper and UI responsibilities to
 * specialized display classes.
 * 
 * Workflow:
 * 1. Initialize browser and Puppeteer page
 * 2. Discovery phase: Build page graph using BFS
 * 3. User confirmation with interactive prompts
 * 4. Execution phase: Scrape all discovered pages
 * 5. Link rewriting for offline browsing
 * 6. Integrity audit
 * 7. Statistics reporting
 * 
 * @see RecursiveScraper
 * @see PageProcessor
 * @see LinkRewriter
 * @see PlanDisplayer
 * @see StatisticsDisplayer
 * @see UserPrompt
 */
class NotionScraper {
  /**
   * @param {Config} [config=null] - Optional configuration object. Creates default Config if not provided.
   */
  constructor(config = null) {
    this.config = config || new Config();
    this.logger = new Logger();
    this.browser = null;
    this.page = null;
    this.currentPlan = null;
    this.stateManager = StateManager.getInstance();
    
    // Initialize UI components
    this.planDisplayer = new PlanDisplayer(this.logger);
    this.statisticsDisplayer = new StatisticsDisplayer(this.logger);
    this.userPrompt = new UserPrompt();
    
    // Initialize processing components
    this.cookieHandler = new CookieHandler(this.config, this.logger);
    this.contentExpander = new ContentExpander(this.config, this.logger);
    this.linkExtractor = new LinkExtractor(this.config, this.logger);
    this.assetDownloader = new AssetDownloader(this.config, this.logger);
    this.cssDownloader = new CssDownloader(this.config, this.logger);
    this.fileDownloader = new FileDownloader(this.config, this.logger);
    
    this.pageProcessor = new PageProcessor(
      this.config,
      this.logger,
      this.cookieHandler,
      this.contentExpander,
      this.linkExtractor,
      this.assetDownloader,
      this.fileDownloader
    );
    
    this.linkRewriter = new LinkRewriter(
      this.config,
      this.logger,
      this.cssDownloader
    );
    
    this.recursiveScraper = new RecursiveScraper(
      this.config,
      this.logger,
      this.pageProcessor,
      this.linkRewriter
    );
    
    this.integrityAuditor = new IntegrityAuditor(this.config, this.logger);
  }
  
  /**
   * @summary Initialize browser and Puppeteer page instance.
   * 
   * Launches a headless Chromium browser and creates a new page with configured viewport.
   * 
   * @async
   * @returns {Promise<void>}
   * @throws {Error} If browser initialization fails.
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
    await this.page.setViewport({ width: 1920, height: 1080 });
    
    this.logger.success('MAIN', 'Browser initialized successfully');
  }
  
  /**
   * @summary Run the complete scraping workflow with interactive user confirmation.
   * 
   * Orchestrates the full scraping process:
   * 1. Discovery phase with optional depth adjustment
   * 2. User confirmation via interactive prompts
   * 3. Execution phase with full content download
   * 4. Link rewriting for offline browsing
   * 5. Integrity audit
   * 6. Statistics reporting
   * 
   * @async
   * @param {Object} [options={}] - Run configuration options.
   * @param {boolean} [options.dryRunOnly=false] - If true, only performs discovery without scraping.
   * @param {boolean} [options.autoConfirm=false] - If true, skips user confirmation prompts.
   * @param {number} [options.initialMaxDepth] - Starting depth for discovery (defaults to config value).
   * @returns {Promise<void>}
   * @throws {Error} If any phase of the scraping process fails.
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
        this.planDisplayer.displayTree(this.currentPlan.rootContext);

        if (runOptions.dryRunOnly) {
          this.logger.info('MAIN', 'Dry run flag detected. Skipping execution phase.');
          return;
        }

        if (runOptions.autoConfirm) {
          this.logger.info('MAIN', '--yes flag provided. Skipping confirmation prompt.');
          break;
        }

        const userChoice = await this.userPrompt.promptForPlanDecision();
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

      this._prepareExecutionPhase(this.currentPlan);

      this.logger.info('MAIN', 'User confirmed. Starting Execution Phase...');
      const result = await this.recursiveScraper.execute(this.page, this.currentPlan.rootContext);

      const auditSummary = await this.integrityAuditor.audit(result.allContexts);
      
      this.statisticsDisplayer.printStatistics(
        result.rootContext,
        result.totalLinksRewritten,
        auditSummary,
        this.pageProcessor.getStats(),
        this.assetDownloader.getStats(),
        this.fileDownloader.getStats(),
        this.cssDownloader.getStats(),
        this.config.OUTPUT_DIR,
        this.logger.getElapsedTime()
      );
      
      this.logger.success('MAIN', 'All operations completed successfully');
      
    } catch (error) {
      this.logger.error('MAIN', 'Fatal error during scraping', error);
      throw error;
    } finally {
      await this.cleanup();
    }
  }

  /**
   * @summary Plan the scraping by discovering pages without downloading content.
   * 
   * Performs lightweight discovery to build the page graph structure. Uses strict BFS
   * to ensure proper depth ordering. Does not download assets or content.
   * 
   * @async
   * @param {number} maxDepth - Maximum depth to discover (0 = root only, 1 = root + children, etc.).
   * @returns {Promise<{rootContext: PageContext, allContexts: Array<PageContext>}>} The discovery plan.
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
   * @summary Prepare for execution phase by registering all discovered contexts.
   * 
   * Registers all page contexts with the PageProcessor so they can be looked up
   * during link rewriting. Must be called before execution begins.
   * 
   * @param {Object} plan - The discovery plan from the plan() method.
   * @param {PageContext} plan.rootContext - Root page context.
   * @param {Array<PageContext>} plan.allContexts - All discovered page contexts.
   * @throws {Error} If plan is missing or invalid.
   * @private
   */
  _prepareExecutionPhase(plan) {
    if (!plan || !plan.rootContext) {
      throw new Error('Cannot start execution without a validated plan.');
    }
    const contexts = plan.allContexts || [];
    this.pageProcessor.resetContextMap();
    contexts.forEach(ctx => {
      this.pageProcessor.registerPageContext(ctx.url, ctx);
    });
    this.recursiveScraper.allContexts = contexts;
  }

  /**
   * @summary Cleanup browser resources.
   * 
   * Closes the Puppeteer browser instance and releases associated resources.
   * Called automatically in the finally block of run().
   * 
   * @async
   * @returns {Promise<void>}
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
