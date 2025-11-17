const puppeteer = require('puppeteer');
const { Cluster } = require('puppeteer-cluster');
const os = require('os');
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
const StateManager = require('./StateManager');

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
    this.stateManager = StateManager.getInstance();
    this.discoveryCluster = null;
    this.clusterTaskRegistered = false;
    this.clusterEventsBound = false;
    this.currentDiscoveryDepthLimit = this.config.MAX_RECURSION_DEPTH;
    this.cookieHandledInCluster = false;
    
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

      this._prepareExecutionPhase(this.currentPlan);

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

  async plan(maxDepth) {
    this.logger.separator('Discovery Phase');
    if (!this.config.ENABLE_PARALLEL_DISCOVERY) {
      const plan = await this.recursiveScraper.discover(
        this.page,
        this.config.NOTION_PAGE_URL,
        maxDepth
      );
      this.logger.success('PLAN', `Discovery complete with ${plan.allContexts.length} page(s).`);
      return plan;
    }
    return this._runParallelDiscovery(maxDepth);
  }

  async _runParallelDiscovery(maxDepth) {
    const rootContext = this.stateManager.bootstrap(this.config.NOTION_PAGE_URL, 'Main_Page');
    this.currentDiscoveryDepthLimit = maxDepth;

    const cluster = await this._getOrCreateCluster();
    const state = this.stateManager;

    // Track if we've processed the root page for cookie handling
    this.cookieHandledInCluster = false;

    let currentDepth = 0;
    await cluster.idle();
    while (state.hasCurrentLevelWork() && currentDepth <= maxDepth) {
      const levelQueue = state.getCurrentLevelQueue();
      this.logger.info('DISCOVERY', `Dispatching ${levelQueue.length} page(s) at depth ${currentDepth}.`);
      for (const url of levelQueue) {
        const context = state.getContextByUrl(url);
        if (!context) {
          this.logger.warn('DISCOVERY', `No context found for queued URL ${url}; skipping.`);
          continue;
        }
        cluster.queue({ url, depth: context.depth });
      }
      await cluster.idle();
      state.advanceLevel();
      currentDepth += 1;
    }

    const plan = {
      rootContext,
      allContexts: state.getAllContexts()
    };
    this.logger.success('PLAN', `Discovery complete with ${plan.allContexts.length} page(s).`);
    return plan;
  }

  async _getOrCreateCluster() {
    if (this.discoveryCluster) {
      return this.discoveryCluster;
    }

    const { maxConcurrency, freeMemory } = this._calculateClusterConcurrency();
    this.logger.log('MAIN', `System has ${Math.round(freeMemory / (1024 * 1024))}MB free RAM.`);
    this.logger.log('MAIN', `Launching cluster with max concurrency: ${maxConcurrency}`);

    this.discoveryCluster = await Cluster.launch({
      concurrency: Cluster.CONCURRENCY_PAGE,
      maxConcurrency,
      retryLimit: this.config.CLUSTER_RETRY_LIMIT,
      retryDelay: this.config.CLUSTER_RETRY_DELAY,
      timeout: this.config.CLUSTER_TASK_TIMEOUT,
      puppeteerOptions: {
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
      }
    });

    this._bindClusterEvents(this.discoveryCluster);
    await this._registerClusterTask(this.discoveryCluster);
    return this.discoveryCluster;
  }

  _bindClusterEvents(cluster) {
    if (this.clusterEventsBound) {
      return;
    }

    cluster.on('taskerror', (err, data, willRetry) => {
      const target = data && data.url ? data.url : 'unknown job';
      if (willRetry) {
        this.logger.warn('DISCOVERY', `Cluster retry scheduled for ${target}: ${err.message}`);
      } else {
        this.logger.error('DISCOVERY', `Cluster job failed permanently for ${target}`, err);
      }
    });

    this.clusterEventsBound = true;
  }

  async _registerClusterTask(cluster) {
    if (this.clusterTaskRegistered) {
      return;
    }

    await cluster.task(async ({ page, data }) => {
      const url = data && data.url ? data.url : null;
      if (!url) {
        this.logger.warn('DISCOVERY', 'Cluster task missing URL payload. Skipping.');
        return;
      }

      const context = this.stateManager.getContextByUrl(url);
      if (!context) {
        this.logger.warn('DISCOVERY', `Context missing for ${url}. Skipping.`);
        return;
      }

      try {
        // With CONCURRENCY_PAGE, cookies are shared across all workers
        // Only handle cookies on the very first page (depth 0)
        const isFirstPage = context.depth === 0 && !this.cookieHandledInCluster;
        if (isFirstPage) {
          this.cookieHandledInCluster = true;
        }
        
        const info = await this.pageScraper.discoverPageInfo(page, url, isFirstPage);
        if (info && info.title) {
          context.setDisplayTitle(info.title);
        }

        const maxDepth = this.currentDiscoveryDepthLimit;
        const canDiscoverChildren = context.depth < maxDepth;
        if (!canDiscoverChildren) {
          return;
        }

        const links = (info && info.links) || [];
        for (const linkInfo of links) {
          if (!linkInfo || !linkInfo.url) continue;
          if (!this.config.isNotionUrl(linkInfo.url)) continue;
          const childDepth = context.depth + 1;
          if (childDepth > maxDepth) continue;
          this.stateManager.registerOrLink(linkInfo, context);
        }
      } catch (error) {
        this.logger.error('DISCOVERY', `Cluster worker failed for ${url}`, error);
        throw error;
      }
    });

    this.clusterTaskRegistered = true;
  }

  _calculateClusterConcurrency() {
    const BYTES_PER_INSTANCE = 1 * 1024 * 1024 * 1024;
    const OS_RAM_BUFFER = 2 * 1024 * 1024 * 1024;
    const freeMemory = os.freemem();
    const availableMemoryForWorkers = Math.max(0, freeMemory - OS_RAM_BUFFER);
    const rawConcurrency = Math.floor(availableMemoryForWorkers / BYTES_PER_INSTANCE);
    const maxConcurrency = Math.max(1, rawConcurrency);
    return { maxConcurrency, freeMemory };
  }

  _prepareExecutionPhase(plan) {
    if (!plan || !plan.rootContext) {
      throw new Error('Cannot start execution without a validated plan.');
    }
    const contexts = plan.allContexts || [];
    this.pageScraper.resetContextMap();
    contexts.forEach(ctx => {
      this.pageScraper.registerPageContext(ctx.url, ctx);
    });
    this.recursiveScraper.allContexts = contexts;
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

    if (this.discoveryCluster) {
      try {
        await this.discoveryCluster.idle();
      } catch (error) {
        this.logger.warn('MAIN', `Cluster idle check failed during cleanup: ${error.message}`);
      }
      await this.discoveryCluster.close();
      this.discoveryCluster = null;
      this.clusterTaskRegistered = false;
      this.clusterEventsBound = false;
      this.logger.success('MAIN', 'Discovery cluster closed');
    }
  }
}

module.exports = NotionScraper;
