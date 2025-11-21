/**
 * @fileoverview Task Execution Router for Worker Process
 * @module worker/TaskRunner
 * @description Routes IPC commands to appropriate execution logic within the Worker process.
 * Handles both Discovery (metadata extraction) and Download (full scrape) tasks.
 * 
 * **CRITICAL**: Runs in Worker process. Must be stateless between tasks.
 */

const path = require('path');
const { MESSAGE_TYPES, serializeError } = require('../core/ProtocolDefinitions');
const PageContext = require('../domain/PageContext');
const LinkExtractor = require('../extraction/LinkExtractor');
const Config = require('../core/Config');
const Logger = require('../core/Logger');

// Download components
const ContentExpander = require('../processing/ContentExpander');
const CookieHandler = require('../processing/CookieHandler');
const AssetDownloader = require('../download/AssetDownloader');
const CssDownloader = require('../download/CssDownloader');
const FileDownloader = require('../download/FileDownloader');

// Pipeline architecture
const WorkerFileSystem = require('./io/WorkerFileSystem');
const ScrapingPipeline = require('./pipeline/ScrapingPipeline');
const NavigationStep = require('./pipeline/steps/NavigationStep');
const CookieConsentStep = require('./pipeline/steps/CookieConsentStep');
const ExpansionStep = require('./pipeline/steps/ExpansionStep');
const AssetDownloadStep = require('./pipeline/steps/AssetDownloadStep');
const LinkRewriterStep = require('./pipeline/steps/LinkRewriterStep');
const HtmlWriteStep = require('./pipeline/steps/HtmlWriteStep');

/**
 * @class TaskRunner
 * @classdesc Routes task commands to appropriate handlers and manages task execution state
 */
class TaskRunner {
  /**
   * @param {import('puppeteer').Browser} browser - Puppeteer browser instance
   */
  constructor(browser) {
    this.browser = browser;
    this.page = null;
    this.config = new Config();
    this.cookies = [];
    this.titleRegistry = {}; // Cached title registry (sent once at init, updated with deltas)
    this.logger = Logger.getInstance();
    
    // Discovery components
    this.linkExtractor = new LinkExtractor(this.config, this.logger);
    
    // Download components (instantiated once per worker for cache efficiency)
    this.cookieHandler = new CookieHandler(this.config, this.logger);
    this.contentExpander = new ContentExpander(this.config, this.logger);
    this.assetDownloader = new AssetDownloader(this.config, this.logger);
    this.cssDownloader = new CssDownloader(this.config, this.logger);
    this.fileDownloader = new FileDownloader(this.config, this.logger);
    
    // File system abstraction for safe I/O
    this.fileSystem = new WorkerFileSystem(this.logger);
  }
  
  /**
   * Set cookies for this worker (broadcast from Master after first successful auth)
   * @async
   * @param {Array<Object>} cookies - Array of Puppeteer cookie objects
   * @returns {Promise<void>}
   */
  async setCookies(cookies) {
    this.cookies = cookies || [];
    this.logger.info('TaskRunner', `Received ${this.cookies.length} cookie(s) from Master`);
  }
  
  /**
   * Initialize or update title registry (sent once at init, then delta updates)
   * @param {Object} titleRegistry - ID-to-title map (full or delta)
   * @param {boolean} [isDelta=false] - Whether this is a delta update
   * @returns {void}
   */
  setTitleRegistry(titleRegistry, isDelta = false) {
    if (isDelta) {
      // Merge delta update
      this.titleRegistry = { ...this.titleRegistry, ...titleRegistry };
      this.logger.info('TaskRunner', `Updated title registry with ${Object.keys(titleRegistry).length} delta(s)`);
    } else {
      // Full initialization
      this.titleRegistry = titleRegistry || {};
      this.logger.info('TaskRunner', `Initialized title registry with ${Object.keys(this.titleRegistry).length} title(s)`);
    }
  }
  
  /**
   * Execute a task based on type
   * @async
   * @param {string} taskType - Task type (DISCOVER or DOWNLOAD)
   * @param {Object} payload - Task payload
   * @returns {Promise<Object>} Task result (WorkerResult format)
   */
  async execute(taskType, payload) {
    try {
      // Log only the ID to keep the terminal clean (as requested)
      const targetId = payload.pageId || (payload.url ? payload.url.split('/').pop() : 'unknown');
      this.logger.info('TaskRunner', `Executing ${taskType} task for: ${targetId}`);
      
      let result;
      
      if (taskType === MESSAGE_TYPES.DISCOVER) {
        result = await this._executeDiscovery(payload);
      } else if (taskType === MESSAGE_TYPES.DOWNLOAD) {
        result = await this._executeDownload(payload);
      } else {
        throw new Error(`Unknown task type: ${taskType}`);
      }
      
      return {
        type: MESSAGE_TYPES.RESULT,
        taskType,
        data: result
      };
      
    } catch (error) {
      this.logger.error('TaskRunner', `Task failed`, error);
      
      return {
        type: MESSAGE_TYPES.RESULT,
        taskType,
        error: serializeError(error)
      };
    }
  }
  
  /**
   * Execute discovery task (metadata only, no heavy downloads)
   * @private
   * @async
   * @param {import('../core/ProtocolDefinitions').DiscoverPayload} payload - Discovery payload
   * @returns {Promise<import('../core/ProtocolDefinitions').DiscoveryResult>} Discovery result
   */
  async _executeDiscovery(payload) {
    // Create or reuse page
    if (!this.page) {
      this.page = await this.browser.newPage();
      await this.page.setViewport({ width: 1920, height: 1080 });
    }
    
    // Set cookies if available
    if (this.cookies.length > 0) {
      await this.page.setCookie(...this.cookies);
    }
    
    // Navigate to page
    this.logger.info('TaskRunner', `Navigating to: ${payload.url}`);
    await this.page.goto(payload.url, {
      waitUntil: 'networkidle0',
      timeout: 60000
    });
    
    // Extract links using LinkExtractor (returns raw page IDs)
    const links = await this.linkExtractor.extractLinks(this.page, payload.url);
    
    // Normalize link format (LinkExtractor uses 'title', but we need 'text' for compatibility)
    const normalizedLinks = links.map(link => ({
      url: link.url,
      text: link.title,
      section: link.section,
      subsection: link.subsection
    }));
    
    // DEFERRED NAMING: Resolve page name from current URL (after all processing)
    // This is the last operation before returning results
    const finalUrl = this.page.url();
    let pageName = null;
    
    // Extract name from resolved URL (everything before first '-29')
    const nameMatch = finalUrl.match(/\/([^\/]+?)(-29[a-f0-9]{30})/i);
    if (nameMatch && nameMatch[1]) {
      // Found name in URL, use it
      pageName = nameMatch[1].replace(/-/g, ' ');
    } else {
      // Fallback: try to get from page title
      const pageTitle = await this.page.title();
      if (pageTitle && pageTitle !== 'Untitled' && pageTitle !== 'Notion' && !pageTitle.toLowerCase().includes('just a moment')) {
        pageName = pageTitle;
      } else {
        // Last resort: extract from original URL or use 'Untitled'
        const origNameMatch = payload.url.match(/\/([^\/]+?)(-29[a-f0-9]{30})/i);
        pageName = origNameMatch && origNameMatch[1] ? origNameMatch[1].replace(/-/g, ' ') : 'Untitled';
      }
    }
    
    // Capture cookies on first page
    let capturedCookies = null;
    if (payload.isFirstPage) {
      capturedCookies = await this.page.cookies();
      this.logger.info('TaskRunner', `Captured ${capturedCookies.length} cookie(s) from first page`);
    }
    
    // Return results with resolved page name
    return {
      success: true,
      pageId: payload.pageId,
      url: payload.url,
      resolvedTitle: pageName || 'Untitled',
      links: normalizedLinks,
      cookies: capturedCookies,
      metadata: {
        depth: payload.depth,
        parentId: payload.parentId
      }
    };
  }
  
  /**
   * Execute download task (full scrape with assets)
   * @private
   * @async
   * @param {import('../core/ProtocolDefinitions').DownloadPayload} payload - Download payload
   * @returns {Promise<import('../core/ProtocolDefinitions').DownloadResult>} Download result
   */
  async _executeDownload(payload) {
    // Validate payload structure
    this._validateDownloadPayload(payload);
    
    // Get human-readable title for logging (use cached registry)
    const displayTitle = this.titleRegistry[payload.pageId] || 'Unknown';
    this.logger.info('DOWNLOAD', `Starting download: ${displayTitle}`);
    
    // Create or reuse page
    if (!this.page) {
      this.page = await this.browser.newPage();
      await this.page.setViewport({ width: 1920, height: 1080 });
    }
    
    // Initialize pipeline context
    const pipelineContext = {
      browser: this.browser,
      page: this.page,
      config: this.config,
      logger: this.logger,
      payload: payload,
      fileSystem: this.fileSystem,
      stats: {
        assetsDownloaded: 0,
        linksRewritten: 0
      },
      downloadedAssets: []
    };
    
    // Construct the scraping pipeline
    const pipeline = new ScrapingPipeline(
      [
        new NavigationStep(),
        new CookieConsentStep(this.cookieHandler), // Precautionary cookie handling
        new ExpansionStep(this.contentExpander),
        new AssetDownloadStep(
          this.assetDownloader,
          this.cssDownloader,
          this.fileDownloader
        ),
        new LinkRewriterStep(),
        new HtmlWriteStep()
      ],
      this.logger
    );
    
    // Execute the pipeline
    try {
      await pipeline.execute(pipelineContext);
      
      this.logger.success('DOWNLOAD', `Completed: ${displayTitle}`);
      
      // Return truthful result from actual operations
      return {
        success: true,
        pageId: payload.pageId,
        url: payload.url,
        savedPath: payload.savePath,
        assetsDownloaded: pipelineContext.stats.assetsDownloaded,
        linksRewritten: pipelineContext.stats.linksRewritten
      };
      
    } catch (error) {
      this.logger.error('DOWNLOAD', `Failed: ${displayTitle}`, error);
      throw error;
    }
  }
  
  /**
   * Validate download payload structure
   * @private
   * @param {Object} payload - Download payload to validate
   * @throws {Error} If payload is invalid or missing required fields
   */
  _validateDownloadPayload(payload) {
    const requiredFields = ['url', 'pageId', 'savePath'];
    
    for (const field of requiredFields) {
      if (!payload[field]) {
        throw new Error(`Invalid download payload: missing required field '${field}'`);
      }
    }
    
    // Critical: Verify savePath is absolute
    if (!path.isAbsolute(payload.savePath)) {
      throw new Error(
        `Invalid download payload: savePath must be absolute. Received: ${payload.savePath}`
      );
    }
    
    this.logger.debug('VALIDATION', `Payload validated for ${payload.pageId}`);
  }
  
  /**
   * Cleanup resources
   * @async
   * @returns {Promise<void>}
   */
  async cleanup() {
    if (this.page) {
      await this.page.close();
      this.page = null;
    }
  }
}

module.exports = TaskRunner;
