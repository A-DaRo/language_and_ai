/**
 * @fileoverview Task Execution Router for Worker Process
 * @module worker/TaskRunner
 * @description Routes IPC commands to appropriate execution logic within the Worker process.
 * Handles both Discovery (metadata extraction) and Download (full scrape) tasks.
 * 
 * **CRITICAL**: Runs in Worker process. Must be stateless between tasks.
 */

const { MESSAGE_TYPES, serializeError } = require('../core/ProtocolDefinitions');
const PageContext = require('../domain/PageContext');
const LinkExtractor = require('../extraction/LinkExtractor');
const Config = require('../core/Config');
const Logger = require('../core/Logger');

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
    this.logger = new Logger();
    this.linkExtractor = new LinkExtractor(this.config, this.logger);
  }
  
  /**
   * Set cookies for this worker (broadcast from Master after first successful auth)
   * @async
   * @param {Array<Object>} cookies - Array of Puppeteer cookie objects
   * @returns {Promise<void>}
   */
  async setCookies(cookies) {
    this.cookies = cookies || [];
    console.log(`[TaskRunner] Received ${this.cookies.length} cookie(s) from Master`);
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
      console.log(`[TaskRunner] Executing ${taskType} task for: ${payload.url}`);
      
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
      console.error(`[TaskRunner] Task failed:`, error);
      
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
    console.log(`[TaskRunner] Navigating to: ${payload.url}`);
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
      console.log(`[TaskRunner] Captured ${capturedCookies.length} cookie(s) from first page`);
    }
    
    // Return results with resolved page name
    return {
      success: true,
      pageId: payload.pageId,
      url: payload.url,
      title: pageName || 'Untitled',
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
    // Create or reuse page
    if (!this.page) {
      this.page = await this.browser.newPage();
      await this.page.setViewport({ width: 1920, height: 1080 });
    }
    
    // Set cookies
    if (payload.cookies && payload.cookies.length > 0) {
      await this.page.setCookie(...payload.cookies);
    }
    
    // Navigate to page
    console.log(`[TaskRunner] Downloading: ${payload.url}`);
    await this.page.goto(payload.url, {
      waitUntil: 'networkidle0',
      timeout: 60000
    });
    
    // Get page content
    const html = await this.page.content();
    
    // TODO: Implement full PageProcessor logic here
    // For now, return placeholder
    
    return {
      success: true,
      pageId: payload.pageId,
      url: payload.url,
      savedPath: payload.targetFilePath,
      assetsDownloaded: 0,
      linksRewritten: 0
    };
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
