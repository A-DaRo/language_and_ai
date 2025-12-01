const Logger = require('../core/Logger');
const PuppeteerCssStrategy = require('./css/strategies/PuppeteerCssStrategy');
const JsdomCssStrategy = require('./css/strategies/JsdomCssStrategy');

/**
 * Facade for CSS stylesheet downloading and DOM rewriting.
 * Delegates to runtime-specific strategies: Puppeteer for active scraping,
 * JSDOM for post-processing.
 *
 * @classdesc Lightweight orchestrator that routes CSS processing to appropriate strategies.
 * Maintains per-page caches to prevent duplicate downloads within a single page context.
 * 
 * @see PuppeteerCssStrategy
 * @see JsdomCssStrategy
 */
class CssDownloader {
  /**
   * @param {Object} config - Configuration object with timeout and navigation settings
   * @param {Logger} [logger=null] - Logger instance, creates new Logger if not provided
   */
  constructor(config, logger = null) {
    this.config = config;
    this.logger = logger || new Logger();
    this.puppeteerStrategy = new PuppeteerCssStrategy(config, this.logger);
    this.jsdomStrategy = new JsdomCssStrategy(config, this.logger);
  }

  /**
   * @summary Check if a URL appears to reference a CSS file
   * @param {string} url - URL to check
   * @returns {boolean} True if URL likely points to CSS
   */
  isCssFile(url) {
    if (!url || typeof url !== 'string') return false;
    const normalized = url.toLowerCase();
    return normalized.includes('.css') || normalized.includes('stylesheet') || normalized.includes('text/css');
  }

  /**
   * @summary Download stylesheets from a live Puppeteer page during active scraping
   * @description
   * Delegates to PuppeteerCssStrategy for active extraction and DOM rewriting.
   * 
   * @param {import('puppeteer').Page} page - The active Puppeteer page instance
   * @param {string} outputDir - Absolute path to the directory where index.html will be saved
   * @returns {Promise<number>} Count of stylesheets successfully processed
   */
  async downloadFromPuppeteer(page, outputDir) {
    return await this.puppeteerStrategy.process(page, outputDir);
  }

  /**
   * @summary Download all CSS files and rewrite DOM to use local versions (JSDOM mode)
   * @description
   * Main entry point for CSS localization in post-processing phase.
   * Delegates to JsdomCssStrategy for static HTML processing.
   *
   * @param {JSDOM} dom - Parsed DOM of the HTML file
   * @param {string} pageDir - Directory where the HTML file resides
   * @param {string} pageUrl - Absolute URL of the original Notion page
   * @returns {Promise<{stylesheets: number, assets: number, inlineStyles: number, modified: boolean}>}
   */
  async downloadAndRewriteCss(dom, pageDir, pageUrl) {
    return await this.jsdomStrategy.process(dom, pageDir, pageUrl);
  }

  /**
   * @summary Get download statistics
   * @returns {{stylesheetsDownloaded: number, totalCssAssets: number}} Download counts
   */
  getStats() {
    return {
      stylesheetsDownloaded: this.puppeteerStrategy.cssCount + this.jsdomStrategy.cssCount,
      totalCssAssets: this.puppeteerStrategy.assetDownloader.assetCount + 
                      this.jsdomStrategy.assetDownloader.assetCount
    };
  }

  /**
   * @summary Reset download statistics
   */
  reset() {
    this.puppeteerStrategy.reset();
    this.jsdomStrategy.reset();
  }
}

module.exports = CssDownloader;


module.exports = CssDownloader;
