/**
 * @fileoverview Puppeteer-based CSS extraction strategy
 * @module download/css/strategies/PuppeteerCssStrategy
 * @description Handles CSS extraction from a live Puppeteer page during active scraping.
 */

const path = require('path');
const fs = require('fs').promises;
const Logger = require('../../../core/Logger');
const CssContentProcessor = require('../CssContentProcessor');
const CssAssetDownloader = require('../CssAssetDownloader');

/**
 * @class PuppeteerCssStrategy
 * @classdesc Extracts and localizes CSS from a live Puppeteer page.
 */
class PuppeteerCssStrategy {
  /**
   * @param {Object} config - Configuration object with timeout and navigation settings
   * @param {Logger} logger - Logger instance
   */
  constructor(config, logger = null) {
    this.config = config;
    this.logger = logger || new Logger();
    this.contentProcessor = new CssContentProcessor(this.logger);
    this.assetDownloader = new CssAssetDownloader(this.config, this.logger);
    this.cssCount = 0;
  }

  /**
   * @async
   * @method process
   * @summary Extracts CSS from a live browser session
   * @description
   * 1. Evaluates the page to find <link rel="stylesheet"> tags
   * 2. Downloads CSS content via Node.js axios
   * 3. Processes @import and url() references
   * 4. Injects local paths back into the live DOM
   * @param {import('puppeteer').Page} page - The active Puppeteer page
   * @param {string} outputDir - The output directory
   * @returns {Promise<number>} Number of stylesheets processed
   */
  async process(page, outputDir) {
    this.logger.info('CSS', 'Extracting stylesheet URLs from live page...');

    // Extract stylesheet references from live DOM
    const stylesheets = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
        .map(link => ({
          href: link.href,
          originalHref: link.getAttribute('href')
        }))
        .filter(s => s.href && !s.href.startsWith('data:') && !s.href.startsWith('blob:'));
    });

    if (stylesheets.length === 0) {
      this.logger.info('CSS', 'No external stylesheets found');
      return 0;
    }

    this.logger.info('CSS', `Found ${stylesheets.length} stylesheet(s) to process`);

    // Create CSS directory structure
    const cssDir = path.join(outputDir, 'css');
    const assetDir = path.join(cssDir, 'assets');
    await fs.mkdir(cssDir, { recursive: true });
    await fs.mkdir(assetDir, { recursive: true });

    const pageCssCache = new Map();
    const pageAssetCache = new Map();
    let successCount = 0;

    // Process each stylesheet
    for (const sheet of stylesheets) {
      try {
        if (pageCssCache.has(sheet.href)) {
          const cachedPath = pageCssCache.get(sheet.href);
          const relativePath = path.relative(outputDir, cachedPath).replace(/\\/g, '/');

          await page.evaluate((originalHref, newPath) => {
            const links = document.querySelectorAll(`link[rel="stylesheet"]`);
            for (const link of links) {
              if (link.href === originalHref || link.getAttribute('href') === originalHref) {
                link.setAttribute('href', newPath);
              }
            }
          }, sheet.originalHref, relativePath);

          successCount++;
          continue;
        }

        // Download CSS content
        const cssContent = await this._downloadCssContent(sheet.href);
        if (!cssContent) {
          this.logger.warn('CSS', `Failed to download: ${sheet.href}`);
          continue;
        }

        // Process CSS
        const processed = await this.contentProcessor.processCssContent(cssContent, {
          baseUrl: sheet.href,
          cssDir,
          assetDir,
          referenceContext: 'css',
          pageCssCache,
          pageAssetCache,
          downloadCssFile: null,
          downloadCssAsset: this.assetDownloader.downloadCssAsset.bind(this.assetDownloader)
        });

        // Save processed CSS
        const filename = this._extractCssFilename(sheet.href);
        const localCssPath = path.join(cssDir, filename);
        await fs.writeFile(localCssPath, processed.css, 'utf-8');

        pageCssCache.set(sheet.href, localCssPath);
        this.cssCount++;

        // Rewrite href in live DOM
        const relativePath = path.relative(outputDir, localCssPath).replace(/\\/g, '/');

        await page.evaluate((originalHref, newPath) => {
          const links = document.querySelectorAll(`link[rel="stylesheet"]`);
          for (const link of links) {
            if (link.href === originalHref || link.getAttribute('href') === originalHref) {
              link.setAttribute('href', newPath);
            }
          }
        }, sheet.originalHref, relativePath);

        successCount++;
        this.logger.success('CSS', `Localized: ${filename}`);

      } catch (error) {
        this.logger.warn('CSS', `Failed to process ${sheet.href}: ${error.message}`);
      }
    }

    this.logger.success('CSS', `Processed ${successCount}/${stylesheets.length} stylesheet(s)`);
    return successCount;
  }

  /**
   * @private
   * @async
   * @param {string} url - CSS file URL
   * @returns {Promise<string|null>} CSS content or null on failure
   */
  async _downloadCssContent(url) {
    const axios = require('axios');
    try {
      const response = await axios.get(url, {
        responseType: 'text',
        timeout: this.config.TIMEOUT_NAVIGATION,
        maxRedirects: 5,
        headers: { 'User-Agent': 'Mozilla/5.0' }
      });
      return response.data;
    } catch (error) {
      this.logger.warn('CSS', `Download failed for ${url}: ${error.message}`);
      return null;
    }
  }

  /**
   * @private
   * @param {string} url - CSS file URL
   * @returns {string} Sanitized filename with hash prefix
   */
  _extractCssFilename(url) {
    const crypto = require('crypto');
    const FileSystemUtils = require('../../../utils/FileSystemUtils');

    try {
      const urlObj = new URL(url);
      let filename = path.basename(urlObj.pathname) || 'stylesheet.css';
      if (!filename.toLowerCase().endsWith('.css')) {
        filename += '.css';
      }
      filename = FileSystemUtils.sanitizeFilename(filename, '.css');
      const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 8);
      return `${hash}-${filename}`;
    } catch (error) {
      return `${crypto.createHash('md5').update(url).digest('hex').substring(0, 12)}.css`;
    }
  }

  /**
   * @summary Get statistics
   * @returns {Object} Download counts
   */
  getStats() {
    return {
      stylesheetsDownloaded: this.cssCount,
      totalCssAssets: this.assetDownloader.assetCount
    };
  }

  /**
   * @summary Reset statistics
   */
  reset() {
    this.cssCount = 0;
    this.assetDownloader.reset();
  }
}

module.exports = PuppeteerCssStrategy;
