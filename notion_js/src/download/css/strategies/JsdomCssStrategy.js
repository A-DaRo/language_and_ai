/**
 * @fileoverview JSDOM-based CSS rewriting strategy
 * @module download/css/strategies/JsdomCssStrategy
 * @description Handles CSS localization for static HTML files using JSDOM.
 */

const path = require('path');
const fs = require('fs').promises;
const Logger = require('../../../core/Logger');
const CssContentProcessor = require('../CssContentProcessor');
const CssAssetDownloader = require('../CssAssetDownloader');

/**
 * @class JsdomCssStrategy
 * @classdesc Rewrites CSS links in a static DOM for offline browsing.
 */
class JsdomCssStrategy {
  /**
   * @param {Object} config - Configuration object
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
   * @summary Rewrites CSS links in a static DOM
   * @description
   * 1. Scans JSDOM document for <link> and <style> tags
   * 2. Downloads referenced CSS files
   * 3. Localizes assets
   * 4. Updates DOM attributes to point to local files
   * @param {import('jsdom').JSDOM} dom - The parsed DOM
   * @param {string} pageDir - Directory where HTML file resides
   * @param {string} pageUrl - Absolute URL of the original page
   * @returns {Promise<Object>} Statistics about processed resources
   */
  async process(dom, pageDir, pageUrl) {
    const document = dom.window.document;
    const linkElements = Array.from(
      document.querySelectorAll('link[rel="stylesheet"], link[href*=".css"]')
    );

    const cssDir = path.join(pageDir, 'css');
    const assetDir = path.join(cssDir, 'assets');
    await fs.mkdir(cssDir, { recursive: true });
    await fs.mkdir(assetDir, { recursive: true });

    const pageCssCache = new Map();
    const pageAssetCache = new Map();

    let stylesheetsDownloaded = 0;
    let assetsLocalized = 0;

    for (const link of linkElements) {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('data:') || href.startsWith('css/') || href.startsWith('./css/')) {
        continue;
      }

      if (!this._isCssFile(href)) {
        continue;
      }

      const cssUrl = this._resolveUrl(href, pageUrl);
      if (!cssUrl) {
        continue;
      }

      const beforeCss = this.cssCount;
      const beforeAssets = this.assetDownloader.assetCount;

      const cssInfo = await this._downloadCssFile(cssUrl, cssDir, assetDir, pageCssCache, pageAssetCache);
      if (!cssInfo) {
        continue;
      }

      link.setAttribute('href', cssInfo.hrefFromHtml);

      stylesheetsDownloaded += this.cssCount - beforeCss;
      assetsLocalized += this.assetDownloader.assetCount - beforeAssets;
    }

    const inlineResult = await this._processInlineStyles(
      dom,
      cssDir,
      assetDir,
      pageUrl,
      pageCssCache,
      pageAssetCache
    );

    assetsLocalized += inlineResult.assets;

    const modified = stylesheetsDownloaded > 0 || assetsLocalized > 0 || inlineResult.inlineStyles > 0;

    return {
      stylesheets: stylesheetsDownloaded,
      assets: assetsLocalized,
      inlineStyles: inlineResult.inlineStyles,
      modified
    };
  }

  /**
   * @private
   * @param {string} url - URL to check
   * @returns {boolean} True if URL likely points to CSS
   */
  _isCssFile(url) {
    if (!url || typeof url !== 'string') return false;
    const normalized = url.toLowerCase();
    return normalized.includes('.css') || normalized.includes('stylesheet') || normalized.includes('text/css');
  }

  /**
   * @private
   * @async
   * @param {string} url - CSS file URL
   * @param {string} cssDir - CSS directory path
   * @param {string} assetDir - Asset directory path
   * @param {Map} pageCssCache - CSS file cache
   * @param {Map} pageAssetCache - Asset cache
   * @returns {Promise<Object|null>} CSS file info or null on failure
   */
  async _downloadCssFile(url, cssDir, assetDir, pageCssCache, pageAssetCache) {
    if (pageCssCache.has(url)) {
      return pageCssCache.get(url);
    }

    pageCssCache.set(url, null);

    const axios = require('axios');
    const crypto = require('crypto');
    const FileSystemUtils = require('../../../utils/FileSystemUtils');

    const maxRetries = 3;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await axios.get(url, {
          responseType: 'arraybuffer',
          timeout: this.config.TIMEOUT_NAVIGATION,
          maxRedirects: 5,
          headers: { 'User-Agent': 'Mozilla/5.0' }
        });

        const filename = this._extractCssFilename(url);
        const filePath = path.join(cssDir, filename);
        await fs.writeFile(filePath, response.data);

        const cssText = Buffer.from(response.data).toString('utf-8');
        const processed = await this.contentProcessor.processCssContent(cssText, {
          baseUrl: url,
          cssDir,
          assetDir,
          referenceContext: 'css',
          pageCssCache,
          pageAssetCache,
          downloadCssFile: this._downloadCssFile.bind(this),
          downloadCssAsset: this.assetDownloader.downloadCssAsset.bind(this.assetDownloader)
        });

        if (processed.css !== cssText) {
          await fs.writeFile(filePath, processed.css, { encoding: 'utf-8' });
        }

        const info = {
          filename,
          hrefFromHtml: path.posix.join('css', filename)
        };

        pageCssCache.set(url, info);
        this.cssCount += 1;

        const sizeKB = (response.data.length / 1024).toFixed(2);
        this.logger.debug('CSS', `Saved ${filename} (${sizeKB} KB)`);

        return info;

      } catch (error) {
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000;
          this.logger.debug('CSS', `Retry ${attempt}/${maxRetries} for ${url}`);
          await this._sleep(delay);
        } else {
          this.logger.error('CSS', `Failed after ${maxRetries} attempts: ${url}`);
          pageCssCache.delete(url);
          return null;
        }
      }
    }

    pageCssCache.delete(url);
    return null;
  }

  /**
   * @private
   * @async
   * @param {JSDOM} dom - Document object model
   * @param {string} cssDir - CSS directory path
   * @param {string} assetDir - Asset directory path
   * @param {string} pageUrl - Page URL
   * @param {Map} pageCssCache - CSS cache
   * @param {Map} pageAssetCache - Asset cache
   * @returns {Promise<Object>} Processing statistics
   */
  async _processInlineStyles(dom, cssDir, assetDir, pageUrl, pageCssCache, pageAssetCache) {
    const styleElements = Array.from(dom.window.document.querySelectorAll('style'));
    if (styleElements.length === 0) {
      return { inlineStyles: 0, assets: 0 };
    }

    let modifiedBlocks = 0;
    let assetsLocalized = 0;

    for (const styleEl of styleElements) {
      const cssText = styleEl.textContent || '';
      if (!cssText.trim()) continue;

      const beforeAssets = this.assetDownloader.assetCount;
      const processed = await this.contentProcessor.processCssContent(cssText, {
        baseUrl: pageUrl,
        cssDir,
        assetDir,
        referenceContext: 'inline',
        pageCssCache,
        pageAssetCache,
        downloadCssFile: this._downloadCssFile.bind(this),
        downloadCssAsset: this.assetDownloader.downloadCssAsset.bind(this.assetDownloader)
      });

      assetsLocalized += this.assetDownloader.assetCount - beforeAssets;

      if (processed.css !== cssText) {
        styleEl.textContent = processed.css;
        modifiedBlocks++;
      }
    }

    if (modifiedBlocks > 0) {
      this.logger.info('CSS', `Updated ${modifiedBlocks} inline <style> block(s)`);
    }

    return { inlineStyles: modifiedBlocks, assets: assetsLocalized };
  }

  /**
   * @private
   * @param {string} resourceUrl - URL to resolve
   * @param {string} baseUrl - Base URL
   * @returns {string|null} Absolute URL or null
   */
  _resolveUrl(resourceUrl, baseUrl) {
    if (!resourceUrl) return null;

    const trimmed = resourceUrl.trim();
    if (!trimmed || trimmed.startsWith('data:') || trimmed.startsWith('about:') || trimmed.startsWith('#')) {
      return null;
    }

    if (trimmed.startsWith('//')) {
      return `https:${trimmed}`;
    }

    if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
      return trimmed;
    }

    try {
      return new URL(trimmed, baseUrl).href;
    } catch (error) {
      this.logger.debug('CSS', `Unable to resolve URL ${resourceUrl}`);
      return null;
    }
  }

  /**
   * @private
   * @param {string} url - CSS file URL
   * @returns {string} Sanitized filename
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
   * @private
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise<void>}
   */
  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
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

module.exports = JsdomCssStrategy;
