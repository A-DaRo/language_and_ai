const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const Logger = require('../core/Logger');
const FileSystemUtils = require('../utils/FileSystemUtils');
const CssContentProcessor = require('./css/CssContentProcessor');
const CssAssetDownloader = require('./css/CssAssetDownloader');

/**
 * Coordinates CSS stylesheet downloading, dependency resolution, and DOM rewriting for offline browsing.
 * 
 * @classdesc This class orchestrates the multi-pass CSS localization strategy:
 * 1. Downloads external <link> stylesheets
 * 2. Processes @import directives recursively
 * 3. Localizes all url() asset references (fonts, images)
 * 4. Rewrites inline <style> blocks
 * 
 * Works as a coordinator, delegating CSS parsing to CssContentProcessor and
 * asset downloads to CssAssetDownloader. Maintains per-page caches to prevent
 * duplicate downloads within a single page context.
 * 
 * @see CssContentProcessor
 * @see CssAssetDownloader
 * @see LinkRewriter
 */
class CssDownloader {
  /**
   * @param {Object} config - Configuration object with timeout and navigation settings.
   * @param {Logger} [logger=null] - Logger instance, creates new Logger if not provided.
   */
  constructor(config, logger = null) {
    this.config = config;
    this.logger = logger || new Logger();

    this.cssCount = 0;
    
    // Initialize helper components
    this.contentProcessor = new CssContentProcessor(this.logger);
    this.assetDownloader = new CssAssetDownloader(this.config, this.logger);
  }

  /**
   * @summary Check if a URL appears to reference a CSS file.
   * 
   * @param {string} url - URL to check.
   * @returns {boolean} True if URL likely points to CSS.
   */
  isCssFile(url) {
    if (!url || typeof url !== 'string') return false;
    const normalized = url.toLowerCase();
    return normalized.includes('.css') || normalized.includes('stylesheet') || normalized.includes('text/css');
  }

  /**
   * @summary Download stylesheets from a live Puppeteer page during active scraping.
   * 
   * @description
   * This method is designed for the **Active Execution Phase** where we operate on a live
   * Puppeteer Page object. It performs CSS localization in three stages:
   * 1. Extraction: Uses page.evaluate() to scrape <link rel="stylesheet"> hrefs from live DOM
   * 2. Processing: Downloads CSS in Node.js context, processes @import and url() references
   * 3. Rewriting: Updates live DOM hrefs to point to local files before HTML serialization
   * 
   * This approach maintains performance by avoiding JSDOM initialization during scraping
   * while ensuring complete CSS localization for offline browsing.
   * 
   * @param {import('puppeteer').Page} page - The active Puppeteer page instance.
   * @param {string} outputDir - Absolute path to the directory where index.html will be saved.
   * @returns {Promise<number>} Count of stylesheets successfully processed.
   * 
   * @example
   * const cssCount = await cssDownloader.downloadFromPuppeteer(page, '/abs/path/to/output');
   * // Returns: 5 (number of stylesheets downloaded and localized)
   */
  async downloadFromPuppeteer(page, outputDir) {
    this.logger.info('CSS', 'Extracting stylesheet URLs from live page...');
    
    // Stage 1: Extract all stylesheet references from live DOM
    const stylesheets = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
        .map(link => ({
          href: link.href, // Absolute URL resolved by browser
          originalHref: link.getAttribute('href') // For matching during rewrite
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
    
    // Per-page caches
    const pageCssCache = new Map();
    const pageAssetCache = new Map();
    
    let successCount = 0;
    
    // Stage 2: Process each stylesheet in Node.js context
    for (const sheet of stylesheets) {
      try {
        // Skip if already processed
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
        
        // Process CSS: handle @import and url() references
        const processed = await this.contentProcessor.processCssContent(cssContent, {
          baseUrl: sheet.href,
          cssDir: cssDir,
          assetDir: assetDir,
          referenceContext: 'css',
          pageCssCache: pageCssCache,
          pageAssetCache: pageAssetCache,
          downloadCssFile: this._downloadCssFile.bind(this),
          downloadCssAsset: this.assetDownloader.downloadCssAsset.bind(this.assetDownloader)
        });
        
        // Save processed CSS to local file
        const filename = this._extractCssFilename(sheet.href);
        const localCssPath = path.join(cssDir, filename);
        await fs.writeFile(localCssPath, processed.css, 'utf-8');
        
        pageCssCache.set(sheet.href, localCssPath);
        this.cssCount++;
        
        // Stage 3: Rewrite href in live DOM
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
   * @summary Download all CSS files and rewrite DOM to use local versions (JSDOM mode).
   * 
   * Main entry point for CSS localization in post-processing phase. Processes both 
   * external <link> stylesheets and inline <style> blocks. Downloads all CSS files, 
   * their @import dependencies, and referenced assets (fonts, background images).
   * 
   * **Note:** This method is for the post-processing phase using JSDOM. For active
   * scraping with Puppeteer, use downloadFromPuppeteer() instead.
   * 
   * @param {JSDOM} dom - Parsed DOM of the HTML file.
   * @param {string} pageDir - Directory where the HTML file resides.
   * @param {string} pageUrl - Absolute URL of the original Notion page.
   * @returns {Promise<{stylesheets: number, assets: number, inlineStyles: number, modified: boolean}>}
   *          Statistics about processed resources and whether DOM was modified.
   * 
   * @example
   * const result = await cssDownloader.downloadAndRewriteCss(dom, '/path/to/page', 'https://notion.so/page');
   * // Result: { stylesheets: 3, assets: 12, inlineStyles: 2, modified: true }
   */
  async downloadAndRewriteCss(dom, pageDir, pageUrl) {
    const document = dom.window.document;
    const linkElements = Array.from(document.querySelectorAll('link[rel="stylesheet"], link[href*=".css"]'));

    const cssDir = path.join(pageDir, 'css');
    const assetDir = path.join(cssDir, 'assets');
    await fs.mkdir(cssDir, { recursive: true });
    await fs.mkdir(assetDir, { recursive: true });

    // Per-page caches to prevent duplicate downloads
    const pageCssCache = new Map();
    const pageAssetCache = new Map();

    let stylesheetsDownloaded = 0;
    let assetsLocalized = 0;

    for (const link of linkElements) {
      const href = link.getAttribute('href');
      if (!href || href.startsWith('data:') || href.startsWith('css/') || href.startsWith('./css/')) {
        continue;
      }

      if (!this.isCssFile(href)) {
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

      // Update DOM to point to local stylesheet
      link.setAttribute('href', cssInfo.hrefFromHtml);

      stylesheetsDownloaded += this.cssCount - beforeCss;
      assetsLocalized += this.assetDownloader.assetCount - beforeAssets;
    }

    const inlineResult = await this._processInlineStyles(dom, cssDir, assetDir, pageUrl, pageCssCache, pageAssetCache);
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
   * @summary Download CSS file content without saving (for Puppeteer mode).
   * 
   * @description
   * Simplified download method that only fetches CSS content without processing
   * or saving. Used by downloadFromPuppeteer() to get raw CSS for processing.
   * 
   * @param {string} url - URL of CSS file to download.
   * @returns {Promise<string|null>} CSS content as string or null on failure.
   * @private
   */
  async _downloadCssContent(url) {
    try {
      const response = await axios.get(url, {
        responseType: 'text',
        timeout: this.config.TIMEOUT_NAVIGATION,
        maxRedirects: 5,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      
      return response.data;
    } catch (error) {
      this.logger.warn('CSS', `Download failed for ${url}: ${error.message}`);
      return null;
    }
  }
  
  /**
   * @summary Download a CSS file with retry logic and process its dependencies.
   * 
   * Downloads a CSS file, then recursively processes @import directives and
   * url() asset references within it. Uses exponential backoff retry on failure.
   * 
   * @param {string} url - URL of CSS file to download.
   * @param {string} cssDir - Directory to save CSS files.
   * @param {string} assetDir - Directory to save CSS assets.
   * @param {Map<string, Object>} pageCssCache - Cache for CSS files.
   * @param {Map<string, Object>} pageAssetCache - Cache for assets.
   * @returns {Promise<{filename: string, hrefFromHtml: string}|null>} CSS file info or null on failure.
   * @private
   */
  async _downloadCssFile(url, cssDir, assetDir, pageCssCache, pageAssetCache) {
    if (pageCssCache.has(url)) {
      return pageCssCache.get(url);
    }

    // Prevent recursive loops if the same URL reappears during processing
    pageCssCache.set(url, null);

    const maxRetries = 3;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await axios.get(url, {
          responseType: 'arraybuffer',
          timeout: this.config.TIMEOUT_NAVIGATION,
          maxRedirects: 5,
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          }
        });

        const filename = this._extractCssFilename(url);
        const filePath = path.join(cssDir, filename);
        await fs.writeFile(filePath, response.data);

        // Process nested imports and assets referenced from this CSS file
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
          this.logger.debug('CSS', `Retry ${attempt}/${maxRetries} for ${url} after ${delay}ms`);
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
   * @summary Process inline <style> blocks for asset localization.
   * 
   * Scans all <style> elements in the document and rewrites url() references
   * to point to locally downloaded assets.
   * 
   * @param {JSDOM} dom - Document object model.
   * @param {string} cssDir - CSS directory path.
   * @param {string} assetDir - Asset directory path.
   * @param {string} pageUrl - Page URL for resolving relative references.
   * @param {Map<string, Object>} pageCssCache - CSS file cache.
   * @param {Map<string, Object>} pageAssetCache - Asset cache.
   * @returns {Promise<{inlineStyles: number, assets: number}>} Processing statistics.
   * @private
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
      this.logger.info('CSS', `Updated ${modifiedBlocks} inline <style> block(s) with localized assets.`);
    }

    return { inlineStyles: modifiedBlocks, assets: assetsLocalized };
  }

  /**
   * @summary Resolve a potentially relative URL against a base URL.
   * 
   * @param {string} resourceUrl - URL to resolve (may be relative or absolute).
   * @param {string} baseUrl - Base URL for resolution.
   * @returns {string|null} Absolute URL or null if invalid/skippable.
   * @private
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
      this.logger.debug('CSS', `Unable to resolve URL ${resourceUrl} relative to ${baseUrl}`);
      return null;
    }
  }

  /**
   * @summary Extract a sanitized filename for CSS files.
   * 
   * @param {string} url - CSS file URL.
   * @returns {string} Sanitized filename with hash prefix.
   * @private
   */
  _extractCssFilename(url) {
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
   * @summary Sleep for specified milliseconds.
   * @param {number} ms - Milliseconds to sleep.
   * @returns {Promise<void>}
   * @private
   */
  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * @summary Get download statistics.
   * @returns {{stylesheetsDownloaded: number, totalCssAssets: number}} Download counts.
   */
  getStats() {
    return {
      stylesheetsDownloaded: this.cssCount,
      totalCssAssets: this.assetDownloader.assetCount
    };
  }

  /**
   * @summary Reset download statistics.
   */
  reset() {
    this.cssCount = 0;
    this.assetDownloader.reset();
  }
}

module.exports = CssDownloader;
