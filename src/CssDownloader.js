const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const Logger = require('./Logger');

/**
 * Downloads external CSS stylesheets (and their dependencies) and rewrites
 * <link> tags to point to local files. Also localizes assets referenced from
 * CSS such as fonts, background images, and nested @import directives.
 */
class CssDownloader {
  constructor(config, logger = null) {
    this.config = config;
    this.logger = logger || new Logger();

    this.cssCount = 0;
    this.assetCount = 0;
  }

  /** Determine if a link target looks like CSS. */
  isCssFile(url) {
    if (!url || typeof url !== 'string') return false;
    const normalized = url.toLowerCase();
    return normalized.includes('.css') || normalized.includes('stylesheet') || normalized.includes('text/css');
  }

  /**
   * Download all CSS files linked in the HTML and rewrite the DOM to point to
   * the localized versions. Inline <style> blocks are processed to localize
   * referenced assets as well.
   *
   * @param {JSDOM} dom Parsed DOM of the HTML file
   * @param {string} pageDir Directory where the HTML file resides
   * @param {string} pageUrl Absolute URL of the original Notion page
   * @returns {Promise<{stylesheets:number, assets:number, inlineStyles:number, modified:boolean}>}
   */
  async downloadAndRewriteCss(dom, pageDir, pageUrl) {
    const document = dom.window.document;
    const linkElements = Array.from(document.querySelectorAll('link[rel="stylesheet"], link[href*=".css"]'));

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

      if (!this.isCssFile(href)) {
        continue;
      }

      const cssUrl = this._resolveUrl(href, pageUrl);
      if (!cssUrl) {
        continue;
      }

      const beforeCss = this.cssCount;
      const beforeAssets = this.assetCount;

  const cssInfo = await this._downloadCssFile(cssUrl, cssDir, assetDir, pageCssCache, pageAssetCache);
      if (!cssInfo) {
        continue;
      }

      // Update DOM to point to local stylesheet
      link.setAttribute('href', cssInfo.hrefFromHtml);

      stylesheetsDownloaded += this.cssCount - beforeCss;
      assetsLocalized += this.assetCount - beforeAssets;
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

  /** Download a CSS file (with retries) and localize its dependencies. */
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
        const processed = await this._processCssContent(cssText, {
          baseUrl: url,
          cssDir,
          assetDir,
          referenceContext: 'css',
          pageCssCache,
          pageAssetCache
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

  /** Process <style> blocks within the page for inline asset localization. */
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

      const beforeAssets = this.assetCount;
      const processed = await this._processCssContent(cssText, {
        baseUrl: pageUrl,
        cssDir,
        assetDir,
        referenceContext: 'inline',
        pageCssCache,
        pageAssetCache
      });

      assetsLocalized += this.assetCount - beforeAssets;

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
   * Parse a CSS string to localize nested imports and asset URLs.
   * @param {string} cssText Original CSS content
   * @param {{baseUrl:string, cssDir:string, assetDir:string, referenceContext:'css'|'inline'}} options
   * @returns {Promise<{css:string}>}
   */
  async _processCssContent(cssText, options) {
    let modified = cssText;
    modified = await this._rewriteImports(modified, options);
    modified = await this._rewriteAssetUrls(modified, options);
    return { css: modified };
  }

  /** Replace @import rules with localized references. */
  async _rewriteImports(cssText, options) {
    const importRegex = /@import\s+(?:url\()?['"]?([^'")]+)['"]?\)?([^;]*);/gi;
    return await this._replaceAsync(cssText, importRegex, async (match) => {
      const original = match[0];
      const importTarget = match[1];
      const trailing = match[2] || '';

      const resolvedUrl = this._resolveUrl(importTarget, options.baseUrl);
      if (!resolvedUrl) {
        return original;
      }

      const beforeCss = this.cssCount;
      const cssInfo = await this._downloadCssFile(
        resolvedUrl,
        options.cssDir,
        options.assetDir,
        options.pageCssCache,
        options.pageAssetCache
      );
      if (!cssInfo) {
        return original;
      }

      const newCssDownloaded = this.cssCount - beforeCss;
      if (newCssDownloaded > 0) {
        this.logger.debug('CSS', `Processed nested @import -> ${cssInfo.filename}`);
      }

      const referencePath = options.referenceContext === 'inline'
        ? cssInfo.hrefFromHtml
        : `./${cssInfo.filename}`;

      return `@import url("${referencePath}")${trailing};`;
    });
  }

  /** Replace url(...) references with localized assets. */
  async _rewriteAssetUrls(cssText, options) {
    const assetRegex = /url\(\s*(['"]?)([^'")]+)\1\s*\)/gi;
    return await this._replaceAsync(cssText, assetRegex, async (match) => {
      const original = match[0];
      const assetTarget = match[2];

      const resolvedUrl = this._resolveUrl(assetTarget, options.baseUrl);
      if (!resolvedUrl) {
        return original;
      }

  const assetInfo = await this._downloadCssAsset(resolvedUrl, options.assetDir, options.pageAssetCache);
      if (!assetInfo) {
        return original;
      }

      const referencePath = options.referenceContext === 'inline'
        ? assetInfo.hrefFromHtml
        : path.posix.join('assets', assetInfo.filename);

      return `url("${referencePath}")`;
    });
  }

  /** Download a CSS asset (fonts, images) with retry logic. */
  async _downloadCssAsset(url, assetDir, pageAssetCache) {
    if (pageAssetCache.has(url)) {
      return pageAssetCache.get(url);
    }

    pageAssetCache.set(url, null);
    
    const maxRetries = 3;
    const filename = this._extractAssetFilename(url);
    const filePath = path.join(assetDir, filename);

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

        await fs.mkdir(path.dirname(filePath), { recursive: true });
        await fs.writeFile(filePath, response.data);

        const info = {
          filename,
          hrefFromHtml: path.posix.join('css', 'assets', filename)
        };

        pageAssetCache.set(url, info);
        this.assetCount += 1;

        const sizeKB = (response.data.length / 1024).toFixed(2);
        this.logger.debug('CSS', `Localized asset ${filename} (${sizeKB} KB)`);

        return info;

      } catch (error) {
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000;
          this.logger.debug('CSS', `Retry ${attempt}/${maxRetries} for asset ${url} after ${delay}ms`);
          await this._sleep(delay);
        } else {
          this.logger.error('CSS', `Failed downloading asset ${url}: ${error.message}`);
          pageAssetCache.delete(url);
          return null;
        }
      }
    }

    pageAssetCache.delete(url);
    return null;
  }

  /** Helper to resolve relative URLs against a base URL. */
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

  /** Extract a sanitized filename for CSS files. */
  _extractCssFilename(url) {
    try {
      const urlObj = new URL(url);
      let filename = path.basename(urlObj.pathname) || 'stylesheet.css';
      if (!filename.toLowerCase().endsWith('.css')) {
        filename += '.css';
      }
      filename = this._sanitizeFilename(filename, '.css');
      const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 8);
      return `${hash}-${filename}`;
    } catch (error) {
      return `${crypto.createHash('md5').update(url).digest('hex').substring(0, 12)}.css`;
    }
  }

  /** Extract a sanitized filename for CSS assets (fonts, images, etc.). */
  _extractAssetFilename(url) {
    try {
      const urlObj = new URL(url);
      let filename = path.basename(urlObj.pathname) || 'asset';
      const ext = path.extname(filename) || '.bin';
      filename = this._sanitizeFilename(filename, ext);
      const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 8);
      return `${hash}-${filename}`;
    } catch (error) {
      return `${crypto.createHash('md5').update(url).digest('hex').substring(0, 12)}.bin`;
    }
  }

  /** Sanitize filenames for safe filesystem usage. */
  _sanitizeFilename(filename, fallbackExtension = '') {
    let sanitized = decodeURIComponent(filename || '');
    sanitized = sanitized
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
      .replace(/\s+/g, '_')
      .replace(/_{2,}/g, '_')
      .replace(/^[._]+|[._]+$/g, '')
      .trim();

    if (!sanitized) {
      sanitized = `file${fallbackExtension || ''}`;
    }

    if (fallbackExtension && !sanitized.toLowerCase().endsWith(fallbackExtension.toLowerCase())) {
      sanitized += fallbackExtension;
    }

    if (sanitized.length > 200) {
      sanitized = sanitized.substring(0, 200);
    }

    return sanitized;
  }

  /** Utility to perform asynchronous regex replacements. */
  async _replaceAsync(text, regex, replacer) {
    const matches = [];
    let match;
    const clonedRegex = new RegExp(regex.source, regex.flags);
    while ((match = clonedRegex.exec(text)) !== null) {
      matches.push(match);
    }

    if (matches.length === 0) {
      return text;
    }

    let result = '';
    let lastIndex = 0;
    for (const currentMatch of matches) {
      const replacement = await replacer(currentMatch);
      result += text.slice(lastIndex, currentMatch.index) + replacement;
      lastIndex = currentMatch.index + currentMatch[0].length;
    }

    result += text.slice(lastIndex);
    return result;
  }

  _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /** Stats for reporting. */
  getStats() {
    return {
      totalCss: this.cssCount,
      totalCssAssets: this.assetCount
    };
  }

  reset() {
    this.cssCount = 0;
    this.assetCount = 0;
  }
}

module.exports = CssDownloader;
