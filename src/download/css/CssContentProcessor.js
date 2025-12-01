const path = require('path');

/**
 * Parses CSS content to localize @import rules and url() references.
 * Handles both external CSS files and inline <style> blocks, rewriting
 * all remote resource references to point to local files.
 * 
 * @classdesc This class is responsible for the text-level manipulation of CSS content.
 * It identifies and rewrites @import directives and url() references to ensure all
 * CSS dependencies point to locally downloaded files. Works in conjunction with
 * CssAssetDownloader to fetch the actual resources.
 * 
 * @see CssDownloader
 * @see CssAssetDownloader
 */
class CssContentProcessor {
  /**
   * @param {Logger} logger - Logger instance for debugging and error reporting.
   */
  constructor(logger) {
    this.logger = logger;
  }

  /**
   * @summary Process CSS content to localize all external references.
   * 
   * Main entry point for CSS content transformation. Handles both @import rules
   * and url() asset references, rewriting them to point to local files.
   * 
   * @param {string} cssText - Original CSS content to process.
   * @param {Object} options - Processing configuration options.
   * @param {string} options.baseUrl - Base URL for resolving relative references.
   * @param {string} options.cssDir - Directory containing CSS files.
   * @param {string} options.assetDir - Directory for CSS assets (fonts, images).
   * @param {'css'|'inline'} options.referenceContext - Context determines path resolution strategy.
   * @param {Map<string, Object>} options.pageCssCache - Cache to prevent duplicate CSS downloads.
   * @param {Map<string, Object>} options.pageAssetCache - Cache to prevent duplicate asset downloads.
   * @param {Function} options.downloadCssFile - Callback to download nested CSS files.
   * @param {Function} options.downloadCssAsset - Callback to download CSS assets.
   * @returns {Promise<{css: string}>} Processed CSS with localized references.
   * 
   * @example
   * const result = await processor.processCssContent(cssText, {
   *   baseUrl: 'https://example.com/style.css',
   *   cssDir: '/path/to/css',
   *   assetDir: '/path/to/css/assets',
   *   referenceContext: 'css',
   *   pageCssCache: new Map(),
   *   pageAssetCache: new Map(),
   *   downloadCssFile: (url) => downloader.downloadCssFile(url),
   *   downloadCssAsset: (url) => downloader.downloadCssAsset(url)
   * });
   */
  async processCssContent(cssText, options) {
    let modified = cssText;
    modified = await this.rewriteImports(modified, options);
    modified = await this.rewriteAssetUrls(modified, options);
    return { css: modified };
  }

  /**
   * @summary Replace @import rules with localized CSS file references.
   * 
   * Identifies all @import directives in CSS and rewrites them to reference
   * locally downloaded CSS files. Triggers download of nested CSS files if needed.
   * 
   * @param {string} cssText - CSS content to process.
   * @param {Object} options - Processing options (see processCssContent).
   * @returns {Promise<string>} CSS with rewritten @import rules.
   * @private
   */
  async rewriteImports(cssText, options) {
    const importRegex = /@import\s+(?:url\()?['"]?([^'")]+)['"]?\)?([^;]*);/gi;
    return await this._replaceAsync(cssText, importRegex, async (match) => {
      const original = match[0];
      const importTarget = match[1];
      const trailing = match[2] || '';

      const resolvedUrl = this._resolveUrl(importTarget, options.baseUrl);
      if (!resolvedUrl) {
        return original;
      }

      const cssInfo = await options.downloadCssFile(
        resolvedUrl,
        options.cssDir,
        options.assetDir,
        options.pageCssCache,
        options.pageAssetCache
      );
      
      if (!cssInfo) {
        return original;
      }

      const referencePath = options.referenceContext === 'inline'
        ? cssInfo.hrefFromHtml
        : `./${cssInfo.filename}`;

      this.logger.debug('CSS-PROCESSOR', `Rewrote @import: ${importTarget} → ${referencePath}`);
      return `@import url("${referencePath}")${trailing};`;
    });
  }

  /**
   * @summary Replace url(...) references with localized asset paths.
   * 
   * Identifies all url() references to external assets (fonts, background images)
   * and rewrites them to point to locally downloaded files.
   * 
   * @param {string} cssText - CSS content to process.
   * @param {Object} options - Processing options (see processCssContent).
   * @returns {Promise<string>} CSS with rewritten url() references.
   * @private
   */
  async rewriteAssetUrls(cssText, options) {
    const assetRegex = /url\(\s*(['"]?)([^'")]+)\1\s*\)/gi;
    return await this._replaceAsync(cssText, assetRegex, async (match) => {
      const original = match[0];
      const assetTarget = match[2];

      const resolvedUrl = this._resolveUrl(assetTarget, options.baseUrl);
      if (!resolvedUrl) {
        return original;
      }

      const assetInfo = await options.downloadCssAsset(
        resolvedUrl,
        options.assetDir,
        options.pageAssetCache
      );
      
      if (!assetInfo) {
        return original;
      }

      const referencePath = options.referenceContext === 'inline'
        ? assetInfo.hrefFromHtml
        : path.posix.join('assets', assetInfo.filename);

      return `url("${referencePath}")`;
    });
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
      this.logger.debug('CSS-PROCESSOR', `Unable to resolve URL ${resourceUrl} relative to ${baseUrl}`);
      return null;
    }
  }

  /**
   * @summary Utility to perform asynchronous regex replacements.
   * 
   * Since JavaScript's String.replace() doesn't support async replacers,
   * this method collects all matches and processes them sequentially.
   * 
   * @param {string} text - Text to process.
   * @param {RegExp} regex - Regular expression to match.
   * @param {Function} replacer - Async function to generate replacement text.
   * @returns {Promise<string>} Text with all replacements applied.
   * @internal
   */
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
}

module.exports = CssContentProcessor;
