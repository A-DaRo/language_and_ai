const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const FileSystemUtils = require('../../utils/FileSystemUtils');

/**
 * Downloads CSS-referenced assets (fonts, background images) with retry logic and caching.
 * 
 * @classdesc Handles the actual HTTP fetching of CSS-referenced resources such as fonts,
 * background images, and other media. Implements exponential backoff retry strategy
 * to handle transient network failures. Uses per-page caching to prevent duplicate
 * downloads within a single page context.
 * 
 * @see CssDownloader
 * @see CssContentProcessor
 */
class CssAssetDownloader {
  /**
   * @param {Object} config - Configuration object with timeout settings.
   * @param {Logger} logger - Logger instance for debugging and error reporting.
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.assetCount = 0;
  }

  /**
   * @summary Download a CSS asset with retry logic and caching.
   * 
   * Attempts to download a CSS-referenced asset (font, image, etc.) with exponential
   * backoff retry on failure. Uses cache to prevent duplicate downloads.
   * 
   * @param {string} url - URL of the asset to download.
   * @param {string} assetDir - Directory where asset should be saved.
   * @param {Map<string, Object>} cache - Per-page cache to prevent duplicate downloads.
   * @returns {Promise<{filename: string, hrefFromHtml: string}|null>} Asset info or null on failure.
   * 
   * @example
   * const assetInfo = await downloader.downloadCssAsset(
   *   'https://fonts.example.com/font.woff2',
   *   '/path/to/css/assets',
   *   pageAssetCache
   * );
   * // Returns: { filename: '8a3f4b2c-font.woff2', hrefFromHtml: 'css/assets/8a3f4b2c-font.woff2' }
   */
  async downloadCssAsset(url, assetDir, cache) {
    if (cache.has(url)) {
      return cache.get(url);
    }

    cache.set(url, null);
    
    const filename = this._extractAssetFilename(url);
    const filePath = path.join(assetDir, filename);

    try {
      const data = await this._fetchWithRetries(url, 3);
      if (!data) {
        cache.delete(url);
        return null;
      }

      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, data);

      const info = {
        filename,
        hrefFromHtml: path.posix.join('css', 'assets', filename)
      };

      cache.set(url, info);
      this.assetCount += 1;

      const sizeKB = (data.length / 1024).toFixed(2);
      this.logger.debug('CSS-ASSET', `Downloaded ${filename} (${sizeKB} KB)`);

      return info;

    } catch (error) {
      this.logger.error('CSS-ASSET', `Failed downloading asset ${url}: ${error.message}`);
      cache.delete(url);
      return null;
    }
  }

  /**
   * @summary Fetch resource with exponential backoff retry strategy.
   * 
   * Implements retry logic with exponential backoff to handle transient network
   * failures gracefully. Waits 2^attempt * 1000ms between retries.
   * 
   * @param {string} url - URL to fetch.
   * @param {number} maxRetries - Maximum number of retry attempts.
   * @returns {Promise<Buffer|null>} Downloaded data or null after all retries exhausted.
   * @throws {Error} Only after exhausting all retry attempts.
   * @internal
   */
  async _fetchWithRetries(url, maxRetries = 3) {
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

        return response.data;

      } catch (error) {
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000;
          this.logger.debug('CSS-ASSET', `Retry ${attempt}/${maxRetries} for ${url} after ${delay}ms`);
          await this._sleep(delay);
        } else {
          throw error;
        }
      }
    }
    return null;
  }

  /**
   * @summary Extract a sanitized filename for CSS assets.
   * 
   * Generates a filename with hash prefix to prevent collisions and sanitizes
   * the original filename to be filesystem-safe.
   * 
   * @param {string} url - Asset URL.
   * @returns {string} Sanitized filename with hash prefix.
   * @private
   */
  _extractAssetFilename(url) {
    try {
      const urlObj = new URL(url);
      let filename = path.basename(urlObj.pathname) || 'asset';
      const ext = path.extname(filename) || '.bin';
      filename = FileSystemUtils.sanitizeFilename(filename, ext);
      const hash = crypto.createHash('md5').update(url).digest('hex').substring(0, 8);
      return `${hash}-${filename}`;
    } catch (error) {
      return `${crypto.createHash('md5').update(url).digest('hex').substring(0, 12)}.bin`;
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
   * @returns {{totalCssAssets: number}} Asset download count.
   */
  getStats() {
    return {
      totalCssAssets: this.assetCount
    };
  }

  /**
   * @summary Reset download statistics.
   */
  reset() {
    this.assetCount = 0;
  }
}

module.exports = CssAssetDownloader;
