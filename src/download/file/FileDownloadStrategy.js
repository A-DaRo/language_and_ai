/**
 * @fileoverview File Download Strategy for FileDownloader
 * @module download/file/FileDownloadStrategy
 * @description Handles file download with retry logic and caching
 */

const axios = require('axios');
const fs = require('fs/promises');
const path = require('path');

const MIME_EXTENSION_MAP = {
  'application/pdf': 'pdf',
  'application/zip': 'zip',
  'application/json': 'json',
  'application/xml': 'xml',
  'application/msword': 'doc',
  'application/vnd.ms-excel': 'xls',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
  'text/plain': 'txt',
  'text/html': 'html',
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/gif': 'gif',
  'image/svg+xml': 'svg',
  'audio/mpeg': 'mp3',
  'video/mp4': 'mp4',
  'application/octet-stream': 'bin'
};

/**
 * @class FileDownloadStrategy
 * @classdesc Manages file downloads with retry logic and attempt tracking
 */
class FileDownloadStrategy {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.downloadedFiles = new Map(); // URL -> local path mapping
    this.downloadAttempts = new Map();
    this.maxRetries = 3;
  }

  /**
   * Download a file with retry logic
   * @async
   * @param {string} url - File URL to download
   * @param {string} localPath - Local file path for saving
   * @returns {Promise<string|null>} Saved file path or null on failure
   */
  async downloadFileWithRetry(url, localPath) {
    const attemptKey = url;
    const currentAttempts = this.downloadAttempts.get(attemptKey) || 0;

    if (currentAttempts >= this.maxRetries) {
      this.logger.error('FILE', `Max retries reached for ${path.basename(localPath)}`);
      return false;
    }

    try {
      const response = await axios({
        method: 'GET',
        url,
        responseType: 'arraybuffer',
        timeout: 60000,
        maxRedirects: 5,
        validateStatus: (status) => status < 400,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });

      const finalLocalPath = this._resolvePathWithExtension(localPath, response.headers['content-type']);
      await fs.mkdir(path.dirname(finalLocalPath), { recursive: true });
      await fs.writeFile(finalLocalPath, response.data);

      const sizeKB = (response.data.length / 1024).toFixed(2);
      this.logger.info('FILE', `Success: ${path.basename(finalLocalPath)} (${sizeKB} KB)`);
      this.downloadAttempts.delete(attemptKey);
      return finalLocalPath;

    } catch (error) {
      this.downloadAttempts.set(attemptKey, currentAttempts + 1);

      if (currentAttempts + 1 < this.maxRetries) {
        this.logger.warn('FILE', `Retry ${currentAttempts + 1}/${this.maxRetries} for ${path.basename(localPath)}`);
        await new Promise(resolve => setTimeout(resolve, 1000 * (currentAttempts + 1)));
        return await this.downloadFileWithRetry(url, localPath);
      } else {
        this.logger.error('FILE', `Failed to download ${path.basename(localPath)}: ${error.message}`);
        return null;
      }
    }
  }

  /**
   * Record a downloaded file in the cache
   * @param {string} url - Source URL
   * @param {string} relativePath - Relative path to downloaded file
   * @returns {void}
   */
  recordDownload(url, relativePath) {
    this.downloadedFiles.set(url, relativePath);
  }

  /**
   * Check if URL was previously downloaded
   * @param {string} url - File URL
   * @returns {boolean} True if already downloaded
   */
  hasDownloaded(url) {
    return this.downloadedFiles.has(url);
  }

  /**
   * Get local path for downloaded URL
   * @param {string} url - File URL
   * @returns {string|undefined} Local path or undefined
   */
  getDownloadedPath(url) {
    return this.downloadedFiles.get(url);
  }

  /**
   * Get download statistics
   * @returns {Object} Statistics about downloads
   */
  getStats() {
    return {
      totalFiles: this.downloadedFiles.size
    };
  }

  /**
   * Reset download cache
   * @returns {void}
   */
  reset() {
    this.downloadedFiles.clear();
    this.downloadAttempts.clear();
  }

  _resolvePathWithExtension(localPath, contentType) {
    if (path.extname(localPath)) {
      return localPath;
    }

    const extension = this._extensionFromContentType(contentType);
    if (extension) {
      return `${localPath}.${extension}`;
    }
    return localPath;
  }

  _extensionFromContentType(contentType) {
    if (!contentType) {
      return null;
    }

    const [mimeType] = contentType.split(';');
    const cleaned = mimeType.trim().toLowerCase();
    return MIME_EXTENSION_MAP[cleaned] || null;
  }
}

module.exports = FileDownloadStrategy;
