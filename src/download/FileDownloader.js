/**
 * @fileoverview File Download Orchestrator
 * @module download/FileDownloader
 * @description Downloads and localizes embedded files (PDFs, documents, archives, etc.)
 * 
 * Delegates to specialized components:
 * - FileTypeDetector: Identifies downloadable files
 * - FileDownloadStrategy: Handles download with retry logic
 * - FileNameExtractor: Extracts and sanitizes filenames
 */

const path = require('path');
const fs = require('fs/promises');
const FileTypeDetector = require('./file/FileTypeDetector');
const FileDownloadStrategy = require('./file/FileDownloadStrategy');
const FileNameExtractor = require('./file/FileNameExtractor');

/**
 * @class FileDownloader
 * @classdesc Orchestrates file detection, download, and link rewriting
 */
class FileDownloader {
  /**
   * @param {Config} config - Configuration object
   * @param {Logger} logger - Logger instance
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;

    // Initialize component managers
    this.typeDetector = new FileTypeDetector();
    this.downloadStrategy = new FileDownloadStrategy(config, logger);
    this.nameExtractor = new FileNameExtractor();
  }

  /**
   * Check if a URL is a downloadable file
   * @param {string} url - URL to check
   * @param {string} [linkText=''] - Optional link text
   * @returns {boolean} True if downloadable
   */
  isDownloadableFile(url, linkText = '') {
    return this.typeDetector.isDownloadableFile(url, linkText);
  }

  /**
   * Download all embedded files from a page and rewrite their links
   * @async
   * @param {import('puppeteer').Page} page - Puppeteer page instance
   * @param {string} outputDir - Output directory for downloads
   * @returns {Promise<void>}
   */
  async downloadAndRewriteFiles(page, outputDir) {
    this.logger.info('FILE', 'Identifying and downloading embedded files...');

    // Create files directory
    const filesDir = path.join(outputDir, 'files');
    await fs.mkdir(filesDir, { recursive: true });

    // Extract all file links
    const fileLinks = await page.evaluate(() => {
      const results = [];
      const links = document.querySelectorAll('a[href]');

      links.forEach(link => {
        const href = link.getAttribute('href');
        const text = link.textContent.trim();

        if (href && href.startsWith('http')) {
          results.push({ url: href, text: text });
        }
      });

      return results;
    });

    this.logger.info('FILE', `Found ${fileLinks.length} links to examine...`);

    const urlMap = {};
    let downloadCount = 0;

    // Filter and download files
    for (const [index, linkInfo] of fileLinks.entries()) {
      try {
        const fileUrl = linkInfo.url;

        // Check if downloadable
        if (!this.isDownloadableFile(fileUrl, linkInfo.text)) {
          continue;
        }

        this.logger.info('FILE', `Downloading file: ${linkInfo.text || path.basename(fileUrl)}`);

        // Extract and sanitize filename
        const filename = this.nameExtractor.extractFilename(fileUrl, linkInfo.text, index + 1);
        const localFilePath = path.join(filesDir, filename);
        const relativePath = path.posix.join('files', filename);

        // Download if not already cached
        if (!this.downloadStrategy.hasDownloaded(fileUrl)) {
          const success = await this.downloadStrategy.downloadFileWithRetry(fileUrl, localFilePath);
          if (success) {
            this.downloadStrategy.recordDownload(fileUrl, relativePath);
            downloadCount++;
          }
        }

        if (this.downloadStrategy.hasDownloaded(fileUrl)) {
          urlMap[fileUrl] = this.downloadStrategy.getDownloadedPath(fileUrl);
        }
      } catch (error) {
        this.logger.error('FILE', `Error processing file link ${linkInfo.url}`, error);
      }
    }

    // Rewrite file links in HTML
    if (Object.keys(urlMap).length > 0) {
      this.logger.info('FILE', 'Rewriting file links in the HTML...');
      await page.evaluate(map => {
        document.querySelectorAll('a[href]').forEach(link => {
          if (map[link.href]) {
            link.href = map[link.href];
          }
        });
      }, urlMap);
    }

    this.logger.success('FILE', `Downloaded ${downloadCount} embedded files.`);
  }

  /**
   * Get statistics about downloaded files
   * @returns {Object} Download statistics
   */
  getStats() {
    return this.downloadStrategy.getStats();
  }

  /**
   * Reset download cache
   * @returns {void}
   */
  reset() {
    this.downloadStrategy.reset();
  }
}

module.exports = FileDownloader;

