/**
 * @fileoverview File Download Orchestrator
 * @module download/FileDownloader
 * @description Downloads and localizes embedded files (PDFs, documents, archives, etc.)
 * 
 * Delegates to specialized components:
 * - FileTypeDetector: Identifies downloadable files (whitelist policy)
 * - FileDownloadStrategy: Handles download with retry logic
 * - FileNameExtractor: Extracts and sanitizes filenames with extension enforcement
 * - HtmlFacade: Abstraction layer for DOM operations (browser/server contexts)
 * 
 * @design HIDDEN FILE HANDLING
 * Notion often wraps downloadable files in interactive elements with `display: contents`
 * or within nested spans. The improved `_processHiddenFiles` method:
 * 1. Finds elements with `data-popup-origin="true"` attribute
 * 2. Traverses up the DOM to find a clickable parent (with cursor:pointer/zoom-in)
 * 3. Uses programmatic click or fallback to element.click() if needed
 * 4. Intercepts download URLs from new tabs or request interception
 */

const path = require('path');
const fs = require('fs/promises');
const FileTypeDetector = require('./file/FileTypeDetector');
const FileDownloadStrategy = require('./file/FileDownloadStrategy');
const FileNameExtractor = require('./file/FileNameExtractor');
const { HtmlFacadeFactory } = require('../html');

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

    // Create HtmlFacade for DOM operations
    const facade = HtmlFacadeFactory.forPage(page);

    // Extract all file links using HtmlFacade
    const linkElements = await facade.query('a[href]');
    const fileLinks = [];
    
    for (const link of linkElements) {
      const href = await facade.getAttribute(link, 'href');
      const text = await facade.getTextContent(link);
      
      if (href && href.startsWith('http')) {
        fileLinks.push({ url: href, text: text.trim() });
      }
    }

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

        // Extract and sanitize filename with extension enforcement
        const filename = this.nameExtractor.extractFilename(fileUrl, linkInfo.text, index + 1, 'files');
        const localFilePath = path.join(filesDir, filename);

        // Download if not already cached
        if (!this.downloadStrategy.hasDownloaded(fileUrl)) {
          const savedPath = await this.downloadStrategy.downloadFileWithRetry(fileUrl, localFilePath);
          if (savedPath) {
            const savedRelativePath = path.posix.join('files', path.basename(savedPath));
            this.downloadStrategy.recordDownload(fileUrl, savedRelativePath);
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

    // Process hidden files (e.g. in divs)
    try {
      const hiddenCount = await this._processHiddenFiles(page, filesDir);
      downloadCount += hiddenCount;
    } catch (error) {
      this.logger.error('FILE', 'Error processing hidden files', error);
    }

    // Rewrite file links in HTML using HtmlFacade
    if (Object.keys(urlMap).length > 0) {
      this.logger.info('FILE', 'Rewriting file links in the HTML...');
      const allLinks = await facade.query('a[href]');
      
      for (const link of allLinks) {
        const href = await facade.getAttribute(link, 'href');
        if (urlMap[href]) {
          await facade.setAttribute(link, 'href', urlMap[href]);
        }
      }
    }

    this.logger.success('FILE', `Downloaded ${downloadCount} embedded files.`);
  }

  /**
   * Process hidden file links (e.g. in divs) by clicking and intercepting requests.
   * 
   * @description Improved selector strategy that handles:
   * - Elements with `display: contents` that aren't directly clickable
   * - Elements obscured by overlays or nested in non-clickable wrappers
   * - Puppeteer's "Node is not clickable" errors
   * 
   * Algorithm:
   * 1. Find all `div[data-popup-origin="true"]` elements
   * 2. For each element, traverse up DOM to find clickable parent
   * 3. Click the clickable parent to trigger file download
   * 4. Intercept the download URL via request interception or new tab
   * 
   * @param {import('puppeteer').Page} page - Puppeteer page instance
   * @param {string} filesDir - Directory to save files
   * @returns {Promise<number>} Number of files downloaded
   */
  async _processHiddenFiles(page, filesDir) {
    let downloadCount = 0;
    
    // Create HtmlFacade for DOM operations
    const facade = HtmlFacadeFactory.forPage(page);
    
    // Selector for hidden file elements using HtmlFacade
    const hiddenElements = await facade.query('div[data-popup-origin="true"]');
    
    if (hiddenElements.length > 0) {
      this.logger.info('FILE', `Found ${hiddenElements.length} potential hidden file elements. Scanning...`);
    }

    for (const element of hiddenElements) {
      try {
        const text = (await facade.getTextContent(element)).trim();
        
        // Check if text implies a file (e.g. "slides.pdf")
        const hasExtension = ['.pdf', '.zip', '.docx', '.pptx', '.xlsx'].some(ext => text.toLowerCase().includes(ext));
        
        if (!hasExtension) continue;

        this.logger.info('FILE', `Processing hidden file candidate: ${text}`);

        let downloadUrl = null;
        
        // 1. Listener for new targets (tabs/windows)
        const targetHandler = async (target) => {
            try {
              const opener = target.opener();
              if (opener) {
                  const openerPage = await opener.page();
                  if (openerPage === page) {
                      const newPage = await target.page();
                      if (newPage) {
                          downloadUrl = newPage.url();
                          await newPage.close();
                      }
                  }
              }
            } catch (e) { /* Ignore target errors */ }
        };
        page.browser().on('targetcreated', targetHandler);

        // 2. Listener for requests (same tab)
        await page.setRequestInterception(true);
        const requestHandler = (request) => {
            if (this.typeDetector.isDownloadableFile(request.url())) {
                downloadUrl = request.url();
                request.abort(); // Prevent browser download
            } else {
                request.continue();
            }
        };
        page.on('request', requestHandler);

        try {
            // Improved click strategy: find clickable parent element
            // Unwrap PuppeteerHtmlElement to get raw ElementHandle for click operations
            const rawElement = element.handle;
            const clickSuccess = await this._clickElementOrParent(page, rawElement);
            
            if (!clickSuccess) {
                this.logger.debug('FILE', `Could not click element for: ${text}`);
                continue;
            }
            
            // Wait for reaction (request or new tab)
            await new Promise(resolve => setTimeout(resolve, 3000));
        } catch (e) {
            this.logger.debug('FILE', `Interaction error: ${e.message}`);
        } finally {
            page.browser().off('targetcreated', targetHandler);
            page.off('request', requestHandler);
            await page.setRequestInterception(false);
        }

        if (downloadUrl) {
             this.logger.info('FILE', `Captured hidden download URL: ${downloadUrl}`);
             
             const filename = this.nameExtractor.extractFilename(downloadUrl, text, 'files');
             const localFilePath = path.join(filesDir, filename);
             
             if (!this.downloadStrategy.hasDownloaded(downloadUrl)) {
                const savedPath = await this.downloadStrategy.downloadFileWithRetry(downloadUrl, localFilePath);
                if (savedPath) {
                    const savedRelativePath = path.posix.join('files', path.basename(savedPath));
                    this.downloadStrategy.recordDownload(downloadUrl, savedRelativePath);
                    downloadCount++;
                    
                    // Rewrite DOM: Replace div with anchor
                    // Use rawElement (unwrapped handle) for page.evaluate
                    await page.evaluate((el, newHref, newText) => {
                        const link = document.createElement('a');
                        link.href = newHref;
                        link.textContent = newText;
                        link.style.display = 'block';
                        link.style.fontWeight = 'bold';
                        link.setAttribute('data-restored-file', 'true');
                        
                        // Try to replace the parent wrapper if it exists and is just a wrapper
                        if (el.parentNode && el.parentNode.tagName === 'DIV' && el.parentNode.children.length === 1) {
                           el.parentNode.replaceWith(link);
                        } else {
                           el.replaceWith(link);
                        }
                    }, element.handle, savedRelativePath, text);
                }
             }
        }
      } catch (error) {
        this.logger.error('FILE', `Error processing hidden element`, error);
      }
    }
    return downloadCount;
  }

  /**
   * Click an element or traverse up to find a clickable parent.
   * 
   * @description Handles Puppeteer's "Node is not clickable" error by:
   * 1. First attempting to find a clickable parent with cursor styles
   * 2. Scrolling the element into view
   * 3. Using programmatic click as fallback
   * 
   * @private
   * @param {import('puppeteer').Page} page - Puppeteer page instance
   * @param {import('puppeteer').ElementHandle} element - Element to click
   * @returns {Promise<boolean>} True if click was successful
   */
  async _clickElementOrParent(page, element) {
    try {
      // Strategy 1: Find clickable parent element
      const clickableTarget = await page.evaluateHandle((el) => {
        // Look for parent with clickable cursor styles
        const clickableCursors = ['pointer', 'zoom-in', 'zoom-out', 'grab'];
        let current = el;
        
        // Traverse up to 5 levels looking for clickable parent
        for (let i = 0; i < 5 && current; i++) {
          const style = window.getComputedStyle(current);
          const cursor = style.cursor;
          const display = style.display;
          
          // Skip display:contents elements (they're not clickable)
          if (display === 'contents') {
            current = current.parentElement;
            continue;
          }
          
          // Check if this element has a clickable cursor
          if (clickableCursors.includes(cursor)) {
            return current;
          }
          
          // Check for interactive elements
          if (current.tagName === 'BUTTON' || 
              current.tagName === 'A' || 
              current.getAttribute('role') === 'button' ||
              current.onclick) {
            return current;
          }
          
          current = current.parentElement;
        }
        
        // Fallback: return immediate parent or element itself
        return el.parentElement || el;
      }, element);

      // Strategy 2: Scroll into view and click
      await page.evaluate((target) => {
        target.scrollIntoView({ block: 'center', behavior: 'instant' });
      }, clickableTarget);
      
      // Small delay for scroll to complete
      await new Promise(resolve => setTimeout(resolve, 100));

      // Strategy 3: Try standard click first
      try {
        await clickableTarget.click({ delay: 50 });
        return true;
      } catch (clickError) {
        // Strategy 4: Fallback to programmatic click
        this.logger.debug('FILE', `Standard click failed, trying programmatic click: ${clickError.message}`);
        
        await page.evaluate((target) => {
          // Dispatch click event programmatically
          const event = new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window
          });
          target.dispatchEvent(event);
        }, clickableTarget);
        
        return true;
      }
    } catch (error) {
      this.logger.debug('FILE', `All click strategies failed: ${error.message}`);
      return false;
    }
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

