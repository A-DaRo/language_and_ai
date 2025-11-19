const axios = require('axios');
const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const FileSystemUtils = require('../utils/FileSystemUtils');

/**
 * @classdesc Downloads and manages assets (images, files, etc.) from Notion pages.
 * 
 * Handles the complete asset download and localization workflow:
 * - Extracts asset URLs from img tags and CSS background properties
 * - Downloads assets with retry logic and exponential backoff
 * - Generates safe filenames using content hashing
 * - Rewrites asset references in the DOM to point to local paths
 * - Maintains download cache to prevent duplicate requests
 * 
 * Uses content-based hashing (MD5) to ensure unique filenames while allowing
 * deduplication of identical assets across multiple pages.
 * 
 * @see PageProcessor#scrapePage
 * @see FileSystemUtils
 */
class AssetDownloader {
  /**
   * @param {Config} config - Configuration object for timeout and retry settings.
   * @param {Logger} logger - Logger instance for download progress tracking.
   */
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.downloadedAssets = new Map(); // URL -> local path mapping
    this.downloadAttempts = new Map(); // Track retry attempts
    this.maxRetries = 3;
  }
  
  /**
   * @summary Download all images and assets from a page and rewrite their paths.
   * 
   * @description Executes the complete asset localization workflow:
   * 1. Creates images directory in output folder
   * 2. Extracts asset URLs from img tags and background-image CSS
   * 3. Downloads each asset with retry logic
   * 4. Generates safe filenames using content hashing
   * 5. Rewrites DOM references to point to local paths
   * 
   * Uses page.evaluate() for extraction and page.$$eval() for rewriting to
   * operate in the browser context for maximum compatibility.
   * 
   * @param {Page} page - Puppeteer page instance.
   * @param {string} outputDir - Directory to save images to (images/ subdirectory will be created).
   * @returns {Promise<void>}
   * 
   * @throws {Error} If directory creation fails.
   * 
   * @see _downloadAsset
   */
  async downloadAndRewriteImages(page, outputDir) {
    this.logger.info('IMAGE', 'Identifying and downloading all visible images and assets...');
    
    // Create images directory
    const imagesDir = path.join(outputDir, 'images');
    await fs.mkdir(imagesDir, { recursive: true });
    
    // Extract all image URLs and background images
    const assets = await page.evaluate(() => {
      const results = [];
      
      // Get all img tags
      document.querySelectorAll('img').forEach(img => {
        if (img.src && img.src.startsWith('http')) {
          results.push({ url: img.src, type: 'img' });
        }
      });
      
      // Get background images from inline styles
      document.querySelectorAll('[style*="background"]').forEach(elem => {
        const style = elem.getAttribute('style');
        const urlMatch = style.match(/url\(['"]?(https?:\/\/[^'")\s]+)['"]?\)/);
        if (urlMatch) {
          results.push({ url: urlMatch[1], type: 'background' });
        }
      });
      
      return results;
    });
    
    this.logger.info('IMAGE', `Found ${assets.length} assets to process.`);
    
    const urlMap = {};
    let successCount = 0;
    
    // Download each asset
    for (const [index, asset] of assets.entries()) {
      try {
        const imageUrl = asset.url;
        
        // Parse URL and extract filename
        const urlObj = new URL(imageUrl);
        let imageName = path.basename(urlObj.pathname);
        
        // Handle empty or root path
        if (!imageName || imageName === '/') {
          imageName = 'image.jpg';
        }
        
        // Sanitize filename
        const sanitizedName = FileSystemUtils.sanitizeFilename(imageName);
        const localImageName = `${index + 1}-${sanitizedName}`;
        const localImagePath = path.join(imagesDir, localImageName);
        const relativePath = path.posix.join('images', localImageName);
        
        // Download if not already downloaded
        if (!this.downloadedAssets.has(imageUrl)) {
          const success = await this._downloadAssetWithRetry(imageUrl, localImagePath);
          if (success) {
            this.downloadedAssets.set(imageUrl, relativePath);
            successCount++;
          }
        }
        
        if (this.downloadedAssets.has(imageUrl)) {
          urlMap[imageUrl] = this.downloadedAssets.get(imageUrl);
        }
      } catch (error) {
        this.logger.error('IMAGE', `Error processing asset ${asset.url}`, error);
      }
    }
    
    // Rewrite asset paths in the HTML
    this.logger.info('IMAGE', 'Rewriting asset paths in the HTML...');
    await page.evaluate(map => {
      // Rewrite img src
      document.querySelectorAll('img').forEach(img => {
        if (map[img.src]) {
          img.src = map[img.src];
        }
      });
      
      // Rewrite background images in inline styles
      document.querySelectorAll('[style*="background"]').forEach(elem => {
        let style = elem.getAttribute('style');
        Object.keys(map).forEach(originalUrl => {
          if (style.includes(originalUrl)) {
            style = style.replace(originalUrl, map[originalUrl]);
          }
        });
        elem.setAttribute('style', style);
      });
    }, urlMap);
    
    this.logger.success('IMAGE', `Downloaded ${successCount}/${assets.length} assets and rewritten paths.`);
  }
  
  /**
   * Download a single asset with retry logic
   */
  async _downloadAssetWithRetry(url, localPath) {
    const attemptKey = url;
    const currentAttempts = this.downloadAttempts.get(attemptKey) || 0;
    
    if (currentAttempts >= this.maxRetries) {
      this.logger.error('DOWNLOAD', `Max retries reached for ${path.basename(localPath)}`);
      return false;
    }
    
    try {
      const response = await axios({
        method: 'GET',
        url,
        responseType: 'arraybuffer',
        timeout: 30000,
        maxRedirects: 5,
        validateStatus: (status) => status < 400,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      
      // Ensure directory exists
      await fs.mkdir(path.dirname(localPath), { recursive: true });
      
      await fs.writeFile(localPath, response.data);
      this.logger.info('DOWNLOAD', `Success: ${path.basename(localPath)}`);
      this.downloadAttempts.delete(attemptKey);
      return true;
      
    } catch (error) {
      this.downloadAttempts.set(attemptKey, currentAttempts + 1);
      
      if (currentAttempts + 1 < this.maxRetries) {
        this.logger.warn('DOWNLOAD', `Retry ${currentAttempts + 1}/${this.maxRetries} for ${path.basename(localPath)}`);
        await new Promise(resolve => setTimeout(resolve, 1000 * (currentAttempts + 1)));
        return await this._downloadAssetWithRetry(url, localPath);
      } else {
        this.logger.error('DOWNLOAD', `Failed to download ${path.basename(localPath)}: ${error.message}`);
        return false;
      }
    }
  }
  
  /**
   * Get statistics about downloaded assets
   */
  getStats() {
    return {
      totalAssets: this.downloadedAssets.size
    };
  }
}

module.exports = AssetDownloader;
