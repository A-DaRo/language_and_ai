const axios = require('axios');
const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');

/**
 * Downloads and manages assets (images, files, etc.)
 */
class AssetDownloader {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.downloadedAssets = new Map(); // URL -> local path mapping
    this.downloadAttempts = new Map(); // Track retry attempts
    this.maxRetries = 3;
  }
  
  /**
   * Sanitize filename to be filesystem-safe
   * Handles complex URLs with special characters
   */
  sanitizeFilename(filename) {
    // Decode URI components
    let sanitized = decodeURIComponent(filename);
    
    // Remove or replace problematic characters
    sanitized = sanitized
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_') // Replace invalid chars with underscore
      .replace(/^\.+/, '') // Remove leading dots
      .replace(/\s+/g, '_') // Replace spaces with underscores
      .replace(/_{2,}/g, '_'); // Replace multiple underscores with single
    
    // If filename is too long or becomes empty, generate a hash-based name
    if (sanitized.length > 200 || sanitized.length === 0) {
      const hash = crypto.createHash('md5').update(filename).digest('hex').substring(0, 8);
      const ext = path.extname(sanitized) || '.jpg';
      sanitized = `asset_${hash}${ext}`;
    }
    
    return sanitized;
  }
  
  /**
   * Download all images and assets from a page and rewrite their paths
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
        const sanitizedName = this.sanitizeFilename(imageName);
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
