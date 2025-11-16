const axios = require('axios');
const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');

/**
 * Downloads embedded files (PDFs, code files, documents, etc.) from Notion pages
 */
class FileDownloader {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    this.downloadedFiles = new Map(); // URL -> local path mapping
    this.downloadAttempts = new Map();
    this.maxRetries = 3;
    
    // File extensions to download
    this.fileExtensions = [
      '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
      '.zip', '.rar', '.7z', '.tar', '.gz',
      '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
      '.ipynb', '.json', '.xml', '.csv', '.txt', '.md',
      '.mp4', '.avi', '.mov', '.mp3', '.wav',
      '.jpg', '.jpeg', '.png', '.gif', '.svg'
    ];
    
    // Notion file URL patterns
    this.notionFilePatterns = [
      /notion\.so\/signed\//,
      /s3.*amazonaws\.com.*prod-files-secure/,
      /notion-static\.com/,
      /\/file\//
    ];
  }
  
  /**
   * Sanitize filename to be filesystem-safe
   */
  sanitizeFilename(filename) {
    let sanitized = decodeURIComponent(filename);
    
    sanitized = sanitized
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
      .replace(/^\.+/, '')
      .replace(/\s+/g, '_')
      .replace(/_{2,}/g, '_');
    
    if (sanitized.length > 200 || sanitized.length === 0) {
      const hash = crypto.createHash('md5').update(filename).digest('hex').substring(0, 8);
      const ext = path.extname(sanitized) || '';
      sanitized = `file_${hash}${ext}`;
    }
    
    return sanitized;
  }
  
  /**
   * Check if a URL is a downloadable file
   */
  isDownloadableFile(url, linkText = '') {
    try {
      // Check Notion file patterns
      for (const pattern of this.notionFilePatterns) {
        if (pattern.test(url)) {
          return true;
        }
      }
      
      // Check file extension
      const urlLower = url.toLowerCase();
      for (const ext of this.fileExtensions) {
        if (urlLower.includes(ext)) {
          return true;
        }
      }
      
      // Check link text for common download indicators
      const textLower = linkText.toLowerCase();
      if (textLower.includes('download') || 
          textLower.includes('.pdf') || 
          textLower.includes('.zip') ||
          textLower.includes('file')) {
        return true;
      }
      
      return false;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Download all embedded files from a page and rewrite their links
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
          results.push({
            url: href,
            text: text
          });
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
        
        // Check if this is a downloadable file
        if (!this.isDownloadableFile(fileUrl, linkInfo.text)) {
          continue;
        }
        
        this.logger.info('FILE', `Downloading file: ${linkInfo.text || path.basename(fileUrl)}`);
        
        // Extract filename
        let filename = this._extractFilename(fileUrl, linkInfo.text);
        
        // Sanitize filename
        const sanitizedName = this.sanitizeFilename(filename);
        const localFileName = `${index + 1}-${sanitizedName}`;
        const localFilePath = path.join(filesDir, localFileName);
        const relativePath = path.posix.join('files', localFileName);
        
        // Download if not already downloaded
        if (!this.downloadedFiles.has(fileUrl)) {
          const success = await this._downloadFileWithRetry(fileUrl, localFilePath);
          if (success) {
            this.downloadedFiles.set(fileUrl, relativePath);
            downloadCount++;
          }
        }
        
        if (this.downloadedFiles.has(fileUrl)) {
          urlMap[fileUrl] = this.downloadedFiles.get(fileUrl);
        }
      } catch (error) {
        this.logger.error('FILE', `Error processing file link ${linkInfo.url}`, error);
      }
    }
    
    // Rewrite file links in the HTML
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
   * Extract filename from URL or link text
   */
  _extractFilename(url, linkText) {
    // Try to get filename from URL
    try {
      const urlObj = new URL(url);
      let filename = path.basename(urlObj.pathname);
      
      // If filename is empty or just a UUID, try link text
      if (!filename || filename.length < 3 || /^[a-f0-9-]{32,}$/.test(filename)) {
        if (linkText && linkText.length > 0 && linkText.length < 100) {
          // Use link text as filename
          filename = linkText;
          
          // Add extension if missing
          if (!path.extname(filename)) {
            // Try to guess extension from URL
            const urlLower = url.toLowerCase();
            for (const ext of this.fileExtensions) {
              if (urlLower.includes(ext)) {
                filename += ext;
                break;
              }
            }
          }
        }
      }
      
      return filename || 'downloaded_file';
    } catch (error) {
      return 'downloaded_file';
    }
  }
  
  /**
   * Download a file with retry logic
   */
  async _downloadFileWithRetry(url, localPath) {
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
        timeout: 60000, // 1 minute timeout for large files
        maxRedirects: 5,
        validateStatus: (status) => status < 400,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      
      await fs.mkdir(path.dirname(localPath), { recursive: true });
      await fs.writeFile(localPath, response.data);
      
      const sizeKB = (response.data.length / 1024).toFixed(2);
      this.logger.info('FILE', `Success: ${path.basename(localPath)} (${sizeKB} KB)`);
      this.downloadAttempts.delete(attemptKey);
      return true;
      
    } catch (error) {
      this.downloadAttempts.set(attemptKey, currentAttempts + 1);
      
      if (currentAttempts + 1 < this.maxRetries) {
        this.logger.warn('FILE', `Retry ${currentAttempts + 1}/${this.maxRetries} for ${path.basename(localPath)}`);
        await new Promise(resolve => setTimeout(resolve, 1000 * (currentAttempts + 1)));
        return await this._downloadFileWithRetry(url, localPath);
      } else {
        this.logger.error('FILE', `Failed to download ${path.basename(localPath)}: ${error.message}`);
        return false;
      }
    }
  }
  
  /**
   * Get statistics about downloaded files
   */
  getStats() {
    return {
      totalFiles: this.downloadedFiles.size
    };
  }
}

module.exports = FileDownloader;
