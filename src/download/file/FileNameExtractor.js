/**
 * @fileoverview File Naming Strategy for FileDownloader
 * @module download/file/FileNameExtractor
 * @description Extracts and sanitizes filenames from URLs and link text
 */

const path = require('path');
const FileSystemUtils = require('../../utils/FileSystemUtils');

/**
 * @class FileNameExtractor
 * @classdesc Extracts meaningful filenames from URLs and link text
 */
class FileNameExtractor {
  constructor() {
    // File extensions for fallback detection
    this.fileExtensions = [
      '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
      '.zip', '.rar', '.7z', '.tar', '.gz',
      '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
      '.ipynb', '.json', '.xml', '.txt', '.md',
      '.mp4', '.avi', '.mov', '.mp3', '.wav',
      '.jpg', '.jpeg', '.png', '.gif', '.svg'
    ];
  }

  /**
   * Extract filename from URL or link text
   * @param {string} url - Source URL
   * @param {string} [linkText=''] - Optional link text
   * @param {number} [index=0] - Optional index for uniqueness
   * @returns {string} Sanitized filename
   */
  extractFilename(url, linkText = '', index = 0) {
    try {
      const urlObj = new URL(url);
      let filename = path.basename(urlObj.pathname);

      // If filename is empty or UUID-like, try link text
      if (!filename || filename.length < 3 || /^[a-f0-9-]{32,}$/.test(filename)) {
        if (linkText && linkText.length > 0 && linkText.length < 100) {
          filename = linkText;

          // Add extension if missing
          if (!path.extname(filename)) {
            const ext = this._guessExtension(url);
            if (ext) {
              filename += ext;
            }
          }
        }
      }

      const sanitized = FileSystemUtils.sanitizeFilename(filename || 'downloaded_file');
      return index > 0 ? `${index}-${sanitized}` : sanitized;
    } catch (error) {
      return index > 0 ? `${index}-downloaded_file` : 'downloaded_file';
    }
  }

  /**
   * Guess file extension from URL
   * @private
   * @param {string} url - Source URL
   * @returns {string|null} File extension or null
   */
  _guessExtension(url) {
    const urlLower = url.toLowerCase();
    for (const ext of this.fileExtensions) {
      if (urlLower.includes(ext)) {
        return ext;
      }
    }
    return null;
  }
}

module.exports = FileNameExtractor;
