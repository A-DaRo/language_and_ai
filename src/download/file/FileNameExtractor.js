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
      const pathname = urlObj.pathname || '';
      const pathnameFilename = path.basename(pathname);
      const pathnameExt = path.extname(pathname);
      let filename = '';

      if (pathnameExt && pathnameFilename) {
        filename = pathnameFilename;
      } else if (linkText && linkText.trim().length > 0 && linkText.length < 100) {
        filename = linkText;

        if (!path.extname(filename)) {
          const ext = this._guessExtension(url);
          if (ext) {
            filename += ext;
          }
        }
      } else if (pathnameFilename && pathnameFilename.length >= 3) {
        filename = pathnameFilename;
      }

      if (!filename) {
        const ext = this._guessExtension(url) || '';
        filename = `downloaded_file${ext}`;
      }

      const sanitized = FileSystemUtils.sanitizeFilename(filename);
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
