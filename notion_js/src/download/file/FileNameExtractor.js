/**
 * @fileoverview File Naming Strategy for FileDownloader
 * @module download/file/FileNameExtractor
 * @description Extracts and sanitizes filenames from URLs and link text.
 * 
 * @design EXTENSION ENFORCEMENT POLICY
 * This extractor enforces file extensions on all downloaded files:
 * 1. Extract filename from URL path or link text
 * 2. If no extension detected, attempt to guess from URL patterns
 * 3. If still no extension:
 *    - For image-related URLs/contexts: append '.jpg'
 *    - For other files: append '.bin' as fallback
 * This ensures all downloaded files have proper extensions for OS compatibility.
 */

const path = require('path');
const FileSystemUtils = require('../../utils/FileSystemUtils');

/**
 * @class FileNameExtractor
 * @classdesc Extracts meaningful filenames from URLs and link text with extension enforcement.
 * 
 * @example
 * const extractor = new FileNameExtractor();
 * extractor.extractFilename('https://ex.com/doc.pdf', 'My Doc'); // 'My_Doc.pdf'
 * extractor.extractFilename('https://ex.com/file', '', 1, 'images'); // '1-file.jpg'
 */
class FileNameExtractor {
  constructor() {
    /**
     * @property {string[]} knownExtensions - File extensions for detection and validation.
     */
    this.knownExtensions = [
      // Documents
      '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
      // Archives
      '.zip', '.rar', '.7z', '.tar', '.gz', '.tgz',
      // Code files
      '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.go',
      // Data files .csv excluded.
      '.ipynb', '.json', '.xml', '.txt', '.md', '.yaml', '.yml',
      // Media
      '.mp4', '.avi', '.mov', '.mp3', '.wav', '.ogg', '.webm',
      // Images
      '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp', '.ico'
    ];

    /**
     * @property {string[]} imageExtensions - Extensions considered as images.
     */
    this.imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp', '.ico'];

    /**
     * @property {Object} mimeToExtension - MIME type to extension mapping for Content-Type hints.
     */
    this.mimeToExtension = {
      'application/pdf': '.pdf',
      'application/zip': '.zip',
      'application/x-zip-compressed': '.zip',
      'application/msword': '.doc',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
      'application/vnd.ms-excel': '.xls',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
      'application/vnd.ms-powerpoint': '.ppt',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
      'image/jpeg': '.jpg',
      'image/png': '.png',
      'image/gif': '.gif',
      'image/svg+xml': '.svg',
      'image/webp': '.webp',
      'text/plain': '.txt',
      'text/markdown': '.md',
      'application/json': '.json',
      'application/xml': '.xml',
      'text/csv': '.csv'
    };
  }

  /**
   * Extract filename from URL or link text with extension enforcement.
   * 
   * @param {string} url - Source URL
   * @param {string} [linkText=''] - Optional link text that may contain filename
   * @param {number} [index=0] - Optional index for uniqueness (prepended if > 0)
   * @param {string} [targetFolder='files'] - Target folder context for extension guessing
   * @returns {string} Sanitized filename with guaranteed extension
   * 
   * @example
   * extractFilename('https://ex.com/doc.pdf', 'My Doc', 0); // 'My_Doc.pdf'
   * extractFilename('https://ex.com/file', '', 1, 'images'); // '1-file.jpg'
   * extractFilename('https://ex.com/data', 'data', 0, 'files'); // 'data.bin'
   */
  extractFilename(url, linkText = '', index = 0, targetFolder = 'files') {
    try {
      const urlObj = new URL(url);
      const pathname = urlObj.pathname || '';
      const pathnameFilename = path.basename(pathname);
      const pathnameExt = this._extractExtension(pathname);
      
      let filename = '';
      let detectedExt = null;

      // Strategy 1: Use pathname if it has an extension
      if (pathnameExt && pathnameFilename) {
        filename = pathnameFilename;
        detectedExt = pathnameExt;
      } 
      // Strategy 2: Use link text if reasonable length and has extension
      else if (linkText && linkText.trim().length > 0 && linkText.length < 100) {
        const linkExt = this._extractExtension(linkText);
        filename = linkText.trim();
        
        if (linkExt) {
          detectedExt = linkExt;
        }
      } 
      // Strategy 3: Use pathname filename even without extension
      else if (pathnameFilename && pathnameFilename.length >= 3) {
        filename = pathnameFilename;
      }

      // Fallback filename
      if (!filename) {
        filename = 'downloaded_file';
      }

      // Ensure filename has an extension
      if (!detectedExt) {
        detectedExt = this._guessExtension(url, targetFolder);
        
        // Only append extension if filename doesn't already have one
        const currentExt = this._extractExtension(filename);
        if (!currentExt && detectedExt) {
          filename = filename + detectedExt;
        }
      }

      // Final fallback: ensure there's always an extension
      const finalExt = this._extractExtension(filename);
      if (!finalExt) {
        const fallbackExt = this._getFallbackExtension(targetFolder);
        filename = filename + fallbackExt;
      }

      // Sanitize and add index prefix if needed
      const sanitized = FileSystemUtils.sanitizeFilename(filename);
      return index > 0 ? `${index}-${sanitized}` : sanitized;
      
    } catch (error) {
      // Error parsing URL - use fallback with index
      const fallbackExt = this._getFallbackExtension(targetFolder);
      const fallbackName = `downloaded_file${fallbackExt}`;
      return index > 0 ? `${index}-${fallbackName}` : fallbackName;
    }
  }

  /**
   * Extract extension from a string (URL path, filename, or link text)
   * @private
   * @param {string} str - String to extract extension from
   * @returns {string|null} Extension including dot (e.g., '.pdf') or null
   */
  _extractExtension(str) {
    if (!str) return null;
    
    // Remove query string and hash
    let clean = str.split('?')[0].split('#')[0];
    
    // Find the last dot
    const lastDot = clean.lastIndexOf('.');
    if (lastDot === -1 || lastDot === clean.length - 1) return null;
    
    const ext = clean.substring(lastDot).toLowerCase();
    
    // Validate: must be a known extension or reasonable length (1-10 chars)
    if (this.knownExtensions.includes(ext)) {
      return ext;
    }
    
    // Accept unknown but reasonable extensions
    if (ext.length > 1 && ext.length <= 11 && /^\.[a-z0-9]+$/i.test(ext)) {
      return ext;
    }
    
    return null;
  }

  /**
   * Guess file extension from URL patterns and context.
   * @private
   * @param {string} url - Source URL
   * @param {string} targetFolder - Target folder context
   * @returns {string|null} Guessed extension or null
   */
  _guessExtension(url, targetFolder) {
    const urlLower = url.toLowerCase();
    
    // Check for extension anywhere in URL (common in CDN URLs)
    for (const ext of this.knownExtensions) {
      // Look for extension followed by ? or / or end of string
      const patterns = [
        ext + '?',
        ext + '/',
        ext + '&'
      ];
      
      if (urlLower.endsWith(ext) || patterns.some(p => urlLower.includes(p))) {
        return ext;
      }
    }
    
    // Context-based guessing
    if (targetFolder === 'images' || urlLower.includes('/image') || urlLower.includes('img')) {
      return '.jpg';
    }
    
    // Notion-specific patterns
    if (urlLower.includes('notion') && urlLower.includes('secure')) {
      // Notion signed URLs often are images
      return '.jpg';
    }
    
    return null;
  }

  /**
   * Get fallback extension based on target folder context.
   * @private
   * @param {string} targetFolder - Target folder context ('images', 'files', etc.)
   * @returns {string} Fallback extension with dot
   */
  _getFallbackExtension(targetFolder) {
    if (targetFolder === 'images') {
      return '.jpg';
    }
    return '.bin';
  }

  /**
   * Get extension from MIME type (Content-Type header).
   * @param {string} mimeType - MIME type string
   * @returns {string|null} Extension or null if unknown
   */
  getExtensionFromMime(mimeType) {
    if (!mimeType) return null;
    
    // Extract base MIME type (remove charset, etc.)
    const baseMime = mimeType.split(';')[0].trim().toLowerCase();
    
    return this.mimeToExtension[baseMime] || null;
  }

  /**
   * Check if extension is for an image file.
   * @param {string} ext - Extension to check (with or without dot)
   * @returns {boolean} True if image extension
   */
  isImageExtension(ext) {
    if (!ext) return false;
    const normalizedExt = ext.startsWith('.') ? ext.toLowerCase() : '.' + ext.toLowerCase();
    return this.imageExtensions.includes(normalizedExt);
  }
}

module.exports = FileNameExtractor;
