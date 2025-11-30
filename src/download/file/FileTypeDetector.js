/**
 * @fileoverview File Type Detection Strategy for FileDownloader
 * @module download/file/FileTypeDetector
 * @description Identifies if URLs point to downloadable files using a strict whitelist approach.
 * 
 * @design WHITELIST POLICY
 * This detector uses a strict whitelist approach instead of blacklisting.
 * A URL is only considered downloadable if:
 * 1. It matches a known Notion file hosting pattern, OR
 * 2. The URL path ends with a whitelisted file extension
 * 
 * This prevents accidental downloads of HTML pages, API responses, or other
 * non-file resources that may slip through a blacklist-based approach.
 */

/**
 * @class FileTypeDetector
 * @classdesc Detects whether a URL is a downloadable file using strict whitelist policy.
 * 
 * @example
 * const detector = new FileTypeDetector();
 * detector.isDownloadableFile('https://example.com/doc.pdf'); // true
 * detector.isDownloadableFile('https://example.com/page'); // false
 * detector.isDownloadableFile('https://notion.so/signed/xyz'); // true (Notion pattern)
 */
class FileTypeDetector {
  constructor() {
    /**
     * @property {string[]} whitelistedExtensions - File extensions allowed for download.
     * Only URLs ending with these extensions (before query string) are considered files.
     */
    this.whitelistedExtensions = [
      // Documents
      '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
      // Archives
      '.zip', '.rar', '.7z', '.tar', '.gz', '.tar.gz', '.tgz',
      // Code files
      '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.go',
      // Data files
      '.ipynb', '.json', '.xml', '.txt', '.md', '.csv', '.yaml', '.yml',
      // Media
      '.mp4', '.avi', '.mov', '.mp3', '.wav', '.ogg', '.webm',
      // Images
      '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp', '.ico'
    ];

    /**
     * @property {RegExp[]} notionFilePatterns - URL patterns that indicate Notion-hosted files.
     * These bypass extension checking as Notion uses signed URLs without extensions.
     */
    this.notionFilePatterns = [
      /notion\.so\/signed\//i,
      /s3.*amazonaws\.com.*prod-files-secure/i,
      /notion-static\.com/i,
      /secure\.notion-static\.com/i
    ];
  }

  /**
   * Check if a URL is a downloadable file using strict whitelist policy.
   * 
   * @param {string} url - URL to check
   * @param {string} [linkText=''] - Optional link text for additional context
   * @returns {boolean} True if URL matches whitelist criteria
   * 
   * @description Detection algorithm:
   * 1. Check if URL matches Notion file hosting patterns (always allow)
   * 2. Extract the path from URL (remove query string and hash)
   * 3. Check if path ends with a whitelisted extension
   * 4. Optionally check link text for strong file indicators (filename with extension)
   * 
   * @example
   * isDownloadableFile('https://ex.com/file.pdf'); // true - whitelisted extension
   * isDownloadableFile('https://ex.com/file.pdf?v=1'); // true - extension before query
   * isDownloadableFile('https://ex.com/page'); // false - no extension
   * isDownloadableFile('https://notion.so/signed/abc'); // true - Notion pattern
   */
  isDownloadableFile(url, linkText = '') {
    if (!url || typeof url !== 'string') return false;

    try {
      // 1. Check Notion file patterns first (these use signed URLs without extensions)
      for (const pattern of this.notionFilePatterns) {
        if (pattern.test(url)) {
          return true;
        }
      }

      // 2. Extract clean path from URL
      const cleanPath = this._extractCleanPath(url);
      if (!cleanPath) return false;

      // 3. Check if path ends with a whitelisted extension
      if (this._hasWhitelistedExtension(cleanPath)) {
        return true;
      }

      // 4. Check link text for strong file indicators (filename.ext pattern)
      if (linkText && this._isFilenameLikeText(linkText)) {
        return true;
      }

      return false;
    } catch (error) {
      return false;
    }
  }

  /**
   * Extract the extension from a URL or filename
   * @param {string} urlOrPath - URL or file path
   * @returns {string|null} Extension including dot (e.g., '.pdf') or null
   */
  getExtension(urlOrPath) {
    if (!urlOrPath) return null;
    
    const cleanPath = this._extractCleanPath(urlOrPath);
    if (!cleanPath) return null;

    const lastDot = cleanPath.lastIndexOf('.');
    if (lastDot === -1 || lastDot === cleanPath.length - 1) return null;

    const ext = cleanPath.substring(lastDot).toLowerCase();
    
    // Validate it's a reasonable extension (1-10 chars after dot)
    if (ext.length > 1 && ext.length <= 11) {
      return ext;
    }
    return null;
  }

  /**
   * Extract clean path from URL (removes query string and hash)
   * @private
   * @param {string} url - URL to process
   * @returns {string} Clean path portion
   */
  _extractCleanPath(url) {
    try {
      // Handle both full URLs and relative paths
      let path = url;
      
      // Remove hash
      const hashIndex = path.indexOf('#');
      if (hashIndex !== -1) {
        path = path.substring(0, hashIndex);
      }
      
      // Remove query string
      const queryIndex = path.indexOf('?');
      if (queryIndex !== -1) {
        path = path.substring(0, queryIndex);
      }

      // Try to parse as URL for proper path extraction
      try {
        const urlObj = new URL(url);
        path = urlObj.pathname;
      } catch {
        // Not a valid URL, use the cleaned string as-is
      }

      return path.toLowerCase();
    } catch {
      return url.toLowerCase();
    }
  }

  /**
   * Check if path ends with a whitelisted extension
   * @private
   * @param {string} path - Clean URL path
   * @returns {boolean} True if whitelisted extension found
   */
  _hasWhitelistedExtension(path) {
    const pathLower = path.toLowerCase();
    
    // Check compound extensions first (e.g., .tar.gz)
    for (const ext of this.whitelistedExtensions) {
      if (ext.includes('.') && ext.split('.').length > 2) {
        // Compound extension like .tar.gz
        if (pathLower.endsWith(ext)) {
          return true;
        }
      }
    }

    // Check simple extensions
    for (const ext of this.whitelistedExtensions) {
      if (pathLower.endsWith(ext)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if link text looks like a filename with extension
   * @private
   * @param {string} text - Link text to check
   * @returns {boolean} True if text appears to be a filename
   */
  _isFilenameLikeText(text) {
    if (!text || text.length > 100) return false;
    
    const trimmed = text.trim().toLowerCase();
    
    // Strong file indicators: text contains filename-like pattern
    const strongIndicators = ['.pdf', '.zip', '.docx', '.pptx', '.xlsx', '.rar', '.7z', '.ipynb'];
    
    for (const ext of strongIndicators) {
      // Check if text ends with extension or contains "filename.ext" pattern
      if (trimmed.endsWith(ext) || /\w+\.(pdf|zip|docx|pptx|xlsx|rar|7z|ipynb)$/i.test(trimmed)) {
        return true;
      }
    }

    return false;
  }
}

module.exports = FileTypeDetector;
