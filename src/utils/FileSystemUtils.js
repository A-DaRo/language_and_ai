const crypto = require('crypto');
const path = require('path');

/**
 * Utility class for file system operations and path sanitization.
 */
class FileSystemUtils {
  /**
   * Sanitize a filename or directory name to be filesystem-safe.
   * Handles complex URLs, special characters, and length limits.
   * 
   * @param {string} filename - The original filename or title.
   * @returns {string} A safe, sanitized filename.
   */
  static sanitizeFilename(filename) {
    if (!filename) return 'untitled';

    // Decode URI components if it looks like a URL segment
    let sanitized = decodeURIComponent(filename);

    // Remove or replace problematic characters
    // Windows reserved chars: < > : " / \ | ? *
    // Control chars: \x00-\x1F
    sanitized = sanitized
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_') // Replace invalid chars with underscore
      .replace(/[^\w\s\-\.]/g, '') // Remove non-alphanumeric except spaces, hyphens, dots (stricter than AssetDownloader, closer to PageContext)
      .replace(/^\.+/, '') // Remove leading dots
      .replace(/\s+/g, '_') // Replace spaces with underscores
      .replace(/_{2,}/g, '_') // Replace multiple underscores with single
      .replace(/^[._]+|[._]+$/g, '') // Remove leading/trailing dots and underscores
      .trim();

    // If filename is empty after sanitization, provide a fallback
    if (!sanitized) sanitized = 'asset';

    // Limit length while preserving file extension if present
    // Max filename length is usually 255, but we keep it safer around 100-200
    if (sanitized.length > 150) {
      const hash = crypto.createHash('md5').update(filename).digest('hex').substring(0, 8);
      const ext = path.extname(sanitized);
      
      // If extension is too long, it might not be a real extension
      if (ext.length > 10) {
         sanitized = sanitized.substring(0, 140) + '_' + hash;
      } else {
         const namePart = sanitized.substring(0, 140 - ext.length);
         sanitized = `${namePart}_${hash}${ext}`;
      }
    }

    return sanitized;
  }
}

module.exports = FileSystemUtils;
