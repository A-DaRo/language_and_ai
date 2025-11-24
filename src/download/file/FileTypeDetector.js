/**
 * @fileoverview File Type Detection Strategy for FileDownloader
 * @module download/file/FileTypeDetector
 * @description Identifies if URLs point to downloadable files
 */

/**
 * @class FileTypeDetector
 * @classdesc Detects whether a URL is a downloadable file
 */
class FileTypeDetector {
  constructor() {
    // File extensions to download
    this.fileExtensions = [
      '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
      '.zip', '.rar', '.7z', '.tar', '.gz',
      '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
      '.ipynb', '.json', '.xml', '.txt', '.md',
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

    // Blacklists
    this.blacklistedDomains = ['surf.nl', 'surfdrive.surf.nl'];
    this.blacklistedExtensions = ['.csv'];
  }

  /**
   * Check if a URL is a downloadable file
   * @param {string} url - URL to check
   * @param {string} [linkText=''] - Optional link text for context
   * @returns {boolean} True if URL is downloadable file
   */
  isDownloadableFile(url, linkText = '') {
    if (!url) return false;

    try {
      const urlLower = url.toLowerCase();

      // 1. Check Blacklists
      if (this.blacklistedDomains.some(domain => urlLower.includes(domain))) {
        return false;
      }
      
      // Check for blacklisted extensions (checking end of path or query param start)
      if (this.blacklistedExtensions.some(ext => urlLower.includes(ext))) {
         // Simple include check might be too aggressive (e.g. .csv.pdf?), but for .csv it's likely fine.
         // Better: check if it ends with extension or has extension followed by ? or /
         const isBlacklistedExt = this.blacklistedExtensions.some(ext => 
           urlLower.endsWith(ext) || urlLower.includes(`${ext}?`) || urlLower.includes(`${ext}/`)
         );
         if (isBlacklistedExt) return false;
      }

      // 2. Check Notion file patterns (Always allow these)
      for (const pattern of this.notionFilePatterns) {
        if (pattern.test(url)) {
          return true;
        }
      }

      // 3. Check Allowed Extensions
      for (const ext of this.fileExtensions) {
        if (urlLower.includes(ext)) {
          return true;
        }
      }

      // 4. Check link text for strong file indicators
      // We are stricter here: only accept if text looks like a filename with extension
      const textLower = linkText.toLowerCase();
      const strongIndicators = ['.pdf', '.zip', '.docx', '.pptx', '.xlsx', '.rar', '.7z'];
      
      if (strongIndicators.some(ind => textLower.includes(ind))) {
        return true;
      }

      // Removed generic 'download' or 'file' checks to avoid false positives like "Download Profile"
      
      return false;
    } catch (error) {
      return false;
    }
  }
}

module.exports = FileTypeDetector;
