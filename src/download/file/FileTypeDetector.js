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
  }

  /**
   * Check if a URL is a downloadable file
   * @param {string} url - URL to check
   * @param {string} [linkText=''] - Optional link text for context
   * @returns {boolean} True if URL is downloadable file
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

      // Check link text for download indicators
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
}

module.exports = FileTypeDetector;
