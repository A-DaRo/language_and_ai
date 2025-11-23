/**
 * @fileoverview Path Calculation Strategy for PageContext
 * @module domain/path/PathCalculator
 * @description Calculates filesystem paths based on page hierarchy
 */

const path = require('path');
const FileSystemUtils = require('../../utils/FileSystemUtils');

/**
 * @class PathCalculator
 * @classdesc Calculates relative and absolute paths for page context in hierarchy
 */
class PathCalculator {
  /**
   * Calculate relative path from page hierarchy
   * @param {Object} context - Page context with parent chain
   * @returns {string} Relative path using segment names
   */
  calculateRelativePath(context) {
    const segments = [];
    let current = context;

    while (current) {
      if (current.title && current.title !== 'untitled' && current.depth > 0) {
        const safeName = FileSystemUtils.sanitizeFilename(current.title);
        segments.unshift(safeName);
      }
      current = current.parentContext;
    }

    return path.posix.join(...segments);
  }

  /**
   * Calculate directory path for context
   * @param {string} baseDir - Base output directory
   * @param {Object} context - Page context
   * @returns {string} Full directory path for saving files
   */
  calculateDirectoryPath(baseDir, context) {
    const relativePath = this.calculateRelativePath(context);

    if (relativePath) {
      return path.join(baseDir, relativePath);
    }

    return baseDir;
  }

  /**
   * Calculate HTML file path
   * @param {string} baseDir - Base output directory
   * @param {Object} context - Page context
   * @returns {string} Full path to index.html for this page
   */
  calculateFilePath(baseDir, context) {
    const dirPath = this.calculateDirectoryPath(baseDir, context);
    return path.join(dirPath, 'index.html');
  }

  /**
   * Calculate relative path from one context to another
   * @param {Object} sourceContext - Source page context
   * @param {Object} targetContext - Target page context to navigate to
   * @returns {string} Relative path (e.g., '../sibling/child/index.html')
   */
  calculateRelativePathBetween(sourceContext, targetContext) {
    const sourcePath = this.calculateRelativePath(sourceContext);
    const targetPath = this.calculateRelativePath(targetContext);

    const sourceSegments = sourcePath.split(path.posix.sep).filter(s => s);
    const targetSegments = targetPath.split(path.posix.sep).filter(s => s);

    // Find common ancestor
    let commonDepth = 0;
    while (commonDepth < sourceSegments.length && 
           commonDepth < targetSegments.length && 
           sourceSegments[commonDepth] === targetSegments[commonDepth]) {
      commonDepth++;
    }

    const upLevels = sourceSegments.length - commonDepth;
    const downSegments = targetSegments.slice(commonDepth);

    let relativePath = '';
    for (let i = 0; i < upLevels; i++) {
      relativePath += '..' + path.posix.sep;
    }

    relativePath += downSegments.join(path.posix.sep);
    if (relativePath && !relativePath.endsWith('index.html')) {
      relativePath += path.posix.sep + 'index.html';
    }

    return relativePath || 'index.html';
  }
}

module.exports = PathCalculator;
