/**
 * @fileoverview Path Calculation Strategy for PageContext
 * @module domain/path/PathCalculator
 * @description Calculates filesystem paths based on page hierarchy.
 * 
 * @deprecated This class is being superseded by the PathStrategy pattern.
 * For new code, use PathStrategyFactory with IntraPathStrategy/InterPathStrategy.
 * This class is retained for backward compatibility with existing code.
 * 
 * @design PATH CALCULATION STRATEGY (Legacy)
 * This calculator handles all path-related operations for the scraper:
 * 
 * 1. **Relative Path (from hierarchy)**: Build path segments from parent chain
 *    - Root (depth=0) → empty string (files go in baseDir directly)
 *    - Child (depth>0) → parent segments + own title
 * 
 * 2. **Directory Path**: baseDir + relativePath
 * 
 * 3. **File Path**: directoryPath + 'index.html'
 * 
 * 4. **Relative Path Between Pages**: Calculate navigation path from source to target
 *    - Finds common ancestor depth
 *    - Calculates upLevels (../) needed
 *    - Appends target path segments
 * 
 * @migration For cross-page navigation, prefer:
 * ```javascript
 * const factory = new PathStrategyFactory(config, logger);
 * const path = factory.resolvePath(source, target, { blockId, blockMapCache });
 * ```
 * 
 * @fixes
 * - Root page links: Pages at depth=1 linking to root now correctly get '../index.html'
 *   instead of './index.html' (fixed upLevels calculation)
 * - Empty path handling: When target is root, ensure proper '../' prefix based on source depth
 * 
 * @see PathStrategyFactory - New unified path resolution
 * @see InterPathStrategy - Cross-page navigation
 * @see IntraPathStrategy - Same-page anchors
 */

const path = require('path');
const FileSystemUtils = require('../../utils/FileSystemUtils');

/**
 * @class PathCalculator
 * @classdesc Calculates relative and absolute paths for page context in hierarchy.
 * 
 * @example
 * const calc = new PathCalculator();
 * 
 * // Root page (depth=0)
 * calc.calculateRelativePath(rootContext); // ''
 * 
 * // Child page (depth=1)
 * calc.calculateRelativePath(childContext); // 'ChildTitle'
 * 
 * // Grandchild page (depth=2)
 * calc.calculateRelativePath(grandchildContext); // 'ChildTitle/GrandchildTitle'
 * 
 * // Link from depth=1 to root
 * calc.calculateRelativePathBetween(childContext, rootContext); // '../index.html'
 */
class PathCalculator {
  /**
   * Calculate relative path from page hierarchy.
   * 
   * @description Builds path by traversing parent chain and collecting title segments.
   * Root pages (depth=0) return empty string since they live in baseDir directly.
   * 
   * @param {Object} context - Page context with parent chain
   * @param {string} context.title - Sanitized page title
   * @param {number} context.depth - Page depth in hierarchy
   * @param {Object|null} context.parentContext - Reference to parent context
   * @returns {string} Relative path using segment names (e.g., 'Section/Page')
   */
  calculateRelativePath(context) {
    const segments = [];
    let current = context;

    while (current) {
      // Only include pages with depth > 0 (skip root in path)
      if (current.title && current.title !== 'untitled' && current.depth > 0) {
        const safeName = FileSystemUtils.sanitizeFilename(current.title);
        segments.unshift(safeName);
      }
      current = current.parentContext;
    }

    return path.posix.join(...segments);
  }

  /**
   * Calculate directory path for context.
   * 
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
   * Calculate HTML file path.
   * 
   * @param {string} baseDir - Base output directory
   * @param {Object} context - Page context
   * @returns {string} Full path to index.html for this page
   */
  calculateFilePath(baseDir, context) {
    const dirPath = this.calculateDirectoryPath(baseDir, context);
    return path.join(dirPath, 'index.html');
  }

  /**
   * Calculate relative path from one context to another.
   * 
   * @description Calculates the relative filesystem path needed to navigate
   * from source page's location to target page's location.
   * 
   * Algorithm:
   * 1. Get path segments for both source and target
   * 2. Find common ancestor (matching prefix segments)
   * 3. Calculate upLevels: how many '../' needed from source to common ancestor
   * 4. Append target segments from common ancestor downward
   * 5. Always append '/index.html' at the end
   * 
   * @param {Object} sourceContext - Source page context
   * @param {Object} targetContext - Target page context to navigate to
   * @returns {string} Relative path (e.g., '../sibling/child/index.html')
   * 
   * @example
   * // Source: depth=1 (Section), Target: depth=0 (Root)
   * // sourceSegments: ['Section'], targetSegments: []
   * // commonDepth: 0, upLevels: 1, downSegments: []
   * // Result: '../index.html'
   * 
   * @example
   * // Source: depth=2 (Section/Page), Target: depth=1 (OtherSection)
   * // sourceSegments: ['Section', 'Page'], targetSegments: ['OtherSection']
   * // commonDepth: 0, upLevels: 2, downSegments: ['OtherSection']
   * // Result: '../../OtherSection/index.html'
   */
  calculateRelativePathBetween(sourceContext, targetContext) {
    const sourcePath = this.calculateRelativePath(sourceContext);
    const targetPath = this.calculateRelativePath(targetContext);

    const sourceSegments = sourcePath ? sourcePath.split(path.posix.sep).filter(s => s) : [];
    const targetSegments = targetPath ? targetPath.split(path.posix.sep).filter(s => s) : [];

    // Find common ancestor depth
    let commonDepth = 0;
    while (commonDepth < sourceSegments.length && 
           commonDepth < targetSegments.length && 
           sourceSegments[commonDepth] === targetSegments[commonDepth]) {
      commonDepth++;
    }

    // Calculate up-levels needed from source to common ancestor
    const upLevels = sourceSegments.length - commonDepth;
    
    // Get segments from common ancestor to target
    const downSegments = targetSegments.slice(commonDepth);

    // Build relative path
    let relativePath = '';
    
    // Add '../' for each level up
    for (let i = 0; i < upLevels; i++) {
      relativePath += '..' + path.posix.sep;
    }

    // Add target segments
    if (downSegments.length > 0) {
      relativePath += downSegments.join(path.posix.sep);
    }

    // Ensure path ends with index.html
    if (relativePath) {
      // Remove trailing separator if present
      if (relativePath.endsWith(path.posix.sep)) {
        relativePath += 'index.html';
      } else if (!relativePath.endsWith('index.html')) {
        relativePath += path.posix.sep + 'index.html';
      }
    } else {
      // Same page or root-to-root navigation
      relativePath = 'index.html';
    }

    return relativePath;
  }
}

module.exports = PathCalculator;
