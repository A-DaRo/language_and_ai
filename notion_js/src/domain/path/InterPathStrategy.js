/**
 * @fileoverview Inter-Page Path Strategy
 * @module domain/path/InterPathStrategy
 * @description Handles cross-page navigation path resolution.
 * 
 * @design INTER-PAGE NAVIGATION
 * This strategy handles links between different pages in the scraped hierarchy.
 * It computes relative filesystem paths based on the page hierarchy structure.
 * 
 * Algorithm:
 * 1. Build path segments from hierarchy for both source and target contexts
 * 2. Find common ancestor depth (matching prefix segments)
 * 3. Calculate '../' prefix count for navigating up from source
 * 4. Append target path segments for navigating down to target
 * 5. Always append 'index.html' at the end
 * 6. Optionally append block anchor hash if block ID provided
 * 
 * Responsibilities:
 * - Computing relative filesystem paths between different pages
 * - Handling hierarchical depth calculations (parent chains)
 * - Appending optional anchor hashes for block ID targets
 * 
 * @see PathStrategy - Base interface
 * @see IntraPathStrategy - Same-page anchor resolution
 * @see PathCalculator - Original implementation (to be deprecated)
 */

const path = require('path');
const PathStrategy = require('./PathStrategy');
const BlockIDMapper = require('../../processing/BlockIDMapper');
const FileSystemUtils = require('../../utils/FileSystemUtils');

/**
 * @class InterPathStrategy
 * @extends PathStrategy
 * @classdesc Resolves cross-page navigation paths.
 * 
 * Handles the complex task of computing relative paths between pages
 * at different depths in the page hierarchy. Uses the parent chain
 * to build path segments and finds the common ancestor to minimize
 * the path length.
 * 
 * @example
 * const strategy = new InterPathStrategy();
 * 
 * // Child to sibling (depth 1 → depth 1)
 * strategy.resolve(childA, childB, {});
 * // Returns: '../ChildB/index.html'
 * 
 * // Grandchild to root (depth 2 → depth 0)
 * strategy.resolve(grandchild, root, {});
 * // Returns: '../../index.html'
 * 
 * // With block anchor
 * strategy.resolve(pageA, pageB, { blockId: '29d979eeca9f' });
 * // Returns: '../PageB/index.html#29d979ee-ca9f-...'
 */
class InterPathStrategy extends PathStrategy {
  /**
   * Create an InterPathStrategy instance.
   * 
   * @param {BlockIDMapper} [blockIdMapper] - Optional custom block ID mapper.
   *   If not provided, a default instance is created.
   */
  constructor(blockIdMapper = null) {
    super();
    this.blockIdMapper = blockIdMapper || new BlockIDMapper();
  }

  /**
   * Check if this strategy handles the given context pair.
   * 
   * @override
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @returns {boolean} True if contexts refer to different internal pages
   * 
   * @description Returns true when:
   * - Both contexts are valid internal pages (have IDs)
   * - Source and target have different page IDs
   */
  supports(sourceContext, targetContext) {
    // Guard against null/undefined contexts
    if (!sourceContext || !targetContext) {
      return false;
    }

    // Must be different pages
    if (sourceContext.id === targetContext.id) {
      return false;
    }

    // Both must be valid internal pages
    return this._isInternalPage(sourceContext) && this._isInternalPage(targetContext);
  }

  /**
   * Get the path type handled by this strategy.
   * 
   * @override
   * @returns {string} PathType.INTER
   */
  getType() {
    return PathStrategy.PathType.INTER;
  }

  /**
   * Resolve the path for a cross-page link.
   * 
   * @override
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context
   * @param {Object} [options={}] - Resolution options
   * @param {string} [options.blockId] - Target block ID for anchor
   * @param {Map<string, Map<string, string>>} [options.blockMapCache] - Block ID mapping cache
   * @returns {string} Relative path (e.g., '../sibling/index.html#block-id')
   * 
   * @description Computes the relative filesystem path from source to target:
   * 1. Gets path segments for both contexts
   * 2. Finds common ancestor
   * 3. Builds '../' prefix for upward navigation
   * 4. Appends target segments for downward navigation
   * 5. Adds 'index.html' filename
   * 6. Optionally appends block anchor hash
   */
  resolve(sourceContext, targetContext, options = {}) {
    const { blockId, blockMapCache } = options;

    // Calculate the relative navigation path
    const relativePath = this._calculateRelativeNavigation(sourceContext, targetContext);

    // Append anchor hash if block ID provided
    const anchorHash = blockId
      ? this._buildAnchorHash(blockId, targetContext, blockMapCache)
      : '';

    return relativePath + anchorHash;
  }

  /**
   * Calculate filesystem-relative path from source to target.
   * 
   * @private
   * @param {PageContext} sourceContext - Origin page
   * @param {PageContext} targetContext - Destination page
   * @returns {string} Relative path (e.g., '../sibling/index.html')
   * 
   * @description Algorithm:
   * 1. Get path segments for source and target (sanitized titles from parent chain)
   * 2. Find common ancestor depth by comparing prefix segments
   * 3. Calculate upLevels = sourceSegments.length - commonDepth
   * 4. Get downSegments = targetSegments.slice(commonDepth)
   * 5. Build path: '../' × upLevels + downSegments.join('/') + '/index.html'
   * 
   * @example
   * // Source: ['Section', 'Page'], Target: ['OtherSection']
   * // commonDepth: 0, upLevels: 2, downSegments: ['OtherSection']
   * // Result: '../../OtherSection/index.html'
   */
  _calculateRelativeNavigation(sourceContext, targetContext) {
    const sourceSegments = this._getPathSegments(sourceContext);
    const targetSegments = this._getPathSegments(targetContext);

    // Find common ancestor depth
    let commonDepth = 0;
    while (
      commonDepth < sourceSegments.length &&
      commonDepth < targetSegments.length &&
      sourceSegments[commonDepth] === targetSegments[commonDepth]
    ) {
      commonDepth++;
    }

    // Calculate up-levels (../) needed to reach common ancestor
    const upLevels = sourceSegments.length - commonDepth;

    // Get down-path segments from common ancestor to target
    const downSegments = targetSegments.slice(commonDepth);

    // Build the relative path
    let result = '';

    // Add '../' for each level up
    for (let i = 0; i < upLevels; i++) {
      result += '../';
    }

    // Add target path segments
    if (downSegments.length > 0) {
      result += downSegments.join('/') + '/';
    }

    // Always end with index.html
    result += 'index.html';

    return result;
  }

  /**
   * Get path segments for a context.
   * Prefers pre-computed pathSegments (survives IPC serialization).
   * Falls back to parent chain traversal if not available.
   * 
   * @private
   * @param {PageContext} context - Page context to get segments for
   * @returns {string[]} Array of sanitized path segments (excluding root)
   * 
   * @description 
   * Priority order:
   * 1. Pre-computed pathSegments array (IPC-safe, computed at construction)
   * 2. getPathSegments() method if available
   * 3. Manual parent chain traversal (only works pre-serialization)
   * 
   * @example
   * // For a page at depth 2 with parents: Root → Section → Page
   * _getPathSegments(pageContext);
   * // Returns: ['Section', 'Page']
   */
  _getPathSegments(context) {
    // Priority 1: Use pre-computed pathSegments (survives IPC serialization)
    if (context.pathSegments && Array.isArray(context.pathSegments) && context.pathSegments.length > 0) {
      return [...context.pathSegments];  // Return copy to prevent mutation
    }

    // Priority 2: Use getPathSegments() method if available
    if (typeof context.getPathSegments === 'function') {
      const segments = context.getPathSegments();
      if (segments && segments.length > 0) {
        return [...segments];
      }
    }

    // Priority 3: Fallback to parent chain traversal (only works pre-serialization)
    const segments = [];
    let current = context;

    while (current) {
      // Only include pages with depth > 0 (skip root in path)
      if (current.depth > 0 && current.title && current.title !== 'untitled') {
        const safeName = FileSystemUtils.sanitizeFilename(current.title);
        segments.unshift(safeName); // Add at beginning to maintain order
      }
      current = current.parentContext;
    }

    return segments;
  }

  /**
   * Build anchor hash for block ID.
   * 
   * @private
   * @param {string} rawBlockId - Raw block ID from URL
   * @param {PageContext} targetContext - Target page for block map lookup
   * @param {Map<string, Map<string, string>>} blockMapCache - Cache of block ID maps
   * @returns {string} Formatted anchor hash (e.g., '#29d979ee-ca9f-...')
   * 
   * @description Formats the raw block ID using the block map cache if available,
   * falling back to standard UUID formatting if not found.
   */
  _buildAnchorHash(rawBlockId, targetContext, blockMapCache) {
    if (!rawBlockId) {
      return '';
    }

    let formattedId;

    // Try to find formatted ID in cache
    if (blockMapCache && targetContext && targetContext.id) {
      const pageBlockMap = blockMapCache.get(targetContext.id);
      if (pageBlockMap && pageBlockMap.size > 0) {
        formattedId = this.blockIdMapper.getFormattedId(rawBlockId, pageBlockMap);
      }
    }

    // Fallback to direct formatting if not found in cache
    if (!formattedId) {
      formattedId = this.blockIdMapper.getFormattedId(rawBlockId, null);
    }

    return `#${formattedId}`;
  }

  /**
   * Check if context represents an internal scraped page.
   * 
   * @private
   * @param {PageContext} context - Context to check
   * @returns {boolean} True if context is a valid internal page
   */
  _isInternalPage(context) {
    return context && context.id && typeof context.id === 'string';
  }
}

module.exports = InterPathStrategy;
