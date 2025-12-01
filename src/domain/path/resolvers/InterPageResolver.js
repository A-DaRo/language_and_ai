/**
 * @fileoverview Inter-Page Resolver
 * @module domain/path/resolvers/InterPageResolver
 * @description Cross-page navigation path resolution using the unified PathResolver interface.
 * 
 * This resolver handles links between different pages in the scraped hierarchy.
 * It computes relative filesystem paths based on the page hierarchy structure.
 * 
 * @see PathResolver - Base interface
 */

'use strict';

const path = require('path');
const PathResolver = require('../PathResolver');
const BlockIDMapper = require('../../../processing/BlockIDMapper');
const FileSystemUtils = require('../../../utils/FileSystemUtils');

/**
 * @class InterPageResolver
 * @extends PathResolver
 * @classdesc Resolves cross-page navigation paths using the PathResolver interface.
 * 
 * Handles the complex task of computing relative paths between pages
 * at different depths in the page hierarchy.
 * 
 * Algorithm:
 * 1. Get path segments for both contexts (sanitized titles from parent chain)
 * 2. Find common ancestor depth by comparing prefix segments
 * 3. Calculate upLevels = sourceSegments.length - commonDepth
 * 4. Get downSegments = targetSegments.slice(commonDepth)
 * 5. Build path: '../' × upLevels + downSegments.join('/') + '/index.html'
 * 
 * @example
 * const resolver = new InterPageResolver();
 * 
 * // Child to sibling (depth 1 → depth 1)
 * resolver.resolve({ source: childA, target: childB });
 * // Returns: '../ChildB/index.html'
 * 
 * // Grandchild to root (depth 2 → depth 0)
 * resolver.resolve({ source: grandchild, target: root });
 * // Returns: '../../index.html'
 */
class InterPageResolver extends PathResolver {
  /**
   * Create an InterPageResolver instance.
   * @param {BlockIDMapper} [blockIdMapper] - Optional custom block ID mapper
   */
  constructor(blockIdMapper = null) {
    super();
    this.blockIdMapper = blockIdMapper || new BlockIDMapper();
  }

  /**
   * Get the path type handled by this resolver.
   * @override
   * @returns {string} PathResolver.Types.INTER
   */
  getType() {
    return PathResolver.Types.INTER;
  }

  /**
   * Check if this resolver handles the given context.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} [context.source] - Source page context
   * @param {PageContext} [context.target] - Target page context
   * @returns {boolean} True if contexts refer to different internal pages
   */
  supports(context) {
    const { source, target } = context || {};

    // Guard against null/undefined contexts
    if (!source || !target) {
      return false;
    }

    // Must be different pages
    if (source.id === target.id) {
      return false;
    }

    // Both must be valid internal pages
    return this._isInternalPage(source) && this._isInternalPage(target);
  }

  /**
   * Resolve the path for a cross-page link.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} context.source - Source page context
   * @param {PageContext} context.target - Target page context
   * @param {string} [context.blockId] - Target block ID for anchor
   * @param {Map<string, Map<string, string>>} [context.blockMapCache] - Block ID mapping cache
   * @returns {string} Relative path (e.g., '../sibling/index.html#block-id')
   */
  resolve(context) {
    const { source, target, blockId, blockMapCache } = context || {};

    // Calculate the relative navigation path
    const relativePath = this._calculateRelativeNavigation(source, target);

    // Append anchor hash if block ID provided
    const anchorHash = blockId
      ? this._buildAnchorHash(blockId, target, blockMapCache)
      : '';

    return relativePath + anchorHash;
  }

  /**
   * Calculate filesystem-relative path from source to target.
   * @private
   * @param {PageContext} source - Source page
   * @param {PageContext} target - Target page
   * @returns {string} Relative path (e.g., '../sibling/index.html')
   */
  _calculateRelativeNavigation(source, target) {
    const sourceSegments = this._getPathSegments(source);
    const targetSegments = this._getPathSegments(target);

    // Find common ancestor depth
    let commonDepth = 0;
    while (
      commonDepth < sourceSegments.length &&
      commonDepth < targetSegments.length &&
      sourceSegments[commonDepth] === targetSegments[commonDepth]
    ) {
      commonDepth++;
    }

    // Calculate up-levels needed to reach common ancestor
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
   * @private
   * @param {PageContext} context - Page context
   * @returns {string[]} Array of sanitized path segments
   */
  _getPathSegments(context) {
    // Priority 1: Pre-computed pathSegments (survives IPC serialization)
    if (context.pathSegments && Array.isArray(context.pathSegments) && context.pathSegments.length > 0) {
      return [...context.pathSegments];
    }

    // Priority 2: Use getPathSegments() method if available
    if (typeof context.getPathSegments === 'function') {
      const segments = context.getPathSegments();
      if (segments && segments.length > 0) {
        return [...segments];
      }
    }

    // Priority 3: Fallback to parent chain traversal
    const segments = [];
    let current = context;

    while (current) {
      if (current.depth > 0 && current.title && current.title !== 'untitled') {
        const safeName = FileSystemUtils.sanitizeFilename(current.title);
        segments.unshift(safeName);
      }
      current = current.parentContext;
    }

    return segments;
  }

  /**
   * Build anchor hash for block ID.
   * @private
   * @param {string} rawBlockId - Raw block ID from URL
   * @param {PageContext} target - Target page context
   * @param {Map<string, Map<string, string>>} blockMapCache - Block ID cache
   * @returns {string} Formatted anchor hash (e.g., '#29d979ee-ca9f-...')
   */
  _buildAnchorHash(rawBlockId, target, blockMapCache) {
    if (!rawBlockId) {
      return '';
    }

    let formattedId;

    // Try to find formatted ID in cache
    if (blockMapCache && target && target.id) {
      const pageBlockMap = blockMapCache.get(target.id);
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
   * @private
   * @param {PageContext} context - Context to check
   * @returns {boolean} True if valid internal page
   */
  _isInternalPage(context) {
    return context && context.id && typeof context.id === 'string';
  }
}

module.exports = InterPageResolver;
