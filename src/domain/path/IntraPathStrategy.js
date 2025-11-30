/**
 * @fileoverview Intra-Page Path Strategy
 * @module domain/path/IntraPathStrategy
 * @description Handles same-page anchor link resolution.
 * 
 * @design INTRA-PAGE LINKS
 * This strategy handles the case where source and target pages are the same.
 * In this scenario, only an anchor hash is needed (e.g., #block-id).
 * 
 * Responsibilities:
 * - Detecting same-page links (sourceContext.id === targetContext.id)
 * - Formatting block IDs as UUID-style anchor hashes
 * - Block ID mapping lookup via cache for accurate formatting
 * 
 * @see PathStrategy - Base interface
 * @see InterPathStrategy - Cross-page navigation
 * @see BlockIDMapper - Block ID formatting utility
 */

const PathStrategy = require('./PathStrategy');
const BlockIDMapper = require('../../processing/BlockIDMapper');

/**
 * @class IntraPathStrategy
 * @extends PathStrategy
 * @classdesc Resolves same-page anchor links.
 * 
 * When a link points to a different section of the same page,
 * only an anchor hash is needed. This prevents unnecessary page reloads
 * and enables smooth in-page navigation.
 * 
 * @example
 * const strategy = new IntraPathStrategy();
 * 
 * // Same page with block anchor
 * strategy.resolve(pageA, pageA, { blockId: '29d979eeca9f4abc' });
 * // Returns: '#29d979ee-ca9f-4abc-...'
 * 
 * // Same page without anchor
 * strategy.resolve(pageA, pageA, {});
 * // Returns: '' (empty string - current location)
 */
class IntraPathStrategy extends PathStrategy {
  /**
   * Create an IntraPathStrategy instance.
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
   * @returns {boolean} True if both contexts refer to the same page
   * 
   * @description Returns true when source and target have the same page ID,
   * indicating an in-page link (anchor) rather than cross-page navigation.
   */
  supports(sourceContext, targetContext) {
    // Guard against null/undefined contexts
    if (!sourceContext || !targetContext) {
      return false;
    }

    // Same page if IDs match
    return sourceContext.id === targetContext.id;
  }

  /**
   * Get the path type handled by this strategy.
   * 
   * @override
   * @returns {string} PathType.INTRA
   */
  getType() {
    return PathStrategy.PathType.INTRA;
  }

  /**
   * Resolve the path for a same-page link.
   * 
   * @override
   * @param {PageContext} sourceContext - Origin page context
   * @param {PageContext} targetContext - Destination page context (same as source)
   * @param {Object} [options={}] - Resolution options
   * @param {string} [options.blockId] - Target block ID for anchor
   * @param {Map<string, Map<string, string>>} [options.blockMapCache] - Block ID mapping cache
   * @returns {string} Anchor hash (e.g., '#29d979ee-ca9f-...') or empty string
   * 
   * @description For same-page links:
   * - If blockId provided: Returns formatted anchor hash
   * - If no blockId: Returns empty string (link to current location)
   * 
   * The block ID is formatted using the block map cache if available,
   * falling back to standard UUID formatting.
   */
  resolve(sourceContext, targetContext, options = {}) {
    const { blockId, blockMapCache } = options;

    // No block ID means link to current location
    if (!blockId) {
      return '';
    }

    // Format block ID using cache or fallback
    const formattedId = this._formatBlockId(blockId, targetContext, blockMapCache);
    return `#${formattedId}`;
  }

  /**
   * Format raw block ID to anchor-compatible UUID format.
   * 
   * @private
   * @param {string} rawBlockId - Raw block ID from URL (e.g., '29d979eeca9f4abc...')
   * @param {PageContext} targetContext - Target page for block map lookup
   * @param {Map<string, Map<string, string>>} blockMapCache - Cache of block ID maps
   * @returns {string} Formatted block ID (e.g., '29d979ee-ca9f-4abc-...')
   * 
   * @description Attempts to find the formatted ID in the block map cache.
   * If found, uses the cached mapping. Otherwise, falls back to
   * the BlockIDMapper's fallback formatting (standard UUID format).
   */
  _formatBlockId(rawBlockId, targetContext, blockMapCache) {
    // Try to find formatted ID in cache
    if (blockMapCache && targetContext && targetContext.id) {
      const pageBlockMap = blockMapCache.get(targetContext.id);
      if (pageBlockMap && pageBlockMap.size > 0) {
        const formattedFromMap = this.blockIdMapper.getFormattedId(rawBlockId, pageBlockMap);
        if (formattedFromMap) {
          return formattedFromMap;
        }
      }
    }

    // Fallback to direct formatting (UUID-style)
    return this.blockIdMapper.getFormattedId(rawBlockId, null);
  }
}

module.exports = IntraPathStrategy;
