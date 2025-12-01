/**
 * @fileoverview Intra-Page Resolver
 * @module domain/path/resolvers/IntraPageResolver
 * @description Same-page anchor link resolution using the unified PathResolver interface.
 * 
 * This resolver handles the case where source and target pages are the same.
 * In this scenario, only an anchor hash is needed (e.g., #block-id).
 * 
 * @see PathResolver - Base interface
 */

'use strict';

const PathResolver = require('../PathResolver');
const BlockIDMapper = require('../../../processing/BlockIDMapper');

/**
 * @class IntraPageResolver
 * @extends PathResolver
 * @classdesc Resolves same-page anchor links using the PathResolver interface.
 * 
 * Wraps IntraPathStrategy to provide:
 * - Unified supports() context handling
 * - Anchor-only detection from href
 * - Block ID formatting with cache support
 * 
 * @example
 * const resolver = new IntraPageResolver();
 * 
 * // Same page with block anchor
 * resolver.resolve({ source: pageA, target: pageA, blockId: '29d979eeca9f' });
 * // Returns: '#29d979ee-ca9f-...'
 * 
 * // Anchor-only href
 * resolver.resolve({ source: pageA, href: '#block-id' });
 * // Returns: '#block-id'
 */
class IntraPageResolver extends PathResolver {
  /**
   * Create an IntraPageResolver instance.
   * @param {BlockIDMapper} [blockIdMapper] - Optional custom block ID mapper
   */
  constructor(blockIdMapper = null) {
    super();
    this.blockIdMapper = blockIdMapper || new BlockIDMapper();
  }

  /**
   * Get the path type handled by this resolver.
   * @override
   * @returns {string} PathResolver.Types.INTRA
   */
  getType() {
    return PathResolver.Types.INTRA;
  }

  /**
   * Check if this resolver handles the given context.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} [context.source] - Source page context
   * @param {PageContext} [context.target] - Target page context
   * @param {string} [context.href] - Original href being resolved
   * @returns {boolean} True if this is a same-page link
   * 
   * @description Returns true when:
   * 1. The href is anchor-only (starts with #) - always intra-page
   * 2. Source and target have the same page ID
   */
  supports(context) {
    const { source, target, href } = context || {};

    // Method 1: Anchor-only links are always intra-page
    if (this._isAnchorOnly(href)) {
      return true;
    }

    // Guard against null source
    if (!source) {
      return false;
    }

    // Method 2: Guard against null target for non-anchor links
    if (!target) {
      return false;
    }

    // Method 3: Same page if IDs match
    return source.id === target.id;
  }

  /**
   * Resolve the path for a same-page link.
   * 
   * @override
   * @param {Object} context - Resolution context
   * @param {PageContext} [context.source] - Source page context
   * @param {PageContext} [context.target] - Target page context (same as source)
   * @param {string} [context.blockId] - Target block ID for anchor
   * @param {string} [context.href] - Original href (for anchor-only extraction)
   * @param {Map<string, Map<string, string>>} [context.blockMapCache] - Block ID mapping cache
   * @returns {string} Anchor hash (e.g., '#29d979ee-ca9f-...') or empty string
   */
  resolve(context) {
    const { source, target, blockId, href, blockMapCache } = context || {};

    // If we have an anchor-only href, handle it directly
    if (this._isAnchorOnly(href)) {
      const extractedBlockId = this._extractBlockIdFromHref(href);
      if (extractedBlockId) {
        const formattedId = this._formatBlockId(extractedBlockId, target || source, blockMapCache);
        return `#${formattedId}`;
      }
      // Return the anchor as-is if we can't extract a block ID
      return href;
    }

    // No block ID means link to current location
    if (!blockId) {
      return '';
    }

    // Format block ID using cache or fallback
    const formattedId = this._formatBlockId(blockId, target, blockMapCache);
    return `#${formattedId}`;
  }

  /**
   * Check if href is an anchor-only link.
   * @private
   * @param {string} href - Link href to check
   * @returns {boolean} True if href starts with '#'
   */
  _isAnchorOnly(href) {
    return href && typeof href === 'string' && href.startsWith('#');
  }

  /**
   * Extract block ID from anchor href.
   * @private
   * @param {string} href - Link href starting with #
   * @returns {string|null} Block ID or null
   */
  _extractBlockIdFromHref(href) {
    if (!href || typeof href !== 'string') {
      return null;
    }

    const hashIndex = href.indexOf('#');
    if (hashIndex !== -1) {
      const fragment = href.substring(hashIndex + 1);
      // Validate it looks like a Notion block ID
      if (/^[a-f0-9-]{32,36}$/i.test(fragment)) {
        return fragment;
      }
    }
    return null;
  }

  /**
   * Format raw block ID to anchor-compatible UUID format.
   * @private
   * @param {string} rawBlockId - Raw block ID
   * @param {PageContext} targetContext - Target page for block map lookup
   * @param {Map<string, Map<string, string>>} blockMapCache - Cache of block ID maps
   * @returns {string} Formatted block ID
   */
  _formatBlockId(rawBlockId, targetContext, blockMapCache) {
    if (blockMapCache && targetContext && targetContext.id) {
      const pageBlockMap = blockMapCache.get(targetContext.id);
      if (pageBlockMap && pageBlockMap.size > 0) {
        const formattedFromMap = this.blockIdMapper.getFormattedId(rawBlockId, pageBlockMap);
        if (formattedFromMap) {
          return formattedFromMap;
        }
      }
    }

    return this.blockIdMapper.getFormattedId(rawBlockId, null);
  }
}

module.exports = IntraPageResolver;
